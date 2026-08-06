from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


Probe = Callable[[], dict[str, Any]]
CACHE_FILENAME = "capabilities-cache.json"
REFRESH_FLAG_FILENAME = "capabilities-refresh.flag"
REFRESH_LOCK_FILENAME = "capabilities-refresh.lock"
LOCK_STALE_AFTER_SECONDS = 15 * 60


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _paths(runtime_dir: Path) -> tuple[Path, Path, Path]:
    root = Path(runtime_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root / CACHE_FILENAME, root / REFRESH_FLAG_FILENAME, root / REFRESH_LOCK_FILENAME


def _atomic_write_text(path: Path, value: str) -> None:
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(value, encoding="utf-8")
    os.replace(temp_path, path)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def _read_snapshot(path: Path) -> dict[str, Any] | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    payload = record.get("payload")
    return record if isinstance(payload, dict) else None


def _refresh_requested(flag_path: Path) -> bool:
    try:
        return flag_path.read_text(encoding="utf-8").strip() == "1"
    except FileNotFoundError:
        return False
    except OSError:
        return False


def _claim_refresh(lock_path: Path) -> bool:
    try:
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            if time.time() - lock_path.stat().st_mtime <= LOCK_STALE_AFTER_SECONDS:
                return False
            lock_path.unlink()
        except (FileNotFoundError, OSError):
            return False
        return _claim_refresh(lock_path)
    with os.fdopen(descriptor, "w", encoding="utf-8") as lock_file:
        lock_file.write(f"pid={os.getpid()}\nstartedAt={_now_iso()}\n")
    return True


def _record_from_probe(probe: Probe, previous: dict[str, Any] | None) -> dict[str, Any]:
    payload = probe()
    revision = int((previous or {}).get("revision") or 0) + 1
    return {
        "schemaVersion": 1,
        "revision": revision,
        "refreshedAt": _now_iso(),
        "payload": payload,
    }


def _refresh_claimed(
    cache_path: Path,
    flag_path: Path,
    lock_path: Path,
    probe: Probe,
    logger: logging.Logger,
) -> dict[str, Any] | None:
    try:
        record = _record_from_probe(probe, _read_snapshot(cache_path))
        _atomic_write_json(cache_path, record)
        _atomic_write_text(flag_path, "0\n")
        return record
    except Exception:
        logger.exception("capabilities snapshot refresh failed")
        return None
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _start_background_refresh(
    cache_path: Path,
    flag_path: Path,
    lock_path: Path,
    probe: Probe,
    logger: logging.Logger,
) -> bool:
    if not _claim_refresh(lock_path):
        return lock_path.exists()
    threading.Thread(
        target=_refresh_claimed,
        args=(cache_path, flag_path, lock_path, probe, logger),
        name="capabilities-refresh",
        daemon=True,
    ).start()
    return True


def get_capabilities_snapshot(
    runtime_dir: Path,
    probe: Probe,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Return a durable snapshot and refresh it in the background when flagged.

    The cached payload lives on disk, so it survives API restarts and is shared by
    multiple workers. Writing ``1`` to ``capabilities-refresh.flag`` invalidates
    the snapshot. A request still receives the previous snapshot immediately while
    one worker refreshes it and atomically resets the flag to ``0``.
    """

    cache_path, flag_path, lock_path = _paths(runtime_dir)
    active_logger = logger or logging.getLogger("seme2e_capabilities")
    snapshot = _read_snapshot(cache_path)
    requested = _refresh_requested(flag_path)

    if snapshot is None:
        claimed = _claim_refresh(lock_path)
        if claimed:
            snapshot = _refresh_claimed(cache_path, flag_path, lock_path, probe, active_logger)
        else:
            deadline = time.monotonic() + 60
            while snapshot is None and time.monotonic() < deadline:
                time.sleep(0.05)
                snapshot = _read_snapshot(cache_path)
        if snapshot is None:
            raise RuntimeError("capabilities snapshot could not be initialized")
        cache_hit = False
        refreshing = False
    else:
        cache_hit = True
        refreshing = requested and _start_background_refresh(cache_path, flag_path, lock_path, probe, active_logger)

    payload = copy.deepcopy(snapshot["payload"])
    payload["cache"] = {
        "hit": cache_hit,
        "revision": int(snapshot.get("revision") or 1),
        "refreshedAt": snapshot.get("refreshedAt"),
        "refreshRequested": requested,
        "refreshing": refreshing,
        "strategy": "disk-snapshot-stale-while-revalidate",
    }
    return payload
