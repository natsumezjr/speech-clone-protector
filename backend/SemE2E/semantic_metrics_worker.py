"""Isolated semantic token/drift worker for one audio pair."""

from __future__ import annotations

import json
import os
import sys
import traceback
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@contextmanager
def _worker_output_to_stderr() -> Iterator[None]:
    stdout_fd: int | None = None
    saved_stdout_fd: int | None = None
    try:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
        sys.stdout.flush()
        saved_stdout_fd = os.dup(stdout_fd)
        os.dup2(stderr_fd, stdout_fd)
    except (AttributeError, OSError):
        if saved_stdout_fd is not None:
            os.close(saved_stdout_fd)
        with redirect_stdout(sys.stderr):
            yield
        return
    try:
        with redirect_stdout(sys.stderr):
            yield
    finally:
        if stdout_fd is not None and saved_stdout_fd is not None:
            os.dup2(saved_stdout_fd, stdout_fd)
            os.close(saved_stdout_fd)


def execute(request: dict[str, Any]) -> dict[str, Any]:
    from metric_definitions import compute_semantic_token_metrics

    clean_path = Path(str(request.get("originalPath") or "")).expanduser().resolve()
    protected_path = Path(str(request.get("protectedPath") or "")).expanduser().resolve()
    if not clean_path.is_file():
        raise FileNotFoundError(f"originalPath does not exist: {clean_path}")
    if not protected_path.is_file():
        raise FileNotFoundError(f"protectedPath does not exist: {protected_path}")
    config = request.get("config") if isinstance(request.get("config"), dict) else {}
    return {"ok": True, "metrics": compute_semantic_token_metrics(clean_path, protected_path, config)}


def main() -> int:
    exit_code = 0
    try:
        request = json.loads(sys.stdin.read())
        if not isinstance(request, dict):
            raise TypeError("semantic metrics worker request must be a JSON object")
        with _worker_output_to_stderr():
            response = execute(request)
    except Exception as exc:
        response = {
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        exit_code = 1
    sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
