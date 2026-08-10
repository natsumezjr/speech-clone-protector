from __future__ import annotations

import argparse
import json
import math
import os
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Sequence
from urllib.parse import unquote, urlsplit

import result_adapter as adapter
from result_schema import utc_now_iso


V21_IDENTITY_SOURCE = "VoiceShield_v2.1_clone_identity"


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def missing_clone_metric_groups(clone_result: dict[str, Any]) -> list[str]:
    clone_eval = clone_result.get("cloneEval")
    if not isinstance(clone_eval, dict):
        return ["asr", "semantic", "dnsmos", "v2.1"]

    missing: list[str] = []
    if not (
        clone_eval.get("cloneAsrStatus") == "available"
        and isinstance(clone_eval.get("cleanCloneTranscription"), str)
        and isinstance(clone_eval.get("protectedCloneTranscription"), str)
    ):
        missing.append("asr")

    if not (
        clone_eval.get("cloneSemanticStatus") == "available"
        and _finite(clone_eval.get("cloneTokenChangeRate"))
        and _finite(clone_eval.get("cloneSemanticDrift"))
        and _finite(clone_eval.get("cloneSemanticScore"))
        and _finite(clone_eval.get("semanticBaselineWeight"))
    ):
        missing.append("semantic")

    if not (
        clone_eval.get("cloneQualityStatus") == "available"
        and _finite(clone_eval.get("cleanCloneQualityMos"))
        and _finite(clone_eval.get("protectedCloneQualityMos"))
        and _finite(clone_eval.get("cloneQualityRawScore"))
        and _finite(clone_eval.get("cloneQualityRelevance"))
        and _finite(clone_eval.get("cloneQualityScore"))
        and _finite(clone_eval.get("qualityBaselineWeight"))
    ):
        missing.append("dnsmos")

    sources = clone_eval.get("_metricSources")
    identity_source = None
    if isinstance(sources, dict):
        identity_meta = sources.get("cloneEval.cloneIdentityScore")
        if isinstance(identity_meta, dict):
            identity_source = identity_meta.get("source")
    if not (
        clone_eval.get("status") == "available"
        and clone_eval.get("cloneIdentityStatus") == "available"
        and _finite(clone_eval.get("cloneIdentityScore"))
        and _finite(clone_eval.get("identityBaselineWeight"))
        and _finite(clone_eval.get("cloneDefenseScore"))
        and identity_source == V21_IDENTITY_SOURCE
    ):
        missing.append("v2.1")
    return missing


def clone_metrics_complete(clone_result: dict[str, Any]) -> bool:
    return not missing_clone_metric_groups(clone_result)


def _clone_locator(clone_result: dict[str, Any], index: int) -> dict[str, Any]:
    request = clone_result.get("request") if isinstance(clone_result.get("request"), dict) else {}
    original_meta = clone_result.get("originalCloneAudio") if isinstance(clone_result.get("originalCloneAudio"), dict) else {}
    protected_meta = clone_result.get("protectedCloneAudio") if isinstance(clone_result.get("protectedCloneAudio"), dict) else {}
    return {
        "cloneId": str(clone_result.get("cloneId") or "") or None,
        "cloneSubId": str(clone_result.get("cloneSubId") or "") or None,
        "index": index,
        "fingerprint": (
            str(request.get("model") or ""),
            str(original_meta.get("filename") or ""),
            str(protected_meta.get("filename") or ""),
            str(clone_result.get("createdAt") or ""),
        ),
    }


def _find_clone_by_locator(result: dict[str, Any], locator: dict[str, Any]) -> dict[str, Any]:
    clones = [item for item in result.get("cloneResults") or [] if isinstance(item, dict)]
    clone_id = locator.get("cloneId")
    clone_sub_id = locator.get("cloneSubId")
    if clone_id:
        matches = [item for item in clones if str(item.get("cloneId") or "") == clone_id]
    elif clone_sub_id:
        matches = [item for item in clones if str(item.get("cloneSubId") or "") == clone_sub_id]
    else:
        matches = [
            item
            for index, item in enumerate(clones)
            if _clone_locator(item, index).get("fingerprint") == locator.get("fingerprint")
        ]
    if len(matches) != 1:
        label = clone_id or clone_sub_id or f"index {locator.get('index')}"
        raise RuntimeError(f"clone record changed while metrics were being computed: {label}")
    return matches[0]


def _same_clone(candidate: dict[str, Any], canonical: dict[str, Any]) -> bool:
    candidate_clone_id = str(candidate.get("cloneId") or "")
    canonical_clone_id = str(canonical.get("cloneId") or "")
    if candidate_clone_id and canonical_clone_id:
        return candidate_clone_id == canonical_clone_id
    candidate_sub_id = str(candidate.get("cloneSubId") or "")
    canonical_sub_id = str(canonical.get("cloneSubId") or "")
    return bool(candidate_sub_id and canonical_sub_id and candidate_sub_id == canonical_sub_id)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("x", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _sync_existing_clone_copies(task_id: str, canonical: dict[str, Any]) -> dict[str, list[str]]:
    task_dir = adapter.TASK_DIR / task_id
    synced: list[str] = []
    errors: list[str] = []

    def sync_result_file(path: Path) -> None:
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and _same_clone(payload, canonical):
                _write_json_atomic(path, canonical)
                synced.append(str(path))
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    sync_result_file(task_dir / "clone_result.json")
    history_dir = task_dir / "clone_results"
    if history_dir.exists():
        for history_path in sorted(history_dir.glob("*.json")):
            sync_result_file(history_path)

    status_path = task_dir / "status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status_changed = False
            if isinstance(status, dict):
                root_clone_result = status.get("cloneResult")
                root_matches = (
                    isinstance(root_clone_result, dict) and _same_clone(root_clone_result, canonical)
                ) or (
                    str(status.get("cloneSubId") or "")
                    and str(status.get("cloneSubId") or "") == str(canonical.get("cloneSubId") or "")
                )
                if root_matches:
                    status["cloneResult"] = canonical
                    status_changed = True

                for key in ["cloneTask"]:
                    snapshot = status.get(key)
                    if not isinstance(snapshot, dict):
                        continue
                    snapshot_result = snapshot.get("cloneResult")
                    if (
                        isinstance(snapshot_result, dict) and _same_clone(snapshot_result, canonical)
                    ) or (
                        str(snapshot.get("cloneSubId") or "")
                        and str(snapshot.get("cloneSubId") or "") == str(canonical.get("cloneSubId") or "")
                    ):
                        snapshot["cloneResult"] = canonical
                        status_changed = True

                clone_tasks = status.get("cloneTasks")
                if isinstance(clone_tasks, list):
                    for snapshot in clone_tasks:
                        if not isinstance(snapshot, dict):
                            continue
                        snapshot_result = snapshot.get("cloneResult")
                        if (
                            isinstance(snapshot_result, dict) and _same_clone(snapshot_result, canonical)
                        ) or (
                            str(snapshot.get("cloneSubId") or "")
                            and str(snapshot.get("cloneSubId") or "") == str(canonical.get("cloneSubId") or "")
                        ):
                            snapshot["cloneResult"] = canonical
                            status_changed = True

                clone_batches = status.get("cloneBatches")
                if isinstance(clone_batches, list):
                    for batch in clone_batches:
                        if not isinstance(batch, dict) or not isinstance(batch.get("items"), list):
                            continue
                        for batch_item in batch["items"]:
                            if not isinstance(batch_item, dict):
                                continue
                            batch_result = batch_item.get("cloneResult")
                            if (
                                isinstance(batch_result, dict) and _same_clone(batch_result, canonical)
                            ) or (
                                str(batch_item.get("cloneSubId") or "")
                                and str(batch_item.get("cloneSubId") or "") == str(canonical.get("cloneSubId") or "")
                            ):
                                batch_item["cloneResult"] = canonical
                                status_changed = True
                if status_changed:
                    _write_json_atomic(status_path, status)
                    synced.append(str(status_path))
        except Exception as exc:
            errors.append(f"{status_path}: {exc}")
    return {"synced": synced, "errors": errors}


def _selected_clones(
    result: dict[str, Any],
    clone_sub_id: str | None,
) -> list[tuple[int, dict[str, Any]]]:
    clones = [
        (index, item)
        for index, item in enumerate(result.get("cloneResults") or [])
        if isinstance(item, dict)
    ]
    if clone_sub_id is None:
        return clones
    matches = [
        item
        for item in clones
        if str(item[1].get("cloneSubId") or "") == clone_sub_id
    ]
    if not matches:
        matches = [
            item
            for item in clones
            if str(item[1].get("cloneId") or "") == clone_sub_id
        ]
    if not matches:
        raise ValueError(f"clone subtask not found: {clone_sub_id}")
    if len(matches) > 1:
        raise ValueError(f"clone subtask is not unique: {clone_sub_id}")
    return matches


def _safe_existing_candidate(candidate: Path, clones_root: Path) -> Path | None:
    resolved_root = clones_root.resolve()
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError:
        return None
    return resolved_candidate if resolved_candidate.is_file() else None


def _resolve_clone_audio_path(
    task_id: str,
    clone_result: dict[str, Any],
    field: str,
) -> Path:
    metadata = clone_result.get(field)
    if not isinstance(metadata, dict):
        raise FileNotFoundError(f"{field} metadata is missing")
    task_dir = adapter.TASK_DIR / task_id
    clones_root = task_dir / "clones"
    filename = Path(str(metadata.get("filename") or "")).name
    clone_id = str(clone_result.get("cloneId") or "").strip()
    candidates: list[Path] = []
    if clone_id and Path(clone_id).name == clone_id and filename:
        candidates.append(clones_root / clone_id / filename)

    audio_url = str(metadata.get("audioUrl") or metadata.get("downloadUrl") or "").strip()
    if audio_url:
        url_path = unquote(urlsplit(audio_url).path)
        marker = f"/api/artifacts/{task_id}/clones/"
        if marker in url_path:
            relative_url = url_path.split(marker, 1)[1]
            relative_parts = PurePosixPath(relative_url).parts
            if relative_parts:
                candidates.append(clones_root.joinpath(*relative_parts))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = _safe_existing_candidate(candidate, clones_root)
        if resolved is not None and resolved not in seen:
            return resolved
        if resolved is not None:
            seen.add(resolved)

    if filename and clones_root.exists():
        matches: list[Path] = []
        for candidate in clones_root.rglob("*"):
            if not candidate.is_file() or candidate.name != filename:
                continue
            resolved = _safe_existing_candidate(candidate, clones_root)
            if resolved is not None and resolved not in matches:
                matches.append(resolved)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(f"{field} artifact is ambiguous: {filename}")
    raise FileNotFoundError(f"{field} artifact is missing: {filename or '<unknown>'}")


def _compute_clone_metrics(
    task_result: dict[str, Any],
    original_reference_path: Path,
    protected_reference_path: Path,
    original_clone_path: Path,
    protected_clone_path: Path,
    clone_result: dict[str, Any],
) -> dict[str, Any]:
    request = dict(clone_result.get("request") or {})
    existing_eval = clone_result.get("cloneEval") if isinstance(clone_result.get("cloneEval"), dict) else {}
    if not request.get("text") and existing_eval.get("targetText"):
        request["text"] = existing_eval.get("targetText")
    if not request.get("asrModel") and existing_eval.get("cloneAsrModel"):
        request["asrModel"] = existing_eval.get("cloneAsrModel")
    task_request = task_result.get("request") if isinstance(task_result.get("request"), dict) else {}
    semantic_config = task_request.get("semantic") if isinstance(task_request.get("semantic"), dict) else {}

    transcription = adapter._transcribe_clone_pair_isolated(
        original_clone_path,
        protected_clone_path,
        request,
    )
    semantic_metrics = adapter._compute_clone_semantic_isolated(
        original_clone_path,
        protected_clone_path,
        semantic_config,
    )
    quality_metrics = adapter._evaluate_dnsmos_pair_isolated(
        original_clone_path,
        protected_clone_path,
    )
    clone_eval = adapter.compute_clone_eval(
        original_reference_path,
        original_clone_path,
        protected_clone_path,
        clone_result,
        protected_audio_path=protected_reference_path,
        clone_transcription=transcription,
        semantic_metrics=semantic_metrics,
        quality_metrics=quality_metrics,
    )
    existing_created_at = clone_result.get("createdAt") or existing_eval.get("createdAt")
    if existing_created_at:
        clone_eval["createdAt"] = existing_created_at
    return clone_eval


def _metric_status(clone_eval: dict[str, Any]) -> dict[str, Any]:
    return {
        "asr": clone_eval.get("cloneAsrStatus"),
        "semantic": clone_eval.get("cloneSemanticStatus"),
        "dnsmos": clone_eval.get("cloneQualityStatus"),
        "v2.1": clone_eval.get("status"),
    }


def backfill_clone_metrics(
    task_id: str,
    *,
    clone_sub_id: str | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    original_reference_path, protected_reference_path, task_result = adapter._task_audio_paths(task_id)
    selected = _selected_clones(task_result, clone_sub_id)
    if not selected:
        raise ValueError(f"task {task_id} has no clone results")

    items: list[dict[str, Any]] = []
    pending_updates: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for index, clone_result in selected:
        locator = _clone_locator(clone_result, index)
        missing = missing_clone_metric_groups(clone_result)
        item: dict[str, Any] = {
            "cloneId": locator.get("cloneId"),
            "cloneSubId": locator.get("cloneSubId"),
            "model": ((clone_result.get("request") or {}).get("model") if isinstance(clone_result.get("request"), dict) else None),
            "missing": missing,
        }
        items.append(item)
        if not force and not missing:
            item["status"] = "skipped_complete"
            continue
        try:
            original_clone_path = _resolve_clone_audio_path(task_id, clone_result, "originalCloneAudio")
            protected_clone_path = _resolve_clone_audio_path(task_id, clone_result, "protectedCloneAudio")
            item["artifacts"] = {
                "original": str(original_clone_path),
                "protected": str(protected_clone_path),
            }
            if dry_run:
                item["status"] = "would_update"
                continue
            clone_eval = _compute_clone_metrics(
                task_result,
                original_reference_path,
                protected_reference_path,
                original_clone_path,
                protected_clone_path,
                clone_result,
            )
            item["status"] = "computed"
            item["metricStatus"] = _metric_status(clone_eval)
            pending_updates.append((locator, clone_eval, item))
        except Exception as exc:
            item["status"] = "failed"
            item["reason"] = str(exc)

    if pending_updates:
        applied_updates: list[tuple[dict[str, Any], dict[str, Any]]] = []

        def apply_updates(latest_result: dict[str, Any]) -> bool:
            changed = False
            for locator, clone_eval, item in pending_updates:
                try:
                    current_clone = _find_clone_by_locator(latest_result, locator)
                    if not force and clone_metrics_complete(current_clone):
                        item["status"] = "skipped_complete"
                        item["reason"] = "metrics were completed by another writer"
                        continue
                    existing_eval = current_clone.get("cloneEval") if isinstance(current_clone.get("cloneEval"), dict) else {}
                    existing_created_at = current_clone.get("createdAt") or existing_eval.get("createdAt")
                    if existing_created_at:
                        clone_eval["createdAt"] = existing_created_at
                    current_clone["cloneEval"] = clone_eval
                    for key in adapter.CLONE_EVAL_MIRROR_FIELDS:
                        current_clone.pop(key, None)
                    adapter._sync_clone_eval_fields(current_clone, clone_eval)
                    item["status"] = "updated" if clone_metrics_complete(current_clone) else "updated_partial"
                    item["missingAfter"] = missing_clone_metric_groups(current_clone)
                    applied_updates.append((locator, item))
                    changed = True
                except Exception as exc:
                    item["status"] = "failed"
                    item["reason"] = str(exc)

            if not changed:
                return False
            latest_clone_eval = None
            for current_clone in latest_result.get("cloneResults") or []:
                if isinstance(current_clone, dict) and isinstance(current_clone.get("cloneEval"), dict):
                    latest_clone_eval = current_clone["cloneEval"]
            if isinstance(latest_clone_eval, dict):
                latest_result.setdefault("summary", {}).setdefault("metricSources", {}).update(
                    latest_clone_eval.get("_metricSources") or {}
                )
            adapter.refresh_result_scores(latest_result)
            latest_result["updatedAt"] = utc_now_iso()
            return True

        def sync_snapshots(latest_result: dict[str, Any]) -> None:
            for locator, item in applied_updates:
                try:
                    current_clone = _find_clone_by_locator(latest_result, locator)
                    sync_result = _sync_existing_clone_copies(task_id, current_clone)
                    item["syncedSnapshots"] = sync_result["synced"]
                    if sync_result["errors"]:
                        item["snapshotSyncErrors"] = sync_result["errors"]
                        if item.get("status") == "updated":
                            item["status"] = "updated_snapshot_partial"
                except Exception as exc:
                    item["snapshotSyncErrors"] = [str(exc)]
                    if item.get("status") == "updated":
                        item["status"] = "updated_snapshot_partial"

        adapter.update_result_safely(task_id, apply_updates, after_write=sync_snapshots)

    statuses = [str(item.get("status") or "") for item in items]
    return {
        "taskId": task_id,
        "cloneSubId": clone_sub_id,
        "dryRun": dry_run,
        "force": force,
        "selected": len(items),
        "updated": sum(status == "updated" for status in statuses),
        "updatedPartial": sum(status in {"updated_partial", "updated_snapshot_partial"} for status in statuses),
        "skipped": sum(status == "skipped_complete" for status in statuses),
        "wouldUpdate": sum(status == "would_update" for status in statuses),
        "failed": sum(status == "failed" for status in statuses),
        "items": items,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill metrics for existing clone audio artifacts without running TTS.",
    )
    parser.add_argument("--task-id", required=True, help="Task id containing result.json and clone artifacts")
    parser.add_argument("--clone-sub-id", help="Optional cloneSubId; cloneId is accepted as a compatibility fallback")
    parser.add_argument("--dry-run", action="store_true", help="Validate selection and artifacts without computing or writing")
    parser.add_argument("--force", action="store_true", help="Recompute records that already have complete metrics")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = backfill_clone_metrics(
            args.task_id,
            clone_sub_id=args.clone_sub_id,
            dry_run=args.dry_run,
            force=args.force,
        )
    except Exception as exc:
        summary = {
            "taskId": args.task_id,
            "cloneSubId": args.clone_sub_id,
            "dryRun": args.dry_run,
            "status": "failed",
            "reason": str(exc),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
