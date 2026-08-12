from __future__ import annotations

import csv
import io
import json
import logging
import multiprocessing
import os
import re
import traceback
import shutil
import threading
import time
import uuid
import zipfile
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from audio_preprocess import probe_audio_metadata
from capability_cache import get_capabilities_snapshot
from dnsmos_quality import dnsmos_model_status
from result_adapter import (
    ASR_WORKER_MAX_CONCURRENCY,
    GPU_ACQUIRE_TIMEOUT_MESSAGE,
    TASK_DIR,
    UPLOAD_DIR,
    AudioPreprocessError,
    CloneBackendUnavailableError,
    ProtectGenerationError,
    _worker_gpu_candidates,
    acquire_gpu_slot,
    create_asr_eval,
    create_clone_voice,
    clone_worker_capacity_snapshot,
    create_psychoacoustic_slice,
    create_task,
    diagnose_capabilities,
    ensure_protection_dnsmos,
    ensure_runtime_dirs,
    load_result,
    new_task_id,
    new_file_id,
    refresh_result_scores,
    release_gpu_slot,
    runtime_config,
    update_result_safely,
    supported_tts_languages,
    tts_model_requires_reference_text,
)
from result_schema import utc_now_iso

ensure_runtime_dirs()

app = FastAPI(title="SemE2E API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FILES: dict[str, dict[str, Any]] = {}
LOG_DIR = TASK_DIR.parent / "logs" / "tasks"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("seme2e_api")
logging.basicConfig(level=logging.INFO)
TASK_CANCEL_EVENTS: dict[str, threading.Event] = {}
TASK_THREADS: dict[str, threading.Thread] = {}
PROTECT_PROCESS_CONTEXT = multiprocessing.get_context("spawn")
TASK_PROCESSES: dict[str, multiprocessing.Process] = {}
DELETED_TASK_IDS: set[str] = set()
DELETED_SUBTASK_KEYS: set[str] = set()
TASK_REGISTRY_LOCK = threading.Lock()
SUBTASK_TOMBSTONE_LOCK = threading.RLock()
TASK_STATUS_WRITE_LOCK = threading.RLock()
EVALUATION_COORDINATION_LOCK = threading.RLock()
EVALUATION_COLLECTION_DELETIONS: set[str] = set()


def _positive_env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


PROTECT_MAX_CONCURRENCY = min(
    2,
    _positive_env_int("SEME2E_PROTECT_MAX_CONCURRENCY", 2),
)
PROTECT_PENDING_TASKS: deque[dict[str, Any]] = deque()
PROTECT_ACTIVE_TASK_IDS: set[str] = set()
PROTECT_QUEUE_LOCK = threading.RLock()
PROTECT_DISPATCH_RETRY_TIMER: threading.Timer | None = None
DELETED_TASK_DIR = TASK_DIR.parent / "deleted_tasks"
DELETED_TASK_DIR.mkdir(parents=True, exist_ok=True)


class TaskCancelledError(RuntimeError):
    pass


def register_task_runtime(task_id: str, cancel_event: threading.Event, thread: threading.Thread | None = None, process: multiprocessing.Process | None = None, *, runtime_id: str | None = None) -> None:
    key = runtime_id or task_id
    with TASK_REGISTRY_LOCK:
        TASK_CANCEL_EVENTS[key] = cancel_event
        if thread is not None:
            TASK_THREADS[key] = thread
        if process is not None:
            TASK_PROCESSES[key] = process


def cleanup_task_runtime(task_id: str, *, runtime_id: str | None = None) -> None:
    key = runtime_id or task_id
    with TASK_REGISTRY_LOCK:
        TASK_CANCEL_EVENTS.pop(key, None)
        TASK_THREADS.pop(key, None)
        TASK_PROCESSES.pop(key, None)


def cleanup_protect_process_runtime(task_id: str, process: multiprocessing.Process, cancel_event: Any) -> None:
    """Remove only the runtime entries owned by this protection process."""
    with TASK_REGISTRY_LOCK:
        if TASK_PROCESSES.get(task_id) is process:
            TASK_PROCESSES.pop(task_id, None)
        if TASK_CANCEL_EVENTS.get(task_id) is cancel_event:
            TASK_CANCEL_EVENTS.pop(task_id, None)


def request_task_cancel(task_id: str) -> tuple[threading.Event | None, threading.Thread | None, multiprocessing.Process | None]:
    with TASK_REGISTRY_LOCK:
        prefix = f"{task_id}:"
        cancel_events = [value for key, value in TASK_CANCEL_EVENTS.items() if key == task_id or key.startswith(prefix)]
        threads = [value for key, value in TASK_THREADS.items() if key == task_id or key.startswith(prefix)]
        processes = [value for key, value in TASK_PROCESSES.items() if key == task_id or key.startswith(prefix)]
    for cancel_event in cancel_events:
        cancel_event.set()
    cancel_event = cancel_events[0] if cancel_events else None
    thread = threads[0] if threads else None
    process = processes[0] if processes else None
    return cancel_event, thread, process


def request_all_task_cancels(task_id: str) -> tuple[list[threading.Event], list[threading.Thread], list[multiprocessing.Process]]:
    with TASK_REGISTRY_LOCK:
        prefix = f"{task_id}:"
        cancel_events = [value for key, value in TASK_CANCEL_EVENTS.items() if key == task_id or key.startswith(prefix)]
        threads = [value for key, value in TASK_THREADS.items() if key == task_id or key.startswith(prefix)]
        processes = [value for key, value in TASK_PROCESSES.items() if key == task_id or key.startswith(prefix)]
    for cancel_event in cancel_events:
        cancel_event.set()
    return (
        list({id(value): value for value in cancel_events}.values()),
        list({id(value): value for value in threads}.values()),
        list({id(value): value for value in processes}.values()),
    )


def _subtask_runtime_key(task_id: str, subtask_id: str) -> str:
    return f"{task_id}:{subtask_id}"


def _evaluation_collection_key(task_id: str, batch_type: str) -> str:
    return f"{task_id}:{batch_type}"


def _begin_evaluation_submission(task_id: str, batch_type: str) -> None:
    if _evaluation_collection_key(task_id, batch_type) in EVALUATION_COLLECTION_DELETIONS:
        raise HTTPException(
            status_code=409,
            detail=f"{batch_type} evaluation history is being deleted; retry after deletion completes",
        )


def _begin_evaluation_collection_deletion(task_id: str, batch_type: str) -> str:
    key = _evaluation_collection_key(task_id, batch_type)
    if key in EVALUATION_COLLECTION_DELETIONS:
        raise HTTPException(status_code=409, detail=f"{batch_type} evaluation history deletion is already in progress")
    EVALUATION_COLLECTION_DELETIONS.add(key)
    return key


def _finish_evaluation_collection_deletion(key: str) -> None:
    with EVALUATION_COORDINATION_LOCK:
        EVALUATION_COLLECTION_DELETIONS.discard(key)


def mark_subtask_deleted(task_id: str, subtask_id: str) -> None:
    with SUBTASK_TOMBSTONE_LOCK:
        DELETED_SUBTASK_KEYS.add(_subtask_runtime_key(task_id, subtask_id))


def is_subtask_deleted(task_id: str, subtask_id: str) -> bool:
    with SUBTASK_TOMBSTONE_LOCK:
        return _subtask_runtime_key(task_id, subtask_id) in DELETED_SUBTASK_KEYS


def request_subtask_cancel(task_id: str, subtask_id: str) -> tuple[threading.Event | None, threading.Thread | None, multiprocessing.Process | None]:
    key = _subtask_runtime_key(task_id, subtask_id)
    with TASK_REGISTRY_LOCK:
        cancel_event = TASK_CANCEL_EVENTS.get(key)
        thread = TASK_THREADS.get(key)
        process = TASK_PROCESSES.get(key)
    if cancel_event is not None:
        cancel_event.set()
    return cancel_event, thread, process


def cleanup_all_task_runtimes(task_id: str) -> None:
    prefix = f"{task_id}:"
    with TASK_REGISTRY_LOCK:
        for registry in (TASK_CANCEL_EVENTS, TASK_THREADS, TASK_PROCESSES):
            for key in [item for item in registry if item == task_id or item.startswith(prefix)]:
                registry.pop(key, None)


def mark_task_deleted(task_id: str) -> None:
    with TASK_REGISTRY_LOCK:
        DELETED_TASK_IDS.add(task_id)
    (DELETED_TASK_DIR / f"{task_id}.deleted").write_text(utc_now_iso(), encoding="utf-8")


def is_task_deleted(task_id: str) -> bool:
    with TASK_REGISTRY_LOCK:
        if task_id in DELETED_TASK_IDS:
            return True
    return (DELETED_TASK_DIR / f"{task_id}.deleted").exists()


def ensure_task_not_cancelled(task_id: str, cancel_event: threading.Event) -> None:
    if cancel_event.is_set() or is_task_deleted(task_id):
        raise TaskCancelledError(f"task cancelled: {task_id}")


def protection_progress_status(event: dict[str, Any]) -> dict[str, Any]:
    step = event.get("step")
    if step is None and event.get("stage"):
        try:
            stage_progress = float(event.get("progress"))
        except (TypeError, ValueError):
            stage_progress = 0.18
        return {
            "status": "running",
            "progress": round(min(0.99, max(0.0, stage_progress)), 3),
            "stage": str(event["stage"]),
            "message": str(event.get("message") or "Backend is processing the task"),
            "error": None,
            "progressSource": "backend_stage",
        }
    total = event.get("total_steps") or event.get("total") or 1
    try:
        step_value = max(0, int(step))
    except (TypeError, ValueError):
        step_value = 0
    try:
        total_value = max(1, int(total))
    except (TypeError, ValueError):
        total_value = 1
    try:
        algorithm_progress = float(event.get("progress"))
    except (TypeError, ValueError):
        algorithm_progress = step_value / total_value
    algorithm_progress = min(1.0, max(0.0, algorithm_progress))
    optimization_metrics = {
        key: event.get(key)
        for key in (
            "current_lr",
            "total_loss",
            "feature_loss",
            "semantic_loss",
            "psychoacoustic_loss",
            "l2_loss",
            "stft_loss",
            "snr_loss",
            "current_snr_db",
        )
        if event.get(key) is not None
    }
    return {
        "status": "running",
        "progress": round(0.18 + algorithm_progress * 0.77, 3),
        "stage": "protect_generation",
        "message": f"Protect optimization step {step_value}/{total_value}" if step_value else "Backend is generating protected audio",
        "error": None,
        "currentStep": step_value or None,
        "totalSteps": total_value,
        "stageProgress": round(algorithm_progress, 3),
        "progressSource": "core.guard.VoiceShield",
        "optimizationMetrics": optimization_metrics,
    }


class ProtectTaskRequest(BaseModel):
    fileId: str | None = None
    mode: str | None = "custom"
    targets: list[str] | None = None
    semantic: dict[str, Any] | None = None
    timbre: dict[str, Any] | None = None
    psychoacoustic: dict[str, Any] | None = None
    optimization: dict[str, Any] | None = None
    referenceText: str | None = None
    reference_text: str | None = None


class CloneVoiceRequest(BaseModel):
    text: str
    model: str | None = "default"
    asrModel: str | None = None
    language: str | None = "auto"
    speed: float | None = 1.0
    speakerPrompt: str | None = None
    annotationSource: str | None = None
    annotationAsrSubId: str | None = None
    batchId: str | None = None
    batchItemId: str | None = None


class AsrEvalRequest(BaseModel):
    model: str
    language: str | None = None
    referenceText: str | None = None
    reference_text: str | None = None
    batchId: str | None = None
    batchItemId: str | None = None


class EvaluationBatchRequest(BaseModel):
    batchId: str
    type: str
    items: list[dict[str, Any]]


def public_file_url(file_id: str, filename: str) -> str:
    return f"/api/files/{file_id}/{filename}"


def find_uploaded_file(file_id: str) -> dict[str, Any]:
    if file_id in FILES:
        return FILES[file_id]
    candidates = sorted(UPLOAD_DIR.glob(f"{file_id}_*"))
    if candidates:
        path = candidates[0]
        storage_prefix = f"{file_id}_"
        filename = path.name[len(storage_prefix):] if path.name.startswith(storage_prefix) else path.name
        data = {"fileId": file_id, "filename": filename, "path": path}
        FILES[file_id] = data
        return data
    raise HTTPException(status_code=404, detail=f"fileId not found: {file_id}")


def task_result_path(task_id: str) -> Path:
    path = TASK_DIR / task_id / "result.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"task not found: {task_id}")
    return path


def task_status_path(task_id: str) -> Path:
    return TASK_DIR / task_id / "status.json"


EVALUATION_BATCH_LABEL = "全模型一键测试"
EVALUATION_BATCH_TYPES = {"asr", "clone"}
EVALUATION_BATCH_SUCCESS_STATUSES = {"completed", "success", "available", "computed"}
EVALUATION_BATCH_FAILURE_STATUSES = {"failed", "error", "cancelled", "unavailable"}
EVALUATION_BATCH_TERMINAL_STATUSES = EVALUATION_BATCH_SUCCESS_STATUSES | EVALUATION_BATCH_FAILURE_STATUSES


def _load_task_status_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _save_task_status_document(task_id: str, current: dict[str, Any], path: Path, now: str) -> None:
    current.setdefault("taskId", task_id)
    current.setdefault("createdAt", now)
    current["updatedAt"] = now
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _batch_storage_key(batch_type: str) -> str:
    return "asrBatches" if batch_type == "asr" else "cloneBatches"


def _batch_subtask_fields(batch_type: str) -> tuple[str, str, str, str, str]:
    if batch_type == "asr":
        return "asrTasks", "asrTask", "asrSubId", "asrRequest", "asrResult"
    return "cloneTasks", "cloneTask", "cloneSubId", "cloneRequest", "cloneResult"


def _normalized_status(value: Any) -> str:
    return str(value or "queued").strip().lower()


def _bounded_progress(value: Any) -> float:
    try:
        progress = float(value)
    except (TypeError, ValueError):
        return 0.0
    if progress != progress or progress in {float("inf"), float("-inf")}:
        return 0.0
    return min(1.0, max(0.0, progress))


def _batch_item_progress(item: dict[str, Any]) -> float:
    if _normalized_status(item.get("status")) in EVALUATION_BATCH_TERMINAL_STATUSES:
        return 1.0
    return _bounded_progress(item.get("progress"))


def _aggregate_evaluation_batch(batch: dict[str, Any], now: str) -> dict[str, Any]:
    items = [dict(item) for item in batch.get("items", []) if isinstance(item, dict)]
    try:
        declared_total = max(0, int(batch.get("totalCount") or 0))
    except (TypeError, ValueError):
        declared_total = 0
    total_count = max(declared_total, len(items))
    missing_count = max(0, total_count - len(items))
    statuses = [_normalized_status(item.get("status")) for item in items]
    completed_count = sum(status in EVALUATION_BATCH_SUCCESS_STATUSES for status in statuses)
    failed_count = sum(status in EVALUATION_BATCH_FAILURE_STATUSES for status in statuses)
    terminal_count = completed_count + failed_count
    progress_values = [_batch_item_progress(item) for item in items]
    progress_values.extend(0.0 for _ in range(missing_count))
    progress = min(progress_values) if progress_values else 0.0

    if total_count > 0 and terminal_count == total_count and missing_count == 0:
        if failed_count == 0:
            status = "completed"
        elif completed_count == 0:
            status = "failed"
        else:
            status = "partial_failed"
        progress = 1.0
    elif all(status == "queued" for status in statuses) and missing_count + len(items) == total_count:
        status = "queued"
    else:
        status = "running"

    elapsed_values: list[float] = []
    for item in items:
        try:
            elapsed = float(item.get("elapsedSec"))
        except (TypeError, ValueError):
            continue
        if elapsed == elapsed and elapsed not in {float("inf"), float("-inf")} and elapsed >= 0:
            elapsed_values.append(elapsed)
    finished_count = completed_count + failed_count
    if status == "queued":
        message = f"{EVALUATION_BATCH_LABEL}：等待执行 0/{total_count}"
    elif status == "running":
        message = f"{EVALUATION_BATCH_LABEL}：已结束 {finished_count}/{total_count}，失败 {failed_count}"
    elif status == "completed":
        message = f"{EVALUATION_BATCH_LABEL}：全部 {total_count} 项已完成"
    elif status == "partial_failed":
        message = f"{EVALUATION_BATCH_LABEL}：完成 {completed_count}/{total_count}，失败 {failed_count}"
    else:
        message = f"{EVALUATION_BATCH_LABEL}：全部 {failed_count} 项失败"

    first_error = next((item.get("error") for item in items if _normalized_status(item.get("status")) in EVALUATION_BATCH_FAILURE_STATUSES and item.get("error") is not None), None)
    batch.update(
        {
            "label": EVALUATION_BATCH_LABEL,
            "status": status,
            "progress": round(progress, 3),
            "message": message,
            "elapsedSec": round(max(elapsed_values), 3) if elapsed_values else 0.0,
            "completedCount": completed_count,
            "failedCount": failed_count,
            "totalCount": total_count,
            "error": first_error,
            "items": items,
            "updatedAt": now,
        }
    )
    return batch


def _find_subtask_snapshot(current: dict[str, Any], batch_type: str, subtask_id: str) -> dict[str, Any] | None:
    history_key, latest_key, sub_id_key, _, _ = _batch_subtask_fields(batch_type)
    for item in current.get(history_key, []):
        if isinstance(item, dict) and item.get(sub_id_key) == subtask_id:
            return item
    latest = current.get(latest_key)
    if isinstance(latest, dict) and latest.get(sub_id_key) == subtask_id:
        return latest
    return None


def _sync_evaluation_batch(current: dict[str, Any], updates: dict[str, Any], batch_type: str, now: str) -> None:
    _, _, sub_id_key, request_key, result_key = _batch_subtask_fields(batch_type)
    subtask_id = updates.get(sub_id_key)
    if not subtask_id:
        return
    snapshot = _find_subtask_snapshot(current, batch_type, str(subtask_id))
    if not snapshot:
        return
    request_payload = updates.get(request_key)
    if not isinstance(request_payload, dict):
        request_payload = snapshot.get(request_key)
    if not isinstance(request_payload, dict):
        return
    batch_id = str(request_payload.get("batchId") or "").strip()
    batch_item_id = str(request_payload.get("batchItemId") or "").strip()
    if not batch_id or not batch_item_id:
        return

    storage_key = _batch_storage_key(batch_type)
    batches = [dict(item) for item in current.get(storage_key, []) if isinstance(item, dict)]
    batch_index = next((index for index, item in enumerate(batches) if str(item.get("batchId") or "") == batch_id), None)
    if batch_index is None:
        return
    batch = batches[batch_index]
    items = [dict(item) for item in batch.get("items", []) if isinstance(item, dict)]
    item_index = next((index for index, item in enumerate(items) if str(item.get("batchItemId") or "") == batch_item_id), None)
    if item_index is None:
        return

    item = items[item_index]
    item[sub_id_key] = subtask_id
    item[request_key] = request_payload
    if not item.get("model") and request_payload.get("model"):
        item["model"] = request_payload.get("model")
    for field in ["status", "progress", "stage", "message", "elapsedSec", "error"]:
        if field in snapshot:
            item[field] = snapshot.get(field)
    if result_key in snapshot:
        item[result_key] = snapshot.get(result_key)
    item.setdefault("createdAt", batch.get("createdAt") or now)
    item["updatedAt"] = now
    if _normalized_status(item.get("status")) in EVALUATION_BATCH_TERMINAL_STATUSES:
        item["completedAt"] = now
    items[item_index] = item
    batch["items"] = items
    batches[batch_index] = _aggregate_evaluation_batch(batch, now)
    current[storage_key] = batches


def _mark_evaluation_batch_item_failed(task_id: str, batch_type: str, batch_id: str | None, batch_item_id: str | None, error: Any, message: str | None = None) -> bool:
    normalized_batch_id = str(batch_id or "").strip()
    normalized_item_id = str(batch_item_id or "").strip()
    if batch_type not in EVALUATION_BATCH_TYPES or not normalized_batch_id or not normalized_item_id:
        return False
    with TASK_STATUS_WRITE_LOCK:
        task_dir = TASK_DIR / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        path = task_status_path(task_id)
        current = _load_task_status_document(path)
        storage_key = _batch_storage_key(batch_type)
        batches = [dict(item) for item in current.get(storage_key, []) if isinstance(item, dict)]
        batch_index = next((index for index, item in enumerate(batches) if str(item.get("batchId") or "") == normalized_batch_id), None)
        if batch_index is None:
            return False
        batch = batches[batch_index]
        items = [dict(item) for item in batch.get("items", []) if isinstance(item, dict)]
        item_index = next((index for index, item in enumerate(items) if str(item.get("batchItemId") or "") == normalized_item_id), None)
        if item_index is None:
            return False
        now = utc_now_iso()
        item = items[item_index]
        error_message = message
        if not error_message and isinstance(error, dict):
            error_message = str(error.get("message") or "Evaluation request failed before submission")
        item.update(
            {
                "status": "failed",
                "progress": 1.0,
                "message": error_message or str(error or "Evaluation request failed before submission"),
                "elapsedSec": 0.0,
                "error": error,
                "updatedAt": now,
                "completedAt": now,
            }
        )
        items[item_index] = item
        batch["items"] = items
        batches[batch_index] = _aggregate_evaluation_batch(batch, now)
        current[storage_key] = batches
        _save_task_status_document(task_id, current, path, now)
        return True


def _json_response_error(response: JSONResponse) -> dict[str, Any]:
    try:
        payload = json.loads(response.body.decode("utf-8"))
    except Exception:
        payload = {}
    error = payload.get("error") if isinstance(payload, dict) else None
    return error if isinstance(error, dict) else {"code": "EVALUATION_REQUEST_FAILED", "message": "Evaluation request failed before submission"}


def _merge_subtask_status(current: dict[str, Any], updates: dict[str, Any], *, stage: str, key: str, history_key: str, result_key: str, sub_id_key: str) -> None:
    if updates.get("stage") != stage and result_key not in updates and sub_id_key not in updates:
        return
    now = utc_now_iso()
    subtask_id = updates.get(sub_id_key)
    if not subtask_id:
        previous = current.get(key)
        subtask_id = previous.get(sub_id_key) if isinstance(previous, dict) else None
    if not subtask_id:
        return
    history = [dict(item) for item in current.get(history_key, []) if isinstance(item, dict)]
    legacy = current.get(key)
    if isinstance(legacy, dict) and legacy.get(sub_id_key) and not any(item.get(sub_id_key) == legacy.get(sub_id_key) for item in history):
        history.append(dict(legacy))
    previous = next((item for item in history if item.get(sub_id_key) == subtask_id), None)
    if previous is None:
        if isinstance(legacy, dict) and legacy.get(sub_id_key) == subtask_id:
            previous = legacy
    subtask = dict(previous) if isinstance(previous, dict) else {}
    if not subtask.get("createdAt"):
        subtask["createdAt"] = now
    for field in ["status", "progress", "stage", "message", "elapsedSec", "error", "asrRequest", "cloneRequest"]:
        if field in updates:
            subtask[field] = updates.get(field)
    subtask["stage"] = stage
    if result_key in updates:
        subtask[result_key] = updates.get(result_key)
    subtask[sub_id_key] = subtask_id
    subtask["updatedAt"] = now
    replaced = False
    for index, item in enumerate(history):
        if item.get(sub_id_key) == subtask_id:
            history[index] = subtask
            replaced = True
            break
    if not replaced:
        history.append(subtask)
    current[history_key] = history
    current[key] = history[-1]


def write_task_status(task_id: str, **updates: Any) -> dict[str, Any]:
    if is_task_deleted(task_id):
        raise TaskCancelledError(f"task deleted: {task_id}")
    subtask_id = str(updates.get("asrSubId") or updates.get("cloneSubId") or "").strip()
    if subtask_id and is_subtask_deleted(task_id, subtask_id):
        return _load_task_status_document(task_status_path(task_id))
    with TASK_STATUS_WRITE_LOCK:
        if subtask_id and is_subtask_deleted(task_id, subtask_id):
            return _load_task_status_document(task_status_path(task_id))
        task_dir = TASK_DIR / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        path = task_status_path(task_id)
        current = _load_task_status_document(path)
        now = utc_now_iso()
        current.update(updates)
        _merge_subtask_status(current, updates, stage="asr_eval", key="asrTask", history_key="asrTasks", result_key="asrResult", sub_id_key="asrSubId")
        _merge_subtask_status(current, updates, stage="downstream_tts_eval", key="cloneTask", history_key="cloneTasks", result_key="cloneResult", sub_id_key="cloneSubId")
        _sync_evaluation_batch(current, updates, "asr", now)
        _sync_evaluation_batch(current, updates, "clone", now)
        _save_task_status_document(task_id, current, path, now)
        return current


def _latest_subtask_snapshot(current: dict[str, Any], history_key: str) -> dict[str, Any] | None:
    history = [item for item in current.get(history_key, []) if isinstance(item, dict)]
    return history[-1] if history else None


def _restore_protection_status(current: dict[str, Any], result: dict[str, Any] | None) -> None:
    if result is not None:
        current["status"] = result.get("status") or "completed"
        current["progress"] = result.get("progress") if result.get("progress") is not None else 1
        current["stage"] = result.get("stage") or "report_generation"
        current["message"] = result.get("message") or "任务已完成"
        current["elapsedSec"] = result.get("elapsedSec")
        current["error"] = result.get("error")
        return
    current["status"] = current.get("protectionStatus") or "queued"
    current["progress"] = current.get("protectionProgress") or 0
    current["stage"] = current.get("protectionStage") or "protect_generation"
    current["message"] = current.get("protectionMessage") or "保护任务等待执行"
    current["elapsedSec"] = current.get("protectionElapsedSec")
    current["error"] = current.get("protectionError")


def _remove_subtask_from_batches(current: dict[str, Any], batch_type: str, subtask_id: str, now: str) -> None:
    storage_key = _batch_storage_key(batch_type)
    _, _, sub_id_key, _, _ = _batch_subtask_fields(batch_type)
    next_batches: list[dict[str, Any]] = []
    for raw_batch in current.get(storage_key, []):
        if not isinstance(raw_batch, dict):
            continue
        batch = dict(raw_batch)
        items = [
            dict(item)
            for item in batch.get("items", [])
            if isinstance(item, dict) and str(item.get(sub_id_key) or "") != subtask_id
        ]
        if not items:
            continue
        batch["items"] = items
        batch["totalCount"] = len(items)
        next_batches.append(_aggregate_evaluation_batch(batch, now))
    current[storage_key] = next_batches


def _remove_subtask_status(task_id: str, batch_type: str, subtask_id: str) -> bool:
    history_key, latest_key, sub_id_key, request_key, result_key = _batch_subtask_fields(batch_type)
    with TASK_STATUS_WRITE_LOCK:
        path = task_status_path(task_id)
        current = _load_task_status_document(path)
        history = [dict(item) for item in current.get(history_key, []) if isinstance(item, dict)]
        legacy = current.get(latest_key)
        was_top_level = str(current.get(sub_id_key) or "") == subtask_id
        existed = any(str(item.get(sub_id_key) or "") == subtask_id for item in history)
        existed = existed or (isinstance(legacy, dict) and str(legacy.get(sub_id_key) or "") == subtask_id)
        filtered = [item for item in history if str(item.get(sub_id_key) or "") != subtask_id]
        current[history_key] = filtered
        latest = filtered[-1] if filtered else None
        if latest is None:
            current.pop(latest_key, None)
        else:
            current[latest_key] = latest
        if was_top_level:
            for field in (sub_id_key, request_key, result_key):
                current.pop(field, None)
        now = utc_now_iso()
        _remove_subtask_from_batches(current, batch_type, subtask_id, now)
        if was_top_level and current.get("stage") in {"asr_eval", "downstream_tts_eval"}:
            other_latest = _latest_subtask_snapshot(current, "cloneTasks" if batch_type == "asr" else "asrTasks")
            same_latest = _latest_subtask_snapshot(current, history_key)
            active_latest = next(
                (
                    item
                    for item in (other_latest, same_latest)
                    if isinstance(item, dict) and _normalized_status(item.get("status")) in {"queued", "running"}
                ),
                None,
            )
            if active_latest is not None:
                for field in ("status", "progress", "stage", "message", "elapsedSec", "error"):
                    current[field] = active_latest.get(field)
            else:
                result_path = TASK_DIR / task_id / "result.json"
                result = load_result(task_id) if result_path.exists() else None
                _restore_protection_status(current, result)
        _save_task_status_document(task_id, current, path, now)
        return existed


def _remove_clone_artifacts(task_dir: Path, clone_result: dict[str, Any]) -> None:
    clone_id = str(clone_result.get("cloneId") or "").strip()
    if not clone_id or not clone_id.startswith("clone_") or Path(clone_id).name != clone_id:
        return
    clones_dir = (task_dir / "clones").resolve()
    clone_dir = (clones_dir / clone_id).resolve()
    if clone_dir.parent != clones_dir:
        return
    if clone_dir.exists():
        shutil.rmtree(clone_dir)


def _remove_subtask_result(task_id: str, batch_type: str, subtask_id: str) -> tuple[bool, dict[str, Any] | None]:
    task_dir = TASK_DIR / task_id
    collection_key = "asrResults" if batch_type == "asr" else "cloneResults"
    sub_id_key = "asrSubId" if batch_type == "asr" else "cloneSubId"
    removed_results: list[dict[str, Any]] = []

    def update(result: dict[str, Any]) -> bool:
        values = [item for item in result.get(collection_key, []) if isinstance(item, dict)]
        removed_results.extend(item for item in values if str(item.get(sub_id_key) or "") == subtask_id)
        if not removed_results:
            return False
        result[collection_key] = [item for item in values if str(item.get(sub_id_key) or "") != subtask_id]
        details = result.setdefault("details", {})
        primary = result.setdefault("summary", {}).setdefault("primaryMetrics", {})
        if batch_type == "asr":
            remaining = result[collection_key]
            latest_asr = (remaining[-1].get("asr") if remaining else None) or {}
            details["asr"] = latest_asr or {"status": "unavailable", "reason": "尚未执行独立 ASR 测试"}
            result["asrModel"] = latest_asr.get("model") if latest_asr else None
            for key in ("wer", "cer"):
                if latest_asr.get(key) is None:
                    primary.pop(key, None)
                else:
                    primary[key] = latest_asr.get(key)
        else:
            remaining = result[collection_key]
            latest_clone = remaining[-1] if remaining else None
            latest_eval = latest_clone.get("cloneEval") if isinstance(latest_clone, dict) else None
            details["cloneEval"] = latest_eval or {"status": "unavailable", "reason": "尚未执行独立语音克隆测试"}
            downstream = details.setdefault("downstreamTts", {})
            if latest_clone is None:
                downstream.clear()
                downstream.update({"enabled": False, "status": "unavailable", "reason": "尚未执行独立语音克隆测试"})
            else:
                request = latest_clone.get("request") or {}
                downstream.update({
                    "enabled": True,
                    "ttsModel": request.get("model"),
                    "status": "computed",
                    "source": latest_clone.get("source"),
                    "lastCloneId": latest_clone.get("cloneId"),
                    "cloneText": request.get("text"),
                })
        refresh_result_scores(result)
        result["updatedAt"] = utc_now_iso()
        return True

    result_path = task_dir / "result.json"
    updated_result: dict[str, Any] | None = None
    changed = False
    if result_path.exists():
        updated_result, changed = update_result_safely(task_id, update)
    if batch_type == "clone":
        for clone_result in removed_results:
            _remove_clone_artifacts(task_dir, clone_result)
    history_path = task_dir / ("asr_results" if batch_type == "asr" else "clone_results") / f"{subtask_id}.json"
    history_path.unlink(missing_ok=True)
    latest_path = task_dir / ("asr_result.json" if batch_type == "asr" else "clone_result.json")
    remaining = [item for item in (updated_result or {}).get(collection_key, []) if isinstance(item, dict)]
    if remaining:
        latest_path.write_text(json.dumps(remaining[-1], ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        latest_path.unlink(missing_ok=True)
    return changed, updated_result


def _read_json_value(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _collect_string_field(value: Any, field: str, *, prefix: str) -> set[str]:
    collected: set[str] = set()
    if isinstance(value, dict):
        candidate = value.get(field)
        if isinstance(candidate, str) and candidate.startswith(prefix):
            collected.add(candidate)
        for nested in value.values():
            collected.update(_collect_string_field(nested, field, prefix=prefix))
    elif isinstance(value, list):
        for nested in value:
            collected.update(_collect_string_field(nested, field, prefix=prefix))
    return collected


def _evaluation_subtask_ids(
    task_id: str,
    batch_type: str,
    status: dict[str, Any],
    result: dict[str, Any] | None,
) -> set[str]:
    history_key, latest_key, sub_id_key, _, _ = _batch_subtask_fields(batch_type)
    prefix = "asr_" if batch_type == "asr" else "clone_"
    collection_key = "asrResults" if batch_type == "asr" else "cloneResults"
    values: list[Any] = [
        status.get(history_key),
        status.get(latest_key),
        status.get(_batch_storage_key(batch_type)),
        {sub_id_key: status.get(sub_id_key)},
        (result or {}).get(collection_key),
    ]
    task_dir = TASK_DIR / task_id
    latest_value = _read_json_value(task_dir / ("asr_result.json" if batch_type == "asr" else "clone_result.json"))
    if latest_value is not None:
        values.append(latest_value)
    history_dir = task_dir / ("asr_results" if batch_type == "asr" else "clone_results")
    if history_dir.exists():
        for path in history_dir.glob("*.json"):
            values.append(_read_json_value(path))
            if path.stem.startswith(prefix):
                values.append({sub_id_key: path.stem})
    collected: set[str] = set()
    for value in values:
        collected.update(_collect_string_field(value, sub_id_key, prefix=prefix))
    runtime_prefix = f"{task_id}:{prefix}"
    with TASK_REGISTRY_LOCK:
        for registry in (TASK_CANCEL_EVENTS, TASK_THREADS, TASK_PROCESSES):
            collected.update(key[len(task_id) + 1:] for key in registry if key.startswith(runtime_prefix))
    return collected


def _referenced_asr_subtask_ids(task_id: str, status: dict[str, Any], result: dict[str, Any] | None) -> set[str]:
    task_dir = TASK_DIR / task_id
    clone_values: list[Any] = [
        status.get("cloneTasks"),
        status.get("cloneTask"),
        status.get("cloneBatches"),
        status.get("cloneRequest"),
        status.get("cloneResult"),
        (result or {}).get("cloneResults"),
        ((result or {}).get("details") or {}).get("downstreamTts"),
    ]
    latest_clone = _read_json_value(task_dir / "clone_result.json")
    if latest_clone is not None:
        clone_values.append(latest_clone)
    clone_history_dir = task_dir / "clone_results"
    if clone_history_dir.exists():
        clone_values.extend(_read_json_value(path) for path in clone_history_dir.glob("*.json"))
    references: set[str] = set()
    for value in clone_values:
        references.update(_collect_string_field(value, "annotationAsrSubId", prefix="asr_"))
    return references


def _evaluation_sidecar_results(task_id: str, batch_type: str) -> list[dict[str, Any]]:
    sub_id_key = "asrSubId" if batch_type == "asr" else "cloneSubId"
    task_dir = TASK_DIR / task_id
    history_dir = task_dir / ("asr_results" if batch_type == "asr" else "clone_results")
    latest_path = task_dir / ("asr_result.json" if batch_type == "asr" else "clone_result.json")
    paths = sorted(history_dir.glob("*.json")) if history_dir.exists() else []
    if latest_path.exists():
        paths.append(latest_path)

    results: list[dict[str, Any]] = []
    indexes: dict[str, int] = {}
    for path in paths:
        payload = _read_json_value(path)
        if not isinstance(payload, dict):
            continue
        subtask_id = str(payload.get(sub_id_key) or "").strip()
        if not subtask_id:
            continue
        if subtask_id in indexes:
            results[indexes[subtask_id]] = payload
        else:
            indexes[subtask_id] = len(results)
            results.append(payload)
    return results


def _latest_active_evaluation_snapshot(current: dict[str, Any]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for history_key in ("asrTasks", "cloneTasks"):
        candidates.extend(item for item in current.get(history_key, []) if isinstance(item, dict))
    for storage_key in ("asrBatches", "cloneBatches"):
        for batch in current.get(storage_key, []):
            if isinstance(batch, dict):
                candidates.extend(item for item in batch.get("items", []) if isinstance(item, dict))
    active = [item for item in candidates if _normalized_status(item.get("status")) in {"queued", "running"}]
    return active[-1] if active else None


def _retain_evaluation_status(task_id: str, batch_type: str, keep_subtask_ids: set[str]) -> tuple[set[str], set[str]]:
    history_key, latest_key, sub_id_key, request_key, result_key = _batch_subtask_fields(batch_type)
    storage_key = _batch_storage_key(batch_type)
    path = task_status_path(task_id)
    if not path.exists():
        return set(), set()
    with TASK_STATUS_WRITE_LOCK:
        current = _load_task_status_document(path)
        existing_ids = _collect_string_field(
            [current.get(history_key), current.get(latest_key), current.get(storage_key), {sub_id_key: current.get(sub_id_key)}],
            sub_id_key,
            prefix="asr_" if batch_type == "asr" else "clone_",
        )
        history = [dict(item) for item in current.get(history_key, []) if isinstance(item, dict)]
        legacy = current.get(latest_key)
        if isinstance(legacy, dict):
            legacy_subtask_id = str(legacy.get(sub_id_key) or "").strip()
            if legacy_subtask_id:
                matching_index = next(
                    (index for index, item in enumerate(history) if str(item.get(sub_id_key) or "") == legacy_subtask_id),
                    None,
                )
                if matching_index is None:
                    history.append(dict(legacy))
                else:
                    history[matching_index] = {**history[matching_index], **legacy}
        kept_history = [item for item in history if str(item.get(sub_id_key) or "") in keep_subtask_ids]
        current[history_key] = kept_history
        latest = kept_history[-1] if kept_history else None
        if latest is None:
            current.pop(latest_key, None)
            for field in (sub_id_key, request_key, result_key):
                current.pop(field, None)
        else:
            current[latest_key] = latest
            current[sub_id_key] = latest.get(sub_id_key)
            for field in (request_key, result_key):
                if field in latest:
                    current[field] = latest.get(field)
                else:
                    current.pop(field, None)

        removed_batch_ids: set[str] = set()
        kept_batches: list[dict[str, Any]] = []
        now = utc_now_iso()
        for raw_batch in current.get(storage_key, []):
            if not isinstance(raw_batch, dict):
                continue
            batch = dict(raw_batch)
            items = [
                dict(item)
                for item in batch.get("items", [])
                if isinstance(item, dict) and str(item.get(sub_id_key) or "") in keep_subtask_ids
            ]
            if not items:
                batch_id = str(batch.get("batchId") or "").strip()
                if batch_id:
                    removed_batch_ids.add(batch_id)
                continue
            batch["items"] = items
            batch["totalCount"] = len(items)
            kept_batches.append(_aggregate_evaluation_batch(batch, now))
        current[storage_key] = kept_batches

        relevant_stage = "asr_eval" if batch_type == "asr" else "downstream_tts_eval"
        if current.get("stage") == relevant_stage:
            active = _latest_active_evaluation_snapshot(current)
            if active is not None:
                for field in ("status", "progress", "stage", "message", "elapsedSec", "error"):
                    current[field] = active.get(field)
            else:
                result_path = TASK_DIR / task_id / "result.json"
                protection_result = load_result(task_id) if result_path.exists() else None
                _restore_protection_status(current, protection_result)
        _save_task_status_document(task_id, current, path, now)
        return existing_ids, removed_batch_ids


def _strip_metric_source_prefix(result: dict[str, Any], prefix: str) -> None:
    for container in (result.get("metricSources"), (result.get("summary") or {}).get("metricSources")):
        if not isinstance(container, dict):
            continue
        for key in [key for key in container if str(key).startswith(prefix)]:
            container.pop(key, None)


def _retain_evaluation_results(task_id: str, batch_type: str, keep_subtask_ids: set[str]) -> tuple[set[str], dict[str, Any] | None]:
    task_dir = TASK_DIR / task_id
    result_path = task_dir / "result.json"
    if not result_path.exists():
        return set(), None
    collection_key = "asrResults" if batch_type == "asr" else "cloneResults"
    sub_id_key = "asrSubId" if batch_type == "asr" else "cloneSubId"
    removed_ids: set[str] = set()
    removed_clone_results: list[dict[str, Any]] = []
    sidecar_results = _evaluation_sidecar_results(task_id, batch_type)

    def update(result: dict[str, Any]) -> bool:
        values = [item for item in result.get(collection_key, []) if isinstance(item, dict)]
        merged_values: list[dict[str, Any]] = []
        indexes: dict[str, int] = {}
        for item in values:
            subtask_id = str(item.get(sub_id_key) or "").strip()
            if not subtask_id:
                continue
            indexes[subtask_id] = len(merged_values)
            merged_values.append(item)
        for item in sidecar_results:
            subtask_id = str(item.get(sub_id_key) or "").strip()
            if not subtask_id or subtask_id in indexes:
                continue
            indexes[subtask_id] = len(merged_values)
            merged_values.append(item)
        retained = [item for item in merged_values if str(item.get(sub_id_key) or "") in keep_subtask_ids]
        removed = [item for item in merged_values if str(item.get(sub_id_key) or "") not in keep_subtask_ids]
        removed_ids.update(str(item.get(sub_id_key)) for item in removed if item.get(sub_id_key))
        if batch_type == "clone":
            removed_clone_results.extend(removed)
        result[collection_key] = retained
        details = result.setdefault("details", {})
        primary = result.setdefault("summary", {}).setdefault("primaryMetrics", {})
        if batch_type == "asr":
            latest_asr = (retained[-1].get("asr") if retained else None) or {}
            details["asr"] = latest_asr or {"status": "unavailable", "reason": "尚未执行独立 ASR 测试"}
            result["asrModel"] = latest_asr.get("model") if latest_asr else None
            result.pop("asrEval", None)
            for key in ("wer", "cer"):
                if latest_asr.get(key) is None:
                    primary.pop(key, None)
                else:
                    primary[key] = latest_asr.get(key)
            _strip_metric_source_prefix(result, "asrEval")
        else:
            latest_clone = retained[-1] if retained else None
            latest_eval = latest_clone.get("cloneEval") if isinstance(latest_clone, dict) else None
            details["cloneEval"] = latest_eval or {"status": "unavailable", "reason": "尚未执行独立语音克隆测试"}
            result.pop("cloneEval", None)
            downstream = details.setdefault("downstreamTts", {})
            if latest_clone is None:
                downstream.clear()
                downstream.update({"enabled": False, "status": "unavailable", "reason": "尚未执行独立语音克隆测试"})
            else:
                request = latest_clone.get("request") or {}
                downstream.clear()
                downstream.update({
                    "enabled": True,
                    "ttsModel": request.get("model"),
                    "status": "computed",
                    "source": latest_clone.get("source"),
                    "lastCloneId": latest_clone.get("cloneId"),
                    "cloneText": request.get("text"),
                })
            _strip_metric_source_prefix(result, "cloneEval")
        refresh_result_scores(result)
        result["updatedAt"] = utc_now_iso()
        return True

    def cleanup_removed_clone_artifacts(_: dict[str, Any]) -> None:
        for clone_result in removed_clone_results:
            _remove_clone_artifacts(task_dir, clone_result)

    updated_result, _ = update_result_safely(
        task_id,
        update,
        after_write=cleanup_removed_clone_artifacts if batch_type == "clone" else None,
    )
    return removed_ids, updated_result


def _cleanup_evaluation_artifacts(
    task_id: str,
    batch_type: str,
    keep_subtask_ids: set[str],
    updated_result: dict[str, Any] | None,
) -> None:
    task_dir = TASK_DIR / task_id
    history_dir = task_dir / ("asr_results" if batch_type == "asr" else "clone_results")
    if history_dir.exists():
        for path in history_dir.glob("*.json"):
            payload = _read_json_value(path)
            sub_id_key = "asrSubId" if batch_type == "asr" else "cloneSubId"
            subtask_id = str((payload or {}).get(sub_id_key) or path.stem) if isinstance(payload, dict) else path.stem
            if subtask_id not in keep_subtask_ids:
                path.unlink(missing_ok=True)
        if not any(history_dir.iterdir()):
            history_dir.rmdir()

    collection_key = "asrResults" if batch_type == "asr" else "cloneResults"
    remaining = [item for item in (updated_result or {}).get(collection_key, []) if isinstance(item, dict)]
    latest_path = task_dir / ("asr_result.json" if batch_type == "asr" else "clone_result.json")
    if remaining:
        latest_path.write_text(json.dumps(remaining[-1], ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        latest_path.unlink(missing_ok=True)
    if batch_type == "clone":
        clone_dir = task_dir / "clones"
        if clone_dir.exists():
            shutil.rmtree(clone_dir)


def read_task_status(task_id: str) -> dict[str, Any]:
    task_dir = TASK_DIR / task_id
    result_path = TASK_DIR / task_id / "result.json"
    status_path = task_status_path(task_id)
    status: dict[str, Any] | None = None
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            status = None
        if status and (
            status.get("stage") in {"asr_eval", "downstream_tts_eval"}
            or status.get("status") in {"queued", "running", "failed", "error", "cancelled"}
            or bool(status.get("asrTasks") or status.get("cloneTasks"))
            or bool(status.get("asrBatches") or status.get("cloneBatches"))
            or not result_path.exists()
        ):
            return status
    if result_path.exists():
        result = load_result(task_id)
        return {
            "taskId": task_id,
            "status": result.get("status", "completed"),
            "progress": 1,
            "stage": "report_generation",
            "message": "任务已完成",
            "createdAt": result.get("createdAt"),
            "updatedAt": result.get("completedAt"),
            "elapsedSec": result.get("elapsedSec"),
            "error": None,
        }
    if status:
        return status
    if task_dir.exists():
        now = utc_now_iso()
        return {
            "taskId": task_id,
            "status": "queued",
            "progress": 0.05,
            "stage": "file_preprocess",
            "message": "Task status is initializing",
            "createdAt": now,
            "updatedAt": now,
            "error": None,
        }
    raise HTTPException(status_code=404, detail=f"task not found: {task_id}")


def request_id() -> str:
    return f"req_{uuid.uuid4().hex[:12]}"


def structured_error(
    *,
    code: str,
    message: str,
    status_code: int,
    request_id_value: str,
    task_id: str | None = None,
    stage: str = "api",
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "requestId": request_id_value,
                "taskId": task_id,
                "stage": stage,
                "details": details or {},
            }
        },
    )


def write_task_log(task_id: str | None, payload: dict[str, Any]) -> None:
    target = LOG_DIR / f"{task_id or 'api'}.log"
    with target.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, default=str))
        file.write("\n")


def _model_values(options: list[Any] | None) -> set[str]:
    values: set[str] = set()
    for option in options or []:
        if isinstance(option, str):
            values.add(option)
        elif isinstance(option, dict):
            for key in ("value", "backendValue", "label"):
                value = option.get(key)
                if isinstance(value, str) and value:
                    values.add(value)
    return values


def _canonical_supported(value: Any, allowed: set[str]) -> str | None:
    raw = str(value or "").strip()
    for item in allowed:
        if raw.lower() == item.lower():
            return item
    return None


def _legacy_weight_error(config: dict[str, Any], new_key: str, legacy_key: str, min_acceptable: float | None, max_acceptable: float | None = None) -> dict[str, Any] | None:
    if new_key in config or legacy_key not in config:
        return None
    try:
        value = float(config.get(legacy_key))
    except (TypeError, ValueError):
        return {"field": legacy_key, "value": config.get(legacy_key), "reason": f"{legacy_key} is deprecated; use {new_key}."}
    if min_acceptable is not None and value < min_acceptable:
        return {"field": legacy_key, "value": value, "reason": f"{legacy_key} looks like a UI-normalized value; use {new_key} with real algorithm scale."}
    if max_acceptable is not None and value > max_acceptable:
        return {"field": legacy_key, "value": value, "reason": f"{legacy_key} is outside the formal algorithm scale; use {new_key}."}
    return None


def validate_protection_config(payload: ProtectTaskRequest, req_id: str) -> JSONResponse | None:
    config = runtime_config()
    models = config.get("models") or {}
    semantic = payload.semantic or {}
    timbre = payload.timbre or {}
    asr_models = semantic.get("asrModels") if isinstance(semantic.get("asrModels"), list) else None
    asr_model = semantic.get("asrModel")
    requested_asr = [str(item) for item in (asr_models or [asr_model]) if item]
    feature_models = timbre.get("encoders") or []
    allowed_asr = _model_values(models.get("asr"))
    allowed_feature = _model_values(models.get("feature"))
    allowed_semantic = _model_values(models.get("semantic"))
    unsupported_asr = [item for item in requested_asr if allowed_asr and _canonical_supported(item, allowed_asr) is None]
    if unsupported_asr:
        return structured_error(
            code="UNSUPPORTED_ASR_MODEL",
            message="ASR 模型不在后端支持配置中。",
            status_code=400,
            request_id_value=req_id,
            stage="file_preprocess",
            details={"asrModels": unsupported_asr, "supported": sorted(allowed_asr)},
        )
    semantic_models = semantic.get("encoders") or []
    unsupported_semantic = [str(item) for item in semantic_models if allowed_semantic and _canonical_supported(item, allowed_semantic) is None]
    if unsupported_semantic:
        return structured_error(
            code="UNSUPPORTED_SEMANTIC_ENCODER",
            message="语义编码器不在后端支持配置中。",
            status_code=400,
            request_id_value=req_id,
            stage="file_preprocess",
            details={"semanticEncoders": unsupported_semantic, "supported": sorted(allowed_semantic)},
        )
    unsupported_features = [str(item) for item in feature_models if allowed_feature and _canonical_supported(item, allowed_feature) is None]
    if unsupported_features:
        return structured_error(
            code="UNSUPPORTED_FEATURE_MODEL",
            message="身份编码器不在后端支持配置中。",
            status_code=400,
            request_id_value=req_id,
            stage="file_preprocess",
            details={"featureModels": unsupported_features, "supported": sorted(allowed_feature)},
        )
    timbre_weight_check = dict(timbre)
    if "weightIdentity" in timbre_weight_check and "weightFeature" not in timbre_weight_check:
        timbre_weight_check["weightFeature"] = timbre_weight_check.get("weightIdentity")
    legacy_errors = [
        _legacy_weight_error(semantic, "weightSemantic", "lambdaSemantic", 1.0),
        _legacy_weight_error(timbre_weight_check, "weightFeature", "lambdaTimbre", 1.0),
        _legacy_weight_error(psychoacoustic := (payload.psychoacoustic or {}), "weightPsy", "lambdaPsy", None, 0.01),
        _legacy_weight_error(payload.optimization or {}, "weightL2", "lambdaL2", 0.05),
    ]
    legacy_errors = [item for item in legacy_errors if item]
    if legacy_errors:
        return structured_error(
            code="DEPRECATED_NORMALIZED_WEIGHT",
            message="请求包含旧版 lambda 字段或归一化权重，不能作为真实算法 weight 上传。",
            status_code=400,
            request_id_value=req_id,
            stage="file_preprocess",
            details={"legacyFields": legacy_errors, "requiredFields": ["weightIdentity", "weightSemantic", "weightPsy", "weightL2"]},
        )
    steps = ((payload.optimization or {}).get("steps"))
    steps_range = (config.get("ranges") or {}).get("steps") or {}
    try:
        steps_value = int(steps)
    except (TypeError, ValueError):
        steps_value = None
    if steps_value is not None and not (int(steps_range.get("min", 1)) <= steps_value <= int(steps_range.get("max", 100))):
        return structured_error(
            code="UNSUPPORTED_STEPS",
            message="优化步数超出后端支持范围。",
            status_code=400,
            request_id_value=req_id,
            stage="file_preprocess",
            details={"steps": steps_value, "range": steps_range},
        )
    return None


def validate_clone_config(payload: CloneVoiceRequest, req_id: str, task_id: str) -> JSONResponse | None:
    config = runtime_config()
    models = config.get("models") or {}
    clone = config.get("clone") or {}
    tts_options = models.get("tts") or []
    allowed_models = _model_values(tts_options)
    allowed_languages = set(str(item) for item in clone.get("languages") or [])
    allowed_speeds = set(float(item) for item in clone.get("speeds") or [])
    model = payload.model or ""
    language = payload.language or ""
    speed = float(payload.speed if payload.speed is not None else (clone.get("defaults") or {}).get("speed", 1.0))
    if allowed_models and _canonical_supported(model, allowed_models) is None:
        return structured_error(
            code="UNSUPPORTED_TTS_MODEL",
            message="克隆模型不在后端支持配置中。",
            status_code=400,
            request_id_value=req_id,
            task_id=task_id,
            stage="downstream_tts_eval",
            details={"model": model, "supported": sorted(allowed_models)},
        )
    selected_model = next(
        (
            option
            for option in tts_options
            if isinstance(option, dict)
            and any(
                model.lower() == str(option.get(key) or "").lower()
                for key in ("value", "backendValue", "label")
            )
        ),
        None,
    )
    if selected_model is not None and selected_model.get("status") != "available":
        return structured_error(
            code="TTS_MODEL_UNAVAILABLE",
            message="所选克隆模型当前不可在线执行。",
            status_code=409,
            request_id_value=req_id,
            task_id=task_id,
            stage="downstream_tts_eval",
            details={
                "model": model,
                "status": selected_model.get("status"),
                "reason": selected_model.get("reason") or "模型运行环境尚未就绪",
            },
        )
    model_languages = set(supported_tts_languages(model))
    if model_languages and language and language not in model_languages:
        return structured_error(
            code="UNSUPPORTED_TTS_LANGUAGE",
            message="克隆语言不在所选 TTS 模型支持范围中。",
            status_code=400,
            request_id_value=req_id,
            task_id=task_id,
            stage="downstream_tts_eval",
            details={"model": model, "language": language, "supported": sorted(model_languages)},
        )
    if allowed_languages and _canonical_supported(language, allowed_languages) is None:
        return structured_error(
            code="UNSUPPORTED_TTS_LANGUAGE",
            message="克隆语言不在后端支持配置中。",
            status_code=400,
            request_id_value=req_id,
            task_id=task_id,
            stage="downstream_tts_eval",
            details={"language": language, "supported": sorted(allowed_languages)},
        )
    if allowed_speeds and speed not in allowed_speeds:
        return structured_error(
            code="UNSUPPORTED_TTS_SPEED",
            message="克隆语速不在后端支持配置中。",
            status_code=400,
            request_id_value=req_id,
            task_id=task_id,
            stage="downstream_tts_eval",
            details={"speed": speed, "supported": sorted(allowed_speeds)},
        )
    return None


def _clear_clone_annotation_fields(payload: dict[str, Any]) -> None:
    payload.update(
        {
            "annotationSource": None,
            "speakerPrompt": None,
            "originalSpeakerPrompt": None,
            "protectedSpeakerPrompt": None,
            "annotationAsrSubId": None,
            "annotationAsrModel": None,
            "annotationCreatedAt": None,
        }
    )


def resolve_clone_annotation(task_id: str, payload: CloneVoiceRequest, req_id: str) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    resolved = payload.model_dump()
    if not tts_model_requires_reference_text(payload.model):
        _clear_clone_annotation_fields(resolved)
        return resolved, None

    annotation_source = str(payload.annotationSource or "manual").strip().lower()
    if annotation_source not in {"manual", "asr"}:
        return None, structured_error(
            code="UNSUPPORTED_ANNOTATION_SOURCE",
            message="克隆参考音频标注来源必须是人工标注或 ASR 标注。",
            status_code=400,
            request_id_value=req_id,
            task_id=task_id,
            stage="downstream_tts_eval",
            details={"annotationSource": annotation_source, "supported": ["manual", "asr"]},
        )
    resolved["annotationSource"] = annotation_source
    if annotation_source == "manual":
        manual_text = str(payload.speakerPrompt or "").strip() or None
        if manual_text is None:
            return None, structured_error(
                code="REFERENCE_TEXT_REQUIRED",
                message="所选克隆模型需要参考音频对应文本，请填写人工标注或选择已有 ASR 标注。",
                status_code=400,
                request_id_value=req_id,
                task_id=task_id,
                stage="downstream_tts_eval",
                details={"model": payload.model, "annotationSource": annotation_source},
            )
        resolved["speakerPrompt"] = manual_text
        resolved["originalSpeakerPrompt"] = manual_text
        resolved["protectedSpeakerPrompt"] = manual_text
        return resolved, None

    result = load_result(task_id)
    status = read_task_status(task_id)
    candidates: list[dict[str, Any]] = []
    for item in result.get("asrResults") or []:
        if isinstance(item, dict):
            candidates.append(item)
    for item in status.get("asrTasks") or []:
        if isinstance(item, dict):
            candidates.append(item)
    legacy_task = status.get("asrTask")
    if isinstance(legacy_task, dict):
        candidates.append(legacy_task)

    requested_sub_id = str(payload.annotationAsrSubId or "").strip()
    if requested_sub_id:
        candidates = [item for item in candidates if str(item.get("asrSubId") or "") == requested_sub_id]
    candidates.sort(key=lambda item: str(item.get("createdAt") or item.get("updatedAt") or ""), reverse=True)
    for item in candidates:
        asr_result = item.get("asrResult") if isinstance(item.get("asrResult"), dict) else item
        asr = asr_result.get("asr") if isinstance(asr_result, dict) and isinstance(asr_result.get("asr"), dict) else {}
        original_text = str(asr.get("originalText") or "").strip()
        protected_text = str(asr.get("protectedText") or "").strip()
        asr_sub_id = str(item.get("asrSubId") or asr_result.get("asrSubId") or "") if isinstance(asr_result, dict) else ""
        if not original_text or not protected_text or not asr_sub_id:
            continue
        resolved.update(
            {
                "speakerPrompt": original_text,
                "originalSpeakerPrompt": original_text,
                "protectedSpeakerPrompt": protected_text,
                "annotationAsrSubId": asr_sub_id,
                "annotationAsrModel": asr.get("model") or asr.get("asrModel"),
                "annotationCreatedAt": item.get("createdAt") or asr_result.get("createdAt"),
            }
        )
        return resolved, None

    return None, structured_error(
        code="ASR_ANNOTATION_NOT_FOUND",
        message="当前保护任务还没有同时包含原始音频和保护音频转写的 ASR 标注，请先完成 ASR 测试。",
        status_code=409,
        request_id_value=req_id,
        task_id=task_id,
        stage="downstream_tts_eval",
        details={"annotationAsrSubId": requested_sub_id or None},
    )


def validate_asr_eval_config(payload: AsrEvalRequest, req_id: str, task_id: str) -> JSONResponse | None:
    allowed_asr = _model_values((runtime_config().get("models") or {}).get("asr"))
    if allowed_asr and _canonical_supported(payload.model, allowed_asr) is None:
        return structured_error(
            code="UNSUPPORTED_ASR_MODEL",
            message="ASR 模型不在后端支持配置中。",
            status_code=400,
            request_id_value=req_id,
            task_id=task_id,
            stage="asr_eval",
            details={"asrModel": payload.model, "supported": sorted(allowed_asr)},
        )
    return None


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    req_id = request.headers.get("x-request-id") or request_id()
    detail = exc.detail if isinstance(exc.detail, str) else "请求失败"
    code = "TASK_NOT_FOUND" if exc.status_code == 404 and "task" in detail.lower() else "API_ERROR"
    if exc.status_code == 404 and "fileId" in detail:
        code = "INPUT_FILE_NOT_FOUND"
    if exc.status_code == 400 and "fileId" in detail:
        code = "MISSING_FILE_ID"
    return structured_error(
        code=code,
        message=detail,
        status_code=exc.status_code,
        request_id_value=req_id,
        stage="api",
        details={"path": str(request.url.path)},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    req_id = request.headers.get("x-request-id") or request_id()
    write_task_log(
        None,
        {
            "requestId": req_id,
            "currentStage": "api",
            "path": str(request.url.path),
            "exceptionType": type(exc).__name__,
            "exceptionMessage": str(exc),
            "stackTrace": traceback.format_exc(),
        },
    )
    return structured_error(
        code="INTERNAL_ERROR",
        message="后端内部错误，请查看日志。",
        status_code=500,
        request_id_value=req_id,
        stage="api",
        details={"exceptionType": type(exc).__name__},
    )


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in {float("inf"), float("-inf")} else None


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _frontend_audio(meta: dict[str, Any] | None, fallback_name: str) -> dict[str, Any]:
    meta = meta or {}
    stored_filename = str(meta.get("filename") or fallback_name)
    display_filename = re.sub(r"^(?:file_[0-9a-fA-F]{12}_)+", "", stored_filename) or stored_filename
    return {
        "fileId": meta.get("fileId"),
        "filename": display_filename,
        "durationSec": meta.get("durationSec") or meta.get("duration"),
        "duration": meta.get("duration") or meta.get("durationSec"),
        "sampleRate": meta.get("sampleRate"),
        "channels": meta.get("channels"),
        "bitDepth": meta.get("bitDepth"),
        "sizeBytes": meta.get("sizeBytes") or 0,
        "format": meta.get("format") or Path(display_filename).suffix.lstrip(".").upper() or "AUDIO",
        "src": meta.get("src"),
        "audioUrl": meta.get("audioUrl"),
        "downloadUrl": meta.get("downloadUrl"),
        "uploadedAt": meta.get("uploadedAt"),
        "fingerprint": meta.get("fingerprint"),
    }


def _frontend_clone(clone: dict[str, Any]) -> dict[str, Any]:
    clone_eval = clone.get("cloneEval") if isinstance(clone.get("cloneEval"), dict) else {}

    def clone_metric(key: str) -> Any:
        nested = clone_eval.get(key)
        return nested if nested is not None else clone.get(key)

    return {
        "cloneId": clone.get("cloneId"),
        "cloneSubId": clone.get("cloneSubId"),
        "taskId": clone.get("taskId"),
        "status": clone.get("status", "partial"),
        "source": clone.get("source"),
        "message": clone.get("message"),
        "request": clone.get("request") or {},
        "originalCloneAudio": _frontend_audio(clone.get("originalCloneAudio"), "original_clone.wav"),
        "protectedCloneAudio": _frontend_audio(clone.get("protectedCloneAudio"), "protected_clone.wav"),
        "cloneEval": clone_eval or None,
        "directSimilarity": clone_metric("directSimilarity"),
        "originalSimilarity": clone_metric("originalSimilarity"),
        "protectedSimilarity": clone_metric("protectedSimilarity"),
        "similarityDropRate": clone_metric("similarityDropRate"),
        "embeddingDistanceBefore": clone_metric("embeddingDistanceBefore"),
        "embeddingDistanceAfter": clone_metric("embeddingDistanceAfter"),
        "embeddingDistanceDelta": clone_metric("embeddingDistanceDelta"),
        "embeddingDistanceIncreaseRate": clone_metric("embeddingDistanceIncreaseRate"),
        "cloneIdentityScore": clone_metric("cloneIdentityScore"),
        "identityBaselineWeight": clone_metric("identityBaselineWeight"),
        "cloneSemanticScore": clone_metric("cloneSemanticScore"),
        "semanticBaselineWeight": clone_metric("semanticBaselineWeight"),
        "clonePairPesq": clone_metric("clonePairPesq"),
        "clonePairStoi": clone_metric("clonePairStoi"),
        "cloneQualityBefore": clone_metric("cloneQualityBefore"),
        "cloneQualityAfter": clone_metric("cloneQualityAfter"),
        "cloneQualityDropRate": clone_metric("cloneQualityDropRate"),
        "clonePesqDegradationScore": clone_metric("clonePesqDegradationScore"),
        "cloneStoiDegradationScore": clone_metric("cloneStoiDegradationScore"),
        "cloneDnsMosDegradationScore": clone_metric("cloneDnsMosDegradationScore"),
        "cloneQualityComponents": clone_metric("cloneQualityComponents"),
        "cloneQualityRawScore": clone_metric("cloneQualityRawScore"),
        "cloneQualityRelevance": clone_metric("cloneQualityRelevance"),
        "cloneQualityScore": clone_metric("cloneQualityScore"),
        "qualityBaselineWeight": clone_metric("qualityBaselineWeight"),
        "cloneConfidenceBefore": clone.get("cloneConfidenceBefore"),
        "cloneConfidenceAfter": clone.get("cloneConfidenceAfter"),
        "cloneConfidenceDropRate": clone.get("cloneConfidenceDropRate"),
        "cloneRadar": clone.get("cloneRadar"),
        "cloneTrend": clone.get("cloneTrend"),
        "cloneDefenseScore": clone.get("cloneDefenseScore"),
        "createdAt": clone.get("createdAt"),
        "fineTune": clone.get("fineTune"),
    }


def frontend_result(result: dict[str, Any]) -> dict[str, Any]:
    refresh_result_scores(result)
    summary = result.get("summary") or {}
    primary = summary.get("primaryMetrics") or {}
    details = result.get("details") or {}
    audio = result.get("audio") or {}
    score = _number(summary.get("score"))
    snr = _number(_coalesce(primary.get("snr"), (details.get("perception") or {}).get("snr")))
    pesq = _number(_coalesce(primary.get("pesq"), (details.get("perception") or {}).get("pesq")))
    sim_after = _number(_coalesce(primary.get("speakerSimilarity"), (details.get("speaker") or {}).get("simOriginalProtected")))
    sim_before = None
    asr = details.get("asr") or {}
    semantic_eval = details.get("semantic") if isinstance(details.get("semantic"), dict) else None
    generation = details.get("generation") or {}
    perception = details.get("perception") or {}
    charts = result.get("charts") or {}
    metric_sources = summary.get("metricSources") or result.get("metricSources") or {}
    request = result.get("request") or {}
    optimization = request.get("optimization") or {}
    loss_final = generation.get("lossFinal")
    loss_weights = generation.get("lossWeights") or {}
    asr_status = asr.get("status")
    asr_has_result = asr_status in {"available", "computed", "partial", "completed", "success"}
    asr_eval = None
    if asr_has_result:
        asr_eval = {
            "model": asr.get("model"),
            "asrModel": asr.get("model"),
            "language": asr.get("language"),
            "referenceText": asr.get("referenceText"),
            "originalText": asr.get("cleanTranscription"),
            "protectedText": asr.get("protectedTranscription"),
            "wer": asr.get("wer"),
            "cer": asr.get("cer"),
            "substituteRate": (asr.get("breakdown") or {}).get("substituteRate"),
            "insertRate": (asr.get("breakdown") or {}).get("insertRate"),
            "deleteRate": (asr.get("breakdown") or {}).get("deleteRate"),
            "editCounts": asr.get("editCounts"),
            "errorShares": asr.get("errorShares"),
            "metricLevel": asr.get("metricLevel"),
            "tokenErrorRate": asr.get("tokenErrorRate"),
            "tokenChangeRate": asr.get("tokenChangeRate"),
            "semanticDrift": asr.get("semanticDrift"),
            "asrProtectionScore": asr.get("asrProtectionScore"),
            "diffOps": asr.get("diffOps"),
            "trend": asr.get("trend"),
            "createdAt": asr.get("createdAt") or result.get("updatedAt"),
            "status": asr_status,
        }
    clone_results = [_frontend_clone(item) for item in result.get("cloneResults", [])]
    latest_clone = clone_results[-1] if clone_results else None
    clone_eval = None
    if latest_clone:
        clone_eval = latest_clone.get("cloneEval") or {
            "cloneModel": (latest_clone.get("request") or {}).get("model"),
            "speakerEvalModel": latest_clone.get("speakerEvalModel"),
            "targetText": (latest_clone.get("request") or {}).get("text"),
            "originalCloneAudio": latest_clone.get("originalCloneAudio"),
            "protectedCloneAudio": latest_clone.get("protectedCloneAudio"),
            "directSimilarity": latest_clone.get("directSimilarity"),
            "originalSimilarity": latest_clone.get("originalSimilarity"),
            "protectedSimilarity": latest_clone.get("protectedSimilarity"),
            "similarityDropRate": latest_clone.get("similarityDropRate"),
            "embeddingDistanceBefore": latest_clone.get("embeddingDistanceBefore"),
            "embeddingDistanceAfter": latest_clone.get("embeddingDistanceAfter"),
            "embeddingDistanceIncreaseRate": latest_clone.get("embeddingDistanceIncreaseRate"),
            "cloneConfidenceBefore": latest_clone.get("cloneConfidenceBefore"),
            "cloneConfidenceAfter": latest_clone.get("cloneConfidenceAfter"),
            "cloneConfidenceDropRate": latest_clone.get("cloneConfidenceDropRate"),
            "cloneRadar": latest_clone.get("cloneRadar"),
            "cloneTrend": latest_clone.get("cloneTrend"),
            "cloneDefenseScore": latest_clone.get("cloneDefenseScore"),
            "createdAt": latest_clone.get("createdAt") or result.get("updatedAt"),
        }
    detail_clone_eval = details.get("cloneEval")
    if clone_eval is None and isinstance(detail_clone_eval, dict):
        has_clone_metric = any(
            detail_clone_eval.get(key) is not None
            for key in [
                "originalSimilarity",
                "protectedSimilarity",
                "similarityDropRate",
                "embeddingDistanceBefore",
                "embeddingDistanceAfter",
                "embeddingDistanceIncreaseRate",
            ]
        )
        if has_clone_metric or detail_clone_eval.get("status") not in {None, "unavailable", "not_run"}:
            clone_eval = detail_clone_eval

    original_audio = _frontend_audio(audio.get("original"), "original.wav")
    protected_audio = _frontend_audio(audio.get("protected"), "protected.wav")

    return {
        "taskId": result.get("taskId"),
        "status": result.get("status", "completed"),
        "mode": result.get("mode", "joint"),
        "dataMode": result.get("dataMode", "backend"),
        "verdict": summary.get("verdict") or "防护结果已生成",
        "score": score,
        "createdAt": result.get("createdAt"),
        "submittedAt": result.get("submittedAt") or result.get("createdAt"),
        "completedAt": result.get("completedAt") or result.get("createdAt") or "-",
        "elapsedSec": _number(result.get("elapsedSec")),
        "inputSource": "后端 API",
        "language": asr.get("language") or "未标注",
        "processingModel": (details.get("generation") or {}).get("source") or (result.get("backend") or {}).get("version"),
        "optimizationTarget": (details.get("generation") or {}).get("mode") or result.get("mode", "joint"),
        "asrModel": asr.get("model"),
        "artifacts": [
            {"label": "原始音频", "filename": original_audio["filename"], "sizeBytes": original_audio["sizeBytes"]},
            {"label": "保护音频", "filename": protected_audio["filename"], "sizeBytes": protected_audio["sizeBytes"]},
            {"label": "结果 JSON", "filename": "result.json"},
        ],
        "originalAudio": original_audio,
        "protectedAudio": protected_audio,
        "perturbation": perception.get("perturbation")
        or {
            "l2Norm": _coalesce(perception.get("l2Norm"), loss_final.get("L2") if isinstance(loss_final, dict) else None, loss_final.get("l2") if isinstance(loss_final, dict) else None),
            "l2Rms": perception.get("l2Rms"),
            "linfNorm": perception.get("linfNorm"),
            "epsilon": _coalesce(perception.get("epsilon"), optimization.get("epsilon")),
            "epsilonNorm": _coalesce(perception.get("epsilonNorm"), optimization.get("epsilonNorm"), optimization.get("epsilon_norm")),
            "epsilonUsageRate": perception.get("epsilonUsageRate"),
            "epsilonUsageRateRaw": perception.get("epsilonUsageRateRaw"),
            "epsilonToleranceRate": perception.get("epsilonToleranceRate"),
            "epsilonExceeded": perception.get("epsilonExceeded"),
            "snr": snr,
            "clippingRate": perception.get("clippingRate"),
        },
        "protectionQuality": perception.get("protectionQuality")
        or {
            "snr": snr,
            "pesq": pesq,
            "stoi": perception.get("stoi"),
            "mos": perception.get("mos"),
            "mosLqo": perception.get("mosLqo"),
            "dnsMos": perception.get("dnsMos"),
            "dnsMosScore": perception.get("dnsMosScore"),
            "dnsMosStatus": perception.get("dnsMosStatus"),
            "dnsMosReason": perception.get("dnsMosReason"),
            "qualityScore": perception.get("qualityScore"),
            "qualityLevel": perception.get("qualityLevel"),
        },
        "psychoacoustic": perception.get("psychoacoustic")
        or {
            "lPsy": _coalesce(perception.get("lPsy"), loss_final.get("Lpsy") if isinstance(loss_final, dict) else None, loss_final.get("lPsy") if isinstance(loss_final, dict) else None),
            "overMaskRate": _coalesce(perception.get("overMaskRate"), perception.get("psychoacousticViolationRate")),
            "maskingThreshold": perception.get("maskingThreshold"),
            "perturbationSpectrum": perception.get("perturbationSpectrum"),
        },
        "lossFinal": loss_final,
        "lossWeights": {
            "lambdaId": _coalesce(loss_weights.get("lambdaId"), loss_weights.get("weight_identity"), loss_weights.get("lambdaFeat"), loss_weights.get("weight_feature")),
            "lambdaFeat": _coalesce(loss_weights.get("lambdaFeat"), loss_weights.get("weight_feature")),
            "lambdaSem": _coalesce(loss_weights.get("lambdaSem"), loss_weights.get("weight_semantic")),
            "lambdaPsy": _coalesce(loss_weights.get("lambdaPsy"), loss_weights.get("weight_psy")),
            "lambda2": _coalesce(loss_weights.get("lambda2"), loss_weights.get("weight_l2")),
        },
        "optimizationTrace": generation.get("optimizationTrace") or [],
        "averageStepSec": generation.get("averageStepSec"),
        "selectedStep": generation.get("selectedStep"),
        "effectiveConfig": generation.get("effectiveConfig"),
        "presetName": generation.get("presetName"),
        "asrEval": asr_eval,
        "semanticEval": semantic_eval,
        "asrResults": result.get("asrResults") or [],
        "cloneEval": clone_eval,
        "cloneResults": clone_results,
        "protectionEvaluation": result.get("protectionEvaluation") or details.get("protectionEvaluation"),
        "asr": {
            "referenceText": asr.get("referenceText") if asr_has_result else None,
            "originalText": asr.get("cleanTranscription") if asr_has_result else None,
            "protectedText": asr.get("protectedTranscription") if asr_has_result else None,
            "wer": _coalesce(primary.get("wer"), asr.get("wer")) if asr_has_result else None,
            "cer": _coalesce(primary.get("cer"), asr.get("cer")) if asr_has_result else None,
            "tokenErrorRate": asr.get("tokenErrorRate") if asr_has_result else None,
            "tokenChangeRate": asr.get("tokenChangeRate") if asr_has_result else None,
            "semanticDrift": asr.get("semanticDrift") if asr_has_result else None,
            "insertRate": ((asr.get("breakdown") or {}).get("insertRate")) if asr_has_result else None,
            "deleteRate": ((asr.get("breakdown") or {}).get("deleteRate")) if asr_has_result else None,
            "substituteRate": ((asr.get("breakdown") or {}).get("substituteRate")) if asr_has_result else None,
            "editCounts": asr.get("editCounts") if asr_has_result else None,
            "errorShares": asr.get("errorShares") if asr_has_result else None,
            "status": asr.get("status"),
        },
        "speaker": {
            "simBefore": (details.get("speaker") or {}).get("simBefore") if (details.get("speaker") or {}).get("simBefore") is not None else sim_before,
            "simAfter": _coalesce((details.get("speaker") or {}).get("simAfter"), sim_after),
            "simDropRate": (details.get("speaker") or {}).get("simDropRate"),
            "embeddingDistanceBefore": (details.get("speaker") or {}).get("embeddingDistanceBefore"),
            "embeddingDistanceAfter": _coalesce((details.get("speaker") or {}).get("embeddingDistanceAfter"), (details.get("speaker") or {}).get("embeddingDistance")),
            "simOriginalProtected": (details.get("speaker") or {}).get("simOriginalProtected"),
            "embeddingDistance": (details.get("speaker") or {}).get("embeddingDistance"),
            "directDistance": (details.get("speaker") or {}).get("directDistance"),
            "directIdentityScore": (details.get("speaker") or {}).get("directIdentityScore"),
            "scoreStatus": (details.get("speaker") or {}).get("scoreStatus"),
            "scoreReason": (details.get("speaker") or {}).get("scoreReason"),
            "source": ((metric_sources.get("speaker.*") or {}).get("source")),
            "status": (details.get("speaker") or {}).get("status"),
        },
        "quality": {
            "snr": snr,
            "pesq": pesq,
            "mosLqo": (details.get("perception") or {}).get("mosLqo"),
            "l2Norm": (details.get("perception") or {}).get("l2Norm"),
            "psychoacousticViolationRate": (details.get("perception") or {}).get("psychoacousticViolationRate"),
            "status": (details.get("perception") or {}).get("status"),
        },
        "metricSources": metric_sources,
        "generation": {
            "lossFinal": generation.get("lossFinal"),
            "optimizationTrace": generation.get("optimizationTrace") or [],
            "steps": generation.get("steps"),
            "maxSteps": generation.get("maxSteps"),
            "selectedStep": generation.get("selectedStep"),
            "snrDb": generation.get("snrDb"),
            "presetName": generation.get("presetName"),
            "effectiveConfig": generation.get("effectiveConfig"),
            "averageStepSec": generation.get("averageStepSec"),
            "realProtect": generation.get("realProtect"),
            "source": generation.get("source"),
            "status": generation.get("status"),
            "mode": generation.get("mode"),
        },
        "raw": result,
        "charts": {
            "psychoacoustic": charts.get("psychoacoustic") or [],
            "trend": charts.get("trend") or charts.get("optimizationTrend") or [],
            "optimizationTrend": charts.get("optimizationTrend") or generation.get("optimizationTrace") or [],
            "radarBefore": charts.get("radarBefore"),
            "radarAfter": charts.get("radarAfter"),
            "chainRadar": charts.get("chainRadar") or [],
        },
    }


def tiny_pdf_bytes(title: str, body: str) -> bytes:
    text = f"{title}\\n\\n{body}".replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 760 Td ({text}) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream.encode('utf-8'))} >>\nstream\n{stream}\nendstream".encode("utf-8"),
    ]
    output = io.BytesIO()
    output.write(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(output.tell())
        output.write(f"{index} 0 obj\n".encode("ascii"))
        output.write(obj)
        output.write(b"\nendobj\n")
    xref_at = output.tell()
    output.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        output.write(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.write(f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode("ascii"))
    return output.getvalue()


def protect_queue_snapshot() -> dict[str, int]:
    with PROTECT_QUEUE_LOCK:
        return {
            "maxConcurrency": PROTECT_MAX_CONCURRENCY,
            "activeCount": len(PROTECT_ACTIVE_TASK_IDS),
            "queuedCount": len(PROTECT_PENDING_TASKS),
        }


def runtime_concurrency_snapshot() -> dict[str, Any]:
    clone_capacity = clone_worker_capacity_snapshot()
    clone_max_concurrency = int(clone_capacity["maxConcurrency"])
    shared_asr_clone_max = int(
        clone_capacity.get(
            "asrCloneMaxConcurrency",
            ASR_WORKER_MAX_CONCURRENCY + clone_max_concurrency,
        )
    )
    protect_shares_worker_gpu = bool(clone_capacity.get("protectSharesWorkerGpu", False))
    total = (
        max(PROTECT_MAX_CONCURRENCY, shared_asr_clone_max)
        if protect_shares_worker_gpu
        else PROTECT_MAX_CONCURRENCY + shared_asr_clone_max
    )
    return {
        "protect": PROTECT_MAX_CONCURRENCY,
        "asr": ASR_WORKER_MAX_CONCURRENCY,
        "clone": clone_max_concurrency,
        "asrCloneShared": shared_asr_clone_max,
        "protectSharesWorkerGpu": protect_shares_worker_gpu,
        "total": total,
        "unit": "worker",
        "definition": (
            "保护与 ASR/克隆共享同一 GPU 时按两者较大容量计算；使用独立 GPU 时容量相加，不包含 HTTP 请求线程。"
            if protect_shares_worker_gpu
            else "保护线程上限与独立 ASR/克隆共享 GPU 池容量相加，不包含 HTTP 请求线程。"
        ),
        "cloneBackends": clone_capacity["backendLimits"],
        "cloneGpuSlots": {
            "limitPerGpu": clone_capacity["gpuSlotLimit"],
            "keys": clone_capacity["gpuKeys"],
            "asr": clone_capacity.get("asrGpuKeys", []),
        },
    }


def latest_runtime_performance_snapshot() -> dict[str, Any]:
    task_dirs = sorted(
        (path for path in TASK_DIR.iterdir() if path.is_dir()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for task_dir in task_dirs:
        result_path = task_dir / "result.json"
        if not result_path.exists():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if result.get("status") not in {"completed", "success"}:
            continue
        details = result.get("details") if isinstance(result.get("details"), dict) else {}
        generation = details.get("generation") if isinstance(details.get("generation"), dict) else {}
        average_step_sec = _number(
            _coalesce(
                result.get("averageStepSec"),
                generation.get("averageStepSec"),
                generation.get("average_step_sec"),
            )
        )
        if average_step_sec is None:
            trace = generation.get("optimizationTrace")
            step_times = [
                value
                for item in (trace if isinstance(trace, list) else []) if isinstance(item, dict)
                for value in [_number(_coalesce(item.get("stepElapsedSec"), item.get("step_elapsed_sec")))]
                if value is not None and value > 0
            ]
            average_step_sec = sum(step_times) / len(step_times) if step_times else None
        if average_step_sec is not None and average_step_sec > 0:
            return {
                "averageStepSec": average_step_sec,
                "sourceTaskId": result.get("taskId") or task_dir.name,
                "source": "latest_completed_protection_result",
            }
    return {
        "averageStepSec": None,
        "sourceTaskId": None,
        "source": "no_completed_protection_timing",
    }


def cached_capabilities() -> dict[str, Any]:
    payload = get_capabilities_snapshot(TASK_DIR.parent, diagnose_capabilities, logger=logger)
    chains = payload.setdefault("chains", {})
    perception = dict(chains.get("perception_eval") or {})
    available = [item for item in perception.get("available", []) if item not in {"mos", "mosLqo", "dnsMos"}]
    unavailable = [item for item in perception.get("unavailable", []) if item not in {"mos", "mosLqo", "dnsMos"}]
    quality_model = dnsmos_model_status()
    target = available if quality_model.get("status") == "available" else unavailable
    target.append("dnsMos")
    perception.update(
        {
            "status": "available" if not unavailable else "partial",
            "available": available,
            "unavailable": unavailable,
            "qualityModel": quality_model,
            "reason": None if not unavailable else "部分语音质量指标尚未生成",
        }
    )
    chains["perception_eval"] = perception
    return payload


@app.get("/api/health")
def health() -> dict[str, Any]:
    capabilities_payload = cached_capabilities()
    tts_chain = (capabilities_payload.get("chains") or {}).get("downstream_tts_eval") or {}
    tts_status = str(tts_chain.get("status") or "unavailable")
    tts_reason = tts_chain.get("reason")
    return {
        "ok": True,
        "version": "sem-e2e-api-0.1",
        "time": utc_now_iso(),
        "device": capabilities_payload["device"],
        "availableChains": ["protect", "semantic", "asr", "speaker", "perception"],
        "optionalChains": {
            "tts": f"{tts_status}: {tts_reason}" if tts_reason else tts_status,
            "pesq": "unavailable: PESQ is only returned when a real evaluator is installed",
        },
        "protectQueue": protect_queue_snapshot(),
        "chains": capabilities_payload["chains"],
    }


@app.get("/api/capabilities")
def capabilities() -> dict[str, Any]:
    payload = cached_capabilities()
    # Model/checkpoint probing is intentionally cached on disk, but form
    # defaults are cheap configuration and must never be held back by an old
    # capability snapshot after a backend deployment.
    config_payload = runtime_config()
    payload["config"] = config_payload
    payload["modelTypes"] = config_payload.get("modelTypes", payload.get("modelTypes", {}))
    payload["runtimeConcurrency"] = runtime_concurrency_snapshot()
    payload["runtimePerformance"] = latest_runtime_performance_snapshot()
    payload["time"] = utc_now_iso()
    payload["version"] = "sem-e2e-api-0.1"
    return payload


@app.get("/api/config")
def config() -> dict[str, Any]:
    config_payload = runtime_config()
    return {
        "ok": True,
        "time": utc_now_iso(),
        "modelTypes": config_payload.get("modelTypes", {}),
        "config": config_payload,
        "protectQueue": protect_queue_snapshot(),
        "runtimeConcurrency": runtime_concurrency_snapshot(),
        "runtimePerformance": latest_runtime_performance_snapshot(),
        "capabilitiesCache": {
            "strategy": "disk-snapshot-stale-while-revalidate",
            "refreshFlag": "seme2e-runtime/capabilities-refresh.flag",
            "refreshValue": 1,
        },
    }


@app.post("/api/files/upload")
async def upload_file(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")
    file_id = new_file_id()
    safe_name = Path(file.filename).name
    path = UPLOAD_DIR / f"{file_id}_{safe_name}"
    with path.open("wb") as target:
        shutil.copyfileobj(file.file, target)
    data = {
        "fileId": file_id,
        "filename": safe_name,
        "sizeBytes": path.stat().st_size,
        "format": path.suffix.lstrip(".").upper() or "AUDIO",
        "audioUrl": public_file_url(file_id, safe_name),
        "downloadUrl": public_file_url(file_id, safe_name),
        "uploadedAt": utc_now_iso(),
        "path": path,
    }
    data.update(probe_audio_metadata(path))
    FILES[file_id] = data
    return {key: value for key, value in data.items() if key != "path"}


@app.get("/api/files/{file_id}/{filename}")
def get_uploaded_file(file_id: str, filename: str) -> FileResponse:
    data = find_uploaded_file(file_id)
    path = Path(data["path"])
    if filename != data["filename"]:
        raise HTTPException(status_code=404, detail="filename mismatch")
    return FileResponse(path, filename=data["filename"])


def run_protect_task_process(
    task_id: str,
    req_id: str,
    uploaded_path: str,
    uploaded_filename: str | None,
    file_id: str,
    payload_dict: dict[str, Any],
    cancel_event: Any,
    selected_gpu: str | None = None,
) -> None:
    if selected_gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu
        os.environ["SEME2E_API_DEVICE"] = "cuda:0"
        os.environ["SEME2E_PROTECT_CUDA_VISIBLE_DEVICES"] = selected_gpu
        os.environ["SEME2E_TOKENIZER_DEVICE"] = "cuda:0"
        os.environ["SEME2E_SEMANTIC_ENCODER_DEVICE"] = "cuda:0"
        os.environ["SEME2E_PROTECT_SELECTED_GPU"] = selected_gpu
    uploaded = {"path": uploaded_path, "filename": uploaded_filename}

    def ensure_process_task_not_deleted() -> None:
        if is_task_deleted(task_id):
            raise TaskCancelledError(f"task deleted: {task_id}")

    try:
        ensure_process_task_not_deleted()
        write_task_status(
            task_id,
            status="running",
            progress=0.08,
            stage="file_preprocess",
            message="后端正在预处理录音/音频",
            queuePosition=None,
            maxConcurrency=PROTECT_MAX_CONCURRENCY,
            error=None,
        )

        def on_protect_progress(**event: Any) -> None:
            ensure_process_task_not_deleted()
            write_task_status(task_id, **protection_progress_status(event))

        result = create_task(
            Path(uploaded_path),
            file_id,
            payload_dict,
            input_filename=uploaded_filename,
            request_id=req_id,
            task_id=task_id,
            progress_callback=on_protect_progress,
            cancel_event=cancel_event,
        )
        ensure_process_task_not_deleted()
        write_task_status(
            task_id,
            status=result.get("status", "completed"),
            progress=1,
            stage="report_generation",
            message="Task completed",
            error=None,
            resultUrl=f"/api/tasks/{task_id}/result",
        )
    except TaskCancelledError:
        return
    except AudioPreprocessError as exc:
        diagnostics = exc.diagnostics
        write_task_log(
            task_id,
            {
                "requestId": req_id,
                "taskId": task_id,
                "fileId": file_id,
                "inputAudioPath": uploaded_path,
                "inputAudioExists": Path(uploaded_path).exists(),
                "currentStage": "file_preprocess",
                "reason": exc.reason,
                "diagnostics": diagnostics,
                "exceptionType": type(exc).__name__,
                "exceptionMessage": str(exc),
                "stackTrace": traceback.format_exc(),
            },
        )
        suggestion = (
            "请使用 Python 3.11 安装 backend/SemE2E/requirements.txt，或通过 SEME2E_FFMPEG_PATH 指定 FFmpeg。"
            if exc.code == "AUDIO_DECODER_UNAVAILABLE"
            else "请重新录音，或上传可正常播放的 WAV、FLAC、MP3、M4A、OGG、WebM/Opus 音频。"
        )
        write_task_status(
            task_id,
            status="failed",
            stage="file_preprocess",
            message=str(exc),
            error={
                "code": exc.code,
                "message": str(exc),
                "requestId": req_id,
                "taskId": task_id,
                "stage": "file_preprocess",
                "details": {
                    "fileId": file_id,
                    "reason": exc.reason,
                    "suggestion": suggestion,
                    **diagnostics,
                },
            },
        )
    except ProtectGenerationError as exc:
        diagnostics = exc.diagnostics
        write_task_log(
            exc.task_id,
            {
                "requestId": req_id,
                "taskId": exc.task_id,
                "fileId": file_id,
                "inputAudioPath": uploaded_path,
                "inputAudioExists": Path(uploaded_path).exists(),
                "outputPath": diagnostics.get("outputPath"),
                "outputPathExists": diagnostics.get("outputPathExists"),
                "cwd": diagnostics.get("cwd"),
                "pythonExecutable": diagnostics.get("pythonExecutable"),
                "device": diagnostics.get("device"),
                "allowFallback": diagnostics.get("allowFallback"),
                "mode": diagnostics.get("mode"),
                "epsilon": diagnostics.get("epsilon"),
                "steps": diagnostics.get("steps"),
                "protectCall": diagnostics.get("protectCall"),
                "protectReturnType": diagnostics.get("protectReturnType"),
                "protectReturnHasOutputWav": diagnostics.get("protectReturnHasOutputWav"),
                "protectReturnHasOutputPath": diagnostics.get("protectReturnHasOutputPath"),
                "protectReturnHasProtectedAudioPath": diagnostics.get("protectReturnHasProtectedAudioPath"),
                "protectReturnedPath": diagnostics.get("protectReturnedPath"),
                "protectReturnedPathExists": diagnostics.get("protectReturnedPathExists"),
                "protectReturnSummary": diagnostics.get("protectReturnSummary"),
                "reason": exc.reason,
                "payload": payload_dict,
                "currentStage": "protect_generation",
                "exceptionType": type(exc).__name__,
                "exceptionMessage": str(exc),
                "stackTrace": diagnostics.get("stackTrace") or traceback.format_exc(),
                "capabilities": diagnostics.get("capabilities"),
            },
        )
        details = {
            "fileId": file_id,
            "inputAudioPath": uploaded_path,
            "inputAudioExists": Path(uploaded_path).exists(),
            "allowFallback": diagnostics.get("allowFallback"),
            "expectedOutputPath": diagnostics.get("outputPath"),
            "reason": exc.reason,
            "exceptionType": diagnostics.get("exceptionType"),
            "exceptionMessage": diagnostics.get("exceptionMessage"),
            "capabilities": diagnostics.get("capabilities", {}).get("chains"),
            "suggestion": (
                "请缩短并发任务数量或释放内存后重试。"
                if exc.reason == "resource_exhausted"
                else "请查看任务日志中的 exceptionType、exceptionMessage 和 stackTrace。"
                if exc.reason == "algorithm_runtime_error"
                else "请检查后端依赖和模型 checkpoint。"
            ),
        }
        failure_message = (
            f"保护算法执行失败：{diagnostics.get('exceptionMessage')}"
            if diagnostics.get("exceptionMessage")
            else "保护音频生成失败：后端算法未生成保护音频。"
        )
        write_task_status(
            task_id,
            status="failed",
            stage="protect_generation",
            message=failure_message,
            error={
                "code": "PROTECT_GENERATION_FAILED",
                "message": failure_message,
                "requestId": req_id,
                "taskId": exc.task_id,
                "stage": "protect_generation",
                "details": details,
            },
        )
    except Exception as exc:
        if is_task_deleted(task_id):
            return
        write_task_log(
            task_id,
            {
                "requestId": req_id,
                "taskId": task_id,
                "fileId": file_id,
                "currentStage": "protect_generation",
                "exceptionType": type(exc).__name__,
                "exceptionMessage": str(exc),
                "stackTrace": traceback.format_exc(),
            },
        )
        write_task_status(
            task_id,
            status="failed",
            stage="protect_generation",
            message=str(exc),
            error={
                "code": "PROTECT_GENERATION_FAILED",
                "message": str(exc),
                "requestId": req_id,
                "taskId": task_id,
                "stage": "protect_generation",
                "details": {"fileId": file_id},
            },
        )


def _refresh_protect_queue_statuses_locked() -> None:
    for position, job in enumerate(PROTECT_PENDING_TASKS, start=1):
        task_id = str(job["task_id"])
        cancel_event = job["cancel_event"]
        if cancel_event.is_set() or is_task_deleted(task_id):
            continue
        try:
            write_task_status(
                task_id,
                status="queued",
                progress=0.05,
                stage="file_preprocess",
                message=f"任务正在排队，前方还有 {position - 1} 个任务",
                queuePosition=position,
                maxConcurrency=PROTECT_MAX_CONCURRENCY,
                error=None,
            )
        except TaskCancelledError:
            continue


def _watch_protect_process(
    task_id: str,
    process: multiprocessing.Process,
    cancel_event: Any,
    gpu_slot: threading.BoundedSemaphore | None = None,
) -> None:
    try:
        process.join()
        if not is_task_deleted(task_id):
            try:
                status = read_task_status(task_id)
            except Exception:
                status = {}
            if status.get("status") in {"queued", "running"}:
                exit_code = process.exitcode
                message = f"保护任务进程意外退出（exit code: {exit_code}）"
                write_task_status(
                    task_id,
                    status="failed",
                    stage=str(status.get("stage") or "protect_generation"),
                    message=message,
                    error={
                        "code": "PROTECT_PROCESS_EXITED",
                        "message": message,
                        "taskId": task_id,
                        "stage": str(status.get("stage") or "protect_generation"),
                        "details": {"exitCode": exit_code},
                    },
                )
    finally:
        if gpu_slot is not None:
            release_gpu_slot(gpu_slot)
        with PROTECT_QUEUE_LOCK:
            PROTECT_ACTIVE_TASK_IDS.discard(task_id)
            cleanup_protect_process_runtime(task_id, process, cancel_event)
            _dispatch_protect_tasks_locked()


def _start_protect_job_locked(
    job: dict[str, Any],
    *,
    selected_gpu: str | None = None,
    gpu_slot: threading.BoundedSemaphore | None = None,
) -> bool:
    task_id = str(job["task_id"])
    cancel_event = job["cancel_event"]
    if cancel_event.is_set() or is_task_deleted(task_id):
        if gpu_slot is not None:
            release_gpu_slot(gpu_slot)
        cleanup_task_runtime(task_id)
        return False

    process = PROTECT_PROCESS_CONTEXT.Process(
        target=run_protect_task_process,
        args=(
            task_id,
            job["request_id"],
            job["uploaded_path"],
            job.get("uploaded_filename"),
            job["file_id"],
            job["payload"],
            cancel_event,
            selected_gpu,
        ),
        daemon=True,
    )
    PROTECT_ACTIVE_TASK_IDS.add(task_id)
    register_task_runtime(task_id, cancel_event, process=process)
    try:
        process.start()
    except Exception as exc:
        if gpu_slot is not None:
            release_gpu_slot(gpu_slot)
        PROTECT_ACTIVE_TASK_IDS.discard(task_id)
        cleanup_protect_process_runtime(task_id, process, cancel_event)
        write_task_status(
            task_id,
            status="failed",
            stage="file_preprocess",
            message=f"无法启动保护任务进程：{exc}",
            error={
                "code": "PROTECT_PROCESS_START_FAILED",
                "message": str(exc),
                "taskId": task_id,
                "stage": "file_preprocess",
                "details": {"exceptionType": type(exc).__name__},
            },
        )
        return False

    watcher = threading.Thread(
        target=_watch_protect_process,
        args=(task_id, process, cancel_event, gpu_slot),
        name=f"protect-watch-{task_id}",
        daemon=True,
    )
    watcher.start()
    return True


def _protect_gpu_candidates() -> tuple[str, ...]:
    requested_device = os.getenv("SEME2E_PROTECT_DEVICE") or os.getenv("SEME2E_API_DEVICE", "cpu")
    pool_env = (
        "SEME2E_PROTECT_GPU_POOL"
        if os.getenv("SEME2E_PROTECT_GPU_POOL", "").strip()
        else "SEME2E_PROTECT_CUDA_VISIBLE_DEVICES"
    )
    return _worker_gpu_candidates(
        requested_device,
        pool_env,
        explicit_device=bool(os.getenv("SEME2E_PROTECT_DEVICE", "").strip()),
    )


def _try_acquire_protect_gpu() -> tuple[str | None, threading.BoundedSemaphore | None, bool]:
    candidates = _protect_gpu_candidates()
    if not candidates:
        return None, None, True
    retry_window = max(0.01, float(os.getenv("SEME2E_PROTECT_GPU_DISPATCH_WINDOW_SECONDS", "0.05")))
    try:
        selected_gpu, gpu_slot = acquire_gpu_slot(
            candidates,
            minimum_free_mib=max(
                0,
                _positive_env_int(
                    "SEME2E_PROTECT_GPU_MIN_FREE_MIB",
                    _positive_env_int("SEME2E_GPU_MIN_FREE_MIB", 1),
                ),
            ),
            deadline=time.monotonic() + retry_window,
        )
        return selected_gpu, gpu_slot, True
    except RuntimeError as exc:
        if str(exc) == GPU_ACQUIRE_TIMEOUT_MESSAGE:
            return None, None, False
        raise


def _retry_protect_dispatch() -> None:
    global PROTECT_DISPATCH_RETRY_TIMER
    with PROTECT_QUEUE_LOCK:
        PROTECT_DISPATCH_RETRY_TIMER = None
        _dispatch_protect_tasks_locked()


def _schedule_protect_dispatch_retry_locked() -> None:
    global PROTECT_DISPATCH_RETRY_TIMER
    if PROTECT_DISPATCH_RETRY_TIMER is not None and PROTECT_DISPATCH_RETRY_TIMER.is_alive():
        return
    retry_seconds = max(0.05, float(os.getenv("SEME2E_PROTECT_GPU_RETRY_SECONDS", "0.5")))
    timer = threading.Timer(retry_seconds, _retry_protect_dispatch)
    timer.daemon = True
    PROTECT_DISPATCH_RETRY_TIMER = timer
    timer.start()


def _dispatch_protect_tasks_locked() -> None:
    while len(PROTECT_ACTIVE_TASK_IDS) < PROTECT_MAX_CONCURRENCY and PROTECT_PENDING_TASKS:
        job = PROTECT_PENDING_TASKS[0]
        cancel_event = job["cancel_event"]
        if cancel_event.is_set() or is_task_deleted(str(job["task_id"])):
            PROTECT_PENDING_TASKS.popleft()
            cleanup_task_runtime(str(job["task_id"]))
            continue
        try:
            selected_gpu, gpu_slot, ready = _try_acquire_protect_gpu()
        except Exception as exc:
            PROTECT_PENDING_TASKS.popleft()
            write_task_status(
                str(job["task_id"]),
                status="failed",
                stage="protect_generation",
                message=f"无法分配保护任务 GPU：{exc}",
                error={
                    "code": "PROTECT_GPU_ALLOCATION_FAILED",
                    "message": str(exc),
                    "taskId": str(job["task_id"]),
                    "stage": "protect_generation",
                },
            )
            cleanup_task_runtime(str(job["task_id"]))
            continue
        if not ready:
            _schedule_protect_dispatch_retry_locked()
            break
        PROTECT_PENDING_TASKS.popleft()
        _start_protect_job_locked(job, selected_gpu=selected_gpu, gpu_slot=gpu_slot)
    _refresh_protect_queue_statuses_locked()


def enqueue_protect_job(job: dict[str, Any]) -> None:
    task_id = str(job["task_id"])
    register_task_runtime(task_id, job["cancel_event"])
    with PROTECT_QUEUE_LOCK:
        PROTECT_PENDING_TASKS.append(job)
        _dispatch_protect_tasks_locked()


def remove_pending_protect_job(task_id: str) -> bool:
    with PROTECT_QUEUE_LOCK:
        for job in tuple(PROTECT_PENDING_TASKS):
            if job.get("task_id") == task_id:
                PROTECT_PENDING_TASKS.remove(job)
                _refresh_protect_queue_statuses_locked()
                return True
    return False


@app.on_event("startup")
def recover_protection_queue() -> None:
    """Recover queued work and make interrupted protection tasks terminal after a restart."""
    with PROTECT_QUEUE_LOCK:
        if PROTECT_PENDING_TASKS or PROTECT_ACTIVE_TASK_IDS:
            return

    for task_dir in sorted(
        (path for path in TASK_DIR.iterdir() if path.is_dir()),
        key=lambda item: item.stat().st_mtime,
    ):
        task_id = task_dir.name
        try:
            status = read_task_status(task_id)
        except Exception:
            continue
        if status.get("stage") not in {"file_preprocess", "protect_generation"}:
            continue
        payload = status.get("payload")
        file_id = status.get("fileId")
        if not isinstance(payload, dict) or not isinstance(file_id, str) or not file_id:
            continue

        if status.get("status") == "running":
            message = "后端服务曾重启，原保护任务进程已中断，请重试"
            write_task_status(
                task_id,
                status="failed",
                stage=str(status.get("stage") or "protect_generation"),
                message=message,
                error={
                    "code": "PROTECT_SERVER_RESTARTED",
                    "message": message,
                    "taskId": task_id,
                    "stage": str(status.get("stage") or "protect_generation"),
                    "details": {"previousStatus": "running"},
                },
            )
            continue
        if status.get("status") != "queued":
            continue

        try:
            uploaded = find_uploaded_file(file_id)
        except HTTPException as exc:
            message = f"排队任务恢复失败：{exc.detail}"
            write_task_status(
                task_id,
                status="failed",
                stage="file_preprocess",
                message=message,
                error={
                    "code": "PROTECT_QUEUE_RECOVERY_FAILED",
                    "message": message,
                    "taskId": task_id,
                    "stage": "file_preprocess",
                    "details": {"fileId": file_id},
                },
            )
            continue

        enqueue_protect_job(
            {
                "task_id": task_id,
                "request_id": request_id(),
                "uploaded_path": str(uploaded["path"]),
                "uploaded_filename": uploaded.get("filename"),
                "file_id": file_id,
                "payload": payload,
                "cancel_event": PROTECT_PROCESS_CONTEXT.Event(),
            }
        )


@app.post("/api/tasks/protect")
def protect_task(payload: ProtectTaskRequest) -> dict[str, Any]:
    req_id = request_id()
    if not payload.fileId:
        return structured_error(
            code="MISSING_FILE_ID",
            message="保护任务缺少 fileId，请先上传音频文件。",
            status_code=400,
            request_id_value=req_id,
            stage="file_preprocess",
            details={"fileId": None},
        )
    validation_error = validate_protection_config(payload, req_id)
    if validation_error is not None:
        return validation_error
    uploaded = find_uploaded_file(payload.fileId)
    task_id = new_task_id()
    write_task_status(
        task_id,
        status="queued",
        progress=0.05,
        stage="file_preprocess",
        message="任务已排入队列",
        fileId=payload.fileId,
        filename=uploaded.get("filename"),
        mode=payload.mode,
        payload=payload.model_dump(),
        maxConcurrency=PROTECT_MAX_CONCURRENCY,
        error=None,
    )
    cancel_event = PROTECT_PROCESS_CONTEXT.Event()
    enqueue_protect_job(
        {
            "task_id": task_id,
            "request_id": req_id,
            "uploaded_path": str(uploaded["path"]),
            "uploaded_filename": uploaded.get("filename"),
            "file_id": payload.fileId,
            "payload": payload.model_dump(),
            "cancel_event": cancel_event,
        }
    )
    return {"taskId": task_id, "status": "queued", "maxConcurrency": PROTECT_MAX_CONCURRENCY}

    cancel_event = threading.Event()
    register_task_runtime(task_id, cancel_event)

    def run_background() -> None:
        try:
            ensure_task_not_cancelled(task_id, cancel_event)
            write_task_status(
                task_id,
                status="running",
                progress=0.18,
                stage="protect_generation",
                message="后端正在生成保护音频",
                error=None,
            )

            def on_protect_progress(**event: Any) -> None:
                ensure_task_not_cancelled(task_id, cancel_event)
                write_task_status(task_id, **protection_progress_status(event))

            result = create_task(
                Path(uploaded["path"]),
                payload.fileId,
                payload.model_dump(),
                input_filename=uploaded.get("filename"),
                request_id=req_id,
                task_id=task_id,
                progress_callback=on_protect_progress,
                cancel_event=cancel_event,
            )
            ensure_task_not_cancelled(task_id, cancel_event)
            write_task_status(
                task_id,
                status=result.get("status", "completed"),
                progress=1,
                stage="report_generation",
                message="任务已完成",
                error=None,
                resultUrl=f"/api/tasks/{task_id}/result",
            )
        except TaskCancelledError:
            if (TASK_DIR / task_id).exists():
                write_task_status(
                    task_id,
                    status="cancelled",
                    stage="protect_generation",
                    message="任务已被删除请求取消",
                    error=None,
                )
        except ProtectGenerationError as exc:
            diagnostics = exc.diagnostics
            write_task_log(
                exc.task_id,
                {
                    "requestId": req_id,
                    "taskId": exc.task_id,
                    "fileId": payload.fileId,
                    "inputAudioPath": uploaded.get("path"),
                    "inputAudioExists": Path(uploaded["path"]).exists(),
                    "outputPath": diagnostics.get("outputPath"),
                    "outputPathExists": diagnostics.get("outputPathExists"),
                    "cwd": diagnostics.get("cwd"),
                    "pythonExecutable": diagnostics.get("pythonExecutable"),
                    "device": diagnostics.get("device"),
                    "allowFallback": diagnostics.get("allowFallback"),
                    "mode": diagnostics.get("mode"),
                    "epsilon": diagnostics.get("epsilon"),
                    "steps": diagnostics.get("steps"),
                    "protectCall": diagnostics.get("protectCall"),
                    "protectReturnType": diagnostics.get("protectReturnType"),
                    "protectReturnHasOutputWav": diagnostics.get("protectReturnHasOutputWav"),
                    "protectReturnHasOutputPath": diagnostics.get("protectReturnHasOutputPath"),
                    "protectReturnHasProtectedAudioPath": diagnostics.get("protectReturnHasProtectedAudioPath"),
                    "protectReturnedPath": diagnostics.get("protectReturnedPath"),
                    "protectReturnedPathExists": diagnostics.get("protectReturnedPathExists"),
                    "protectReturnSummary": diagnostics.get("protectReturnSummary"),
                    "reason": exc.reason,
                    "payload": payload.model_dump(),
                    "currentStage": "protect_generation",
                    "exceptionType": type(exc).__name__,
                    "exceptionMessage": str(exc),
                    "stackTrace": diagnostics.get("stackTrace") or traceback.format_exc(),
                    "capabilities": diagnostics.get("capabilities"),
                },
            )
            details = {
                "fileId": payload.fileId,
                "inputAudioPath": str(uploaded.get("path")),
                "inputAudioExists": Path(uploaded["path"]).exists(),
                "allowFallback": diagnostics.get("allowFallback"),
                "expectedOutputPath": diagnostics.get("outputPath"),
                "reason": exc.reason,
                "capabilities": diagnostics.get("capabilities", {}).get("chains"),
                "suggestion": "Install/check backend dependencies and model checkpoints.",
            }
            write_task_status(
                task_id,
                status="failed",
                stage="protect_generation",
                message="保护音频生成失败：后端算法未生成 protected audio。",
                error={
                    "code": "PROTECT_GENERATION_FAILED",
                    "message": "保护音频生成失败：后端算法未生成 protected audio。",
                    "requestId": req_id,
                    "taskId": exc.task_id,
                    "stage": "protect_generation",
                    "details": details,
                },
            )
        except RuntimeError as exc:
            if str(exc) == "TASK_CANCELLED":
                if (TASK_DIR / task_id).exists():
                    write_task_status(
                        task_id,
                        status="cancelled",
                        stage="protect_generation",
                        message="Task cancelled by delete request",
                        error=None,
                    )
                return
            write_task_log(
                task_id,
                {
                    "requestId": req_id,
                    "taskId": task_id,
                    "fileId": payload.fileId,
                    "currentStage": "protect_generation",
                    "exceptionType": type(exc).__name__,
                    "exceptionMessage": str(exc),
                    "stackTrace": traceback.format_exc(),
                },
            )
            write_task_status(
                task_id,
                status="failed",
                stage="protect_generation",
                message=str(exc),
                error={
                    "code": "PROTECT_GENERATION_FAILED",
                    "message": str(exc),
                    "requestId": req_id,
                    "taskId": task_id,
                    "stage": "protect_generation",
                    "details": {"fileId": payload.fileId},
                },
            )
        except Exception as exc:
            write_task_log(
                task_id,
                {
                    "requestId": req_id,
                    "taskId": task_id,
                    "fileId": payload.fileId,
                    "currentStage": "protect_generation",
                    "exceptionType": type(exc).__name__,
                    "exceptionMessage": str(exc),
                    "stackTrace": traceback.format_exc(),
                },
            )
            write_task_status(
                task_id,
                status="failed",
                stage="protect_generation",
                message=str(exc),
                error={
                    "code": "PROTECT_GENERATION_FAILED",
                    "message": str(exc),
                    "requestId": req_id,
                    "taskId": task_id,
                    "stage": "protect_generation",
                    "details": {"fileId": payload.fileId},
                },
            )
        finally:
            cleanup_task_runtime(task_id)

    thread = threading.Thread(target=run_background, daemon=True)
    register_task_runtime(task_id, cancel_event, thread)
    thread.start()
    return {"taskId": task_id, "status": "queued"}


@app.post("/api/tasks/{task_id}/retry")
def retry_protection_task(task_id: str) -> JSONResponse:
    req_id = request_id()
    status = read_task_status(task_id)
    current_status = status.get("status")
    if current_status not in {"failed", "error"}:
        return structured_error(
            code="TASK_NOT_RETRYABLE",
            message="仅保护失败的任务可以重试。",
            status_code=409,
            request_id_value=req_id,
            task_id=task_id,
            stage=str(status.get("stage") or "protect_generation"),
            details={"status": current_status},
        )

    original_payload = status.get("payload")
    if not isinstance(original_payload, dict):
        return structured_error(
            code="TASK_RETRY_PAYLOAD_MISSING",
            message="原任务缺少保护参数，无法重试。",
            status_code=409,
            request_id_value=req_id,
            task_id=task_id,
            stage="protect_generation",
            details={"status": current_status},
        )

    try:
        retry_payload = ProtectTaskRequest.model_validate(original_payload)
    except Exception as exc:
        return structured_error(
            code="TASK_RETRY_PAYLOAD_INVALID",
            message="原任务保护参数无效，无法重试。",
            status_code=409,
            request_id_value=req_id,
            task_id=task_id,
            stage="protect_generation",
            details={"exceptionType": type(exc).__name__, "exceptionMessage": str(exc)},
        )

    created = protect_task(retry_payload)
    if isinstance(created, dict):
        return JSONResponse({**created, "retryOfTaskId": task_id})
    return created


@app.get("/api/tasks")
def list_tasks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def terminal_status_from_chain(status_value: Any) -> str | None:
        if status_value in {"available", "computed", "partial", "completed", "success"}:
            return "completed"
        if status_value in {"running", "queued", "failed", "error", "cancelled"}:
            return str(status_value)
        if status_value in {"unavailable", "missing", "disabled"}:
            return "failed"
        return None

    for task_dir in sorted((path for path in TASK_DIR.iterdir() if path.is_dir()), key=lambda item: item.stat().st_mtime, reverse=True):
        task_id = task_dir.name
        result_path = task_dir / "result.json"
        status_path = task_status_path(task_id)
        asr_result_path = task_dir / "asr_result.json"
        clone_result_path = task_dir / "clone_result.json"
        result: dict[str, Any] | None = None
        status: dict[str, Any] | None = None
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                result = None
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception:
                status = None
        if result is None and status is None:
            continue
        payload = result or status or {}
        current_stage = (status or {}).get("stage")
        asr_tasks = [item for item in (status or {}).get("asrTasks", []) if isinstance(item, dict)]
        clone_tasks = [item for item in (status or {}).get("cloneTasks", []) if isinstance(item, dict)]
        asr_batches = [item for item in (status or {}).get("asrBatches", []) if isinstance(item, dict)]
        clone_batches = [item for item in (status or {}).get("cloneBatches", []) if isinstance(item, dict)]
        latest_asr_batch = asr_batches[-1] if asr_batches else None
        latest_clone_batch = clone_batches[-1] if clone_batches else None
        asr_task = asr_tasks[-1] if asr_tasks else (status or {}).get("asrTask") if isinstance((status or {}).get("asrTask"), dict) else {}
        clone_task = clone_tasks[-1] if clone_tasks else (status or {}).get("cloneTask") if isinstance((status or {}).get("cloneTask"), dict) else {}
        audio = payload.get("audio") or {}
        primary = (payload.get("summary") or {}).get("primaryMetrics") or {}
        details = payload.get("details") or {}
        asr_details = details.get("asr") or {}
        frontend_payload = frontend_result(payload) if result is not None else {}
        asr_eval = frontend_payload.get("asrEval") or {}
        clone_eval = frontend_payload.get("cloneEval") or {}
        protection_quality = frontend_payload.get("protectionQuality") or {}
        downstream_tts = details.get("downstreamTts") or {}
        clone_results = payload.get("cloneResults") or []
        request_payload = payload.get("request") or payload.get("payload") or {}
        targets = request_payload.get("targets") or []
        has_semantic = "semantic" in targets or bool((request_payload.get("semantic") or {}).get("enabled"))
        has_timbre = "timbre" in targets or bool((request_payload.get("timbre") or {}).get("enabled"))
        target_mode = "joint" if has_semantic and has_timbre else "semantic" if has_semantic else "timbre" if has_timbre else None
        semantic_cfg = request_payload.get("semantic") or {}
        timbre_cfg = request_payload.get("timbre") or {}
        psychoacoustic_cfg = request_payload.get("psychoacoustic") or {}
        optimization_cfg = request_payload.get("optimization") or {}
        protection_status = payload.get("status") or (status or {}).get("status") or "queued"
        if current_stage in {"asr_eval", "downstream_tts_eval"} and result is not None:
            protection_progress = 1 if protection_status in {"completed", "success"} else payload.get("progress")
            protection_stage = payload.get("stage") or "report_generation"
            protection_message = payload.get("message")
            protection_elapsed = payload.get("elapsedSec")
            protection_error = payload.get("error")
        else:
            protection_progress = (status or payload).get("progress")
            protection_stage = (status or payload).get("stage")
            protection_message = (status or payload).get("message")
            protection_elapsed = (status or payload).get("elapsedSec")
            protection_error = (status or payload).get("error")

        has_current_asr = current_stage == "asr_eval" or asr_task.get("status") in {"queued", "running"}
        has_asr_result = bool(latest_asr_batch) or bool(asr_task) or has_current_asr or asr_result_path.exists() or asr_details.get("status") in {"available", "computed", "partial", "failed", "error"}
        if has_current_asr:
            asr_task_status = asr_task.get("status") or (status or {}).get("status")
            asr_progress = asr_task.get("progress") if asr_task.get("progress") is not None else (status or {}).get("progress")
            asr_message = asr_task.get("message") or (status or {}).get("message")
            asr_elapsed = asr_task.get("elapsedSec") if asr_task.get("elapsedSec") is not None else (status or {}).get("elapsedSec")
            asr_error = asr_task.get("error") if asr_task.get("error") is not None else (status or {}).get("error")
        elif asr_task:
            asr_task_status = terminal_status_from_chain(asr_task.get("status")) or str(asr_task.get("status") or "completed")
            asr_progress = asr_task.get("progress") if asr_task.get("progress") is not None else (1 if asr_task_status in {"completed", "success"} else None)
            asr_message = asr_task.get("message")
            asr_elapsed = asr_task.get("elapsedSec")
            asr_error = asr_task.get("error")
        elif has_asr_result:
            asr_task_status = terminal_status_from_chain(asr_details.get("status")) or "completed"
            asr_progress = 1
            asr_message = asr_details.get("reason")
            asr_elapsed = None
            asr_error = asr_details.get("error")
        else:
            asr_task_status = None
            asr_progress = None
            asr_message = None
            asr_elapsed = None
            asr_error = None

        has_current_clone = current_stage == "downstream_tts_eval" or clone_task.get("status") in {"queued", "running"}
        has_clone_result = bool(latest_clone_batch) or bool(clone_task) or has_current_clone or clone_result_path.exists() or bool(clone_results) or downstream_tts.get("status") in {"computed", "partial", "failed", "error"}
        if has_current_clone:
            clone_task_status = clone_task.get("status") or (status or {}).get("status")
            clone_progress = clone_task.get("progress") if clone_task.get("progress") is not None else (status or {}).get("progress")
            clone_message = clone_task.get("message") or (status or {}).get("message")
            clone_elapsed = clone_task.get("elapsedSec") if clone_task.get("elapsedSec") is not None else (status or {}).get("elapsedSec")
            clone_error = clone_task.get("error") if clone_task.get("error") is not None else (status or {}).get("error")
        elif clone_task:
            clone_task_status = terminal_status_from_chain(clone_task.get("status")) or str(clone_task.get("status") or "completed")
            clone_progress = clone_task.get("progress") if clone_task.get("progress") is not None else (1 if clone_task_status in {"completed", "success"} else None)
            clone_message = clone_task.get("message")
            clone_elapsed = clone_task.get("elapsedSec")
            clone_error = clone_task.get("error")
        elif has_clone_result:
            clone_task_status = terminal_status_from_chain(downstream_tts.get("status")) or "completed"
            clone_progress = 1
            clone_message = downstream_tts.get("reason")
            clone_elapsed = None
            clone_error = downstream_tts.get("error")
        else:
            clone_task_status = None
            clone_progress = None
            clone_message = None
            clone_elapsed = None
            clone_error = None

        batch_terminal_statuses = {"completed", "failed", "partial_failed", "cancelled"}
        asr_started_at = asr_task.get("createdAt") if asr_task else None
        asr_completed_at = asr_task.get("updatedAt") if asr_task_status in {"completed", "success", "failed", "error", "cancelled"} else None
        clone_started_at = clone_task.get("createdAt") if clone_task else None
        clone_completed_at = clone_task.get("updatedAt") if clone_task_status in {"completed", "success", "failed", "error", "cancelled"} else None
        asr_model = asr_eval.get("asrModel") or asr_eval.get("model") or asr_details.get("model")
        clone_model = clone_eval.get("cloneModel") or clone_eval.get("model")
        if latest_asr_batch:
            asr_task_status = str(latest_asr_batch.get("status") or "queued")
            asr_progress = latest_asr_batch.get("progress")
            asr_message = latest_asr_batch.get("message")
            asr_elapsed = latest_asr_batch.get("elapsedSec")
            asr_error = latest_asr_batch.get("error")
            asr_started_at = latest_asr_batch.get("createdAt")
            asr_completed_at = latest_asr_batch.get("updatedAt") if asr_task_status in batch_terminal_statuses else None
            asr_model = EVALUATION_BATCH_LABEL
            latest_asr_batch_items = [item for item in latest_asr_batch.get("items", []) if isinstance(item, dict)]
            if latest_asr_batch_items:
                asr_task = latest_asr_batch_items[-1]
        if latest_clone_batch:
            clone_task_status = str(latest_clone_batch.get("status") or "queued")
            clone_progress = latest_clone_batch.get("progress")
            clone_message = latest_clone_batch.get("message")
            clone_elapsed = latest_clone_batch.get("elapsedSec")
            clone_error = latest_clone_batch.get("error")
            clone_started_at = latest_clone_batch.get("createdAt")
            clone_completed_at = latest_clone_batch.get("updatedAt") if clone_task_status in batch_terminal_statuses else None
            clone_model = EVALUATION_BATCH_LABEL
            latest_clone_batch_items = [item for item in latest_clone_batch.get("items", []) if isinstance(item, dict)]
            if latest_clone_batch_items:
                clone_task = latest_clone_batch_items[-1]

        rows.append(
            {
                "taskId": payload.get("taskId", task_id),
                "filename": (frontend_payload.get("originalAudio") or {}).get("filename") or (audio.get("original") or {}).get("filename") or payload.get("filename") or "-",
                "protectedFilename": (frontend_payload.get("protectedAudio") or {}).get("filename") or (audio.get("protected") or {}).get("filename") or "-",
                "mode": request_payload.get("mode") or payload.get("mode", "joint"),
                "targetMode": target_mode,
                "parameters": {
                    "weightSemantic": semantic_cfg.get("weightSemantic", semantic_cfg.get("lambdaSemantic")),
                    "weightIdentity": timbre_cfg.get("weightIdentity", timbre_cfg.get("lambdaId", timbre_cfg.get("weightFeature", timbre_cfg.get("lambdaTimbre")))),
                    "weightFeature": timbre_cfg.get("weightFeature", timbre_cfg.get("lambdaTimbre")),
                    "weightPsy": psychoacoustic_cfg.get("weightPsy", psychoacoustic_cfg.get("lambdaPsy")),
                    "weightL2": optimization_cfg.get("weightL2", optimization_cfg.get("lambdaL2")),
                },
                "dataMode": "backend",
                "status": protection_status,
                "progress": protection_progress,
                "stage": protection_stage,
                "message": protection_message,
                "protectionStatus": protection_status,
                "protectionProgress": protection_progress,
                "protectionStage": protection_stage,
                "protectionMessage": protection_message,
                "protectionElapsedSec": protection_elapsed,
                "protectionCompletedAt": payload.get("completedAt"),
                "protectionError": protection_error,
                "asrStatus": asr_task_status,
                "asrProgress": asr_progress,
                "asrStage": "asr_eval" if has_asr_result else None,
                "asrMessage": asr_message,
                "asrElapsedSec": asr_elapsed,
                "asrStartedAt": asr_started_at,
                "asrCompletedAt": asr_completed_at,
                "asrError": asr_error,
                "asrSubId": asr_task.get("asrSubId") if asr_task else None,
                "cloneStatus": clone_task_status,
                "cloneProgress": clone_progress,
                "cloneStage": "downstream_tts_eval" if has_clone_result else None,
                "cloneMessage": clone_message,
                "cloneElapsedSec": clone_elapsed,
                "cloneStartedAt": clone_started_at,
                "cloneCompletedAt": clone_completed_at,
                "cloneError": clone_error,
                "cloneSubId": clone_task.get("cloneSubId") if clone_task else None,
                "hasAsrResult": bool(has_asr_result),
                "hasCloneResult": bool(has_clone_result),
                "asrTaskCount": len(asr_tasks) if asr_tasks else (1 if asr_task else 0),
                "cloneTaskCount": len(clone_tasks) if clone_tasks else (1 if clone_task else 0),
                "processingModel": frontend_payload.get("processingModel") or (details.get("generation") or {}).get("source"),
                "asrModel": asr_model,
                "cloneModel": clone_model,
                "createdAt": payload.get("createdAt") or (status or {}).get("createdAt"),
                "updatedAt": payload.get("updatedAt") or (status or {}).get("updatedAt"),
                "elapsedSec": protection_elapsed,
                "error": protection_error,
            }
        )
    return rows


@app.get("/api/tasks/{task_id}")
def task_status(task_id: str) -> dict[str, Any]:
    return read_task_status(task_id)


@app.get("/api/tasks/{task_id}/status")
def task_status_alias(task_id: str) -> dict[str, Any]:
    return task_status(task_id)


@app.get("/api/tasks/{task_id}/events")
def task_events(task_id: str) -> StreamingResponse:
    def stream():
        last_state: dict[str, Any] | None = None
        for _ in range(300):
            state = read_task_status(task_id)
            if state != last_state:
                payload = json.dumps(state, ensure_ascii=False)
                yield f"event: stage\ndata: {payload}\n\n"
                last_state = state
            if state.get("status") in {"completed", "success"}:
                completed = json.dumps({"taskId": task_id, "resultUrl": f"/api/tasks/{task_id}/result"}, ensure_ascii=False)
                yield f"event: completed\ndata: {completed}\n\n"
                return
            if state.get("status") in {"failed", "error"}:
                error_payload = json.dumps(
                    {
                        "taskId": task_id,
                        "stage": state.get("stage") or "protect_generation",
                        "message": state.get("message") or "Task failed",
                        "error": state.get("error"),
                    },
                    ensure_ascii=False,
                )
                yield f"event: error\ndata: {error_payload}\n\n"
                return
            time.sleep(1)
        final = json.dumps({"taskId": task_id, "status": "timeout", "message": "No terminal event received"}, ensure_ascii=False)
        yield f"event: error\ndata: {final}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/tasks/{task_id}/result")
def task_result(task_id: str) -> JSONResponse:
    status = read_task_status(task_id)
    if not (TASK_DIR / task_id / "result.json").exists():
        return structured_error(
            code="TASK_RESULT_NOT_READY",
            message=status.get("message") or "Task result is not ready yet.",
            status_code=409,
            request_id_value=request_id(),
            task_id=task_id,
            stage=status.get("stage") or "protect_generation",
            details={"status": status.get("status"), "progress": status.get("progress"), "error": status.get("error")},
        )
    result = ensure_protection_dnsmos(task_id, load_result(task_id))
    return JSONResponse(frontend_result(result))


@app.get("/api/tasks/{task_id}/psychoacoustic-slice")
def task_psychoacoustic_slice(task_id: str, mode: str = "mean", timeSec: float | None = None) -> JSONResponse:
    if not (TASK_DIR / task_id / "result.json").exists():
        status = read_task_status(task_id)
        return structured_error(
            code="TASK_RESULT_NOT_READY",
            message=status.get("message") or "Task result is not ready yet.",
            status_code=409,
            request_id_value=request_id(),
            task_id=task_id,
            stage=status.get("stage") or "protect_generation",
            details={"status": status.get("status"), "progress": status.get("progress"), "error": status.get("error")},
        )
    try:
        return JSONResponse(create_psychoacoustic_slice(task_id, mode=mode, time_sec=timeSec))
    except ValueError as exc:
        return structured_error(
            code="INVALID_PSYCHOACOUSTIC_SLICE_REQUEST",
            message=str(exc),
            status_code=400,
            request_id_value=request_id(),
            task_id=task_id,
            stage="psychoacoustic_slice",
            details={"mode": mode, "timeSec": timeSec},
        )
    except Exception as exc:
        return structured_error(
            code="PSYCHOACOUSTIC_SLICE_FAILED",
            message="Failed to compute psychoacoustic slice.",
            status_code=500,
            request_id_value=request_id(),
            task_id=task_id,
            stage="psychoacoustic_slice",
            details={"mode": mode, "timeSec": timeSec, "error": str(exc)},
        )


@app.get("/api/tasks/{task_id}/details")
def task_details(task_id: str) -> JSONResponse:
    if not (TASK_DIR / task_id / "result.json").exists():
        status = read_task_status(task_id)
        return structured_error(
            code="TASK_RESULT_NOT_READY",
            message=status.get("message") or "Task result is not ready yet.",
            status_code=409,
            request_id_value=request_id(),
            task_id=task_id,
            stage=status.get("stage") or "protect_generation",
            details={"status": status.get("status"), "progress": status.get("progress"), "error": status.get("error")},
        )
    result = ensure_protection_dnsmos(task_id, load_result(task_id))
    refresh_result_scores(result)
    return JSONResponse(result)


@app.post("/api/tasks/{task_id}/evaluation-batches")
def create_evaluation_batch(task_id: str, payload: EvaluationBatchRequest) -> JSONResponse:
    task_result_path(task_id)
    batch_type = str(payload.type or "").strip().lower()
    batch_id = str(payload.batchId or "").strip()
    if batch_type not in EVALUATION_BATCH_TYPES:
        return structured_error(
            code="INVALID_EVALUATION_BATCH_TYPE",
            message="Evaluation batch type must be asr or clone.",
            status_code=400,
            request_id_value=request_id(),
            task_id=task_id,
            stage="evaluation_batch",
            details={"type": payload.type, "supported": sorted(EVALUATION_BATCH_TYPES)},
        )
    if not batch_id:
        return structured_error(
            code="INVALID_EVALUATION_BATCH_ID",
            message="batchId is required.",
            status_code=400,
            request_id_value=request_id(),
            task_id=task_id,
            stage="evaluation_batch",
        )
    if not payload.items:
        return structured_error(
            code="EMPTY_EVALUATION_BATCH",
            message="Evaluation batch items must not be empty.",
            status_code=400,
            request_id_value=request_id(),
            task_id=task_id,
            stage="evaluation_batch",
        )

    with EVALUATION_COORDINATION_LOCK:
        _begin_evaluation_submission(task_id, batch_type)
        now = utc_now_iso()
        seen_item_ids: set[str] = set()
        items: list[dict[str, Any]] = []
        for index, raw_item in enumerate(payload.items):
            batch_item_id = str(raw_item.get("batchItemId") or "").strip()
            if not batch_item_id:
                return structured_error(
                    code="INVALID_EVALUATION_BATCH_ITEM",
                    message="Every evaluation batch item requires batchItemId.",
                    status_code=400,
                    request_id_value=request_id(),
                    task_id=task_id,
                    stage="evaluation_batch",
                    details={"index": index},
                )
            if batch_item_id in seen_item_ids:
                return structured_error(
                    code="DUPLICATE_EVALUATION_BATCH_ITEM",
                    message="batchItemId values must be unique within a batch.",
                    status_code=409,
                    request_id_value=request_id(),
                    task_id=task_id,
                    stage="evaluation_batch",
                    details={"batchItemId": batch_item_id},
                )
            seen_item_ids.add(batch_item_id)
            item = dict(raw_item)
            if batch_type == "clone" and not tts_model_requires_reference_text(str(item.get("model") or "")):
                _clear_clone_annotation_fields(item)
            annotation_asr_sub_id = str(item.get("annotationAsrSubId") or "").strip()
            if batch_type == "clone" and annotation_asr_sub_id:
                _begin_evaluation_submission(task_id, "asr")
                if is_subtask_deleted(task_id, annotation_asr_sub_id):
                    return structured_error(
                        code="ASR_ANNOTATION_DELETED",
                        message="所选 ASR 标注正在删除或已被删除，请重新执行 ASR 测试后再创建克隆测试批次。",
                        status_code=409,
                        request_id_value=request_id(),
                        task_id=task_id,
                        stage="evaluation_batch",
                        details={"annotationAsrSubId": annotation_asr_sub_id, "batchItemId": batch_item_id},
                    )
            item.update(
                {
                    "batchId": batch_id,
                    "batchItemId": batch_item_id,
                    "status": "queued",
                    "progress": 0.0,
                    "message": None,
                    "elapsedSec": 0.0,
                    "error": None,
                    "createdAt": now,
                    "updatedAt": now,
                }
            )
            items.append(item)

        batch = _aggregate_evaluation_batch(
            {
                "taskId": task_id,
                "batchId": batch_id,
                "type": batch_type,
                "label": EVALUATION_BATCH_LABEL,
                "status": "queued",
                "progress": 0.0,
                "completedCount": 0,
                "failedCount": 0,
                "totalCount": len(items),
                "createdAt": now,
                "updatedAt": now,
                "items": items,
            },
            now,
        )
        with TASK_STATUS_WRITE_LOCK:
            path = task_status_path(task_id)
            current = _load_task_status_document(path)
            current.setdefault("status", "completed")
            current.setdefault("progress", 1.0)
            current.setdefault("stage", "report_generation")
            current.setdefault("message", "Task completed")
            current.setdefault("error", None)
            storage_key = _batch_storage_key(batch_type)
            batches = [dict(item) for item in current.get(storage_key, []) if isinstance(item, dict)]
            active_batch = next(
                (
                    item
                    for item in reversed(batches)
                    if _normalized_status(item.get("status")) in {"queued", "running"}
                ),
                None,
            )
            if active_batch is not None:
                active_status = _normalized_status(active_batch.get("status"))
                return structured_error(
                    code="EVALUATION_BATCH_ACTIVE",
                    message="An evaluation batch of this type is already queued or running.",
                    status_code=409,
                    request_id_value=request_id(),
                    task_id=task_id,
                    stage="evaluation_batch",
                    details={
                        "batchId": active_batch.get("batchId"),
                        "status": active_status,
                        "type": str(active_batch.get("type") or batch_type),
                    },
                )
            if any(str(item.get("batchId") or "") == batch_id for item in batches):
                return structured_error(
                    code="EVALUATION_BATCH_EXISTS",
                    message="An evaluation batch with this batchId already exists.",
                    status_code=409,
                    request_id_value=request_id(),
                    task_id=task_id,
                    stage="evaluation_batch",
                    details={"batchId": batch_id, "type": batch_type},
                )
            batches.append(batch)
            current[storage_key] = batches
            _save_task_status_document(task_id, current, path, now)
    return JSONResponse(batch)


@app.post("/api/tasks/{task_id}/asr-eval")
def run_asr_eval(task_id: str, payload: AsrEvalRequest) -> JSONResponse:
    req_id = request_id()
    validation_error = validate_asr_eval_config(payload, req_id, task_id)
    if validation_error is not None:
        error = _json_response_error(validation_error)
        _mark_evaluation_batch_item_failed(task_id, "asr", payload.batchId, payload.batchItemId, error, str(error.get("message") or "ASR request validation failed"))
        return validation_error
    with EVALUATION_COORDINATION_LOCK:
        _begin_evaluation_submission(task_id, "asr")
        task_result_path(task_id)
        asr_sub_id = f"asr_{uuid.uuid4().hex[:8]}"
        asr_runtime_id = f"{task_id}:{asr_sub_id}"
        asr_started_at = time.time()
        cancel_event = threading.Event()

        def write_asr_status(**updates: Any) -> None:
            write_task_status(task_id, asrSubId=asr_sub_id, **updates)

        def persist_asr_result(result: dict[str, Any]) -> None:
            if is_subtask_deleted(task_id, asr_sub_id):
                return
            serialized = json.dumps(result, ensure_ascii=False, indent=2)
            history_dir = TASK_DIR / task_id / "asr_results"
            with SUBTASK_TOMBSTONE_LOCK:
                if is_subtask_deleted(task_id, asr_sub_id):
                    return
                (TASK_DIR / task_id / "asr_result.json").write_text(serialized, encoding="utf-8")
                history_dir.mkdir(parents=True, exist_ok=True)
                (history_dir / f"{asr_sub_id}.json").write_text(serialized, encoding="utf-8")

        def run_asr_background() -> None:
            try:
                ensure_task_not_cancelled(task_id, cancel_event)
                write_asr_status(
                    status="running",
                    progress=0.15,
                    stage="asr_eval",
                    message="后端正在执行 ASR 评估",
                    error=None,
                    asrResult=None,
                    elapsedSec=round(time.time() - asr_started_at, 3),
                )
                result = create_asr_eval(
                    task_id,
                    {**payload.model_dump(), "asrSubId": asr_sub_id},
                    cancel_event=cancel_event,
                    persist_allowed=lambda: not is_subtask_deleted(task_id, asr_sub_id),
                )
                ensure_task_not_cancelled(task_id, cancel_event)
                asr_payload = result.get("asr") if isinstance(result, dict) else {}
                asr_status = (asr_payload or {}).get("status") if isinstance(asr_payload, dict) else None
                if asr_status not in {"available", "computed", "partial"}:
                    reason = (asr_payload or {}).get("error") or (asr_payload or {}).get("reason") or "ASR evaluator did not generate transcriptions"
                    worker_diagnostics = asr_payload.get("diagnostics") if isinstance(asr_payload, dict) else None
                    write_task_log(
                        task_id,
                        {
                            "requestId": req_id,
                            "taskId": task_id,
                            "currentStage": "asr_eval",
                            "reason": str(reason),
                            **({"diagnostics": worker_diagnostics} if worker_diagnostics else {}),
                        },
                    )
                    write_asr_status(
                        status="failed",
                        progress=1,
                        stage="asr_eval",
                        message=str(reason),
                        error={
                            "code": "ASR_EVAL_FAILED",
                            "message": str(reason),
                            "requestId": req_id,
                            "taskId": task_id,
                            "stage": "asr_eval",
                            "details": {
                                "model": payload.model,
                                "reason": str(reason),
                                **({"diagnostics": worker_diagnostics} if worker_diagnostics else {}),
                            },
                        },
                        asrResult=result,
                        elapsedSec=round(time.time() - asr_started_at, 3),
                    )
                    persist_asr_result(result)
                    return
                write_asr_status(
                    status="completed",
                    progress=1,
                    stage="asr_eval",
                    message="ASR 评估已完成",
                    error=None,
                    asrResult=result,
                    elapsedSec=round(time.time() - asr_started_at, 3),
                )
                persist_asr_result(result)
            except TaskCancelledError:
                if (TASK_DIR / task_id).exists():
                    write_asr_status(
                        status="cancelled",
                        stage="asr_eval",
                        message="Task cancelled by delete request",
                        error=None,
                        elapsedSec=round(time.time() - asr_started_at, 3),
                    )
            except Exception as exc:
                if isinstance(exc, RuntimeError) and str(exc) == "TASK_CANCELLED":
                    if (TASK_DIR / task_id).exists():
                        write_asr_status(
                            status="cancelled",
                            stage="asr_eval",
                            message="Task cancelled by delete request",
                            error=None,
                            elapsedSec=round(time.time() - asr_started_at, 3),
                        )
                    return
                write_task_log(
                    task_id,
                    {
                        "requestId": req_id,
                        "taskId": task_id,
                        "currentStage": "asr_eval",
                        "exceptionType": type(exc).__name__,
                        "exceptionMessage": str(exc),
                        "stackTrace": traceback.format_exc(),
                    },
                )
                write_asr_status(
                    status="failed",
                    progress=1,
                    stage="asr_eval",
                    message=str(exc),
                    error={
                        "code": "ASR_EVAL_FAILED",
                        "message": str(exc),
                        "requestId": req_id,
                        "taskId": task_id,
                        "stage": "asr_eval",
                        "details": {"model": payload.model, "reason": str(exc)},
                    },
                    elapsedSec=round(time.time() - asr_started_at, 3),
                )
            finally:
                cleanup_task_runtime(task_id, runtime_id=asr_runtime_id)

        write_task_status(
            task_id,
            status="queued",
            progress=0.05,
            stage="asr_eval",
            message="后端已排入 ASR 评估队列",
            error=None,
            asrSubId=asr_sub_id,
            asrRequest=payload.model_dump(),
            asrResult=None,
            elapsedSec=0.0,
        )
        thread = threading.Thread(target=run_asr_background, daemon=True)
        register_task_runtime(task_id, cancel_event, thread, runtime_id=asr_runtime_id)
        thread.start()
    return JSONResponse({"taskId": task_id, "asrSubId": asr_sub_id, "status": "queued"})


@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: str) -> dict[str, Any]:
    task_dir = TASK_DIR / task_id
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail=f"task not found: {task_id}")
    status: dict[str, Any] = {}
    try:
        status = read_task_status(task_id)
    except Exception:
        status = {}
    cancel_events, threads, processes = request_all_task_cancels(task_id)
    mark_task_deleted(task_id)
    removed_from_queue = remove_pending_protect_job(task_id)
    cancelled = bool(cancel_events or processes) or status.get("status") in {"queued", "running"}
    process_pids = [process.pid for process in processes if process.pid is not None]
    process_deadline = time.monotonic() + 15.0
    for process in processes:
        if not process.is_alive():
            continue
        process.terminate()
        process.join(timeout=max(0.0, process_deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(timeout=5.0)
    thread_deadline = time.monotonic() + 15.0
    for thread in threads:
        if thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, thread_deadline - time.monotonic()))
    cleanup_all_task_runtimes(task_id)
    last_error: Exception | None = None
    for _ in range(8):
        try:
            if task_dir.exists():
                shutil.rmtree(task_dir)
            return {
                "taskId": task_id,
                "status": "deleted",
                "cancelled": cancelled,
                "removedFromQueue": removed_from_queue,
                "threadStopped": all(not thread.is_alive() for thread in threads),
                "processPid": process_pids[0] if process_pids else None,
                "processPids": process_pids,
                "processStopped": all(not process.is_alive() for process in processes),
            }
        except Exception as exc:
            last_error = exc
            time.sleep(0.25)
    raise HTTPException(status_code=409, detail=f"task delete pending: {last_error}")


def _delete_evaluation_subtask(task_id: str, batch_type: str, subtask_id: str) -> dict[str, Any]:
    if batch_type not in EVALUATION_BATCH_TYPES:
        raise HTTPException(status_code=400, detail=f"unknown evaluation type: {batch_type}")
    normalized_subtask_id = str(subtask_id or "").strip()
    expected_prefix = "asr_" if batch_type == "asr" else "clone_"
    if not normalized_subtask_id or not normalized_subtask_id.startswith(expected_prefix):
        raise HTTPException(status_code=400, detail=f"invalid {batch_type} subtask id")
    task_dir = TASK_DIR / task_id
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail=f"task not found: {task_id}")
    with EVALUATION_COORDINATION_LOCK:
        _begin_evaluation_submission(task_id, batch_type)
        if batch_type == "asr":
            status = _load_task_status_document(task_status_path(task_id))
            result_path = task_dir / "result.json"
            result = load_result(task_id) if result_path.exists() else None
            if normalized_subtask_id in _referenced_asr_subtask_ids(task_id, status, result):
                raise HTTPException(
                    status_code=409,
                    detail=f"ASR subtask is still referenced by a clone task: {normalized_subtask_id}",
                )
        mark_subtask_deleted(task_id, normalized_subtask_id)
        cancel_event, thread, process = request_subtask_cancel(task_id, normalized_subtask_id)
    process_pid = process.pid if process is not None else None
    if process is not None and process.is_alive():
        process.terminate()
        process.join(timeout=10.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=5.0)
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=15.0)

    status_removed = _remove_subtask_status(task_id, batch_type, normalized_subtask_id)
    result_removed, _ = _remove_subtask_result(task_id, batch_type, normalized_subtask_id)
    cleanup_task_runtime(task_id, runtime_id=_subtask_runtime_key(task_id, normalized_subtask_id))
    if not status_removed and not result_removed:
        raise HTTPException(status_code=404, detail=f"{batch_type} subtask not found: {normalized_subtask_id}")
    return {
        "taskId": task_id,
        "subtaskType": batch_type,
        "subtaskId": normalized_subtask_id,
        "status": "deleted",
        "cancelled": cancel_event is not None or process is not None,
        "processPid": process_pid,
    }


@app.delete("/api/tasks/{task_id}/asr-eval/{asr_sub_id}")
def delete_asr_eval(task_id: str, asr_sub_id: str) -> dict[str, Any]:
    return _delete_evaluation_subtask(task_id, "asr", asr_sub_id)


@app.delete("/api/tasks/{task_id}/clone-voice/{clone_sub_id}")
def delete_clone_voice(task_id: str, clone_sub_id: str) -> dict[str, Any]:
    return _delete_evaluation_subtask(task_id, "clone", clone_sub_id)


def _cancel_evaluation_runtimes(task_id: str, subtask_ids: set[str]) -> int:
    runtimes: list[tuple[str, threading.Event | None, threading.Thread | None, multiprocessing.Process | None]] = []
    for subtask_id in sorted(subtask_ids):
        mark_subtask_deleted(task_id, subtask_id)
        cancel_event, thread, process = request_subtask_cancel(task_id, subtask_id)
        runtimes.append((subtask_id, cancel_event, thread, process))
    for _, _, _, process in runtimes:
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=10.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
    for _, _, thread, _ in runtimes:
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=15.0)
    for subtask_id, _, _, _ in runtimes:
        cleanup_task_runtime(task_id, runtime_id=_subtask_runtime_key(task_id, subtask_id))
    return sum(1 for _, cancel_event, _, process in runtimes if cancel_event is not None or process is not None)


def _delete_evaluation_collection(task_id: str, batch_type: str) -> dict[str, Any]:
    if batch_type not in EVALUATION_BATCH_TYPES:
        raise HTTPException(status_code=400, detail=f"unknown evaluation type: {batch_type}")
    with EVALUATION_COORDINATION_LOCK:
        task_dir = TASK_DIR / task_id
        if not task_dir.exists():
            raise HTTPException(status_code=404, detail=f"task not found: {task_id}")
        deletion_key = _begin_evaluation_collection_deletion(task_id, batch_type)
        status = _load_task_status_document(task_status_path(task_id))
        result_path = task_dir / "result.json"
        result = load_result(task_id) if result_path.exists() else None
        all_subtask_ids = _evaluation_subtask_ids(task_id, batch_type, status, result)
        referenced_asr_ids = _referenced_asr_subtask_ids(task_id, status, result) if batch_type == "asr" else set()
        keep_subtask_ids = all_subtask_ids & referenced_asr_ids
        delete_subtask_ids = all_subtask_ids - keep_subtask_ids
        for subtask_id in delete_subtask_ids:
            mark_subtask_deleted(task_id, subtask_id)

    try:
        cancelled_count = _cancel_evaluation_runtimes(task_id, delete_subtask_ids)
        status_ids, _ = _retain_evaluation_status(task_id, batch_type, keep_subtask_ids)
        result_ids, updated_result = _retain_evaluation_results(task_id, batch_type, keep_subtask_ids)
        delete_subtask_ids.update((status_ids | result_ids) - keep_subtask_ids)
        _cleanup_evaluation_artifacts(task_id, batch_type, keep_subtask_ids, updated_result)

        return {
            "taskId": task_id,
            "subtaskType": batch_type,
            "status": "deleted",
            "deletedCount": len(delete_subtask_ids),
            "deletedSubtaskIds": sorted(delete_subtask_ids),
            "cancelledCount": cancelled_count,
            "preservedReferencedCount": len(keep_subtask_ids),
            "preservedAsrSubIds": sorted(keep_subtask_ids) if batch_type == "asr" else [],
        }
    finally:
        _finish_evaluation_collection_deletion(deletion_key)


@app.delete("/api/tasks/{task_id}/asr-evals")
def delete_asr_evals(task_id: str) -> dict[str, Any]:
    return _delete_evaluation_collection(task_id, "asr")


@app.delete("/api/tasks/{task_id}/clone-voices")
def delete_clone_voices(task_id: str) -> dict[str, Any]:
    return _delete_evaluation_collection(task_id, "clone")


@app.get("/api/tasks/{task_id}/download/protected-audio")
def download_protected_audio(task_id: str) -> FileResponse:
    result = load_result(task_id)
    filename = ((result.get("audio") or {}).get("protected") or {}).get("filename")
    path = TASK_DIR / task_id / "protected" / filename if filename else None
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="protected audio not found")
    return FileResponse(path, media_type="audio/wav", filename=filename)


@app.get("/api/tasks/{task_id}/download")
def download_task_file(task_id: str, type: str) -> Response:
    if type == "protected_audio":
        return download_protected_audio(task_id)
    if type == "report_pdf":
        return export_pdf({"taskId": task_id})
    if type == "evidence_zip":
        return evidence_zip(task_id)
    if type == "asr_report":
        path = TASK_DIR / task_id / "asr_result.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="ASR report not generated")
        return FileResponse(path, media_type="application/json", filename=f"{task_id}-asr-report.json")
    if type == "clone_result_zip":
        task_result_path(task_id)
        task_dir = TASK_DIR / task_id
        clone_result_path = task_dir / "clone_result.json"
        clone_dir = task_dir / "clones"
        if not clone_result_path.exists() and not clone_dir.exists():
            raise HTTPException(status_code=404, detail="clone result not generated")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            if clone_result_path.exists():
                archive.write(clone_result_path, clone_result_path.name)
            if clone_dir.exists():
                for path in clone_dir.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(task_dir).as_posix())
        headers = {"Content-Disposition": f'attachment; filename="{task_id}-clone-result.zip"'}
        return Response(buffer.getvalue(), media_type="application/zip", headers=headers)
    raise HTTPException(status_code=404, detail=f"download type not generated: {type}")


@app.post("/api/tasks/{task_id}/clone-voice")
def clone_voice(task_id: str, payload: CloneVoiceRequest) -> JSONResponse:
    req_id = request_id()
    if not payload.text.strip():
        error = {
            "code": "INVALID_CLONE_REQUEST",
            "message": "text is required",
            "requestId": req_id,
            "taskId": task_id,
            "stage": "downstream_tts_eval",
            "details": {},
        }
        _mark_evaluation_batch_item_failed(task_id, "clone", payload.batchId, payload.batchItemId, error, "text is required")
        raise HTTPException(status_code=400, detail="text is required")
    validation_error = validate_clone_config(payload, req_id, task_id)
    if validation_error is not None:
        error = _json_response_error(validation_error)
        _mark_evaluation_batch_item_failed(task_id, "clone", payload.batchId, payload.batchItemId, error, str(error.get("message") or "Clone request validation failed"))
        return validation_error
    with EVALUATION_COORDINATION_LOCK:
        _begin_evaluation_submission(task_id, "clone")
        task_result_path(task_id)
        resolved_payload, annotation_error = resolve_clone_annotation(task_id, payload, req_id)
        if annotation_error is not None:
            error = _json_response_error(annotation_error)
            _mark_evaluation_batch_item_failed(task_id, "clone", payload.batchId, payload.batchItemId, error, str(error.get("message") or "Clone annotation resolution failed"))
            return annotation_error
        assert resolved_payload is not None
        annotation_asr_sub_id = str(resolved_payload.get("annotationAsrSubId") or "").strip()
        if annotation_asr_sub_id:
            _begin_evaluation_submission(task_id, "asr")
            if is_subtask_deleted(task_id, annotation_asr_sub_id):
                return structured_error(
                    code="ASR_ANNOTATION_DELETED",
                    message="所选 ASR 标注正在删除或已被删除，请重新执行 ASR 测试后再提交克隆任务。",
                    status_code=409,
                    request_id_value=req_id,
                    task_id=task_id,
                    stage="downstream_tts_eval",
                    details={"annotationAsrSubId": annotation_asr_sub_id},
                )
        clone_sub_id = f"clone_{uuid.uuid4().hex[:8]}"
        resolved_payload["cloneSubId"] = clone_sub_id
        clone_runtime_id = f"{task_id}:{clone_sub_id}"
        cancel_event = threading.Event()

        def write_clone_status(**updates: Any) -> None:
            write_task_status(task_id, cloneSubId=clone_sub_id, **updates)

        def run_clone_background() -> None:
            clone_started_at = time.time()
            try:
                ensure_task_not_cancelled(task_id, cancel_event)
                write_clone_status(
                    status="running",
                    progress=0.12,
                    elapsedSec=round(time.time() - clone_started_at, 3),
                    stage="downstream_tts_eval",
                    message="后端正在生成下游 TTS 克隆音频",
                    error=None,
                )

                def on_clone_progress(**event: Any) -> None:
                    ensure_task_not_cancelled(task_id, cancel_event)
                    try:
                        progress_value = float(event.get("progress"))
                    except (TypeError, ValueError):
                        progress_value = 0.12
                    write_clone_status(
                        status="running",
                        progress=round(min(0.95, max(0.0, progress_value)), 3),
                        elapsedSec=round(time.time() - clone_started_at, 3),
                        stage="downstream_tts_eval",
                        message=str(event.get("message") or "后端正在生成下游 TTS 克隆音频"),
                        error=None,
                    )

                result = create_clone_voice(
                    task_id,
                    resolved_payload,
                    progress_callback=on_clone_progress,
                    cancel_event=cancel_event,
                    persist_allowed=lambda: not is_subtask_deleted(task_id, clone_sub_id),
                )
                ensure_task_not_cancelled(task_id, cancel_event)
                write_clone_status(
                    status="completed",
                    progress=1,
                    stage="downstream_tts_eval",
                    message=result.get("message") or "克隆已完成",
                    elapsedSec=round(time.time() - clone_started_at, 3),
                    error=None,
                    cloneResult=result,
                )
                serialized = json.dumps(result, ensure_ascii=False, indent=2)
                history_dir = TASK_DIR / task_id / "clone_results"
                with SUBTASK_TOMBSTONE_LOCK:
                    if is_subtask_deleted(task_id, clone_sub_id):
                        return
                    (TASK_DIR / task_id / "clone_result.json").write_text(serialized, encoding="utf-8")
                    history_dir.mkdir(parents=True, exist_ok=True)
                    (history_dir / f"{clone_sub_id}.json").write_text(serialized, encoding="utf-8")
            except (ValueError, FileNotFoundError) as exc:
                write_clone_status(
                    status="failed",
                    progress=1,
                    stage="downstream_tts_eval",
                    message=str(exc),
                    elapsedSec=round(time.time() - clone_started_at, 3),
                    error={
                        "code": "CLONE_BACKEND_UNAVAILABLE",
                        "message": str(exc),
                        "requestId": req_id,
                        "taskId": task_id,
                        "stage": "downstream_tts_eval",
                        "details": {"reason": "input_audio_missing"},
                    },
                )
            except TaskCancelledError:
                if (TASK_DIR / task_id).exists():
                    write_clone_status(
                        status="cancelled",
                        elapsedSec=round(time.time() - clone_started_at, 3),
                        stage="downstream_tts_eval",
                        message="任务已被删除请求取消",
                        error=None,
                    )
            except CloneBackendUnavailableError as exc:
                write_clone_status(
                    status="failed",
                    progress=1,
                    stage="downstream_tts_eval",
                    message=str(exc),
                    elapsedSec=round(time.time() - clone_started_at, 3),
                    error={
                        "code": "CLONE_BACKEND_UNAVAILABLE",
                        "message": str(exc),
                        "requestId": req_id,
                        "taskId": task_id,
                        "stage": "downstream_tts_eval",
                        "details": {"reason": exc.reason, **exc.diagnostics},
                    },
                )
                write_task_log(
                    task_id,
                    {
                        "requestId": req_id,
                        "taskId": task_id,
                        "currentStage": "downstream_tts_eval",
                        "reason": exc.reason,
                        "diagnostics": exc.diagnostics,
                        "exceptionType": type(exc).__name__,
                        "exceptionMessage": str(exc),
                    },
                )
            except RuntimeError as exc:
                if str(exc) == "TASK_CANCELLED":
                    if (TASK_DIR / task_id).exists():
                        write_clone_status(
                            status="cancelled",
                            elapsedSec=round(time.time() - clone_started_at, 3),
                            stage="downstream_tts_eval",
                            message="Task cancelled by delete request",
                            error=None,
                        )
                    return
                write_clone_status(
                    status="failed",
                    progress=1,
                    stage="downstream_tts_eval",
                    message=str(exc),
                    elapsedSec=round(time.time() - clone_started_at, 3),
                    error={
                        "code": "CLONE_BACKEND_UNAVAILABLE",
                        "message": str(exc),
                        "requestId": req_id,
                        "taskId": task_id,
                        "stage": "downstream_tts_eval",
                        "details": {"reason": "runtime_error"},
                    },
                )
            finally:
                cleanup_task_runtime(task_id, runtime_id=clone_runtime_id)

        write_clone_status(
            status="queued",
            progress=0.05,
            stage="downstream_tts_eval",
            message="后端已排入下游 TTS 克隆音频生成队列",
            error=None,
            elapsedSec=0.0,
            cloneResult=None,
            cloneRequest=resolved_payload,
        )
        thread = threading.Thread(target=run_clone_background, daemon=True)
        register_task_runtime(task_id, cancel_event, thread, runtime_id=clone_runtime_id)
        thread.start()
    return JSONResponse({"taskId": task_id, "cloneSubId": clone_sub_id, "status": "queued"})


@app.get("/api/artifacts/{task_id}/result.json")
def artifact_result_json(task_id: str) -> FileResponse:
    path = task_result_path(task_id)
    return FileResponse(path, media_type="application/json", filename="result.json")


@app.get("/api/artifacts/{task_id}/{kind}/{filename}")
def artifact_file(task_id: str, kind: str, filename: str) -> FileResponse:
    if kind not in {"source", "original", "protected"}:
        raise HTTPException(status_code=404, detail="unknown artifact kind")
    path = TASK_DIR / task_id / kind / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    media_type = "audio/wav" if path.suffix.lower() == ".wav" else "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=filename)


@app.get("/api/artifacts/{task_id}/clones/{clone_id}/{filename}")
def clone_artifact_file(task_id: str, clone_id: str, filename: str) -> FileResponse:
    path = TASK_DIR / task_id / "clones" / clone_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="clone artifact not found")
    media_type = "audio/wav" if path.suffix.lower() == ".wav" else "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=filename)


@app.post("/api/reports/export")
def export_pdf(payload: dict[str, Any] | None = None) -> Response:
    task_id = str((payload or {}).get("taskId") or "")
    if task_id:
        task_result_path(task_id)
    pdf = tiny_pdf_bytes("Voice Protection Report", f"Task: {task_id or 'unknown'}")
    headers = {"Content-Disposition": f'attachment; filename="{task_id or "task"}-report.pdf"'}
    return Response(pdf, media_type="application/pdf", headers=headers)


@app.get("/api/tasks/{task_id}/export/csv")
def export_csv(task_id: str) -> Response:
    result = load_result(task_id)
    frontend = frontend_result(result)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["metric", "value"])
    writer.writerow(["taskId", frontend["taskId"]])
    writer.writerow(["status", frontend["status"]])
    writer.writerow(["score", frontend["score"]])
    writer.writerow(["wer", frontend["asr"].get("wer")])
    writer.writerow(["cer", frontend["asr"].get("cer")])
    writer.writerow(["simBefore", frontend["speaker"].get("simBefore")])
    writer.writerow(["simAfter", frontend["speaker"].get("simAfter")])
    writer.writerow(["snr", frontend["quality"].get("snr")])
    writer.writerow(["pesq", frontend["quality"].get("pesq")])
    headers = {"Content-Disposition": f'attachment; filename="{task_id}-metrics.csv"'}
    return Response(buffer.getvalue(), media_type="text/csv; charset=utf-8", headers=headers)


@app.get("/api/tasks/{task_id}/download/evidence")
def evidence_zip(task_id: str) -> Response:
    task_result_path(task_id)
    task_dir = TASK_DIR / task_id
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
      for path in task_dir.rglob("*"):
          if path.is_file():
              archive.write(path, path.relative_to(task_dir).as_posix())
    headers = {"Content-Disposition": f'attachment; filename="{task_id}-evidence.zip"'}
    return Response(buffer.getvalue(), media_type="application/zip", headers=headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=int(os.getenv("SEME2E_API_PORT", "8000")), reload=False)
