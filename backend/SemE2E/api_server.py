from __future__ import annotations

import csv
import io
import json
import logging
import multiprocessing
import traceback
import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from result_adapter import (
    TASK_DIR,
    UPLOAD_DIR,
    CloneBackendUnavailableError,
    ProtectGenerationError,
    create_asr_eval,
    create_clone_voice,
    create_task,
    diagnose_capabilities,
    ensure_runtime_dirs,
    load_result,
    new_task_id,
    new_file_id,
    runtime_config,
    supported_tts_languages,
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
TASK_PROCESSES: dict[str, multiprocessing.Process] = {}
DELETED_TASK_IDS: set[str] = set()
TASK_REGISTRY_LOCK = threading.Lock()
DELETED_TASK_DIR = TASK_DIR.parent / "deleted_tasks"
DELETED_TASK_DIR.mkdir(parents=True, exist_ok=True)


class TaskCancelledError(RuntimeError):
    pass


def register_task_runtime(task_id: str, cancel_event: threading.Event, thread: threading.Thread | None = None, process: multiprocessing.Process | None = None) -> None:
    with TASK_REGISTRY_LOCK:
        TASK_CANCEL_EVENTS[task_id] = cancel_event
        if thread is not None:
            TASK_THREADS[task_id] = thread
        if process is not None:
            TASK_PROCESSES[task_id] = process


def cleanup_task_runtime(task_id: str) -> None:
    with TASK_REGISTRY_LOCK:
        TASK_CANCEL_EVENTS.pop(task_id, None)
        TASK_THREADS.pop(task_id, None)
        TASK_PROCESSES.pop(task_id, None)


def request_task_cancel(task_id: str) -> tuple[threading.Event | None, threading.Thread | None, multiprocessing.Process | None]:
    with TASK_REGISTRY_LOCK:
        cancel_event = TASK_CANCEL_EVENTS.get(task_id)
        thread = TASK_THREADS.get(task_id)
        process = TASK_PROCESSES.get(task_id)
    if cancel_event is not None:
        cancel_event.set()
    return cancel_event, thread, process


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
    language: str | None = "auto"
    speed: float | None = 1.0
    speakerPrompt: str | None = None


class AsrEvalRequest(BaseModel):
    model: str
    referenceText: str | None = None
    reference_text: str | None = None


def public_file_url(file_id: str, filename: str) -> str:
    return f"/api/files/{file_id}/{filename}"


def find_uploaded_file(file_id: str) -> dict[str, Any]:
    if file_id in FILES:
        return FILES[file_id]
    candidates = sorted(UPLOAD_DIR.glob(f"{file_id}_*"))
    if candidates:
        path = candidates[0]
        data = {"fileId": file_id, "filename": path.name.split("_", 1)[1], "path": path}
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


def _merge_subtask_status(current: dict[str, Any], updates: dict[str, Any], *, stage: str, key: str, result_key: str, sub_id_key: str) -> None:
    if updates.get("stage") != stage and result_key not in updates and sub_id_key not in updates:
        return
    now = utc_now_iso()
    previous = current.get(key)
    subtask = dict(previous) if isinstance(previous, dict) else {}
    if not subtask.get("createdAt") or updates.get(sub_id_key):
        subtask["createdAt"] = now
    for field in ["status", "progress", "stage", "message", "elapsedSec", "error"]:
        if field in updates:
            subtask[field] = updates.get(field)
    subtask["stage"] = stage
    if result_key in updates:
        subtask[result_key] = updates.get(result_key)
    if sub_id_key in updates:
        subtask[sub_id_key] = updates.get(sub_id_key)
    subtask["updatedAt"] = now
    current[key] = subtask


def write_task_status(task_id: str, **updates: Any) -> dict[str, Any]:
    if is_task_deleted(task_id):
        raise TaskCancelledError(f"task deleted: {task_id}")
    task_dir = TASK_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_status_path(task_id)
    current: dict[str, Any] = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    now = utc_now_iso()
    current.update(updates)
    _merge_subtask_status(current, updates, stage="asr_eval", key="asrTask", result_key="asrResult", sub_id_key="asrSubId")
    _merge_subtask_status(current, updates, stage="downstream_tts_eval", key="cloneTask", result_key="cloneResult", sub_id_key="cloneSubId")
    current.setdefault("taskId", task_id)
    current.setdefault("createdAt", now)
    current["updatedAt"] = now
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)
    return current


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
            message="Feature 编码器不在后端支持配置中。",
            status_code=400,
            request_id_value=req_id,
            stage="file_preprocess",
            details={"featureModels": unsupported_features, "supported": sorted(allowed_feature)},
        )
    legacy_errors = [
        _legacy_weight_error(semantic, "weightSemantic", "lambdaSemantic", 1.0),
        _legacy_weight_error(timbre, "weightFeature", "lambdaTimbre", 1.0),
        _legacy_weight_error(psychoacoustic := (payload.psychoacoustic or {}), "weightPsy", "lambdaPsy", None, 0.001),
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
            details={"legacyFields": legacy_errors, "requiredFields": ["weightFeature", "weightSemantic", "weightPsy", "weightL2"]},
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
    allowed_models = _model_values(models.get("tts"))
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
    return {
        "fileId": meta.get("fileId"),
        "filename": meta.get("filename") or fallback_name,
        "durationSec": meta.get("durationSec") or meta.get("duration"),
        "duration": meta.get("duration") or meta.get("durationSec"),
        "sampleRate": meta.get("sampleRate"),
        "channels": meta.get("channels"),
        "bitDepth": meta.get("bitDepth"),
        "sizeBytes": meta.get("sizeBytes") or 0,
        "format": meta.get("format") or Path(fallback_name).suffix.lstrip(".").upper() or "AUDIO",
        "src": meta.get("src"),
        "audioUrl": meta.get("audioUrl"),
        "downloadUrl": meta.get("downloadUrl"),
        "uploadedAt": meta.get("uploadedAt"),
        "fingerprint": meta.get("fingerprint"),
    }


def _frontend_clone(clone: dict[str, Any]) -> dict[str, Any]:
    return {
        "cloneId": clone.get("cloneId"),
        "taskId": clone.get("taskId"),
        "status": clone.get("status", "partial"),
        "source": clone.get("source"),
        "message": clone.get("message"),
        "request": clone.get("request") or {},
        "originalCloneAudio": _frontend_audio(clone.get("originalCloneAudio"), "original_clone.wav"),
        "protectedCloneAudio": _frontend_audio(clone.get("protectedCloneAudio"), "protected_clone.wav"),
        "cloneEval": clone.get("cloneEval"),
        "originalSimilarity": clone.get("originalSimilarity"),
        "protectedSimilarity": clone.get("protectedSimilarity"),
        "similarityDropRate": clone.get("similarityDropRate"),
        "embeddingDistanceBefore": clone.get("embeddingDistanceBefore"),
        "embeddingDistanceAfter": clone.get("embeddingDistanceAfter"),
        "embeddingDistanceIncreaseRate": clone.get("embeddingDistanceIncreaseRate"),
        "cloneConfidenceBefore": clone.get("cloneConfidenceBefore"),
        "cloneConfidenceAfter": clone.get("cloneConfidenceAfter"),
        "cloneConfidenceDropRate": clone.get("cloneConfidenceDropRate"),
        "cloneRadar": clone.get("cloneRadar"),
        "cloneTrend": clone.get("cloneTrend"),
        "cloneDefenseScore": clone.get("cloneDefenseScore"),
        "createdAt": clone.get("createdAt"),
    }


def frontend_result(result: dict[str, Any]) -> dict[str, Any]:
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
    semantic = details.get("semantic") or {}
    generation = details.get("generation") or {}
    perception = details.get("perception") or {}
    charts = result.get("charts") or {}
    metric_sources = summary.get("metricSources") or result.get("metricSources") or {}
    request = result.get("request") or {}
    optimization = request.get("optimization") or {}
    loss_final = generation.get("lossFinal")
    loss_weights = generation.get("lossWeights") or {}
    asr_status = asr.get("status")
    asr_eval = None
    if asr_status in {"available", "computed", "partial", "completed", "success"}:
        asr_eval = {
            "model": asr.get("model"),
            "asrModel": asr.get("model"),
            "language": asr.get("language"),
            "originalText": asr.get("referenceText") or asr.get("cleanTranscription"),
            "protectedText": asr.get("protectedTranscription"),
            "wer": asr.get("wer"),
            "cer": asr.get("cer"),
            "substituteRate": (asr.get("breakdown") or {}).get("substituteRate"),
            "insertRate": (asr.get("breakdown") or {}).get("insertRate"),
            "deleteRate": (asr.get("breakdown") or {}).get("deleteRate"),
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
            "lambdaFeat": _coalesce(loss_weights.get("lambdaFeat"), loss_weights.get("weight_feature")),
            "lambdaSem": _coalesce(loss_weights.get("lambdaSem"), loss_weights.get("weight_semantic")),
            "lambdaPsy": _coalesce(loss_weights.get("lambdaPsy"), loss_weights.get("weight_psy")),
            "lambda2": _coalesce(loss_weights.get("lambda2"), loss_weights.get("weight_l2")),
        },
        "optimizationTrace": generation.get("optimizationTrace") or [],
        "averageStepSec": generation.get("averageStepSec"),
        "asrEval": asr_eval,
        "cloneEval": clone_eval,
        "cloneResults": clone_results,
        "asr": {
            "originalText": asr.get("referenceText") or asr.get("cleanTranscription"),
            "protectedText": asr.get("protectedTranscription"),
            "wer": _coalesce(primary.get("wer"), asr.get("wer")),
            "cer": _coalesce(primary.get("cer"), asr.get("cer")),
            "tokenErrorRate": _coalesce(primary.get("tokenErrorRate"), asr.get("tokenErrorRate"), semantic.get("tokenErrorRate")),
            "tokenChangeRate": _coalesce(primary.get("tokenChangeRate"), asr.get("tokenChangeRate"), semantic.get("tokenChangeRate")),
            "semanticDrift": _coalesce(primary.get("semanticDrift"), asr.get("semanticDrift"), semantic.get("semanticDrift")),
            "insertRate": ((asr.get("breakdown") or {}).get("insertRate")),
            "deleteRate": ((asr.get("breakdown") or {}).get("deleteRate")),
            "substituteRate": ((asr.get("breakdown") or {}).get("substituteRate")),
            "status": asr.get("status"),
        },
        "speaker": {
            "simBefore": (details.get("speaker") or {}).get("simBefore") if (details.get("speaker") or {}).get("simBefore") is not None else sim_before,
            "simAfter": _coalesce((details.get("speaker") or {}).get("simAfter"), sim_after),
            "simDropRate": (details.get("speaker") or {}).get("simDropRate"),
            "embeddingDistanceBefore": (details.get("speaker") or {}).get("embeddingDistanceBefore"),
            "embeddingDistanceAfter": _coalesce((details.get("speaker") or {}).get("embeddingDistanceAfter"), (details.get("speaker") or {}).get("embeddingDistance")),
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


@app.get("/api/health")
def health() -> dict[str, Any]:
    capabilities_payload = diagnose_capabilities()
    return {
        "ok": True,
        "version": "sem-e2e-api-0.1",
        "time": utc_now_iso(),
        "device": capabilities_payload["device"],
        "availableChains": ["protect", "semantic", "asr", "speaker", "perception"],
        "optionalChains": {
            "tts": "unavailable: downstream TTS is not enabled by default",
            "pesq": "unavailable: PESQ is only returned when a real evaluator is installed",
        },
        "chains": capabilities_payload["chains"],
    }


@app.get("/api/capabilities")
def capabilities() -> dict[str, Any]:
    payload = diagnose_capabilities()
    payload["time"] = utc_now_iso()
    payload["version"] = "sem-e2e-api-0.1"
    return payload


@app.get("/api/config")
def config() -> dict[str, Any]:
    return {
        "ok": True,
        "time": utc_now_iso(),
        "config": runtime_config(),
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
    FILES[file_id] = data
    return {key: value for key, value in data.items() if key != "path"}


@app.get("/api/files/{file_id}/{filename}")
def get_uploaded_file(file_id: str, filename: str) -> FileResponse:
    data = find_uploaded_file(file_id)
    path = Path(data["path"])
    if filename != data["filename"]:
        raise HTTPException(status_code=404, detail="filename mismatch")
    return FileResponse(path, filename=data["filename"])


def run_protect_task_process(task_id: str, req_id: str, uploaded_path: str, uploaded_filename: str | None, file_id: str, payload_dict: dict[str, Any]) -> None:
    uploaded = {"path": uploaded_path, "filename": uploaded_filename}

    def ensure_process_task_not_deleted() -> None:
        if is_task_deleted(task_id):
            raise TaskCancelledError(f"task deleted: {task_id}")

    try:
        ensure_process_task_not_deleted()
        write_task_status(
            task_id,
            status="running",
            progress=0.18,
            stage="protect_generation",
            message="后端正在生成保护音频",
            error=None,
        )

        def on_protect_progress(**event: Any) -> None:
            ensure_process_task_not_deleted()
            explicit_progress = event.get("progress")
            if explicit_progress is not None:
                try:
                    progress_value = float(explicit_progress)
                except (TypeError, ValueError):
                    progress_value = 0.18
                write_task_status(
                    task_id,
                    status="running",
                    progress=round(min(0.99, max(0.0, progress_value)), 3),
                    stage=str(event.get("stage") or "protect_generation"),
                    message=str(event.get("message") or "后端正在处理保护任务"),
                    error=None,
                    progressSource="backend_stage",
                )
                return
            step = event.get("step")
            total = event.get("total") or 1
            try:
                ratio = (float(step) + 1.0) / max(float(total), 1.0)
            except (TypeError, ValueError):
                ratio = 0.0
            progress = 0.18 + min(1.0, max(0.0, ratio)) * 0.77
            write_task_status(
                task_id,
                status="running",
                progress=round(progress, 3),
                stage="protect_generation",
                message=f"Protect optimization step {int(step) + 1}/{int(total)}" if step is not None else "Backend is generating protected audio",
                error=None,
                currentStep=int(step) + 1 if step is not None else None,
                totalSteps=int(total),
                stageProgress=round(min(1.0, max(0.0, ratio)), 3),
                progressSource="semantic_vguard_step",
            )

        result = create_task(
            Path(uploaded_path),
            file_id,
            payload_dict,
            request_id=req_id,
            task_id=task_id,
            progress_callback=on_protect_progress,
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
            "capabilities": diagnostics.get("capabilities", {}).get("chains"),
            "suggestion": "Install/check backend dependencies and model checkpoints.",
        }
        write_task_status(
            task_id,
            status="failed",
            stage="protect_generation",
            message="保护音频生成失败：后端算法未生成保护音频。",
            error={
                "code": "PROTECT_GENERATION_FAILED",
                "message": "保护音频生成失败：后端算法未生成保护音频。",
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


@app.post("/api/tasks/protect")
def protect_task(payload: ProtectTaskRequest) -> dict[str, str]:
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
        error=None,
    )
    cancel_event = threading.Event()
    process = multiprocessing.Process(
        target=run_protect_task_process,
        args=(task_id, req_id, str(uploaded["path"]), uploaded.get("filename"), payload.fileId, payload.model_dump()),
        daemon=True,
    )
    register_task_runtime(task_id, cancel_event, process=process)
    process.start()
    return {"taskId": task_id, "status": "queued"}

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
                explicit_progress = event.get("progress")
                if explicit_progress is not None:
                    try:
                        progress_value = float(explicit_progress)
                    except (TypeError, ValueError):
                        progress_value = 0.18
                    write_task_status(
                        task_id,
                        status="running",
                        progress=round(min(0.99, max(0.0, progress_value)), 3),
                        stage=str(event.get("stage") or "protect_generation"),
                        message=str(event.get("message") or "后端正在处理保护任务"),
                        error=None,
                        progressSource="backend_stage",
                    )
                    return
                step = event.get("step")
                total = event.get("total") or 1
                try:
                    ratio = (float(step) + 1.0) / max(float(total), 1.0)
                except (TypeError, ValueError):
                    ratio = 0.0
                progress = 0.18 + min(1.0, max(0.0, ratio)) * 0.77
                write_task_status(
                    task_id,
                    status="running",
                    progress=round(progress, 3),
                    stage="protect_generation",
                    message=f"Protect optimization step {int(step) + 1}/{int(total)}" if step is not None else "Backend is generating protected audio",
                    error=None,
                    currentStep=int(step) + 1 if step is not None else None,
                    totalSteps=int(total),
                    stageProgress=round(min(1.0, max(0.0, ratio)), 3),
                    progressSource="semantic_vguard_step",
                )

            result = create_task(
                Path(uploaded["path"]),
                payload.fileId,
                payload.model_dump(),
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
        asr_task = (status or {}).get("asrTask") if isinstance((status or {}).get("asrTask"), dict) else {}
        clone_task = (status or {}).get("cloneTask") if isinstance((status or {}).get("cloneTask"), dict) else {}
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
        has_asr_result = bool(asr_task) or has_current_asr or asr_result_path.exists() or asr_details.get("status") in {"available", "computed", "partial", "failed", "error"}
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
        has_clone_result = bool(clone_task) or has_current_clone or clone_result_path.exists() or bool(clone_results) or downstream_tts.get("status") in {"computed", "partial", "failed", "error"}
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

        rows.append(
            {
                "taskId": payload.get("taskId", task_id),
                "filename": (audio.get("original") or {}).get("filename") or payload.get("filename") or "-",
                "protectedFilename": (audio.get("protected") or {}).get("filename") or "-",
                "mode": request_payload.get("mode") or payload.get("mode", "joint"),
                "targetMode": target_mode,
                "parameters": {
                    "weightSemantic": semantic_cfg.get("weightSemantic", semantic_cfg.get("lambdaSemantic")),
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
                "protectionError": protection_error,
                "asrStatus": asr_task_status,
                "asrProgress": asr_progress,
                "asrStage": "asr_eval" if has_asr_result else None,
                "asrMessage": asr_message,
                "asrElapsedSec": asr_elapsed,
                "asrError": asr_error,
                "cloneStatus": clone_task_status,
                "cloneProgress": clone_progress,
                "cloneStage": "downstream_tts_eval" if has_clone_result else None,
                "cloneMessage": clone_message,
                "cloneElapsedSec": clone_elapsed,
                "cloneError": clone_error,
                "hasAsrResult": bool(has_asr_result),
                "hasCloneResult": bool(has_clone_result),
                "wer": asr_eval.get("wer") if asr_eval else None,
                "simDropRate": _coalesce(clone_eval.get("similarityDropRate") if clone_eval else None, (details.get("speaker") or {}).get("simDropRate")),
                "pesq": protection_quality.get("pesq") if protection_quality else None,
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
    result = load_result(task_id)
    return JSONResponse(frontend_result(result))


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
    return JSONResponse(load_result(task_id))


@app.post("/api/tasks/{task_id}/asr-eval")
def run_asr_eval(task_id: str, payload: AsrEvalRequest) -> JSONResponse:
    task_result_path(task_id)
    req_id = request_id()
    validation_error = validate_asr_eval_config(payload, req_id, task_id)
    if validation_error is not None:
        return validation_error
    asr_sub_id = f"asr_{uuid.uuid4().hex[:8]}"
    cancel_event = threading.Event()
    write_task_status(
        task_id,
        status="running",
        progress=0.05,
        stage="asr_eval",
            message="后端已排入 ASR 评估队列",
        error=None,
        asrSubId=asr_sub_id,
        asrResult=None,
    )

    def run_asr_background() -> None:
        register_task_runtime(task_id, cancel_event, threading.current_thread())
        try:
            ensure_task_not_cancelled(task_id, cancel_event)
            write_task_status(
                task_id,
                status="running",
                progress=0.15,
                stage="asr_eval",
                message="后端正在执行 ASR 评估",
                error=None,
                asrResult=None,
            )
            result = create_asr_eval(task_id, payload.model_dump())
            ensure_task_not_cancelled(task_id, cancel_event)
            asr_payload = result.get("asr") if isinstance(result, dict) else {}
            asr_status = (asr_payload or {}).get("status") if isinstance(asr_payload, dict) else None
            if asr_status not in {"available", "computed", "partial"}:
                reason = (asr_payload or {}).get("error") or (asr_payload or {}).get("reason") or "ASR evaluator did not generate transcriptions"
                write_task_status(
                    task_id,
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
                        "details": {"model": payload.model, "reason": str(reason)},
                    },
                    asrResult=result,
                )
                result_path = TASK_DIR / task_id / "asr_result.json"
                result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                return
            write_task_status(
                task_id,
                status="completed",
                progress=1,
                stage="asr_eval",
                message="ASR 评估已完成",
                error=None,
                asrResult=result,
            )
            result_path = TASK_DIR / task_id / "asr_result.json"
            result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        except TaskCancelledError:
            if (TASK_DIR / task_id).exists():
                write_task_status(
                    task_id,
                    status="cancelled",
                    stage="asr_eval",
                    message="Task cancelled by delete request",
                    error=None,
                )
        except Exception as exc:
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
            write_task_status(
                task_id,
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
            )
        finally:
            cleanup_task_runtime(task_id)

    thread = threading.Thread(target=run_asr_background, daemon=True)
    register_task_runtime(task_id, cancel_event, thread)
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
    cancel_event, thread, process = request_task_cancel(task_id)
    mark_task_deleted(task_id)
    cancelled = cancel_event is not None or process is not None or status.get("status") in {"queued", "running"}
    process_pid = process.pid if process is not None else None
    if process is not None and process.is_alive():
        process.terminate()
        process.join(timeout=15.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=5.0)
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=15.0)
    cleanup_task_runtime(task_id)
    last_error: Exception | None = None
    for _ in range(8):
        try:
            if task_dir.exists():
                shutil.rmtree(task_dir)
            return {
                "taskId": task_id,
                "status": "deleted",
                "cancelled": cancelled,
                "threadStopped": thread is None or not thread.is_alive(),
                "processPid": process_pid,
                "processStopped": process is None or not process.is_alive(),
            }
        except Exception as exc:
            last_error = exc
            time.sleep(0.25)
    raise HTTPException(status_code=409, detail=f"task delete pending: {last_error}")


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
    task_result_path(task_id)
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    req_id = request_id()
    validation_error = validate_clone_config(payload, req_id, task_id)
    if validation_error is not None:
        return validation_error
    clone_sub_id = f"clone_{uuid.uuid4().hex[:8]}"
    cancel_event = threading.Event()
    write_task_status(
        task_id,
        status="running",
        progress=0.08,
        stage="downstream_tts_eval",
        message="后端已排入下游 TTS 克隆音频生成队列",
        error=None,
        elapsedSec=None,
        cloneResult=None,
        cloneRequest=payload.model_dump(),
        cloneSubId=clone_sub_id,
    )

    def run_clone_background() -> None:
        clone_started_at = time.time()
        register_task_runtime(task_id, cancel_event, threading.current_thread())
        try:
            ensure_task_not_cancelled(task_id, cancel_event)
            write_task_status(
                task_id,
                status="running",
                progress=0.12,
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
                write_task_status(
                    task_id,
                    status="running",
                    progress=round(min(0.95, max(0.0, progress_value)), 3),
                    stage="downstream_tts_eval",
                    message=str(event.get("message") or "后端正在生成下游 TTS 克隆音频"),
                    error=None,
                )

            result = create_clone_voice(task_id, payload.model_dump(), progress_callback=on_clone_progress, cancel_event=cancel_event)
            ensure_task_not_cancelled(task_id, cancel_event)
            write_task_status(
                task_id,
                status="completed",
                progress=1,
                stage="downstream_tts_eval",
                message=result.get("message") or "克隆已完成",
                elapsedSec=round(time.time() - clone_started_at, 3),
                error=None,
                cloneResult=result,
            )
            result_path = TASK_DIR / task_id / "clone_result.json"
            result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        except (ValueError, FileNotFoundError) as exc:
            write_task_status(
                task_id,
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
                write_task_status(
                    task_id,
                    status="cancelled",
                    stage="downstream_tts_eval",
                    message="任务已被删除请求取消",
                    error=None,
                )
        except CloneBackendUnavailableError as exc:
            write_task_status(
                task_id,
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
                    write_task_status(
                        task_id,
                        status="cancelled",
                        stage="downstream_tts_eval",
                        message="Task cancelled by delete request",
                        error=None,
                    )
                return
            write_task_status(
                task_id,
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
            cleanup_task_runtime(task_id)

    thread = threading.Thread(target=run_clone_background, daemon=True)
    register_task_runtime(task_id, cancel_event, thread)
    thread.start()
    return JSONResponse({"taskId": task_id, "cloneSubId": clone_sub_id, "status": "queued"})


@app.get("/api/artifacts/{task_id}/result.json")
def artifact_result_json(task_id: str) -> FileResponse:
    path = task_result_path(task_id)
    return FileResponse(path, media_type="application/json", filename="result.json")


@app.get("/api/artifacts/{task_id}/{kind}/{filename}")
def artifact_file(task_id: str, kind: str, filename: str) -> FileResponse:
    if kind not in {"original", "protected"}:
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
    import os

    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=int(os.getenv("SEME2E_API_PORT", "8000")), reload=False)
