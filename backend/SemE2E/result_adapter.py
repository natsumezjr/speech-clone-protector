from __future__ import annotations

import json
import math
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import wave
import zipfile
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from audio_preprocess import AudioPreprocessError, audio_preprocess_capabilities, preprocess_audio
from capability_cache import get_capabilities_snapshot
from dnsmos_quality import dnsmos_model_status
from metric_definitions import (
    align_audio_pair,
    compute_asr_metrics,
    compute_clone_eval,
    compute_clone_identity_score,
    compute_clone_quality_score,
    compute_direct_identity_score,
    compute_direct_speaker_metrics,
    compute_loss_summary,
    compute_overall_score,
    compute_perturbation_metrics,
    compute_psychoacoustic_metrics,
    compute_psychoacoustic_slice,
    compute_quality_metrics,
    compute_protection_quality_score,
    compute_protection_semantic_score,
    compute_semantic_token_metrics,
    metric_source,
)
from result_schema import default_chains, empty_charts, empty_details, empty_primary_metrics, utc_now_iso

ProgressCallback = Callable[..., None]
RESULT_WRITE_LOCK = threading.RLock()
DNSMOS_TASK_FLIGHTS_GUARD = threading.Lock()
DNSMOS_TASK_FLIGHTS: dict[str, threading.Event] = {}

CLONE_EVAL_MIRROR_FIELDS = (
    "directSimilarity",
    "originalSimilarity",
    "protectedSimilarity",
    "similarityDropRate",
    "embeddingDistanceBefore",
    "embeddingDistanceAfter",
    "embeddingDistanceDelta",
    "embeddingDistanceIncreaseRate",
    "cloneIdentityScore",
    "identityBaselineWeight",
    "cloneIdentityStatus",
    "cloneIdentityReason",
    "cleanCloneTranscription",
    "protectedCloneTranscription",
    "cloneAsrModel",
    "cloneAsrStatus",
    "cloneAsrReason",
    "cleanCloneTextAccuracy",
    "cleanCloneTextError",
    "protectedCloneTextAccuracy",
    "protectedCloneTextError",
    "cloneTextChangeAccuracy",
    "cloneTextChangeRate",
    "semanticBaselineWeight",
    "cloneTokenChangeRate",
    "cloneSemanticDrift",
    "cloneTokenScore",
    "cloneDriftScore",
    "cloneSemanticScore",
    "cloneSemanticStatus",
    "cloneSemanticReason",
    "cleanCloneQualityMos",
    "protectedCloneQualityMos",
    "cloneQualityDropRate",
    "cloneQualityRawScore",
    "cloneQualityRelevance",
    "cloneQualityScore",
    "qualityBaselineWeight",
    "cloneQualityModel",
    "cloneQualityModelPath",
    "cloneQualityStatus",
    "cloneQualityReason",
    "cloneConfidenceBefore",
    "cloneConfidenceAfter",
    "cloneConfidenceDropRate",
    "cloneRadar",
    "cloneTrend",
    "cloneDefenseScore",
    "createdAt",
)

ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = Path(os.getenv("SEME2E_RUNTIME_DIR", ROOT.parents[1] / "seme2e-runtime"))
UPLOAD_DIR = RUNTIME_DIR / "uploads"
TASK_DIR = RUNTIME_DIR / "tasks"
PROJECT_TTS_CACHE_DIR = ROOT / "checkpoints" / "tts"
os.environ.setdefault("TTS_HOME", str(PROJECT_TTS_CACHE_DIR))


class ProtectGenerationError(RuntimeError):
    def __init__(self, message: str, *, task_id: str, diagnostics: dict[str, Any], reason: str = "unknown") -> None:
        super().__init__(message)
        self.task_id = task_id
        self.diagnostics = diagnostics
        self.reason = reason


class CloneBackendUnavailableError(RuntimeError):
    def __init__(self, message: str, *, task_id: str, diagnostics: dict[str, Any], reason: str = "unknown") -> None:
        super().__init__(message)
        self.task_id = task_id
        self.diagnostics = diagnostics
        self.reason = reason


class IsolatedWorkerError(RuntimeError):
    def __init__(self, message: str, *, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def ensure_runtime_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    TASK_DIR.mkdir(parents=True, exist_ok=True)


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sync_clone_eval_fields(clone_result: dict[str, Any], clone_eval: dict[str, Any]) -> None:
    """Keep the per-clone canonical eval and legacy flattened fields consistent."""

    for key in CLONE_EVAL_MIRROR_FIELDS:
        nested_value = clone_eval.get(key)
        flat_value = clone_result.get(key)
        if nested_value is None and flat_value is not None:
            clone_eval[key] = flat_value
        elif key in clone_eval:
            clone_result[key] = nested_value


def read_wav_meta(path: Path) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            sample_rate = wav.getframerate()
            return {
                "durationSec": frames / sample_rate if sample_rate else None,
                "sampleRate": sample_rate,
                "channels": wav.getnchannels(),
                "bitDepth": wav.getsampwidth() * 8,
                "format": "WAV",
            }
    except Exception:
        return {
            "durationSec": None,
            "sampleRate": None,
            "channels": None,
            "bitDepth": None,
            "format": path.suffix.lstrip(".").upper() or "AUDIO",
        }


def read_wav_float(path: Path) -> tuple[np.ndarray, int] | None:
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
        if sample_width != 2:
            return None
        audio = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return audio, sample_rate
    except Exception:
        return None


def write_wav_float(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def audio_meta(path: Path, url: str, file_id: str | None = None) -> dict[str, Any]:
    meta = read_wav_meta(path)
    return {
        "fileId": file_id,
        "filename": path.name,
        "durationSec": meta["durationSec"],
        "sampleRate": meta["sampleRate"],
        "channels": meta["channels"],
        "bitDepth": meta["bitDepth"],
        "sizeBytes": path.stat().st_size if path.exists() else 0,
        "format": meta["format"],
        "audioUrl": url,
        "downloadUrl": url,
    }


def compute_snr_numpy(clean: np.ndarray, protected: np.ndarray) -> float | None:
    n = min(clean.shape[0], protected.shape[0])
    if n <= 0:
        return None
    clean = clean[:n]
    protected = protected[:n]
    noise = protected - clean
    signal_power = float(np.sum(clean * clean) + 1.0e-12)
    noise_power = float(np.sum(noise * noise) + 1.0e-12)
    return 10.0 * math.log10(signal_power / noise_power)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _env_float(name: str, default: float) -> float:
    value = to_float(os.getenv(name))
    return value if value is not None else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


def _apply_environment_overrides(
    environment: dict[str, str],
    env_overrides: Mapping[str, str | None] | None,
) -> None:
    if not env_overrides:
        return
    for name, value in env_overrides.items():
        if value is None:
            environment.pop(str(name), None)
        else:
            environment[str(name)] = str(value)


def _cuda_worker_runtime(
    requested_device: str,
    visible_devices_env: str,
) -> tuple[str, dict[str, str] | None]:
    visible_devices = os.getenv(visible_devices_env, "").strip()
    if not visible_devices:
        return requested_device, None
    return "cuda:0", {"CUDA_VISIBLE_DEVICES": visible_devices}


def _isolated_process_group_options() -> dict[str, Any]:
    if os.name == "posix":
        return {"start_new_session": True}
    if os.name == "nt":
        return {
            "creationflags": (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        }
    return {}


def _stop_isolated_worker(process: subprocess.Popen[str], *, grace_seconds: float = 5.0) -> None:
    root_process_exited = process.poll() is not None
    if root_process_exited and os.name != "posix":
        return
    grace_seconds = max(0.1, float(grace_seconds))
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=grace_seconds,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            try:
                process.terminate()
            except (OSError, ProcessLookupError):
                pass
        try:
            process.wait(timeout=grace_seconds)
            return
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except (OSError, ProcessLookupError):
                pass
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            pass
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass


def _run_cancellable_subprocess(
    command: list[str],
    *,
    cwd: str | Path,
    env: Mapping[str, str] | None,
    timeout_seconds: int,
    cancel_event: Any | None = None,
) -> subprocess.CompletedProcess[str]:
    timeout_seconds = max(1, int(timeout_seconds))
    popen_kwargs: dict[str, Any] = {
        "cwd": str(cwd),
        "env": dict(env) if env is not None else None,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    popen_kwargs.update(_isolated_process_group_options())
    process = subprocess.Popen(command, **popen_kwargs)
    started_at = time.monotonic()
    stdout = ""
    stderr = ""
    while True:
        if cancel_event is not None and cancel_event.is_set():
            _stop_isolated_worker(process)
            stdout, stderr = process.communicate()
            raise RuntimeError("TASK_CANCELLED")
        remaining = timeout_seconds - (time.monotonic() - started_at)
        if remaining <= 0:
            _stop_isolated_worker(process)
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(
                command,
                timeout_seconds,
                output=stdout,
                stderr=stderr,
            )
        try:
            stdout, stderr = process.communicate(timeout=min(0.25, remaining))
            break
        except subprocess.TimeoutExpired:
            continue
    return subprocess.CompletedProcess(command, int(process.returncode or 0), stdout, stderr)


def _run_isolated_json_worker(
    worker_path: Path,
    request_payload: dict[str, Any],
    *,
    timeout_seconds: int,
    cancel_event: Any | None = None,
    env_overrides: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    worker_path = worker_path.resolve()
    timeout_seconds = max(1, int(timeout_seconds))
    environment = os.environ.copy()
    existing_python_path = environment.get("PYTHONPATH", "").strip()
    environment["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing_python_path if existing_python_path else "")
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    _apply_environment_overrides(environment, env_overrides)
    command = [sys.executable, "-u", str(worker_path)]
    popen_kwargs: dict[str, Any] = {
        "cwd": str(ROOT),
        "env": environment,
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    popen_kwargs.update(_isolated_process_group_options())

    try:
        process = subprocess.Popen(command, **popen_kwargs)
    except Exception as exc:
        raise IsolatedWorkerError(
            f"Unable to start isolated worker {worker_path.name}: {exc}",
            diagnostics={
                "worker": str(worker_path),
                "pythonExecutable": sys.executable,
                "cwd": str(ROOT),
                "exceptionType": type(exc).__name__,
                "exceptionMessage": str(exc),
                "stackTrace": traceback.format_exc(),
            },
        ) from exc

    request_text: str | None = json.dumps(request_payload, ensure_ascii=False, allow_nan=False)
    started_at = time.monotonic()
    stdout = ""
    stderr = ""
    while True:
        if cancel_event is not None and cancel_event.is_set():
            _stop_isolated_worker(process)
            stdout, stderr = process.communicate()
            raise RuntimeError("TASK_CANCELLED")
        elapsed = time.monotonic() - started_at
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            _stop_isolated_worker(process)
            stdout, stderr = process.communicate()
            raise IsolatedWorkerError(
                f"Isolated worker {worker_path.name} timed out after {timeout_seconds}s",
                diagnostics={
                    "worker": str(worker_path),
                    "pythonExecutable": sys.executable,
                    "cwd": str(ROOT),
                    "returnCode": process.returncode,
                    "timeoutSec": timeout_seconds,
                    "stdoutTail": stdout[-4000:].strip(),
                    "stderrTail": stderr[-8000:].strip(),
                },
            )
        try:
            stdout, stderr = process.communicate(input=request_text, timeout=min(0.25, remaining))
            break
        except subprocess.TimeoutExpired:
            request_text = None

    response: dict[str, Any] | None = None
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            response = decoded
            break

    diagnostics: dict[str, Any] = {
        "worker": str(worker_path),
        "pythonExecutable": sys.executable,
        "cwd": str(ROOT),
        "returnCode": process.returncode,
        "timeoutSec": timeout_seconds,
        "elapsedSec": round(time.monotonic() - started_at, 3),
        "stdoutTail": stdout[-4000:].strip(),
        "stderrTail": stderr[-8000:].strip(),
    }
    if response is not None:
        diagnostics["response"] = response
    if response is None:
        raise IsolatedWorkerError(
            f"Isolated worker {worker_path.name} did not return a JSON response",
            diagnostics=diagnostics,
        )
    if process.returncode != 0 or response.get("ok") is not True:
        worker_error = response.get("error") if isinstance(response.get("error"), dict) else {}
        message = str(worker_error.get("message") or f"worker exited with code {process.returncode}")
        raise IsolatedWorkerError(message, diagnostics=diagnostics)
    return response


def _evaluate_dnsmos_pair_isolated(
    clean_path: Path,
    protected_path: Path,
    *,
    cancel_event: Any | None = None,
) -> dict[str, Any]:
    status = dnsmos_model_status()
    if status.get("status") != "available":
        return status
    try:
        _acquire_worker_slot(DNSMOS_WORKER_SLOTS, cancel_event)
        try:
            response = _run_isolated_json_worker(
                ROOT / "dnsmos_worker.py",
                {
                    "modelPath": status.get("modelPath"),
                    "originalPath": str(clean_path.resolve()),
                    "protectedPath": str(protected_path.resolve()),
                },
                timeout_seconds=_env_int("SEME2E_DNSMOS_WORKER_TIMEOUT_SECONDS", 180),
                cancel_event=cancel_event,
            )
        finally:
            DNSMOS_WORKER_SLOTS.release()
        return {
            "status": "available",
            "model": response.get("model") or status.get("model"),
            "modelPath": response.get("modelPath") or status.get("modelPath"),
            "provider": response.get("provider"),
            "clean": response.get("clean"),
            "protected": response.get("protected"),
            "cleanMos": to_float(response.get("cleanMos")),
            "protectedMos": to_float(response.get("protectedMos")),
            "reason": None,
        }
    except IsolatedWorkerError as exc:
        return {
            **status,
            "status": "error",
            "reason": f"语音质量评分生成失败：{exc}",
            "diagnostics": exc.diagnostics,
        }
    except RuntimeError as exc:
        if str(exc) == "TASK_CANCELLED":
            raise
        return {
            **status,
            "status": "error",
            "reason": f"语音质量评分生成失败：{exc}",
            "diagnostics": {
                "exceptionType": type(exc).__name__,
                "exceptionMessage": str(exc),
                "stackTrace": traceback.format_exc(),
            },
        }
    except Exception as exc:
        return {
            **status,
            "status": "error",
            "reason": f"语音质量评分生成失败：{exc}",
            "diagnostics": {
                "exceptionType": type(exc).__name__,
                "exceptionMessage": str(exc),
                "stackTrace": traceback.format_exc(),
            },
        }


def _dnsmos_fields_from_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    dns_value = to_float(evaluation.get("protectedMos"))
    invalid_value = dns_value is not None and not 1.0 <= dns_value <= 5.0
    if invalid_value:
        dns_value = None
    dns_status = "available" if dns_value is not None else str(evaluation.get("status") or "unavailable")
    if invalid_value:
        dns_status = "error"
    dns_reason = None
    if dns_value is None:
        dns_reason = str(
            evaluation.get("reason")
            or ("语音质量评分超出 1–5 分范围" if invalid_value else "语音质量评分尚未生成")
        )
    return {
        "dnsMos": dns_value,
        "dnsMosStatus": dns_status,
        "dnsMosReason": dns_reason,
        "dnsMosModel": evaluation.get("model"),
        "dnsMosModelPath": evaluation.get("modelPath"),
        "dnsMosProvider": evaluation.get("provider"),
        "dnsMosDiagnostics": evaluation.get("diagnostics"),
    }


def _transcribe_clone_pair_isolated(
    clean_clone_path: Path,
    protected_clone_path: Path,
    payload: dict[str, Any],
    *,
    cancel_event: Any | None = None,
    preferred_gpu: str | None = None,
) -> dict[str, Any]:
    model = str(
        payload.get("asrModel")
        or os.getenv("SEME2E_CLONE_ASR_MODEL")
        or os.getenv("SEME2E_ASR_MODEL")
        or "openai-whisper:base"
    ).strip()
    language = str(payload.get("language") or "auto").strip()
    if language.lower() in {"auto", "default", ""}:
        target_text = str(payload.get("text") or "")
        language = "zh" if any("\u4e00" <= char <= "\u9fff" for char in target_text) else "en"
    requested_device = os.getenv("SEME2E_CLONE_ASR_DEVICE") or os.getenv("SEME2E_API_DEVICE", "cpu")
    worker_timeout = _env_int("SEME2E_CLONE_ASR_WORKER_TIMEOUT_SECONDS", 600)
    try:
        def run_worker(
            worker_device: str,
            worker_env: Mapping[str, str] | None,
            _selected_gpu: str | None,
            attempt_timeout_seconds: float,
        ) -> dict[str, Any]:
            return _run_isolated_json_worker(
                ROOT / "asr_worker.py",
                {
                    "model": model,
                    "device": worker_device,
                    "language": language,
                    "originalPath": str(clean_clone_path.resolve()),
                    "protectedPath": str(protected_clone_path.resolve()),
                },
                timeout_seconds=attempt_timeout_seconds,
                cancel_event=cancel_event,
                **({"env_overrides": worker_env} if worker_env else {}),
            )

        response, selected_gpu, gpu_attempts = _run_gpu_worker_with_retry(
            operation_name=f"clone_asr:{model}",
            worker_slot=ASR_WORKER_SLOTS,
            requested_device=requested_device,
            visible_devices_env="SEME2E_CLONE_ASR_CUDA_VISIBLE_DEVICES",
            cancel_event=cancel_event,
            operation=run_worker,
            timeout_seconds=worker_timeout,
            explicit_device=bool(os.getenv("SEME2E_CLONE_ASR_DEVICE", "").strip()),
            preferred_gpu=preferred_gpu,
            minimum_free_mib=max(
                0,
                _env_int(
                    "SEME2E_ASR_GPU_MIN_FREE_MIB",
                    _env_int("SEME2E_GPU_MIN_FREE_MIB", 0),
                ),
            ),
        )
    except RuntimeError as exc:
        if str(exc) == "TASK_CANCELLED":
            raise
        return {
            "status": "error",
            "model": model,
            "originalText": None,
            "protectedText": None,
            "reason": f"克隆语音文本尚未生成：{exc}",
            **({"diagnostics": exc.diagnostics} if isinstance(exc, IsolatedWorkerError) else {}),
        }
    except Exception as exc:
        return {
            "status": "error",
            "model": model,
            "originalText": None,
            "protectedText": None,
            "reason": f"克隆语音文本尚未生成：{exc}",
            **({"diagnostics": exc.diagnostics} if isinstance(exc, IsolatedWorkerError) else {}),
        }
    return {
        "status": "available",
        "model": response.get("model") or model,
        "language": response.get("language") or language,
        "originalText": str(response.get("originalText") or ""),
        "protectedText": str(response.get("protectedText") or ""),
        "reason": None,
        "elapsedSec": response.get("elapsedSec"),
        "gpu": selected_gpu,
        **({"gpuAttempts": gpu_attempts} if gpu_attempts else {}),
    }


def _semantic_child_device(parent_value: str | None, worker_device: str) -> str:
    raw_value = str(parent_value or "").strip()
    if not raw_value:
        return worker_device
    normalized = raw_value.lower()
    if normalized.startswith("cpu"):
        return raw_value
    if normalized.startswith("cuda"):
        return "cuda:0" if worker_device.lower().startswith("cuda") else worker_device
    return raw_value


def _compute_clone_semantic_isolated(
    clean_clone_path: Path,
    protected_clone_path: Path,
    config: dict[str, Any],
    *,
    cancel_event: Any | None = None,
    preferred_gpu: str | None = None,
) -> dict[str, Any]:
    requested_device = os.getenv("SEME2E_SEMANTIC_DEVICE") or os.getenv("SEME2E_API_DEVICE", "cpu")
    worker_timeout = _env_int("SEME2E_SEMANTIC_WORKER_TIMEOUT_SECONDS", 900)
    try:
        def run_worker(
            worker_device: str,
            worker_env: Mapping[str, str] | None,
            _selected_gpu: str | None,
            attempt_timeout_seconds: float,
        ) -> dict[str, Any]:
            semantic_env = dict(worker_env or {})
            semantic_env["SEME2E_API_DEVICE"] = worker_device
            semantic_env["SEME2E_SEMANTIC_DEVICE"] = _semantic_child_device(
                os.getenv("SEME2E_SEMANTIC_DEVICE"),
                worker_device,
            )
            semantic_env["SEME2E_TOKENIZER_DEVICE"] = _semantic_child_device(
                os.getenv("SEME2E_TOKENIZER_DEVICE"),
                worker_device,
            )
            semantic_env["SEME2E_SEMANTIC_ENCODER_DEVICE"] = _semantic_child_device(
                os.getenv("SEME2E_SEMANTIC_ENCODER_DEVICE"),
                worker_device,
            )
            return _run_isolated_json_worker(
                ROOT / "semantic_metrics_worker.py",
                {
                    "originalPath": str(clean_clone_path.resolve()),
                    "protectedPath": str(protected_clone_path.resolve()),
                    "config": config,
                },
                timeout_seconds=attempt_timeout_seconds,
                cancel_event=cancel_event,
                **({"env_overrides": semantic_env} if semantic_env else {}),
            )

        response, selected_gpu, gpu_attempts = _run_gpu_worker_with_retry(
            operation_name="clone_semantic_metrics",
            worker_slot=SEMANTIC_WORKER_SLOTS,
            requested_device=requested_device,
            visible_devices_env="SEME2E_SEMANTIC_CUDA_VISIBLE_DEVICES",
            cancel_event=cancel_event,
            operation=run_worker,
            timeout_seconds=worker_timeout,
            explicit_device=bool(os.getenv("SEME2E_SEMANTIC_DEVICE", "").strip()),
            preferred_gpu=preferred_gpu,
            minimum_free_mib=max(
                0,
                _env_int(
                    "SEME2E_SEMANTIC_GPU_MIN_FREE_MIB",
                    _env_int("SEME2E_GPU_MIN_FREE_MIB", 0),
                ),
            ),
        )
        metrics = response.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("semantic worker did not return metrics")
        metrics["gpu"] = selected_gpu
        if gpu_attempts:
            metrics["gpuAttempts"] = gpu_attempts
        if not metrics.get("reason") and metrics.get("status") not in {"available", "partial"}:
            metrics["reason"] = metrics.get("error") or "克隆语义指标尚未生成"
        return metrics
    except RuntimeError as exc:
        if str(exc) == "TASK_CANCELLED":
            raise
        return {
            "status": "error",
            "tokenChangeRate": None,
            "semanticDrift": None,
            "reason": f"克隆语义指标尚未生成：{exc}",
            **({"diagnostics": exc.diagnostics} if isinstance(exc, IsolatedWorkerError) else {}),
        }
    except Exception as exc:
        return {
            "status": "error",
            "tokenChangeRate": None,
            "semanticDrift": None,
            "reason": f"克隆语义指标尚未生成：{exc}",
        }


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or default


ASR_WORKER_MAX_CONCURRENCY = max(1, _env_int("SEME2E_ASR_WORKER_MAX_CONCURRENCY", 2))
SEMANTIC_WORKER_MAX_CONCURRENCY = max(1, _env_int("SEME2E_SEMANTIC_WORKER_MAX_CONCURRENCY", 2))
COQUI_TTS_WORKER_MAX_CONCURRENCY = max(1, _env_int("SEME2E_COQUI_TTS_WORKER_MAX_CONCURRENCY", 2))
COSYVOICE_WORKER_MAX_CONCURRENCY = max(1, _env_int("SEME2E_COSYVOICE_WORKER_MAX_CONCURRENCY", 1))
GPT_SOVITS_WORKER_MAX_CONCURRENCY = min(
    2,
    max(1, _env_int("SEME2E_GPT_SOVITS_WORKER_MAX_CONCURRENCY", 2)),
)
DNSMOS_WORKER_MAX_CONCURRENCY = max(1, _env_int("SEME2E_DNSMOS_WORKER_MAX_CONCURRENCY", 1))
CLONE_GPU_MAX_CONCURRENCY = max(1, _env_int("SEME2E_CLONE_GPU_MAX_CONCURRENCY", 1))
ASR_WORKER_SLOTS = threading.BoundedSemaphore(ASR_WORKER_MAX_CONCURRENCY)
SEMANTIC_WORKER_SLOTS = threading.BoundedSemaphore(SEMANTIC_WORKER_MAX_CONCURRENCY)
COQUI_TTS_WORKER_SLOTS = threading.BoundedSemaphore(COQUI_TTS_WORKER_MAX_CONCURRENCY)
COSYVOICE_WORKER_SLOTS = threading.BoundedSemaphore(COSYVOICE_WORKER_MAX_CONCURRENCY)
DNSMOS_WORKER_SLOTS = threading.BoundedSemaphore(DNSMOS_WORKER_MAX_CONCURRENCY)
GPT_SOVITS_WORKER_SLOTS = threading.BoundedSemaphore(GPT_SOVITS_WORKER_MAX_CONCURRENCY)
CLONE_GPU_SLOTS_GUARD = threading.Lock()
CLONE_GPU_SLOTS: dict[str, threading.BoundedSemaphore] = {}
GPU_SLOT_CONDITION = threading.Condition()
GPT_SOVITS_GPU_LEASE_CONDITION = threading.Condition()
GPT_SOVITS_GPU_LEASES: set[str] = set()
GPT_SOVITS_GPU_LEASE_KEYS: dict[str, str] = {}
GPU_INVENTORY_CACHE_GUARD = threading.Lock()
GPU_INVENTORY_CACHE_AT = 0.0
GPU_INVENTORY_CACHE: tuple[tuple[str, ...], dict[str, int], dict[str, str]] = ((), {}, {})
GPU_ACQUIRE_TIMEOUT_MESSAGE = "等待可用 GPU 超时，请稍后重试，或释放显存后重新运行。"


def _wait_for_gpu_slot_change(
    cancel_event: Any | None,
    deadline: float,
    *,
    maximum_wait_seconds: float | None = None,
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("TASK_CANCELLED")
    now = time.monotonic()
    remaining = deadline - now
    if remaining <= 0:
        raise RuntimeError(GPU_ACQUIRE_TIMEOUT_MESSAGE)
    poll_seconds = max(0.05, _env_float("SEME2E_GPU_WAIT_POLL_SECONDS", 1.0))
    if maximum_wait_seconds is None:
        with GPU_SLOT_CONDITION:
            GPU_SLOT_CONDITION.wait(timeout=min(poll_seconds, remaining))
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("TASK_CANCELLED")
        return
    wait_deadline = min(deadline, now + max(0.05, maximum_wait_seconds))
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("TASK_CANCELLED")
        now = time.monotonic()
        remaining = min(deadline, wait_deadline) - now
        if remaining <= 0:
            return
        with GPU_SLOT_CONDITION:
            notified = GPU_SLOT_CONDITION.wait(timeout=min(poll_seconds, remaining))
        if notified:
            return


def _notify_gpu_slot_waiters() -> None:
    with GPU_SLOT_CONDITION:
        GPU_SLOT_CONDITION.notify_all()


def _acquire_worker_slot(
    semaphore: threading.BoundedSemaphore,
    cancel_event: Any | None,
    *,
    deadline: float | None = None,
) -> None:
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("TASK_CANCELLED")
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(GPU_ACQUIRE_TIMEOUT_MESSAGE)
            wait_seconds = min(0.25, remaining)
        else:
            wait_seconds = 0.25
        if not semaphore.acquire(timeout=wait_seconds):
            continue
        if cancel_event is not None and cancel_event.is_set():
            semaphore.release()
            raise RuntimeError("TASK_CANCELLED")
        return


def _gpu_slot_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return 0, int(value)
    except ValueError:
        return 1, value


def _visible_gpu_tokens(raw_value: str) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_value.split(","):
        item = raw_item.strip()
        if not item or item.lower() in {"-1", "none", "nodevfiles"} or item in seen:
            continue
        seen.add(item)
        values.append(item)
    return tuple(values)


def _query_nvidia_gpu_inventory() -> tuple[tuple[str, ...], dict[str, int], dict[str, str]]:
    """Return GPU launch indices, free memory aliases, and canonical physical keys."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.free",
        "--format=csv,noheader,nounits",
    ]
    process_options: dict[str, Any] = {}
    if os.name == "nt":
        process_options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(1, _env_int("SEME2E_NVIDIA_SMI_TIMEOUT_SECONDS", 3)),
            **process_options,
        )
    except (OSError, subprocess.SubprocessError):
        return (), {}, {}
    if completed.returncode != 0:
        return (), {}, {}

    indices: list[str] = []
    free_memory: dict[str, int] = {}
    canonical_keys: dict[str, str] = {}
    for raw_line in completed.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) < 3:
            continue
        index, gpu_uuid, free_raw = parts[0], parts[1], parts[2]
        try:
            free_mib = int(float(free_raw))
        except ValueError:
            continue
        if index and index not in indices:
            indices.append(index)
        canonical_key = (gpu_uuid or index).lower()
        if index:
            free_memory[index] = free_mib
            canonical_keys[index] = canonical_key
        if gpu_uuid:
            free_memory[gpu_uuid] = free_mib
            free_memory[gpu_uuid.lower()] = free_mib
            canonical_keys[gpu_uuid] = canonical_key
            canonical_keys[gpu_uuid.lower()] = canonical_key
        if canonical_key:
            free_memory[canonical_key] = free_mib
            canonical_keys[canonical_key] = canonical_key
    return tuple(indices), free_memory, canonical_keys


def _nvidia_gpu_inventory(
    *,
    force: bool = False,
) -> tuple[tuple[str, ...], dict[str, int], dict[str, str]]:
    global GPU_INVENTORY_CACHE_AT, GPU_INVENTORY_CACHE
    ttl_seconds = max(0.1, _env_float("SEME2E_GPU_INVENTORY_CACHE_SECONDS", 1.0))
    now = time.monotonic()
    with GPU_INVENTORY_CACHE_GUARD:
        # Normal polling is collapsed by the TTL. A retryable worker failure may
        # request one fresh sample so its post-failure free-memory baseline is not
        # confused with the pre-launch cache entry used to select the card.
        if not force and GPU_INVENTORY_CACHE_AT > 0 and now - GPU_INVENTORY_CACHE_AT < ttl_seconds:
            indices, free_memory, canonical_keys = GPU_INVENTORY_CACHE
            return indices, dict(free_memory), dict(canonical_keys)
        GPU_INVENTORY_CACHE = _query_nvidia_gpu_inventory()
        GPU_INVENTORY_CACHE_AT = time.monotonic()
        indices, free_memory, canonical_keys = GPU_INVENTORY_CACHE
        return indices, dict(free_memory), dict(canonical_keys)


def _unpack_gpu_inventory(
    inventory: tuple[Any, ...],
) -> tuple[tuple[str, ...], dict[str, int], dict[str, str]]:
    """Accept the current inventory shape and older two-field test fixtures."""
    indices = tuple(str(item) for item in (inventory[0] if inventory else ()))
    free_memory = dict(inventory[1]) if len(inventory) > 1 else {}
    if len(inventory) > 2:
        canonical_keys = {str(key): str(value) for key, value in dict(inventory[2]).items()}
    else:
        canonical_keys = {str(key): str(key) for key in free_memory}
    return indices, free_memory, canonical_keys


def _canonical_gpu_slot_key(token: str, canonical_keys: Mapping[str, str] | None = None) -> str:
    raw_token = str(token).strip()
    if not raw_token:
        return raw_token
    if canonical_keys is None:
        _, _, canonical_keys = _unpack_gpu_inventory(_nvidia_gpu_inventory())
    exact = canonical_keys.get(raw_token) or canonical_keys.get(raw_token.lower())
    if exact:
        return exact
    lowered = raw_token.lower()
    if lowered.startswith(("gpu-", "mig-")):
        prefix_matches = {
            canonical
            for alias, canonical in canonical_keys.items()
            if str(alias).lower().startswith(lowered)
        }
        if len(prefix_matches) == 1:
            return next(iter(prefix_matches))
        return lowered
    return raw_token


def _unique_gpu_tokens(
    candidates: tuple[str, ...],
    canonical_keys: Mapping[str, str],
) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        slot_key = _canonical_gpu_slot_key(candidate, canonical_keys)
        if not candidate or slot_key in seen:
            continue
        seen.add(slot_key)
        values.append(candidate)
    return tuple(values)


def _gpu_free_memory_mib(
    token: str,
    free_memory: Mapping[str, int],
    canonical_keys: Mapping[str, str],
) -> int | None:
    if token in free_memory:
        return free_memory[token]
    lowered = token.lower()
    if lowered in free_memory:
        return free_memory[lowered]
    return free_memory.get(_canonical_gpu_slot_key(token, canonical_keys))


def _cuda_device_index(requested_device: str) -> int:
    normalized = (requested_device or "").strip().lower()
    if ":" not in normalized:
        return 0
    try:
        return max(0, int(normalized.split(":", 1)[1]))
    except ValueError:
        return 0


def _worker_gpu_candidates(
    requested_device: str,
    visible_devices_env: str,
    *,
    explicit_device: bool = False,
) -> tuple[str, ...]:
    """Resolve physical GPU candidates without changing the API process device."""
    explicit_visible_devices = _visible_gpu_tokens(os.getenv(visible_devices_env, ""))
    if explicit_visible_devices:
        return explicit_visible_devices

    normalized_device = (requested_device or "").strip().lower()
    if not normalized_device.startswith("cuda"):
        return ()

    shared_pool = _visible_gpu_tokens(os.getenv("SEME2E_GPU_POOL", ""))
    parent_visible_devices = _visible_gpu_tokens(os.getenv("CUDA_VISIBLE_DEVICES", ""))
    logical_index = _cuda_device_index(normalized_device)
    if explicit_device:
        if 0 <= logical_index < len(shared_pool):
            return (shared_pool[logical_index],)
        if 0 <= logical_index < len(parent_visible_devices):
            return (parent_visible_devices[logical_index],)
        return (str(logical_index),)

    if shared_pool:
        return shared_pool
    if parent_visible_devices:
        return parent_visible_devices
    physical_indices, _, _ = _unpack_gpu_inventory(_nvidia_gpu_inventory())
    if physical_indices:
        return physical_indices
    return (str(logical_index),)


def _rank_gpu_candidates_by_free_memory(
    candidates: tuple[str, ...],
    *,
    force_refresh: bool = False,
) -> tuple[tuple[str, ...], dict[str, int]]:
    inventory = _unpack_gpu_inventory(_nvidia_gpu_inventory(force=force_refresh))
    _, free_memory, canonical_keys = inventory
    unique_candidates = _unique_gpu_tokens(candidates, canonical_keys)
    if not unique_candidates:
        return (), {}
    original_order = {candidate: index for index, candidate in enumerate(unique_candidates)}
    ranked = tuple(
        sorted(
            unique_candidates,
            key=lambda candidate: (
                0 if _gpu_free_memory_mib(candidate, free_memory, canonical_keys) is not None else 1,
                -(_gpu_free_memory_mib(candidate, free_memory, canonical_keys) or 0),
                original_order[candidate],
            ),
        )
    )
    return ranked, {
        candidate: free_mib
        for candidate in ranked
        if (free_mib := _gpu_free_memory_mib(candidate, free_memory, canonical_keys)) is not None
    }


def acquire_gpu_slot(
    candidates: tuple[str, ...],
    cancel_event: Any | None = None,
    minimum_free_mib: int = 0,
    *,
    allow_low_memory: bool = False,
    deadline: float | None = None,
    excluded_gpu_keys: set[str] | None = None,
) -> tuple[str, threading.BoundedSemaphore]:
    _, _, canonical_keys = _unpack_gpu_inventory(_nvidia_gpu_inventory())
    unique_candidates = _unique_gpu_tokens(candidates, canonical_keys)
    if not unique_candidates:
        raise RuntimeError("GPU worker requires at least one CUDA GPU candidate")
    if deadline is None:
        timeout_seconds = max(0.01, _env_float("SEME2E_GPU_ACQUIRE_TIMEOUT_SECONDS", 900.0))
        deadline = time.monotonic() + timeout_seconds
    excluded = {
        _canonical_gpu_slot_key(candidate, canonical_keys)
        for candidate in (excluded_gpu_keys or set())
    }
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("TASK_CANCELLED")
        if time.monotonic() >= deadline:
            raise RuntimeError(GPU_ACQUIRE_TIMEOUT_MESSAGE)
        ranked, free_memory = _rank_gpu_candidates_by_free_memory(unique_candidates)
        for candidate in ranked:
            if _canonical_gpu_slot_key(candidate, canonical_keys) in excluded:
                continue
            known_free_mib = free_memory.get(candidate)
            if (
                not allow_low_memory
                and minimum_free_mib > 0
                and known_free_mib is not None
                and known_free_mib < minimum_free_mib
            ):
                continue
            gpu_slot = _clone_gpu_slot(candidate)
            if not gpu_slot.acquire(blocking=False):
                continue
            if cancel_event is not None and cancel_event.is_set():
                release_gpu_slot(gpu_slot)
                raise RuntimeError("TASK_CANCELLED")
            return candidate, gpu_slot
        _wait_for_gpu_slot_change(cancel_event, deadline)


def release_gpu_slot(slot: threading.BoundedSemaphore) -> None:
    slot.release()
    _notify_gpu_slot_waiters()


def _acquire_best_gpu_slot(
    candidates: tuple[str, ...],
    cancel_event: Any | None,
    *,
    minimum_free_mib: int = 0,
    allow_low_memory: bool = False,
    deadline: float | None = None,
    excluded_gpu_keys: set[str] | None = None,
) -> tuple[str, threading.BoundedSemaphore]:
    return acquire_gpu_slot(
        candidates,
        cancel_event,
        minimum_free_mib,
        allow_low_memory=allow_low_memory,
        deadline=deadline,
        excluded_gpu_keys=excluded_gpu_keys,
    )


@contextmanager
def _isolated_worker_gpu_lease(
    worker_slot: threading.BoundedSemaphore,
    requested_device: str,
    visible_devices_env: str,
    cancel_event: Any | None,
    *,
    explicit_device: bool = False,
    preferred_gpu: str | None = None,
    minimum_free_mib: int = 0,
    candidate_overrides: tuple[str, ...] | None = None,
    deadline: float | None = None,
    excluded_gpu_keys: set[str] | None = None,
):
    acquired_slots: list[threading.BoundedSemaphore] = []
    try:
        _acquire_worker_slot(worker_slot, cancel_event, deadline=deadline)
        acquired_slots.append(worker_slot)
        explicit_visible_devices = _visible_gpu_tokens(os.getenv(visible_devices_env, ""))
        candidates = candidate_overrides
        if candidates is None:
            candidates = _worker_gpu_candidates(
                requested_device,
                visible_devices_env,
                explicit_device=explicit_device,
            )
        if (
            preferred_gpu
            and not explicit_visible_devices
            and not explicit_device
            and preferred_gpu in candidates
        ):
            candidates = (preferred_gpu, *(candidate for candidate in candidates if candidate != preferred_gpu))
        if candidates:
            selected_gpu, gpu_slot = _acquire_best_gpu_slot(
                candidates,
                cancel_event,
                minimum_free_mib=minimum_free_mib,
                allow_low_memory=(bool(explicit_visible_devices) and len(candidates) == 1) or explicit_device,
                deadline=deadline,
                excluded_gpu_keys=excluded_gpu_keys,
            )
            acquired_slots.append(gpu_slot)
            yield "cuda:0", {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": selected_gpu,
            }, selected_gpu
        else:
            yield requested_device, None, None
    finally:
        _release_worker_slots(acquired_slots)


def _clone_gpu_slot_keys(requested_device: str, visible_devices_env: str) -> tuple[str, ...]:
    explicit_visible_devices = os.getenv(visible_devices_env, "").strip()
    if explicit_visible_devices:
        return tuple(sorted(_visible_gpu_tokens(explicit_visible_devices), key=_gpu_slot_sort_key))

    normalized_device = (requested_device or "").strip().lower()
    if not normalized_device.startswith("cuda"):
        return ()

    parent_visible_devices = _visible_gpu_tokens(os.getenv("CUDA_VISIBLE_DEVICES", ""))
    logical_index = 0
    if ":" in normalized_device:
        try:
            logical_index = int(normalized_device.split(":", 1)[1])
        except ValueError:
            logical_index = 0
    if parent_visible_devices:
        if 0 <= logical_index < len(parent_visible_devices):
            return (parent_visible_devices[logical_index],)
        return tuple(sorted(parent_visible_devices, key=_gpu_slot_sort_key))
    return (str(logical_index),)


def _clone_gpu_slot(key: str) -> threading.BoundedSemaphore:
    canonical_key = _canonical_gpu_slot_key(key)
    with CLONE_GPU_SLOTS_GUARD:
        slot = CLONE_GPU_SLOTS.get(canonical_key)
        if slot is None:
            slot = threading.BoundedSemaphore(CLONE_GPU_MAX_CONCURRENCY)
            CLONE_GPU_SLOTS[canonical_key] = slot
        return slot


def _gpt_sovits_gpu_candidates(requested_device: str) -> tuple[str, ...]:
    configured_pool = os.getenv("SEME2E_GPT_SOVITS_GPU_POOL", "").strip()
    if not configured_pool:
        configured_pool = os.getenv("SEME2E_GPT_SOVITS_CUDA_VISIBLE_DEVICES", "").strip()
    candidates = _visible_gpu_tokens(configured_pool)
    if not candidates:
        candidates = _worker_gpu_candidates(
            requested_device,
            "SEME2E_GPT_SOVITS_CUDA_VISIBLE_DEVICES",
            explicit_device=bool(os.getenv("SEME2E_TTS_DEVICE", "").strip()),
        )
    if not candidates:
        raise RuntimeError("GPT-SoVITS requires at least one configured CUDA GPU")
    return candidates


def _release_gpt_sovits_gpu_lease(candidate: str) -> None:
    with GPT_SOVITS_GPU_LEASE_CONDITION:
        canonical_candidate = GPT_SOVITS_GPU_LEASE_KEYS.pop(candidate, candidate)
        GPT_SOVITS_GPU_LEASES.discard(canonical_candidate)
        GPT_SOVITS_GPU_LEASE_CONDITION.notify_all()
    _notify_gpu_slot_waiters()


def _gpt_sovits_minimum_free_mib() -> int:
    return max(
        0,
        _env_int(
            "SEME2E_GPT_SOVITS_GPU_MIN_FREE_MIB",
            max(
                24576,
                _env_int(
                    "SEME2E_CLONE_GPU_MIN_FREE_MIB",
                    _env_int("SEME2E_GPU_MIN_FREE_MIB", 0),
                ),
            ),
        ),
    )


def _acquire_gpt_sovits_gpu_resources(
    candidates: tuple[str, ...],
    cancel_event: Any | None,
    *,
    deadline: float | None = None,
    excluded_gpu_keys: set[str] | None = None,
) -> tuple[str, threading.BoundedSemaphore]:
    minimum_free_mib = _gpt_sovits_minimum_free_mib()
    if deadline is None:
        timeout_seconds = max(0.01, _env_float("SEME2E_GPU_ACQUIRE_TIMEOUT_SECONDS", 900.0))
        deadline = time.monotonic() + timeout_seconds
    _, _, initial_canonical_keys = _unpack_gpu_inventory(_nvidia_gpu_inventory())
    excluded = {
        _canonical_gpu_slot_key(candidate, initial_canonical_keys)
        for candidate in (excluded_gpu_keys or set())
    }
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("TASK_CANCELLED")
        if time.monotonic() >= deadline:
            raise RuntimeError(GPU_ACQUIRE_TIMEOUT_MESSAGE)
        ranked, free_memory = _rank_gpu_candidates_by_free_memory(candidates)
        for candidate in ranked:
            canonical_candidate = _canonical_gpu_slot_key(candidate, initial_canonical_keys)
            if canonical_candidate in excluded:
                continue
            known_free_mib = free_memory.get(candidate)
            if (
                minimum_free_mib > 0
                and known_free_mib is not None
                and known_free_mib < minimum_free_mib
            ):
                continue
            shared_gpu_slot = _clone_gpu_slot(candidate)
            if not shared_gpu_slot.acquire(blocking=False):
                continue
            with GPT_SOVITS_GPU_LEASE_CONDITION:
                if canonical_candidate in GPT_SOVITS_GPU_LEASES:
                    release_gpu_slot(shared_gpu_slot)
                    continue
                GPT_SOVITS_GPU_LEASES.add(canonical_candidate)
                GPT_SOVITS_GPU_LEASE_KEYS[candidate] = canonical_candidate
            if cancel_event is not None and cancel_event.is_set():
                release_gpu_slot(shared_gpu_slot)
                _release_gpt_sovits_gpu_lease(candidate)
                raise RuntimeError("TASK_CANCELLED")
            return candidate, shared_gpu_slot
        _wait_for_gpu_slot_change(cancel_event, deadline)


@contextmanager
def _gpt_sovits_gpu_resource_lease(
    requested_device: str,
    cancel_event: Any | None,
    *,
    candidate_overrides: tuple[str, ...] | None = None,
    deadline: float | None = None,
    excluded_gpu_keys: set[str] | None = None,
):
    candidate, shared_gpu_slot = _acquire_gpt_sovits_gpu_resources(
        candidate_overrides or _gpt_sovits_gpu_candidates(requested_device),
        cancel_event,
        deadline=deadline,
        excluded_gpu_keys=excluded_gpu_keys,
    )
    try:
        yield candidate
    finally:
        try:
            release_gpu_slot(shared_gpu_slot)
        finally:
            _release_gpt_sovits_gpu_lease(candidate)


def _release_worker_slots(slots: list[threading.BoundedSemaphore]) -> None:
    for slot in reversed(slots):
        slot.release()
    if slots:
        _notify_gpu_slot_waiters()


def _gpu_retry_deadline(_worker_timeout_seconds: float) -> tuple[float, float]:
    acquire_timeout = max(
        0.01,
        _env_float("SEME2E_GPU_ACQUIRE_TIMEOUT_SECONDS", 900.0),
    )
    retry_timeout = max(
        0.01,
        _env_float("SEME2E_GPU_RETRY_TIMEOUT_SECONDS", acquire_timeout),
    )
    return time.monotonic() + retry_timeout, retry_timeout


def _exception_diagnostics(exc: BaseException) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "exceptionType": type(exc).__name__,
        "exceptionMessage": str(exc),
    }
    if isinstance(exc, IsolatedWorkerError):
        diagnostics["workerDiagnostics"] = exc.diagnostics
    elif isinstance(exc, subprocess.TimeoutExpired):
        diagnostics.update(
            {
                "command": list(exc.cmd) if isinstance(exc.cmd, (list, tuple)) else str(exc.cmd),
                "timeoutSec": exc.timeout,
                "stdoutTail": str(exc.output or "")[-4000:].strip(),
                "stderrTail": str(exc.stderr or "")[-8000:].strip(),
            }
        )
    return diagnostics


def _gpu_resource_error_kind(exc: BaseException) -> str | None:
    diagnostics = _exception_diagnostics(exc)
    try:
        text = json.dumps(diagnostics, ensure_ascii=False, default=str).lower()
    except Exception:
        text = f"{type(exc).__name__}: {exc}".lower()
    memory_markers = (
        "cuda out of memory",
        "cuda error: out of memory",
        "hip out of memory",
        "cudnn_status_alloc_failed",
        "cublas_status_alloc_failed",
        "failed to allocate cuda",
    )
    if any(marker in text for marker in memory_markers):
        return "gpu_memory_exhausted"
    gpu_context_markers = (
        "cuda",
        "gpu",
        "cudnn",
        "cublas",
        "hip",
        "nccl",
    )
    generic_memory_markers = (
        "failed to allocate memory",
        "cannot allocate memory",
        "out of memory",
        "memoryerror",
    )
    if any(marker in text for marker in gpu_context_markers) and any(
        marker in text for marker in generic_memory_markers
    ):
        return "gpu_memory_exhausted"
    busy_markers = (
        "cuda-capable device is busy or unavailable",
        "all cuda-capable devices are busy or unavailable",
        "cuda device is busy or unavailable",
        "cuda_error_devices_unavailable",
        "cuda_error_device_already_in_use",
    )
    if any(marker in text for marker in busy_markers):
        return "gpu_temporarily_unavailable"
    if "resource temporarily unavailable" in text and any(
        marker in text for marker in ("cuda", "gpu", "cudnn", "cublas")
    ):
        return "gpu_temporarily_unavailable"
    cudnn_initialization_markers = (
        "cudnn_status_internal_error",
        "cudnncreate",
    )
    if any(marker in text for marker in cudnn_initialization_markers) and any(
        marker in text for marker in ("cuda", "gpu", "cudnn", "onnxruntime")
    ):
        return "gpu_temporarily_unavailable"
    return None


def _gpu_retry_attempt(
    operation_name: str,
    selected_gpu: str,
    exc: BaseException,
    attempt_number: int,
) -> dict[str, Any]:
    return {
        "attempt": attempt_number,
        "operation": operation_name,
        "gpu": selected_gpu,
        "reason": _gpu_resource_error_kind(exc),
        **_exception_diagnostics(exc),
    }


def _append_gpu_retry_attempt(
    attempts: list[dict[str, Any]],
    attempt: dict[str, Any],
) -> None:
    history_limit = max(3, _env_int("SEME2E_GPU_RETRY_DIAGNOSTIC_LIMIT", 12))
    if len(attempts) >= history_limit:
        # Keep the first failure and the most recent failures. The actual attempt
        # number remains on every row and is also exposed as gpuAttemptCount.
        del attempts[1]
    attempts.append(attempt)


def _gpu_failure_state(
    candidate: str,
    kind: str | None,
) -> tuple[str, dict[str, Any]]:
    _, free_memory, canonical_keys = _unpack_gpu_inventory(
        _nvidia_gpu_inventory(force=True)
    )
    canonical_key = _canonical_gpu_slot_key(candidate, canonical_keys)
    return canonical_key, {
        "kind": kind,
        "freeMemoryMiB": _gpu_free_memory_mib(candidate, free_memory, canonical_keys),
    }


def _recovered_gpu_keys(
    candidates: tuple[str, ...],
    failure_states: Mapping[str, Mapping[str, Any]],
    *,
    minimum_free_mib: int,
    cooldown_deadlines: Mapping[str, float] | None = None,
) -> set[str]:
    _, free_memory, canonical_keys = _unpack_gpu_inventory(
        _nvidia_gpu_inventory(force=True)
    )
    recovery_mib = max(1, _env_int("SEME2E_GPU_RETRY_MEMORY_RECOVERY_MIB", 512))
    now = time.monotonic()
    recovered: set[str] = set()
    for candidate in candidates:
        canonical_key = _canonical_gpu_slot_key(candidate, canonical_keys)
        cooldown_until = max(
            float((cooldown_deadlines or {}).get(candidate, 0.0)),
            float((cooldown_deadlines or {}).get(canonical_key, 0.0)),
        )
        if cooldown_until > now:
            continue
        state = failure_states.get(canonical_key) or failure_states.get(candidate)
        if state is None:
            recovered.update((candidate, canonical_key))
            continue
        if state.get("kind") != "gpu_memory_exhausted":
            recovered.update((candidate, canonical_key))
            continue
        previous_free = to_float(state.get("freeMemoryMiB"))
        current_free = _gpu_free_memory_mib(candidate, free_memory, canonical_keys)
        if previous_free is None or current_free is None:
            # When nvidia-smi cannot provide a stable sample, use the bounded
            # retry backoff rather than permanently excluding the device.
            recovered.update((candidate, canonical_key))
            continue
        required_free = max(
            int(previous_free) + recovery_mib,
            max(0, int(minimum_free_mib)),
        )
        if current_free >= required_free:
            recovered.update((candidate, canonical_key))
    return recovered


def _gpu_retry_backoff_seconds(exhausted_rounds: int) -> float:
    base = max(0.05, _env_float("SEME2E_GPU_RETRY_BACKOFF_SECONDS", 1.0))
    maximum = max(base, _env_float("SEME2E_GPU_RETRY_MAX_BACKOFF_SECONDS", 30.0))
    return min(maximum, base * (2 ** min(max(0, exhausted_rounds), 8)))


def _next_gpu_retry_wait_seconds(
    candidates: tuple[str, ...],
    excluded_gpu_keys: set[str],
    cooldown_deadlines: Mapping[str, float],
    exhausted_rounds: int,
) -> float:
    now = time.monotonic()
    _, _, canonical_keys = _unpack_gpu_inventory(_nvidia_gpu_inventory())
    pending_cooldowns: list[float] = []
    for candidate in candidates:
        canonical_key = _canonical_gpu_slot_key(candidate, canonical_keys)
        if candidate not in excluded_gpu_keys and canonical_key not in excluded_gpu_keys:
            continue
        cooldown_until = max(
            float(cooldown_deadlines.get(candidate, 0.0)),
            float(cooldown_deadlines.get(canonical_key, 0.0)),
        )
        if cooldown_until > now:
            pending_cooldowns.append(cooldown_until - now)
    if pending_cooldowns:
        return max(0.05, min(pending_cooldowns))
    return _gpu_retry_backoff_seconds(exhausted_rounds)


def _gpu_retry_error(
    operation_name: str,
    retry_timeout_seconds: float,
    attempts: list[dict[str, Any]],
    final_error: BaseException,
) -> IsolatedWorkerError:
    return IsolatedWorkerError(
        f"{operation_name} 在等待或切换 GPU 后仍未完成：{final_error}",
        diagnostics={
            "operation": operation_name,
            "retryTimeoutSec": retry_timeout_seconds,
            "retryableResourceError": True,
            "gpuAttemptCount": int(attempts[-1].get("attempt") or len(attempts)) if attempts else 0,
            "gpuAttemptsTruncated": bool(
                attempts and int(attempts[-1].get("attempt") or len(attempts)) > len(attempts)
            ),
            "gpuAttempts": attempts,
            "finalError": _exception_diagnostics(final_error),
        },
    )


def _remaining_gpu_candidates(
    candidates: tuple[str, ...],
    excluded_gpu_keys: set[str],
) -> tuple[str, ...]:
    if not excluded_gpu_keys:
        return candidates
    _, _, canonical_keys = _unpack_gpu_inventory(_nvidia_gpu_inventory())
    return tuple(
        candidate
        for candidate in candidates
        if candidate not in excluded_gpu_keys
        and _canonical_gpu_slot_key(candidate, canonical_keys) not in excluded_gpu_keys
    )


def _run_gpu_worker_with_retry(
    *,
    operation_name: str,
    worker_slot: threading.BoundedSemaphore,
    requested_device: str,
    visible_devices_env: str,
    cancel_event: Any | None,
    operation: Callable[[str, Mapping[str, str] | None, str | None, float], Any],
    timeout_seconds: float,
    explicit_device: bool = False,
    preferred_gpu: str | None = None,
    minimum_free_mib: int = 0,
    before_retry: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Any, str | None, list[dict[str, Any]]]:
    candidates = _worker_gpu_candidates(
        requested_device,
        visible_devices_env,
        explicit_device=explicit_device,
    )
    deadline, retry_timeout = _gpu_retry_deadline(timeout_seconds)
    attempts: list[dict[str, Any]] = []
    excluded_gpu_keys: set[str] = set()
    failure_states: dict[str, dict[str, Any]] = {}
    failure_counts: dict[str, int] = {}
    cooldown_deadlines: dict[str, float] = {}
    attempt_count = 0
    exhausted_rounds = 0
    last_resource_error: BaseException | None = None

    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("TASK_CANCELLED")
        if time.monotonic() >= deadline:
            if last_resource_error is not None:
                raise _gpu_retry_error(
                    operation_name,
                    retry_timeout,
                    attempts,
                    last_resource_error,
                ) from last_resource_error
            raise RuntimeError(GPU_ACQUIRE_TIMEOUT_MESSAGE)

        remaining_candidates = _remaining_gpu_candidates(candidates, excluded_gpu_keys)
        if candidates and not remaining_candidates:
            try:
                _wait_for_gpu_slot_change(
                    cancel_event,
                    deadline,
                    maximum_wait_seconds=_next_gpu_retry_wait_seconds(
                        candidates,
                        excluded_gpu_keys,
                        cooldown_deadlines,
                        exhausted_rounds,
                    ),
                )
            except RuntimeError as wait_error:
                if (
                    str(wait_error) == GPU_ACQUIRE_TIMEOUT_MESSAGE
                    and last_resource_error is not None
                ):
                    raise _gpu_retry_error(
                        operation_name,
                        retry_timeout,
                        attempts,
                        last_resource_error,
                    ) from last_resource_error
                raise
            recovered_keys = _recovered_gpu_keys(
                candidates,
                failure_states,
                minimum_free_mib=minimum_free_mib,
                cooldown_deadlines=cooldown_deadlines,
            )
            if recovered_keys:
                excluded_gpu_keys.difference_update(recovered_keys)
                for recovered_key in recovered_keys:
                    cooldown_deadlines.pop(recovered_key, None)
                exhausted_rounds = 0
            else:
                exhausted_rounds += 1
            continue

        selected_gpu: str | None = None
        retry_error: BaseException | None = None
        try:
            with _isolated_worker_gpu_lease(
                worker_slot,
                requested_device,
                visible_devices_env,
                cancel_event,
                explicit_device=explicit_device,
                preferred_gpu=preferred_gpu,
                minimum_free_mib=minimum_free_mib,
                candidate_overrides=remaining_candidates if candidates else (),
                deadline=deadline,
            ) as (worker_device, worker_env, selected_gpu):
                try:
                    remaining_attempt_seconds = deadline - time.monotonic()
                    if remaining_attempt_seconds <= 0:
                        raise RuntimeError(GPU_ACQUIRE_TIMEOUT_MESSAGE)
                    attempt_timeout_seconds = min(
                        max(0.01, float(timeout_seconds)),
                        remaining_attempt_seconds,
                    )
                    result = operation(
                        worker_device,
                        worker_env,
                        selected_gpu,
                        attempt_timeout_seconds,
                    )
                except Exception as exc:
                    if selected_gpu is None or _gpu_resource_error_kind(exc) is None:
                        raise
                    retry_error = exc
        except RuntimeError as exc:
            if str(exc) == "TASK_CANCELLED":
                raise
            if attempts and str(exc) == GPU_ACQUIRE_TIMEOUT_MESSAGE and last_resource_error is not None:
                raise _gpu_retry_error(
                    operation_name,
                    retry_timeout,
                    attempts,
                    last_resource_error,
                ) from last_resource_error
            raise

        if retry_error is None:
            return result, selected_gpu, attempts

        last_resource_error = retry_error
        attempt_count += 1
        attempt = _gpu_retry_attempt(
            operation_name,
            str(selected_gpu),
            retry_error,
            attempt_count,
        )
        canonical_key, failure_state = _gpu_failure_state(
            str(selected_gpu),
            str(attempt.get("reason") or "") or None,
        )
        attempt["freeMemoryMiBAfterFailure"] = failure_state.get("freeMemoryMiB")
        failure_count = failure_counts.get(canonical_key, 0) + 1
        failure_counts[canonical_key] = failure_count
        retry_cooldown_seconds = _gpu_retry_backoff_seconds(failure_count - 1)
        cooldown_until = time.monotonic() + retry_cooldown_seconds
        cooldown_deadlines[canonical_key] = cooldown_until
        cooldown_deadlines[str(selected_gpu)] = cooldown_until
        attempt["retryCooldownSec"] = retry_cooldown_seconds
        _append_gpu_retry_attempt(attempts, attempt)
        failure_states[canonical_key] = failure_state
        failure_states[str(selected_gpu)] = failure_state
        excluded_gpu_keys.update((str(selected_gpu), canonical_key))
        if before_retry is not None:
            try:
                before_retry(attempt)
            except Exception as cleanup_error:
                attempt["retryCleanupError"] = _exception_diagnostics(cleanup_error)
                raise _gpu_retry_error(
                    operation_name,
                    retry_timeout,
                    attempts,
                    cleanup_error,
                ) from cleanup_error


def _acquire_clone_worker_slots(
    model_slot: threading.BoundedSemaphore,
    gpu_keys: tuple[str, ...],
    cancel_event: Any | None,
) -> list[threading.BoundedSemaphore]:
    acquired_slots: list[threading.BoundedSemaphore] = []
    try:
        _acquire_worker_slot(model_slot, cancel_event)
        acquired_slots.append(model_slot)
        if gpu_keys:
            _, gpu_slot = _acquire_best_gpu_slot(gpu_keys, cancel_event)
            acquired_slots.append(gpu_slot)
        return acquired_slots
    except Exception:
        _release_worker_slots(acquired_slots)
        raise


def maximum_clone_worker_concurrency(
    *,
    coqui_limit: int,
    cosyvoice_limit: int,
    gpt_sovits_limit: int,
    clone_gpu_limit: int,
    coqui_gpu_keys: tuple[str, ...],
    cosyvoice_gpu_keys: tuple[str, ...],
    gpt_sovits_gpu_keys: tuple[str, ...],
) -> int:
    """Return the largest clone workload that can run under shared GPU slots."""
    worker_limits = {
        "coquiTts": max(0, int(coqui_limit)),
        "cosyVoice": max(0, int(cosyvoice_limit)),
        "gptSoVits": max(0, int(gpt_sovits_limit)),
    }
    worker_gpu_keys = {
        "coquiTts": tuple(dict.fromkeys(coqui_gpu_keys)),
        "cosyVoice": tuple(dict.fromkeys(cosyvoice_gpu_keys)),
        "gptSoVits": tuple(dict.fromkeys(gpt_sovits_gpu_keys)),
    }
    unconstrained_cpu_workers = sum(
        worker_limits[name]
        for name in ("coquiTts", "cosyVoice")
        if not worker_gpu_keys[name]
    )
    return unconstrained_cpu_workers + maximum_gpu_worker_concurrency(
        worker_limits=worker_limits,
        worker_gpu_keys=worker_gpu_keys,
        gpu_slot_limit=clone_gpu_limit,
    )


def maximum_gpu_worker_concurrency(
    *,
    worker_limits: Mapping[str, int],
    worker_gpu_keys: Mapping[str, tuple[str, ...]],
    gpu_slot_limit: int,
) -> int:
    """Maximum bipartite matching of workers to physical GPU slot tokens."""
    slot_limit = max(1, int(gpu_slot_limit))
    _, _, canonical_keys = _unpack_gpu_inventory(_nvidia_gpu_inventory())
    normalized_worker_gpu_keys = {
        worker_name: tuple(
            dict.fromkeys(_canonical_gpu_slot_key(key, canonical_keys) for key in keys)
        )
        for worker_name, keys in worker_gpu_keys.items()
    }
    gpu_slots: list[tuple[str, int]] = []
    all_gpu_keys = sorted(
        {
            key
            for keys in normalized_worker_gpu_keys.values()
            for key in keys
        },
        key=_gpu_slot_sort_key,
    )
    for gpu_key in all_gpu_keys:
        gpu_slots.extend((gpu_key, slot_index) for slot_index in range(slot_limit))

    jobs: list[tuple[str, int, tuple[str, ...]]] = []
    for worker_name, raw_limit in worker_limits.items():
        candidates = normalized_worker_gpu_keys.get(worker_name, ())
        for worker_index in range(max(0, int(raw_limit))):
            jobs.append((worker_name, worker_index, candidates))
    jobs.sort(key=lambda item: (len(item[2]) if item[2] else sys.maxsize, item[0], item[1]))

    matched_slots: dict[tuple[str, int], tuple[str, int, tuple[str, ...]]] = {}

    def assign(job: tuple[str, int, tuple[str, ...]], visited: set[tuple[str, int]]) -> bool:
        candidate_keys = set(job[2])
        for gpu_slot in gpu_slots:
            if gpu_slot in visited or gpu_slot[0] not in candidate_keys:
                continue
            visited.add(gpu_slot)
            previous_job = matched_slots.get(gpu_slot)
            if previous_job is None or assign(previous_job, visited):
                matched_slots[gpu_slot] = job
                return True
        return False

    for job in jobs:
        if job[2]:
            assign(job, set())
    return len(matched_slots)


def _optional_env_bool(name: str) -> bool | None:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return None
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _gpu_key_set(keys: tuple[str, ...]) -> set[str]:
    if not keys:
        return set()
    _, _, canonical_keys = _unpack_gpu_inventory(_nvidia_gpu_inventory())
    return {_canonical_gpu_slot_key(key, canonical_keys) for key in keys if key}


def clone_worker_capacity_snapshot() -> dict[str, Any]:
    requested_device = os.getenv("SEME2E_TTS_DEVICE") or os.getenv("SEME2E_API_DEVICE", "cpu")
    explicit_tts_device = bool(os.getenv("SEME2E_TTS_DEVICE", "").strip())
    coqui_gpu_keys = _worker_gpu_candidates(
        requested_device,
        "SEME2E_COQUI_TTS_CUDA_VISIBLE_DEVICES",
        explicit_device=explicit_tts_device,
    )
    cosyvoice_gpu_keys = _worker_gpu_candidates(
        requested_device,
        "SEME2E_COSYVOICE_CUDA_VISIBLE_DEVICES",
        explicit_device=explicit_tts_device,
    )
    try:
        gpt_sovits_gpu_keys = _gpt_sovits_gpu_candidates(requested_device)
    except RuntimeError:
        gpt_sovits_gpu_keys = ()
    asr_device = os.getenv("SEME2E_ASR_DEVICE") or os.getenv("SEME2E_API_DEVICE", "cpu")
    asr_gpu_keys = _worker_gpu_candidates(
        asr_device,
        "SEME2E_ASR_CUDA_VISIBLE_DEVICES",
        explicit_device=bool(os.getenv("SEME2E_ASR_DEVICE", "").strip()),
    )
    backend_limits = {
        "coquiTts": COQUI_TTS_WORKER_MAX_CONCURRENCY,
        "cosyVoice": COSYVOICE_WORKER_MAX_CONCURRENCY,
        "gptSoVits": GPT_SOVITS_WORKER_MAX_CONCURRENCY,
    }
    maximum = maximum_clone_worker_concurrency(
        coqui_limit=COQUI_TTS_WORKER_MAX_CONCURRENCY,
        cosyvoice_limit=COSYVOICE_WORKER_MAX_CONCURRENCY,
        gpt_sovits_limit=GPT_SOVITS_WORKER_MAX_CONCURRENCY,
        clone_gpu_limit=CLONE_GPU_MAX_CONCURRENCY,
        coqui_gpu_keys=coqui_gpu_keys,
        cosyvoice_gpu_keys=cosyvoice_gpu_keys,
        gpt_sovits_gpu_keys=gpt_sovits_gpu_keys,
    )
    combined_worker_limits = {
        "asr": ASR_WORKER_MAX_CONCURRENCY,
        "coquiTts": COQUI_TTS_WORKER_MAX_CONCURRENCY,
        "cosyVoice": COSYVOICE_WORKER_MAX_CONCURRENCY,
        "gptSoVits": GPT_SOVITS_WORKER_MAX_CONCURRENCY,
    }
    combined_gpu_keys = {
        "asr": asr_gpu_keys,
        "coquiTts": coqui_gpu_keys,
        "cosyVoice": cosyvoice_gpu_keys,
        "gptSoVits": gpt_sovits_gpu_keys,
    }
    unconstrained_cpu_workers = sum(
        combined_worker_limits[name]
        for name in ("asr", "coquiTts", "cosyVoice")
        if not combined_gpu_keys[name]
    )
    asr_clone_maximum = unconstrained_cpu_workers + maximum_gpu_worker_concurrency(
        worker_limits=combined_worker_limits,
        worker_gpu_keys=combined_gpu_keys,
        gpu_slot_limit=CLONE_GPU_MAX_CONCURRENCY,
    )
    api_device = os.getenv("SEME2E_API_DEVICE", "cpu")
    protect_gpu_keys = _clone_gpu_slot_keys(api_device, "SEME2E_PROTECT_CUDA_VISIBLE_DEVICES")
    configured_shared = _optional_env_bool("SEME2E_PROTECT_GPU_SHARED_WITH_WORKERS")
    worker_gpu_keys = tuple(
        dict.fromkeys(
            (
                *asr_gpu_keys,
                *coqui_gpu_keys,
                *cosyvoice_gpu_keys,
                *gpt_sovits_gpu_keys,
            )
        )
    )
    protect_shares_worker_gpu = (
        configured_shared
        if configured_shared is not None
        else bool(_gpu_key_set(protect_gpu_keys) & _gpu_key_set(worker_gpu_keys))
    )
    return {
        "maxConcurrency": maximum,
        "asrCloneMaxConcurrency": asr_clone_maximum,
        "protectSharesWorkerGpu": protect_shares_worker_gpu,
        "protectGpuKeys": list(protect_gpu_keys),
        "backendLimits": backend_limits,
        "gpuSlotLimit": CLONE_GPU_MAX_CONCURRENCY,
        "asrGpuKeys": list(asr_gpu_keys),
        "gpuKeys": {
            "coquiTts": list(coqui_gpu_keys),
            "cosyVoice": list(cosyvoice_gpu_keys),
            "gptSoVits": list(gpt_sovits_gpu_keys),
        },
    }


FORMAL_EPSILON = 4 / 255
FORMAL_STEPS = 200
FORMAL_WEIGHT_FEATURE = 150.0
FORMAL_WEIGHT_SEMANTIC = 300.0
FORMAL_WEIGHT_PSY = 0.001
FORMAL_WEIGHT_L2 = 0.1
FIXED_WEIGHT_STFT = 150.0
FIXED_WEIGHT_SNR = 20.0
FIXED_TARGET_SNR_DB = 25.0
FIXED_SELECTION_SNR_DB = 25.0
FIXED_STEP_SIZE = 0.00012
FIXED_INIT_NOISE = "zero"
FIXED_L2_REDUCTION = "rms"
FIXED_MIN_LR = 1.0e-6
FORMAL_PRESET_NAME = "lq25_large_balanced"
FORMAL_SEMANTIC_ENCODERS = ["S3", "HuBERT", "Whisper", "MFCC"]
FORMAL_TIMBRE_ENCODERS = ["VITS", "GPT-SoVITS", "MFCC", "WavLM", "CosyVoice"]
FORMAL_ASR_MODEL = "openai/whisper-small"
FORMAL_TTS_BACKEND = "xtts_v2"
MODEL_TYPES = {
    "tts": [
        {
            "value": "zero_shot",
            "name": "零样本克隆",
            "information": "只需短参考语音即可直接复刻目标声音，代表低门槛、即时式语音克隆风险。",
        },
        {
            "value": "fine_tuning",
            "name": "微调式克隆",
            "information": "收集多条目标语音并进一步训练模型，使其稳定学习目标声音，代表公开语音被长期收集后的训练式滥用风险。",
        },
        {
            "value": "llm_based",
            "name": "LLM 语音克隆",
            "information": "利用 Speech Tokenizer 与语言模型进行语音建模，代表当前语音 Token 化和大模型驱动的新型克隆链路。",
        },
    ],
    "asr": [
        {"value": "generative_asr", "name": "通用生成式 ASR", "information": "通过生成式解码得到识别文本的通用语音识别路线。"},
        {"value": "ctc_asr", "name": "CTC 语音识别", "information": "使用 CTC 对齐完成语音到文本映射的识别路线。"},
        {"value": "self_supervised_asr", "name": "自监督语音识别", "information": "基于自监督预训练语音表示构建的识别路线。"},
        {"value": "non_autoregressive_asr", "name": "非自回归 ASR", "information": "不依赖逐字自回归生成的高效语音识别路线。"},
        {"value": "chinese_asr", "name": "中文语音识别", "information": "针对中文语音识别场景训练或优化的模型。"},
    ],
    "semantic": [
        {"value": "speech_tokenizer", "name": "语音 Tokenizer", "information": "将连续语音转换为模型可处理的语音表示或离散 Token。"},
        {"value": "semantic_encoder", "name": "语义编码器", "information": "提取语音内容与发音相关的高层表示。"},
        {"value": "llm_frontend", "name": "语音大模型前端", "information": "位于原始语音与语音大模型之间的前端表示模块。"},
        {"value": "self_supervised_representation", "name": "自监督语音表示", "information": "从大规模无标注语音中学习的通用表示。"},
        {"value": "asr_encoder", "name": "ASR 编码器", "information": "语音识别模型在生成文本之前使用的编码前端。"},
        {"value": "acoustic_feature", "name": "声学特征", "information": "描述基础频谱、发音和音色结构的声学表示。"},
    ],
    "identity": [
        {"value": "tts_encoder", "name": "TTS 编码器", "information": "语音合成系统用于提取音色或声音条件的编码模块。"},
        {"value": "voice_identity_encoder", "name": "声音身份编码器", "information": "提取说话人身份与音色信息的编码模块。"},
        {"value": "clone_encoder", "name": "克隆系统编码器", "information": "语音克隆系统用于提取参考声音条件的编码模块。"},
        {"value": "fine_tuning_related", "name": "微调相关编码器", "information": "训练式语音克隆在数据适配或微调过程中使用的声音表示模块。"},
        {"value": "acoustic_feature", "name": "声学特征", "information": "描述基础频谱、发音和音色结构的声学表示。"},
        {"value": "self_supervised_representation", "name": "自监督语音表示", "information": "从大规模无标注语音中学习的通用表示。"},
        {"value": "speaker_encoder", "name": "说话人编码器", "information": "将语音转换为可比较的说话人身份表示。"},
        {"value": "speaker_verification", "name": "说话人验证", "information": "用于比较两段语音是否来自同一说话人的模型。"},
    ],
    "evaluation": [
        {"value": "speaker_verification", "name": "说话人验证", "information": "用于比较两段语音是否来自同一说话人的模型。"},
        {"value": "evaluation_model", "name": "独立评估模型", "information": "只用于结果评价，不参与 VoiceShield 扰动生成。"},
    ],
}
MODEL_METADATA = {
    "S3": ("S3 Tokenizer Encoder", ["speech_tokenizer", "semantic_encoder", "llm_frontend"], "CosyVoice S3 前端把连续语音编码为离散语音 Token，作为语义扰动优化目标之一。"),
    "HuBERT": ("HuBERT Large", ["self_supervised_representation", "semantic_encoder"], "HuBERT Large 自监督语音表示用于约束扰动对高层语音内容特征的影响。"),
    "Whisper": ("Whisper Large-v3 Encoder", ["asr_encoder", "semantic_encoder"], "Whisper Large-v3 编码前端提供面向语音识别的语义表示，不在此处直接输出转写文本。"),
    "MFCC": ("MFCC", ["acoustic_feature"], "MFCC 是轻量声学基线，用于描述语音的频谱包络；同一模型可参与语义和声音身份两条分支。"),
    "VITS": ("VITS Posterior Encoder", ["tts_encoder", "voice_identity_encoder"], "VITS 后验编码器提取合成模型使用的声学与音色条件，作为声音身份扰动目标。"),
    "GPT-SoVITS": ("GPT-SoVITS Encoder", ["clone_encoder", "fine_tuning_related", "voice_identity_encoder", "llm_frontend"], "GPT-SoVITS 编码组件代表训练式、语义 Token 驱动的语音克隆特征路径。"),
    "WavLM": ("WavLM", ["self_supervised_representation", "voice_identity_encoder"], "WavLM 自监督语音表示同时保留说话人和声学信息，用于声音身份防护。"),
    "CosyVoice": ("CosyVoice CAM++", ["speaker_encoder", "speaker_verification", "voice_identity_encoder"], "CosyVoice CAM++ 提取参考音频的说话人条件，参与声音身份扰动优化。"),
    "openai/whisper-small": ("Whisper Small", ["generative_asr"], "Whisper Small 是本项目英文演示的主 ASR，使用服务器本地权重执行真实转写。"),
    "openai-whisper:tiny": ("Whisper Tiny", ["generative_asr"], "Whisper Tiny 提供低资源生成式 ASR 基线，使用本地 OpenAI Whisper 检查点。"),
    "openai-whisper:base": ("Whisper Base", ["generative_asr"], "Whisper Base 提供较 Tiny 更强的本地生成式 ASR 对照。"),
    "openai-whisper:medium": ("Whisper Medium", ["generative_asr"], "Whisper Medium 是 VoiceShield 作品实验中的强 ASR 评估模型，用于验证保护效果面对更稳定识别能力时的迁移性。"),
    "facebook/wav2vec2-base-960h": ("Wav2Vec2 Base 960h", ["ctc_asr", "self_supervised_asr"], "Wav2Vec2 Base 960h 使用自监督预训练与 CTC 解码，作为不同于 Whisper 的真实 ASR 路线。"),
    "funasr:paraformer-zh": ("FunASR Paraformer 中文", ["non_autoregressive_asr", "chinese_asr"], "Paraformer 是面向中文的非自回归 ASR，用于补充中文识别路线。"),
    "speechbrain/spkrec-ecapa-voxceleb": ("ECAPA-TDNN", ["speaker_verification", "evaluation_model"], "ECAPA-TDNN 只用于独立计算说话人相似度和克隆防护效果，不参与 VoiceShield 扰动优化。"),
}
SUPPORTED_TTS_MODELS = [
    {
        "label": "XTTS-v2",
        "name": "XTTS-v2",
        "value": "XTTS-v2",
        "type": ["zero_shot"],
        "information": "仅需短参考语音和目标文本即可生成相似声音，是 VoiceShield 用于验证低门槛零样本克隆风险的主要后端。",
        "backendValue": "xtts_v2",
        "cacheName": "tts_models--multilingual--multi-dataset--xtts_v2",
        "aliases": ["default", "xtts", "xtts-v2", "xtts_v2", "coquitts:xtts_v2"],
        "languages": ["en", "zh-cn"],
        "description": "Coqui XTTS-v2 voice cloning backend.",
        "backend": "CoquiTTS",
        "requiresReferenceText": False,
    },
    {
        "label": "XTTS-v1.1",
        "name": "XTTS-v1.1",
        "value": "XTTS-v1.1",
        "type": ["zero_shot"],
        "information": "XTTS 的早期跨语言零样本克隆版本，用于验证保护效果能否迁移到同系列的不同模型版本。",
        "backendValue": "tts_models/multilingual/multi-dataset/xtts_v1.1",
        "cacheName": "tts_models--multilingual--multi-dataset--xtts_v1.1",
        "aliases": ["xtts-v1.1", "xtts_v1.1", "xtts-v1", "xtts_v1", "coquitts:xtts_v1.1"],
        "languages": ["en", "zh-cn"],
        "description": "Coqui XTTS-v1.1 cross-language voice cloning backend.",
        "backend": "CoquiTTS",
        "requiresReferenceText": False,
        "frontendVisible": False,
    },
    {
        "label": "YourTTS",
        "name": "YourTTS",
        "value": "YourTTS",
        "type": ["zero_shot"],
        "information": "面向少量参考语音的跨说话人合成模型，用于补充验证传统零样本语音克隆路径。",
        "backendValue": "tts_models/multilingual/multi-dataset/your_tts",
        "cacheName": "tts_models--multilingual--multi-dataset--your_tts",
        "aliases": ["your-tts", "your_tts", "coquitts:your_tts"],
        "languages": ["en"],
        "description": "Coqui YourTTS voice cloning backend.",
        "backend": "CoquiTTS",
        "requiresReferenceText": False,
    },
    {
        "label": "CosyVoice2-0.5B",
        "name": "CosyVoice2-0.5B",
        "value": "CosyVoice2-0.5B",
        "type": ["zero_shot", "llm_based"],
        "information": "使用 Qwen2.5 驱动的 0.5B 语音 Token 语言模型进行零样本克隆；在 VoiceShield 中作为 LLM 语音克隆代表模型，原始和保护参考音频由同一次模型加载连续评测。",
        "backendValue": "cosyvoice2:0.5b",
        "aliases": ["cosyvoice2", "cosyvoice2-0.5b", "cosyvoice2:0.5b"],
        "languages": ["en", "zh-cn"],
        "description": "Official FunAudioLLM CosyVoice2 0.5B zero-shot voice cloning backend.",
        "backend": "CosyVoice2",
        "requiresReferenceText": True,
        "promptRequired": True,
    },
    {
        "label": "GPT-SoVITS 微调链路",
        "name": "GPT-SoVITS",
        "value": "GPT-SoVITS",
        "type": ["fine_tuning", "llm_based"],
        "information": "结合语义 GPT 与 SoVITS 声学生成器的训练式克隆链路。每次使用当前保护任务的原始音频和保护音频分别现场微调，再生成两侧克隆语音进行对比。训练时长由上传音频自动确定，过长音频会在训练上限处截取。",
        "backendValue": "gpt-sovits:finetune",
        "aliases": ["gpt-sovits", "gpt-sovits:finetune"],
        "languages": ["en", "zh-cn"],
        "description": "GPT-SoVITS live per-upload fine-tuning evaluation chain.",
        "backend": "GPT-SoVITS",
        "online": True,
        "requiresReferenceText": True,
        "promptRequired": True,
        "fineTuneMode": "live_fine_tune",
    },
]

COSYVOICE_MODEL_DIR = Path(os.getenv("SEME2E_COSYVOICE_MODEL_DIR", ROOT / "checkpoints" / "CosyVoice2-0.5B"))
COSYVOICE_REPO_DIR = Path(os.getenv("SEME2E_COSYVOICE_REPO_DIR", ROOT.parents[1] / ".runtime" / "cosyvoice" / "CosyVoice"))
COSYVOICE_PYTHON = Path(os.getenv("SEME2E_COSYVOICE_PYTHON", ROOT.parents[1] / ".runtime" / "cosyvoice" / ".venv" / "bin" / "python"))
COSYVOICE_REQUIRED_FILES = (
    "cosyvoice2.yaml",
    "llm.pt",
    "flow.pt",
    "hift.pt",
    "campplus.onnx",
    "speech_tokenizer_v2.onnx",
)
COSYVOICE_READY_MARKER = COSYVOICE_MODEL_DIR / ".voiceshield-ready.json"
GPT_SOVITS_RUNTIME_DIR = Path(os.getenv("SEME2E_GPT_SOVITS_RUNTIME_DIR", ROOT.parents[1] / ".runtime" / "gpt-sovits"))
GPT_SOVITS_REPO_DIR = Path(os.getenv("SEME2E_GPT_SOVITS_REPO_DIR", GPT_SOVITS_RUNTIME_DIR / "GPT-SoVITS"))
GPT_SOVITS_PYTHON = Path(os.getenv("SEME2E_GPT_SOVITS_PYTHON", GPT_SOVITS_RUNTIME_DIR / ".venv" / "bin" / "python"))
GPT_SOVITS_CNHUBERT = Path(
    os.getenv(
        "SEME2E_GPT_SOVITS_CNHUBERT",
        ROOT / "checkpoints" / "GSV" / "base_models" / "chinese-hubert-base",
    )
)
GPT_SOVITS_BERT = Path(
    os.getenv(
        "SEME2E_GPT_SOVITS_BERT",
        ROOT / "checkpoints" / "GSV" / "base_models" / "chinese-roberta-wwm-ext-large",
    )
)
GPT_SOVITS_PRETRAINED_DIR = Path(
    os.getenv(
        "SEME2E_GPT_SOVITS_PRETRAINED_DIR",
        ROOT / "checkpoints" / "GSV" / "base_models" / "gsv-v2final-pretrained",
    )
)
GPT_SOVITS_PRETRAINED_S1 = GPT_SOVITS_PRETRAINED_DIR / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
GPT_SOVITS_PRETRAINED_S2G = GPT_SOVITS_PRETRAINED_DIR / "s2G2333k.pth"
GPT_SOVITS_PRETRAINED_S2D = GPT_SOVITS_PRETRAINED_DIR / "s2D2333k.pth"


def _tts_cache_dir() -> Path:
    return Path(os.getenv("TTS_HOME", str(PROJECT_TTS_CACHE_DIR)))


def _hf_snapshot_path(repo_id: str, project_path: Path | None = None) -> tuple[Path | None, str | None]:
    candidates = [project_path] if project_path is not None else []
    for candidate in candidates:
        if candidate is not None and _model_directory_ready(candidate):
            return candidate, None
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=repo_id, local_files_only=True)), None
    except Exception as exc:
        return None, f"local Hugging Face snapshot unavailable: {repo_id} ({type(exc).__name__}: {exc})"


def _model_directory_ready(path: Path | None) -> bool:
    if path is None or not path.is_dir() or not (path / "config.json").is_file():
        return False
    weight_patterns = ("*.safetensors", "*.bin", "*.pt", "*.pth")
    return any(next(path.glob(pattern), None) is not None for pattern in weight_patterns)


def _tts_model_cache_status(cache_name: str) -> tuple[str, str | None, str]:
    path = _tts_cache_dir() / cache_name
    config_path = path / "config.json"
    checkpoint_path = path / "model.pth"
    if not checkpoint_path.exists():
        checkpoint_path = path / "model_file.pth"
    if not config_path.exists() or not checkpoint_path.exists():
        return "download_required", f"missing local Coqui TTS cache: {path}", str(path)
    ready, reason = _torch_checkpoint_ready(checkpoint_path)
    if not ready:
        return "unavailable", reason, str(path)
    return "available", None, str(path)


def _cosyvoice_model_status() -> tuple[str, str | None, str]:
    missing = [name for name in COSYVOICE_REQUIRED_FILES if not (COSYVOICE_MODEL_DIR / name).is_file()]
    if not (COSYVOICE_MODEL_DIR / "CosyVoice-BlankEN").is_dir():
        missing.append("CosyVoice-BlankEN/")
    if missing:
        return "download_required", "incomplete local CosyVoice2 model: " + ", ".join(missing), str(COSYVOICE_MODEL_DIR)
    runtime_missing = []
    if not COSYVOICE_PYTHON.is_file():
        runtime_missing.append(str(COSYVOICE_PYTHON))
    if not (COSYVOICE_REPO_DIR / "cosyvoice" / "cli" / "cosyvoice.py").is_file():
        runtime_missing.append(str(COSYVOICE_REPO_DIR))
    if runtime_missing:
        return "unavailable", "missing isolated CosyVoice2 runtime: " + ", ".join(runtime_missing), str(COSYVOICE_MODEL_DIR)
    if not COSYVOICE_READY_MARKER.is_file():
        return "unavailable", "CosyVoice2 snapshot is complete, but the server runtime has not passed the VoiceShield generation benchmark", str(COSYVOICE_MODEL_DIR)
    return "available", None, str(COSYVOICE_MODEL_DIR)


def _gpt_sovits_model_status() -> tuple[str, str | None, str]:
    required_paths = [
        GPT_SOVITS_PYTHON,
        GPT_SOVITS_REPO_DIR / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py",
        GPT_SOVITS_CNHUBERT,
        GPT_SOVITS_BERT,
        ROOT / "gpt_sovits_worker.py",
        ROOT / "gpt_sovits_live_finetune.py",
        GPT_SOVITS_PRETRAINED_S1,
        GPT_SOVITS_PRETRAINED_S2G,
        GPT_SOVITS_PRETRAINED_S2D,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        return "unavailable", "missing isolated GPT-SoVITS training runtime: " + ", ".join(missing), str(GPT_SOVITS_RUNTIME_DIR)
    return "available", None, str(GPT_SOVITS_RUNTIME_DIR)


def _tts_catalog_status(item: dict[str, Any], *, coqui_available: bool) -> tuple[str, str | None, str | None]:
    backend = str(item.get("backend") or "CoquiTTS")
    if backend == "CosyVoice2":
        return _cosyvoice_model_status()
    if backend == "GPT-SoVITS":
        return _gpt_sovits_model_status()
    cache_status, cache_reason, cache_path = _tts_model_cache_status(str(item["cacheName"]))
    if not coqui_available:
        return "unavailable", "Coqui TTS package is not installed", cache_path
    return cache_status, cache_reason, cache_path


def _torch_checkpoint_ready(path: Path) -> tuple[bool, str | None]:
    try:
        if not path.is_file() or path.stat().st_size < 1024 * 1024:
            return False, f"incomplete checkpoint: {path}"
        with path.open("rb") as checkpoint_file:
            signature = checkpoint_file.read(4)
        if signature.startswith(b"PK"):
            with zipfile.ZipFile(path) as checkpoint_zip:
                if not checkpoint_zip.namelist():
                    return False, f"empty checkpoint archive: {path}"
    except (OSError, zipfile.BadZipFile) as exc:
        return False, f"invalid checkpoint archive: {path} ({type(exc).__name__}: {exc})"
    return True, None


def _portable_tts_config(model_dir: Path, config_path: Path) -> Path:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return config_path

    changed = False

    def rewrite(value: Any) -> Any:
        nonlocal changed
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, str):
            normalized = value.replace("\\", "/")
            if normalized.startswith("/") or ":/" in normalized:
                candidate = model_dir / Path(normalized).name
                if candidate.exists():
                    changed = True
                    return str(candidate)
        return value

    rewritten = rewrite(config)
    if not changed:
        return config_path

    local_config_path = model_dir / "config.local.json"
    local_config_path.write_text(json.dumps(rewritten, ensure_ascii=False, indent=2), encoding="utf-8")
    return local_config_path


def _supported_tts_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for item in SUPPORTED_TTS_MODELS:
        backend_value = str(item["backendValue"])
        for value in [item["label"], item["value"], backend_value, *item.get("aliases", [])]:
            aliases[str(value).lower()] = backend_value
    return aliases


def supported_tts_languages(model: str | None) -> list[str]:
    backend_value = normalize_tts_model(model)
    for item in SUPPORTED_TTS_MODELS:
        if str(item["backendValue"]).lower() == backend_value.lower():
            return [str(language) for language in item.get("languages", [])]
    return []


def tts_model_requires_reference_text(model: str | None) -> bool:
    backend_value = normalize_tts_model(model)
    for item in SUPPORTED_TTS_MODELS:
        if str(item["backendValue"]).lower() == backend_value.lower():
            if "requiresReferenceText" in item:
                return bool(item.get("requiresReferenceText"))
            return bool(item.get("promptRequired"))
    return False


def tts_model_requires_prompt(model: str | None) -> bool:
    """Compatibility alias for older API consumers."""
    return tts_model_requires_reference_text(model)


def _checkpoint_status() -> dict[str, Any]:
    whisper_cache_dir = Path(os.getenv("WHISPER_CACHE_DIR", Path.home() / ".cache" / "whisper"))
    required = {
        "VITS": ROOT / "checkpoints" / "VITS" / "pretrained_ljs.pth",
        "GPT-SoVITS": ROOT / "checkpoints" / "GSV" / "base_models" / "gsv-v2final-pretrained" / "s2G2333k.pth",
        "CosyVoiceTokenizer": ROOT / "checkpoints" / "CosyVoice" / "speech_tokenizer_v1.onnx",
        "CosyVoiceCAMPP": ROOT / "checkpoints" / "CosyVoice" / "base_models" / "CosyVoice-300M" / "campplus.onnx",
        "WavLM": ROOT / "checkpoints" / "wavlm" / "pytorch_model.bin",
        "ESpeakNG": ROOT / "vendor" / "espeak-ng" / "libespeak-ng.dll",
        "ESpeakNGData": ROOT / "vendor" / "espeak-ng" / "espeak-ng-data",
        "ASRWhisperSmall": ROOT / "checkpoints" / "asr" / "openai-whisper-small" / "config.json",
        "ASRWav2Vec2Base960h": ROOT / "checkpoints" / "hf" / "facebook" / "wav2vec2-base-960h" / "config.json",
        "ASRWhisperTiny": whisper_cache_dir / "tiny.pt",
        "ASRWhisperBase": whisper_cache_dir / "base.pt",
        "ASRWhisperMedium": whisper_cache_dir / "medium.pt",
        "ASRParaformerZh": ROOT / "checkpoints" / "modelscope" / "damo" / "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" / "model.pt",
        "ECAPA": ROOT / "checkpoints" / "ecapa" / "embedding_model.ckpt",
        "CosyVoice2LLM": COSYVOICE_MODEL_DIR / "cosyvoice2.yaml",
    }
    entries = {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "status": "available" if path.exists() else "unavailable",
            "reason": None if path.exists() else f"missing path: {path}",
        }
        for name, path in required.items()
    }
    hf_models = {
        "HuBERTLargeLL60K": (
            "facebook/hubert-large-ll60k",
            ROOT / "checkpoints" / "hf" / "facebook" / "hubert-large-ll60k",
        ),
        "WhisperLargeV3": (
            "openai/whisper-large-v3",
            ROOT / "checkpoints" / "hf" / "openai" / "whisper-large-v3",
        ),
    }
    for name, (repo_id, project_path) in hf_models.items():
        found_path, reason = _hf_snapshot_path(repo_id, project_path)
        ready = _model_directory_ready(found_path)
        entries[name] = {
            "path": str(project_path),
            "foundPath": str(found_path) if found_path is not None else None,
            "exists": ready,
            "status": "available" if ready else "unavailable",
            "reason": None if ready else reason or f"incomplete model directory: {found_path}",
            "repoId": repo_id,
        }
    missing = [name for name, item in entries.items() if not item["exists"]]
    return {
        "missing": missing,
        "required": {name: item.get("path") for name, item in entries.items()},
        "entries": entries,
    }


def _component_available(checkpoints: dict[str, Any], names: list[str]) -> tuple[str, str | None]:
    missing = [name for name in names if name in checkpoints["missing"]]
    if missing:
        return "unavailable", "missing files: " + ", ".join(missing)
    return "available", None


def _model_option(label: str, value: str, backend_value: str, branch: str, *, status: str = "available", reason: str | None = None, **extra: Any) -> dict[str, Any]:
    metadata = MODEL_METADATA.get(value)
    metadata_name = metadata[0] if metadata else label
    metadata_types = list(metadata[1]) if metadata else []
    metadata_information = metadata[2] if metadata else str(extra.get("description") or f"{label} 由后端运行时能力清单提供。")
    return {
        "label": label,
        "name": extra.pop("name", metadata_name),
        "value": value,
        "backendValue": backend_value,
        "branch": branch,
        "type": extra.pop("type", metadata_types),
        "information": extra.pop("information", metadata_information),
        "status": status,
        "reason": reason,
        **extra,
    }


def _profile_defaults(profile: str, *, steps: int, semantic_encoders: list[str], timbre_encoders: list[str]) -> dict[str, Any]:
    return {
        "profile": profile,
        "presetName": FORMAL_PRESET_NAME,
        "realProtect": True,
        "mode": "standard",
        "targets": ["semantic", "timbre"],
        "semantic": {
            "enabled": True,
            "asrModel": FORMAL_ASR_MODEL,
            "asrModels": [FORMAL_ASR_MODEL],
            "encoders": semantic_encoders,
            "tokenizerPath": "checkpoints/CosyVoice/speech_tokenizer_v1.onnx",
            "hubertPath": "facebook/hubert-large-ll60k",
            "whisperPath": "openai/whisper-large-v3",
            "weightSemantic": FORMAL_WEIGHT_SEMANTIC,
        },
        "timbre": {
            "enabled": True,
            "mode": "untargeted",
            "encoders": timbre_encoders,
            "weightIdentity": FORMAL_WEIGHT_FEATURE,
            "weightFeature": FORMAL_WEIGHT_FEATURE,
        },
        "psychoacoustic": {
            "enabled": True,
            "weightPsy": FORMAL_WEIGHT_PSY,
        },
        "optimization": {
            "epsilon": round(FORMAL_EPSILON, 9),
            "steps": steps,
            "weightL2": FORMAL_WEIGHT_L2,
        },
    }


def runtime_config() -> dict[str, Any]:
    audio_preprocessing = audio_preprocess_capabilities()
    checkpoints = _checkpoint_status()
    vits_status, vits_reason = _component_available(checkpoints, ["VITS"])
    gsv_status, gsv_reason = _component_available(checkpoints, ["GPT-SoVITS"])
    s3_status, s3_reason = _component_available(checkpoints, ["CosyVoiceTokenizer"])
    cosy_status, cosy_reason = _component_available(checkpoints, ["CosyVoiceCAMPP"])
    transformers_available = _module_available("transformers")
    whisper_small_available = transformers_available and bool((checkpoints.get("entries") or {}).get("ASRWhisperSmall", {}).get("exists"))
    whisper_small_reason = None if whisper_small_available else "missing local Whisper Small checkpoint" if transformers_available else "transformers not installed"
    wav2vec_available = transformers_available and bool((checkpoints.get("entries") or {}).get("ASRWav2Vec2Base960h", {}).get("exists"))
    wav2vec_reason = None if wav2vec_available else "missing local Wav2Vec2 Base 960h checkpoint" if transformers_available else "transformers not installed"
    whisper_package_available = _module_available("whisper")
    whisper_tiny_available = whisper_package_available and bool((checkpoints.get("entries") or {}).get("ASRWhisperTiny", {}).get("exists"))
    whisper_base_available = whisper_package_available and bool((checkpoints.get("entries") or {}).get("ASRWhisperBase", {}).get("exists"))
    whisper_medium_available = whisper_package_available and bool((checkpoints.get("entries") or {}).get("ASRWhisperMedium", {}).get("exists"))
    funasr_available = _module_available("funasr")
    paraformer_available = funasr_available and bool((checkpoints.get("entries") or {}).get("ASRParaformerZh", {}).get("exists"))
    tts_available = _module_available("TTS")

    semantic_options = [
        _model_option("S3 Tokenizer Encoder", "S3", "s3", "semantic", status=s3_status, reason=s3_reason),
        _model_option("HuBERT Large", "HuBERT", "hubert", "semantic", status="available" if transformers_available and "HuBERTLargeLL60K" not in checkpoints["missing"] else "unavailable", reason=None if transformers_available and "HuBERTLargeLL60K" not in checkpoints["missing"] else (checkpoints["entries"]["HuBERTLargeLL60K"].get("reason") if transformers_available else "transformers not installed"), defaultPath="facebook/hubert-large-ll60k"),
        _model_option("Whisper Large-v3 Encoder", "Whisper", "whisper", "semantic", status="available" if transformers_available and "WhisperLargeV3" not in checkpoints["missing"] else "unavailable", reason=None if transformers_available and "WhisperLargeV3" not in checkpoints["missing"] else (checkpoints["entries"]["WhisperLargeV3"].get("reason") if transformers_available else "transformers not installed"), defaultPath="openai/whisper-large-v3"),
        _model_option("MFCC", "MFCC", "mfcc", "semantic"),
    ]
    timbre_options = [
        _model_option("VITS Posterior Encoder", "VITS", "vits", "timbre", status=vits_status, reason=vits_reason),
        _model_option("GPT-SoVITS Encoder", "GPT-SoVITS", "gsv", "timbre", status=gsv_status, reason=gsv_reason),
        _model_option("MFCC", "MFCC", "mfcc", "timbre"),
        _model_option("WavLM", "WavLM", "wavlm", "timbre", status="available" if transformers_available else "unavailable", reason=None if transformers_available else "transformers not installed"),
        _model_option("CosyVoice CAM++", "CosyVoice", "cosyvoice", "timbre", status=cosy_status, reason=cosy_reason),
    ]
    asr_options = [
        _model_option("Whisper Small", "openai/whisper-small", str(ROOT / "checkpoints" / "asr" / "openai-whisper-small"), "asr", status="available" if whisper_small_available else "unavailable", reason=whisper_small_reason, backend="transformers", localPath=str(ROOT / "checkpoints" / "asr" / "openai-whisper-small")),
        _model_option("OpenAI Whisper Tiny", "openai-whisper:tiny", "openai-whisper:tiny", "asr", status="available" if whisper_tiny_available else "unavailable", reason=None if whisper_tiny_available else "missing local Whisper Tiny checkpoint" if whisper_package_available else "openai-whisper package not installed", backend="openai-whisper"),
        _model_option("OpenAI Whisper Base", "openai-whisper:base", "openai-whisper:base", "asr", status="available" if whisper_base_available else "unavailable", reason=None if whisper_base_available else "missing local Whisper Base checkpoint" if whisper_package_available else "openai-whisper package not installed", backend="openai-whisper"),
        _model_option("OpenAI Whisper Medium", "openai-whisper:medium", "openai-whisper:medium", "asr", status="available" if whisper_medium_available else "unavailable", reason=None if whisper_medium_available else "missing local Whisper Medium checkpoint" if whisper_package_available else "openai-whisper package not installed", backend="openai-whisper", localPath=str(Path.home() / ".cache" / "whisper" / "medium.pt")),
        _model_option("Wav2Vec2 Base 960h", "facebook/wav2vec2-base-960h", str(ROOT / "checkpoints" / "hf" / "facebook" / "wav2vec2-base-960h"), "asr", status="available" if wav2vec_available else "unavailable", reason=wav2vec_reason, backend="transformers", localPath=str(ROOT / "checkpoints" / "hf" / "facebook" / "wav2vec2-base-960h")),
        _model_option("FunASR Paraformer", "funasr:paraformer-zh", "funasr:paraformer-zh", "asr", status="available" if paraformer_available else "unavailable", reason=None if paraformer_available else "missing local Paraformer checkpoint" if funasr_available else "funasr not installed", backend="funasr"),
    ]
    tts_options = []
    for item in SUPPORTED_TTS_MODELS:
        if item.get("frontendVisible") is False:
            continue
        status, reason, cache_path = _tts_catalog_status(item, coqui_available=tts_available)
        tts_options.append(
            _model_option(
                str(item["label"]),
                str(item["value"]),
                str(item["backendValue"]),
                "tts",
                status=status,
                reason=reason,
                name=str(item["name"]),
                type=list(item["type"]),
                information=str(item["information"]),
                backend=str(item.get("backend") or "CoquiTTS"),
                localPath=cache_path,
                languages=item.get("languages", []),
                description=item.get("description"),
                requiresReferenceText=tts_model_requires_reference_text(str(item["backendValue"])),
                promptRequired=tts_model_requires_reference_text(str(item["backendValue"])),
                annotationSources=["manual", "asr"] if tts_model_requires_reference_text(str(item["backendValue"])) else [],
                fineTuneMode=item.get("fineTuneMode"),
            )
        )
    configured_tts_backend = normalize_tts_model(os.getenv("SEME2E_API_DEFAULT_TTS_MODEL"))
    configured_tts_option = next(
        (
            option
            for option in tts_options
            if str(option.get("backendValue") or "").lower() == configured_tts_backend.lower()
        ),
        None,
    )
    default_tts_option = (
        configured_tts_option
        if configured_tts_option is not None and configured_tts_option.get("status") == "available"
        else next((option for option in tts_options if option.get("status") == "available"), None)
        or (tts_options[0] if tts_options else None)
    )
    default_tts_backend = (
        str(default_tts_option["backendValue"])
        if default_tts_option is not None
        else None
    )
    evaluation_options = [
        _model_option(
            "ECAPA-TDNN",
            "speechbrain/spkrec-ecapa-voxceleb",
            str(ROOT / "checkpoints" / "ecapa"),
            "evaluation",
            status="available" if _module_available("speechbrain") and "ECAPA" not in checkpoints["missing"] else "unavailable",
            reason=None if _module_available("speechbrain") and "ECAPA" not in checkpoints["missing"] else "SpeechBrain or local ECAPA checkpoint is unavailable",
            backend="SpeechBrain",
            localPath=str(ROOT / "checkpoints" / "ecapa"),
        )
    ]

    formal = _profile_defaults("formal", steps=FORMAL_STEPS, semantic_encoders=FORMAL_SEMANTIC_ENCODERS, timbre_encoders=FORMAL_TIMBRE_ENCODERS)
    fields = {
        "epsilon": {"label": "扰动强度 ε", "path": "optimization.epsilon", "default": round(FORMAL_EPSILON, 9), "min": 0.001, "max": 0.08, "step": 0.001, "unit": "waveform amplitude", "description": "高保真默认值为 4/255。"},
        "steps": {"label": "优化步数", "path": "optimization.steps", "default": FORMAL_STEPS, "min": 1, "max": 500, "step": 1, "description": "默认 200，最大 500。"},
        "weightIdentity": {"label": "Identity 权重", "path": "timbre.weightIdentity", "default": FORMAL_WEIGHT_FEATURE, "min": 0, "max": 1000, "step": 1},
        "weightFeature": {"label": "Identity 权重（legacy）", "path": "timbre.weightFeature", "default": FORMAL_WEIGHT_FEATURE, "min": 0, "max": 1000, "step": 1},
        "weightSemantic": {"label": "Semantic 权重", "path": "semantic.weightSemantic", "default": FORMAL_WEIGHT_SEMANTIC, "min": 0, "max": 500, "step": 1},
        "weightPsy": {"label": "心理声学权重", "path": "psychoacoustic.weightPsy", "default": FORMAL_WEIGHT_PSY, "min": 0, "max": 0.01, "step": 0.000001},
        "weightL2": {"label": "L2 权重", "path": "optimization.weightL2", "default": FORMAL_WEIGHT_L2, "min": 0, "max": 1, "step": 0.01},
    }
    ranges = {
        key: {"min": value["min"], "max": value["max"], "step": value["step"]}
        for key, value in fields.items()
    }
    mode_presets = {
        "standard": {
            "profile": "formal",
            "semantic": {"weightSemantic": FORMAL_WEIGHT_SEMANTIC},
            "timbre": {"weightIdentity": FORMAL_WEIGHT_FEATURE, "weightFeature": FORMAL_WEIGHT_FEATURE},
            "psychoacoustic": {"weightPsy": FORMAL_WEIGHT_PSY},
            "optimization": {"epsilon": round(FORMAL_EPSILON, 9), "steps": FORMAL_STEPS, "weightL2": FORMAL_WEIGHT_L2},
        },
        "strong": {
            "profile": "formal",
            "semantic": {"weightSemantic": 140.0},
            "timbre": {"weightIdentity": 675.0, "weightFeature": 675.0},
            "psychoacoustic": {"weightPsy": 0.0000075},
            "optimization": {"epsilon": round(FORMAL_EPSILON, 9), "steps": FORMAL_STEPS, "weightL2": 0.08},
        },
        "high_fidelity": {
            "profile": "formal",
            "semantic": {"weightSemantic": 80.0},
            "timbre": {"weightIdentity": 400.0, "weightFeature": 400.0},
            "psychoacoustic": {"weightPsy": 0.000015},
            "optimization": {"epsilon": 0.020392156, "steps": FORMAL_STEPS, "weightL2": 0.15},
        },
        "custom": {
            "profile": "formal",
            "semantic": {"weightSemantic": FORMAL_WEIGHT_SEMANTIC},
            "timbre": {"weightIdentity": FORMAL_WEIGHT_FEATURE, "weightFeature": FORMAL_WEIGHT_FEATURE},
            "psychoacoustic": {"weightPsy": FORMAL_WEIGHT_PSY},
            "optimization": {"epsilon": round(FORMAL_EPSILON, 9), "steps": FORMAL_STEPS, "weightL2": FORMAL_WEIGHT_L2},
        },
    }
    active_profile = os.getenv("SEME2E_API_DEFAULT_PROFILE", "formal")
    if active_profile != "formal":
        active_profile = "formal"
    form_schema = {
        "defaults": {"formal": formal},
        "activeDefaultProfile": active_profile,
        "profiles": [
            {"value": "formal", "label": "正式保护", "description": "使用 lq25_large_balanced 高保真默认参数，steps=200。"},
        ],
        "fields": fields,
        "modelOptions": {
            "semanticEncoders": semantic_options,
            "timbreEncoders": timbre_options,
            "asrModels": asr_options,
            "ttsModels": tts_options,
        },
    }
    return {
        "modelTypes": MODEL_TYPES,
        "defaults": formal,
        "profiles": {"formal": formal},
        "activeDefaultProfile": form_schema["activeDefaultProfile"],
        "ranges": ranges,
        "models": {
            "semantic": semantic_options,
            "asr": asr_options,
            "feature": timbre_options,
            "timbre": timbre_options,
            "tts": tts_options,
            "evaluation": evaluation_options,
        },
        "formSchema": form_schema,
        "constraints": {
            "maxAudioSizeBytes": _env_int("SEME2E_API_MAX_AUDIO_SIZE_BYTES", 200 * 1024 * 1024),
            "audioPreprocessing": audio_preprocessing,
        },
        "fixedOptimization": {
            "weight_stft": FIXED_WEIGHT_STFT,
            "weight_snr": FIXED_WEIGHT_SNR,
            "target_snr_db": FIXED_TARGET_SNR_DB,
            "selection_snr_db": FIXED_SELECTION_SNR_DB,
            "step_size": FIXED_STEP_SIZE,
            "init_noise": FIXED_INIT_NOISE,
            "l2_reduction": FIXED_L2_REDUCTION,
            "min_lr": FIXED_MIN_LR,
            "readOnly": True,
        },
        "clone": {
            "defaults": {
                "model": default_tts_backend,
                "backendValue": default_tts_backend,
                "language": os.getenv("SEME2E_API_DEFAULT_TTS_LANGUAGE", "en"),
                "uiPreferredLanguage": os.getenv("SEME2E_API_UI_TTS_LANGUAGE", "zh-cn"),
                "speed": _env_float("SEME2E_API_DEFAULT_TTS_SPEED", 1.0),
            },
            "languages": _env_list("SEME2E_API_TTS_LANGUAGES", ["en", "zh-cn"]),
            "speeds": [float(value) for value in _env_list("SEME2E_API_TTS_SPEEDS", ["0.75", "1", "1.25"])],
        },
        "modes": [
            {"value": "standard", "label": "标准保护", "description": "正式默认", "targetPolicy": "selectable"},
            {"value": "strong", "label": "强保护", "description": "强化防护", "targetPolicy": "selectable"},
            {"value": "high_fidelity", "label": "高保真", "description": "听感优先", "targetPolicy": "selectable"},
            {"value": "custom", "label": "自定义", "description": "手动联合", "targetPolicy": "joint_only"},
        ],
        "targets": [
            {"value": "semantic", "label": "语义防护", "description": "干扰识别"},
            {"value": "timbre", "label": "声音身份防护", "description": "阻断特征"},
            {"value": "joint", "label": "联合防护", "description": "双重防护"},
        ],
        "modePresets": mode_presets,
    }


def diagnose_capabilities() -> dict[str, Any]:
    device = os.getenv("SEME2E_API_DEVICE", "cpu")
    audio_preprocessing = audio_preprocess_capabilities()
    checkpoints = _checkpoint_status()
    missing = checkpoints["missing"]
    protect_required = [
        "VITS",
        "GPT-SoVITS",
        "CosyVoiceTokenizer",
        "CosyVoiceCAMPP",
        "WavLM",
        "HuBERTLargeLL60K",
        "WhisperLargeV3",
    ]
    protect_missing = [name for name in missing if name in set(protect_required)]
    whisper_available = _module_available("whisper") or _module_available("transformers")
    speaker_available = _module_available("speechbrain")
    pesq_available = _module_available("pesq")
    stoi_available = _module_available("pystoi")
    dnsmos_status = dnsmos_model_status()
    dnsmos_available = dnsmos_status.get("status") == "available"
    tts_available = _module_available("TTS")
    cosyvoice_status, cosyvoice_reason, _ = _cosyvoice_model_status()
    gpt_sovits_status, gpt_sovits_reason, _ = _gpt_sovits_model_status()
    perception_available = ["snr", "maskingCurve"] + (["pesq"] if pesq_available else []) + (["stoi"] if stoi_available else []) + (["dnsMos"] if dnsmos_available else [])
    perception_unavailable = ([] if pesq_available else ["pesq"]) + ([] if stoi_available else ["stoi"]) + ([] if dnsmos_available else ["dnsMos"])
    return {
        "ok": True,
        "modelTypes": MODEL_TYPES,
        "device": device,
        "python": sys.executable,
        "cwd": os.getcwd(),
        "checkpoints": checkpoints,
        "config": runtime_config(),
        "chains": {
            "audio_preprocessing": {
                "status": audio_preprocessing["status"],
                "recordingSupported": audio_preprocessing["recordingSupported"],
                "reason": audio_preprocessing["reason"],
                "decoder": audio_preprocessing["decoder"],
                "output": audio_preprocessing["output"],
            },
            "protect_generation": {
                "status": "available" if not protect_missing else "unavailable",
                "reason": None if not protect_missing else f"missing checkpoints: {', '.join(protect_missing)}",
            },
            "semantic_tokenizer_eval": {
                "status": "available" if "CosyVoiceTokenizer" not in missing else "unavailable",
                "reason": None if "CosyVoiceTokenizer" not in missing else "missing CosyVoiceTokenizer checkpoint file",
            },
            "asr_eval": {
                "status": "available" if whisper_available else "unavailable",
                "reason": None if whisper_available else "whisper/transformers ASR backend not installed",
            },
            "speaker_eval": {
                "status": "available" if speaker_available else "unavailable",
                "reason": None if speaker_available else "speaker model dependency speechbrain not installed",
            },
            "perception_eval": {
                "status": "available" if not perception_unavailable else "partial",
                "available": perception_available,
                "unavailable": perception_unavailable,
                "qualityModel": dnsmos_status,
                "reason": None if not perception_unavailable else "部分语音质量指标尚未生成",
            },
            "downstream_tts_eval": {
                "status": "available" if tts_available or cosyvoice_status == "available" or gpt_sovits_status == "available" else "unavailable",
                "reason": None if tts_available or cosyvoice_status == "available" or gpt_sovits_status == "available" else gpt_sovits_reason or cosyvoice_reason or "No real TTS clone backend is available",
            },
        },
    }


def classify_generation_reason(exc: BaseException | None, output_exists: bool) -> str:
    protect_required = {
        "VITS",
        "GPT-SoVITS",
        "CosyVoiceTokenizer",
        "CosyVoiceCAMPP",
        "WavLM",
        "HuBERTLargeLL60K",
        "WhisperLargeV3",
    }
    checkpoint_missing = [name for name in _checkpoint_status()["missing"] if name in protect_required]
    if checkpoint_missing:
        return "checkpoint_missing"
    if exc is not None:
        text = f"{type(exc).__name__}: {exc}".lower()
        if any(token in text for token in ["import", "module", "dependency", "not installed", "no module named"]):
            return "dependency_missing"
        if any(token in text for token in ["out of memory", "cannot allocate memory", "memoryerror"]):
            return "resource_exhausted"
        return "algorithm_runtime_error"
    if not output_exists:
        return "output_file_missing"
    return "unknown"


def summarize_return(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = repr(value)
    return text[:2000]


def output_path_from_result(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    for key in ["output_wav", "output_path", "protectedAudioPath", "protected_audio_path"]:
        candidate = value.get(key)
        if candidate:
            return str(candidate)
    return None


def _enabled_encoder_names(values: Any, fallback: list[str]) -> set[str]:
    if not values:
        values = fallback
    if isinstance(values, str):
        raw_values = [values]
    elif isinstance(values, list):
        raw_values = values
    else:
        raw_values = fallback
    normalized = {str(value).strip().lower().replace("_", "-") for value in raw_values if str(value).strip()}
    aliases = {
        "gpt-sovits": "gsv",
        "gpt_sovits": "gsv",
        "gsv": "gsv",
        "vits": "vits",
        "mfcc": "mfcc",
        "wavlm": "wavlm",
        "wavlm-large": "wavlm",
        "cosyvoice": "cosyvoice",
        "styletts2": "style",
        "style": "style",
    }
    return {aliases.get(value, value) for value in normalized}


def _read_weight(config: dict[str, Any], new_key: str, legacy_key: str, default: float, warnings: list[str]) -> float:
    value = to_float(config.get(new_key))
    if value is not None:
        return value
    legacy_value = to_float(config.get(legacy_key))
    if legacy_value is not None:
        warnings.append(f"{legacy_key} is deprecated; use {new_key}.")
        if legacy_key in {"lambdaSemantic", "lambdaTimbre"} and legacy_value < 1:
            warnings.append(f"{legacy_key}={legacy_value} looks like a UI-normalized value; using formal default {default}.")
            return default
        if legacy_key == "lambdaPsy" and legacy_value > 0.01:
            warnings.append(f"{legacy_key}={legacy_value} is outside formal psychoacoustic scale; using formal default {default}.")
            return default
        if legacy_key == "lambdaL2" and legacy_value < 0.05:
            warnings.append(f"{legacy_key}={legacy_value} looks like an old UI value; using formal default {default}.")
            return default
        return legacy_value
    return default


def _read_identity_weight(config: dict[str, Any], default: float, warnings: list[str]) -> float:
    value = to_float(config.get("weightIdentity"))
    if value is not None:
        return value
    lambda_id = to_float(config.get("lambdaId"))
    if lambda_id is not None:
        warnings.append("lambdaId is deprecated for request payload weights; use weightIdentity.")
        return lambda_id
    legacy_value = _read_weight(config, "weightFeature", "lambdaTimbre", default, warnings)
    if "weightFeature" in config or "lambdaTimbre" in config:
        warnings.append("weightFeature is a deprecated legacy alias of weightIdentity.")
    return legacy_value


def run_protection(
    input_path: Path,
    output_path: Path,
    payload: dict[str, Any],
    *,
    request_id: str,
    task_id: str,
    file_id: str | None,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Any | None = None,
) -> dict[str, Any]:
    optimization = payload.get("optimization") or {}
    timbre = payload.get("timbre") or {}
    psychoacoustic = payload.get("psychoacoustic") or {}
    semantic = payload.get("semantic") or {}
    config_defaults = runtime_config()["defaults"]
    optimization_defaults = config_defaults["optimization"]
    default_timbre_encoders = (config_defaults.get("timbre") or {}).get("encoders") or []
    active_timbre_encoders = _enabled_encoder_names(timbre.get("encoders"), default_timbre_encoders)
    epsilon_value = to_float(optimization.get("epsilon"))
    epsilon = epsilon_value if epsilon_value is not None else float(optimization_defaults["epsilon"])
    steps_raw = optimization.get("steps")
    steps = int(steps_raw) if steps_raw is not None else int(optimization_defaults["steps"])
    weight_warnings: list[str] = []
    weight_identity = _read_identity_weight(timbre, FORMAL_WEIGHT_FEATURE, weight_warnings)
    weight_semantic = _read_weight(semantic, "weightSemantic", "lambdaSemantic", FORMAL_WEIGHT_SEMANTIC, weight_warnings)
    weight_psy = _read_weight(psychoacoustic, "weightPsy", "lambdaPsy", FORMAL_WEIGHT_PSY, weight_warnings)
    weight_l2 = _read_weight(optimization, "weightL2", "lambdaL2", FORMAL_WEIGHT_L2, weight_warnings)
    device = os.getenv("SEME2E_API_DEVICE", "cpu")
    real_guard_enabled = os.getenv("SEME2E_API_REAL_GUARD", "1") == "1"
    diagnostics: dict[str, Any] = {
        "requestId": request_id,
        "taskId": task_id,
        "fileId": file_id,
        "inputAudioPath": str(input_path),
        "inputAudioExists": input_path.exists(),
        "outputPath": str(output_path),
        "outputPathExists": output_path.exists(),
        "cwd": os.getcwd(),
        "pythonExecutable": sys.executable,
        "device": device,
        "allowFallback": False,
        "realGuardEnabled": real_guard_enabled,
        "mode": timbre.get("mode") or payload.get("mode") or "untargeted",
        "epsilon": epsilon,
        "steps": steps,
        "weights": {
            "weightIdentity": weight_identity,
            "weightFeature": weight_identity,
            "weightSemantic": weight_semantic,
            "weightPsy": weight_psy,
            "weightL2": weight_l2,
        },
        "fixedOptimization": {
            "weight_stft": FIXED_WEIGHT_STFT,
            "weight_snr": FIXED_WEIGHT_SNR,
            "target_snr_db": FIXED_TARGET_SNR_DB,
            "selection_snr_db": FIXED_SELECTION_SNR_DB,
            "step_size": FIXED_STEP_SIZE,
            "init_noise": FIXED_INIT_NOISE,
            "l2_reduction": FIXED_L2_REDUCTION,
            "min_lr": FIXED_MIN_LR,
        },
        "deprecationWarnings": weight_warnings,
        "selectedSemanticEncoders": semantic.get("encoders"),
        "selectedTimbreEncoders": timbre.get("encoders"),
        "activeTimbreEncoders": sorted(active_timbre_encoders),
        "capabilities": get_capabilities_snapshot(RUNTIME_DIR, diagnose_capabilities),
        "protectCall": {
            "class": "VoiceShield",
            "inputPath": str(input_path),
            "outputPath": str(output_path),
        },
    }

    if real_guard_enabled:
        result: Any = None
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("TASK_CANCELLED")
            import torch
            from core.guard import VoiceShield

            semantic_defaults = config_defaults.get("semantic") or {}
            tokenizer_path = semantic.get("tokenizerPath") or semantic_defaults.get("tokenizerPath")
            if tokenizer_path and not Path(str(tokenizer_path)).is_absolute():
                tokenizer_path = str((ROOT / str(tokenizer_path)).resolve())
            hubert_path = semantic.get("hubertPath") or semantic_defaults.get("hubertPath") or "facebook/hubert-large-ll60k"
            whisper_path = semantic.get("whisperPath") or semantic_defaults.get("whisperPath") or "openai/whisper-large-v3"

            guard = VoiceShield(
                epsilon=epsilon,
                max_items=steps,
                device=torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu"),
                tokenizer_path=tokenizer_path,
                hubert_path=hubert_path,
                whisper_path=whisper_path,
                use_vits="vits" in active_timbre_encoders,
                use_gsv="gsv" in active_timbre_encoders,
                use_mfcc_timbre="mfcc" in active_timbre_encoders,
                use_wavlm="wavlm" in active_timbre_encoders,
                use_cosyvoice="cosyvoice" in active_timbre_encoders,
                weight_feature=weight_identity,
                weight_semantic=weight_semantic,
                weight_psy=weight_psy,
                weight_l2=weight_l2,
                l2_reduction=FIXED_L2_REDUCTION,
                init_noise=FIXED_INIT_NOISE,
                step_size=FIXED_STEP_SIZE,
                weight_stft=FIXED_WEIGHT_STFT,
                weight_snr=FIXED_WEIGHT_SNR,
                target_snr_db=FIXED_TARGET_SNR_DB,
                selection_snr_db=FIXED_SELECTION_SNR_DB,
            )
            diagnostics["protectCall"]["guardInit"] = "ok"
            result = guard.protect(input_path, output_path, progress_callback=progress_callback, cancel_event=cancel_event)
            returned_path = output_path_from_result(result)
            returned_path_exists = Path(returned_path).exists() if returned_path else None
            output_exists = output_path.exists()
            diagnostics.update(
                {
                    "protectReturnType": type(result).__name__,
                    "protectReturnHasOutputWav": isinstance(result, dict) and bool(result.get("output_wav")),
                    "protectReturnHasOutputPath": isinstance(result, dict) and bool(result.get("output_path")),
                    "protectReturnHasProtectedAudioPath": isinstance(result, dict) and bool(result.get("protectedAudioPath")),
                    "protectReturnedPath": returned_path,
                    "protectReturnedPathExists": returned_path_exists,
                    "outputPathExists": output_exists,
                    "protectReturnSummary": summarize_return(result),
                }
            )
            if not output_exists and returned_path and returned_path_exists:
                shutil.copyfile(returned_path, output_path)
                output_exists = output_path.exists()
                diagnostics["outputCopiedFromReturnedPath"] = returned_path
                diagnostics["outputPathExists"] = output_exists
            if not output_exists:
                reason = classify_generation_reason(None, output_exists)
                diagnostics["reason"] = reason
                raise ProtectGenerationError(
                     "保护音频生成失败：后端算法未生成保护音频。",
                    task_id=task_id,
                    diagnostics=diagnostics,
                    reason=reason,
                )
            if not isinstance(result, dict):
                result = {"raw_return": repr(result)}
            result["source"] = "VoiceShield.protect"
            result["preset_name"] = FORMAL_PRESET_NAME
            result["guardDiagnostics"] = diagnostics
            return result
        except Exception as exc:
            if isinstance(exc, ProtectGenerationError):
                raise
            output_exists = output_path.exists()
            diagnostics.update(
                {
                    "protectReturnType": type(result).__name__ if result is not None else None,
                    "protectReturnSummary": summarize_return(result) if result is not None else None,
                    "outputPathExists": output_exists,
                    "exceptionType": type(exc).__name__,
                    "exceptionMessage": str(exc),
                    "stackTrace": traceback.format_exc(),
                }
            )
            reason = classify_generation_reason(exc, output_exists)
            diagnostics["reason"] = reason
            raise ProtectGenerationError(
                "保护音频生成失败：后端算法未生成保护音频。",
                task_id=task_id,
                diagnostics=diagnostics,
                reason=reason,
            ) from exc

    diagnostics["reason"] = "real_guard_disabled"
    raise ProtectGenerationError(
        "保护音频生成失败：真实保护链路未启用。",
        task_id=task_id,
        diagnostics=diagnostics,
        reason="real_guard_disabled",
    )


def compute_perception(clean_path: Path, protected_path: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    optimization = payload.get("optimization") or {}
    epsilon = to_float(optimization.get("epsilon"))
    epsilon_norm = str(optimization.get("epsilonNorm") or optimization.get("epsilon_norm") or "linf")
    perception = {
        "snr": None,
        "pesq": None,
        "stoi": None,
        "mos": None,
        "mosLqo": None,
        "dnsMos": None,
        "dnsMosScore": None,
        "dnsMosStatus": "unavailable",
        "dnsMosReason": "语音质量评分尚未生成",
        "dnsMosModel": None,
        "dnsMosModelPath": None,
        "dnsMosProvider": None,
        "dnsMosDiagnostics": None,
        "qualityScore": None,
        "qualityLevel": None,
        "l2Norm": None,
        "l2Rms": None,
        "linfNorm": None,
        "epsilon": epsilon,
        "epsilonNorm": epsilon_norm,
        "epsilonUsageRate": None,
        "clippingRate": None,
        "lPsy": None,
        "overMaskRate": None,
        "psychoacousticViolationRate": None,
        "maskingThreshold": None,
        "perturbationSpectrum": None,
        "maskingCurve": [],
        "perturbation": None,
        "protectionQuality": None,
        "psychoacoustic": None,
        "status": "unavailable",
        "source": "metric_definitions.py",
        "_metricSources": {},
    }

    try:
        dns_mos = _evaluate_dnsmos_pair_isolated(clean_path, protected_path)
    except IsolatedWorkerError as exc:
        dns_mos = {
            "status": "error",
            "model": "DNSMOS P.835 OVRL",
            "reason": f"语音质量评分生成失败：{exc}",
            "diagnostics": exc.diagnostics,
        }
    except RuntimeError as exc:
        if str(exc) == "TASK_CANCELLED":
            raise
        dns_mos = {
            "status": "error",
            "model": "DNSMOS P.835 OVRL",
            "reason": f"语音质量评分生成失败：{exc}",
            "diagnostics": {
                "exceptionType": type(exc).__name__,
                "exceptionMessage": str(exc),
                "stackTrace": traceback.format_exc(),
            },
        }
    except Exception as exc:
        dns_mos = {
            "status": "error",
            "model": "DNSMOS P.835 OVRL",
            "reason": f"语音质量评分生成失败：{exc}",
            "diagnostics": {
                "exceptionType": type(exc).__name__,
                "exceptionMessage": str(exc),
                "stackTrace": traceback.format_exc(),
            },
        }

    dns_fields = _dnsmos_fields_from_evaluation(dns_mos)
    dns_score_payload = compute_protection_quality_score(None, None, None, dns_fields["dnsMos"])
    if dns_fields["dnsMos"] is None:
        dns_score_payload["dnsMosStatus"] = dns_fields["dnsMosStatus"]
        dns_score_payload["dnsMosReason"] = dns_fields["dnsMosReason"]
    quality: dict[str, Any] = {
        "snr": None,
        "pesq": None,
        "stoi": None,
        "mos": None,
        "mosLqo": None,
        **dns_score_payload,
        **dns_fields,
    }
    perception.update(
        {
            **dns_fields,
            "dnsMosScore": dns_score_payload.get("dnsMosScore"),
            "qualityScore": dns_score_payload.get("qualityScore"),
            "qualityLevel": dns_score_payload.get("qualityLevel"),
            "protectionQuality": quality,
        }
    )
    dns_source = metric_source(
        str(dns_fields["dnsMosStatus"]),
        str(dns_fields.get("dnsMosModel") or "DNSMOS P.835 OVRL"),
        reason=dns_fields.get("dnsMosReason"),
        formula="100*(DNSMOS_OVRL-1)/4 when 1<=DNSMOS_OVRL<=5",
    )
    perception["_metricSources"]["protectionQuality.dnsMos"] = dns_source

    try:
        x, xp, delta, sr = align_audio_pair(clean_path, protected_path)
        perturbation = compute_perturbation_metrics(x, xp, delta, sr, epsilon=epsilon, epsilon_norm=epsilon_norm)
        quality = compute_quality_metrics(
            x,
            xp,
            delta,
            sr,
            perturbation,
            dns_mos=dns_fields["dnsMos"],
            dns_mos_status=str(dns_fields["dnsMosStatus"]),
            dns_mos_reason=dns_fields.get("dnsMosReason"),
        )
        quality.update(dns_fields)
        quality_sources = quality.pop("_metricSources", {})
        quality_sources["protectionQuality.dnsMos"] = dns_source
        perception.update(perturbation)
        perception.update(
            {
                "snr": quality.get("snr"),
                "pesq": quality.get("pesq"),
                "stoi": quality.get("stoi"),
                "mos": quality.get("mos"),
                "mosLqo": quality.get("mosLqo"),
                "dnsMos": quality.get("dnsMos"),
                "dnsMosScore": quality.get("dnsMosScore"),
                "dnsMosStatus": quality.get("dnsMosStatus"),
                "dnsMosReason": quality.get("dnsMosReason"),
                "dnsMosModel": quality.get("dnsMosModel"),
                "dnsMosModelPath": quality.get("dnsMosModelPath"),
                "dnsMosProvider": quality.get("dnsMosProvider"),
                "dnsMosDiagnostics": quality.get("dnsMosDiagnostics"),
                "qualityScore": quality.get("qualityScore"),
                "qualityLevel": quality.get("qualityLevel"),
                "perturbation": perturbation,
                "protectionQuality": quality,
                "status": "available",
            }
        )
        sources = dict(perception["_metricSources"])
        perturbation_source = metric_source(
            "available",
            "align_audio_pair + compute_perturbation_metrics",
            formula="delta=xp-x; l2Norm=sqrt(sum(delta^2)); snr=10*log10((P_signal+1e-12)/(P_noise+1e-12))",
        )
        sources["perturbation.*"] = perturbation_source
        for key in perturbation:
            sources[f"perturbation.{key}"] = perturbation_source
        sources.update(quality_sources)
        perception["_metricSources"] = sources
    except Exception as exc:
        if str(exc) == "TASK_CANCELLED":
            raise
        reason = str(exc)
        perception["error"] = reason
        perception["_metricSources"].update(
            {
                "perturbation.*": metric_source("error", "align_audio_pair", reason=reason, formula="read/resample/mono/truncate audio pair"),
                "protectionQuality.snr": metric_source("unavailable", "compute_perturbation_metrics", reason=reason, formula="10*log10((P_signal+1e-12)/(P_noise+1e-12))"),
                "protectionQuality.pesq": metric_source("unavailable", "pesq", reason="Audio pair alignment failed before PESQ", formula="pesq(sr,x,xp,mode)"),
                "protectionQuality.stoi": metric_source("unavailable", "pystoi", reason="Audio pair alignment failed before STOI", formula="stoi(x,xp,sr)"),
                "protectionQuality.mosLqo": metric_source("unavailable", "objective_mos_lqo_model", reason="No explicit MOS-LQO objective model is configured", formula="None"),
                "protectionQuality.qualityScore": metric_source("unavailable", "VoiceShield_v2.1_piecewise_quality", reason=quality.get("scoreReason")),
                "protectionQuality.qualityLevel": metric_source("unavailable", "qualityScore_thresholds", reason="qualityScore is unavailable"),
                "psychoacoustic.*": metric_source("error", "engineering_stft_masking_threshold", reason=reason, formula="V=max(0,PSD_delta-Theta)"),
            }
        )
        return perception

    try:
        psycho = compute_psychoacoustic_metrics(x, xp, delta, sr)
        psycho_sources = psycho.pop("_metricSources", {})
        perception.update(
            {
                "lPsy": psycho.get("lPsy"),
                "overMaskRate": psycho.get("overMaskRate"),
                "psychoacousticViolationRate": psycho.get("overMaskRate"),
                "maskingThreshold": psycho.get("maskingThreshold"),
                "perturbationSpectrum": psycho.get("perturbationSpectrum"),
                "maskingCurve": psycho.get("chart") or [],
                "psychoacoustic": {
                    "lPsy": psycho.get("lPsy"),
                    "overMaskRate": psycho.get("overMaskRate"),
                    "frameCount": psycho.get("frameCount"),
                    "sampleRate": psycho.get("sampleRate"),
                    "hopLength": psycho.get("hopLength"),
                    "nFft": psycho.get("nFft"),
                    "aggregation": psycho.get("aggregation"),
                    "maskingThreshold": psycho.get("maskingThreshold"),
                    "perturbationSpectrum": psycho.get("perturbationSpectrum"),
                },
            }
        )
        perception["_metricSources"].update(psycho_sources)
        for key in ["lPsy", "overMaskRate", "maskingThreshold", "perturbationSpectrum"]:
            perception["_metricSources"][f"psychoacoustic.{key}"] = perception["_metricSources"]["psychoacoustic.*"]
    except Exception as exc:
        if str(exc) == "TASK_CANCELLED":
            raise
        reason = str(exc)
        perception["status"] = "partial"
        perception["psychoacousticError"] = reason
        psycho_source = metric_source(
            "error",
            "engineering_stft_masking_threshold",
            reason=reason,
            formula="V=max(0,PSD_delta-Theta)",
        )
        perception["_metricSources"]["psychoacoustic.*"] = psycho_source
        for key in ["lPsy", "overMaskRate", "maskingThreshold", "perturbationSpectrum"]:
            perception["_metricSources"][f"psychoacoustic.{key}"] = psycho_source
    return perception


def compute_mfcc_semantic(clean_path: Path, protected_path: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    details = compute_semantic_token_metrics(clean_path, protected_path, config or {})
    details.update(
        compute_protection_semantic_score(
            details.get("tokenChangeRate"),
            details.get("semanticDrift"),
        )
    )
    if not details.get("encoderDistances"):
        details["encoderDistances"] = empty_details()["semantic"]["encoderDistances"]
    return details


def refresh_result_scores(result: dict[str, Any]) -> dict[str, Any]:
    """Attach the v2.1 score contract without fabricating missing dimensions."""

    details = result.setdefault("details", {})
    perception = details.setdefault("perception", {})
    quality = perception.get("protectionQuality")
    if not isinstance(quality, dict):
        quality = {}
        perception["protectionQuality"] = quality
    explicit_dns_status = quality.get("dnsMosStatus") or perception.get("dnsMosStatus")
    explicit_dns_reason = quality.get("dnsMosReason") or perception.get("dnsMosReason")
    quality_score = compute_protection_quality_score(
        quality.get("snr", perception.get("snr")),
        quality.get("stoi", perception.get("stoi")),
        quality.get("pesq", perception.get("pesq")),
        quality.get("dnsMos", perception.get("dnsMos")),
    )
    quality.update(quality_score)
    if quality_score.get("dnsMos") is None:
        quality["dnsMosStatus"] = explicit_dns_status or quality_score.get("dnsMosStatus")
        quality["dnsMosReason"] = explicit_dns_reason or quality_score.get("dnsMosReason")
    perception.update(
        {
            "qualityScore": quality_score.get("qualityScore"),
            "qualityLevel": quality_score.get("qualityLevel"),
            "dnsMos": quality_score.get("dnsMos"),
            "dnsMosScore": quality_score.get("dnsMosScore"),
            "dnsMosStatus": quality.get("dnsMosStatus") or quality_score.get("dnsMosStatus"),
            "dnsMosReason": quality.get("dnsMosReason") or quality_score.get("dnsMosReason"),
        }
    )
    quality["qualityLevel"] = quality_score.get("qualityLevel")

    semantic = details.get("semantic")
    if not isinstance(semantic, dict):
        semantic = {}
        details["semantic"] = semantic
    semantic.update(
        compute_protection_semantic_score(
            semantic.get("tokenChangeRate"),
            semantic.get("semanticDrift"),
        )
    )
    speaker = details.get("speaker")
    if not isinstance(speaker, dict):
        speaker = {}
        details["speaker"] = speaker
    direct_similarity = to_float(speaker.get("simOriginalProtected", speaker.get("simAfter")))
    if direct_similarity is None:
        direct_distance = to_float(speaker.get("embeddingDistanceAfter", speaker.get("embeddingDistance")))
        if direct_distance is not None:
            direct_similarity = 1.0 - direct_distance
    if direct_similarity is None:
        direct_similarity = to_float(((result.get("summary") or {}).get("primaryMetrics") or {}).get("speakerSimilarity"))
    speaker.update(compute_direct_identity_score(direct_similarity))

    latest_clone_eval: dict[str, Any] | None = None
    for clone_result in result.get("cloneResults") or []:
        if not isinstance(clone_result, dict):
            continue
        clone_eval = clone_result.get("cloneEval")
        if not isinstance(clone_eval, dict):
            continue
        _sync_clone_eval_fields(clone_result, clone_eval)
        if clone_eval.get("cloneIdentityScore") is None:
            identity = compute_clone_identity_score(
                clone_eval.get("originalSimilarity", clone_result.get("originalSimilarity")),
                clone_eval.get("protectedSimilarity", clone_result.get("protectedSimilarity")),
            )
            clone_eval.update(identity)
            clone_eval["cloneDefenseScore"] = identity.get("cloneIdentityScore")
        elif clone_eval.get("identityBaselineWeight") is None:
            identity = compute_clone_identity_score(
                clone_eval.get("originalSimilarity", clone_result.get("originalSimilarity")),
                clone_eval.get("protectedSimilarity", clone_result.get("protectedSimilarity")),
            )
            if identity.get("identityBaselineWeight") is not None:
                clone_eval["identityBaselineWeight"] = identity["identityBaselineWeight"]
        clean_clone_mos = clone_eval.get("cleanCloneQualityMos", clone_result.get("cleanCloneQualityMos"))
        protected_clone_mos = clone_eval.get("protectedCloneQualityMos", clone_result.get("protectedCloneQualityMos"))
        if to_float(clean_clone_mos) is not None and to_float(protected_clone_mos) is not None:
            clone_eval.update(
                compute_clone_quality_score(
                    clean_clone_mos,
                    protected_clone_mos,
                    identity_baseline_weight=clone_eval.get("identityBaselineWeight"),
                    clone_identity_score=clone_eval.get("cloneIdentityScore"),
                    clone_semantic_score=clone_eval.get("cloneSemanticScore"),
                )
            )
        _sync_clone_eval_fields(clone_result, clone_eval)
        latest_clone_eval = clone_eval
    if latest_clone_eval is not None:
        details["cloneEval"] = latest_clone_eval

    summary_score = compute_overall_score(result)
    evaluation = summary_score["protectionEvaluation"]
    details["protectionEvaluation"] = evaluation
    result["protectionEvaluation"] = evaluation
    summary = result.setdefault("summary", {})
    summary["score"] = summary_score["score"]
    summary["verdict"] = summary_score["verdict"]
    metric_sources = summary.setdefault("metricSources", {})
    metric_sources.update(summary_score.get("_metricSources") or {})
    result["metricSources"] = metric_sources
    return summary_score


def maybe_asr_eval(
    clean_path: Path,
    protected_path: Path,
    payload: dict[str, Any],
    *,
    cancel_event: Any | None = None,
) -> dict[str, Any]:
    asr = empty_details()["asr"]
    reference_text = payload.get("referenceText") or payload.get("reference_text")
    asr["referenceText"] = reference_text
    semantic = payload.get("semantic") or {}
    configured_models = semantic.get("asrModels") if isinstance(semantic.get("asrModels"), list) else None
    requested_models = [str(item).strip() for item in (configured_models or [semantic.get("asrModel") or os.getenv("SEME2E_ASR_MODEL") or "openai-whisper:tiny"]) if str(item).strip()]
    asr_aliases = {
        "paraformer-large": "openai-whisper:tiny",
        "whisper-large-v3": "openai-whisper:tiny",
        "conformer-ctc": "openai-whisper:tiny",
        "whisper-tiny": "openai-whisper:tiny",
        "whisper-base": "openai-whisper:base",
    }
    actual_models: list[str] = []
    for model in requested_models:
        actual = asr_aliases.get(model.lower(), model)
        if actual not in actual_models:
            actual_models.append(actual)
    if not actual_models:
        actual_models = ["openai-whisper:tiny"]
    asr["requestedModels"] = requested_models
    asr["models"] = actual_models
    asr["model"] = actual_models[0]
    if os.getenv("SEME2E_ENABLE_ASR", "0") != "1" and payload.get("forceAsrEval") is not True:
        asr["status"] = "unavailable"
        asr["reason"] = "当前运行环境未启用 ASR 评估。"
        asr["_metricSources"] = {"asrEval.*": metric_source("not_run", "ASRTranscriber", reason=asr["reason"], formula="独立 ASR 转写与编辑距离评估")}
        return asr

    requested_device = os.getenv("SEME2E_ASR_DEVICE") or os.getenv("SEME2E_API_DEVICE", "cpu")
    worker_timeout = _env_int("SEME2E_ASR_WORKER_TIMEOUT_SECONDS", 600)
    try:
        evaluations = []
        for model in actual_models:
            def run_worker(
                worker_device: str,
                worker_env: Mapping[str, str] | None,
                _selected_gpu: str | None,
                attempt_timeout_seconds: float,
                *,
                current_model: str = model,
            ) -> dict[str, Any]:
                return _run_isolated_json_worker(
                    ROOT / "asr_worker.py",
                    {
                        "model": current_model,
                        "device": worker_device,
                        "language": str(payload.get("language") or "en"),
                        "originalPath": str(clean_path.resolve()),
                        "protectedPath": str(protected_path.resolve()),
                    },
                    timeout_seconds=attempt_timeout_seconds,
                    cancel_event=cancel_event,
                    **({"env_overrides": worker_env} if worker_env else {}),
                )

            worker_result, selected_gpu, gpu_attempts = _run_gpu_worker_with_retry(
                operation_name=f"asr:{model}",
                worker_slot=ASR_WORKER_SLOTS,
                requested_device=requested_device,
                visible_devices_env="SEME2E_ASR_CUDA_VISIBLE_DEVICES",
                cancel_event=cancel_event,
                operation=run_worker,
                timeout_seconds=worker_timeout,
                explicit_device=bool(os.getenv("SEME2E_ASR_DEVICE", "").strip()),
                minimum_free_mib=max(
                    0,
                    _env_int(
                        "SEME2E_ASR_GPU_MIN_FREE_MIB",
                        _env_int("SEME2E_GPU_MIN_FREE_MIB", 0),
                    ),
                ),
            )
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("TASK_CANCELLED")
            clean_text = str(worker_result.get("originalText") or "")
            protected_text = str(worker_result.get("protectedText") or "")
            item = compute_asr_metrics(
                clean_text,
                protected_text,
                reference_text=reference_text,
                language=payload.get("language"),
                model=model,
            )
            item["gpu"] = selected_gpu
            if gpu_attempts:
                item["gpuAttempts"] = gpu_attempts
            evaluations.append(item)
        if evaluations:
            asr.update(evaluations[0])
            asr["model"] = evaluations[0]["model"]
            asr["evaluations"] = evaluations
            asr["status"] = "available"
    except IsolatedWorkerError as exc:
        asr["status"] = "unavailable"
        asr["error"] = str(exc)
        asr["diagnostics"] = exc.diagnostics
        asr["_metricSources"] = {"asrEval.*": metric_source("error", "isolated_asr_worker", reason=str(exc), formula="transcribe(original/protected)+Levenshtein")}
    except RuntimeError as exc:
        if str(exc) == "TASK_CANCELLED":
            raise
        asr["status"] = "unavailable"
        asr["error"] = str(exc)
        asr["_metricSources"] = {"asrEval.*": metric_source("error", "ASRTranscriber", reason=str(exc), formula="transcribe(original/protected)+Levenshtein")}
    except Exception as exc:
        asr["status"] = "unavailable"
        asr["error"] = str(exc)
        asr["_metricSources"] = {"asrEval.*": metric_source("error", "ASRTranscriber", reason=str(exc), formula="transcribe(original/protected)+Levenshtein")}
    return asr


def create_asr_eval(task_id: str, payload: dict[str, Any], *, cancel_event: Any | None = None) -> dict[str, Any]:
    original_path, protected_path, result = _task_audio_paths(task_id)
    request_semantic = ((result.get("request") or {}).get("semantic") or {}) if isinstance(result.get("request"), dict) else {}
    payload_semantic = payload.get("semantic") if isinstance(payload.get("semantic"), dict) else {}
    semantic_config = {
        **request_semantic,
        **payload_semantic,
        "asrModel": payload.get("model") or payload_semantic.get("asrModel") or request_semantic.get("asrModel"),
    }
    asr = maybe_asr_eval(
        original_path,
        protected_path,
        {
            "referenceText": payload.get("referenceText") or payload.get("reference_text"),
            "language": payload.get("language"),
            "semantic": semantic_config,
            "forceAsrEval": True,
        },
        cancel_event=cancel_event,
    )
    created_at = utc_now_iso()
    response = {
        "taskId": task_id,
        "asrSubId": payload.get("asrSubId"),
        "status": asr.get("status") or "available",
        "asr": asr,
        "request": {
            "model": payload.get("model"),
            "language": payload.get("language"),
            "referenceText": payload.get("referenceText") or payload.get("reference_text"),
        },
        "createdAt": created_at,
    }
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("TASK_CANCELLED")
    with RESULT_WRITE_LOCK:
        result_path = TASK_DIR / task_id / "result.json"
        latest_result = load_result(task_id) if result_path.exists() else result
        details = latest_result.setdefault("details", {})
        details["asr"] = asr
        primary = latest_result.setdefault("summary", {}).setdefault("primaryMetrics", {})
        for key in ["wer", "cer"]:
            if key in asr:
                primary[key] = asr.get(key)
        metric_sources = latest_result.setdefault("summary", {}).setdefault("metricSources", {})
        metric_sources.update(asr.get("_metricSources") or {})
        latest_result["asrModel"] = asr.get("model")
        latest_result["updatedAt"] = created_at
        asr_results = latest_result.setdefault("asrResults", [])
        asr_sub_id = response.get("asrSubId")
        if asr_sub_id:
            asr_results[:] = [item for item in asr_results if item.get("asrSubId") != asr_sub_id]
        asr_results.append(response)
        refresh_result_scores(latest_result)
        save_result(TASK_DIR / task_id, latest_result)
    return response


def maybe_speaker_eval(clean_path: Path, protected_path: Path) -> dict[str, Any]:
    return compute_direct_speaker_metrics(clean_path, protected_path)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def update_chain(chains: list[dict[str, Any]], chain_id: str, status: str, metrics: dict[str, Any] | None = None) -> None:
    for chain in chains:
        if chain["chainId"] == chain_id:
            chain["status"] = status
            if metrics is not None:
                chain["metrics"] = metrics
            return


def build_task_payload(
    task_id: str,
    payload: dict[str, Any],
    input_path: Path,
    protected_path: Path,
    uploaded_file_id: str | None,
    started_at: str,
    completed_at: str,
    protection_result: dict[str, Any],
    source_path: Path | None = None,
    preprocess_meta: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    base_url = f"/api/artifacts/{task_id}"
    source_url = f"{base_url}/source/{source_path.name}" if source_path is not None else None
    original_url = f"{base_url}/original/{input_path.name}"
    protected_url = f"{base_url}/protected/{protected_path.name}"
    result_json_url = f"{base_url}/result.json"

    details = empty_details()
    chains = default_chains()

    optimization = payload.get("optimization") or {}
    timbre = payload.get("timbre") or {}
    semantic_cfg = payload.get("semantic") or {}
    psychoacoustic = payload.get("psychoacoustic") or {}
    meta = read_wav_meta(input_path)
    loss_summary = compute_loss_summary(protection_result, payload)
    loss_final = loss_summary["lossFinal"]
    loss_weights = loss_summary["lossWeights"]
    trace = loss_summary["optimizationTrace"]
    metric_sources: dict[str, Any] = {}
    metric_sources.update(loss_summary.get("_metricSources") or {})
    details["generation"].update(
        {
            "mode": timbre.get("mode") or payload.get("mode") or "untargeted",
            "epsilon": to_float(optimization.get("epsilon")),
            "steps": int(optimization.get("steps") or 0) or None,
            "maxSteps": protection_result.get("max_steps") or int(optimization.get("steps") or 0) or None,
            "selectedStep": loss_summary.get("selectedStep"),
            "snrDb": to_float(protection_result.get("snr_db", protection_result.get("snr"))),
            "presetName": protection_result.get("preset_name") or FORMAL_PRESET_NAME,
            "sampleRate": meta["sampleRate"],
            "durationSec": meta["durationSec"],
            "lossFinal": loss_final,
            "lossWeights": {
                "weight_identity": loss_weights.get("lambdaId"),
                "weight_feature": loss_weights.get("lambdaFeat"),
                "weight_semantic": loss_weights.get("lambdaSem"),
                "weight_psy": loss_weights.get("lambdaPsy"),
                "weight_l2": loss_weights.get("lambda2"),
                "lambdaId": loss_weights.get("lambdaId"),
                "lambdaFeat": loss_weights.get("lambdaFeat"),
                "lambdaSem": loss_weights.get("lambdaSem"),
                "lambdaPsy": loss_weights.get("lambdaPsy"),
                "lambda2": loss_weights.get("lambda2"),
                "lambdaStft": FIXED_WEIGHT_STFT,
                "lambdaSnr": FIXED_WEIGHT_SNR,
                "targetSnrDb": FIXED_TARGET_SNR_DB,
                "selectionSnrDb": FIXED_SELECTION_SNR_DB,
            },
            "optimizationTrace": trace,
            "internalOptimizationTrace": protection_result.get("optimization_trace") or [],
            "averageStepSec": loss_summary["averageStepSec"],
            "effectiveConfig": protection_result.get("effective_config"),
            "lossItems": protection_result.get("loss_items"),
            "models": protection_result.get("models"),
            "checkpoints": protection_result.get("checkpoints"),
            "source": protection_result.get("source") or "VoiceShield.protect",
            "status": "computed" if protected_path.exists() else "unavailable",
            "realProtect": True,
            "preprocessing": preprocess_meta,
        }
    )
    if protection_result.get("warning"):
        details["generation"]["warning"] = protection_result["warning"]
    if protection_result.get("guardError"):
        details["generation"]["guardError"] = protection_result["guardError"]
    if protection_result.get("guardSkipped"):
        details["generation"]["guardSkipped"] = protection_result["guardSkipped"]
    if protection_result.get("guardDiagnostics"):
        details["generation"]["diagnostics"] = protection_result["guardDiagnostics"]
    if protection_result.get("guardDiagnostics", {}).get("deprecationWarnings"):
        details["generation"]["deprecationWarnings"] = protection_result["guardDiagnostics"]["deprecationWarnings"]
    update_chain(chains, "protect_generation", details["generation"]["status"], details["generation"]["lossFinal"] or {})

    if progress_callback is not None:
        progress_callback(progress=0.96, stage="result_evaluation", message="正在计算扰动与可听性指标")
    details["perception"] = compute_perception(input_path, protected_path, payload)
    metric_sources.update(details["perception"].get("_metricSources") or {})
    if details["perception"]["snr"] is None:
        details["perception"]["snr"] = to_float(protection_result.get("snr"))
    update_chain(
        chains,
        "perception_eval",
        details["perception"]["status"],
        {"snr": details["perception"]["snr"], "pesq": details["perception"]["pesq"]},
    )

    if progress_callback is not None:
        progress_callback(progress=0.97, stage="result_evaluation", message="正在计算语义与 tokenizer 指标")
    details["semantic"] = compute_mfcc_semantic(input_path, protected_path, semantic_cfg)
    metric_sources.update(details["semantic"].get("_metricSources") or {})
    update_chain(
        chains,
        "semantic_tokenizer_eval",
        details["semantic"]["status"],
        {
            "tokenErrorRate": details["semantic"]["tokenErrorRate"],
            "semanticDrift": details["semantic"]["semanticDrift"],
        },
    )

    details["asr"] = empty_details()["asr"]
    details["asr"]["reason"] = "ASR 评估与语音保护独立执行；请在工作台选择识别模型后运行测试。"
    metric_sources["asrEval.*"] = metric_source("not_run", "ASRTranscriber", reason="尚未执行独立 ASR 测试", formula="运行 ASR 测试后生成")
    update_chain(chains, "asr_eval", details["asr"]["status"], {"wer": details["asr"]["wer"], "cer": details["asr"]["cer"]})

    if progress_callback is not None:
        progress_callback(progress=0.985, stage="result_evaluation", message="正在计算声音身份相似度指标")
    details["speaker"] = maybe_speaker_eval(input_path, protected_path)
    metric_sources.update(details["speaker"].get("_metricSources") or {})
    update_chain(chains, "speaker_eval", details["speaker"]["status"], {"speakerSimilarity": details["speaker"]["simOriginalProtected"]})
    metric_sources["cloneEval.*"] = metric_source("not_run", "clone-voice", reason="尚未执行独立语音克隆测试", formula="运行语音克隆测试后生成")
    metric_sources["cloneEval.cloneConfidenceBefore"] = metric_source("unavailable", "confidence_calibrator", reason="No confidence calibrator is configured", formula="sigmoid(A*similarity+B)")
    metric_sources["cloneEval.cloneConfidenceAfter"] = metric_source("unavailable", "confidence_calibrator", reason="No confidence calibrator is configured", formula="sigmoid(A*similarity+B)")
    metric_sources["cloneEval.cloneConfidenceDropRate"] = metric_source("unavailable", "confidence_calibrator", reason="No confidence calibrator is configured", formula="(cloneConfidenceBefore-cloneConfidenceAfter)/max(cloneConfidenceBefore,EPS)")
    metric_sources["cloneEval.cloneTrend"] = metric_source("not_run", "multi_checkpoint_clone_eval", reason="clone trend is disabled; only final clone evaluation is reported", formula="None")

    primary = empty_primary_metrics()
    primary.update(
        {
            "wer": details["asr"]["wer"],
            "cer": details["asr"]["cer"],
            "tokenChangeRate": details["semantic"]["tokenChangeRate"],
            "tokenErrorRate": details["semantic"]["tokenErrorRate"],
            "semanticDrift": details["semantic"]["semanticDrift"],
            "speakerSimilarity": details["speaker"]["simAfter"],
            "snr": details["perception"]["snr"],
            "pesq": details["perception"]["pesq"],
        }
    )

    charts = empty_charts()
    charts["optimizationTrend"] = trace
    charts["psychoacoustic"] = details["perception"]["maskingCurve"]

    elapsed = None
    try:
        from datetime import datetime

        def _parse_custom(dt_str: str) -> datetime:
            return datetime.strptime(dt_str, "%Y.%m.%d %H:%M:%S")

        elapsed = (_parse_custom(completed_at) - _parse_custom(started_at)).total_seconds()
    except Exception:
        elapsed = None

    result = {
        "taskId": task_id,
        "status": "completed",
        "mode": payload.get("mode") or "joint",
        "dataMode": "backend",
        "createdAt": started_at,
        "submittedAt": started_at,
        "completedAt": completed_at,
        "elapsedSec": elapsed,
        "summary": {
            "verdict": "防护结果已生成",
            "score": None,
            "primaryMetrics": primary,
            "metricSources": metric_sources,
        },
        "artifacts": {
            "sourceAudioUrl": source_url,
            "originalAudioUrl": original_url,
            "protectedAudioUrl": protected_url,
            "resultJsonUrl": result_json_url,
        },
        "audio": {
            "source": audio_meta(source_path, source_url, uploaded_file_id) if source_path is not None and source_url is not None else None,
            "original": audio_meta(input_path, original_url, uploaded_file_id),
            "protected": audio_meta(protected_path, protected_url),
        },
        "preprocessing": preprocess_meta,
        "details": details,
        "chains": chains,
        "charts": charts,
        "request": {
            "mode": payload.get("mode"),
            "targets": payload.get("targets") or [],
            "semantic": {
                "enabled": bool(semantic_cfg.get("enabled")),
                "encoders": semantic_cfg.get("encoders") or [],
                "tokenizerPath": semantic_cfg.get("tokenizerPath"),
                "hubertPath": semantic_cfg.get("hubertPath"),
                "whisperPath": semantic_cfg.get("whisperPath"),
                "weightSemantic": loss_weights.get("lambdaSem"),
                "lambdaSemantic": loss_weights.get("lambdaSem"),
            },
            "timbre": {
                "enabled": bool(timbre.get("enabled")),
                "mode": timbre.get("mode"),
                "encoders": timbre.get("encoders") or [],
                "weightIdentity": loss_weights.get("lambdaId"),
                "lambdaId": loss_weights.get("lambdaId"),
                "weightFeature": loss_weights.get("lambdaFeat"),
                "lambdaTimbre": loss_weights.get("lambdaFeat"),
            },
            "psychoacoustic": {
                "enabled": bool(psychoacoustic.get("enabled")),
                "weightPsy": loss_weights.get("lambdaPsy"),
                "lambdaPsy": loss_weights.get("lambdaPsy"),
            },
            "optimization": {
                "epsilon": to_float(optimization.get("epsilon")),
                "steps": int(optimization.get("steps") or 0) or None,
                "weightL2": loss_weights.get("lambda2"),
                "lambdaL2": loss_weights.get("lambda2"),
            },
        },
        "metricSources": metric_sources,
        "realProtect": True,
        "selectedStep": loss_summary.get("selectedStep"),
        "effectiveConfig": protection_result.get("effective_config"),
        "presetName": protection_result.get("preset_name") or FORMAL_PRESET_NAME,
        "warning": None,
        "backend": {
            "version": "SemE2E API adapter",
            "commit": git_commit(),
            "python": sys.version.split()[0],
        },
    }
    result["summary"]["metricSources"] = metric_sources
    result["metricSources"] = metric_sources
    refresh_result_scores(result)
    return result


def _lock_result_file(lock_file: Any) -> None:
    if os.name == "nt":
        import msvcrt

        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        while True:
            try:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(0.05)
    elif os.name == "posix":
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)


def _unlock_result_file(lock_file: Any) -> None:
    if os.name == "nt":
        import msvcrt

        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    elif os.name == "posix":
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _result_file_transaction(task_dir: Path) -> Any:
    task_dir.mkdir(parents=True, exist_ok=True)
    with RESULT_WRITE_LOCK:
        with (task_dir / ".result.lock").open("a+b") as lock_file:
            _lock_result_file(lock_file)
            try:
                yield
            finally:
                _unlock_result_file(lock_file)


def _write_result_atomic_unlocked(task_dir: Path, result: dict[str, Any]) -> None:
    result_path = task_dir / "result.json"
    temporary_path = task_dir / f".result.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary_path.open("x", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, result_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_result(task_dir: Path, result: dict[str, Any]) -> None:
    with _result_file_transaction(task_dir):
        _write_result_atomic_unlocked(task_dir, result)


def update_result_safely(
    task_id: str,
    updater: Callable[[dict[str, Any]], bool],
    *,
    after_write: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], bool]:
    task_dir = TASK_DIR / task_id
    result_path = task_dir / "result.json"
    with _result_file_transaction(task_dir):
        if not result_path.exists():
            raise FileNotFoundError(f"task {task_id} result.json is missing")
        with result_path.open("r", encoding="utf-8") as file:
            result = json.load(file)
        changed = bool(updater(result))
        if changed:
            _write_result_atomic_unlocked(task_dir, result)
            if after_write is not None:
                after_write(result)
        return result, changed


def load_result(task_id: str) -> dict[str, Any]:
    with (TASK_DIR / task_id / "result.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def create_psychoacoustic_slice(task_id: str, mode: str = "mean", time_sec: float | None = None) -> dict[str, Any]:
    original_path, protected_path, result = _task_audio_paths(task_id)
    x, xp, delta, sr = align_audio_pair(original_path, protected_path)
    audio = result.get("audio") or {}
    protected_meta = audio.get("protected") or {}
    original_meta = audio.get("original") or {}
    duration_sec = to_float(protected_meta.get("durationSec") or protected_meta.get("duration"))
    if duration_sec is None:
        duration_sec = to_float(original_meta.get("durationSec") or original_meta.get("duration"))
    return compute_psychoacoustic_slice(x, xp, delta, sr, mode=mode, time_sec=time_sec, duration_sec=duration_sec)


def new_task_id() -> str:
    return f"task_{uuid.uuid4().hex[:12]}"


def new_file_id() -> str:
    return f"file_{uuid.uuid4().hex[:12]}"


def create_task(
    input_path: Path,
    uploaded_file_id: str | None,
    payload: dict[str, Any],
    input_filename: str | None = None,
    request_id: str | None = None,
    task_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Any | None = None,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    task_id = task_id or new_task_id()
    task_dir = TASK_DIR / task_id
    source_dir = task_dir / "source"
    original_dir = task_dir / "original"
    protected_dir = task_dir / "protected"
    source_dir.mkdir(parents=True, exist_ok=True)
    original_dir.mkdir(parents=True, exist_ok=True)
    protected_dir.mkdir(parents=True, exist_ok=True)

    started_at = utc_now_iso()
    display_filename = Path(input_filename).name if input_filename else input_path.name
    if not display_filename or display_filename in {".", ".."}:
        display_filename = input_path.name
    source_path = source_dir / display_filename
    if input_path.resolve() != source_path.resolve():
        shutil.copyfile(input_path, source_path)
    display_stem = Path(display_filename).stem
    original_path = original_dir / f"{display_stem}.wav"
    protected_path = protected_dir / f"{display_stem}_protected.wav"
    preprocess_meta = preprocess_audio(
        source_path,
        original_path,
        target_sample_rate=24_000,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
    preprocess_meta["source"]["fileId"] = uploaded_file_id
    preprocess_path = task_dir / "preprocess.json"
    preprocess_temp_path = task_dir / ".preprocess.json.tmp"
    with preprocess_temp_path.open("w", encoding="utf-8") as file:
        json.dump(preprocess_meta, file, ensure_ascii=False, indent=2)
    os.replace(preprocess_temp_path, preprocess_path)
    if progress_callback is not None:
        progress_callback(progress=0.18, stage="encoder_loading", message="录音预处理完成，正在加载防护模型")

    protection_result = run_protection(
        original_path,
        protected_path,
        payload,
        request_id=request_id or "",
        task_id=task_id,
        file_id=uploaded_file_id,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("TASK_CANCELLED")
    if progress_callback is not None:
        progress_callback(progress=0.955, stage="result_evaluation", message="防护音频已生成，正在准备结果评估")
    completed_at = utc_now_iso()
    result = build_task_payload(
        task_id,
        payload,
        original_path,
        protected_path,
        uploaded_file_id,
        started_at,
        completed_at,
        protection_result,
        source_path=source_path,
        preprocess_meta=preprocess_meta,
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        progress_callback(progress=0.99, stage="report_generation", message="后端正在写入结果报告")
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("TASK_CANCELLED")
    save_result(task_dir, result)
    return result


def _task_audio_paths(task_id: str) -> tuple[Path, Path, dict[str, Any]]:
    result = load_result(task_id)
    audio = result.get("audio") or {}
    original_name = (audio.get("original") or {}).get("filename")
    protected_name = (audio.get("protected") or {}).get("filename")
    if not original_name or not protected_name:
        raise FileNotFoundError(f"task {task_id} does not include original/protected audio metadata")
    original_path = TASK_DIR / task_id / "original" / original_name
    protected_path = TASK_DIR / task_id / "protected" / protected_name
    if not original_path.exists() or not protected_path.exists():
        raise FileNotFoundError(f"task {task_id} audio artifacts are missing")
    return original_path, protected_path, result


def _join_dnsmos_task_flight(task_id: str) -> tuple[threading.Event, bool]:
    with DNSMOS_TASK_FLIGHTS_GUARD:
        existing = DNSMOS_TASK_FLIGHTS.get(task_id)
        if existing is not None:
            return existing, False
        flight = threading.Event()
        DNSMOS_TASK_FLIGHTS[task_id] = flight
        return flight, True


def _finish_dnsmos_task_flight(task_id: str, flight: threading.Event) -> None:
    with DNSMOS_TASK_FLIGHTS_GUARD:
        if DNSMOS_TASK_FLIGHTS.get(task_id) is flight:
            DNSMOS_TASK_FLIGHTS.pop(task_id, None)
        flight.set()


def ensure_protection_dnsmos(task_id: str, result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Backfill DNSMOS while coalescing concurrent attempts for the same persisted task."""

    current = result if isinstance(result, dict) else load_result(task_id)
    current_perception = ((current.get("details") or {}).get("perception") or {})
    current_quality = current_perception.get("protectionQuality")
    if not isinstance(current_quality, dict):
        current_quality = {}
    if to_float(current_quality.get("dnsMos", current_perception.get("dnsMos"))) is not None:
        refresh_result_scores(current)
        return current

    flight, is_leader = _join_dnsmos_task_flight(task_id)
    if not is_leader:
        flight.wait()
        latest = load_result(task_id)
        refresh_result_scores(latest)
        return latest

    try:
        with RESULT_WRITE_LOCK:
            latest = load_result(task_id)
            latest_perception = ((latest.get("details") or {}).get("perception") or {})
            latest_quality = latest_perception.get("protectionQuality")
            if not isinstance(latest_quality, dict):
                latest_quality = {}
            if to_float(latest_quality.get("dnsMos", latest_perception.get("dnsMos"))) is not None:
                refresh_result_scores(latest)
                return latest

        try:
            original_path, protected_path, _ = _task_audio_paths(task_id)
            evaluation = _evaluate_dnsmos_pair_isolated(original_path, protected_path)
        except IsolatedWorkerError as exc:
            evaluation = {
                "status": "error",
                "model": "DNSMOS P.835 OVRL",
                "reason": f"语音质量评分生成失败：{exc}",
                "diagnostics": exc.diagnostics,
            }
        except RuntimeError as exc:
            if str(exc) == "TASK_CANCELLED":
                raise
            evaluation = {
                "status": "error",
                "model": "DNSMOS P.835 OVRL",
                "reason": f"语音质量评分生成失败：{exc}",
                "diagnostics": {
                    "exceptionType": type(exc).__name__,
                    "exceptionMessage": str(exc),
                    "stackTrace": traceback.format_exc(),
                },
            }
        except Exception as exc:
            evaluation = {
                "status": "error",
                "model": "DNSMOS P.835 OVRL",
                "reason": f"语音质量评分生成失败：{exc}",
                "diagnostics": {
                    "exceptionType": type(exc).__name__,
                    "exceptionMessage": str(exc),
                    "stackTrace": traceback.format_exc(),
                },
            }

        dns_fields = _dnsmos_fields_from_evaluation(evaluation)
        dns_source = metric_source(
            str(dns_fields["dnsMosStatus"]),
            str(dns_fields.get("dnsMosModel") or "DNSMOS P.835 OVRL"),
            reason=dns_fields.get("dnsMosReason"),
            formula="100*(DNSMOS_OVRL-1)/4 when 1<=DNSMOS_OVRL<=5",
        )

        with RESULT_WRITE_LOCK:
            latest = load_result(task_id)
            details = latest.setdefault("details", {})
            perception = details.setdefault("perception", {})
            quality = perception.get("protectionQuality")
            if not isinstance(quality, dict):
                quality = {}
                perception["protectionQuality"] = quality
            existing = to_float(quality.get("dnsMos", perception.get("dnsMos")))
            if existing is None:
                quality.update(dns_fields)
                perception.update(dns_fields)
                perception.setdefault("_metricSources", {})["protectionQuality.dnsMos"] = dns_source
                summary = latest.setdefault("summary", {})
                metric_sources = summary.setdefault("metricSources", {})
                metric_sources["protectionQuality.dnsMos"] = dns_source
                refresh_result_scores(latest)
                metric_sources["protectionQuality.qualityScore"] = metric_source(
                    "available" if quality.get("qualityScore") is not None else "unavailable",
                    "VoiceShield_v2.1_piecewise_quality",
                    reason=quality.get("scoreReason"),
                    formula="without DNSMOS: (.40*S_snr+.35*S_stoi+.15*S_pesq)/.90; with DNSMOS: .40*S_snr+.35*S_stoi+.15*S_pesq+.10*S_dnsmos",
                )
                latest["metricSources"] = metric_sources
                save_result(TASK_DIR / task_id, latest)
            else:
                refresh_result_scores(latest)
            return latest
    finally:
        _finish_dnsmos_task_flight(task_id, flight)


def _local_tts_model_files(model: str) -> tuple[Path, Path] | None:
    backend_value = normalize_tts_model(model)
    for item in SUPPORTED_TTS_MODELS:
        if str(item["backendValue"]).lower() != backend_value.lower():
            continue
        model_dir = _tts_cache_dir() / str(item["cacheName"])
        config_path = model_dir / "config.json"
        model_path = model_dir / "model.pth"
        if not model_path.exists():
            model_path = model_dir / "model_file.pth"
        if config_path.exists() and model_path.exists():
            ready, reason = _torch_checkpoint_ready(model_path)
            if not ready:
                raise RuntimeError(reason or f"invalid local TTS checkpoint: {model_path}")
            config_path = _portable_tts_config(model_dir, config_path)
            if "xtts" in backend_value.lower():
                return model_dir, config_path
            return model_path, config_path
    return None


def _tts_clone_to_file(reference_path: Path, text: str, output_path: Path, *, model: str, language: str, speed: float, device: str) -> str:
    from TTS.api import TTS
    from TTS.tts.models import xtts as xtts_module
    import torch
    import torchaudio.functional as ta_functional
    import soundfile as sf

    def load_audio_without_torchcodec(audiopath: str, sampling_rate: int) -> Any:
        audio_np, loaded_sr = sf.read(str(audiopath), dtype="float32", always_2d=True)
        audio = torch.from_numpy(audio_np.T)
        if audio.size(0) != 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if loaded_sr != sampling_rate:
            audio = ta_functional.resample(audio, loaded_sr, sampling_rate)
        if torch.any(audio > 10) or not torch.any(audio < 0):
            print(f"Error with {audiopath}. Max={audio.max()} min={audio.min()}")
        audio.clip_(-1, 1)
        return audio

    original_torch_load = torch.load
    original_xtts_load_audio = xtts_module.load_audio

    def torch_load_for_trusted_tts_checkpoint(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = torch_load_for_trusted_tts_checkpoint
    xtts_module.load_audio = load_audio_without_torchcodec
    try:
        local_model = _local_tts_model_files(model)
        if local_model is not None:
            model_dir, config_path = local_model
            tts = TTS(model_path=str(model_dir), config_path=str(config_path), progress_bar=False)
        else:
            tts = TTS(model)
    except Exception:
        xtts_module.load_audio = original_xtts_load_audio
        raise
    finally:
        torch.load = original_torch_load
    if hasattr(tts, "to"):
        try:
            tts.to(device)
        except Exception:
            tts.to("cpu")
    if getattr(tts, "model_name", None) is None:
        tts.model_name = model
    if getattr(tts, "config", None) is None and getattr(tts, "synthesizer", None) is not None:
        tts.config = getattr(tts.synthesizer, "tts_config", None)
    if local_model is not None and "xtts" not in normalize_tts_model(model).lower():
        tts._check_arguments = lambda *args, **kwargs: None
    kwargs = {
        "text": text,
        "speaker_wav": str(reference_path),
        "language": language,
        "file_path": str(output_path),
    }
    if speed:
        kwargs["speed"] = speed
    try:
        tts.tts_to_file(**kwargs)
    except TypeError:
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("speed", None)
        try:
            tts.tts_to_file(**fallback_kwargs)
        except TypeError:
            fallback_kwargs.pop("language", None)
            tts.tts_to_file(**fallback_kwargs)
    finally:
        xtts_module.load_audio = original_xtts_load_audio
    return model


def _coqui_tts_clone_pair(
    original_reference: Path,
    protected_reference: Path,
    original_output: Path,
    protected_output: Path,
    *,
    text: str,
    model: str,
    language: str,
    speed: float,
    device: str,
    task_id: str,
    clone_sub_id: str | None,
    cancel_event: Any | None,
    runtime_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    local_model = _local_tts_model_files(model)
    request_payload: dict[str, Any] = {
        "taskId": task_id,
        "cloneSubId": clone_sub_id,
        "model": model,
        "text": text,
        "language": language,
        "speed": speed,
        "ttsHome": str(PROJECT_TTS_CACHE_DIR.resolve()),
        "originalReferencePath": str(original_reference.resolve()),
        "protectedReferencePath": str(protected_reference.resolve()),
        "originalOutputPath": str(original_output.resolve()),
        "protectedOutputPath": str(protected_output.resolve()),
    }
    if local_model is not None:
        model_path, config_path = local_model
        request_payload["modelPath"] = str(model_path.resolve())
        request_payload["configPath"] = str(config_path.resolve())
    worker_timeout = _env_int("SEME2E_COQUI_TTS_WORKER_TIMEOUT_SECONDS", 900)

    def run_worker(
        worker_device: str,
        worker_env: Mapping[str, str] | None,
        _selected_gpu: str | None,
        attempt_timeout_seconds: float,
    ) -> dict[str, Any]:
        request_payload["device"] = worker_device
        return _run_isolated_json_worker(
            ROOT / "coqui_tts_worker.py",
            request_payload,
            timeout_seconds=attempt_timeout_seconds,
            cancel_event=cancel_event,
            **({"env_overrides": worker_env} if worker_env else {}),
        )

    def cleanup_retry(_attempt: dict[str, Any]) -> None:
        original_output.unlink(missing_ok=True)
        protected_output.unlink(missing_ok=True)

    worker_result, selected_gpu, gpu_attempts = _run_gpu_worker_with_retry(
        operation_name=f"coqui_tts:{model}",
        worker_slot=COQUI_TTS_WORKER_SLOTS,
        requested_device=device,
        visible_devices_env="SEME2E_COQUI_TTS_CUDA_VISIBLE_DEVICES",
        cancel_event=cancel_event,
        operation=run_worker,
        timeout_seconds=worker_timeout,
        explicit_device=bool(os.getenv("SEME2E_TTS_DEVICE", "").strip()),
        minimum_free_mib=max(
            0,
            _env_int(
                "SEME2E_CLONE_GPU_MIN_FREE_MIB",
                _env_int("SEME2E_GPU_MIN_FREE_MIB", 0),
            ),
        ),
        before_retry=cleanup_retry,
    )
    if runtime_context is not None and selected_gpu:
        runtime_context["gpuKey"] = selected_gpu
        if gpu_attempts:
            runtime_context["gpuAttempts"] = gpu_attempts
    return worker_result


def _is_cosyvoice_model(model: str) -> bool:
    return normalize_tts_model(model).lower() == "cosyvoice2:0.5b"


def _is_gpt_sovits_model(model: str) -> bool:
    return normalize_tts_model(model).lower() == "gpt-sovits:finetune"


def _keep_gpt_sovits_work_dir() -> bool:
    return any(
        str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}
        for name in (
            "SEME2E_KEEP_GPT_SOVITS_WORK_DIR",
            "SEME2E_GPT_SOVITS_KEEP_WORK_DIR",
            "SEME2E_DEBUG_KEEP_WORK_DIR",
        )
    )


def _remove_or_isolate_gpt_sovits_work_dir(work_dir: Path, *, label: str) -> dict[str, Any]:
    if not work_dir.exists():
        return {"status": "absent", "path": None}
    try:
        shutil.rmtree(work_dir)
        return {"status": "removed", "path": None}
    except Exception as remove_error:
        isolated_path = work_dir.with_name(f"{work_dir.name}.{label}-{uuid.uuid4().hex[:8]}")
        try:
            work_dir.replace(isolated_path)
        except Exception as isolate_error:
            raise RuntimeError(
                f"无法清理或隔离 GPT-SoVITS 临时目录 {work_dir}: {isolate_error}"
            ) from isolate_error
        try:
            shutil.rmtree(isolated_path)
            return {
                "status": "removed_after_isolation",
                "path": None,
                "initialError": _exception_diagnostics(remove_error),
            }
        except Exception as isolated_remove_error:
            return {
                "status": "isolated",
                "path": str(isolated_path),
                "initialError": _exception_diagnostics(remove_error),
                "cleanupError": _exception_diagnostics(isolated_remove_error),
            }


def _reset_gpt_sovits_work_dir(
    work_dir: Path,
    *,
    attempt_number: int,
    keep_work_dir: bool,
) -> dict[str, Any]:
    if not work_dir.exists():
        return {"status": "absent", "path": None}
    if keep_work_dir:
        retained_path = work_dir.with_name(
            f"{work_dir.name}.failed-attempt-{attempt_number}-{uuid.uuid4().hex[:8]}"
        )
        work_dir.replace(retained_path)
        return {"status": "retained_for_debug", "path": str(retained_path)}
    return _remove_or_isolate_gpt_sovits_work_dir(
        work_dir,
        label=f"failed-attempt-{attempt_number}",
    )


def _path_is_inside(path_value: str, parent: Path) -> bool:
    try:
        return Path(path_value).resolve().is_relative_to(parent.resolve())
    except (OSError, RuntimeError, ValueError):
        return False


def _sanitize_gpt_sovits_result_paths(
    payload: dict[str, Any],
    work_dir: Path,
    cleanup: dict[str, Any],
) -> dict[str, Any]:
    def sanitize_mapping(value: dict[str, Any]) -> None:
        for key, item in list(value.items()):
            if isinstance(item, dict):
                sanitize_mapping(item)
            elif isinstance(item, str) and _path_is_inside(item, work_dir):
                value[key] = None

    sanitize_mapping(payload)
    payload["workDirRetained"] = False
    payload["workDirCleanup"] = cleanup
    return payload


def _cosyvoice_clone_pair(
    original_reference: Path,
    protected_reference: Path,
    original_output: Path,
    protected_output: Path,
    *,
    text: str,
    original_prompt_text: str,
    protected_prompt_text: str,
    speed: float,
    device: str,
    cancel_event: Any | None = None,
    runtime_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status, reason, _ = _cosyvoice_model_status()
    if status != "available":
        raise RuntimeError(reason or "CosyVoice2 runtime is unavailable")
    worker = ROOT / "cosyvoice_worker.py"
    worker_timeout = _env_int("SEME2E_COSYVOICE_TIMEOUT_SECONDS", 900)

    def run_worker(
        worker_device: str,
        worker_env: Mapping[str, str] | None,
        _selected_gpu: str | None,
        attempt_timeout_seconds: float,
    ) -> dict[str, Any]:
        command = [
            str(COSYVOICE_PYTHON),
            str(worker),
            "--model-dir",
            str(COSYVOICE_MODEL_DIR),
            "--cosyvoice-repo",
            str(COSYVOICE_REPO_DIR),
            "--original-reference",
            str(original_reference),
            "--protected-reference",
            str(protected_reference),
            "--original-output",
            str(original_output),
            "--protected-output",
            str(protected_output),
            "--text",
            text,
            "--original-prompt-text",
            original_prompt_text,
            "--protected-prompt-text",
            protected_prompt_text,
            "--speed",
            str(speed),
            "--device",
            worker_device,
        ]
        environment = os.environ.copy()
        _apply_environment_overrides(environment, worker_env)
        completed = _run_cancellable_subprocess(
            command,
            cwd=str(COSYVOICE_REPO_DIR),
            env=environment,
            timeout_seconds=attempt_timeout_seconds,
            cancel_event=cancel_event,
        )
        marker = "VOICE_SHIELD_COSYVOICE_RESULT="
        result_line = next((line for line in reversed(completed.stdout.splitlines()) if line.startswith(marker)), None)
        if completed.returncode != 0 or result_line is None:
            stderr_tail = completed.stderr[-4000:].strip()
            stdout_tail = completed.stdout[-2000:].strip()
            raise RuntimeError(
                f"CosyVoice2 worker failed (exit={completed.returncode}): "
                f"{stderr_tail or stdout_tail or 'missing worker result'}"
            )
        return json.loads(result_line[len(marker):])

    def cleanup_retry(_attempt: dict[str, Any]) -> None:
        original_output.unlink(missing_ok=True)
        protected_output.unlink(missing_ok=True)

    worker_result, selected_gpu, gpu_attempts = _run_gpu_worker_with_retry(
        operation_name="cosyvoice2_clone",
        worker_slot=COSYVOICE_WORKER_SLOTS,
        requested_device=device,
        visible_devices_env="SEME2E_COSYVOICE_CUDA_VISIBLE_DEVICES",
        cancel_event=cancel_event,
        operation=run_worker,
        timeout_seconds=worker_timeout,
        explicit_device=bool(os.getenv("SEME2E_TTS_DEVICE", "").strip()),
        minimum_free_mib=max(
            0,
            _env_int(
                "SEME2E_CLONE_GPU_MIN_FREE_MIB",
                _env_int("SEME2E_GPU_MIN_FREE_MIB", 0),
            ),
        ),
        before_retry=cleanup_retry,
    )
    if runtime_context is not None and selected_gpu:
        runtime_context["gpuKey"] = selected_gpu
        if gpu_attempts:
            runtime_context["gpuAttempts"] = gpu_attempts
    return worker_result


def _gpt_sovits_clone_pair(
    original_reference: Path,
    protected_reference: Path,
    original_output: Path,
    protected_output: Path,
    *,
    original_transcript: str,
    protected_transcript: str,
    text: str,
    language: str,
    speed: float,
    device: str,
    cancel_event: Any | None = None,
    runtime_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status, reason, _ = _gpt_sovits_model_status()
    if status != "available":
        raise RuntimeError(reason or "GPT-SoVITS live fine-tuning runtime is unavailable")
    worker = ROOT / "gpt_sovits_live_finetune.py"
    work_dir = original_output.parent / "fine_tune"
    timeout_seconds = _env_int("SEME2E_GPT_SOVITS_TIMEOUT_SECONDS", 900)
    base_command = [
        str(GPT_SOVITS_PYTHON),
        str(worker),
        "--repo",
        str(GPT_SOVITS_REPO_DIR),
        "--python",
        str(GPT_SOVITS_PYTHON),
        "--work-dir",
        str(work_dir),
        "--original-audio",
        str(original_reference),
        "--protected-audio",
        str(protected_reference),
        "--original-transcript",
        original_transcript,
        "--protected-transcript",
        protected_transcript,
        "--text",
        text,
        "--language",
        language,
        "--speed",
        str(speed),
        "--original-output",
        str(original_output),
        "--protected-output",
        str(protected_output),
        "--device",
        "cuda:0",
        "--cnhubert",
        str(GPT_SOVITS_CNHUBERT),
        "--bert",
        str(GPT_SOVITS_BERT),
        "--pretrained-s1",
        str(GPT_SOVITS_PRETRAINED_S1),
        "--pretrained-s2g",
        str(GPT_SOVITS_PRETRAINED_S2G),
        "--pretrained-s2d",
        str(GPT_SOVITS_PRETRAINED_S2D),
        "--timeout",
        str(timeout_seconds),
        "--min-reference-seconds",
        "3.0",
        "--max-reference-seconds",
        "10.0",
    ]
    candidates = _gpt_sovits_gpu_candidates(device)
    deadline, retry_timeout = _gpu_retry_deadline(timeout_seconds * 3)
    attempts: list[dict[str, Any]] = []
    excluded_gpu_keys: set[str] = set()
    failure_states: dict[str, dict[str, Any]] = {}
    failure_counts: dict[str, int] = {}
    cooldown_deadlines: dict[str, float] = {}
    attempt_count = 0
    exhausted_rounds = 0
    last_resource_error: BaseException | None = None
    keep_work_dir = _keep_gpt_sovits_work_dir()
    initial_work_dir_cleanup: dict[str, Any] = {"status": "absent", "path": None}

    if work_dir.exists():
        try:
            initial_work_dir_cleanup = _reset_gpt_sovits_work_dir(
                work_dir,
                attempt_number=0,
                keep_work_dir=keep_work_dir,
            )
        except Exception as cleanup_error:
            initial_work_dir_cleanup = {
                "status": "cleanup_failed",
                "path": str(work_dir),
                "error": _exception_diagnostics(cleanup_error),
            }
            if runtime_context is not None:
                runtime_context["initialWorkDirCleanup"] = initial_work_dir_cleanup
            raise IsolatedWorkerError(
                "GPT-SoVITS 启动前无法清理旧微调目录",
                diagnostics={"initialWorkDirCleanup": initial_work_dir_cleanup},
            ) from cleanup_error
    if runtime_context is not None:
        runtime_context["initialWorkDirCleanup"] = initial_work_dir_cleanup

    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("TASK_CANCELLED")
            if time.monotonic() >= deadline:
                if last_resource_error is not None:
                    raise _gpu_retry_error(
                        "gpt_sovits_live_fine_tune",
                        retry_timeout,
                        attempts,
                        last_resource_error,
                    ) from last_resource_error
                raise RuntimeError(GPU_ACQUIRE_TIMEOUT_MESSAGE)

            remaining_candidates = _remaining_gpu_candidates(candidates, excluded_gpu_keys)
            if not remaining_candidates:
                try:
                    _wait_for_gpu_slot_change(
                        cancel_event,
                        deadline,
                        maximum_wait_seconds=_next_gpu_retry_wait_seconds(
                            candidates,
                            excluded_gpu_keys,
                            cooldown_deadlines,
                            exhausted_rounds,
                        ),
                    )
                except RuntimeError as wait_error:
                    if (
                        str(wait_error) == GPU_ACQUIRE_TIMEOUT_MESSAGE
                        and last_resource_error is not None
                    ):
                        raise _gpu_retry_error(
                            "gpt_sovits_live_fine_tune",
                            retry_timeout,
                            attempts,
                            last_resource_error,
                        ) from last_resource_error
                    raise
                recovered_keys = _recovered_gpu_keys(
                    candidates,
                    failure_states,
                    minimum_free_mib=_gpt_sovits_minimum_free_mib(),
                    cooldown_deadlines=cooldown_deadlines,
                )
                if recovered_keys:
                    excluded_gpu_keys.difference_update(recovered_keys)
                    for recovered_key in recovered_keys:
                        cooldown_deadlines.pop(recovered_key, None)
                    exhausted_rounds = 0
                else:
                    exhausted_rounds += 1
                continue

            leased_gpu: str | None = None
            retry_error: BaseException | None = None
            worker_result: dict[str, Any] | None = None
            _acquire_worker_slot(GPT_SOVITS_WORKER_SLOTS, cancel_event, deadline=deadline)
            try:
                with _gpt_sovits_gpu_resource_lease(
                    device,
                    cancel_event,
                    candidate_overrides=remaining_candidates,
                    deadline=deadline,
                ) as leased_gpu:
                    # GPT-SoVITS s2_train.py rewrites CUDA_VISIBLE_DEVICES from gpu_numbers.
                    # Pass the physical lease there, while the worker itself uses logical cuda:0.
                    leased_command = [
                        *base_command,
                        "--gpu-numbers",
                        leased_gpu,
                        "--cuda-visible-devices",
                        leased_gpu,
                    ]
                    environment = os.environ.copy()
                    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                    environment["CUDA_VISIBLE_DEVICES"] = leased_gpu
                    try:
                        remaining_attempt_seconds = deadline - time.monotonic()
                        if remaining_attempt_seconds <= 0:
                            if last_resource_error is not None:
                                raise _gpu_retry_error(
                                    "gpt_sovits_live_fine_tune",
                                    retry_timeout,
                                    attempts,
                                    last_resource_error,
                                ) from last_resource_error
                            raise RuntimeError(GPU_ACQUIRE_TIMEOUT_MESSAGE)
                        completed = _run_cancellable_subprocess(
                            leased_command,
                            cwd=str(GPT_SOVITS_REPO_DIR),
                            env=environment,
                            timeout_seconds=min(
                                max(0.01, float(timeout_seconds * 3)),
                                remaining_attempt_seconds,
                            ),
                            cancel_event=cancel_event,
                        )
                        marker = "VOICE_SHIELD_GPT_SOVITS_LIVE_RESULT="
                        result_line = next(
                            (
                                line
                                for line in reversed(completed.stdout.splitlines())
                                if line.startswith(marker)
                            ),
                            None,
                        )
                        if completed.returncode != 0 or result_line is None:
                            stderr_tail = completed.stderr[-4000:].strip()
                            stdout_tail = completed.stdout[-3000:].strip()
                            message = (
                                f"GPT-SoVITS live fine-tuning worker failed (exit={completed.returncode}): "
                                f"{stderr_tail or stdout_tail or 'missing worker result'}"
                            )
                            raise IsolatedWorkerError(
                                message,
                                diagnostics={
                                    "worker": str(worker),
                                    "returnCode": completed.returncode,
                                    "stdoutTail": stdout_tail,
                                    "stderrTail": stderr_tail,
                                    "gpu": leased_gpu,
                                    "workDir": str(work_dir),
                                },
                            )
                        worker_result = json.loads(result_line[len(marker):])
                    except Exception as exc:
                        if _gpu_resource_error_kind(exc) is None:
                            raise
                        retry_error = exc
            finally:
                _release_worker_slots([GPT_SOVITS_WORKER_SLOTS])

            if retry_error is None and worker_result is not None:
                if runtime_context is not None and leased_gpu:
                    runtime_context["gpuKey"] = leased_gpu
                    if attempts:
                        runtime_context["gpuAttempts"] = attempts
                if keep_work_dir:
                    worker_result["workDirRetained"] = True
                    worker_result["workDirCleanup"] = {
                        "status": "retained_for_debug",
                        "path": str(work_dir),
                    }
                else:
                    cleanup = _remove_or_isolate_gpt_sovits_work_dir(
                        work_dir,
                        label="completed",
                    )
                    _sanitize_gpt_sovits_result_paths(worker_result, work_dir, cleanup)
                worker_result["initialWorkDirCleanup"] = initial_work_dir_cleanup
                if attempts:
                    worker_result["gpuAttempts"] = attempts
                return worker_result

            if retry_error is None or leased_gpu is None:
                raise RuntimeError("GPT-SoVITS worker ended without a result")
            last_resource_error = retry_error
            attempt_count += 1
            attempt = _gpu_retry_attempt(
                "gpt_sovits_live_fine_tune",
                leased_gpu,
                retry_error,
                attempt_count,
            )
            canonical_key, failure_state = _gpu_failure_state(
                leased_gpu,
                str(attempt.get("reason") or "") or None,
            )
            attempt["freeMemoryMiBAfterFailure"] = failure_state.get("freeMemoryMiB")
            failure_count = failure_counts.get(canonical_key, 0) + 1
            failure_counts[canonical_key] = failure_count
            retry_cooldown_seconds = _gpu_retry_backoff_seconds(failure_count - 1)
            cooldown_until = time.monotonic() + retry_cooldown_seconds
            cooldown_deadlines[canonical_key] = cooldown_until
            cooldown_deadlines[leased_gpu] = cooldown_until
            attempt["retryCooldownSec"] = retry_cooldown_seconds
            _append_gpu_retry_attempt(attempts, attempt)
            failure_states[canonical_key] = failure_state
            failure_states[leased_gpu] = failure_state
            excluded_gpu_keys.update((leased_gpu, canonical_key))
            original_output.unlink(missing_ok=True)
            protected_output.unlink(missing_ok=True)
            attempt["workDirCleanup"] = _reset_gpt_sovits_work_dir(
                work_dir,
                attempt_number=attempt_count,
                keep_work_dir=keep_work_dir,
            )
    except Exception as exc:
        cleanup: dict[str, Any] | None = None
        if work_dir.exists() and not keep_work_dir:
            try:
                cleanup = _remove_or_isolate_gpt_sovits_work_dir(
                    work_dir,
                    label="failed",
                )
            except Exception as cleanup_error:
                cleanup = {
                    "status": "cleanup_failed",
                    "path": str(work_dir),
                    "error": _exception_diagnostics(cleanup_error),
                }
        if isinstance(exc, IsolatedWorkerError) and cleanup is not None:
            exc.diagnostics["workDirCleanup"] = cleanup
        if isinstance(exc, IsolatedWorkerError):
            exc.diagnostics["initialWorkDirCleanup"] = initial_work_dir_cleanup
        raise


def normalize_tts_model(value: str | None) -> str:
    raw = (value or os.getenv("SEME2E_TTS_MODEL") or "CosyVoice2-0.5B").strip()
    return _supported_tts_aliases().get(raw.lower(), raw)


def normalize_tts_language(value: str | None) -> str:
    raw = (value or os.getenv("SEME2E_TTS_LANGUAGE") or "zh-cn").strip().lower()
    aliases = {
        "zh": "zh-cn",
        "cn": "zh-cn",
        "zh_cn": "zh-cn",
        "chinese": "zh-cn",
    }
    return aliases.get(raw, raw)


def _tts_catalog_entry(model: str) -> dict[str, Any] | None:
    backend_value = normalize_tts_model(model)
    for item in SUPPORTED_TTS_MODELS:
        if str(item["backendValue"]).lower() == backend_value.lower():
            return item
    return None


def create_clone_voice(task_id: str, payload: dict[str, Any], progress_callback: ProgressCallback | None = None, cancel_event: Any | None = None) -> dict[str, Any]:
    text = str(payload.get("text") or "").strip()
    if not text:
        raise ValueError("text is required")
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("TASK_CANCELLED")
    original_path, protected_path, result = _task_audio_paths(task_id)
    model = normalize_tts_model(str(payload.get("model") or "") or None)
    language = normalize_tts_language(str(payload.get("language") or "") or None)
    speed = to_float(payload.get("speed")) or 1.0
    device = os.getenv("SEME2E_TTS_DEVICE") or os.getenv("SEME2E_API_DEVICE", "cpu")
    cosyvoice_model = _is_cosyvoice_model(model)
    gpt_sovits_model = _is_gpt_sovits_model(model)
    requires_reference_text = tts_model_requires_reference_text(model)
    prompt_text = str(payload.get("speakerPrompt") or "").strip()
    original_prompt_text = str(payload.get("originalSpeakerPrompt") or prompt_text).strip()
    protected_prompt_text = str(payload.get("protectedSpeakerPrompt") or prompt_text).strip()
    cosyvoice_status, cosyvoice_reason, _ = _cosyvoice_model_status()
    gpt_sovits_status, gpt_sovits_reason, _ = _gpt_sovits_model_status()
    catalog_entry = _tts_catalog_entry(model)
    catalog_status, catalog_reason, catalog_path = (
        _tts_catalog_status(catalog_entry, coqui_available=_module_available("TTS"))
        if catalog_entry is not None
        else ("unavailable", f"unsupported TTS model: {model}", None)
    )
    diagnostics = {
        "taskId": task_id,
        "ttsPackageAvailable": _module_available("TTS"),
        "cosyVoiceStatus": cosyvoice_status,
        "cosyVoiceReason": cosyvoice_reason,
        "gptSoVitsStatus": gpt_sovits_status,
        "gptSoVitsReason": gpt_sovits_reason,
        "fineTuneAudioDurationSec": read_wav_meta(original_path).get("durationSec"),
        "catalogStatus": catalog_status,
        "catalogReason": catalog_reason,
        "catalogPath": catalog_path,
        "model": model,
        "language": language,
        "speed": speed,
        "requiresReferenceText": requires_reference_text,
        "device": device,
        "originalReferencePath": str(original_path),
        "protectedReferencePath": str(protected_path),
    }
    if progress_callback is not None:
        progress_callback(progress=0.18, message="正在加载真实 TTS 克隆后端")
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("TASK_CANCELLED")
    if requires_reference_text and (not original_prompt_text or not protected_prompt_text):
        raise CloneBackendUnavailableError(
            "当前克隆模型需要原始参考音频和保护参考音频各自对应的标注文本。",
            task_id=task_id,
            diagnostics=diagnostics,
            reason="speaker_prompt_required",
        )
    if catalog_status != "available":
        downstream = result.setdefault("details", {}).setdefault("downstreamTts", {})
        unavailable_reason = catalog_reason or "TTS model is unavailable"
        downstream.update(
            {
                "enabled": False,
                "ttsModel": model,
                "status": "unavailable",
                "source": "CosyVoice2" if cosyvoice_model else "GPT-SoVITS" if gpt_sovits_model else "CoquiTTS.xtts_v2",
                "reason": unavailable_reason,
                "cloneText": text,
            }
        )
        save_result(TASK_DIR / task_id, result)
        raise CloneBackendUnavailableError(
            f"真实 TTS 语音克隆后端不可用：{unavailable_reason}",
            task_id=task_id,
            diagnostics=diagnostics,
            reason="dependency_missing",
        )

    clone_id = f"clone_{uuid.uuid4().hex[:10]}"
    clone_dir = TASK_DIR / task_id / "clones" / clone_id
    clone_dir.mkdir(parents=True, exist_ok=True)

    original_clone_path = clone_dir / f"{clone_id}_original_clone.wav"
    protected_clone_path = clone_dir / f"{clone_id}_protected_clone.wav"
    worker_result: dict[str, Any] | None = None
    clone_runtime: dict[str, Any] = {}
    evaluation_reference_path = original_path
    evaluation_protected_path = protected_path
    try:
        if progress_callback is not None:
            progress_callback(progress=0.32, message="正在从原始参考音频生成克隆音频")
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("TASK_CANCELLED")
        if cosyvoice_model:
            if progress_callback is not None:
                progress_callback(progress=0.46, message="CosyVoice2 正在一次加载中评测原始与保护参考音频")
            worker_result = _cosyvoice_clone_pair(
                original_path,
                protected_path,
                original_clone_path,
                protected_clone_path,
                text=text,
                original_prompt_text=original_prompt_text,
                protected_prompt_text=protected_prompt_text,
                speed=speed,
                device=device,
                cancel_event=cancel_event,
                runtime_context=clone_runtime,
            )
            diagnostics["workerResult"] = worker_result
            source_model = model
        elif gpt_sovits_model:
            if progress_callback is not None:
                progress_callback(progress=0.42, message="GPT-SoVITS 正在使用当前原始音频和保护音频进行现场微调")
            worker_result = _gpt_sovits_clone_pair(
                original_path,
                protected_path,
                original_clone_path,
                protected_clone_path,
                original_transcript=original_prompt_text,
                protected_transcript=protected_prompt_text,
                text=text,
                language=language,
                speed=speed,
                device=device,
                cancel_event=cancel_event,
                runtime_context=clone_runtime,
            )
            diagnostics["workerResult"] = worker_result
            source_model = model
        else:
            if progress_callback is not None:
                progress_callback(progress=0.46, message="Coqui TTS 正在独立子进程中加载模型并生成两组克隆音频")
            worker_result = _coqui_tts_clone_pair(
                original_path,
                protected_path,
                original_clone_path,
                protected_clone_path,
                text=text,
                model=model,
                language=language,
                speed=speed,
                device=device,
                task_id=task_id,
                clone_sub_id=str(payload.get("cloneSubId") or "") or None,
                cancel_event=cancel_event,
                runtime_context=clone_runtime,
            )
            diagnostics["workerResult"] = worker_result
            source_model = str(worker_result.get("sourceModel") or model)
            if progress_callback is not None:
                progress_callback(progress=0.82, message="Coqui TTS 独立子进程已生成原始与保护克隆音频")
        if progress_callback is not None:
            progress_callback(progress=0.84, message="克隆音频已生成，正在准备结果评估")
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("TASK_CANCELLED")
    except Exception as exc:
        if isinstance(exc, RuntimeError) and str(exc) == "TASK_CANCELLED":
            original_clone_path.unlink(missing_ok=True)
            protected_clone_path.unlink(missing_ok=True)
            raise
        if isinstance(exc, IsolatedWorkerError):
            diagnostics["workerDiagnostics"] = exc.diagnostics
        if clone_runtime:
            diagnostics["cloneRuntime"] = dict(clone_runtime)
        original_output_exists = original_clone_path.exists()
        protected_output_exists = protected_clone_path.exists()
        original_clone_path.unlink(missing_ok=True)
        protected_clone_path.unlink(missing_ok=True)
        diagnostics.update(
            {
                "exceptionType": type(exc).__name__,
                "exceptionMessage": str(exc),
                "stackTrace": traceback.format_exc(),
                "originalOutputExists": original_output_exists,
                "protectedOutputExists": protected_output_exists,
            }
        )
        raise CloneBackendUnavailableError(
            "真实 TTS 语音克隆失败：后端未生成克隆音频。",
            task_id=task_id,
            diagnostics=diagnostics,
            reason="tts_generation_failed",
        ) from exc

    if not original_clone_path.exists() or not protected_clone_path.exists():
        diagnostics.update(
            {
                "originalOutputExists": original_clone_path.exists(),
                "protectedOutputExists": protected_clone_path.exists(),
            }
        )
        raise CloneBackendUnavailableError(
            "真实 TTS 语音克隆失败：后端未生成克隆音频。",
            task_id=task_id,
            diagnostics=diagnostics,
            reason="output_file_missing",
        )

    clone_gpu_key = str(clone_runtime.get("gpuKey") or "").strip() or None
    if clone_gpu_key:
        diagnostics["workerGpu"] = clone_gpu_key
    clone_gpu_attempts = clone_runtime.get("gpuAttempts")
    if isinstance(clone_gpu_attempts, list) and clone_gpu_attempts:
        diagnostics["gpuAttempts"] = clone_gpu_attempts
    clone_runtime_diagnostics = dict(clone_runtime)
    if clone_gpu_key:
        clone_runtime_diagnostics["workerGpu"] = clone_gpu_key
    if isinstance(clone_gpu_attempts, list) and clone_gpu_attempts:
        clone_runtime_diagnostics["gpuAttempts"] = clone_gpu_attempts
    base_url = f"/api/artifacts/{task_id}/clones/{clone_id}"
    response = {
        "cloneId": clone_id,
        "cloneSubId": payload.get("cloneSubId"),
        "taskId": task_id,
        "status": "completed",
        "source": f"CosyVoice2:{source_model}" if cosyvoice_model else f"GPT-SoVITS:{source_model}:live" if gpt_sovits_model else f"CoquiTTS:{source_model}",
        "message": "GPT-SoVITS 已使用当前音频完成现场微调与生成。" if gpt_sovits_model else "真实 TTS 克隆音频已生成。",
        "request": {
            "text": text,
            "model": model,
            "asrModel": payload.get("asrModel") or os.getenv("SEME2E_CLONE_ASR_MODEL") or os.getenv("SEME2E_ASR_MODEL") or "openai-whisper:base",
            "language": language,
            "speed": speed,
            "speakerPrompt": prompt_text or None,
            "originalSpeakerPrompt": original_prompt_text or None,
            "protectedSpeakerPrompt": protected_prompt_text or None,
            "annotationSource": payload.get("annotationSource"),
            "annotationAsrSubId": payload.get("annotationAsrSubId"),
            "annotationAsrModel": payload.get("annotationAsrModel"),
            "annotationCreatedAt": payload.get("annotationCreatedAt"),
        },
        "originalCloneAudio": audio_meta(original_clone_path, f"{base_url}/{original_clone_path.name}"),
        "protectedCloneAudio": audio_meta(protected_clone_path, f"{base_url}/{protected_clone_path.name}"),
        "runtimeDiagnostics": clone_runtime_diagnostics,
    }
    if gpt_sovits_model and worker_result is not None:
        response["fineTune"] = {
            key: value
            for key, value in worker_result.items()
            if key not in {"cleanReferencePath", "protectedReferencePath"}
        }
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("TASK_CANCELLED")
    if progress_callback is not None:
        progress_callback(progress=0.86, message="正在生成克隆语音文本")
    clone_transcription = _transcribe_clone_pair_isolated(
        original_clone_path,
        protected_clone_path,
        payload,
        cancel_event=cancel_event,
        preferred_gpu=clone_gpu_key,
    )
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("TASK_CANCELLED")
    if progress_callback is not None:
        progress_callback(progress=0.91, message="正在计算克隆语义与语音质量")
    request_semantic = ((result.get("request") or {}).get("semantic") or {}) if isinstance(result.get("request"), dict) else {}
    clone_semantic_metrics = _compute_clone_semantic_isolated(
        original_clone_path,
        protected_clone_path,
        request_semantic if isinstance(request_semantic, dict) else {},
        cancel_event=cancel_event,
        preferred_gpu=clone_gpu_key,
    )
    clone_quality_metrics = _evaluate_dnsmos_pair_isolated(
        original_clone_path,
        protected_clone_path,
        cancel_event=cancel_event,
    )
    clone_runtime_diagnostics["cloneAsr"] = {
        key: clone_transcription.get(key)
        for key in ("status", "reason", "model", "gpu", "gpuAttempts", "diagnostics")
        if clone_transcription.get(key) is not None
    }
    clone_runtime_diagnostics["cloneSemantic"] = {
        key: clone_semantic_metrics.get(key)
        for key in ("status", "reason", "gpu", "gpuAttempts", "diagnostics")
        if clone_semantic_metrics.get(key) is not None
    }
    clone_runtime_diagnostics["cloneQuality"] = {
        key: clone_quality_metrics.get(key)
        for key in ("status", "reason", "provider", "diagnostics")
        if clone_quality_metrics.get(key) is not None
    }
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("TASK_CANCELLED")
    clone_eval = compute_clone_eval(
        evaluation_reference_path,
        original_clone_path,
        protected_clone_path,
        response,
        protected_audio_path=evaluation_protected_path,
        clone_transcription=clone_transcription,
        semantic_metrics=clone_semantic_metrics,
        quality_metrics=clone_quality_metrics,
    )
    clone_eval["runtimeDiagnostics"] = clone_runtime_diagnostics
    clone_eval_sources = clone_eval.get("_metricSources") or {}
    response["cloneEval"] = clone_eval
    _sync_clone_eval_fields(response, clone_eval)
    response.update(
        {
            "directSimilarity": clone_eval.get("directSimilarity"),
            "originalSimilarity": clone_eval.get("originalSimilarity"),
            "protectedSimilarity": clone_eval.get("protectedSimilarity"),
            "similarityDropRate": clone_eval.get("similarityDropRate"),
            "embeddingDistanceBefore": clone_eval.get("embeddingDistanceBefore"),
            "embeddingDistanceAfter": clone_eval.get("embeddingDistanceAfter"),
            "embeddingDistanceDelta": clone_eval.get("embeddingDistanceDelta"),
            "embeddingDistanceIncreaseRate": clone_eval.get("embeddingDistanceIncreaseRate"),
            "cloneIdentityScore": clone_eval.get("cloneIdentityScore"),
            "identityBaselineWeight": clone_eval.get("identityBaselineWeight"),
            "cloneSemanticScore": clone_eval.get("cloneSemanticScore"),
            "semanticBaselineWeight": clone_eval.get("semanticBaselineWeight"),
            "cloneQualityRawScore": clone_eval.get("cloneQualityRawScore"),
            "cloneQualityRelevance": clone_eval.get("cloneQualityRelevance"),
            "cloneQualityScore": clone_eval.get("cloneQualityScore"),
            "qualityBaselineWeight": clone_eval.get("qualityBaselineWeight"),
            "cloneConfidenceBefore": clone_eval.get("cloneConfidenceBefore"),
            "cloneConfidenceAfter": clone_eval.get("cloneConfidenceAfter"),
            "cloneConfidenceDropRate": clone_eval.get("cloneConfidenceDropRate"),
            "cloneRadar": clone_eval.get("cloneRadar"),
            "cloneTrend": clone_eval.get("cloneTrend"),
            "cloneDefenseScore": clone_eval.get("cloneDefenseScore"),
            "createdAt": clone_eval.get("createdAt"),
        }
    )

    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("TASK_CANCELLED")
    with RESULT_WRITE_LOCK:
        result_path = TASK_DIR / task_id / "result.json"
        latest_result = load_result(task_id) if result_path.exists() else result
        clones = latest_result.setdefault("cloneResults", [])
        clone_sub_id = response.get("cloneSubId")
        if clone_sub_id:
            clones[:] = [item for item in clones if item.get("cloneSubId") != clone_sub_id]
        clones.append(response)
        downstream = latest_result.setdefault("details", {}).setdefault("downstreamTts", {})
        latest_result.setdefault("details", {})["cloneEval"] = clone_eval
        downstream.update(
            {
                "enabled": True,
                "ttsModel": response["request"]["model"],
                "status": "computed",
                "source": response["source"],
                "lastCloneId": clone_id,
                "cloneText": text,
                "simCleanClone": clone_eval.get("originalSimilarity"),
                "simProtectedClone": clone_eval.get("protectedSimilarity"),
                "simDropRate": clone_eval.get("similarityDropRate"),
                "fineTune": response.get("fineTune"),
            }
        )
        metric_sources = latest_result.setdefault("summary", {}).setdefault("metricSources", {})
        metric_sources.update(clone_eval_sources)
        refresh_result_scores(latest_result)
        latest_result["updatedAt"] = utc_now_iso()
        save_result(TASK_DIR / task_id, latest_result)
    return response
