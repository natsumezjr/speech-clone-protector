from __future__ import annotations

"""Isolated Coqui TTS clone worker.

The worker reads exactly one JSON object from stdin and writes exactly one JSON
object to stdout.  Third-party diagnostic output is redirected to stderr so the
caller can parse stdout without marker scanning.

Required request fields (camelCase and snake_case aliases are accepted):

* ``model``
* ``originalReferencePath`` / ``original_reference_path``
* ``protectedReferencePath`` / ``protected_reference_path``
* ``originalOutputPath`` / ``original_output_path``
* ``protectedOutputPath`` / ``protected_output_path``
* ``text``

Optional fields include ``modelPath``, ``configPath``, ``ttsHome``,
``language``, ``speed``, and ``device``.  A local model path is required for an
unknown model name; known VoiceShield Coqui models are resolved below TTS_HOME.
"""

import contextlib
import json
import os
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, TextIO


ROOT = Path(__file__).resolve().parent
DEFAULT_TTS_HOME = ROOT / "checkpoints" / "tts"

MODEL_SPECS: dict[str, dict[str, str]] = {
    "xtts_v2": {
        "cacheName": "tts_models--multilingual--multi-dataset--xtts_v2",
        "kind": "xtts",
    },
    "tts_models/multilingual/multi-dataset/xtts_v1.1": {
        "cacheName": "tts_models--multilingual--multi-dataset--xtts_v1.1",
        "kind": "xtts",
    },
    "tts_models/multilingual/multi-dataset/your_tts": {
        "cacheName": "tts_models--multilingual--multi-dataset--your_tts",
        "kind": "coqui",
    },
}

MODEL_ALIASES = {
    "default": "xtts_v2",
    "xtts": "xtts_v2",
    "xtts-v2": "xtts_v2",
    "xtts_v2": "xtts_v2",
    "coquitts:xtts_v2": "xtts_v2",
    "xtts-v1": "tts_models/multilingual/multi-dataset/xtts_v1.1",
    "xtts_v1": "tts_models/multilingual/multi-dataset/xtts_v1.1",
    "xtts-v1.1": "tts_models/multilingual/multi-dataset/xtts_v1.1",
    "xtts_v1.1": "tts_models/multilingual/multi-dataset/xtts_v1.1",
    "coquitts:xtts_v1.1": "tts_models/multilingual/multi-dataset/xtts_v1.1",
    "yourtts": "tts_models/multilingual/multi-dataset/your_tts",
    "your-tts": "tts_models/multilingual/multi-dataset/your_tts",
    "your_tts": "tts_models/multilingual/multi-dataset/your_tts",
    "coquitts:your_tts": "tts_models/multilingual/multi-dataset/your_tts",
}


class RequestError(ValueError):
    """The worker request is missing or contains an invalid value."""


def _first(payload: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    return None


def _required_text(payload: dict[str, Any], *names: str) -> str:
    value = _first(payload, *names)
    normalized = str(value or "").strip()
    if not normalized:
        raise RequestError(f"missing required field: {names[0]}")
    return normalized


def _optional_path(payload: dict[str, Any], *names: str) -> Path | None:
    value = _first(payload, *names)
    if value is None or not str(value).strip():
        return None
    return Path(str(value)).expanduser().resolve(strict=False)


def _required_path(payload: dict[str, Any], *names: str) -> Path:
    value = _optional_path(payload, *names)
    if value is None:
        raise RequestError(f"missing required field: {names[0]}")
    return value


def _normalize_model(value: str) -> str:
    normalized = value.strip()
    return MODEL_ALIASES.get(normalized.lower(), normalized)


def _validate_input_file(path: Path, field_name: str) -> None:
    if not path.is_file():
        raise RequestError(f"{field_name} does not exist or is not a file: {path}")


def _validate_checkpoint(path: Path) -> None:
    if not path.is_file():
        raise RequestError(f"model checkpoint does not exist: {path}")
    if path.stat().st_size < 1024 * 1024:
        raise RequestError(f"model checkpoint is incomplete: {path}")
    try:
        with path.open("rb") as checkpoint_file:
            signature = checkpoint_file.read(4)
        if signature.startswith(b"PK"):
            with zipfile.ZipFile(path) as checkpoint_zip:
                if not checkpoint_zip.namelist():
                    raise RequestError(f"model checkpoint archive is empty: {path}")
    except zipfile.BadZipFile as exc:
        raise RequestError(f"model checkpoint archive is invalid: {path}") from exc


def _resolve_model_files(
    payload: dict[str, Any],
    source_model: str,
) -> tuple[Path, Path, Path, str]:
    """Return (model argument, config path, model directory, model kind)."""

    explicit_model = _optional_path(
        payload,
        "modelPath",
        "model_path",
        "modelDir",
        "model_dir",
    )
    explicit_config = _optional_path(payload, "configPath", "config_path")
    spec = MODEL_SPECS.get(source_model)

    if explicit_model is None:
        if spec is None:
            raise RequestError(
                f"modelPath is required for unsupported local Coqui model: {source_model}"
            )
        tts_home = _optional_path(payload, "ttsHome", "tts_home")
        if tts_home is None:
            tts_home = Path(
                os.getenv("TTS_HOME", str(DEFAULT_TTS_HOME))
            ).expanduser().resolve(strict=False)
        explicit_model = tts_home / spec["cacheName"]

    if explicit_model.is_dir():
        model_dir = explicit_model
        checkpoint = model_dir / "model.pth"
        if not checkpoint.is_file():
            checkpoint = model_dir / "model_file.pth"
    else:
        checkpoint = explicit_model
        model_dir = checkpoint.parent

    config_path = explicit_config or model_dir / "config.json"
    if not config_path.is_file():
        raise RequestError(f"model config does not exist: {config_path}")
    _validate_checkpoint(checkpoint)

    model_kind = (spec or {}).get("kind") or (
        "xtts" if "xtts" in source_model.lower() else "coqui"
    )
    model_argument = model_dir if model_kind == "xtts" else checkpoint
    return model_argument, config_path, model_dir, model_kind


def _portable_config_path(
    model_dir: Path,
    config_path: Path,
    temporary_dir: Path,
) -> Path:
    """Rewrite stale absolute asset paths into a temporary local config."""

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

    portable_path = temporary_dir / "config.json"
    portable_path.write_text(
        json.dumps(rewritten, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return portable_path


def _safe_output_status(path: Path) -> dict[str, Any]:
    try:
        exists = path.is_file()
        return {
            "path": str(path),
            "exists": exists,
            "sizeBytes": path.stat().st_size if exists else 0,
        }
    except OSError as exc:
        return {
            "path": str(path),
            "exists": False,
            "sizeBytes": 0,
            "inspectionError": f"{type(exc).__name__}: {exc}",
        }


def _audio_metadata(path: Path, generation_sec: float) -> dict[str, Any]:
    import soundfile as sf

    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Coqui TTS did not generate a non-empty audio file: {path}")
    info = sf.info(str(path))
    return {
        "path": str(path),
        "generationSec": round(generation_sec, 4),
        "sizeBytes": path.stat().st_size,
        "sampleRate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "durationSec": round(float(info.duration), 4),
        "format": str(info.format),
        "subtype": str(info.subtype),
    }


def _generate_one(
    tts: Any,
    reference_path: Path,
    output_path: Path,
    *,
    text: str,
    language: str,
    speed: float,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "text": text,
        "speaker_wav": str(reference_path),
        "language": language,
        "file_path": str(output_path),
    }
    if speed:
        kwargs["speed"] = speed

    started = time.perf_counter()
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
    return _audio_metadata(output_path, time.perf_counter() - started)


def _model_runtime_metadata(tts: Any) -> dict[str, Any]:
    synthesizer = getattr(tts, "synthesizer", None)
    model = getattr(synthesizer, "tts_model", None)
    if model is None or not hasattr(model, "parameters"):
        return {"modelClass": None, "modelDevice": None, "modelDtype": None}
    try:
        parameter = next(model.parameters())
    except (StopIteration, TypeError):
        return {
            "modelClass": f"{type(model).__module__}.{type(model).__name__}",
            "modelDevice": None,
            "modelDtype": None,
        }
    return {
        "modelClass": f"{type(model).__module__}.{type(model).__name__}",
        "modelDevice": str(parameter.device),
        "modelDtype": str(parameter.dtype),
    }


def _run_worker(payload: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    state["stage"] = "validate_request"
    model_value = _required_text(payload, "model")
    source_model = _normalize_model(model_value)
    state["sourceModel"] = source_model

    original_reference = _required_path(
        payload,
        "originalReferencePath",
        "original_reference_path",
        "originalReference",
        "original_reference",
    )
    protected_reference = _required_path(
        payload,
        "protectedReferencePath",
        "protected_reference_path",
        "protectedReference",
        "protected_reference",
    )
    original_output = _required_path(
        payload,
        "originalOutputPath",
        "original_output_path",
        "originalOutput",
        "original_output",
    )
    protected_output = _required_path(
        payload,
        "protectedOutputPath",
        "protected_output_path",
        "protectedOutput",
        "protected_output",
    )
    text = _required_text(payload, "text")
    language = str(_first(payload, "language") or "en").strip() or "en"
    requested_device = str(_first(payload, "device") or "cpu").strip() or "cpu"
    try:
        speed = float(_first(payload, "speed") or 1.0)
    except (TypeError, ValueError) as exc:
        raise RequestError("speed must be a number") from exc
    if speed <= 0:
        raise RequestError("speed must be greater than zero")

    _validate_input_file(original_reference, "originalReferencePath")
    _validate_input_file(protected_reference, "protectedReferencePath")
    if original_output == protected_output:
        raise RequestError("originalOutputPath and protectedOutputPath must differ")
    input_paths = {original_reference, protected_reference}
    if original_output in input_paths or protected_output in input_paths:
        raise RequestError("output paths must not overwrite either reference audio file")

    state["outputs"] = {
        "original": _safe_output_status(original_output),
        "protected": _safe_output_status(protected_output),
    }
    state["stage"] = "resolve_model"
    model_argument, config_path, model_dir, model_kind = _resolve_model_files(
        payload,
        source_model,
    )

    state["stage"] = "import_runtime"
    import soundfile as sf
    import torch
    import torchaudio.functional as ta_functional
    from TTS.api import TTS
    from TTS.tts.models import xtts as xtts_module

    del sf  # The module is imported here so import failures use this stage.

    requested_cuda = requested_device.lower().startswith("cuda")
    if requested_cuda and not torch.cuda.is_available():
        requested_device = "cpu"
        device_fallback = "CUDA was requested but is unavailable"
    else:
        device_fallback = None

    original_torch_load = torch.load
    original_xtts_load_audio = xtts_module.load_audio

    def torch_load_for_trusted_tts_checkpoint(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    def load_audio_without_torchcodec(audiopath: str, sampling_rate: int) -> Any:
        import soundfile as soundfile

        audio_np, loaded_sr = soundfile.read(
            str(audiopath),
            dtype="float32",
            always_2d=True,
        )
        audio = torch.from_numpy(audio_np.T)
        if audio.size(0) != 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if loaded_sr != sampling_rate:
            audio = ta_functional.resample(audio, loaded_sr, sampling_rate)
        if torch.any(audio > 10) or not torch.any(audio < 0):
            print(
                f"Audio range warning for {audiopath}: "
                f"max={audio.max()} min={audio.min()}",
                file=sys.stderr,
                flush=True,
            )
        audio.clip_(-1, 1)
        return audio

    worker_started = time.perf_counter()
    torch.load = torch_load_for_trusted_tts_checkpoint
    xtts_module.load_audio = load_audio_without_torchcodec
    try:
        with tempfile.TemporaryDirectory(prefix="voiceshield-coqui-") as temp_name:
            portable_config = _portable_config_path(
                model_dir,
                config_path,
                Path(temp_name),
            )
            state["stage"] = "load_model"
            load_started = time.perf_counter()
            tts = TTS(
                model_path=str(model_argument),
                config_path=str(portable_config),
                progress_bar=False,
            )
            load_sec = time.perf_counter() - load_started

            state["stage"] = "move_model"
            actual_device = requested_device
            if hasattr(tts, "to"):
                try:
                    moved = tts.to(requested_device)
                    if moved is not None:
                        tts = moved
                except Exception as exc:
                    if requested_device.lower() == "cpu":
                        raise
                    print(
                        f"Could not move Coqui TTS to {requested_device}; "
                        f"falling back to CPU: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    moved = tts.to("cpu")
                    if moved is not None:
                        tts = moved
                    actual_device = "cpu"
                    device_fallback = f"{type(exc).__name__}: {exc}"

            if getattr(tts, "model_name", None) is None:
                tts.model_name = source_model
            if (
                getattr(tts, "config", None) is None
                and getattr(tts, "synthesizer", None) is not None
            ):
                tts.config = getattr(tts.synthesizer, "tts_config", None)
            if model_kind != "xtts":
                tts._check_arguments = lambda *args, **kwargs: None

            runtime_metadata = _model_runtime_metadata(tts)
            state["stage"] = "generate_original"
            original_result = _generate_one(
                tts,
                original_reference,
                original_output,
                text=text,
                language=language,
                speed=speed,
            )
            state["outputs"]["original"] = _safe_output_status(original_output)

            state["stage"] = "generate_protected"
            protected_result = _generate_one(
                tts,
                protected_reference,
                protected_output,
                text=text,
                language=language,
                speed=speed,
            )
            state["outputs"]["protected"] = _safe_output_status(protected_output)
    finally:
        xtts_module.load_audio = original_xtts_load_audio
        torch.load = original_torch_load

    state["stage"] = "completed"
    result: dict[str, Any] = {
        "ok": True,
        "sourceModel": source_model,
        "requestedModel": model_value,
        "modelPath": str(model_argument),
        "configPath": str(config_path),
        "requestedDevice": str(_first(payload, "device") or "cpu"),
        "device": actual_device,
        "deviceFallback": device_fallback,
        "loadSec": round(load_sec, 4),
        "totalSec": round(time.perf_counter() - worker_started, 4),
        "original": original_result,
        "protected": protected_result,
        **runtime_metadata,
    }
    for key in ("requestId", "taskId", "cloneSubId"):
        if key in payload:
            result[key] = payload[key]
    return result


def _error_response(
    exc: BaseException,
    state: dict[str, Any],
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    stage = str(state.get("stage") or "parse_request")
    response: dict[str, Any] = {
        "ok": False,
        "sourceModel": state.get("sourceModel"),
        "error": {
            "code": (
                "INVALID_REQUEST"
                if isinstance(exc, (RequestError, json.JSONDecodeError))
                else "COQUI_TTS_WORKER_FAILED"
            ),
            "stage": stage,
            "exceptionType": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
        "outputs": state.get("outputs") or {},
    }
    if isinstance(payload, dict):
        for key in ("requestId", "taskId", "cloneSubId"):
            if key in payload:
                response[key] = payload[key]
    return response


def _write_response(stream: TextIO, response: dict[str, Any]) -> None:
    stream.write(json.dumps(response, ensure_ascii=False, allow_nan=False))
    stream.write("\n")
    stream.flush()


def main() -> int:
    response_stream = sys.stdout
    state: dict[str, Any] = {"stage": "parse_request"}
    payload: dict[str, Any] | None = None
    try:
        raw_request = sys.stdin.read()
        if not raw_request.strip():
            raise RequestError("stdin did not contain a JSON request")
        decoded = json.loads(raw_request)
        if not isinstance(decoded, dict):
            raise RequestError("JSON request must be an object")
        payload = decoded
        # Coqui TTS emits informational messages with print().  Redirect them to
        # stderr so stdout remains a machine-readable, one-line JSON channel.
        with contextlib.redirect_stdout(sys.stderr):
            response = _run_worker(payload, state)
        _write_response(response_stream, response)
        return 0
    except BaseException as exc:
        response = _error_response(exc, state, payload)
        _write_response(response_stream, response)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
