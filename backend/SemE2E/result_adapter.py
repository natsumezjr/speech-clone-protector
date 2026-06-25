from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import traceback
import uuid
import wave
import importlib.util
from pathlib import Path
from typing import Any, Callable

import numpy as np

from result_schema import default_chains, empty_charts, empty_details, empty_primary_metrics, utc_now_iso

ProgressCallback = Callable[..., None]

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


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or default


FORMAL_EPSILON = 8 / 255
FORMAL_STEPS = 50
FORMAL_WEIGHT_FEATURE = 500.0
FORMAL_WEIGHT_SEMANTIC = 100.0
FORMAL_WEIGHT_PSY = 1.0e-5
FORMAL_WEIGHT_L2 = 0.1
FORMAL_SEMANTIC_ENCODERS = ["S3", "HuBERT", "Whisper", "MFCC"]
FORMAL_TIMBRE_ENCODERS = ["VITS", "GPT-SoVITS", "MFCC", "WavLM", "CosyVoice", "StyleTTS2"]
FORMAL_ASR_MODEL = "openai/whisper-small"
FORMAL_TTS_BACKEND = "tts_models/multilingual/multi-dataset/xtts_v2"
SUPPORTED_TTS_MODELS = [
    {
        "label": "XTTS-v2",
        "value": "XTTS-v2",
        "backendValue": "tts_models/multilingual/multi-dataset/xtts_v2",
        "cacheName": "tts_models--multilingual--multi-dataset--xtts_v2",
        "aliases": ["default", "xtts", "xtts-v2", "xtts_v2", "coquitts:xtts_v2"],
        "languages": ["en", "zh-cn"],
        "description": "Coqui XTTS-v2 voice cloning backend.",
    },
    {
        "label": "XTTS-v1.1",
        "value": "XTTS-v1.1",
        "backendValue": "tts_models/multilingual/multi-dataset/xtts_v1.1",
        "cacheName": "tts_models--multilingual--multi-dataset--xtts_v1.1",
        "aliases": ["xtts-v1.1", "xtts_v1.1", "xtts-v1", "xtts_v1", "coquitts:xtts_v1.1"],
        "languages": ["en", "zh-cn"],
        "description": "Coqui XTTS-v1.1 cross-language voice cloning backend.",
    },
    {
        "label": "YourTTS",
        "value": "YourTTS",
        "backendValue": "tts_models/multilingual/multi-dataset/your_tts",
        "cacheName": "tts_models--multilingual--multi-dataset--your_tts",
        "aliases": ["your-tts", "your_tts", "coquitts:your_tts"],
        "languages": ["en"],
        "description": "Coqui YourTTS voice cloning backend.",
    },
]


def _tts_cache_dir() -> Path:
    return Path(os.getenv("TTS_HOME", str(PROJECT_TTS_CACHE_DIR)))


def _tts_model_cache_status(cache_name: str) -> tuple[str, str | None, str]:
    path = _tts_cache_dir() / cache_name
    config_path = path / "config.json"
    checkpoint_files = list(path.glob("*.pth")) if path.exists() else []
    if config_path.exists() and checkpoint_files:
        return "available", None, str(path)
    return "download_required", f"missing local Coqui TTS cache: {path}", str(path)


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


def _checkpoint_status() -> dict[str, Any]:
    required = {
        "VITS": ROOT / "checkpoints" / "VITS" / "pretrained_ljs.pth",
        "GPT-SoVITS": ROOT / "checkpoints" / "GSV" / "base_models" / "gsv-v2final-pretrained" / "s2G2333k.pth",
        "CosyVoiceTokenizer": ROOT / "checkpoints" / "CosyVoice" / "speech_tokenizer_v1.onnx",
        "CosyVoiceCAMPP": ROOT / "checkpoints" / "CosyVoice" / "base_models" / "CosyVoice-300M" / "campplus.onnx",
        "StyleTTS2Config": ROOT / "checkpoints" / "StyleTTS2" / "base_models" / "config.yml",
        "StyleTTS2Checkpoint": ROOT / "checkpoints" / "StyleTTS2" / "base_models" / "epochs_2nd_00020.pth",
        "StyleTTS2ASR": ROOT / "tts_models" / "styletts2" / "Utils" / "ASR" / "epoch_00080.pth",
        "StyleTTS2JDC": ROOT / "tts_models" / "styletts2" / "Utils" / "JDC" / "bst.t7",
        "StyleTTS2PLBERT": ROOT / "tts_models" / "styletts2" / "Utils" / "PLBERT" / "step_1000000.t7",
        "ESpeakNG": ROOT / "vendor" / "espeak-ng" / "libespeak-ng.dll",
        "ESpeakNGData": ROOT / "vendor" / "espeak-ng" / "espeak-ng-data",
        "ASRWhisperSmall": ROOT / "checkpoints" / "asr" / "openai-whisper-small" / "config.json",
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
    missing = [name for name, item in entries.items() if not item["exists"]]
    return {
        "missing": missing,
        "required": {name: item["path"] for name, item in entries.items()},
        "entries": entries,
    }


def _component_available(checkpoints: dict[str, Any], names: list[str]) -> tuple[str, str | None]:
    missing = [name for name in names if name in checkpoints["missing"]]
    if missing:
        return "unavailable", "missing files: " + ", ".join(missing)
    return "available", None


def _model_option(label: str, value: str, backend_value: str, branch: str, *, status: str = "available", reason: str | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "label": label,
        "value": value,
        "backendValue": backend_value,
        "branch": branch,
        "status": status,
        "reason": reason,
        **extra,
    }


def _profile_defaults(profile: str, *, steps: int, semantic_encoders: list[str], timbre_encoders: list[str]) -> dict[str, Any]:
    return {
        "profile": profile,
        "realProtect": True,
        "mode": "standard",
        "targets": ["semantic", "timbre"],
        "semantic": {
            "enabled": True,
            "asrModel": FORMAL_ASR_MODEL,
            "asrModels": [FORMAL_ASR_MODEL],
            "encoders": semantic_encoders,
            "tokenizerPath": "checkpoints/CosyVoice/speech_tokenizer_v1.onnx",
            "hubertPath": "facebook/hubert-base-ls960",
            "whisperPath": "openai/whisper-small",
            "weightSemantic": FORMAL_WEIGHT_SEMANTIC,
        },
        "timbre": {
            "enabled": True,
            "mode": "untargeted",
            "encoders": timbre_encoders,
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
    checkpoints = _checkpoint_status()
    vits_status, vits_reason = _component_available(checkpoints, ["VITS"])
    gsv_status, gsv_reason = _component_available(checkpoints, ["GPT-SoVITS"])
    s3_status, s3_reason = _component_available(checkpoints, ["CosyVoiceTokenizer"])
    cosy_status, cosy_reason = _component_available(checkpoints, ["CosyVoiceCAMPP"])
    style_status, style_reason = _component_available(checkpoints, ["StyleTTS2Config", "StyleTTS2Checkpoint", "StyleTTS2ASR", "StyleTTS2JDC", "StyleTTS2PLBERT", "ESpeakNG", "ESpeakNGData"])
    transformers_available = _module_available("transformers")
    whisper_small_available = transformers_available and bool((checkpoints.get("entries") or {}).get("ASRWhisperSmall", {}).get("exists"))
    whisper_small_reason = None if whisper_small_available else "missing local Whisper Small checkpoint" if transformers_available else "transformers not installed"
    whisper_package_available = _module_available("whisper")
    funasr_available = _module_available("funasr")
    tts_available = _module_available("TTS")

    semantic_options = [
        _model_option("S3 Tokenizer Encoder", "S3", "s3", "semantic", status=s3_status, reason=s3_reason),
        _model_option("HuBERT", "HuBERT", "hubert", "semantic", status="available" if transformers_available else "unavailable", reason=None if transformers_available else "transformers not installed", defaultPath="facebook/hubert-base-ls960"),
        _model_option("Whisper Encoder", "Whisper", "whisper", "semantic", status="available" if transformers_available else "unavailable", reason=None if transformers_available else "transformers not installed", defaultPath="openai/whisper-small"),
        _model_option("MFCC", "MFCC", "mfcc", "semantic"),
    ]
    timbre_options = [
        _model_option("VITS Posterior Encoder", "VITS", "vits", "timbre", status=vits_status, reason=vits_reason),
        _model_option("GPT-SoVITS Encoder", "GPT-SoVITS", "gsv", "timbre", status=gsv_status, reason=gsv_reason),
        _model_option("MFCC", "MFCC", "mfcc", "timbre"),
        _model_option("WavLM", "WavLM", "wavlm", "timbre", status="available" if transformers_available else "unavailable", reason=None if transformers_available else "transformers not installed"),
        _model_option("CosyVoice CAM++", "CosyVoice", "cosyvoice", "timbre", status=cosy_status, reason=cosy_reason),
        _model_option("StyleTTS2 Style Encoder", "StyleTTS2", "style", "timbre", status=style_status, reason=style_reason),
    ]
    asr_options = [
        _model_option("Whisper Small", "openai/whisper-small", str(ROOT / "checkpoints" / "asr" / "openai-whisper-small"), "asr", status="available" if whisper_small_available else "unavailable", reason=whisper_small_reason, backend="transformers", localPath=str(ROOT / "checkpoints" / "asr" / "openai-whisper-small")),
        _model_option("OpenAI Whisper Tiny", "openai-whisper:tiny", "openai-whisper:tiny", "asr", status="available" if whisper_package_available else "unavailable", reason=None if whisper_package_available else "openai-whisper package not installed", backend="openai-whisper"),
        _model_option("OpenAI Whisper Base", "openai-whisper:base", "openai-whisper:base", "asr", status="available" if whisper_package_available else "unavailable", reason=None if whisper_package_available else "openai-whisper package not installed", backend="openai-whisper"),
        _model_option("FunASR Paraformer", "funasr:paraformer-zh", "funasr:paraformer-zh", "asr", status="available" if funasr_available else "unavailable", reason=None if funasr_available else "funasr not installed", backend="funasr"),
    ]
    tts_options = []
    for item in SUPPORTED_TTS_MODELS:
        cache_status, cache_reason, cache_path = _tts_model_cache_status(str(item["cacheName"]))
        if not tts_available:
            status = "unavailable"
            reason = "Coqui TTS package is not installed"
        else:
            status = cache_status
            reason = cache_reason
        tts_options.append(
            _model_option(
                str(item["label"]),
                str(item["value"]),
                str(item["backendValue"]),
                "tts",
                status=status,
                reason=reason,
                backend="CoquiTTS",
                localPath=cache_path,
                languages=item.get("languages", []),
                description=item.get("description"),
            )
        )

    formal = _profile_defaults("formal", steps=FORMAL_STEPS, semantic_encoders=FORMAL_SEMANTIC_ENCODERS, timbre_encoders=FORMAL_TIMBRE_ENCODERS)
    fields = {
        "epsilon": {"label": "扰动强度 ε", "path": "optimization.epsilon", "default": round(FORMAL_EPSILON, 9), "min": 0.001, "max": 0.08, "step": 0.001, "unit": "waveform amplitude", "description": "正式默认值为 8/255。"},
        "steps": {"label": "优化步数", "path": "optimization.steps", "default": FORMAL_STEPS, "min": 1, "max": 500, "step": 1, "description": "默认 50，最大 500。"},
        "weightFeature": {"label": "Feature 权重", "path": "timbre.weightFeature", "default": FORMAL_WEIGHT_FEATURE, "min": 0, "max": 1000, "step": 1},
        "weightSemantic": {"label": "Semantic 权重", "path": "semantic.weightSemantic", "default": FORMAL_WEIGHT_SEMANTIC, "min": 0, "max": 500, "step": 1},
        "weightPsy": {"label": "心理声学权重", "path": "psychoacoustic.weightPsy", "default": FORMAL_WEIGHT_PSY, "min": 0, "max": 0.001, "step": 0.000001},
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
            "timbre": {"weightFeature": FORMAL_WEIGHT_FEATURE},
            "psychoacoustic": {"weightPsy": FORMAL_WEIGHT_PSY},
            "optimization": {"epsilon": round(FORMAL_EPSILON, 9), "steps": FORMAL_STEPS, "weightL2": FORMAL_WEIGHT_L2},
        },
        "strong": {
            "profile": "formal",
            "semantic": {"weightSemantic": 140.0},
            "timbre": {"weightFeature": 675.0},
            "psychoacoustic": {"weightPsy": 0.0000075},
            "optimization": {"epsilon": round(FORMAL_EPSILON, 9), "steps": FORMAL_STEPS, "weightL2": 0.08},
        },
        "high_fidelity": {
            "profile": "formal",
            "semantic": {"weightSemantic": 80.0},
            "timbre": {"weightFeature": 400.0},
            "psychoacoustic": {"weightPsy": 0.000015},
            "optimization": {"epsilon": 0.020392156, "steps": FORMAL_STEPS, "weightL2": 0.15},
        },
        "custom": {
            "profile": "formal",
            "semantic": {"weightSemantic": FORMAL_WEIGHT_SEMANTIC},
            "timbre": {"weightFeature": FORMAL_WEIGHT_FEATURE},
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
            {"value": "formal", "label": "正式保护", "description": "使用论文/原始后端默认参数，steps=50。"},
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
        },
        "formSchema": form_schema,
        "constraints": {
            "maxAudioSizeBytes": _env_int("SEME2E_API_MAX_AUDIO_SIZE_BYTES", 200 * 1024 * 1024),
        },
        "clone": {
            "defaults": {
                "model": normalize_tts_model(os.getenv("SEME2E_API_DEFAULT_TTS_MODEL")),
                "backendValue": normalize_tts_model(os.getenv("SEME2E_API_DEFAULT_TTS_MODEL")),
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
            {"value": "timbre", "label": "语音特征防护", "description": "阻断特征"},
            {"value": "joint", "label": "联合防护", "description": "双重防护"},
        ],
        "modePresets": mode_presets,
    }


def diagnose_capabilities() -> dict[str, Any]:
    device = os.getenv("SEME2E_API_DEVICE", "cpu")
    checkpoints = _checkpoint_status()
    missing = checkpoints["missing"]
    protect_required = [
        "VITS",
        "GPT-SoVITS",
        "CosyVoiceTokenizer",
        "CosyVoiceCAMPP",
        "StyleTTS2Config",
        "StyleTTS2Checkpoint",
        "StyleTTS2ASR",
        "StyleTTS2JDC",
        "StyleTTS2PLBERT",
        "ESpeakNG",
        "ESpeakNGData",
    ]
    protect_missing = [name for name in missing if name in set(protect_required)]
    whisper_available = _module_available("whisper") or _module_available("transformers")
    speaker_available = _module_available("speechbrain")
    pesq_available = _module_available("pesq")
    tts_available = _module_available("TTS")
    return {
        "ok": True,
        "device": device,
        "python": sys.executable,
        "cwd": os.getcwd(),
        "checkpoints": checkpoints,
        "config": runtime_config(),
        "chains": {
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
                "status": "partial" if not pesq_available else "available",
                "available": ["snr"] + (["pesq"] if pesq_available else []),
                "unavailable": ([] if pesq_available else ["pesq"]) + ["mosLqo", "maskingCurve"],
                "reason": None if pesq_available else "PESQ/MOS-LQO/masking curve evaluators are not installed",
            },
            "downstream_tts_eval": {
                "status": "available" if tts_available else "unavailable",
                "reason": None if tts_available else "Coqui TTS/XTTS package is not installed",
            },
        },
    }


def classify_generation_reason(exc: BaseException | None, output_exists: bool) -> str:
    checkpoint_missing = _checkpoint_status()["missing"]
    if checkpoint_missing:
        return "checkpoint_missing"
    if exc is not None:
        text = f"{type(exc).__name__}: {exc}".lower()
        if any(token in text for token in ["import", "module", "dependency", "not installed", "no module named"]):
            return "dependency_missing"
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
        if legacy_key == "lambdaPsy" and legacy_value > 0.001:
            warnings.append(f"{legacy_key}={legacy_value} is outside formal psychoacoustic scale; using formal default {default}.")
            return default
        if legacy_key == "lambdaL2" and legacy_value < 0.05:
            warnings.append(f"{legacy_key}={legacy_value} looks like an old UI value; using formal default {default}.")
            return default
        return legacy_value
    return default


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
    epsilon = to_float(optimization.get("epsilon")) or float(optimization_defaults["epsilon"])
    steps = int(optimization.get("steps") or int(optimization_defaults["steps"]))
    weight_warnings: list[str] = []
    weight_feature = _read_weight(timbre, "weightFeature", "lambdaTimbre", FORMAL_WEIGHT_FEATURE, weight_warnings)
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
            "weightFeature": weight_feature,
            "weightSemantic": weight_semantic,
            "weightPsy": weight_psy,
            "weightL2": weight_l2,
        },
        "deprecationWarnings": weight_warnings,
        "selectedSemanticEncoders": semantic.get("encoders"),
        "selectedTimbreEncoders": timbre.get("encoders"),
        "activeTimbreEncoders": sorted(active_timbre_encoders),
        "capabilities": diagnose_capabilities(),
        "protectCall": {
            "class": "SemanticE2EVGuard",
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
            from semantic_vguard import SemanticE2EVGuard

            guard = SemanticE2EVGuard(
                epsilon=epsilon,
                max_items=steps,
                device=torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu"),
                timbre_mode=timbre.get("mode") or "untargeted",
                use_vits="vits" in active_timbre_encoders,
                use_gsv="gsv" in active_timbre_encoders,
                use_mfcc_timbre="mfcc" in active_timbre_encoders,
                use_wavlm="wavlm" in active_timbre_encoders,
                use_cosyvoice="cosyvoice" in active_timbre_encoders,
                use_style="style" in active_timbre_encoders,
                weight_feature=weight_feature,
                weight_semantic=weight_semantic,
                weight_psy=weight_psy,
                weight_l2=weight_l2,
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
            result["source"] = "SemanticE2EVGuard.protect"
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


def compute_perception(clean_path: Path, protected_path: Path) -> dict[str, Any]:
    perception = {
        "snr": None,
        "pesq": None,
        "mosLqo": None,
        "l2Norm": None,
        "psychoacousticViolationRate": None,
        "maskingCurve": [],
        "status": "unavailable",
        "source": "audio_utils.py",
    }
    try:
        clean_data = read_wav_float(clean_path)
        protected_data = read_wav_float(protected_path)
        if clean_data is None or protected_data is None:
            raise ValueError("default perception metrics currently support 16-bit PCM WAV")
        clean, _ = clean_data
        protected, _ = protected_data
        n = min(len(clean), len(protected))
        l2_norm = float(np.linalg.norm(protected[:n] - clean[:n])) if n else None
        perception.update(
            {
                "snr": compute_snr_numpy(clean, protected),
                "l2Norm": l2_norm,
                "status": "computed",
            }
        )
    except Exception as exc:
        perception["error"] = str(exc)
    return perception


def compute_mfcc_semantic(clean_path: Path, protected_path: Path) -> dict[str, Any]:
    details = {
        "tokenErrorRate": None,
        "tokenChangeCount": None,
        "tokenTotal": None,
        "semanticDrift": None,
        "encoderDistances": empty_details()["semantic"]["encoderDistances"],
        "status": "unavailable",
    }
    if os.getenv("SEME2E_ENABLE_MFCC", "0") != "1":
        details["reason"] = "Set SEME2E_ENABLE_MFCC=1 to compute MFCC semantic distance."
        return details
    try:
        import librosa

        clean, sr = librosa.load(str(clean_path), sr=16000)
        protected, _ = librosa.load(str(protected_path), sr=16000)
        n = min(len(clean), len(protected))
        clean = clean[:n]
        protected = protected[:n]
        if n == 0:
            return details
        clean_mfcc = librosa.feature.mfcc(y=clean, sr=sr, n_mfcc=20).mean(axis=1)
        protected_mfcc = librosa.feature.mfcc(y=protected, sr=sr, n_mfcc=20).mean(axis=1)
        denom = float(np.linalg.norm(clean_mfcc) * np.linalg.norm(protected_mfcc) + 1.0e-12)
        cosine = float(np.dot(clean_mfcc, protected_mfcc) / denom)
        distance = float(np.linalg.norm(clean_mfcc - protected_mfcc))
        drift = max(0.0, min(1.0, 1.0 - cosine))
        encoders = details["encoderDistances"]
        for item in encoders:
            if item["encoder"] == "MFCC":
                item.update(
                    {
                        "cosineBeforeAfter": cosine,
                        "distance": distance,
                        "status": "computed",
                        "source": "librosa.mfcc",
                    }
                )
        details.update({"semanticDrift": drift, "status": "partial"})
    except Exception as exc:
        details["error"] = str(exc)
    return details


def maybe_asr_eval(clean_path: Path, protected_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
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
        asr["reason"] = "Set SEME2E_ENABLE_ASR=1 to run evaluate_asr.py dependencies."
        return asr

    try:
        from asr_backends import ASRTranscriber
        from text_metrics import cer, wer

        evaluations = []
        for model in actual_models:
            transcriber = ASRTranscriber(model, os.getenv("SEME2E_API_DEVICE", "cpu"))
            clean_text = transcriber.transcribe(clean_path)
            protected_text = transcriber.transcribe(protected_path)
            reference = reference_text or clean_text
            item = {
                "model": model,
                "referenceText": reference,
                "cleanTranscription": clean_text,
                "protectedTranscription": protected_text,
                "wer": wer(reference, protected_text),
                "cer": cer(reference, protected_text),
                "status": "computed",
            }
            evaluations.append(item)
        if evaluations:
            asr.update(evaluations[0])
            asr["model"] = evaluations[0]["model"]
            asr["evaluations"] = evaluations
            asr["status"] = "computed"
    except Exception as exc:
        asr["status"] = "unavailable"
        asr["error"] = str(exc)
    return asr


def create_asr_eval(task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    original_path, protected_path, result = _task_audio_paths(task_id)
    asr = maybe_asr_eval(
        original_path,
        protected_path,
        {
            "referenceText": payload.get("referenceText") or payload.get("reference_text"),
            "semantic": {"asrModel": payload.get("model")},
            "forceAsrEval": True,
        },
    )
    details = result.setdefault("details", {})
    details["asr"] = asr
    primary = result.setdefault("summary", {}).setdefault("primaryMetrics", {})
    for key in ["wer", "cer", "tokenErrorRate", "semanticDrift"]:
        if key in asr:
            primary[key] = asr.get(key)
    result["asrModel"] = asr.get("model")
    result["updatedAt"] = utc_now_iso()
    save_result(TASK_DIR / task_id, result)
    return {
        "taskId": task_id,
        "status": asr.get("status") or "computed",
        "asr": asr,
    }


def maybe_speaker_eval(clean_path: Path, protected_path: Path) -> dict[str, Any]:
    speaker = empty_details()["speaker"]
    metric = os.getenv("SEME2E_SPEAKER_METRIC", "ecapa")
    speaker["metric"] = metric
    if os.getenv("SEME2E_ENABLE_SPEAKER", "0") != "1":
        speaker["reason"] = "Set SEME2E_ENABLE_SPEAKER=1 to run speaker_similarity.py dependencies."
        return speaker
    try:
        from speaker_similarity import build_speaker_similarity

        model = os.getenv("SEME2E_SPEAKER_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
        scorer = build_speaker_similarity(metric, model, os.getenv("SEME2E_API_DEVICE", "cpu"))
        sim = scorer.score(clean_path, protected_path)
        speaker.update(
            {
                "simOriginalProtected": sim,
                "embeddingDistance": 1.0 - sim,
                "status": "computed",
            }
        )
    except Exception as exc:
        speaker["status"] = "unavailable"
        speaker["error"] = str(exc)
    return speaker


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
) -> dict[str, Any]:
    base_url = f"/api/artifacts/{task_id}"
    original_url = f"{base_url}/original/{input_path.name}"
    protected_url = f"{base_url}/protected/{protected_path.name}"
    result_json_url = f"{base_url}/result.json"

    details = empty_details()
    chains = default_chains()

    optimization = payload.get("optimization") or {}
    timbre = payload.get("timbre") or {}
    semantic_cfg = payload.get("semantic") or {}
    psychoacoustic = payload.get("psychoacoustic") or {}
    weight_warnings: list[str] = []
    weight_feature = _read_weight(timbre, "weightFeature", "lambdaTimbre", FORMAL_WEIGHT_FEATURE, weight_warnings)
    weight_semantic = _read_weight(semantic_cfg, "weightSemantic", "lambdaSemantic", FORMAL_WEIGHT_SEMANTIC, weight_warnings)
    weight_psy = _read_weight(psychoacoustic, "weightPsy", "lambdaPsy", FORMAL_WEIGHT_PSY, weight_warnings)
    weight_l2 = _read_weight(optimization, "weightL2", "lambdaL2", FORMAL_WEIGHT_L2, weight_warnings)
    meta = read_wav_meta(input_path)
    loss_items = protection_result.get("loss_items") or {}
    lfea = to_float(loss_items.get("Lfeat") or loss_items.get("Lfea") or loss_items.get("lossFeature") or loss_items.get("loss_timbre") or loss_items.get("L_feature"))
    lsem = to_float(loss_items.get("Lsem") or loss_items.get("lossSemantic") or loss_items.get("loss_semantic") or loss_items.get("L_semantic"))
    lpsy = to_float(loss_items.get("Lpsy") or loss_items.get("lossPsy") or loss_items.get("loss_psy") or loss_items.get("L_psy"))
    l2 = to_float(loss_items.get("L2") or loss_items.get("lossL2") or loss_items.get("loss_l2") or loss_items.get("l2Norm"))
    total = None
    if all(value is not None for value in [lfea, lsem, lpsy, l2]):
        total = (
            lfea * weight_feature
            + lsem * weight_semantic
            + lpsy * weight_psy
            + l2 * weight_l2
        )

    raw_trace = protection_result.get("optimization_trace") or protection_result.get("optimizationTrace") or []
    trace = []
    if isinstance(raw_trace, list):
        for index, item in enumerate(raw_trace):
            if not isinstance(item, dict):
                continue
            point = {
                "step": to_float(item.get("step") or item.get("epoch") or item.get("iteration") or item.get("iter")),
                "Lfeat": to_float(item.get("Lfeat") or item.get("Lfea") or item.get("lossFeature") or item.get("loss_timbre") or item.get("L_feature")),
                "Lsem": to_float(item.get("Lsem") or item.get("lossSemantic") or item.get("loss_semantic") or item.get("L_semantic")),
                "Lpsy": to_float(item.get("Lpsy") or item.get("lossPsy") or item.get("loss_psy") or item.get("L_psy")),
                "L2": to_float(item.get("L2") or item.get("lossL2") or item.get("loss_l2") or item.get("l2Norm")),
                "total": to_float(item.get("total") or item.get("lossTotal") or item.get("loss_total")),
            }
            if point["step"] is None:
                point["step"] = index
            if any(point[key] is not None for key in ["Lfeat", "Lsem", "Lpsy", "L2", "total"]):
                trace.append(point)
    details["generation"].update(
        {
            "mode": timbre.get("mode") or payload.get("mode") or "untargeted",
            "epsilon": to_float(optimization.get("epsilon")),
            "steps": int(optimization.get("steps") or 0) or None,
            "sampleRate": meta["sampleRate"],
            "durationSec": meta["durationSec"],
            "lossFinal": {
                "Lfeat": lfea,
                "Lsem": lsem,
                "Lpsy": lpsy,
                "L2": l2,
                "total": total,
            },
            "lossWeights": {
                "weight_feature": weight_feature,
                "weight_semantic": weight_semantic,
                "weight_psy": weight_psy,
                "weight_l2": weight_l2,
            },
            "optimizationTrace": trace,
            "source": protection_result.get("source") or "SemanticE2EVGuard.protect",
            "status": "computed" if protected_path.exists() else "unavailable",
            "realProtect": True,
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
    if weight_warnings:
        details["generation"]["deprecationWarnings"] = weight_warnings
    update_chain(chains, "protect_generation", details["generation"]["status"], details["generation"]["lossFinal"])

    details["perception"] = compute_perception(input_path, protected_path)
    if details["perception"]["snr"] is None:
        details["perception"]["snr"] = to_float(protection_result.get("snr"))
    update_chain(
        chains,
        "perception_eval",
        details["perception"]["status"],
        {"snr": details["perception"]["snr"], "pesq": details["perception"]["pesq"]},
    )

    details["semantic"] = compute_mfcc_semantic(input_path, protected_path)
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
    details["asr"]["reason"] = f"ASR 评估与保护流程解耦。请使用 /api/tasks/{task_id}/asr-eval 并选定 ASR 模型执行。"
    update_chain(chains, "asr_eval", details["asr"]["status"], {"wer": details["asr"]["wer"], "cer": details["asr"]["cer"]})

    details["speaker"] = maybe_speaker_eval(input_path, protected_path)
    update_chain(chains, "speaker_eval", details["speaker"]["status"], {"speakerSimilarity": details["speaker"]["simOriginalProtected"]})

    primary = empty_primary_metrics()
    primary.update(
        {
            "wer": details["asr"]["wer"],
            "cer": details["asr"]["cer"],
            "tokenErrorRate": details["semantic"]["tokenErrorRate"],
            "semanticDrift": details["semantic"]["semanticDrift"],
            "speakerSimilarity": details["downstreamTts"]["simProtectedClone"]
            if details["downstreamTts"]["simProtectedClone"] is not None
            else details["speaker"]["simOriginalProtected"],
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
            "metricSources": {
                "wer": {"source": "evaluate_asr.py", "status": details["asr"]["status"]},
                "cer": {"source": "evaluate_asr.py", "status": details["asr"]["status"]},
                "tokenErrorRate": {"source": "semantic tokenizer", "status": details["semantic"]["status"]},
                "semanticDrift": {"source": "semantic_encoder_distance", "status": details["semantic"]["status"]},
                "speakerSimilarity": {
                    "source": "downstream_tts.simProtectedClone"
                    if details["downstreamTts"]["simProtectedClone"] is not None
                    else "speaker.simOriginalProtected",
                    "status": details["speaker"]["status"],
                },
                "snr": {"source": details["perception"].get("source") or "audio_utils.audio_metrics", "status": details["perception"]["status"]},
                "pesq": {"source": "PESQ evaluator", "status": "unavailable"},
            },
        },
        "artifacts": {
            "originalAudioUrl": original_url,
            "protectedAudioUrl": protected_url,
            "resultJsonUrl": result_json_url,
        },
        "audio": {
            "original": audio_meta(input_path, original_url, uploaded_file_id),
            "protected": audio_meta(protected_path, protected_url),
        },
        "details": details,
        "chains": chains,
        "charts": charts,
        "request": {
            "mode": payload.get("mode"),
            "targets": payload.get("targets") or [],
            "semantic": {
                "enabled": bool(semantic_cfg.get("enabled")),
                "encoders": semantic_cfg.get("encoders") or [],
                "weightSemantic": weight_semantic,
            },
            "timbre": {
                "enabled": bool(timbre.get("enabled")),
                "mode": timbre.get("mode"),
                "encoders": timbre.get("encoders") or [],
                "weightFeature": weight_feature,
            },
            "psychoacoustic": {
                "enabled": bool(psychoacoustic.get("enabled")),
                "weightPsy": weight_psy,
            },
            "optimization": {
                "epsilon": to_float(optimization.get("epsilon")),
                "steps": int(optimization.get("steps") or 0) or None,
                "weightL2": weight_l2,
            },
        },
        "realProtect": True,
        "warning": None,
        "backend": {
            "version": "SemE2E API adapter",
            "commit": git_commit(),
            "python": sys.version.split()[0],
        },
    }
    return result


def save_result(task_dir: Path, result: dict[str, Any]) -> None:
    with (task_dir / "result.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)


def load_result(task_id: str) -> dict[str, Any]:
    with (TASK_DIR / task_id / "result.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def new_task_id() -> str:
    return f"task_{uuid.uuid4().hex[:12]}"


def new_file_id() -> str:
    return f"file_{uuid.uuid4().hex[:12]}"


def create_task(
    input_path: Path,
    uploaded_file_id: str | None,
    payload: dict[str, Any],
    request_id: str | None = None,
    task_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Any | None = None,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    task_id = task_id or new_task_id()
    task_dir = TASK_DIR / task_id
    original_dir = task_dir / "original"
    protected_dir = task_dir / "protected"
    original_dir.mkdir(parents=True, exist_ok=True)
    protected_dir.mkdir(parents=True, exist_ok=True)

    original_path = original_dir / input_path.name
    if input_path.resolve() != original_path.resolve():
        shutil.copyfile(input_path, original_path)
    protected_path = protected_dir / f"{input_path.stem}_protected.wav"

    started_at = utc_now_iso()
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
        progress_callback(progress=0.97, stage="result_evaluation", message="后端正在评估生成的音频")
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


def normalize_tts_model(value: str | None) -> str:
    raw = (value or os.getenv("SEME2E_TTS_MODEL") or "tts_models/multilingual/multi-dataset/xtts_v2").strip()
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
    diagnostics = {
        "taskId": task_id,
        "ttsPackageAvailable": _module_available("TTS"),
        "model": model,
        "language": language,
        "speed": speed,
        "device": device,
        "originalReferencePath": str(original_path),
        "protectedReferencePath": str(protected_path),
    }
    if progress_callback is not None:
            progress_callback(progress=0.18, message="正在加载真实 TTS 克隆后端")
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("TASK_CANCELLED")
    if not diagnostics["ttsPackageAvailable"]:
        downstream = result.setdefault("details", {}).setdefault("downstreamTts", {})
        downstream.update(
            {
                "enabled": False,
                "ttsModel": model,
                "status": "unavailable",
                "source": "CoquiTTS.xtts_v2",
                "reason": "Coqui TTS/XTTS package is not installed",
                "cloneText": text,
            }
        )
        save_result(TASK_DIR / task_id, result)
        raise CloneBackendUnavailableError(
            "真实 TTS 语音克隆后端不可用：未安装 Coqui TTS/XTTS。请安装 TTS 并配置模型后重试。",
            task_id=task_id,
            diagnostics=diagnostics,
            reason="dependency_missing",
        )

    clone_id = f"clone_{uuid.uuid4().hex[:10]}"
    clone_dir = TASK_DIR / task_id / "clones" / clone_id
    clone_dir.mkdir(parents=True, exist_ok=True)

    original_clone_path = clone_dir / f"{clone_id}_original_clone.wav"
    protected_clone_path = clone_dir / f"{clone_id}_protected_clone.wav"
    try:
        if progress_callback is not None:
            progress_callback(progress=0.32, message="正在从原始参考音频生成克隆音频")
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("TASK_CANCELLED")
        source_model = _tts_clone_to_file(original_path, text, original_clone_path, model=model, language=language, speed=speed, device=device)
        if progress_callback is not None:
            progress_callback(progress=0.62, message="正在从保护参考音频生成克隆音频")
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("TASK_CANCELLED")
        _tts_clone_to_file(protected_path, text, protected_clone_path, model=model, language=language, speed=speed, device=device)
        if progress_callback is not None:
            progress_callback(progress=0.9, message="正在保存下游 TTS 克隆结果")
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("TASK_CANCELLED")
    except Exception as exc:
        diagnostics.update(
            {
                "exceptionType": type(exc).__name__,
                "exceptionMessage": str(exc),
                "stackTrace": traceback.format_exc(),
                "originalOutputExists": original_clone_path.exists(),
                "protectedOutputExists": protected_clone_path.exists(),
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

    base_url = f"/api/artifacts/{task_id}/clones/{clone_id}"
    response = {
        "cloneId": clone_id,
        "taskId": task_id,
        "status": "completed",
        "source": f"CoquiTTS:{source_model}",
        "message": "真实 TTS 克隆音频已生成。",
        "request": {
            "text": text,
            "model": model,
            "language": language,
            "speed": speed,
        },
        "originalCloneAudio": audio_meta(original_clone_path, f"{base_url}/{original_clone_path.name}"),
        "protectedCloneAudio": audio_meta(protected_clone_path, f"{base_url}/{protected_clone_path.name}"),
    }

    clones = result.setdefault("cloneResults", [])
    clones.append(response)
    downstream = (result.setdefault("details", {}).setdefault("downstreamTts", {}))
    downstream.update(
        {
            "enabled": True,
            "ttsModel": response["request"]["model"],
            "status": "computed",
            "source": response["source"],
            "lastCloneId": clone_id,
            "cloneText": text,
        }
    )
    save_result(TASK_DIR / task_id, result)
    return response
