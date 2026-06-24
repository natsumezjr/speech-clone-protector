from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import uuid
import wave
from pathlib import Path
from typing import Any

import numpy as np

from result_schema import default_chains, empty_charts, empty_details, empty_primary_metrics, utc_now_iso

ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = Path(os.getenv("SEME2E_RUNTIME_DIR", ROOT.parents[1] / "seme2e-runtime"))
UPLOAD_DIR = RUNTIME_DIR / "uploads"
TASK_DIR = RUNTIME_DIR / "tasks"


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


def safe_copy_or_perturb(input_path: Path, output_path: Path, epsilon: float | None) -> dict[str, Any]:
    """Connectivity fallback when heavyweight SemE2E dependencies are unavailable."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        wav_data = read_wav_float(input_path)
        if wav_data is None:
            raise ValueError("fallback perturbation currently supports 16-bit PCM WAV")
        audio, sample_rate = wav_data
        scale = min(float(epsilon or 0.003), 0.01)
        rng = np.random.default_rng(20260624)
        noise = rng.normal(0.0, scale / 6.0, size=audio.shape).astype("float32")
        protected = np.clip(audio + noise, -1.0, 1.0)
        write_wav_float(output_path, protected, sample_rate)
        return {
            "output_wav": str(output_path),
            "snr": compute_snr_numpy(audio, protected),
            "loss_items": {},
            "target_speaker": None,
            "source": "fallback_light_perturbation",
            "warning": "SemanticE2EVGuard unavailable; generated a connectivity-only protected audio artifact.",
        }
    except Exception as exc:
        shutil.copyfile(input_path, output_path)
        return {
            "output_wav": str(output_path),
            "snr": None,
            "loss_items": {},
            "target_speaker": None,
            "source": "fallback_copy",
            "warning": f"SemanticE2EVGuard unavailable and perturbation fallback failed: {exc}",
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


def run_protection(input_path: Path, output_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    optimization = payload.get("optimization") or {}
    timbre = payload.get("timbre") or {}
    psychoacoustic = payload.get("psychoacoustic") or {}
    semantic = payload.get("semantic") or {}
    epsilon = to_float(optimization.get("epsilon")) or 8 / 255
    steps = int(optimization.get("steps") or int(os.getenv("SEME2E_API_STEPS", "8")))
    device = os.getenv("SEME2E_API_DEVICE", "cpu")
    real_guard_enabled = os.getenv("SEME2E_API_REAL_GUARD", "1") == "1"
    allow_fallback = os.getenv("SEME2E_API_ALLOW_FALLBACK", "0") == "1"

    if real_guard_enabled:
        try:
            import torch
            from semantic_vguard import SemanticE2EVGuard

            guard = SemanticE2EVGuard(
                epsilon=epsilon,
                max_items=steps,
                device=torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu"),
                timbre_mode=timbre.get("mode") or "untargeted",
                use_vits=False,
                use_gsv=False,
                use_mfcc_timbre=True,
                use_wavlm=False,
                use_cosyvoice=False,
                use_style=False,
                weight_feature=to_float(timbre.get("lambdaTimbre")) or 500.0,
                weight_semantic=to_float(semantic.get("lambdaSemantic")) or 100.0,
                weight_psy=to_float(psychoacoustic.get("lambdaPsy")) or 1.0e-5,
                weight_l2=to_float(optimization.get("lambdaL2")) or 0.1,
            )
            result = guard.protect(input_path, output_path)
            result["source"] = "SemanticE2EVGuard.protect"
            return result
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(
                    "SemanticE2EVGuard.protect failed. The API did not generate a protected audio file. "
                    "Install/check backend dependencies and model checkpoints, or explicitly set "
                    "SEME2E_API_ALLOW_FALLBACK=1 for connectivity-only demo output."
                ) from exc
            fallback = safe_copy_or_perturb(input_path, output_path, epsilon)
            fallback["guardError"] = str(exc)
            return fallback

    fallback = safe_copy_or_perturb(input_path, output_path, epsilon)
    fallback["guardSkipped"] = "SEME2E_API_REAL_GUARD=0; SemanticE2EVGuard.protect was intentionally skipped."
    return fallback


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
    asr["model"] = semantic.get("asrModel") or os.getenv("SEME2E_ASR_MODEL")
    if os.getenv("SEME2E_ENABLE_ASR", "0") != "1":
        asr["status"] = "unavailable"
        asr["reason"] = "Set SEME2E_ENABLE_ASR=1 to run evaluate_asr.py dependencies."
        return asr

    try:
        from asr_backends import ASRTranscriber
        from text_metrics import cer, wer

        transcriber = ASRTranscriber(asr["model"] or "openai/whisper-small", os.getenv("SEME2E_API_DEVICE", "cpu"))
        clean_text = transcriber.transcribe(clean_path)
        protected_text = transcriber.transcribe(protected_path)
        reference = reference_text or clean_text
        asr.update(
            {
                "referenceText": reference,
                "cleanTranscription": clean_text,
                "protectedTranscription": protected_text,
                "wer": wer(reference, protected_text),
                "cer": cer(reference, protected_text),
                "status": "computed",
            }
        )
    except Exception as exc:
        asr["status"] = "unavailable"
        asr["error"] = str(exc)
    return asr


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
    meta = read_wav_meta(input_path)
    loss_items = protection_result.get("loss_items") or {}
    lfea = to_float(loss_items.get("Lfea") or loss_items.get("loss_timbre"))
    lsem = to_float(loss_items.get("Lsem") or loss_items.get("loss_semantic"))
    lpsy = to_float(loss_items.get("Lpsy") or loss_items.get("loss_psy"))
    l2 = to_float(loss_items.get("L2") or loss_items.get("loss_l2"))
    total = None
    if all(value is not None for value in [lfea, lsem, lpsy, l2]):
        total = (
            lfea * (to_float(timbre.get("lambdaTimbre")) or 500.0)
            + lsem * (to_float(semantic_cfg.get("lambdaSemantic")) or 100.0)
            + lpsy * (to_float(psychoacoustic.get("lambdaPsy")) or 1.0e-5)
            + l2 * (to_float(optimization.get("lambdaL2")) or 0.1)
        )

    trace = protection_result.get("optimization_trace") or protection_result.get("optimizationTrace") or []
    details["generation"].update(
        {
            "mode": timbre.get("mode") or payload.get("mode") or "untargeted",
            "epsilon": to_float(optimization.get("epsilon")),
            "steps": int(optimization.get("steps") or 0) or None,
            "sampleRate": meta["sampleRate"],
            "durationSec": meta["durationSec"],
            "lossFinal": {
                "Lfea": lfea,
                "Lsem": lsem,
                "Lpsy": lpsy,
                "L2": l2,
                "total": total,
            },
            "lossWeights": {
                "weight_feature": to_float(timbre.get("lambdaTimbre")),
                "weight_semantic": to_float(semantic_cfg.get("lambdaSemantic")),
                "weight_psy": to_float(psychoacoustic.get("lambdaPsy")),
                "weight_l2": to_float(optimization.get("lambdaL2")),
            },
            "optimizationTrace": trace,
            "source": protection_result.get("source") or "SemanticE2EVGuard.protect",
            "status": "computed" if protected_path.exists() else "unavailable",
        }
    )
    if protection_result.get("warning"):
        details["generation"]["warning"] = protection_result["warning"]
    if protection_result.get("guardError"):
        details["generation"]["guardError"] = protection_result["guardError"]
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

    details["asr"] = maybe_asr_eval(input_path, protected_path, payload)
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

        elapsed = (
            datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
            - datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        ).total_seconds()
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
                "snr": {"source": "audio_utils.audio_metrics", "status": details["perception"]["status"]},
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


def create_task(input_path: Path, uploaded_file_id: str | None, payload: dict[str, Any]) -> dict[str, Any]:
    ensure_runtime_dirs()
    task_id = new_task_id()
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
    protection_result = run_protection(original_path, protected_path, payload)
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


def _clone_fallback(reference_path: Path, output_path: Path, seed: int) -> None:
    wav_data = read_wav_float(reference_path)
    if wav_data is None:
        shutil.copyfile(reference_path, output_path)
        return
    audio, sample_rate = wav_data
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.00035, size=audio.shape).astype("float32")
    write_wav_float(output_path, np.clip(audio + noise, -1.0, 1.0), sample_rate)


def create_clone_voice(task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    text = str(payload.get("text") or "").strip()
    if not text:
        raise ValueError("text is required")
    original_path, protected_path, result = _task_audio_paths(task_id)
    clone_id = f"clone_{uuid.uuid4().hex[:10]}"
    clone_dir = TASK_DIR / task_id / "clones" / clone_id
    clone_dir.mkdir(parents=True, exist_ok=True)

    original_clone_path = clone_dir / f"{clone_id}_original_clone.wav"
    protected_clone_path = clone_dir / f"{clone_id}_protected_clone.wav"

    # Hook point for real downstream TTS. The default keeps the API runnable without
    # large TTS checkpoints and marks the artifact as partial instead of computed.
    _clone_fallback(original_path, original_clone_path, 20260624)
    _clone_fallback(protected_path, protected_clone_path, 20260625)

    base_url = f"/api/artifacts/{task_id}/clones/{clone_id}"
    response = {
        "cloneId": clone_id,
        "taskId": task_id,
        "status": "partial",
        "source": "fallback_reference_audio_clone",
        "message": "TTS clone backend is not enabled; returned playable reference-derived clone artifacts.",
        "request": {
            "text": text,
            "model": payload.get("model") or "default",
            "language": payload.get("language") or "auto",
            "speed": to_float(payload.get("speed")) or 1.0,
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
            "status": "partial",
            "source": response["source"],
            "lastCloneId": clone_id,
            "cloneText": text,
        }
    )
    save_result(TASK_DIR / task_id, result)
    return response
