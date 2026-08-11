from __future__ import annotations

import importlib.util
import math
import os
import re
import threading
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


EPS = 1.0e-12
PCM16_QUANTIZATION_STEP = 1.0 / 32768.0
ROOT = Path(__file__).resolve().parent
_S3_TOKENIZER_CACHE: dict[tuple[str, str], Any] = {}
_SEMANTIC_ENCODER_CACHE: dict[tuple[str, str, str, str], Any] = {}
_SEMANTIC_ENCODER_CACHE_LOCK = threading.Lock()
_SEMANTIC_ENCODER_LAST_ERROR: str | None = None


def _checkpoint_file_ready(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size < 1024 * 1024:
            return False
        with path.open("rb") as file:
            return not file.read(64).startswith(b"version https://git-lfs.github.com/spec")
    except OSError:
        return False


def _default_s3_tokenizer_model() -> str:
    configured = os.getenv("SEME2E_TOKENIZER_MODEL")
    if configured:
        return str(_resolve_local_model_path(configured))
    project_checkpoint = ROOT / "checkpoints" / "CosyVoice" / "speech_tokenizer_v1.onnx"
    if _checkpoint_file_ready(project_checkpoint):
        return str(project_checkpoint.resolve())
    if os.getenv("SEME2E_ALLOW_MODEL_DOWNLOADS", "0") == "1":
        return "speech_tokenizer_v1_25hz"
    raise FileNotFoundError(
        f"local S3 tokenizer checkpoint is missing or incomplete: {project_checkpoint}; "
        "set SEME2E_ALLOW_MODEL_DOWNLOADS=1 only when an online download is intended"
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _positive_env_float(name: str, default: float | None) -> float | None:
    value = finite_float(os.getenv(name))
    if value is None:
        value = default
    return value if value is not None and value > 0 else None


# Lightweight protection-side calibration from four existing real tasks
# (task_10a6a6320505 plus three current pro tasks). The defaults are rounded
# 90th-percentile scales. Clone semantic and quality defaults come from six
# existing real clone pairs in task_4810dbfae886. Every scale remains
# environment-overridable.
SCORE_CALIBRATION: dict[str, float | None] = {
    "tokenChangeRate90": _positive_env_float("SEME2E_TOKEN_CHANGE_R90", 0.90),
    "semanticDrift90": _positive_env_float("SEME2E_SEMANTIC_DRIFT_D90", 0.60),
    "directDistance90": _positive_env_float("SEME2E_DIRECT_DISTANCE_D90", 0.50),
    "cloneTokenChangeRate90": _positive_env_float("SEME2E_CLONE_TOKEN_CHANGE_R90", 1.00),
    "cloneSemanticDrift90": _positive_env_float("SEME2E_CLONE_SEMANTIC_DRIFT_D90", 0.78),
    "cloneQualityWeightedDrop90": _positive_env_float("SEME2E_CLONE_WEIGHTED_QUALITY_DROP_R90", 0.75),
}

SCORE_CALIBRATION_SOURCES: dict[str, str] = {
    "tokenChangeRate90": "real protection tasks p90 rounded: [0.7887,0.8684,0.8940,0.7602]",
    "semanticDrift90": "real protection tasks p90 rounded: [0.5943,0.5535,0.5778,0.4318]",
    "directDistance90": "real protection tasks p90 rounded: [0.4956,0.2618,0.4622,0.2689]",
    "cloneTokenChangeRate90": "existing real clone pairs task_4810dbfae886, n=6, p90=0.9962 rounded",
    "cloneSemanticDrift90": "existing real clone pairs task_4810dbfae886, n=6, p90=0.7784 rounded",
    "cloneQualityWeightedDrop90": "same-text clone pairs across five real protection tasks, n=20, weighted quality-drop p90=0.73953 rounded to 0.75",
}

# Pair-reference PESQ/STOI are the primary clone quality evidence. DNSMOS is a
# no-reference absolute-quality cross-check and therefore carries a smaller
# weight. The same available-component mask is used on the before/after sides;
# a missing metric is never filled with zero.
CLONE_QUALITY_COMPONENT_WEIGHTS: dict[str, float] = {
    "pesq": 0.45,
    "stoi": 0.45,
    "dnsmos": 0.10,
}


def weighted_available_mean(items: dict[str, tuple[float | None, float]]) -> float | None:
    weighted = 0.0
    weight_sum = 0.0
    for value, weight in items.values():
        value = finite_float(value)
        weight = finite_float(weight)
        if value is None or weight is None or weight <= 0:
            continue
        weighted += value * weight
        weight_sum += weight
    if weight_sum <= 0:
        return None
    return weighted / weight_sum


def piecewise_linear_score(value: Any, anchors: Sequence[tuple[float, float]]) -> float | None:
    """Map a real metric to 0..100 using the exact ordered anchor table."""

    number = finite_float(value)
    points = sorted((float(x), float(score)) for x, score in anchors)
    if number is None or not points:
        return None
    if number <= points[0][0]:
        return clamp(points[0][1], 0.0, 100.0)
    if number >= points[-1][0]:
        return clamp(points[-1][1], 0.0, 100.0)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= number <= x1:
            ratio = (number - x0) / max(x1 - x0, EPS)
            return clamp(y0 + ratio * (y1 - y0), 0.0, 100.0)
    return None


def phi_score(value: Any, x90: Any) -> float | None:
    """VoiceShield v2.1 saturation calibration: 100 * (1 - 10^(-x/x90))."""

    number = finite_float(value)
    scale = finite_float(x90)
    if number is None or scale is None or scale <= 0:
        return None
    return clamp(100.0 * (1.0 - 10.0 ** (-max(number, 0.0) / scale)), 0.0, 100.0)


def _smoothstep_weight(value: Any, lower: float, upper: float) -> float | None:
    number = finite_float(value)
    if number is None:
        return None
    u = clamp((number - lower) / max(upper - lower, EPS))
    return u * u * (3.0 - 2.0 * u)


def compute_protection_quality_score(
    snr: Any,
    stoi: Any,
    pesq: Any,
    dns_mos: Any = None,
) -> dict[str, Any]:
    snr_score = piecewise_linear_score(snr, [(10.0, 0.0), (15.0, 55.0), (18.5, 75.0), (25.0, 92.0), (30.0, 100.0)])
    stoi_score = piecewise_linear_score(stoi, [(0.60, 0.0), (0.75, 60.0), (0.90, 95.0), (1.00, 100.0)])
    pesq_score = piecewise_linear_score(pesq, [(1.0, 0.0), (1.5, 45.0), (2.0, 75.0), (3.0, 90.0), (4.5, 100.0)])
    dns_mos_value = finite_float(dns_mos)
    if dns_mos_value is not None and not 1.0 <= dns_mos_value <= 5.0:
        dns_mos_value = None
    dns_mos_score = 100.0 * (dns_mos_value - 1.0) / 4.0 if dns_mos_value is not None else None
    required = {"snr": snr_score, "stoi": stoi_score, "pesq": pesq_score}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        quality_score = None
        status = "unavailable"
        reason = f"保护音频质量指标尚未完整生成：{', '.join(missing)}"
    elif dns_mos_score is None:
        quality_score = (0.40 * snr_score + 0.35 * stoi_score + 0.15 * pesq_score) / 0.90
        status = "available"
        reason = None
    else:
        quality_score = 0.40 * snr_score + 0.35 * stoi_score + 0.15 * pesq_score + 0.10 * dns_mos_score
        status = "available"
        reason = None
    if quality_score is None:
        quality_level = None
    elif quality_score >= 85:
        quality_level = "excellent"
    elif quality_score >= 70:
        quality_level = "good"
    elif quality_score >= 50:
        quality_level = "fair"
    else:
        quality_level = "poor"
    return {
        "snrScore": snr_score,
        "stoiScore": stoi_score,
        "pesqScore": pesq_score,
        "dnsMos": dns_mos_value,
        "dnsMosScore": dns_mos_score,
        "dnsMosStatus": "available" if dns_mos_value is not None else "unavailable",
        "dnsMosReason": None if dns_mos_value is not None else "语音质量评分尚未生成",
        "qualityScore": quality_score,
        "qualityLevel": quality_level,
        "scoreStatus": status,
        "scoreReason": reason,
    }


def compute_protection_semantic_score(token_change_rate: Any, semantic_drift: Any) -> dict[str, Any]:
    token_score = phi_score(token_change_rate, SCORE_CALIBRATION["tokenChangeRate90"])
    drift_score = phi_score(semantic_drift, SCORE_CALIBRATION["semanticDrift90"])
    missing = [name for name, value in {"tokenChangeRate": token_score, "semanticDrift": drift_score}.items() if value is None]
    score = 0.55 * token_score + 0.45 * drift_score if not missing else None
    return {
        "tokenScore": token_score,
        "driftScore": drift_score,
        "protectionSemanticScore": score,
        "scoreStatus": "available" if score is not None else "unavailable",
        "scoreReason": None if score is not None else f"保护语义指标尚未完整生成：{', '.join(missing)}",
    }


def compute_direct_identity_score(similarity: Any) -> dict[str, Any]:
    similarity_value = finite_float(similarity)
    distance = 1.0 - similarity_value if similarity_value is not None else None
    score = phi_score(distance, SCORE_CALIBRATION["directDistance90"])
    return {
        "directDistance": distance,
        "directIdentityScore": score,
        "scoreStatus": "available" if score is not None else "unavailable",
        "scoreReason": None if score is not None else "直接声音身份指标尚未生成",
    }


def compute_clone_identity_score(original_similarity: Any, protected_similarity: Any) -> dict[str, Any]:
    similarity_before = finite_float(original_similarity)
    similarity_after = finite_float(protected_similarity)
    baseline_weight = _smoothstep_weight(similarity_before, 0.25, 0.65)
    distance_before = 1.0 - similarity_before if similarity_before is not None else None
    distance_after = 1.0 - similarity_after if similarity_after is not None else None
    distance_delta = distance_after - distance_before if distance_before is not None and distance_after is not None else None
    score = None
    reason = None
    if distance_before is None or distance_after is None:
        reason = "原始克隆或保护后克隆的声音身份结果尚未生成"
    elif 0.75 - distance_before <= EPS:
        reason = "原始克隆的声音身份结果不足以进行有效比较"
    else:
        progress = clamp((distance_after - distance_before) / (0.75 - distance_before))
        bonus = 5.0 * clamp((distance_after - 0.75) / 0.25)
        score = 95.0 * progress + bonus
    return {
        "embeddingDistanceBefore": distance_before,
        "embeddingDistanceAfter": distance_after,
        "embeddingDistanceDelta": distance_delta,
        "cloneIdentityScore": score,
        "identityBaselineWeight": baseline_weight,
        "cloneIdentityStatus": "available" if score is not None else "unavailable",
        "cloneIdentityReason": reason,
    }


def compute_bounded_text_metrics(reference_text: str | None, hypothesis_text: str | None) -> dict[str, Any]:
    reference = reference_text or ""
    hypothesis = hypothesis_text or ""
    word_ref = _normalize_words(reference)
    word_hyp = _normalize_words(hypothesis)
    char_ref = _normalize_chars(reference)
    char_hyp = _normalize_chars(hypothesis)
    word_distance = _edit_distance(word_ref, word_hyp)
    char_distance = _edit_distance(char_ref, char_hyp)
    word_denominator = max(len(word_ref), len(word_hyp))
    char_denominator = max(len(char_ref), len(char_hyp))
    word_accuracy = 1.0 if word_denominator == 0 else clamp(1.0 - word_distance / word_denominator)
    char_accuracy = 1.0 if char_denominator == 0 else clamp(1.0 - char_distance / char_denominator)
    accuracy = 0.6 * word_accuracy + 0.4 * char_accuracy
    return {
        "accuracy": accuracy,
        "error": 1.0 - accuracy,
        "wordAccuracy": word_accuracy,
        "charAccuracy": char_accuracy,
        "wordEditDistance": word_distance,
        "charEditDistance": char_distance,
        "referenceWordCount": len(word_ref),
        "hypothesisWordCount": len(word_hyp),
        "referenceCharCount": len(char_ref),
        "hypothesisCharCount": len(char_hyp),
        "wordUnit": "latin_word_or_cjk_character",
        "charUnit": "non_whitespace_character",
    }


def compute_clone_semantic_score(
    target_text: str | None,
    clean_transcription: str | None,
    protected_transcription: str | None,
    token_change_rate: Any,
    semantic_drift: Any,
) -> dict[str, Any]:
    if clean_transcription is None or protected_transcription is None:
        return {
            "cleanCloneTextAccuracy": None,
            "cleanCloneTextError": None,
            "protectedCloneTextAccuracy": None,
            "protectedCloneTextError": None,
            "cloneTextChangeAccuracy": None,
            "cloneTextChangeRate": None,
            "semanticBaselineWeight": None,
            "cloneTokenScore": None,
            "cloneDriftScore": None,
            "cloneSemanticScore": None,
            "cloneSemanticStatus": "unavailable",
            "cloneSemanticReason": "克隆语音文本尚未生成",
        }
    clean_metrics = compute_bounded_text_metrics(target_text, clean_transcription)
    protected_metrics = compute_bounded_text_metrics(target_text, protected_transcription)
    change_metrics = compute_bounded_text_metrics(clean_transcription, protected_transcription)
    baseline_weight = _smoothstep_weight(clean_metrics["accuracy"], 0.0, 1.0)
    token_score = phi_score(token_change_rate, SCORE_CALIBRATION["cloneTokenChangeRate90"])
    drift_score = phi_score(semantic_drift, SCORE_CALIBRATION["cloneSemanticDrift90"])
    score = 0.55 * token_score + 0.45 * drift_score if token_score is not None and drift_score is not None else None
    calibration_missing = SCORE_CALIBRATION["cloneTokenChangeRate90"] is None or SCORE_CALIBRATION["cloneSemanticDrift90"] is None
    reason = None
    if score is None:
        reason = "克隆语义评分标定尚未完成" if calibration_missing else "克隆语义指标尚未完整生成"
    return {
        "cleanCloneTextAccuracy": clean_metrics["accuracy"],
        "cleanCloneTextError": clean_metrics["error"],
        "cleanCloneTextMetrics": clean_metrics,
        "protectedCloneTextAccuracy": protected_metrics["accuracy"],
        "protectedCloneTextError": protected_metrics["error"],
        "protectedCloneTextMetrics": protected_metrics,
        "cloneTextChangeAccuracy": change_metrics["accuracy"],
        "cloneTextChangeRate": change_metrics["error"],
        "cloneTextChangeMetrics": change_metrics,
        "semanticBaselineWeight": baseline_weight,
        "cloneTokenChangeRate": finite_float(token_change_rate),
        "cloneSemanticDrift": finite_float(semantic_drift),
        "cloneTokenScore": token_score,
        "cloneDriftScore": drift_score,
        "cloneSemanticScore": score,
        "cloneSemanticStatus": "available" if score is not None else "unavailable",
        "cloneSemanticReason": reason,
    }


def adjust_clone_quality_score(
    raw_score: Any,
    *,
    identity_baseline_weight: Any = None,
    clone_identity_score: Any = None,
    clone_semantic_score: Any = None,
) -> tuple[float | None, float | None]:
    raw_value = finite_float(raw_score)
    score = raw_value
    relevance = None
    identity_weight = finite_float(identity_baseline_weight)
    identity_score = finite_float(clone_identity_score)
    semantic_score = finite_float(clone_semantic_score)
    if raw_value is not None:
        # Missing identity/semantic evidence must never improve the quality score.
        relevance = 1.0
        if identity_weight is not None and identity_score is not None and semantic_score is not None:
            baseline_need = 1.0 - clamp(identity_weight)
            defense_need = 1.0 - clamp(min(identity_score, semantic_score) / 100.0)
            relevance = clamp(max(baseline_need, defense_need))
            score = (1.0 - relevance) * 100.0 + relevance * raw_value
    return score, relevance


def compute_clone_quality_score(
    clean_mos: Any,
    protected_mos: Any,
    *,
    pair_pesq: Any = None,
    pair_stoi: Any = None,
    identity_baseline_weight: Any = None,
    clone_identity_score: Any = None,
    clone_semantic_score: Any = None,
) -> dict[str, Any]:
    clean_value = finite_float(clean_mos)
    protected_value = finite_float(protected_mos)
    if clean_value is not None and not 1.0 <= clean_value <= 5.0:
        clean_value = None
    if protected_value is not None and not 1.0 <= protected_value <= 5.0:
        protected_value = None
    pesq_value = finite_float(pair_pesq)
    stoi_value = finite_float(pair_stoi)

    # PESQ and STOI are reference metrics. The clean/original-side clone is the
    # same-text reference, so its self-reference baseline is defined as 100;
    # the protected-side values are measured against that reference. DNSMOS is
    # no-reference and is therefore measured independently on both sides.
    component_before: dict[str, tuple[float | None, float]] = {}
    component_after: dict[str, tuple[float | None, float]] = {}
    normalized_components: dict[str, dict[str, float] | None] = {
        "pesq": None,
        "stoi": None,
        "dnsmos": None,
    }
    if pesq_value is not None:
        pesq_before = 100.0
        pesq_after = 100.0 * clamp((pesq_value + 0.5) / 5.0)
        component_before["pesq"] = (pesq_before, CLONE_QUALITY_COMPONENT_WEIGHTS["pesq"])
        component_after["pesq"] = (pesq_after, CLONE_QUALITY_COMPONENT_WEIGHTS["pesq"])
        normalized_components["pesq"] = {
            "before": pesq_before,
            "after": pesq_after,
            "weight": CLONE_QUALITY_COMPONENT_WEIGHTS["pesq"],
        }
    if stoi_value is not None:
        stoi_before = 100.0
        stoi_after = 100.0 * clamp(stoi_value)
        component_before["stoi"] = (stoi_before, CLONE_QUALITY_COMPONENT_WEIGHTS["stoi"])
        component_after["stoi"] = (stoi_after, CLONE_QUALITY_COMPONENT_WEIGHTS["stoi"])
        normalized_components["stoi"] = {
            "before": stoi_before,
            "after": stoi_after,
            "weight": CLONE_QUALITY_COMPONENT_WEIGHTS["stoi"],
        }
    if clean_value is not None and protected_value is not None:
        dns_before = 100.0 * clamp((clean_value - 1.0) / 4.0)
        dns_after = 100.0 * clamp((protected_value - 1.0) / 4.0)
        component_before["dnsmos"] = (dns_before, CLONE_QUALITY_COMPONENT_WEIGHTS["dnsmos"])
        component_after["dnsmos"] = (dns_after, CLONE_QUALITY_COMPONENT_WEIGHTS["dnsmos"])
        normalized_components["dnsmos"] = {
            "before": dns_before,
            "after": dns_after,
            "weight": CLONE_QUALITY_COMPONENT_WEIGHTS["dnsmos"],
        }

    quality_before = weighted_available_mean(component_before)
    quality_after = weighted_available_mean(component_after)
    drop_rate = (
        max(0.0, (quality_before - quality_after) / max(quality_before, EPS))
        if quality_before is not None and quality_after is not None and quality_before > EPS
        else None
    )
    raw_score = phi_score(drop_rate, SCORE_CALIBRATION["cloneQualityWeightedDrop90"])
    baseline_weight = _smoothstep_weight(clean_value, 2.5, 4.0)
    if baseline_weight is None and quality_before is not None:
        baseline_weight = _smoothstep_weight(quality_before, 50.0, 90.0)
    score, relevance = adjust_clone_quality_score(
        raw_score,
        identity_baseline_weight=identity_baseline_weight,
        clone_identity_score=clone_identity_score,
        clone_semantic_score=clone_semantic_score,
    )
    degradation_components = {
        name: (
            max(0.0, values["before"] - values["after"])
            if isinstance(values, dict)
            else None
        )
        for name, values in normalized_components.items()
    }
    missing = [name for name, value in normalized_components.items() if value is None]
    reason = None if raw_score is not None else "克隆语音质量结果尚未生成"
    return {
        "cleanCloneQualityMos": clean_value,
        "protectedCloneQualityMos": protected_value,
        "clonePairPesq": pesq_value,
        "clonePairStoi": stoi_value,
        "cloneQualityBefore": quality_before,
        "cloneQualityAfter": quality_after,
        "cloneQualityDropRate": drop_rate,
        "clonePesqDegradationScore": degradation_components["pesq"],
        "cloneStoiDegradationScore": degradation_components["stoi"],
        "cloneDnsMosDegradationScore": degradation_components["dnsmos"],
        "cloneQualityComponents": normalized_components,
        "cloneQualityRawScore": raw_score,
        "cloneQualityRelevance": relevance,
        "cloneQualityScore": score,
        "qualityBaselineWeight": baseline_weight,
        "cloneQualityStatus": "available" if score is not None and not missing else "partial" if score is not None else "unavailable",
        "cloneQualityReason": reason if reason is not None else (f"克隆语音质量指标尚未完整生成：{', '.join(missing)}" if missing else None),
    }


def aggregate_weighted_scores(
    items: Sequence[dict[str, Any]],
    score_key: str,
    weight_key: str,
    empty_reason: str,
) -> tuple[float | None, str | None]:
    weighted_sum = 0.0
    weight_sum = 0.0
    for item in items:
        score = finite_float(item.get(score_key))
        weight = finite_float(item.get(weight_key))
        if score is None or weight is None or weight <= EPS:
            continue
        weighted_sum += weight * score
        weight_sum += weight
    if weight_sum <= EPS:
        return None, empty_reason
    return weighted_sum / weight_sum, None


def metric_source(status: str, source: str, reason: str | None = None, formula: str | None = None, metric: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": status, "source": source}
    if metric:
        payload["metric"] = metric
    if reason:
        payload["reason"] = reason
    if formula:
        payload["formula"] = formula
    return payload


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _resolve_local_model_path(value: Any) -> Any:
    if not isinstance(value, (str, os.PathLike)):
        return value
    raw = str(value)
    if not raw.strip():
        return value
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, ROOT / path]
    if not path.is_absolute() and "/" in raw:
        provider, model_name = raw.split("/", 1)
        candidates.extend(
            [
                ROOT / "checkpoints" / "hf" / provider / model_name,
                ROOT / "checkpoints" / "asr" / f"{provider}-{model_name}",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return raw


def load_s3_tokenizer(model_name_or_path: str | None = None, device: str | None = None) -> Any:
    """Load the S3 semantic tokenizer used for real token metrics."""

    model = _resolve_local_model_path(model_name_or_path or _default_s3_tokenizer_model())
    resolved_device = device or os.getenv("SEME2E_TOKENIZER_DEVICE") or os.getenv("SEME2E_API_DEVICE") or "cpu"
    cache_key = (str(model), str(resolved_device))
    if cache_key not in _S3_TOKENIZER_CACHE:
        import s3tokenizer

        _S3_TOKENIZER_CACHE[cache_key] = s3tokenizer.load_model(model).to(resolved_device).eval()
    return _S3_TOKENIZER_CACHE[cache_key]


def _to_device(value: Any, device: str) -> Any:
    return value.to(device) if hasattr(value, "to") else value


def encode_s3_tokens(audio_path: Path | str) -> list[int]:
    import s3tokenizer
    import torch

    device = os.getenv("SEME2E_TOKENIZER_DEVICE") or os.getenv("SEME2E_API_DEVICE") or "cpu"
    tokenizer = load_s3_tokenizer(device=device)
    try:
        audio = s3tokenizer.load_audio(str(audio_path))
    except Exception:
        waveform, sample_rate = _read_audio(Path(audio_path))
        if sample_rate != 16000:
            waveform = _resample(waveform, sample_rate, 16000)
        audio = torch.from_numpy(waveform.astype(np.float32))
    mel = s3tokenizer.log_mel_spectrogram(audio)
    mels, mels_lens = s3tokenizer.padding([mel])
    codes, codes_lens = tokenizer.quantize(_to_device(mels, device), _to_device(mels_lens, device))
    length = int(codes_lens[0].item() if hasattr(codes_lens[0], "item") else codes_lens[0])
    tokens = codes[0, :length].detach().cpu().long().flatten().tolist()
    return [int(item) for item in tokens]


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        data = np.asarray(audio, dtype=np.float32)
    except Exception:
        import wave

        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            sr = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
        if width == 1:
            raw = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
            data = (raw - 128.0) / 128.0
        elif width == 2:
            data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 4:
            data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported WAV sample width: {width}")
        if channels > 1:
            data = data.reshape(-1, channels)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 1.0:
        data = data / peak
    return np.clip(data, -1.0, 1.0).astype(np.float32), int(sr)


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32)
    try:
        from scipy.signal import resample_poly

        divisor = math.gcd(int(src_sr), int(dst_sr))
        up = int(dst_sr // divisor)
        down = int(src_sr // divisor)
        return resample_poly(audio, up, down).astype(np.float32)
    except Exception:
        duration = len(audio) / float(src_sr)
        target_len = max(1, int(round(duration * dst_sr)))
        old_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        new_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        return np.interp(new_x, old_x, audio).astype(np.float32)


def align_audio_pair(clean_path: Path, protected_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    x, sr = _read_audio(clean_path)
    xp, srp = _read_audio(protected_path)
    if srp != sr:
        xp = _resample(xp, srp, sr)
    n = min(len(x), len(xp))
    if n <= 0:
        raise ValueError("audio pair has no overlapping samples")
    x = x[:n].astype(np.float32)
    xp = xp[:n].astype(np.float32)
    delta = (xp - x).astype(np.float32)
    return x, xp, delta, int(sr)


def compute_clone_pair_perceptual_metrics(
    clean_clone_path: Path,
    protected_clone_path: Path,
) -> dict[str, Any]:
    """Compute reference-based clone-pair PESQ/STOI without introducing SNR."""

    sources: dict[str, dict[str, Any]] = {}
    try:
        clean, protected, _, sample_rate = align_audio_pair(clean_clone_path, protected_clone_path)
    except Exception as exc:
        reason = str(exc)
        return {
            "clonePairPesq": None,
            "clonePairStoi": None,
            "sampleRate": None,
            "status": "error",
            "reason": reason,
            "_metricSources": {
                "cloneEval.clonePairPesq": metric_source("error", "pesq", reason=reason, formula="PESQ(cleanClone,protectedClone)"),
                "cloneEval.clonePairStoi": metric_source("error", "pystoi", reason=reason, formula="STOI(cleanClone,protectedClone)"),
            },
        }

    pesq_value = None
    if not _module_available("pesq"):
        sources["cloneEval.clonePairPesq"] = metric_source(
            "unavailable",
            "pesq",
            reason="Python package 'pesq' is not installed",
            formula="PESQ(cleanClone,protectedClone)",
        )
    else:
        try:
            from pesq import pesq

            pesq_sample_rate = sample_rate if sample_rate in {8000, 16000} else 16000
            pesq_clean = clean if pesq_sample_rate == sample_rate else _resample(clean, sample_rate, pesq_sample_rate)
            pesq_protected = protected if pesq_sample_rate == sample_rate else _resample(protected, sample_rate, pesq_sample_rate)
            pesq_value = float(
                pesq(
                    pesq_sample_rate,
                    pesq_clean,
                    pesq_protected,
                    "wb" if pesq_sample_rate == 16000 else "nb",
                )
            )
            sources["cloneEval.clonePairPesq"] = metric_source(
                "available",
                "pesq",
                reason=None if pesq_sample_rate == sample_rate else f"Audio was resampled from {sample_rate} Hz to 16000 Hz for PESQ compatibility",
                formula="PESQ(cleanClone,protectedClone)",
            )
        except Exception as exc:
            sources["cloneEval.clonePairPesq"] = metric_source(
                "error",
                "pesq",
                reason=str(exc),
                formula="PESQ(cleanClone,protectedClone)",
            )

    stoi_value = None
    if not _module_available("pystoi"):
        sources["cloneEval.clonePairStoi"] = metric_source(
            "unavailable",
            "pystoi",
            reason="Python package 'pystoi' is not installed",
            formula="STOI(cleanClone,protectedClone)",
        )
    else:
        try:
            from pystoi.stoi import stoi

            stoi_value = float(stoi(clean, protected, sample_rate, extended=False))
            sources["cloneEval.clonePairStoi"] = metric_source(
                "available",
                "pystoi",
                formula="STOI(cleanClone,protectedClone)",
            )
        except Exception as exc:
            sources["cloneEval.clonePairStoi"] = metric_source(
                "error",
                "pystoi",
                reason=str(exc),
                formula="STOI(cleanClone,protectedClone)",
            )

    available_count = sum(value is not None for value in (pesq_value, stoi_value))
    reasons = [
        str(source.get("reason"))
        for source in sources.values()
        if source.get("reason") and source.get("status") != "available"
    ]
    return {
        "clonePairPesq": pesq_value,
        "clonePairStoi": stoi_value,
        "sampleRate": sample_rate,
        "status": "available" if available_count == 2 else "partial" if available_count else "unavailable",
        "reason": "; ".join(dict.fromkeys(reasons)) or None,
        "_metricSources": sources,
    }


def compute_perturbation_metrics(
    x: np.ndarray,
    xp: np.ndarray,
    delta: np.ndarray,
    sr: int,
    epsilon: float | None = None,
    epsilon_norm: str = "linf",
) -> dict[str, Any]:
    del sr
    l2_norm = float(np.sqrt(np.sum(delta.astype(np.float64) ** 2)))
    l2_rms = float(np.sqrt(np.mean(delta.astype(np.float64) ** 2)))
    linf_norm = float(np.max(np.abs(delta))) if delta.size else None
    signal_power = float(np.mean(x.astype(np.float64) ** 2))
    noise_power = float(np.mean(delta.astype(np.float64) ** 2))
    snr = 10.0 * math.log10((signal_power + EPS) / (noise_power + EPS))
    clipping_rate = float(np.mean(np.abs(xp) >= 0.999)) if xp.size else None
    epsilon_value = finite_float(epsilon)
    epsilon_norm_value = str(epsilon_norm or "linf").lower()
    epsilon_usage_rate = None
    epsilon_usage_rate_raw = None
    epsilon_tolerance_rate = 0.0
    epsilon_exceeded = None
    if epsilon_value is not None:
        if epsilon_norm_value == "linf" and linf_norm is not None:
            epsilon_usage_rate_raw = linf_norm / max(epsilon_value, EPS)
            # The optimizer projects the floating-point perturbation to epsilon,
            # but exporting and reading a PCM16 WAV can move one sample by one
            # quantization step. Treat only that narrow serialization tolerance
            # as 100% utilization; larger overruns remain visible as violations.
            epsilon_tolerance_rate = PCM16_QUANTIZATION_STEP / max(epsilon_value, EPS)
            epsilon_exceeded = epsilon_usage_rate_raw > 1.0 + epsilon_tolerance_rate + EPS
            epsilon_usage_rate = epsilon_usage_rate_raw if epsilon_exceeded else min(epsilon_usage_rate_raw, 1.0)
        elif epsilon_norm_value == "l2":
            epsilon_usage_rate_raw = l2_norm / max(epsilon_value, EPS)
            epsilon_usage_rate = epsilon_usage_rate_raw
            epsilon_exceeded = epsilon_usage_rate_raw > 1.0 + EPS
    return {
        "l2Norm": l2_norm,
        "l2Rms": l2_rms,
        "linfNorm": linf_norm,
        "epsilon": epsilon_value,
        "epsilonNorm": epsilon_norm_value,
        "epsilonUsageRate": epsilon_usage_rate,
        "epsilonUsageRateRaw": epsilon_usage_rate_raw,
        "epsilonToleranceRate": epsilon_tolerance_rate,
        "epsilonExceeded": epsilon_exceeded,
        "snr": snr,
        "clippingRate": clipping_rate,
    }


def compute_quality_metrics(
    x: np.ndarray,
    xp: np.ndarray,
    delta: np.ndarray,
    sr: int,
    perturbation_metrics: dict[str, Any],
    dns_mos: float | None = None,
    dns_mos_status: str | None = None,
    dns_mos_reason: str | None = None,
) -> dict[str, Any]:
    del delta
    sources: dict[str, dict[str, Any]] = {
        "protectionQuality.snr": metric_source("available", "compute_perturbation_metrics", formula="10*log10((P_signal+1e-12)/(P_noise+1e-12))"),
        "protectionQuality.dnsMos": metric_source(
            "available" if finite_float(dns_mos) is not None else (dns_mos_status or "unavailable"),
            "DNSMOS P.835 OVRL",
            reason=None if finite_float(dns_mos) is not None else (dns_mos_reason or "语音质量评分尚未生成"),
            formula="100*(DNSMOS_OVRL-1)/4 when 1<=DNSMOS_OVRL<=5",
        ),
        "protectionQuality.mosLqo": metric_source("unavailable", "objective_mos_lqo_model", reason="No explicit MOS-LQO objective model is configured", formula="None without an explicit MOS-LQO model"),
    }
    pesq_value = None
    if not _module_available("pesq"):
        sources["protectionQuality.pesq"] = metric_source("unavailable", "pesq", reason="Python package 'pesq' is not installed", formula="pesq(sr, x, xp, mode)")
    else:
        try:
            from pesq import pesq

            pesq_sr = sr if sr in {8000, 16000} else 16000
            pesq_x = x if pesq_sr == sr else _resample(x, sr, pesq_sr)
            pesq_xp = xp if pesq_sr == sr else _resample(xp, sr, pesq_sr)
            pesq_value = float(pesq(pesq_sr, pesq_x, pesq_xp, "wb" if pesq_sr == 16000 else "nb"))
            reason = None if pesq_sr == sr else f"Audio was resampled from {sr} Hz to 16000 Hz for PESQ compatibility"
            sources["protectionQuality.pesq"] = metric_source("available", "pesq", reason=reason, formula="pesq(sr_supported, x, xp, mode)")
        except Exception as exc:
            sources["protectionQuality.pesq"] = metric_source("error", "pesq", reason=str(exc), formula="pesq(sr, x, xp, mode)")

    stoi_value = None
    if not _module_available("pystoi"):
        sources["protectionQuality.stoi"] = metric_source("unavailable", "pystoi", reason="Python package 'pystoi' is not installed", formula="stoi(x, xp, sr)")
    else:
        try:
            from pystoi.stoi import stoi

            stoi_value = float(stoi(x, xp, sr, extended=False))
            sources["protectionQuality.stoi"] = metric_source("available", "pystoi", formula="stoi(x, xp, sr)")
        except Exception as exc:
            sources["protectionQuality.stoi"] = metric_source("error", "pystoi", reason=str(exc), formula="stoi(x, xp, sr)")

    snr = finite_float(perturbation_metrics.get("snr"))
    score_payload = compute_protection_quality_score(snr, stoi_value, pesq_value, dns_mos)
    if score_payload.get("dnsMos") is None:
        score_payload["dnsMosStatus"] = dns_mos_status or "unavailable"
        score_payload["dnsMosReason"] = dns_mos_reason or "语音质量评分尚未生成"
    quality_score = finite_float(score_payload.get("qualityScore"))
    if quality_score is None:
        quality_level = None
    elif quality_score >= 85:
        quality_level = "excellent"
    elif quality_score >= 70:
        quality_level = "good"
    elif quality_score >= 50:
        quality_level = "fair"
    else:
        quality_level = "poor"
    sources["protectionQuality.qualityScore"] = metric_source(
        "available" if quality_score is not None else "unavailable",
        "VoiceShield_v2.1_piecewise_quality",
        reason=score_payload.get("scoreReason"),
        formula="without DNSMOS: (.40*S_snr+.35*S_stoi+.15*S_pesq)/.90; with DNSMOS: .40*S_snr+.35*S_stoi+.15*S_pesq+.10*S_dnsmos",
    )
    sources["protectionQuality.qualityLevel"] = metric_source(
        "available" if quality_level is not None else "unavailable",
        "qualityScore_thresholds",
        reason=None if quality_level is not None else "qualityScore is unavailable",
        formula="excellent>=85, good>=70, fair>=50, else poor",
    )
    return {
        "snr": snr,
        "pesq": pesq_value,
        "stoi": stoi_value,
        "mos": None,
        "mosLqo": None,
        "snrScore": score_payload.get("snrScore"),
        "stoiScore": score_payload.get("stoiScore"),
        "pesqScore": score_payload.get("pesqScore"),
        "dnsMos": score_payload.get("dnsMos"),
        "dnsMosScore": score_payload.get("dnsMosScore"),
        "dnsMosStatus": score_payload.get("dnsMosStatus"),
        "dnsMosReason": score_payload.get("dnsMosReason"),
        "qualityScore": quality_score,
        "qualityLevel": quality_level,
        "scoreStatus": score_payload.get("scoreStatus"),
        "scoreReason": score_payload.get("scoreReason"),
        "_metricSources": sources,
    }


def _stft_with_meta(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        from scipy.signal import stft

        nperseg = min(1024, max(64, len(audio)))
        noverlap = nperseg // 2
        freqs, _, zxx = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary=None)
        return freqs.astype(np.float64), zxx, {"nFft": int(nperseg), "hopLength": int(nperseg - noverlap), "center": False}
    except Exception:
        nfft = min(1024, max(64, int(2 ** math.floor(math.log2(max(len(audio), 64))))))
        hop = max(1, nfft // 2)
        window = np.hanning(nfft).astype(np.float32)
        frames = []
        for start in range(0, max(1, len(audio) - nfft + 1), hop):
            frame = audio[start : start + nfft]
            if len(frame) < nfft:
                frame = np.pad(frame, (0, nfft - len(frame)))
            frames.append(np.fft.rfft(frame * window))
        freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)
        return freqs.astype(np.float64), np.asarray(frames, dtype=np.complex64).T, {"nFft": int(nfft), "hopLength": int(hop), "center": False}


def _stft(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    freqs, zxx, _ = _stft_with_meta(audio, sr)
    return freqs, zxx


def _absolute_threshold_hearing(freqs: np.ndarray) -> np.ndarray:
    khz = np.maximum(freqs / 1000.0, 0.02)
    ath = 3.64 * np.power(khz, -0.8) - 6.5 * np.exp(-0.6 * np.power(khz - 3.3, 2.0)) + 0.001 * np.power(khz, 4.0)
    return np.clip(ath - 80.0, -120.0, 40.0)


def _psychoacoustic_state(x: np.ndarray, delta: np.ndarray, sr: int) -> dict[str, Any]:
    freqs, x_stft, stft_meta = _stft_with_meta(x, sr)
    _, d_stft, _ = _stft_with_meta(delta, sr)
    min_time = min(x_stft.shape[1], d_stft.shape[1])
    x_power = np.abs(x_stft[:, :min_time]) ** 2
    d_power = np.abs(d_stft[:, :min_time]) ** 2
    psd_delta = 10.0 * np.log10(d_power + EPS)
    psd_signal = 10.0 * np.log10(x_power + EPS)
    ath = _absolute_threshold_hearing(freqs)[:, None]
    theta = np.maximum(ath, psd_signal - 18.0)
    violation = np.maximum(0.0, psd_delta - theta)
    return {
        "freqs": freqs,
        "psdDelta": psd_delta,
        "theta": theta,
        "violation": violation,
        "sampleRate": int(sr),
        "hopLength": int(stft_meta["hopLength"]),
        "nFft": int(stft_meta["nFft"]),
        "center": bool(stft_meta.get("center")),
        "frameCount": int(min_time),
    }


def _psychoacoustic_curve(freqs: np.ndarray, threshold: np.ndarray, perturbation: np.ndarray) -> dict[str, Any]:
    stride = max(1, int(math.ceil(len(freqs) / 96)))
    masking_threshold = [
        {"frequencyHz": float(freqs[i]), "thresholdDb": float(threshold[i])}
        for i in range(0, len(freqs), stride)
    ]
    perturbation_spectrum = [
        {"frequencyHz": float(freqs[i]), "powerDb": float(perturbation[i])}
        for i in range(0, len(freqs), stride)
    ]
    chart = [
        {
            "frequency": item["frequencyHz"],
            "maskingThreshold": item["thresholdDb"],
            "perturbation": perturbation_spectrum[index]["powerDb"],
            "perturbationPsd": perturbation_spectrum[index]["powerDb"],
        }
        for index, item in enumerate(masking_threshold)
        if index < len(perturbation_spectrum)
    ]
    return {
        "maskingThreshold": masking_threshold,
        "perturbationSpectrum": perturbation_spectrum,
        "chart": chart,
    }


def compute_psychoacoustic_metrics(x: np.ndarray, xp: np.ndarray, delta: np.ndarray, sr: int) -> dict[str, Any]:
    del xp
    state = _psychoacoustic_state(x, delta, sr)
    freqs = state["freqs"]
    psd_delta = state["psdDelta"]
    theta = state["theta"]
    violation = state["violation"]
    l_psy = float(np.mean(violation)) if violation.size else None
    over_mask_rate = float(np.mean(violation > 0.0)) if violation.size else None
    threshold = np.mean(theta, axis=1) if theta.size else np.asarray([], dtype=np.float64)
    perturbation = np.mean(psd_delta, axis=1) if psd_delta.size else np.asarray([], dtype=np.float64)
    curve = _psychoacoustic_curve(freqs, threshold, perturbation)
    sources = {
        "psychoacoustic.*": metric_source(
            "available",
            "engineering_stft_masking_threshold",
            reason="Engineering approximation from signal STFT PSD and absolute-threshold curve; not a calibrated psychoacoustic model",
            formula="Theta=max(ATH,PSD_signal-18); V=max(0,PSD_delta-Theta); lPsy=mean_{t,f}(V); overMaskRate=mean_{t,f}(V>0); maskingThreshold[f]=mean_t(Theta[t,f]); perturbationSpectrum[f]=mean_t(PSD_delta[t,f])",
        ),
        "psychoacoustic.slice": metric_source(
            "available",
            "stft_psychoacoustic_lazy_slice",
            reason="默认结果只保存时间平均曲线；选择具体时刻后再计算对应单帧曲线",
            formula="mean mode: maskingThreshold[f]=mean_t(Theta[t,f]); perturbationSpectrum[f]=mean_t(PSD_delta[t,f]); lPsy and overMaskRate are full time-frequency statistics",
        ),
    }
    return {
        "lPsy": l_psy,
        "overMaskRate": over_mask_rate,
        "frameCount": state["frameCount"],
        "sampleRate": state["sampleRate"],
        "hopLength": state["hopLength"],
        "nFft": state["nFft"],
        "aggregation": "time_mean",
        "maskingThreshold": curve["maskingThreshold"],
        "perturbationSpectrum": curve["perturbationSpectrum"],
        "chart": curve["chart"],
        "_metricSources": sources,
    }


def compute_psychoacoustic_slice(
    x: np.ndarray,
    xp: np.ndarray,
    delta: np.ndarray,
    sr: int,
    mode: str = "mean",
    time_sec: float | None = None,
    duration_sec: float | None = None,
) -> dict[str, Any]:
    del xp
    mode = (mode or "mean").strip().lower()
    if mode not in {"mean", "frame"}:
        raise ValueError("mode must be 'mean' or 'frame'")

    state = _psychoacoustic_state(x, delta, sr)
    freqs = state["freqs"]
    psd_delta = state["psdDelta"]
    theta = state["theta"]
    violation = state["violation"]
    frame_count = int(state["frameCount"])
    sample_rate = int(state["sampleRate"])
    hop_length = int(state["hopLength"])
    audio_duration = float(duration_sec) if duration_sec is not None and math.isfinite(float(duration_sec)) else (len(x) / float(sr) if sr else 0.0)
    requested_time: float | None = None
    actual_time: float | None = None
    frame_index: int | None = None

    if mode == "frame":
        if time_sec is None:
            raise ValueError("timeSec is required when mode=frame")
        requested_time = float(time_sec)
        if not math.isfinite(requested_time) or requested_time < 0.0 or requested_time > audio_duration:
            raise ValueError(f"timeSec must be between 0 and {audio_duration:.6f} seconds")
        if frame_count <= 0:
            raise ValueError("psychoacoustic STFT has no frames")
        frame_index = int(math.floor((requested_time * sample_rate / max(1, hop_length)) + 0.5))
        frame_index = min(max(frame_index, 0), frame_count - 1)
        actual_time = frame_index * hop_length / float(sample_rate) if sample_rate else None
        threshold = theta[:, frame_index] if theta.size else np.asarray([], dtype=np.float64)
        perturbation = psd_delta[:, frame_index] if psd_delta.size else np.asarray([], dtype=np.float64)
        aggregation = "single_frame"
        formula = (
            "frame mode: frameIndex=round(timeSec*sampleRate/hopLength), clamped to [0,frameCount-1]; "
            "maskingThreshold[f]=Theta[frameIndex,f]; perturbationSpectrum[f]=PSD_delta[frameIndex,f]; "
            "lPsy and overMaskRate are full time-frequency statistics"
        )
    else:
        threshold = np.mean(theta, axis=1) if theta.size else np.asarray([], dtype=np.float64)
        perturbation = np.mean(psd_delta, axis=1) if psd_delta.size else np.asarray([], dtype=np.float64)
        aggregation = "time_mean"
        formula = (
            "mean mode: maskingThreshold[f]=mean_t(Theta[t,f]); perturbationSpectrum[f]=mean_t(PSD_delta[t,f]); "
            "lPsy and overMaskRate are full time-frequency statistics"
        )

    curve = _psychoacoustic_curve(freqs, threshold, perturbation)
    l_psy = float(np.mean(violation)) if violation.size else None
    over_mask_rate = float(np.mean(violation > 0.0)) if violation.size else None
    source_reason = "STFT uses center=False; actualTimeSec is frameIndex*hopLength/sampleRate"
    return {
        "mode": mode,
        "requestedTimeSec": requested_time,
        "actualTimeSec": actual_time,
        "frameIndex": frame_index,
        "frameCount": frame_count,
        "sampleRate": sample_rate,
        "hopLength": hop_length,
        "nFft": int(state["nFft"]),
        "aggregation": aggregation,
        "lPsy": l_psy,
        "overMaskRate": over_mask_rate,
        "maskingThreshold": curve["maskingThreshold"],
        "perturbationSpectrum": curve["perturbationSpectrum"],
        "charts": {"psychoacoustic": curve["chart"]},
        "metricSources": {
            "psychoacoustic.slice": metric_source(
                "available",
                "stft_psychoacoustic_lazy_slice",
                reason=source_reason,
                formula=formula,
            )
        },
    }


def _read_weight(config: dict[str, Any], group: str, primary: str, alias: str) -> float | None:
    section = config.get(group) or {}
    return finite_float(section.get(primary, section.get(alias)))


def _read_identity_weight(config: dict[str, Any]) -> float | None:
    timbre = config.get("timbre") or {}
    optimization = config.get("optimization") or {}
    candidates = [
        timbre.get("lambdaId"),
        timbre.get("lambdaIdentity"),
        timbre.get("weightIdentity"),
        timbre.get("weight_identity"),
        timbre.get("lambdaFeat"),
        timbre.get("lambdaTimbre"),
        timbre.get("weightFeature"),
        timbre.get("weight_feature"),
        optimization.get("lambdaId"),
        optimization.get("weightIdentity"),
        optimization.get("weight_identity"),
    ]
    for candidate in candidates:
        value = finite_float(candidate)
        if value is not None:
            return value
    return None


def _normalize_loss_point(item: dict[str, Any], index: int, weights: dict[str, float | None]) -> dict[str, Any] | None:
    lid = None
    for key in ["Lid", "lId", "lossIdentity", "loss_identity", "Lfeat", "Lfea", "lossFeature", "loss_timbre", "L_feature"]:
        lid = finite_float(item.get(key))
        if lid is not None:
            break
    point = {
        "step": finite_float(item.get("step", item.get("epoch", item.get("iteration", item.get("iter", index))))),
        "Lid": lid,
        "Lfeat": lid,
        "Lsem": finite_float(item.get("Lsem", item.get("lossSemantic", item.get("loss_semantic", item.get("L_semantic"))))),
        "Lpsy": finite_float(item.get("Lpsy", item.get("lossPsy", item.get("loss_psy", item.get("L_psy"))))),
        "L2": finite_float(item.get("L2", item.get("lossL2", item.get("loss_l2", item.get("l2Norm"))))),
        "total": finite_float(item.get("total", item.get("lossTotal", item.get("loss_total", item.get("objective"))))),
        "snr": finite_float(item.get("snr", item.get("SNR"))),
        "stepElapsedSec": finite_float(item.get("stepElapsedSec", item.get("elapsedSec", item.get("step_time")))),
    }
    if point["total"] is None and all(point[key] is not None for key in ["Lid", "Lsem", "Lpsy", "L2"]):
        if all(weights.get(key) is not None for key in ["lambdaId", "lambdaSem", "lambdaPsy", "lambda2"]):
            point["total"] = (
                float(weights["lambdaId"]) * float(point["Lid"])
                + float(weights["lambdaSem"]) * float(point["Lsem"])
                + float(weights["lambdaPsy"]) * float(point["Lpsy"])
                + float(weights["lambda2"]) * float(point["L2"])
            )
    if any(point[key] is not None for key in ["Lid", "Lsem", "Lpsy", "L2", "total"]):
        return point
    return None


def compute_loss_summary(
    protection_details: dict[str, Any],
    request_config: dict[str, Any],
    x: np.ndarray | None = None,
    xp: np.ndarray | None = None,
    delta: np.ndarray | None = None,
) -> dict[str, Any]:
    del x, xp, delta
    lambda_id = _read_identity_weight(request_config)
    weights = {
        "lambdaId": lambda_id,
        "lambdaFeat": lambda_id,
        "lambdaSem": _read_weight(request_config, "semantic", "lambdaSemantic", "weightSemantic"),
        "lambdaPsy": _read_weight(request_config, "psychoacoustic", "lambdaPsy", "weightPsy"),
        "lambda2": _read_weight(request_config, "optimization", "lambdaL2", "weightL2"),
    }
    raw_trace = protection_details.get("optimization_trace") or protection_details.get("optimizationTrace") or []
    trace = []
    if isinstance(raw_trace, list):
        for index, item in enumerate(raw_trace):
            if isinstance(item, dict):
                point = _normalize_loss_point(item, index, weights)
                if point is not None:
                    trace.append(point)
    step_times = [float(item["stepElapsedSec"]) for item in trace if item.get("stepElapsedSec") is not None]
    average_step_sec = float(np.mean(step_times)) if step_times else finite_float(protection_details.get("average_step_sec"))
    selected_step = finite_float(protection_details.get("selected_step", protection_details.get("selectedStep")))
    selected_index = int(selected_step) - 1 if selected_step is not None else None
    if trace and selected_index is not None and 0 <= selected_index < len(trace):
        loss_final = trace[selected_index]
    else:
        loss_final = trace[-1] if trace else None
    source_status = "available" if trace else "unavailable"
    sources = {
        "lossFinal.*": metric_source(
            source_status,
            protection_details.get("source") or "VoiceShield.protect",
            reason=None if trace else "Protection backend did not return an optimization trace",
            formula="lossFinal=optimizationTrace[selectedStep-1] when selectedStep is available, otherwise optimizationTrace[-1]",
        ),
        "optimizationTrace": metric_source(
            source_status,
            protection_details.get("source") or "VoiceShield.protect",
            reason=None if trace else "Protection backend did not return per-step loss records",
            formula="normalized backend trace; total=lambdaId*Lid+lambdaSem*Lsem+lambdaPsy*Lpsy+lambda2*L2 when total is absent",
        ),
        "lossFinal.Lid": metric_source(
            source_status,
            protection_details.get("source") or "VoiceShield.protect",
            reason=None if trace else "Protection backend did not return an optimization trace",
            formula="L_{\\mathrm{id}}",
            metric="Identity loss from VoiceShield optimization trace.",
        ),
        "optimizationTrace.Lid": metric_source(
            source_status,
            protection_details.get("source") or "VoiceShield.protect",
            reason=None if trace else "Protection backend did not return per-step loss records",
            formula="L_{\\mathrm{id}}",
            metric="Identity loss from VoiceShield optimization trace.",
        ),
        "lossFinal.Lfeat": metric_source(
            source_status,
            protection_details.get("source") or "VoiceShield.protect",
            reason="Deprecated legacy alias of Lid.",
            formula="Lfeat := Lid",
        ),
        "optimizationTrace.Lfeat": metric_source(
            source_status,
            protection_details.get("source") or "VoiceShield.protect",
            reason="Deprecated legacy alias of Lid.",
            formula="Lfeat := Lid",
        ),
        "averageStepSec": metric_source(
            "available" if average_step_sec is not None else "unavailable",
            "optimizationTrace.stepElapsedSec",
            reason=None if average_step_sec is not None else "No stepElapsedSec values were present in the optimization trace",
            formula="mean(stepElapsedSec)",
        ),
    }
    return {
        "lossFinal": loss_final,
        "lossWeights": weights,
        "optimizationTrace": trace,
        "averageStepSec": average_step_sec,
        "selectedStep": int(selected_step) if selected_step is not None else None,
        "_metricSources": sources,
    }


def _normalize_words(text: str) -> list[str]:
    text = text.lower()
    # Latin/digit runs remain word units; CJK characters are individual units
    # because ordinary Chinese transcripts do not contain whitespace boundaries.
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*|[\u4e00-\u9fff]", text)


def _normalize_chars(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    return list(text)


def _edit_counts_and_ops(reference: Sequence[Any], hypothesis: Sequence[Any]) -> tuple[int, int, int, list[dict[str, Any]]]:
    n = len(reference)
    m = len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "delete"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "insert"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                best = (dp[i - 1][j - 1], "equal")
            else:
                best = (dp[i - 1][j - 1] + 1, "replace")
            delete = (dp[i - 1][j] + 1, "delete")
            insert = (dp[i][j - 1] + 1, "insert")
            dp[i][j], back[i][j] = min(best, delete, insert, key=lambda item: item[0])
    i, j = n, m
    raw_ops: list[dict[str, Any]] = []
    substitutions = deletions = insertions = 0
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "equal":
            raw_ops.append({"type": "equal", "text": reference[i - 1]})
            i -= 1
            j -= 1
        elif op == "replace":
            substitutions += 1
            raw_ops.append({"type": "replace", "from": reference[i - 1], "to": hypothesis[j - 1]})
            i -= 1
            j -= 1
        elif op == "delete":
            deletions += 1
            raw_ops.append({"type": "delete", "text": reference[i - 1]})
            i -= 1
        else:
            insertions += 1
            raw_ops.append({"type": "insert", "text": hypothesis[j - 1]})
            j -= 1
    raw_ops.reverse()
    return substitutions, deletions, insertions, _merge_diff_ops(raw_ops)


def _merge_diff_ops(raw_ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for op in raw_ops:
        if not merged or op["type"] != merged[-1]["type"] or op["type"] == "replace":
            merged.append(dict(op))
            continue
        if "text" in op:
            joiner = "" if len(str(op["text"])) == 1 and len(str(merged[-1].get("text", ""))) <= 1 else " "
            merged[-1]["text"] = f"{merged[-1]['text']}{joiner}{op['text']}".strip()
    return merged


def compute_asr_metrics(
    original_text: str | None,
    protected_text: str | None,
    reference_text: str | None = None,
    language: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    original = original_text or ""
    protected = protected_text or ""
    reference = reference_text if reference_text is not None else original
    lang = language or None
    metric_level = "char" if (lang or "").lower().startswith("zh") else "word"
    word_ref = _normalize_words(reference)
    word_hyp = _normalize_words(protected)
    char_ref = _normalize_chars(reference)
    char_hyp = _normalize_chars(protected)
    s_word, d_word, i_word, _ = _edit_counts_and_ops(word_ref, word_hyp)
    s_char, d_char, i_char, _ = _edit_counts_and_ops(char_ref, char_hyp)
    wer = (s_word + d_word + i_word) / max(len(word_ref), 1)
    cer = (s_char + d_char + i_char) / max(len(char_ref), 1)
    if metric_level == "char":
        substitutions, deletions, insertions, diff_ops = _edit_counts_and_ops(char_ref, char_hyp)
        normalizer = max(len(char_ref), 1)
        m = cer
        reference_length = len(char_ref)
    else:
        substitutions, deletions, insertions, diff_ops = _edit_counts_and_ops(word_ref, word_hyp)
        normalizer = max(len(word_ref), 1)
        m = wer
        reference_length = len(word_ref)
    total_errors = substitutions + deletions + insertions
    substitute_rate = substitutions / normalizer
    insert_rate = insertions / normalizer
    delete_rate = deletions / normalizer
    error_normalizer = max(total_errors, 1)
    return {
        "model": model,
        "asrModel": model,
        "language": lang,
        "originalText": original,
        "protectedText": protected,
        "referenceText": reference,
        "cleanTranscription": original,
        "protectedTranscription": protected,
        "wer": wer,
        "cer": cer,
        "insertRate": insert_rate,
        "deleteRate": delete_rate,
        "substituteRate": substitute_rate,
        "editCounts": {
            "level": metric_level,
            "referenceLength": reference_length,
            "substitutions": substitutions,
            "insertions": insertions,
            "deletions": deletions,
            "totalErrors": total_errors,
        },
        "errorShares": {
            "substituteShare": substitutions / error_normalizer if total_errors else 0.0,
            "insertShare": insertions / error_normalizer if total_errors else 0.0,
            "deleteShare": deletions / error_normalizer if total_errors else 0.0,
        },
        "breakdown": {
            "insertRate": insert_rate,
            "deleteRate": delete_rate,
            "substituteRate": substitute_rate,
        },
        "metricLevel": metric_level,
        "asrProtectionScore": 100.0 * clamp(m / 0.5),
        "diffOps": diff_ops,
        "trend": None,
        "createdAt": now_iso(),
        "status": "available",
        "_metricSources": {
            "asrEval.*": metric_source("available", "ASRTranscriber + Levenshtein", formula="WER/CER=(S+D+I)/max(N,1); score=100*clamp(m/0.5,0,1)"),
            "asrEval.trend": metric_source("not_run", "ASR trend evaluator", reason="No repeated ASR checkpoints were evaluated", formula="None"),
        },
    }


def _edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    substitutions, deletions, insertions, _ = _edit_counts_and_ops(reference, hypothesis)
    return substitutions + deletions + insertions


def _pool_vector(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            tensor = value.detach().float().cpu()
            if tensor.ndim == 0:
                tensor = tensor.reshape(1)
            if tensor.ndim >= 2:
                tensor = tensor.reshape(-1, tensor.shape[-1]).mean(dim=0)
            return tensor.numpy().astype(np.float64).reshape(-1)
    except Exception:
        pass
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim >= 2:
        arr = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    return arr.reshape(-1)


def _cosine_distance(clean_vec: Any, protected_vec: Any) -> tuple[float, float]:
    clean = _pool_vector(clean_vec)
    protected = _pool_vector(protected_vec)
    dim = min(clean.shape[0], protected.shape[0])
    if dim <= 0:
        raise ValueError("empty semantic vector")
    clean = clean[:dim]
    protected = protected[:dim]
    denom = float(np.linalg.norm(clean) * np.linalg.norm(protected) + EPS)
    cosine = float(np.dot(clean, protected) / denom)
    return cosine, clamp(1.0 - cosine)


def _framewise_cosine_similarity(clean_vec: Any, protected_vec: Any) -> float:
    import torch
    import torch.nn.functional as F

    clean = torch.as_tensor(clean_vec).detach().float().cpu()
    protected = torch.as_tensor(protected_vec).detach().float().cpu()
    if clean.ndim == 0 or protected.ndim == 0:
        raise ValueError("empty semantic vector")
    if clean.ndim == 2:
        clean = clean.unsqueeze(0)
    if protected.ndim == 2:
        protected = protected.unsqueeze(0)
    try:
        return float(F.cosine_similarity(clean, protected, dim=-1, eps=1.0e-6).mean().item())
    except RuntimeError:
        min_len = min(clean.shape[-1], protected.shape[-1])
        if min_len <= 0:
            raise ValueError("empty semantic vector")
        clean = clean[..., :min_len]
        protected = protected[..., :min_len]
        if clean.ndim >= 2 and protected.ndim >= 2:
            min_seq = min(clean.shape[-2], protected.shape[-2])
            if min_seq <= 0:
                raise ValueError("empty semantic sequence")
            clean = clean[..., :min_seq, :]
            protected = protected[..., :min_seq, :]
        return float(F.cosine_similarity(clean, protected, dim=-1, eps=1.0e-6).mean().item())


def _semantic_encoder_weight_map(config: dict[str, Any]) -> dict[str, float]:
    semantic = config.get("semantic") if isinstance(config.get("semantic"), dict) else {}
    raw = (
        config.get("encoderWeights")
        or config.get("semanticEncoderWeights")
        or semantic.get("encoderWeights")
        or semantic.get("semanticEncoderWeights")
    )
    weights: dict[str, float] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            canonical = _canonical_semantic_encoder_name(key)
            weight = finite_float(value)
            if canonical and weight is not None and weight > 0:
                weights[canonical] = weight
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            canonical = _canonical_semantic_encoder_name(item.get("encoder") or item.get("name") or item.get("value"))
            weight = finite_float(item.get("weight"))
            if canonical and weight is not None and weight > 0:
                weights[canonical] = weight
    return weights


def _load_audio_without_torchcodec(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        return mono.astype(np.float32, copy=False), int(sr)
    except Exception:
        pass

    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sr = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32, copy=False), int(sr)


def _canonical_semantic_encoder_name(value: Any) -> str | None:
    normalized = str(value or "").strip().lower().replace("_", "-")
    if not normalized:
        return None
    if normalized in {"s3", "speech-tokenizer", "semantic-tokenizer", "tokenizer", "cosyvoice"}:
        return "s3"
    if "hubert" in normalized:
        return "hubert"
    if "whisper" in normalized:
        return "whisper"
    if normalized == "mfcc" or "mfcc" in normalized:
        return "mfcc"
    return normalized


def _selected_semantic_encoder_names(config: dict[str, Any]) -> set[str] | None:
    semantic = config.get("semantic") if isinstance(config.get("semantic"), dict) else {}
    raw = config.get("encoders") or semantic.get("encoders") or config.get("selectedSemanticEncoders")
    if not isinstance(raw, list) or not raw:
        return None
    selected = {_canonical_semantic_encoder_name(item) for item in raw}
    selected = {item for item in selected if item}
    return selected or None


def _hf_model_is_cached(model_id: str) -> bool:
    model_path = Path(str(model_id)).expanduser()
    if model_path.exists():
        return True
    if "/" not in str(model_id):
        return False
    cache_root = (
        os.getenv("HUGGINGFACE_HUB_CACHE")
        or (str(Path(os.getenv("HF_HOME")).expanduser() / "hub") if os.getenv("HF_HOME") else None)
        or str(Path.home() / ".cache" / "huggingface" / "hub")
    )
    cache_name = "models--" + str(model_id).replace("/", "--")
    return (Path(cache_root) / cache_name).exists()


def _default_hubert_model() -> str:
    preferred = "facebook/hubert-large-ll60k"
    cached_fallbacks = ["facebook/hubert-large-ls960-ft"]
    if _hf_model_is_cached(preferred):
        return preferred
    for fallback in cached_fallbacks:
        if _hf_model_is_cached(fallback):
            return fallback
    return preferred


def _resolve_hf_or_local_model(requested: Any, cached_fallbacks: Sequence[str]) -> str:
    requested_path = Path(str(requested)).expanduser()
    if requested_path.exists():
        return str(requested_path.resolve())
    for fallback in cached_fallbacks:
        resolved_fallback = _resolve_local_model_path(fallback)
        if isinstance(resolved_fallback, str) and Path(resolved_fallback).expanduser().exists():
            return resolved_fallback
    resolved = _resolve_local_model_path(requested)
    if isinstance(resolved, str) and Path(resolved).expanduser().exists():
        return resolved
    if isinstance(resolved, str) and _hf_model_is_cached(resolved):
        return resolved
    for fallback in cached_fallbacks:
        resolved_fallback = _resolve_local_model_path(fallback)
        if isinstance(resolved_fallback, str) and _hf_model_is_cached(resolved_fallback):
            return resolved_fallback
    return str(resolved)


def _load_semantic_encoder_ensemble(config: dict[str, Any]) -> Any | None:
    global _SEMANTIC_ENCODER_LAST_ERROR
    if os.getenv("SEME2E_ENABLE_SEMANTIC_ENCODERS", "1") != "1":
        _SEMANTIC_ENCODER_LAST_ERROR = "Set SEME2E_ENABLE_SEMANTIC_ENCODERS=1 to run SemanticEncoderEnsemble"
        return None
    try:
        from semantic_encoders import SemanticEncoderEnsemble

        device = os.getenv("SEME2E_SEMANTIC_ENCODER_DEVICE") or os.getenv("SEME2E_API_DEVICE") or os.getenv("SEME2E_TOKENIZER_DEVICE") or "cpu"
        tokenizer_path = (
            os.getenv("SEME2E_SEMANTIC_TOKENIZER_MODEL")
            or os.getenv("SEME2E_TOKENIZER_MODEL")
            or config.get("tokenizerPath")
            or _default_s3_tokenizer_model()
        )
        tokenizer_path = _resolve_local_model_path(tokenizer_path)
        requested_hubert = os.getenv("SEME2E_HUBERT_MODEL") or config.get("hubertModel") or config.get("hubertPath") or _default_hubert_model()
        requested_whisper = os.getenv("SEME2E_WHISPER_MODEL") or config.get("whisperModel") or config.get("whisperPath") or "openai/whisper-large-v3"
        hubert_path = _resolve_hf_or_local_model(requested_hubert, [_default_hubert_model(), "facebook/hubert-large-ls960-ft"])
        whisper_path = _resolve_hf_or_local_model(
            requested_whisper,
            [str(ROOT / "checkpoints" / "asr" / "openai-whisper-small"), "openai/whisper-small", "openai/whisper-large-v3"],
        )
        cache_key = (str(device), str(tokenizer_path), str(hubert_path), str(whisper_path))
        ensemble = _SEMANTIC_ENCODER_CACHE.get(cache_key)
        if ensemble is None:
            with _SEMANTIC_ENCODER_CACHE_LOCK:
                ensemble = _SEMANTIC_ENCODER_CACHE.get(cache_key)
                if ensemble is None:
                    ensemble = SemanticEncoderEnsemble(
                        device=device,
                        tokenizer_path=tokenizer_path,
                        hubert_path=hubert_path,
                        whisper_path=whisper_path,
                        sample_rate=16000,
                    )
                    _SEMANTIC_ENCODER_CACHE[cache_key] = ensemble
        _SEMANTIC_ENCODER_LAST_ERROR = None
        return ensemble
    except Exception as exc:
        _SEMANTIC_ENCODER_LAST_ERROR = f"{type(exc).__name__}: {exc}"
        return None


def _compute_semantic_encoder_distances(clean_path: Path, protected_path: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    ensemble = _load_semantic_encoder_ensemble(config)
    if ensemble is None:
        return [], None
    import torch
    import torchaudio

    def load_wave(path: Path) -> Any:
        audio, sr = _load_audio_without_torchcodec(path)
        wave_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        if sr != 16000:
            wave_tensor = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wave_tensor)
        return wave_tensor.to(ensemble.device)

    with torch.no_grad():
        clean_vectors = ensemble.get_vectors(load_wave(clean_path))
        protected_vectors = ensemble.get_vectors(load_wave(protected_path))
    names = list(getattr(ensemble, "vector_names", [])) or ["s3", "hubert", "whisper", "mfcc"]
    display_names = {"s3": "S3", "hubert": "HuBERT", "whisper": "Whisper", "mfcc": "MFCC"}
    selected = _selected_semantic_encoder_names(config)
    weights = _semantic_encoder_weight_map(config)
    distances = []
    for name, clean_vec, protected_vec in zip(names, clean_vectors, protected_vectors):
        canonical_name = _canonical_semantic_encoder_name(name)
        if selected is not None and canonical_name not in selected:
            continue
        cosine = _framewise_cosine_similarity(clean_vec, protected_vec)
        drift = clamp(1.0 - cosine)
        weight = weights.get(str(canonical_name or name).lower(), 1.0)
        distances.append(
            {
                "encoder": display_names.get(str(canonical_name or name).lower(), str(name)),
                "cosineBeforeAfter": cosine,
                "distance": drift,
                "weight": weight,
                "status": "available",
                "source": "SemanticEncoderEnsemble",
            }
        )
    return distances, "SemanticEncoderEnsemble"


def compute_semantic_token_metrics(clean_path: Path, protected_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    config = config or {}
    selected_encoders = _selected_semantic_encoder_names(config)
    details = {
        "tokenChangeRate": None,
        "tokenErrorRate": None,
        "tokenChangeCount": None,
        "tokenTotal": None,
        "semanticDrift": None,
        "encoderDistances": [],
        "status": "unavailable",
        "_metricSources": {
            "asrEval.tokenChangeRate": metric_source("unavailable", "semantic_tokenizer", reason="No real tokenizer configured", formula="edit/token mismatch over encoded tokens"),
            "asrEval.tokenErrorRate": metric_source("unavailable", "semantic_tokenizer", reason="No real tokenizer configured", formula="edit_distance(tokens,tokens_p)/max(len(tokens),1)"),
            "asrEval.semanticDrift": metric_source(
                "unavailable",
                "semantic_encoder",
                reason="No real semantic encoder is configured",
                formula="sum_k(w_k*(1-mean_t CosSim(F_k(x)_t,F_k(xp)_t)))/sum_k(w_k) over selected semantic encoders",
            ),
        },
    }
    if os.getenv("SEME2E_ENABLE_TOKENIZER", "1") == "1":
        try:
            tokens = encode_s3_tokens(clean_path)
            tokens_p = encode_s3_tokens(protected_path)
            length = min(len(tokens), len(tokens_p))
            token_change_count = sum(a != b for a, b in zip(tokens[:length], tokens_p[:length]))
            token_change_rate = token_change_count / max(length, 1)
            token_error_rate = _edit_distance(tokens, tokens_p) / max(len(tokens), 1)
            details.update(
                {
                    "tokenChangeRate": token_change_rate,
                    "tokenErrorRate": token_error_rate,
                    "tokenChangeCount": token_change_count,
                    "tokenTotal": len(tokens),
                    "status": "partial",
                }
            )
            source = os.getenv("SEME2E_TOKENIZER_MODEL") or _default_s3_tokenizer_model()
            details["_metricSources"]["asrEval.tokenChangeRate"] = metric_source(
                "available",
                source,
                formula="sum(tokens[i]!=tokens_p[i] for i<L)/max(L,1)",
            )
            details["_metricSources"]["asrEval.tokenErrorRate"] = metric_source(
                "available",
                source,
                formula="edit_distance(tokens,tokens_p)/max(len(tokens),1)",
            )
        except Exception as exc:
            details["status"] = "error"
            details["error"] = str(exc)
            details["_metricSources"]["asrEval.tokenChangeRate"] = metric_source("error", "semantic_tokenizer", reason=str(exc), formula="edit/token mismatch over encoded tokens")
            details["_metricSources"]["asrEval.tokenErrorRate"] = metric_source("error", "semantic_tokenizer", reason=str(exc), formula="edit_distance(tokens,tokens_p)/max(len(tokens),1)")

    try:
        encoder_distances, encoder_source = _compute_semantic_encoder_distances(clean_path, protected_path, config)
        if encoder_distances:
            weighted_drift = 0.0
            weighted_cosine = 0.0
            weight_sum = 0.0
            for item in encoder_distances:
                drift = finite_float(item.get("distance"))
                cosine = finite_float(item.get("cosineBeforeAfter"))
                weight = finite_float(item.get("weight")) or 1.0
                if drift is None or cosine is None or weight <= 0:
                    continue
                weighted_drift += drift * weight
                weighted_cosine += cosine * weight
                weight_sum += weight
            details.update(
                {
                    "semanticDrift": weighted_drift / weight_sum if weight_sum else None,
                    "semanticCosineWeightedSum": weighted_cosine,
                    "semanticWeightedCosine": weighted_cosine / weight_sum if weight_sum else None,
                    "semanticWeightSum": weight_sum,
                    "encoderDistances": encoder_distances,
                    "status": "available" if details.get("tokenErrorRate") is not None else "partial",
                }
            )
            details["_metricSources"]["asrEval.semanticDrift"] = metric_source(
                "available",
                encoder_source or "SemanticEncoderEnsemble",
                reason=f"selectedSemanticEncoders={sorted(selected_encoders)}" if selected_encoders else None,
                formula="sum_k(w_k*(1-mean_t CosSim(F_k(x)_t,F_k(xp)_t)))/sum_k(w_k) over selected semantic encoders",
            )
            return details
    except Exception as exc:
        details["status"] = "error"
        details["error"] = str(exc)
        details["_metricSources"]["asrEval.semanticDrift"] = metric_source(
            "error",
            "SemanticEncoderEnsemble",
            reason=str(exc),
            formula="sum_k(w_k*(1-mean_t CosSim(F_k(x)_t,F_k(xp)_t)))/sum_k(w_k) over selected semantic encoders",
        )
        return details

    details["reason"] = _SEMANTIC_ENCODER_LAST_ERROR or f"SemanticEncoderEnsemble unavailable for selected semantic encoders: {sorted(selected_encoders) if selected_encoders else 'all configured semantic encoders'}"
    details["_metricSources"]["asrEval.semanticDrift"] = metric_source(
        "error" if _SEMANTIC_ENCODER_LAST_ERROR and not _SEMANTIC_ENCODER_LAST_ERROR.startswith("Set SEME2E_ENABLE_SEMANTIC_ENCODERS") else "unavailable",
        "SemanticEncoderEnsemble",
        reason=details["reason"],
        formula="sum_k(w_k*(1-mean_t CosSim(F_k(x)_t,F_k(xp)_t)))/sum_k(w_k) over selected semantic encoders",
    )
    return details


def _build_speaker_scorer() -> tuple[Any | None, str, dict[str, Any]]:
    metric = os.getenv("SEME2E_SPEAKER_METRIC", "ecapa")
    model = os.getenv("SEME2E_SPEAKER_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
    source = model
    metric_label = "ECAPA-TDNN speaker embedding cosine similarity" if metric.lower() == "ecapa" else f"{metric} speaker embedding cosine similarity"
    if os.getenv("SEME2E_ENABLE_SPEAKER", "1") != "1":
        return None, source, metric_source("unavailable", source, reason="Set SEME2E_ENABLE_SPEAKER=1 to run speaker similarity dependencies", formula="cosine(Emb(a),Emb(b))", metric=metric_label)
    try:
        from speaker_similarity import build_speaker_similarity

        scorer = build_speaker_similarity(metric, model, os.getenv("SEME2E_API_DEVICE", "cpu"))
        return scorer, source, metric_source("available", source, formula="cosine(Emb(a),Emb(b))", metric=metric_label)
    except Exception as exc:
        return None, source, metric_source("error", source, reason=str(exc), formula="cosine(Emb(a),Emb(b))", metric=metric_label)


def compute_direct_speaker_metrics(clean_path: Path, protected_path: Path, speaker_model: Any | None = None) -> dict[str, Any]:
    scorer = speaker_model
    source = "speaker_similarity"
    source_info = metric_source("available", source, formula="cosine(Emb(x),Emb(xp))")
    if scorer is None:
        scorer, source, source_info = _build_speaker_scorer()
    base = {
        "simBefore": None,
        "simAfter": None,
        "simDropRate": None,
        "embeddingDistanceBefore": None,
        "embeddingDistanceAfter": None,
        "simOriginalProtected": None,
        "embeddingDistance": None,
        "directDistance": None,
        "directIdentityScore": None,
        "scoreStatus": "unavailable",
        "scoreReason": source_info.get("reason") or "直接声音身份指标尚未生成",
        "metric": os.getenv("SEME2E_SPEAKER_METRIC", "ecapa"),
        "source": source,
        "status": source_info["status"],
        "_metricSources": {
            "speaker.*": source_info,
        },
    }
    if scorer is None:
        base["reason"] = source_info.get("reason")
        return base
    try:
        direct_similarity = finite_float(scorer.score(clean_path, protected_path))
        if direct_similarity is None:
            raise ValueError("speaker scorer returned a non-finite similarity")
        score_payload = compute_direct_identity_score(direct_similarity)
        base.update(
            {
                "simBefore": 1.0,
                "simAfter": direct_similarity,
                "simDropRate": 1.0 - direct_similarity,
                "embeddingDistanceBefore": 0.0,
                "embeddingDistanceAfter": 1.0 - direct_similarity,
                "simOriginalProtected": direct_similarity,
                "embeddingDistance": 1.0 - direct_similarity,
                **score_payload,
                "status": "available",
            }
        )
        base["_metricSources"]["speaker.*"] = metric_source("available", source, formula="directSimilarity=cosine(Emb(x),Emb(xp)); directDistance=1-directSimilarity", metric=source_info.get("metric"))
        base["_metricSources"]["speaker.directIdentityScore"] = metric_source(
            score_payload["scoreStatus"],
            "VoiceShield_v2.1_phi_calibration",
            reason=score_payload.get("scoreReason"),
            formula="100*(1-10^(-directDistance/directDistance90))",
        )
    except Exception as exc:
        base["status"] = "error"
        base["error"] = str(exc)
        base["_metricSources"]["speaker.*"] = metric_source("error", source, reason=str(exc), formula="directSimilarity=cosine(Emb(x),Emb(xp))", metric=source_info.get("metric"))
    return base


def clone_radar_point(name: str, value: float | None, reason: str | None = None, formula: str | None = None, raw_metric_keys: list[str] | None = None) -> dict[str, Any]:
    point: dict[str, Any] = {
        "name": name,
        "value": value,
        "status": "available" if value is not None else "unavailable",
        "reason": reason,
        "formula": formula,
        "rawMetricKeys": raw_metric_keys or [],
    }
    return point


def build_clone_radar(
    direct_similarity: float | None,
    similarity_drop_rate: float | None,
    embedding_distance_increase_rate: float | None,
    protected_similarity: float | None,
    *,
    direct_reason: str | None = None,
    clone_reason: str | None = None,
) -> list[dict[str, Any]]:
    direct_offset_score = 100.0 * clamp(1.0 - direct_similarity, 0.0, 1.0) if direct_similarity is not None else None
    similarity_drop_score = 100.0 * clamp(similarity_drop_rate / 0.5, 0.0, 1.0) if similarity_drop_rate is not None else None
    distance_increase_score = 100.0 * clamp(embedding_distance_increase_rate / 1.0, 0.0, 1.0) if embedding_distance_increase_rate is not None else None
    protected_clone_defense_score = 100.0 * clamp(1.0 - protected_similarity, 0.0, 1.0) if protected_similarity is not None else None
    return [
        clone_radar_point(
            "直接声纹偏移",
            direct_offset_score,
            direct_reason or "speaker similarity is not available",
            "100*clip(1-directSimilarity,0,1)",
            ["directSimilarity"],
        ),
        clone_radar_point(
            "相似度下降",
            similarity_drop_score,
            clone_reason or "clone similarity metrics are not available",
            "100*clip(similarityDropRate/0.5,0,1)",
            ["originalSimilarity", "protectedSimilarity", "similarityDropRate"],
        ),
        clone_radar_point(
            "嵌入距离增加",
            distance_increase_score,
            clone_reason or "clone embedding distance metrics are not available",
            "100*clip(embeddingDistanceIncreaseRate/1.0,0,1)",
            ["embeddingDistanceBefore", "embeddingDistanceAfter", "embeddingDistanceIncreaseRate"],
        ),
        clone_radar_point(
            "保护后克隆防护",
            protected_clone_defense_score,
            clone_reason or "protected clone similarity is not available",
            "100*clip(1-protectedSimilarity,0,1)",
            ["protectedSimilarity"],
        ),
    ]

def compute_clone_eval(
    original_audio_path: Path,
    original_clone_path: Path,
    protected_clone_path: Path,
    clone_result: dict[str, Any],
    protected_audio_path: Path | None = None,
    speaker_model: Any | None = None,
    confidence_calibrator: Any | None = None,
    clone_transcription: dict[str, Any] | None = None,
    semantic_metrics: dict[str, Any] | None = None,
    quality_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clone_transcription = clone_transcription or {}
    semantic_metrics = semantic_metrics or {}
    quality_metrics = quality_metrics or {}
    scorer = speaker_model
    source = "speaker_similarity"
    source_info = metric_source("available", source, formula="SIM(a,b)=cosine(Emb(a),Emb(b))")
    if scorer is None:
        scorer, source, source_info = _build_speaker_scorer()
    request = clone_result.get("request") or {}
    unavailable_reason = source_info.get("reason") or "克隆声音身份结果尚未生成"
    eval_payload = {
        "cloneModel": request.get("model"),
        "speakerEvalModel": source,
        "targetText": request.get("text"),
        "originalCloneAudio": clone_result.get("originalCloneAudio"),
        "protectedCloneAudio": clone_result.get("protectedCloneAudio"),
        "directSimilarity": None,
        "originalSimilarity": None,
        "protectedSimilarity": None,
        "similarityDropRate": None,
        "embeddingDistanceBefore": None,
        "embeddingDistanceAfter": None,
        "embeddingDistanceDelta": None,
        "embeddingDistanceIncreaseRate": None,
        "cloneIdentityScore": None,
        "identityBaselineWeight": None,
        "cloneIdentityStatus": "unavailable",
        "cloneIdentityReason": unavailable_reason,
        "cleanCloneTranscription": clone_transcription.get("originalText"),
        "protectedCloneTranscription": clone_transcription.get("protectedText"),
        "cloneAsrModel": clone_transcription.get("model"),
        "cloneAsrStatus": clone_transcription.get("status") or "unavailable",
        "cloneAsrReason": clone_transcription.get("reason"),
        "cleanCloneTextAccuracy": None,
        "cleanCloneTextError": None,
        "protectedCloneTextAccuracy": None,
        "protectedCloneTextError": None,
        "cloneTextChangeAccuracy": None,
        "cloneTextChangeRate": None,
        "semanticBaselineWeight": None,
        "cloneTokenChangeRate": finite_float(semantic_metrics.get("tokenChangeRate")),
        "cloneSemanticDrift": finite_float(semantic_metrics.get("semanticDrift")),
        "cloneTokenScore": None,
        "cloneDriftScore": None,
        "cloneSemanticScore": None,
        "cloneSemanticStatus": "unavailable",
        "cloneSemanticReason": semantic_metrics.get("reason") or clone_transcription.get("reason") or "克隆语义指标尚未生成",
        "cleanCloneQualityMos": finite_float(quality_metrics.get("cleanMos")),
        "protectedCloneQualityMos": finite_float(quality_metrics.get("protectedMos")),
        "clonePairPesq": finite_float(quality_metrics.get("clonePairPesq")),
        "clonePairStoi": finite_float(quality_metrics.get("clonePairStoi")),
        "cloneQualityBefore": None,
        "cloneQualityAfter": None,
        "cloneQualityDropRate": None,
        "clonePesqDegradationScore": None,
        "cloneStoiDegradationScore": None,
        "cloneDnsMosDegradationScore": None,
        "cloneQualityComponents": None,
        "cloneQualityRawScore": None,
        "cloneQualityRelevance": None,
        "cloneQualityScore": None,
        "qualityBaselineWeight": None,
        "cloneQualityModel": quality_metrics.get("model") or "PESQ + STOI + DNSMOS P.835 OVRL",
        "cloneQualityModelPath": quality_metrics.get("modelPath"),
        "cloneQualityStatus": quality_metrics.get("status") or "unavailable",
        "cloneQualityReason": quality_metrics.get("reason") or "克隆语音质量结果尚未生成",
        "cloneConfidenceBefore": None,
        "cloneConfidenceAfter": None,
        "cloneConfidenceDropRate": None,
        "cloneRadar": build_clone_radar(
            None,
            None,
            None,
            None,
            direct_reason=unavailable_reason if protected_audio_path is not None else "protected audio path is not available",
            clone_reason=unavailable_reason,
        ),
        "cloneTrend": None,
        "cloneDefenseScore": None,
        "createdAt": now_iso(),
        "status": source_info["status"],
        "_metricSources": {
            "cloneEval.*": source_info,
            "cloneEval.cloneAsr": metric_source(
                clone_transcription.get("status") or "unavailable",
                clone_transcription.get("model") or "isolated_asr_worker",
                reason=clone_transcription.get("reason"),
                formula="transcribe(cleanCloneAudio), transcribe(protectedCloneAudio)",
            ),
            "cloneEval.cloneSemanticScore": metric_source(
                "unavailable",
                "semantic_tokenizer + semantic_encoder",
                reason=semantic_metrics.get("reason") or "克隆语义指标或评分标定尚未完成",
                formula=".55*Phi(cloneTokenChange)+.45*Phi(cloneSemanticDrift)",
            ),
            "cloneEval.cloneQualityScore": metric_source(
                quality_metrics.get("status") or "unavailable",
                quality_metrics.get("model") or "PESQ + STOI + DNSMOS P.835 OVRL",
                reason=quality_metrics.get("reason"),
                formula="Q0=weighted(100,100,N(DNS0)); Q1=weighted(N(PESQ(clean,protected)),N(STOI(clean,protected)),N(DNS1)); Sq_raw=Phi(max(0,(Q0-Q1)/Q0);0.75)",
            ),
            "cloneEval.cloneConfidenceBefore": metric_source("unavailable", "confidence_calibrator", reason="No confidence calibrator is configured", formula="sigmoid(A*similarity+B)"),
            "cloneEval.cloneConfidenceAfter": metric_source("unavailable", "confidence_calibrator", reason="No confidence calibrator is configured", formula="sigmoid(A*similarity+B)"),
            "cloneEval.cloneConfidenceDropRate": metric_source("unavailable", "confidence_calibrator", reason="No confidence calibrator is configured", formula="(cloneConfidenceBefore-cloneConfidenceAfter)/max(cloneConfidenceBefore,EPS)"),
            "cloneEval.cloneTrend": metric_source("not_run", "multi_checkpoint_clone_eval", reason="clone trend is disabled; only final clone evaluation is reported", formula="None"),
        },
    }
    if scorer is not None:
        try:
            original_similarity = finite_float(scorer.score(original_audio_path, original_clone_path))
            protected_similarity = finite_float(scorer.score(original_audio_path, protected_clone_path))
            direct_similarity = None
            direct_reason = "protected audio path is not available"
            if protected_audio_path is not None:
                direct_similarity = finite_float(scorer.score(original_audio_path, protected_audio_path))
                direct_reason = None if direct_similarity is not None else "speaker similarity is not available"
            if original_similarity is None or protected_similarity is None:
                raise ValueError("speaker scorer returned non-finite clone similarity")
            similarity_drop_rate = (original_similarity - protected_similarity) / max(original_similarity, EPS)
            identity = compute_clone_identity_score(original_similarity, protected_similarity)
            embedding_before = identity["embeddingDistanceBefore"]
            embedding_after = identity["embeddingDistanceAfter"]
            embedding_increase = (
                (embedding_after - embedding_before) / max(embedding_before, EPS)
                if embedding_before is not None and embedding_after is not None
                else None
            )
            conf_before = conf_after = conf_drop = None
            if confidence_calibrator is not None:
                conf_before = finite_float(confidence_calibrator(original_similarity))
                conf_after = finite_float(confidence_calibrator(protected_similarity))
                if conf_before is not None and conf_after is not None:
                    conf_drop = (conf_before - conf_after) / max(conf_before, EPS)
            eval_payload.update(
                {
                    "directSimilarity": direct_similarity,
                    "originalSimilarity": original_similarity,
                    "protectedSimilarity": protected_similarity,
                    "similarityDropRate": similarity_drop_rate,
                    **identity,
                    "embeddingDistanceIncreaseRate": embedding_increase,
                    "cloneConfidenceBefore": conf_before,
                    "cloneConfidenceAfter": conf_after,
                    "cloneConfidenceDropRate": conf_drop,
                    "cloneRadar": build_clone_radar(
                        direct_similarity,
                        similarity_drop_rate,
                        embedding_increase,
                        protected_similarity,
                        direct_reason=direct_reason,
                    ),
                    # Legacy alias retained for old clients; the main score now uses
                    # only the v2.1 distance-baseline identity formula.
                    "cloneDefenseScore": identity.get("cloneIdentityScore"),
                }
            )
            eval_payload["_metricSources"]["cloneEval.*"] = metric_source(
                "available",
                source,
                formula="d0=1-SIM(original,cleanClone); d1=1-SIM(original,protectedClone)",
                metric=source_info.get("metric"),
            )
            eval_payload["_metricSources"]["cloneEval.cloneIdentityScore"] = metric_source(
                identity["cloneIdentityStatus"],
                "VoiceShield_v2.1_clone_identity",
                reason=identity.get("cloneIdentityReason"),
                formula="95*clip((d1-d0)/(.75-d0),0,1)+5*clip((d1-.75)/.25,0,1)",
                metric=source_info.get("metric"),
            )
            eval_payload["_metricSources"]["cloneEval.cloneDefenseScore"] = eval_payload["_metricSources"]["cloneEval.cloneIdentityScore"]
            if conf_drop is not None:
                eval_payload["_metricSources"]["cloneEval.cloneConfidenceBefore"] = metric_source("available", "confidence_calibrator", formula="sigmoid(A*similarity+B)")
                eval_payload["_metricSources"]["cloneEval.cloneConfidenceAfter"] = metric_source("available", "confidence_calibrator", formula="sigmoid(A*similarity+B)")
                eval_payload["_metricSources"]["cloneEval.cloneConfidenceDropRate"] = metric_source("available", "confidence_calibrator", formula="(cloneConfidenceBefore-cloneConfidenceAfter)/max(cloneConfidenceBefore,EPS)")
        except Exception as exc:
            eval_payload["cloneIdentityStatus"] = "error"
            eval_payload["cloneIdentityReason"] = str(exc)
            eval_payload["_metricSources"]["cloneEval.*"] = metric_source("error", source, reason=str(exc), formula="SIM(originalAudio,cloneAudio)", metric=source_info.get("metric"))
    else:
        eval_payload["cloneIdentityReason"] = source_info.get("reason") or unavailable_reason

    semantic = compute_clone_semantic_score(
        eval_payload.get("targetText"),
        eval_payload.get("cleanCloneTranscription"),
        eval_payload.get("protectedCloneTranscription"),
        semantic_metrics.get("tokenChangeRate"),
        semantic_metrics.get("semanticDrift"),
    )
    eval_payload.update(semantic)
    eval_payload["_metricSources"]["cloneEval.cloneSemanticScore"] = metric_source(
        semantic["cloneSemanticStatus"],
        "semantic_tokenizer + semantic_encoder + bounded_text_baseline",
        reason=semantic.get("cloneSemanticReason"),
        formula="w_sem=A_clean^2*(3-2*A_clean); score=.55*Phi(tokenChange)+.45*Phi(semanticDrift)",
    )

    quality = compute_clone_quality_score(
        quality_metrics.get("cleanMos"),
        quality_metrics.get("protectedMos"),
        pair_pesq=quality_metrics.get("clonePairPesq"),
        pair_stoi=quality_metrics.get("clonePairStoi"),
        identity_baseline_weight=eval_payload.get("identityBaselineWeight"),
        clone_identity_score=eval_payload.get("cloneIdentityScore"),
        clone_semantic_score=eval_payload.get("cloneSemanticScore"),
    )
    eval_payload.update(quality)
    if quality_metrics.get("status") not in {"available", "computed"}:
        eval_payload["cloneQualityStatus"] = quality_metrics.get("status") or "unavailable"
        eval_payload["cloneQualityReason"] = quality_metrics.get("reason") or quality.get("cloneQualityReason")
    quality_source = metric_source(
        eval_payload["cloneQualityStatus"],
        quality_metrics.get("model") or "PESQ + STOI + DNSMOS P.835 OVRL",
        reason=eval_payload.get("cloneQualityReason"),
        formula="Q0=weighted(100,100,N(DNS0)); Q1=weighted(N(PESQ(clean,protected)),N(STOI(clean,protected)),N(DNS1)); Sq_raw=Phi(max(0,(Q0-Q1)/Q0);0.75)",
    )
    for metric_key in (
        "cloneQualityBefore",
        "cloneQualityAfter",
        "clonePesqDegradationScore",
        "cloneStoiDegradationScore",
        "cloneDnsMosDegradationScore",
        "cloneQualityRawScore",
        "cloneQualityRelevance",
        "cloneQualityScore",
    ):
        eval_payload["_metricSources"][f"cloneEval.{metric_key}"] = quality_source
    pair_metric_sources = quality_metrics.get("pairMetricSources")
    if isinstance(pair_metric_sources, dict):
        eval_payload["_metricSources"].update(pair_metric_sources)

    available_dimensions = sum(
        eval_payload.get(key) == "available"
        for key in ["cloneIdentityStatus", "cloneSemanticStatus", "cloneQualityStatus"]
    )
    eval_payload["status"] = "available" if available_dimensions == 3 else "partial" if available_dimensions else "unavailable"
    reasons = [
        eval_payload.get(key)
        for key in ["cloneIdentityReason", "cloneSemanticReason", "cloneQualityReason"]
        if eval_payload.get(key)
    ]
    if reasons:
        eval_payload["reason"] = "; ".join(dict.fromkeys(str(item) for item in reasons))
    return eval_payload


def compute_overall_score(result: dict[str, Any]) -> dict[str, Any]:
    details = result.get("details") or {}
    perception = details.get("perception") or {}
    semantic_details = details.get("semantic") or {}
    speaker_details = details.get("speaker") or {}
    raw_quality = perception.get("protectionQuality") or {}
    protection_quality = compute_protection_quality_score(
        raw_quality.get("snr", perception.get("snr")),
        raw_quality.get("stoi", perception.get("stoi")),
        raw_quality.get("pesq", perception.get("pesq")),
        raw_quality.get("dnsMos", perception.get("dnsMos")),
    )
    protection_semantic = compute_protection_semantic_score(
        semantic_details.get("tokenChangeRate"),
        semantic_details.get("semanticDrift"),
    )
    primary_metrics = (result.get("summary") or {}).get("primaryMetrics") or {}
    direct_similarity = finite_float(
        speaker_details.get("simOriginalProtected", speaker_details.get("simAfter"))
    )
    if direct_similarity is None:
        direct_distance = finite_float(
            speaker_details.get("embeddingDistanceAfter", speaker_details.get("embeddingDistance"))
        )
        if direct_distance is not None:
            direct_similarity = 1.0 - direct_distance
    if direct_similarity is None:
        direct_similarity = finite_float(primary_metrics.get("speakerSimilarity"))
    direct_identity = compute_direct_identity_score(direct_similarity)

    latest_by_model: dict[str, dict[str, Any]] = {}
    for index, clone_result in enumerate(result.get("cloneResults") or []):
        if not isinstance(clone_result, dict):
            continue
        clone_eval = clone_result.get("cloneEval")
        if not isinstance(clone_eval, dict):
            continue
        for key in (
            "cloneIdentityScore",
            "identityBaselineWeight",
            "cloneSemanticScore",
            "semanticBaselineWeight",
            "cloneQualityScore",
            "qualityBaselineWeight",
        ):
            if clone_eval.get(key) is None and clone_result.get(key) is not None:
                clone_eval[key] = clone_result.get(key)
        if clone_eval.get("cloneIdentityScore") is None:
            clone_eval.update(
                compute_clone_identity_score(
                    clone_eval.get("originalSimilarity", clone_result.get("originalSimilarity")),
                    clone_eval.get("protectedSimilarity", clone_result.get("protectedSimilarity")),
                )
            )
        request = clone_result.get("request") or {}
        model = str(clone_eval.get("cloneModel") or request.get("model") or f"clone-{index}")
        latest_by_model[model] = clone_eval
    clone_evals = list(latest_by_model.values())
    clone_identity, clone_identity_reason = aggregate_weighted_scores(
        clone_evals,
        "cloneIdentityScore",
        "identityBaselineWeight",
        "缺少有效的原始克隆身份结果",
    )
    clone_semantic, clone_semantic_reason = aggregate_weighted_scores(
        clone_evals,
        "cloneSemanticScore",
        "semanticBaselineWeight",
        "缺少有效的原始克隆文本结果",
    )
    clone_quality, clone_quality_reason = aggregate_weighted_scores(
        clone_evals,
        "cloneQualityScore",
        "qualityBaselineWeight",
        "缺少有效的原始克隆语音质量结果",
    )
    clone_quality_status = "available"
    clone_quality_partial_reasons: list[str] = []
    for clone_eval in clone_evals:
        quality_score = finite_float(clone_eval.get("cloneQualityScore"))
        quality_weight = finite_float(clone_eval.get("qualityBaselineWeight"))
        quality_status = str(clone_eval.get("cloneQualityStatus") or "").strip().lower()
        if quality_score is None or quality_weight is None or quality_weight <= EPS:
            clone_quality_status = "partial" if clone_quality is not None else "unavailable"
            reason = clone_eval.get("cloneQualityReason")
            if isinstance(reason, str) and reason.strip():
                clone_quality_partial_reasons.append(reason.strip())
            continue
        if quality_status and quality_status != "available":
            clone_quality_status = "partial"
            reason = clone_eval.get("cloneQualityReason")
            if isinstance(reason, str) and reason.strip():
                clone_quality_partial_reasons.append(reason.strip())
    if clone_quality is None:
        clone_quality_status = "unavailable"
    elif clone_quality_status == "partial":
        clone_quality_reason = "；".join(dict.fromkeys(clone_quality_partial_reasons)) or "部分克隆语音质量指标尚未生成"
    if not clone_evals:
        clone_identity_reason = clone_semantic_reason = clone_quality_reason = "待完成克隆测试"
        clone_quality_status = "pending"

    dimension_specs = [
        ("protectionQuality", "保护音频听感质量", protection_quality.get("qualityScore"), protection_quality.get("scoreReason"), 0.20, None),
        ("cloneQuality", "克隆音频质量下降", clone_quality, clone_quality_reason, 0.10, clone_quality_status),
        ("protectionSemantic", "保护后音频语义干扰", protection_semantic.get("protectionSemanticScore"), protection_semantic.get("scoreReason"), 0.20, None),
        ("cloneSemantic", "克隆后音频语义干扰", clone_semantic, clone_semantic_reason, 0.15, None),
        ("directIdentity", "保护后声音身份直接保护效果", direct_identity.get("directIdentityScore"), direct_identity.get("scoreReason"), 0.15, None),
        ("cloneIdentity", "克隆声音身份保护效果", clone_identity, clone_identity_reason, 0.20, None),
    ]
    dimensions: list[dict[str, Any]] = []
    missing_dimensions: list[str] = []
    for key, label, value, reason, weight, explicit_status in dimension_specs:
        score_value = finite_float(value)
        status = (
            explicit_status
            if score_value is not None and explicit_status is not None
            else "available"
            if score_value is not None
            else "pending"
            if reason == "待完成克隆测试"
            else "unavailable"
        )
        if score_value is None or status != "available":
            missing_dimensions.append(key)
        dimensions.append(
            {
                "key": key,
                "label": label,
                "score": score_value,
                "status": status,
                "reason": None if status == "available" else reason,
                "weight": weight,
            }
        )

    score = None
    if not missing_dimensions:
        score = math.exp(
            sum(item["weight"] * math.log(max(float(item["score"]), 1.0)) for item in dimensions)
        )
    if score is None:
        level = None
        verdict = "待完整评估"
    elif score >= 85:
        level = "优秀"
        verdict = "综合防护效果优秀"
    elif score >= 70:
        level = "中等"
        verdict = "综合防护效果中等"
    else:
        level = "较差"
        verdict = "综合防护效果较差"

    recommendations: list[dict[str, Any]] = []
    if finite_float(direct_identity.get("directIdentityScore")) is not None and float(direct_identity["directIdentityScore"]) < 70:
        recommendations.append({"key": "identity", "message": "直接身份保护偏弱，可适当提高身份保护权重。", "parameters": ["lambdaId"]})
    if finite_float(protection_semantic.get("protectionSemanticScore")) is not None and float(protection_semantic["protectionSemanticScore"]) < 70:
        recommendations.append({"key": "semantic", "message": "语义保护偏弱，可适当提高语义保护权重。", "parameters": ["lambdaSem"]})
    if finite_float(protection_quality.get("qualityScore")) is not None and float(protection_quality["qualityScore"]) < 70:
        recommendations.append({"key": "quality", "message": "保护音频听感质量偏低，可提高心理声学或 L2 权重，或降低扰动上限。", "parameters": ["lambdaPsy", "lambda2", "epsilon"]})
    generation = details.get("generation") or {}
    selected_step = finite_float(generation.get("selectedStep"))
    max_steps = finite_float(generation.get("maxSteps", generation.get("steps")))
    if selected_step is not None and max_steps is not None and selected_step >= max_steps:
        recommendations.append({"key": "convergence", "message": "最优结果出现在迭代末端，可适当增加迭代次数。", "parameters": ["steps"]})

    evaluation = {
        "status": "complete" if not missing_dimensions else "incomplete",
        "overallScore": score,
        "level": level,
        "verdict": verdict,
        "dimensions": dimensions,
        "missingDimensions": missing_dimensions,
        "recommendations": recommendations,
        "calibration": dict(SCORE_CALIBRATION),
        "calibrationSources": dict(SCORE_CALIBRATION_SOURCES),
        "cloneAggregation": {
            "modelCount": len(clone_evals),
            "models": list(latest_by_model),
            "identityReason": clone_identity_reason,
            "semanticReason": clone_semantic_reason,
            "qualityReason": clone_quality_reason,
        },
    }
    source = metric_source(
        "available" if score is not None else "unavailable",
        "VoiceShield_v2.1_weighted_geometric_mean",
        reason=None if score is not None else f"综合评分仍缺少以下项目：{', '.join(missing_dimensions)}",
        formula="exp(sum(alpha_i*ln(max(S_i,1)))) with weights [.20,.10,.20,.15,.15,.20]",
    )
    return {
        "score": score,
        "verdict": verdict,
        "protectionEvaluation": evaluation,
        "protectionQuality": protection_quality,
        "protectionSemantic": protection_semantic,
        "directIdentity": direct_identity,
        "_metricSources": {"score": source, "verdict": source, "protectionEvaluation.overallScore": source},
    }
