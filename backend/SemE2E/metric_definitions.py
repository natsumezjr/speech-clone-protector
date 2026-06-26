from __future__ import annotations

import importlib.util
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


EPS = 1.0e-12


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


def metric_source(status: str, source: str, reason: str | None = None, formula: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": status, "source": source}
    if reason:
        payload["reason"] = reason
    if formula:
        payload["formula"] = formula
    return payload


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


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
    if epsilon_value is not None:
        if epsilon_norm_value == "linf" and linf_norm is not None:
            epsilon_usage_rate = linf_norm / max(epsilon_value, EPS)
        elif epsilon_norm_value == "l2":
            epsilon_usage_rate = l2_norm / max(epsilon_value, EPS)
    return {
        "l2Norm": l2_norm,
        "l2Rms": l2_rms,
        "linfNorm": linf_norm,
        "epsilon": epsilon_value,
        "epsilonNorm": epsilon_norm_value,
        "epsilonUsageRate": epsilon_usage_rate,
        "snr": snr,
        "clippingRate": clipping_rate,
    }


def compute_quality_metrics(
    x: np.ndarray,
    xp: np.ndarray,
    delta: np.ndarray,
    sr: int,
    perturbation_metrics: dict[str, Any],
) -> dict[str, Any]:
    del delta
    sources: dict[str, dict[str, Any]] = {
        "protectionQuality.snr": metric_source("available", "compute_perturbation_metrics", formula="10*log10((P_signal+1e-12)/(P_noise+1e-12))"),
        "protectionQuality.mos": metric_source("unavailable", "human_listening_test", reason="MOS requires human listening test or a declared MOS model", formula="None without an explicit MOS model"),
        "protectionQuality.mosLqo": metric_source("unavailable", "objective_mos_lqo_model", reason="No explicit MOS-LQO objective model is configured", formula="None without an explicit MOS-LQO model"),
    }
    pesq_value = None
    if sr not in {8000, 16000}:
        sources["protectionQuality.pesq"] = metric_source("unavailable", "pesq", reason=f"PESQ supports 8000 or 16000 Hz, got {sr}", formula="pesq(sr, x, xp, mode)")
    elif not _module_available("pesq"):
        sources["protectionQuality.pesq"] = metric_source("unavailable", "pesq", reason="Python package 'pesq' is not installed", formula="pesq(sr, x, xp, mode)")
    else:
        try:
            from pesq import pesq

            pesq_value = float(pesq(sr, x, xp, "wb" if sr == 16000 else "nb"))
            sources["protectionQuality.pesq"] = metric_source("available", "pesq", formula="pesq(sr, x, xp, mode)")
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
    clipping_rate = finite_float(perturbation_metrics.get("clippingRate"))
    q_snr = clamp(((snr or 0.0) - 10.0) / 20.0) if snr is not None else None
    q_pesq = clamp((pesq_value - 1.0) / 3.5) if pesq_value is not None else None
    q_stoi = clamp((stoi_value - 0.5) / 0.5) if stoi_value is not None else None
    q_clip = 1.0 - clamp((clipping_rate or 0.0) / 0.01) if clipping_rate is not None else None
    quality_score_base = weighted_available_mean(
        {
            "q_snr": (q_snr, 0.35),
            "q_pesq": (q_pesq, 0.30),
            "q_stoi": (q_stoi, 0.25),
            "q_clip": (q_clip, 0.10),
        }
    )
    quality_score = 100.0 * quality_score_base if quality_score_base is not None else None
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
        "weighted_available_mean",
        reason=None if quality_score is not None else "No quality submetrics are available",
        formula="100*weighted_available_mean(q_snr:.35,q_pesq:.30,q_stoi:.25,q_clip:.10)",
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
        "qualityScore": quality_score,
        "qualityLevel": quality_level,
        "_metricSources": sources,
    }


def _stft(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.signal import stft

        nperseg = min(1024, max(64, len(audio)))
        noverlap = nperseg // 2
        freqs, _, zxx = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary=None)
        return freqs.astype(np.float64), zxx
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
        return freqs.astype(np.float64), np.asarray(frames, dtype=np.complex64).T


def _absolute_threshold_hearing(freqs: np.ndarray) -> np.ndarray:
    khz = np.maximum(freqs / 1000.0, 0.02)
    ath = 3.64 * np.power(khz, -0.8) - 6.5 * np.exp(-0.6 * np.power(khz - 3.3, 2.0)) + 0.001 * np.power(khz, 4.0)
    return np.clip(ath - 80.0, -120.0, 40.0)


def compute_psychoacoustic_metrics(x: np.ndarray, xp: np.ndarray, delta: np.ndarray, sr: int) -> dict[str, Any]:
    del xp
    freqs, x_stft = _stft(x, sr)
    _, d_stft = _stft(delta, sr)
    min_time = min(x_stft.shape[1], d_stft.shape[1])
    x_power = np.abs(x_stft[:, :min_time]) ** 2
    d_power = np.abs(d_stft[:, :min_time]) ** 2
    psd_delta = 10.0 * np.log10(d_power + EPS)
    psd_signal = 10.0 * np.log10(x_power + EPS)
    ath = _absolute_threshold_hearing(freqs)[:, None]
    theta = np.maximum(ath, psd_signal - 18.0)
    violation = np.maximum(0.0, psd_delta - theta)
    l_psy = float(np.mean(violation)) if violation.size else None
    over_mask_rate = float(np.mean(violation > 0.0)) if violation.size else None
    threshold = np.mean(theta, axis=1) if theta.size else np.asarray([], dtype=np.float64)
    perturbation = 10.0 * np.log10(np.mean(d_power, axis=1) + EPS) if d_power.size else np.asarray([], dtype=np.float64)
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
    sources = {
        "psychoacoustic.*": metric_source(
            "available",
            "engineering_stft_masking_threshold",
            reason="Engineering approximation from signal STFT PSD and absolute-threshold curve; not a calibrated psychoacoustic model",
            formula="V=max(0,PSD_delta-Theta); lPsy=mean(V); overMaskRate=mean(V>0)",
        )
    }
    return {
        "lPsy": l_psy,
        "overMaskRate": over_mask_rate,
        "maskingThreshold": masking_threshold,
        "perturbationSpectrum": perturbation_spectrum,
        "chart": chart,
        "_metricSources": sources,
    }


def _read_weight(config: dict[str, Any], group: str, primary: str, alias: str) -> float | None:
    section = config.get(group) or {}
    return finite_float(section.get(primary, section.get(alias)))


def _normalize_loss_point(item: dict[str, Any], index: int, weights: dict[str, float | None]) -> dict[str, Any] | None:
    point = {
        "step": finite_float(item.get("step", item.get("epoch", item.get("iteration", item.get("iter", index))))),
        "Lfeat": finite_float(item.get("Lfeat", item.get("Lfea", item.get("lossFeature", item.get("loss_timbre", item.get("L_feature")))))),
        "Lsem": finite_float(item.get("Lsem", item.get("lossSemantic", item.get("loss_semantic", item.get("L_semantic"))))),
        "Lpsy": finite_float(item.get("Lpsy", item.get("lossPsy", item.get("loss_psy", item.get("L_psy"))))),
        "L2": finite_float(item.get("L2", item.get("lossL2", item.get("loss_l2", item.get("l2Norm"))))),
        "total": finite_float(item.get("total", item.get("lossTotal", item.get("loss_total", item.get("objective"))))),
        "stepElapsedSec": finite_float(item.get("stepElapsedSec", item.get("elapsedSec", item.get("step_time")))),
    }
    if point["total"] is None and all(point[key] is not None for key in ["Lfeat", "Lsem", "Lpsy", "L2"]):
        if all(weights.get(key) is not None for key in ["lambdaFeat", "lambdaSem", "lambdaPsy", "lambda2"]):
            point["total"] = (
                float(weights["lambdaFeat"]) * float(point["Lfeat"])
                + float(weights["lambdaSem"]) * float(point["Lsem"])
                + float(weights["lambdaPsy"]) * float(point["Lpsy"])
                + float(weights["lambda2"]) * float(point["L2"])
            )
    if any(point[key] is not None for key in ["Lfeat", "Lsem", "Lpsy", "L2", "total"]):
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
    weights = {
        "lambdaFeat": _read_weight(request_config, "timbre", "lambdaTimbre", "weightFeature"),
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
    average_step_sec = float(np.mean(step_times)) if step_times else None
    loss_final = trace[-1] if trace else None
    source_status = "available" if trace else "unavailable"
    sources = {
        "lossFinal.*": metric_source(
            source_status,
            protection_details.get("source") or "SemanticE2EVGuard.protect",
            reason=None if trace else "Protection backend did not return an optimization trace",
            formula="lossFinal=optimizationTrace[-1]",
        ),
        "optimizationTrace": metric_source(
            source_status,
            protection_details.get("source") or "SemanticE2EVGuard.protect",
            reason=None if trace else "Protection backend did not return per-step loss records",
            formula="normalized backend trace; total=lambdaFeat*Lfeat+lambdaSem*Lsem+lambdaPsy*Lpsy+lambda2*L2 when total is absent",
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
        "_metricSources": sources,
    }


def _normalize_words(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\u4e00-\u9fff]+", " ", text)
    return [item for item in re.sub(r"\s+", " ", text).strip().split(" ") if item]


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
    else:
        substitutions, deletions, insertions, diff_ops = _edit_counts_and_ops(word_ref, word_hyp)
        normalizer = max(len(word_ref), 1)
        m = wer
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
        "insertRate": insertions / normalizer,
        "deleteRate": deletions / normalizer,
        "substituteRate": substitutions / normalizer,
        "breakdown": {
            "insertRate": insertions / normalizer,
            "deleteRate": deletions / normalizer,
            "substituteRate": substitutions / normalizer,
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


def compute_semantic_token_metrics(clean_path: Path, protected_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    del config
    details = {
        "tokenChangeRate": None,
        "tokenErrorRate": None,
        "tokenChangeCount": None,
        "tokenTotal": None,
        "semanticDrift": None,
        "encoderDistances": [],
        "status": "unavailable",
        "_metricSources": {
            "asrEval.tokenChangeRate": metric_source("unavailable", "semantic_tokenizer", reason="No real tokenizer is configured", formula="edit/token mismatch over encoded tokens"),
            "asrEval.tokenErrorRate": metric_source("unavailable", "semantic_tokenizer", reason="No real tokenizer is configured", formula="edit_distance(tokens,tokens_p)/max(len(tokens),1)"),
            "asrEval.semanticDrift": metric_source("unavailable", "semantic_encoder", reason="No real semantic encoder is configured", formula="mean(1-cosine(pool(F_k(x)),pool(F_k(xp))))"),
        },
    }
    if os.getenv("SEME2E_ENABLE_MFCC", "0") != "1":
        details["reason"] = "Set SEME2E_ENABLE_MFCC=1 to compute MFCC proxy semantic drift."
        return details
    try:
        import librosa

        x, sr = librosa.load(str(clean_path), sr=16000)
        xp, _ = librosa.load(str(protected_path), sr=16000)
        n = min(len(x), len(xp))
        if n <= 0:
            return details
        clean_mfcc = librosa.feature.mfcc(y=x[:n], sr=sr, n_mfcc=20).mean(axis=1)
        protected_mfcc = librosa.feature.mfcc(y=xp[:n], sr=sr, n_mfcc=20).mean(axis=1)
        denom = float(np.linalg.norm(clean_mfcc) * np.linalg.norm(protected_mfcc) + EPS)
        cosine = float(np.dot(clean_mfcc, protected_mfcc) / denom)
        drift = clamp(1.0 - cosine)
        details.update(
            {
                "semanticDrift": drift,
                "encoderDistances": [
                    {
                        "encoder": "MFCC",
                        "cosineBeforeAfter": cosine,
                        "distance": float(np.linalg.norm(clean_mfcc - protected_mfcc)),
                        "status": "partial",
                        "source": "mfcc_proxy",
                    }
                ],
                "status": "partial",
            }
        )
        details["_metricSources"]["asrEval.semanticDrift"] = metric_source(
            "partial",
            "mfcc_proxy",
            reason="MFCC proxy drift is reported only as semanticDrift; it is not used for tokenChangeRate or tokenErrorRate",
            formula="1-cosine(mean(MFCC(x)),mean(MFCC(xp)))",
        )
    except Exception as exc:
        details["status"] = "error"
        details["error"] = str(exc)
        details["_metricSources"]["asrEval.semanticDrift"] = metric_source("error", "mfcc_proxy", reason=str(exc), formula="1-cosine(mean(MFCC(x)),mean(MFCC(xp)))")
    return details


def _build_speaker_scorer() -> tuple[Any | None, str, dict[str, Any]]:
    metric = os.getenv("SEME2E_SPEAKER_METRIC", "ecapa")
    model = os.getenv("SEME2E_SPEAKER_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
    source = f"speaker_similarity:{metric}:{model}"
    if os.getenv("SEME2E_ENABLE_SPEAKER", "0") != "1":
        return None, source, metric_source("unavailable", source, reason="Set SEME2E_ENABLE_SPEAKER=1 to run speaker similarity dependencies", formula="cosine(Emb(a),Emb(b))")
    try:
        from speaker_similarity import build_speaker_similarity

        scorer = build_speaker_similarity(metric, model, os.getenv("SEME2E_API_DEVICE", "cpu"))
        return scorer, source, metric_source("available", source, formula="cosine(Emb(a),Emb(b))")
    except Exception as exc:
        return None, source, metric_source("error", source, reason=str(exc), formula="cosine(Emb(a),Emb(b))")


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
        base.update(
            {
                "simBefore": 1.0,
                "simAfter": direct_similarity,
                "simDropRate": 1.0 - direct_similarity,
                "embeddingDistanceBefore": 0.0,
                "embeddingDistanceAfter": 1.0 - direct_similarity,
                "simOriginalProtected": direct_similarity,
                "embeddingDistance": 1.0 - direct_similarity,
                "status": "available",
            }
        )
        base["_metricSources"]["speaker.*"] = metric_source("available", source, formula="directSimilarity=cosine(Emb(x),Emb(xp)); simDropRate=1-directSimilarity")
    except Exception as exc:
        base["status"] = "error"
        base["error"] = str(exc)
        base["_metricSources"]["speaker.*"] = metric_source("error", source, reason=str(exc), formula="directSimilarity=cosine(Emb(x),Emb(xp))")
    return base


def compute_clone_eval(
    original_audio_path: Path,
    original_clone_path: Path,
    protected_clone_path: Path,
    clone_result: dict[str, Any],
    speaker_model: Any | None = None,
    confidence_calibrator: Any | None = None,
) -> dict[str, Any]:
    scorer = speaker_model
    source = "speaker_similarity"
    source_info = metric_source("available", source, formula="SIM(a,b)=cosine(Emb(a),Emb(b))")
    if scorer is None:
        scorer, source, source_info = _build_speaker_scorer()
    request = clone_result.get("request") or {}
    eval_payload = {
        "cloneModel": request.get("model"),
        "speakerEvalModel": source,
        "targetText": request.get("text"),
        "originalCloneAudio": clone_result.get("originalCloneAudio"),
        "protectedCloneAudio": clone_result.get("protectedCloneAudio"),
        "originalSimilarity": None,
        "protectedSimilarity": None,
        "similarityDropRate": None,
        "embeddingDistanceBefore": None,
        "embeddingDistanceAfter": None,
        "embeddingDistanceIncreaseRate": None,
        "cloneConfidenceBefore": None,
        "cloneConfidenceAfter": None,
        "cloneConfidenceDropRate": None,
        "cloneRadar": [
            {"name": "相似度下降", "value": None},
            {"name": "嵌入距离增加", "value": None},
            {"name": "置信度下降", "value": None},
        ],
        "cloneTrend": None,
        "cloneDefenseScore": None,
        "createdAt": now_iso(),
        "status": source_info["status"],
        "_metricSources": {
            "cloneEval.*": source_info,
            "cloneEval.cloneConfidenceBefore": metric_source("unavailable", "confidence_calibrator", reason="No confidence calibrator is configured", formula="sigmoid(A*similarity+B)"),
            "cloneEval.cloneConfidenceAfter": metric_source("unavailable", "confidence_calibrator", reason="No confidence calibrator is configured", formula="sigmoid(A*similarity+B)"),
            "cloneEval.cloneTrend": metric_source("not_run", "multi_checkpoint_clone_eval", reason="No repeated checkpoint TTS clone evaluations were run", formula="None"),
        },
    }
    if scorer is None:
        eval_payload["reason"] = source_info.get("reason")
        return eval_payload
    try:
        original_similarity = finite_float(scorer.score(original_audio_path, original_clone_path))
        protected_similarity = finite_float(scorer.score(original_audio_path, protected_clone_path))
        if original_similarity is None or protected_similarity is None:
            raise ValueError("speaker scorer returned non-finite clone similarity")
        similarity_drop_rate = (original_similarity - protected_similarity) / max(original_similarity, EPS)
        embedding_before = 1.0 - original_similarity
        embedding_after = 1.0 - protected_similarity
        embedding_increase = (embedding_after - embedding_before) / max(embedding_before, EPS)
        conf_before = conf_after = conf_drop = None
        if confidence_calibrator is not None:
            conf_before = finite_float(confidence_calibrator(original_similarity))
            conf_after = finite_float(confidence_calibrator(protected_similarity))
            if conf_before is not None and conf_after is not None:
                conf_drop = (conf_before - conf_after) / max(conf_before, EPS)
        s_sim = clamp(similarity_drop_rate / 0.5)
        s_dist = clamp(embedding_increase / 1.0)
        s_conf = clamp(conf_drop / 0.5) if conf_drop is not None else None
        clone_score_base = weighted_available_mean(
            {
                "S_sim": (s_sim, 0.45),
                "S_dist": (s_dist, 0.25),
                "S_conf": (s_conf, 0.15),
            }
        )
        eval_payload.update(
            {
                "originalSimilarity": original_similarity,
                "protectedSimilarity": protected_similarity,
                "similarityDropRate": similarity_drop_rate,
                "embeddingDistanceBefore": embedding_before,
                "embeddingDistanceAfter": embedding_after,
                "embeddingDistanceIncreaseRate": embedding_increase,
                "cloneConfidenceBefore": conf_before,
                "cloneConfidenceAfter": conf_after,
                "cloneConfidenceDropRate": conf_drop,
                "cloneRadar": [
                    {"name": "相似度下降", "value": 100.0 * s_sim},
                    {"name": "嵌入距离增加", "value": 100.0 * s_dist},
                    {"name": "置信度下降", "value": 100.0 * s_conf if s_conf is not None else None},
                ],
                "cloneDefenseScore": 100.0 * clone_score_base if clone_score_base is not None else None,
                "status": "available",
            }
        )
        eval_payload["_metricSources"]["cloneEval.*"] = metric_source(
            "available",
            source,
            formula="originalSimilarity=SIM(originalAudio,originalCloneAudio); protectedSimilarity=SIM(originalAudio,protectedCloneAudio)",
        )
        if conf_drop is not None:
            eval_payload["_metricSources"]["cloneEval.cloneConfidenceBefore"] = metric_source("available", "confidence_calibrator", formula="sigmoid(A*similarity+B)")
            eval_payload["_metricSources"]["cloneEval.cloneConfidenceAfter"] = metric_source("available", "confidence_calibrator", formula="sigmoid(A*similarity+B)")
    except Exception as exc:
        eval_payload["status"] = "error"
        eval_payload["error"] = str(exc)
        eval_payload["_metricSources"]["cloneEval.*"] = metric_source("error", source, reason=str(exc), formula="SIM(originalAudio,cloneAudio)")
    return eval_payload


def compute_overall_score(result: dict[str, Any]) -> dict[str, Any]:
    details = result.get("details") or {}
    perception = details.get("perception") or {}
    generation = details.get("generation") or {}
    asr = details.get("asr") or {}
    clone_results = result.get("cloneResults") or []
    clone_eval = None
    if clone_results:
        clone_eval = (clone_results[-1] or {}).get("cloneEval")
    quality_score = finite_float(perception.get("qualityScore"))
    if quality_score is None:
        quality_score = finite_float((perception.get("protectionQuality") or {}).get("qualityScore"))
    over_mask_rate = finite_float(perception.get("overMaskRate"))
    if over_mask_rate is None:
        over_mask_rate = finite_float(perception.get("psychoacousticViolationRate"))
    epsilon_usage_rate = finite_float(perception.get("epsilonUsageRate"))
    if epsilon_usage_rate is None:
        epsilon_usage_rate = finite_float((perception.get("perturbation") or {}).get("epsilonUsageRate"))
    asr_score = finite_float(asr.get("asrProtectionScore"))
    clone_score = finite_float((clone_eval or {}).get("cloneDefenseScore") if isinstance(clone_eval, dict) else None)
    s_asr = asr_score / 100.0 if asr_score is not None and asr.get("status") in {"available", "computed", "partial", "completed", "success"} else None
    s_clone = clone_score / 100.0 if clone_score is not None else None
    s_audio = quality_score / 100.0 if quality_score is not None else None
    s_psy = 1.0 - clamp(over_mask_rate / 0.2) if over_mask_rate is not None else None
    s_epsilon = 1.0 - clamp(epsilon_usage_rate) if epsilon_usage_rate is not None else None
    score_base = weighted_available_mean(
        {
            "S_asr": (s_asr, 0.25),
            "S_clone": (s_clone, 0.35),
            "S_audio": (s_audio, 0.20),
            "S_psy": (s_psy, 0.10),
            "S_epsilon": (s_epsilon, 0.10),
        }
    )
    score = 100.0 * score_base if score_base is not None else None
    if score is None:
        verdict = "未完成评估"
    elif score >= 85:
        verdict = "强防护"
    elif score >= 70:
        verdict = "有效防护"
    elif score >= 50:
        verdict = "部分防护"
    else:
        verdict = "防护不足"
    source = metric_source(
        "available" if score is not None else "unavailable",
        "weighted_available_mean",
        reason=None if score is not None else "No score submetrics are available",
        formula="100*weighted_available_mean(S_asr:.25,S_clone:.35,S_audio:.20,S_psy:.10,S_epsilon:.10)",
    )
    return {"score": score, "verdict": verdict, "_metricSources": {"score": source, "verdict": source}}
