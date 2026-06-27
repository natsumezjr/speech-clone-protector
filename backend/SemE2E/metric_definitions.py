from __future__ import annotations

import importlib.util
import math
import os
import re
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


EPS = 1.0e-12
ROOT = Path(__file__).resolve().parent
_S3_TOKENIZER_CACHE: dict[tuple[str, str], Any] = {}
_SEMANTIC_ENCODER_CACHE: dict[tuple[str, str, str, str], Any] = {}
_SEMANTIC_ENCODER_LAST_ERROR: str | None = None


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
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return raw


def load_s3_tokenizer(model_name_or_path: str | None = None, device: str | None = None) -> Any:
    """Load the S3 semantic tokenizer used for real token metrics."""

    model = _resolve_local_model_path(model_name_or_path or os.getenv("SEME2E_TOKENIZER_MODEL") or "speech_tokenizer_v1_25hz")
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
            formula="normalized backend trace; total=lambdaId*Lid+lambdaSem*Lsem+lambdaPsy*Lpsy+lambda2*L2 when total is absent",
        ),
        "lossFinal.Lid": metric_source(
            source_status,
            protection_details.get("source") or "SemanticE2EVGuard.protect",
            reason=None if trace else "Protection backend did not return an optimization trace",
            formula="L_{\\mathrm{id}}",
            metric="Identity loss from SemanticE2EVGuard optimization trace.",
        ),
        "optimizationTrace.Lid": metric_source(
            source_status,
            protection_details.get("source") or "SemanticE2EVGuard.protect",
            reason=None if trace else "Protection backend did not return per-step loss records",
            formula="L_{\\mathrm{id}}",
            metric="Identity loss from SemanticE2EVGuard optimization trace.",
        ),
        "lossFinal.Lfeat": metric_source(
            source_status,
            protection_details.get("source") or "SemanticE2EVGuard.protect",
            reason="Deprecated legacy alias of Lid.",
            formula="Lfeat := Lid",
        ),
        "optimizationTrace.Lfeat": metric_source(
            source_status,
            protection_details.get("source") or "SemanticE2EVGuard.protect",
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
    resolved = _resolve_local_model_path(requested)
    if isinstance(resolved, str) and Path(resolved).expanduser().exists():
        return resolved
    for fallback in cached_fallbacks:
        resolved_fallback = _resolve_local_model_path(fallback)
        if isinstance(resolved_fallback, str) and Path(resolved_fallback).expanduser().exists():
            return resolved_fallback
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

        _SEMANTIC_ENCODER_LAST_ERROR = None
        device = os.getenv("SEME2E_SEMANTIC_ENCODER_DEVICE") or os.getenv("SEME2E_API_DEVICE") or os.getenv("SEME2E_TOKENIZER_DEVICE") or "cpu"
        tokenizer_path = (
            os.getenv("SEME2E_SEMANTIC_TOKENIZER_MODEL")
            or os.getenv("SEME2E_TOKENIZER_MODEL")
            or config.get("tokenizerPath")
            or "speech_tokenizer_v1_25hz"
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
        if cache_key not in _SEMANTIC_ENCODER_CACHE:
            _SEMANTIC_ENCODER_CACHE[cache_key] = SemanticEncoderEnsemble(
                device=device,
                tokenizer_path=tokenizer_path,
                hubert_path=hubert_path,
                whisper_path=whisper_path,
                sample_rate=16000,
            )
        return _SEMANTIC_ENCODER_CACHE[cache_key]
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
            source = os.getenv("SEME2E_TOKENIZER_MODEL") or "speech_tokenizer_v1_25hz"
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
        base["_metricSources"]["speaker.*"] = metric_source("available", source, formula="directSimilarity=cosine(Emb(x),Emb(xp)); simDropRate=1-directSimilarity", metric=source_info.get("metric"))
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
) -> dict[str, Any]:
    scorer = speaker_model
    source = "speaker_similarity"
    source_info = metric_source("available", source, formula="SIM(a,b)=cosine(Emb(a),Emb(b))")
    if scorer is None:
        scorer, source, source_info = _build_speaker_scorer()
    request = clone_result.get("request") or {}
    unavailable_reason = source_info.get("reason") or "speaker similarity is not available"
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
        "embeddingDistanceIncreaseRate": None,
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
            "cloneEval.cloneConfidenceBefore": metric_source("unavailable", "confidence_calibrator", reason="No confidence calibrator is configured", formula="sigmoid(A*similarity+B)"),
            "cloneEval.cloneConfidenceAfter": metric_source("unavailable", "confidence_calibrator", reason="No confidence calibrator is configured", formula="sigmoid(A*similarity+B)"),
            "cloneEval.cloneConfidenceDropRate": metric_source("unavailable", "confidence_calibrator", reason="No confidence calibrator is configured", formula="(cloneConfidenceBefore-cloneConfidenceAfter)/max(cloneConfidenceBefore,EPS)"),
            "cloneEval.cloneTrend": metric_source("not_run", "multi_checkpoint_clone_eval", reason="clone trend is disabled; only final clone evaluation is reported", formula="None"),
        },
    }
    if scorer is None:
        eval_payload["reason"] = source_info.get("reason")
        return eval_payload
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
                "S_dist": (s_dist, 0.35),
                "S_conf": (s_conf, 0.20),
            }
        )
        eval_payload.update(
            {
                "directSimilarity": direct_similarity,
                "originalSimilarity": original_similarity,
                "protectedSimilarity": protected_similarity,
                "similarityDropRate": similarity_drop_rate,
                "embeddingDistanceBefore": embedding_before,
                "embeddingDistanceAfter": embedding_after,
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
                "cloneDefenseScore": 100.0 * clone_score_base if clone_score_base is not None else None,
                "status": "available",
            }
        )
        eval_payload["_metricSources"]["cloneEval.*"] = metric_source(
            "available",
            source,
            formula="originalSimilarity=SIM(originalAudio,originalCloneAudio); protectedSimilarity=SIM(originalAudio,protectedCloneAudio); similarityDropRate=(originalSimilarity-protectedSimilarity)/max(originalSimilarity,EPS); embeddingDistanceBefore=1-originalSimilarity; embeddingDistanceAfter=1-protectedSimilarity; embeddingDistanceIncreaseRate=(embeddingDistanceAfter-embeddingDistanceBefore)/max(embeddingDistanceBefore,EPS)",
            metric=source_info.get("metric"),
        )
        if conf_drop is not None:
            eval_payload["_metricSources"]["cloneEval.cloneConfidenceBefore"] = metric_source("available", "confidence_calibrator", formula="sigmoid(A*similarity+B)")
            eval_payload["_metricSources"]["cloneEval.cloneConfidenceAfter"] = metric_source("available", "confidence_calibrator", formula="sigmoid(A*similarity+B)")
            eval_payload["_metricSources"]["cloneEval.cloneConfidenceDropRate"] = metric_source("available", "confidence_calibrator", formula="(cloneConfidenceBefore-cloneConfidenceAfter)/max(cloneConfidenceBefore,EPS)")
    except Exception as exc:
        eval_payload["status"] = "error"
        eval_payload["error"] = str(exc)
        eval_payload["_metricSources"]["cloneEval.*"] = metric_source("error", source, reason=str(exc), formula="SIM(originalAudio,cloneAudio)", metric=source_info.get("metric"))
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
        verdict = "未生成评分"
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
