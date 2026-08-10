from __future__ import annotations

import math
import os
import wave
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DNSMOS_MODEL = ROOT / "checkpoints" / "dnsmos" / "sig_bak_ovr.onnx"
SAMPLE_RATE = 16000
INPUT_LENGTH_SECONDS = 9.01


def resolve_dnsmos_model_path(value: str | os.PathLike[str] | None = None) -> Path:
    configured = value or os.getenv("SEME2E_DNSMOS_MODEL")
    path = Path(configured).expanduser() if configured else DEFAULT_DNSMOS_MODEL
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def dnsmos_model_status(value: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    path = resolve_dnsmos_model_path(value)
    try:
        ready = path.is_file() and path.stat().st_size >= 100_000
        if ready:
            with path.open("rb") as model_file:
                ready = not model_file.read(64).startswith(b"version https://git-lfs.github.com/spec")
    except OSError:
        ready = False
    if not ready:
        return {
            "status": "unavailable",
            "model": "DNSMOS P.835 OVRL",
            "modelPath": str(path),
            "reason": "语音质量评分模型尚未安装",
        }
    try:
        import onnxruntime  # noqa: F401
    except Exception as exc:
        return {
            "status": "unavailable",
            "model": "DNSMOS P.835 OVRL",
            "modelPath": str(path),
            "reason": f"语音质量评分运行依赖不可用：{exc}",
        }
    return {
        "status": "available",
        "model": "DNSMOS P.835 OVRL",
        "modelPath": str(path),
        "reason": None,
    }


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        data = np.asarray(audio, dtype=np.float32)
    except Exception:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
        if width == 1:
            data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
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
    if data.size == 0:
        raise ValueError(f"audio is empty: {path}")
    return np.clip(data, -1.0, 1.0), int(sample_rate)


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32)
    try:
        import librosa

        return librosa.resample(
            audio.astype(np.float32),
            orig_sr=source_rate,
            target_sr=target_rate,
        ).astype(np.float32)
    except Exception:
        pass
    try:
        from scipy.signal import resample_poly

        divisor = math.gcd(int(source_rate), int(target_rate))
        return resample_poly(audio, target_rate // divisor, source_rate // divisor).astype(np.float32)
    except Exception:
        target_length = max(1, int(round(len(audio) * target_rate / source_rate)))
        source_axis = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        target_axis = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
        return np.interp(target_axis, source_axis, audio).astype(np.float32)


def _calibrate(raw_sig: float, raw_bak: float, raw_ovrl: float) -> dict[str, float]:
    # Official non-personalized DNSMOS P.835 polynomial calibration.
    sig = np.polyval([-0.08397278, 1.22083953, 0.00524390], raw_sig)
    bak = np.polyval([-0.13166888, 1.60915514, -0.39604546], raw_bak)
    ovrl = np.polyval([-0.06766283, 1.11546468, 0.04602535], raw_ovrl)
    return {
        "sig": float(np.clip(sig, 1.0, 5.0)),
        "bak": float(np.clip(bak, 1.0, 5.0)),
        "ovrl": float(np.clip(ovrl, 1.0, 5.0)),
    }


def _evaluate_audio(session: Any, audio_path: Path) -> dict[str, Any]:
    audio, sample_rate = _read_audio(audio_path)
    audio = _resample(audio, sample_rate, SAMPLE_RATE)
    segment_length = int(INPUT_LENGTH_SECONDS * SAMPLE_RATE)
    if len(audio) < segment_length:
        repeats = int(math.ceil(segment_length / max(len(audio), 1)))
        audio = np.tile(audio, repeats)
    # Match the Microsoft/SpeechBrain DNSMOS recipe exactly. The floor is
    # applied to whole audio seconds before subtracting the 9.01 s window.
    hop_count = max(1, int(math.floor(len(audio) / SAMPLE_RATE) - INPUT_LENGTH_SECONDS) + 1)
    input_name = session.get_inputs()[0].name
    calibrated_rows: list[dict[str, float]] = []
    raw_rows: list[tuple[float, float, float]] = []
    for index in range(hop_count):
        start = index * SAMPLE_RATE
        segment = audio[start : start + segment_length]
        if len(segment) < segment_length:
            break
        output = np.asarray(
            session.run(None, {input_name: segment.astype(np.float32)[np.newaxis, :]})[0]
        ).reshape(-1)
        if output.size < 3:
            raise ValueError(f"DNSMOS model returned {output.size} values; expected SIG/BAK/OVRL")
        raw_sig, raw_bak, raw_ovrl = (float(output[0]), float(output[1]), float(output[2]))
        raw_rows.append((raw_sig, raw_bak, raw_ovrl))
        calibrated_rows.append(_calibrate(raw_sig, raw_bak, raw_ovrl))
    if not calibrated_rows:
        raise ValueError(f"DNSMOS could not construct a complete input segment: {audio_path}")
    return {
        "sig": float(np.mean([row["sig"] for row in calibrated_rows])),
        "bak": float(np.mean([row["bak"] for row in calibrated_rows])),
        "ovrl": float(np.mean([row["ovrl"] for row in calibrated_rows])),
        "rawSig": float(np.mean([row[0] for row in raw_rows])),
        "rawBak": float(np.mean([row[1] for row in raw_rows])),
        "rawOvrl": float(np.mean([row[2] for row in raw_rows])),
        "segmentCount": len(calibrated_rows),
        "sampleRate": SAMPLE_RATE,
    }


def evaluate_dnsmos_pair(
    clean_audio_path: str | os.PathLike[str],
    protected_audio_path: str | os.PathLike[str],
    model_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    status = dnsmos_model_status(model_path)
    if status["status"] != "available":
        return status
    import onnxruntime as ort

    resolved_model = Path(status["modelPath"])
    session_options = ort.SessionOptions()
    session_options.inter_op_num_threads = 1
    session_options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(resolved_model),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    clean = _evaluate_audio(session, Path(clean_audio_path).expanduser().resolve())
    protected = _evaluate_audio(session, Path(protected_audio_path).expanduser().resolve())
    return {
        **status,
        "provider": "CPUExecutionProvider",
        "clean": clean,
        "protected": protected,
        "cleanMos": clean["ovrl"],
        "protectedMos": protected["ovrl"],
    }
