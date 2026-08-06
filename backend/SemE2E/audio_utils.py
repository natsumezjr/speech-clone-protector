from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as ta_functional


def load_mono(path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    y, in_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr is not None and in_sr != sr:
        waveform = torch.from_numpy(np.ascontiguousarray(y)).float().unsqueeze(0)
        y = ta_functional.resample(waveform, in_sr, sr).squeeze(0).cpu().numpy()
        in_sr = sr
    return np.asarray(y, dtype=np.float32), in_sr


def audio_metrics(clean_path: str | Path, audio_path: str | Path) -> dict[str, float]:
    clean, sr_clean = load_mono(clean_path)
    audio, _ = load_mono(audio_path, sr_clean)

    n = min(len(clean), len(audio))
    clean = clean[:n]
    audio = audio[:n]
    noise = audio - clean

    signal_power = float(np.sum(clean * clean) + 1.0e-12)
    noise_power = float(np.sum(noise * noise) + 1.0e-12)
    corr = float(np.corrcoef(clean, audio)[0, 1]) if n > 1 else 0.0

    return {
        "duration_s": round(n / sr_clean, 4),
        "snr_db": 10.0 * np.log10(signal_power / noise_power),
        "noise_linf": float(np.max(np.abs(noise))) if n else 0.0,
        "noise_rms": float(np.sqrt(np.mean(noise * noise))) if n else 0.0,
        "wave_corr": corr,
    }
