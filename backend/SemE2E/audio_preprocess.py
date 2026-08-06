from __future__ import annotations

import math
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parent
TARGET_SAMPLE_RATE = 24_000
NATIVE_AUDIO_FORMATS = ["wav", "flac", "ogg"]
COMPRESSED_AUDIO_FORMATS = ["webm", "opus", "mp4", "m4a", "aac", "mp3", "ogg"]
ProgressCallback = Callable[..., None]


class AudioPreprocessError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        reason: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.reason = reason
        self.diagnostics = diagnostics or {}


def _is_cancelled(cancel_event: Any | None) -> bool:
    return bool(cancel_event is not None and cancel_event.is_set())


def _check_cancelled(cancel_event: Any | None) -> None:
    if _is_cancelled(cancel_event):
        raise RuntimeError("TASK_CANCELLED")


def _emit(
    progress_callback: ProgressCallback | None,
    *,
    progress: float,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(progress=progress, stage="file_preprocess", message=message)


def _resolve_candidate(candidate: str | Path | None) -> Path | None:
    if not candidate:
        return None
    path = Path(candidate).expanduser()
    if path.is_dir():
        path = path / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    return resolved if resolved.is_file() else None


def resolve_ffmpeg() -> tuple[Path | None, str | None]:
    configured = _resolve_candidate(os.getenv("SEME2E_FFMPEG_PATH"))
    if configured is not None:
        return configured, "SEME2E_FFMPEG_PATH"

    executable_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    vendor_candidates = [
        ROOT / "vendor" / "ffmpeg" / executable_name,
        ROOT / "vendor" / "ffmpeg" / "bin" / executable_name,
        ROOT / "vendor" / executable_name,
    ]
    for candidate in vendor_candidates:
        resolved = _resolve_candidate(candidate)
        if resolved is not None:
            return resolved, "vendor"

    try:
        import imageio_ffmpeg

        resolved = _resolve_candidate(imageio_ffmpeg.get_ffmpeg_exe())
        if resolved is not None:
            return resolved, "imageio-ffmpeg"
    except Exception:
        pass

    system_ffmpeg = shutil.which("ffmpeg")
    resolved = _resolve_candidate(system_ffmpeg)
    if resolved is not None:
        return resolved, "PATH"
    return None, None


def audio_preprocess_capabilities() -> dict[str, Any]:
    ffmpeg_path, ffmpeg_source = resolve_ffmpeg()
    return {
        "status": "available" if ffmpeg_path is not None else "partial",
        "recordingSupported": ffmpeg_path is not None,
        "decoder": {
            "name": "ffmpeg" if ffmpeg_path is not None else None,
            "path": str(ffmpeg_path) if ffmpeg_path is not None else None,
            "source": ffmpeg_source,
        },
        "nativeFormats": NATIVE_AUDIO_FORMATS,
        "compressedFormats": COMPRESSED_AUDIO_FORMATS if ffmpeg_path is not None else [],
        "output": {
            "format": "WAV",
            "codec": "PCM_S16LE",
            "sampleRate": TARGET_SAMPLE_RATE,
            "channels": 1,
            "bitDepth": 16,
        },
        "reason": None if ffmpeg_path is not None else (
            "FFmpeg is unavailable; WAV/FLAC/OGG files readable by libsndfile still work, "
            "but browser WebM/Opus recordings cannot be decoded."
        ),
    }


def _source_diagnostics(source_path: Path) -> dict[str, Any]:
    return {
        "path": str(source_path),
        "filename": source_path.name,
        "format": source_path.suffix.lstrip(".").upper() or "AUDIO",
        "sizeBytes": source_path.stat().st_size if source_path.exists() else None,
    }


def _read_with_soundfile(source_path: Path, target_sample_rate: int) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(source_path), dtype="float32", always_2d=True)
    if audio.shape[0] <= 0 or sample_rate <= 0:
        raise AudioPreprocessError(
            "音频文件不包含可处理的采样数据。",
            code="AUDIO_EMPTY",
            reason="empty_audio",
            diagnostics={"source": _source_diagnostics(source_path)},
        )
    mono = np.nan_to_num(audio.mean(axis=1), nan=0.0, posinf=1.0, neginf=-1.0)
    if sample_rate != target_sample_rate:
        divisor = math.gcd(int(sample_rate), int(target_sample_rate))
        mono = resample_poly(mono, target_sample_rate // divisor, sample_rate // divisor)
    return np.clip(mono, -1.0, 1.0).astype(np.float32, copy=False), int(sample_rate)


def _run_ffmpeg(
    ffmpeg_path: Path,
    source_path: Path,
    temporary_path: Path,
    target_sample_rate: int,
    cancel_event: Any | None,
) -> None:
    command = [
        str(ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(target_sample_rate),
        "-c:a",
        "pcm_s16le",
        str(temporary_path),
    ]
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creation_flags,
    )
    stderr = ""
    try:
        while process.poll() is None:
            if _is_cancelled(cancel_event):
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
                raise RuntimeError("TASK_CANCELLED")
            time.sleep(0.1)
        stderr = process.stderr.read().strip() if process.stderr is not None else ""
    finally:
        if process.stderr is not None:
            process.stderr.close()
    if process.returncode != 0:
        raise AudioPreprocessError(
            "录音解码失败，请重新录音或上传有效的音频文件。",
            code="AUDIO_PREPROCESS_FAILED",
            reason="ffmpeg_decode_failed",
            diagnostics={
                "source": _source_diagnostics(source_path),
                "decoder": {"name": "ffmpeg", "path": str(ffmpeg_path)},
                "returnCode": process.returncode,
                "stderr": stderr[-4000:],
            },
        )


def _validated_output_meta(path: Path, target_sample_rate: int) -> dict[str, Any]:
    try:
        info = sf.info(str(path))
    except Exception as exc:
        raise AudioPreprocessError(
            "预处理输出无法读取。",
            code="AUDIO_PREPROCESS_FAILED",
            reason="normalized_audio_unreadable",
            diagnostics={"outputPath": str(path), "exception": f"{type(exc).__name__}: {exc}"},
        ) from exc
    if info.frames <= 0:
        raise AudioPreprocessError(
            "预处理输出不包含音频采样。",
            code="AUDIO_EMPTY",
            reason="normalized_audio_empty",
            diagnostics={"outputPath": str(path)},
        )
    if info.channels != 1 or info.samplerate != target_sample_rate or info.subtype != "PCM_16":
        raise AudioPreprocessError(
            "预处理输出未满足规范 WAV 要求。",
            code="AUDIO_PREPROCESS_FAILED",
            reason="normalized_audio_invalid",
            diagnostics={
                "outputPath": str(path),
                "sampleRate": info.samplerate,
                "channels": info.channels,
                "subtype": info.subtype,
            },
        )
    return {
        "path": str(path),
        "filename": path.name,
        "format": "WAV",
        "codec": "PCM_S16LE",
        "durationSec": info.duration,
        "sampleRate": info.samplerate,
        "channels": info.channels,
        "bitDepth": 16,
        "frames": info.frames,
        "sizeBytes": path.stat().st_size,
    }


def preprocess_audio(
    source_path: Path,
    output_wav: Path,
    *,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    cancel_event: Any | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    source_path = Path(source_path)
    output_wav = Path(output_wav)
    started_at = time.perf_counter()
    source = _source_diagnostics(source_path)
    if not source_path.is_file():
        raise AudioPreprocessError(
            "待处理的音频文件不存在。",
            code="AUDIO_PREPROCESS_FAILED",
            reason="source_audio_missing",
            diagnostics={"source": source},
        )
    if source_path.stat().st_size <= 0:
        raise AudioPreprocessError(
            "录音文件为空，请重新录音。",
            code="AUDIO_EMPTY",
            reason="source_audio_empty",
            diagnostics={"source": source},
        )

    _check_cancelled(cancel_event)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_wav.with_name(f".{output_wav.stem}.{uuid.uuid4().hex}.tmp.wav")
    decoder: dict[str, Any]
    source_sample_rate: int | None = None
    soundfile_error: str | None = None
    try:
        _emit(progress_callback, progress=0.08, message="正在检查录音格式")
        try:
            audio, source_sample_rate = _read_with_soundfile(source_path, target_sample_rate)
            _check_cancelled(cancel_event)
            _emit(progress_callback, progress=0.12, message="正在转换为规范 WAV")
            sf.write(str(temporary_path), audio, target_sample_rate, format="WAV", subtype="PCM_16")
            decoder = {"name": "libsndfile", "path": None, "source": "soundfile"}
        except AudioPreprocessError:
            raise
        except Exception as exc:
            soundfile_error = f"{type(exc).__name__}: {exc}"
            ffmpeg_path, ffmpeg_source = resolve_ffmpeg()
            if ffmpeg_path is None:
                raise AudioPreprocessError(
                    "当前环境缺少 FFmpeg，无法读取浏览器录制的 WebM/Opus 音频。",
                    code="AUDIO_DECODER_UNAVAILABLE",
                    reason="ffmpeg_unavailable",
                    diagnostics={
                        "source": source,
                        "soundfileError": soundfile_error,
                        "capabilities": audio_preprocess_capabilities(),
                    },
                ) from exc
            _check_cancelled(cancel_event)
            _emit(progress_callback, progress=0.12, message="正在解码浏览器录音")
            _run_ffmpeg(ffmpeg_path, source_path, temporary_path, target_sample_rate, cancel_event)
            decoder = {"name": "ffmpeg", "path": str(ffmpeg_path), "source": ffmpeg_source}

        _check_cancelled(cancel_event)
        temporary_meta = _validated_output_meta(temporary_path, target_sample_rate)
        os.replace(temporary_path, output_wav)
        output = {**temporary_meta, "path": str(output_wav), "filename": output_wav.name}
        _emit(progress_callback, progress=0.17, message="录音预处理完成")
        return {
            "status": "normalized",
            "source": {**source, "sampleRate": source_sample_rate},
            "output": output,
            "decoder": decoder,
            "target": {
                "format": "WAV",
                "codec": "PCM_S16LE",
                "sampleRate": target_sample_rate,
                "channels": 1,
                "bitDepth": 16,
            },
            "soundfileReadError": soundfile_error,
            "elapsedSec": round(time.perf_counter() - started_at, 3),
        }
    except AudioPreprocessError:
        raise
    except RuntimeError as exc:
        if str(exc) == "TASK_CANCELLED":
            raise
        raise AudioPreprocessError(
            "音频预处理失败，请重新录音或上传有效音频。",
            code="AUDIO_PREPROCESS_FAILED",
            reason="runtime_preprocess_error",
            diagnostics={
                "source": source,
                "outputPath": str(output_wav),
                "exception": f"{type(exc).__name__}: {exc}",
                "soundfileError": soundfile_error,
            },
        ) from exc
    except Exception as exc:
        raise AudioPreprocessError(
            "音频预处理失败，请重新录音或上传有效音频。",
            code="AUDIO_PREPROCESS_FAILED",
            reason="unexpected_preprocess_error",
            diagnostics={
                "source": source,
                "outputPath": str(output_wav),
                "exception": f"{type(exc).__name__}: {exc}",
                "soundfileError": soundfile_error,
            },
        ) from exc
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
