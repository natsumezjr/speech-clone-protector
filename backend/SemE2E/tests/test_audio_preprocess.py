from __future__ import annotations

import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_preprocess import (
    AudioPreprocessError,
    TARGET_SAMPLE_RATE,
    audio_preprocess_capabilities,
    preprocess_audio,
    resolve_ffmpeg,
)


class AudioPreprocessTests(unittest.TestCase):
    @staticmethod
    def _write_stereo_wav(path: Path, sample_rate: int = 48_000) -> None:
        duration_sec = 0.5
        time_axis = np.arange(round(sample_rate * duration_sec), dtype=np.float32) / sample_rate
        left = 0.2 * np.sin(2 * np.pi * 220 * time_axis)
        right = 0.1 * np.sin(2 * np.pi * 330 * time_axis)
        sf.write(str(path), np.column_stack([left, right]), sample_rate, subtype="PCM_16")

    def test_normalizes_native_wav_to_mono_pcm16_24khz(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "stereo-48k.wav"
            output = root / "normalized.wav"
            self._write_stereo_wav(source)

            metadata = preprocess_audio(source, output)
            info = sf.info(str(output))

            self.assertEqual(info.samplerate, TARGET_SAMPLE_RATE)
            self.assertEqual(info.channels, 1)
            self.assertEqual(info.subtype, "PCM_16")
            self.assertEqual(metadata["decoder"]["name"], "libsndfile")
            self.assertEqual(metadata["output"]["sampleRate"], TARGET_SAMPLE_RATE)

    def test_decodes_browser_webm_opus_with_resolved_ffmpeg(self) -> None:
        ffmpeg_path, _source = resolve_ffmpeg()
        if ffmpeg_path is None:
            self.skipTest("FFmpeg decoder is unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_wav = root / "recording-source.wav"
            source_webm = root / "recorded_voice.webm"
            output = root / "normalized.wav"
            self._write_stereo_wav(source_wav)
            subprocess.run(
                [
                    str(ffmpeg_path),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source_wav),
                    "-c:a",
                    "libopus",
                    str(source_webm),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            metadata = preprocess_audio(source_webm, output)
            info = sf.info(str(output))

            self.assertEqual(metadata["decoder"]["name"], "ffmpeg")
            self.assertEqual(info.samplerate, TARGET_SAMPLE_RATE)
            self.assertEqual(info.channels, 1)
            self.assertEqual(info.subtype, "PCM_16")

    def test_invalid_recording_returns_structured_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "broken.webm"
            source.write_bytes(b"not a real recording")

            with self.assertRaises(AudioPreprocessError) as context:
                preprocess_audio(source, root / "normalized.wav")

            self.assertEqual(context.exception.code, "AUDIO_PREPROCESS_FAILED")
            self.assertEqual(context.exception.reason, "ffmpeg_decode_failed")

    def test_cancellation_is_checked_before_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.wav"
            self._write_stereo_wav(source)
            cancel_event = threading.Event()
            cancel_event.set()

            with self.assertRaisesRegex(RuntimeError, "TASK_CANCELLED"):
                preprocess_audio(source, root / "normalized.wav", cancel_event=cancel_event)

    def test_capabilities_describe_canonical_output(self) -> None:
        capabilities = audio_preprocess_capabilities()
        self.assertEqual(capabilities["output"]["format"], "WAV")
        self.assertEqual(capabilities["output"]["sampleRate"], TARGET_SAMPLE_RATE)
        self.assertEqual(capabilities["output"]["channels"], 1)
        self.assertEqual(capabilities["output"]["bitDepth"], 16)


if __name__ == "__main__":
    unittest.main()
