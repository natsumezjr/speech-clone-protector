from __future__ import annotations

import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gpt_sovits_live_finetune as live_finetune


def write_silent_wav(path: Path, *, duration_sec: float, sample_rate: int = 1000) -> None:
    frame_count = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(b"\x00\x00" * frame_count)


class GptSovitsLiveFineTuneTests(unittest.TestCase):
    def test_inference_uses_trimmed_reference_when_source_exceeds_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original.wav"
            protected = root / "protected.wav"
            write_silent_wav(original, duration_sec=12.0)
            write_silent_wav(protected, duration_sec=12.0)
            args = SimpleNamespace(
                work_dir=root / "work",
                repo=root / "repo",
                cuda_visible_devices="",
                original_audio=original,
                protected_audio=protected,
                original_transcript="original transcript",
                protected_transcript="protected transcript",
                original_output=root / "original-output.wav",
                protected_output=root / "protected-output.wav",
                max_training_seconds=54.0,
                min_reference_seconds=3.0,
                max_reference_seconds=10.0,
            )
            inference_references: list[Path] = []

            def fake_prepare(**kwargs: object):
                condition_dir = Path(kwargs["condition_dir"])
                return {"prepareWallSec": 0.01}, condition_dir / "phoneme.txt", condition_dir / "semantic.tsv"

            def fake_train(**kwargs: object):
                condition_dir = Path(kwargs["condition_dir"])
                return {"trainWallSec": 0.01}, condition_dir / "gpt.ckpt", condition_dir / "sovits.pth"

            def fake_infer(**kwargs: object) -> float:
                inference_references.append(Path(kwargs["audio_path"]))
                return 0.01

            with (
                mock.patch.object(live_finetune, "_prepare", side_effect=fake_prepare),
                mock.patch.object(live_finetune, "_train", side_effect=fake_train),
                mock.patch.object(live_finetune, "_infer", side_effect=fake_infer),
            ):
                result = live_finetune.run(args)

            self.assertEqual(len(inference_references), 2)
            self.assertTrue(all(path.name.endswith("_reference.wav") for path in inference_references))
            for path in inference_references:
                with wave.open(str(path), "rb") as source:
                    self.assertAlmostEqual(source.getnframes() / source.getframerate(), 10.0, places=3)
            self.assertEqual(result["original"]["sourceDurationSec"], 12.0)
            self.assertEqual(result["original"]["trainingDurationSec"], 12.0)
            self.assertEqual(result["original"]["referenceDurationSec"], 10.0)
            self.assertEqual(result["protected"]["sourceDurationSec"], 12.0)
            self.assertEqual(result["protected"]["trainingDurationSec"], 12.0)
            self.assertEqual(result["protected"]["referenceDurationSec"], 10.0)

    def test_reference_audio_keeps_valid_boundaries_and_rejects_too_short_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            condition_dir = root / "condition"
            condition_dir.mkdir()
            for duration in (9.9, 10.0):
                audio = root / f"valid-{duration}.wav"
                write_silent_wav(audio, duration_sec=duration)
                reference, source_duration, reference_duration = live_finetune._reference_audio(
                    audio,
                    condition_dir,
                    min_seconds=3.0,
                    max_seconds=10.0,
                )
                self.assertEqual(reference, audio)
                self.assertAlmostEqual(source_duration, duration, places=3)
                self.assertAlmostEqual(reference_duration, duration, places=3)

            too_short = root / "too-short.wav"
            write_silent_wav(too_short, duration_sec=2.9)
            with self.assertRaisesRegex(ValueError, "at least 3.00 seconds"):
                live_finetune._reference_audio(
                    too_short,
                    condition_dir,
                    min_seconds=3.0,
                    max_seconds=10.0,
                )


if __name__ == "__main__":
    unittest.main()
