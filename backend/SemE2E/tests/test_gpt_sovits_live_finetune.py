from __future__ import annotations

import os
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
import gpt_sovits_worker


def write_silent_wav(path: Path, *, duration_sec: float, sample_rate: int = 1000) -> None:
    frame_count = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(b"\x00\x00" * frame_count)


class GptSovitsLiveFineTuneTests(unittest.TestCase):
    def test_inference_configures_absolute_bert_path_before_runtime_imports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bert = Path(tmp) / "bert"
            bert.mkdir()
            args = SimpleNamespace(bert=bert)
            with mock.patch.dict(os.environ, {}, clear=True):
                gpt_sovits_worker._configure_runtime_environment(args)
                self.assertEqual(os.environ["bert_path"], str(bert.resolve()))
                self.assertEqual(os.environ["bert_pretrained_dir"], str(bert.resolve()))

    def test_reference_language_is_detected_independently_from_target_language(self) -> None:
        self.assertEqual(
            live_finetune._reference_language("Ladies and gentlemen, good afternoon.", fallback="zh-cn"),
            "en",
        )
        self.assertEqual(
            live_finetune._reference_language("各位女士先生们，大家晚上好。", fallback="en"),
            "zh",
        )
        self.assertEqual(
            live_finetune._reference_language("各位女士先生们，大家晚上好。", configured="en"),
            "en",
        )

    def test_prepared_dataset_validation_accepts_matching_basenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phoneme = root / "2-name2text.txt"
            semantic = root / "6-name2semantic.tsv"
            phoneme.write_text(
                "record.wav\ta b c\t[1, 1, 1]\t测试\n",
                encoding="utf-8",
            )
            semantic.write_text(
                "item_name\tsemantic_audio\nrecord.wav\t1 2 3\n",
                encoding="utf-8",
            )

            live_finetune._validate_prepared_dataset(
                phoneme,
                semantic,
                condition="original",
                transcript_language="zh",
            )

    def test_prepared_dataset_keys_are_rewritten_to_basenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phoneme = root / "2-name2text.txt"
            semantic = root / "6-name2semantic.tsv"
            phoneme.write_text(
                "/tmp/input/record.wav\ta b c\t[1, 1, 1]\t测试\n",
                encoding="utf-8",
            )
            semantic.write_text(
                "item_name\tsemantic_audio\nC:\\input\\record.wav\t1 2 3\n",
                encoding="utf-8",
            )

            live_finetune._normalize_prepared_keys(phoneme, semantic=False)
            live_finetune._normalize_prepared_keys(semantic, semantic=True)

            self.assertTrue(phoneme.read_text(encoding="utf-8").startswith("record.wav\t"))
            self.assertIn("record.wav\t1 2 3", semantic.read_text(encoding="utf-8"))

    def test_prepared_dataset_validation_rejects_empty_or_mismatched_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phoneme = root / "2-name2text.txt"
            semantic = root / "6-name2semantic.tsv"
            phoneme.write_text("\n", encoding="utf-8")
            semantic.write_text(
                "item_name\tsemantic_audio\nrecord.wav\t1 2 3\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "no phoneme entries"):
                live_finetune._validate_prepared_dataset(
                    phoneme,
                    semantic,
                    condition="original",
                    transcript_language="zh",
                )

            phoneme.write_text(
                "phoneme.wav\ta b c\t[1, 1, 1]\ttest\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "key mismatch"):
                live_finetune._validate_prepared_dataset(
                    phoneme,
                    semantic,
                    condition="protected",
                    transcript_language="en",
                )

    def test_prepare_passes_local_bert_path_to_chinese_g2pw_frontend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            condition_dir = root / "condition"
            condition_dir.mkdir()
            audio = root / "reference.wav"
            audio.write_bytes(b"audio")
            bert = root / "bert"
            bert.mkdir()
            prepared_dir = condition_dir / "prepared"
            captured_environments: list[dict[str, str]] = []

            def fake_run(_command: list[str], **kwargs: object) -> float:
                environment = dict(kwargs["environment"])
                captured_environments.append(environment)
                prepared_dir.mkdir(parents=True, exist_ok=True)
                if len(captured_environments) == 1:
                    (prepared_dir / "2-name2text-0.txt").write_text(
                        "reference.wav\ta b\t[1, 1]\t测试\n",
                        encoding="utf-8",
                    )
                elif len(captured_environments) == 3:
                    (prepared_dir / "6-name2semantic-0.tsv").write_text(
                        "reference.wav\t1 2\n",
                        encoding="utf-8",
                    )
                return 0.01

            args = SimpleNamespace(
                language="zh-cn",
                bert=bert,
                cnhubert=root / "cnhubert",
                pretrained_s2g=root / "s2g.pth",
                repo=repo,
                python=root / "python",
                timeout=10,
            )
            with mock.patch.object(live_finetune, "_run", side_effect=fake_run):
                live_finetune._prepare(
                    args=args,
                    condition="original",
                    audio_path=audio,
                    transcript="中文测试",
                    transcript_language="zh",
                    condition_dir=condition_dir,
                    environment={},
                )

            self.assertEqual(len(captured_environments), 3)
            for environment in captured_environments:
                self.assertEqual(environment["bert_pretrained_dir"], str(bert))
                self.assertEqual(environment["bert_path"], str(bert))

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
            preparation_languages: list[str] = []
            inference_languages: list[tuple[str, str]] = []

            def fake_prepare(**kwargs: object):
                condition_dir = Path(kwargs["condition_dir"])
                preparation_languages.append(str(kwargs["transcript_language"]))
                return {"prepareWallSec": 0.01}, condition_dir / "phoneme.txt", condition_dir / "semantic.tsv"

            def fake_train(**kwargs: object):
                condition_dir = Path(kwargs["condition_dir"])
                return {"trainWallSec": 0.01}, condition_dir / "gpt.ckpt", condition_dir / "sovits.pth"

            def fake_infer(**kwargs: object) -> float:
                inference_references.append(Path(kwargs["audio_path"]))
                inference_languages.append(
                    (str(kwargs["prompt_language"]), str(kwargs["text_language"]))
                )
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
            self.assertEqual(preparation_languages, ["en", "en"])
            self.assertEqual(inference_languages, [("en", "en"), ("en", "en")])

    def test_chinese_target_keeps_english_reference_prompt_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original.wav"
            protected = root / "protected.wav"
            write_silent_wav(original, duration_sec=4.0)
            write_silent_wav(protected, duration_sec=4.0)
            args = SimpleNamespace(
                work_dir=root / "work",
                repo=root / "repo",
                cuda_visible_devices="",
                language="zh-cn",
                original_audio=original,
                protected_audio=protected,
                original_transcript="Ladies and gentlemen, good afternoon.",
                protected_transcript="This protected reference still contains English speech.",
                original_output=root / "original-output.wav",
                protected_output=root / "protected-output.wav",
                max_training_seconds=54.0,
                min_reference_seconds=3.0,
                max_reference_seconds=10.0,
            )
            preparation_languages: list[str] = []
            inference_languages: list[tuple[str, str]] = []

            def fake_prepare(**kwargs: object):
                condition_dir = Path(kwargs["condition_dir"])
                preparation_languages.append(str(kwargs["transcript_language"]))
                return {}, condition_dir / "phoneme.txt", condition_dir / "semantic.tsv"

            def fake_train(**kwargs: object):
                condition_dir = Path(kwargs["condition_dir"])
                return {}, condition_dir / "gpt.ckpt", condition_dir / "sovits.pth"

            def fake_infer(**kwargs: object) -> float:
                inference_languages.append(
                    (str(kwargs["prompt_language"]), str(kwargs["text_language"]))
                )
                return 0.01

            with (
                mock.patch.object(live_finetune, "_prepare", side_effect=fake_prepare),
                mock.patch.object(live_finetune, "_train", side_effect=fake_train),
                mock.patch.object(live_finetune, "_infer", side_effect=fake_infer),
            ):
                result = live_finetune.run(args)

            self.assertEqual(preparation_languages, ["en", "en"])
            self.assertEqual(inference_languages, [("en", "zh"), ("en", "zh")])
            self.assertEqual(result["original"]["promptLanguage"], "en")
            self.assertEqual(result["original"]["textLanguage"], "zh")

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
