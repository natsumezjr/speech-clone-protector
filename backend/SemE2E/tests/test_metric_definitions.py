from __future__ import annotations

import math
import json
import os
import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import metric_definitions as metrics
import api_server as api
import result_adapter as adapter
from core import utils as core_utils
from api_server import frontend_result
from result_adapter import build_task_payload


def write_wav(path: Path, samples: np.ndarray, sr: int = 16000) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(pcm.tobytes())


class MetricDefinitionsTest(unittest.TestCase):
    def test_semantic_token_metrics_without_tokenizer_returns_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_path = root / "clean.wav"
            protected_path = root / "protected.wav"
            write_wav(clean_path, np.zeros(160, dtype=np.float32))
            write_wav(protected_path, np.zeros(160, dtype=np.float32))

            with mock.patch.dict(os.environ, {"SEME2E_ENABLE_TOKENIZER": "0", "SEME2E_ENABLE_MFCC": "0", "SEME2E_ENABLE_SEMANTIC_ENCODERS": "0"}, clear=False):
                result = metrics.compute_semantic_token_metrics(clean_path, protected_path, {})

        self.assertIsNone(result["tokenChangeRate"])
        self.assertIsNone(result["tokenErrorRate"])
        self.assertEqual(result["_metricSources"]["asrEval.tokenChangeRate"]["reason"], "No real tokenizer configured")
        self.assertEqual(result["_metricSources"]["asrEval.tokenErrorRate"]["reason"], "No real tokenizer configured")

    def test_semantic_token_metrics_with_tokenizer_computes_token_rates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_path = root / "clean.wav"
            protected_path = root / "protected.wav"
            write_wav(clean_path, np.zeros(160, dtype=np.float32))
            write_wav(protected_path, np.zeros(160, dtype=np.float32))

            def fake_encode(path: Path) -> list[int]:
                return [1, 2, 3, 4] if Path(path) == clean_path else [1, 9, 3]

            with mock.patch.dict(os.environ, {"SEME2E_ENABLE_TOKENIZER": "1", "SEME2E_ENABLE_MFCC": "0", "SEME2E_ENABLE_SEMANTIC_ENCODERS": "0"}, clear=False):
                with mock.patch.object(metrics, "encode_s3_tokens", side_effect=fake_encode):
                    result = metrics.compute_semantic_token_metrics(clean_path, protected_path, {})

        self.assertEqual(result["tokenChangeCount"], 1)
        self.assertEqual(result["tokenTotal"], 4)
        self.assertTrue(math.isclose(result["tokenChangeRate"], 1 / 3, rel_tol=1e-6))
        self.assertTrue(math.isclose(result["tokenErrorRate"], 0.5, rel_tol=1e-6))
        self.assertEqual(result["_metricSources"]["asrEval.tokenChangeRate"]["status"], "available")

    def test_perturbation_metrics_from_wavs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean = np.array([0.0, 0.25, -0.25, 0.5], dtype=np.float32)
            protected = np.array([0.1, 0.20, -0.20, 1.0], dtype=np.float32)
            clean_path = root / "clean.wav"
            protected_path = root / "protected.wav"
            write_wav(clean_path, clean)
            write_wav(protected_path, protected)

            x, xp, delta, sr = metrics.align_audio_pair(clean_path, protected_path)
            result = metrics.compute_perturbation_metrics(x, xp, delta, sr, epsilon=0.5, epsilon_norm="linf")

            expected_delta = xp - x
            self.assertTrue(math.isclose(result["l2Norm"], float(np.sqrt(np.sum(expected_delta**2))), rel_tol=1e-5))
            self.assertTrue(math.isclose(result["l2Rms"], float(np.sqrt(np.mean(expected_delta**2))), rel_tol=1e-5))
            self.assertTrue(math.isclose(result["linfNorm"], float(np.max(np.abs(expected_delta))), rel_tol=1e-5))
            self.assertEqual(result["clippingRate"], 0.25)
            self.assertIsNotNone(result["snr"])

    def _fake_psycho_state(self) -> dict[str, object]:
        theta = np.array([[1.0, 3.0, 5.0], [10.0, 14.0, 18.0]], dtype=np.float64)
        psd_delta = np.array([[0.0, 4.0, 8.0], [20.0, 22.0, 24.0]], dtype=np.float64)
        return {
            "freqs": np.array([100.0, 200.0], dtype=np.float64),
            "psdDelta": psd_delta,
            "theta": theta,
            "violation": np.maximum(0.0, psd_delta - theta),
            "sampleRate": 10,
            "hopLength": 5,
            "nFft": 8,
            "center": False,
            "frameCount": 3,
        }

    def test_psychoacoustic_slice_mean_uses_time_mean_curves_and_global_stats(self) -> None:
        with mock.patch.object(metrics, "_psychoacoustic_state", return_value=self._fake_psycho_state()):
            result = metrics.compute_psychoacoustic_slice(np.zeros(10), np.zeros(10), np.zeros(10), 10, mode="mean", duration_sec=1.0)

        expected_violation = np.maximum(0.0, self._fake_psycho_state()["psdDelta"] - self._fake_psycho_state()["theta"])
        self.assertEqual(result["aggregation"], "time_mean")
        self.assertIsNone(result["frameIndex"])
        self.assertEqual(result["maskingThreshold"][0]["thresholdDb"], 3.0)
        self.assertEqual(result["maskingThreshold"][1]["thresholdDb"], 14.0)
        self.assertEqual(result["perturbationSpectrum"][0]["powerDb"], 4.0)
        self.assertEqual(result["perturbationSpectrum"][1]["powerDb"], 22.0)
        self.assertTrue(math.isclose(result["lPsy"], float(np.mean(expected_violation)), rel_tol=1e-9))
        self.assertTrue(math.isclose(result["overMaskRate"], float(np.mean(expected_violation > 0.0)), rel_tol=1e-9))

    def test_psychoacoustic_slice_frame_uses_selected_frame_not_mean(self) -> None:
        with mock.patch.object(metrics, "_psychoacoustic_state", return_value=self._fake_psycho_state()):
            result = metrics.compute_psychoacoustic_slice(np.zeros(10), np.zeros(10), np.zeros(10), 10, mode="frame", time_sec=0.5, duration_sec=1.0)

        self.assertEqual(result["aggregation"], "single_frame")
        self.assertEqual(result["frameIndex"], 1)
        self.assertEqual(result["actualTimeSec"], 0.5)
        self.assertEqual(result["frameCount"], 3)
        self.assertEqual(result["maskingThreshold"][0]["thresholdDb"], 3.0)
        self.assertEqual(result["maskingThreshold"][1]["thresholdDb"], 14.0)
        self.assertEqual(result["perturbationSpectrum"][0]["powerDb"], 4.0)
        self.assertEqual(result["perturbationSpectrum"][1]["powerDb"], 22.0)
        self.assertNotEqual(result["maskingThreshold"][0]["thresholdDb"], 1.0)

    def test_psychoacoustic_slice_frame_zero_and_duration_are_clamped_to_legal_frames(self) -> None:
        with mock.patch.object(metrics, "_psychoacoustic_state", return_value=self._fake_psycho_state()):
            first = metrics.compute_psychoacoustic_slice(np.zeros(10), np.zeros(10), np.zeros(10), 10, mode="frame", time_sec=0.0, duration_sec=1.0)
            last = metrics.compute_psychoacoustic_slice(np.zeros(10), np.zeros(10), np.zeros(10), 10, mode="frame", time_sec=1.0, duration_sec=1.0)

        self.assertEqual(first["frameIndex"], 0)
        self.assertEqual(last["frameIndex"], 2)
        self.assertEqual(last["frameIndex"], last["frameCount"] - 1)

    def test_psychoacoustic_slice_rejects_time_outside_duration(self) -> None:
        with mock.patch.object(metrics, "_psychoacoustic_state", return_value=self._fake_psycho_state()):
            with self.assertRaisesRegex(ValueError, "between 0 and 1.000000"):
                metrics.compute_psychoacoustic_slice(np.zeros(10), np.zeros(10), np.zeros(10), 10, mode="frame", time_sec=-0.01, duration_sec=1.0)
            with self.assertRaisesRegex(ValueError, "between 0 and 1.000000"):
                metrics.compute_psychoacoustic_slice(np.zeros(10), np.zeros(10), np.zeros(10), 10, mode="frame", time_sec=1.01, duration_sec=1.0)

    def test_psychoacoustic_slice_route_returns_400_for_invalid_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp)
            task_dir = task_root / "task_test"
            task_dir.mkdir()
            (task_dir / "result.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(api, "TASK_DIR", task_root):
                with mock.patch.object(api, "create_psychoacoustic_slice", side_effect=ValueError("timeSec must be between 0 and 1.000000 seconds")):
                    response = api.task_psychoacoustic_slice("task_test", mode="frame", timeSec=-0.01)

        payload = json.loads(response.body.decode("utf-8"))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(payload["error"]["code"], "INVALID_PSYCHOACOUSTIC_SLICE_REQUEST")
        self.assertIn("between 0 and 1.000000", payload["error"]["message"])

    def test_asr_edit_rates_and_error_shares_use_separate_denominators(self) -> None:
        result = metrics.compute_asr_metrics(
            "alpha beta gamma delta",
            "alpha theta gamma extra delta",
            language="en",
            model="fake-asr",
        )

        counts = result["editCounts"]
        shares = result["errorShares"]
        self.assertEqual(counts["referenceLength"], 4)
        self.assertEqual(counts["substitutions"], 1)
        self.assertEqual(counts["insertions"], 1)
        self.assertEqual(counts["deletions"], 0)
        self.assertEqual(counts["totalErrors"], 2)
        self.assertTrue(math.isclose(result["substituteRate"] + result["insertRate"] + result["deleteRate"], result["wer"], rel_tol=1e-6))
        self.assertTrue(math.isclose(shares["substituteShare"] + shares["insertShare"] + shares["deleteShare"], 1.0, rel_tol=1e-6))

    def test_quality_dependency_missing_returns_null_and_reason(self) -> None:
        def fake_module_available(name: str) -> bool:
            return False if name in {"pesq", "pystoi"} else True

        with mock.patch.object(metrics, "_module_available", fake_module_available):
            x = np.linspace(-0.2, 0.2, 160, dtype=np.float32)
            xp = x + 0.001
            delta = xp - x
            perturbation = metrics.compute_perturbation_metrics(x, xp, delta, 16000)

            result = metrics.compute_quality_metrics(x, xp, delta, 16000, perturbation)

        self.assertIsNone(result["pesq"])
        self.assertIsNone(result["stoi"])
        self.assertTrue(result["_metricSources"]["protectionQuality.pesq"]["reason"])
        self.assertTrue(result["_metricSources"]["protectionQuality.stoi"]["reason"])

    def test_semantic_drift_requires_semantic_encoder_ensemble(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_path = root / "clean.wav"
            protected_path = root / "protected.wav"
            write_wav(clean_path, np.sin(np.linspace(0, 1, 320)).astype(np.float32))
            write_wav(protected_path, np.sin(np.linspace(0, 1, 320)).astype(np.float32) * 0.9)

            with mock.patch.dict(os.environ, {"SEME2E_ENABLE_TOKENIZER": "0", "SEME2E_ENABLE_SEMANTIC_ENCODERS": "0"}, clear=False):
                result = metrics.compute_semantic_token_metrics(clean_path, protected_path, {})

        self.assertIsNone(result["semanticDrift"])
        self.assertIsNone(result["tokenChangeRate"])
        self.assertIsNone(result["tokenErrorRate"])
        self.assertEqual(result["_metricSources"]["asrEval.semanticDrift"]["source"], "SemanticEncoderEnsemble")

    def test_semantic_drift_uses_weighted_framewise_cosine_across_encoders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_path = root / "clean.wav"
            protected_path = root / "protected.wav"
            write_wav(clean_path, np.zeros(160, dtype=np.float32))
            write_wav(protected_path, np.zeros(160, dtype=np.float32))

            class FakeEnsemble:
                device = "cpu"
                vector_names = ["s3", "hubert", "whisper", "mfcc"]

                def __init__(self) -> None:
                    self.calls = 0

                def get_vectors(self, _wave):
                    self.calls += 1
                    if self.calls == 1:
                        return [
                            np.array([1.0, 0.0]),
                            np.array([1.0, 0.0]),
                            np.array([1.0, 0.0]),
                            np.array([1.0, 0.0]),
                        ]
                    return [
                        np.array([0.0, 1.0]),
                        np.array([1.0, 0.0]),
                        np.array([0.0, 1.0]),
                        np.array([1.0, 0.0]),
                    ]

            import torch

            fake_torchaudio = SimpleNamespace(
                load=lambda _path: (torch.zeros(1, 160), 16000),
                transforms=SimpleNamespace(Resample=lambda orig_freq, new_freq: (lambda wave: wave)),
            )
            fake_ensemble = FakeEnsemble()
            with mock.patch.dict(os.environ, {"SEME2E_ENABLE_TOKENIZER": "0", "SEME2E_ENABLE_MFCC": "0"}, clear=False):
                with mock.patch.dict(sys.modules, {"torchaudio": fake_torchaudio}):
                    with mock.patch.object(metrics, "_load_semantic_encoder_ensemble", return_value=fake_ensemble):
                        result = metrics.compute_semantic_token_metrics(
                            clean_path,
                            protected_path,
                            {
                                "encoders": ["S3", "HuBERT", "Whisper", "MFCC"],
                                "encoderWeights": {"S3": 2.0, "HuBERT": 1.0, "Whisper": 1.0, "MFCC": 1.0},
                            },
                        )

        self.assertEqual([item["encoder"] for item in result["encoderDistances"]], ["S3", "HuBERT", "Whisper", "MFCC"])
        self.assertEqual([item["weight"] for item in result["encoderDistances"]], [2.0, 1.0, 1.0, 1.0])
        self.assertTrue(math.isclose(result["semanticDrift"], 0.6, rel_tol=1e-6))
        self.assertTrue(math.isclose(result["semanticCosineWeightedSum"], 2.0, rel_tol=1e-6))
        self.assertTrue(math.isclose(result["semanticWeightedCosine"], 0.4, rel_tol=1e-6))
        self.assertTrue(math.isclose(result["semanticWeightSum"], 5.0, rel_tol=1e-6))
        self.assertEqual(result["_metricSources"]["asrEval.semanticDrift"]["source"], "SemanticEncoderEnsemble")
        self.assertIn("sum_k(w_k", result["_metricSources"]["asrEval.semanticDrift"]["formula"])

    def test_default_hubert_model_prefers_cached_fallback(self) -> None:
        def fake_cached(model_id: str) -> bool:
            return model_id == "facebook/hubert-large-ls960-ft"

        with mock.patch.object(metrics, "_hf_model_is_cached", side_effect=fake_cached):
            self.assertEqual(metrics._default_hubert_model(), "facebook/hubert-large-ls960-ft")

    def test_semantic_model_resolution_prefers_existing_local_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            local_model = Path(tmp) / "openai-whisper-small"
            local_model.mkdir()

            with mock.patch.object(metrics, "_hf_model_is_cached", return_value=True):
                resolved = metrics._resolve_hf_or_local_model("openai/whisper-small", [str(local_model)])

        self.assertEqual(resolved, str(local_model.resolve()))

    def test_semantic_audio_loader_reads_wav_without_torchcodec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "audio.wav"
            samples = np.sin(np.linspace(0, 1, 320)).astype(np.float32)
            write_wav(wav_path, samples)

            audio, sr = metrics._load_audio_without_torchcodec(wav_path)

        self.assertEqual(sr, 16000)
        self.assertEqual(audio.ndim, 1)
        self.assertGreater(audio.size, 0)

    def test_semantic_encoder_load_error_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_path = root / "clean.wav"
            protected_path = root / "protected.wav"
            write_wav(clean_path, np.zeros(160, dtype=np.float32))
            write_wav(protected_path, np.zeros(160, dtype=np.float32))

            with mock.patch.dict(os.environ, {"SEME2E_ENABLE_TOKENIZER": "0", "SEME2E_ENABLE_SEMANTIC_ENCODERS": "1"}, clear=False):
                with mock.patch.object(metrics, "_load_semantic_encoder_ensemble", return_value=None):
                    with mock.patch.object(metrics, "_SEMANTIC_ENCODER_LAST_ERROR", "OSError: HuBERT model is not in local cache and Hub is unavailable"):
                        result = metrics.compute_semantic_token_metrics(clean_path, protected_path, {})

        source = result["_metricSources"]["asrEval.semanticDrift"]
        self.assertIsNone(result["semanticDrift"])
        self.assertEqual(source["status"], "error")
        self.assertIn("HuBERT model", source["reason"])

    def test_legacy_weight_norm_context_serializes_global_mutation(self) -> None:
        original_weight_norm = object()
        parametrizations = SimpleNamespace(weight_norm=original_weight_norm)
        fake_torch = SimpleNamespace(nn=SimpleNamespace(utils=SimpleNamespace(parametrizations=parametrizations)))
        first_entered = threading.Event()
        second_started = threading.Event()
        second_entered = threading.Event()
        release_first = threading.Event()
        errors: list[BaseException] = []

        def first_worker() -> None:
            try:
                with core_utils.legacy_weight_norm_for_transformers_audio():
                    self.assertFalse(hasattr(parametrizations, "weight_norm"))
                    first_entered.set()
                    release_first.wait(2)
            except BaseException as exc:  # pragma: no cover - surfaced by assertion below
                errors.append(exc)

        def second_worker() -> None:
            second_started.set()
            try:
                with core_utils.legacy_weight_norm_for_transformers_audio():
                    self.assertFalse(hasattr(parametrizations, "weight_norm"))
                    second_entered.set()
            except BaseException as exc:  # pragma: no cover - surfaced by assertion below
                errors.append(exc)

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            first = threading.Thread(target=first_worker)
            second = threading.Thread(target=second_worker)
            first.start()
            self.assertTrue(first_entered.wait(1))
            second.start()
            self.assertTrue(second_started.wait(1))
            try:
                self.assertFalse(second_entered.wait(0.1))
            finally:
                release_first.set()
                first.join(2)
                second.join(2)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(second_entered.is_set())
        self.assertIs(parametrizations.weight_norm, original_weight_norm)

    def test_semantic_encoder_cache_initializes_once_under_concurrency(self) -> None:
        constructor_started = threading.Event()
        second_cache_miss = threading.Event()
        release_constructor = threading.Event()
        constructor_count = 0
        count_lock = threading.Lock()
        results: list[object] = []
        errors: list[BaseException] = []

        class TrackingCache(dict[tuple[str, str, str, str], object]):
            def get(self, key: tuple[str, str, str, str], default: object = None) -> object:
                if threading.current_thread().name == "semantic-second" and key not in self:
                    second_cache_miss.set()
                return super().get(key, default)

        class FakeSemanticEncoderEnsemble:
            def __init__(self, **_: object) -> None:
                nonlocal constructor_count
                with count_lock:
                    constructor_count += 1
                constructor_started.set()
                release_constructor.wait(2)

        fake_module = ModuleType("semantic_encoders")
        fake_module.SemanticEncoderEnsemble = FakeSemanticEncoderEnsemble

        def load_ensemble() -> None:
            try:
                results.append(metrics._load_semantic_encoder_ensemble({}))
            except BaseException as exc:  # pragma: no cover - surfaced by assertion below
                errors.append(exc)

        env = {
            "SEME2E_ENABLE_SEMANTIC_ENCODERS": "1",
            "SEME2E_SEMANTIC_ENCODER_DEVICE": "cpu",
            "SEME2E_SEMANTIC_TOKENIZER_MODEL": "tokenizer",
            "SEME2E_HUBERT_MODEL": "hubert",
            "SEME2E_WHISPER_MODEL": "whisper",
        }
        with (
            mock.patch.dict(os.environ, env, clear=False),
            mock.patch.dict(sys.modules, {"semantic_encoders": fake_module}),
            mock.patch.object(metrics, "_SEMANTIC_ENCODER_CACHE", TrackingCache()),
            mock.patch.object(metrics, "_resolve_local_model_path", side_effect=lambda value: value),
            mock.patch.object(metrics, "_resolve_hf_or_local_model", side_effect=lambda requested, _: requested),
        ):
            first = threading.Thread(target=load_ensemble, name="semantic-first")
            second = threading.Thread(target=load_ensemble, name="semantic-second")
            first.start()
            self.assertTrue(constructor_started.wait(1))
            second.start()
            self.assertTrue(second_cache_miss.wait(1))
            release_constructor.set()
            first.join(2)
            second.join(2)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(constructor_count, 1)
        self.assertEqual(len(results), 2)
        self.assertIs(results[0], results[1])

    def test_selected_semantic_encoders_require_semantic_encoder_ensemble(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_path = root / "clean.wav"
            protected_path = root / "protected.wav"
            write_wav(clean_path, np.zeros(160, dtype=np.float32))
            write_wav(protected_path, np.zeros(160, dtype=np.float32))

            fake_librosa = SimpleNamespace(
                load=lambda path, sr=16000: (np.ones(320, dtype=np.float32), sr),
                feature=SimpleNamespace(mfcc=lambda y, sr, n_mfcc: np.tile(np.linspace(0.0, 1.0, n_mfcc)[:, None], (1, 4))),
            )
            with mock.patch.dict(os.environ, {"SEME2E_ENABLE_TOKENIZER": "0", "SEME2E_ENABLE_MFCC": "1", "SEME2E_ENABLE_SEMANTIC_ENCODERS": "0"}, clear=False):
                with mock.patch.dict(sys.modules, {"librosa": fake_librosa}):
                    result = metrics.compute_semantic_token_metrics(
                        clean_path,
                        protected_path,
                        {"encoders": ["S3", "MFCC"]},
                    )

        self.assertIsNone(result["semanticDrift"])
        self.assertEqual(result["_metricSources"]["asrEval.semanticDrift"]["source"], "SemanticEncoderEnsemble")
        self.assertIn("SEME2E_ENABLE_SEMANTIC_ENCODERS=1", result["_metricSources"]["asrEval.semanticDrift"]["reason"])

    def test_direct_speaker_metrics_without_model_returns_null_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_path = root / "clean.wav"
            protected_path = root / "protected.wav"
            write_wav(clean_path, np.zeros(160, dtype=np.float32))
            write_wav(protected_path, np.zeros(160, dtype=np.float32))

            with mock.patch.dict(os.environ, {"SEME2E_ENABLE_SPEAKER": "0"}, clear=False):
                result = metrics.compute_direct_speaker_metrics(clean_path, protected_path)

        for key in [
            "simBefore",
            "simAfter",
            "simDropRate",
            "embeddingDistanceBefore",
            "embeddingDistanceAfter",
            "simOriginalProtected",
            "embeddingDistance",
        ]:
            self.assertIsNone(result[key])
        self.assertEqual(result["reason"], "Set SEME2E_ENABLE_SPEAKER=1 to run speaker similarity dependencies")

    def test_direct_speaker_metrics_with_model_computes_similarity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_path = root / "clean.wav"
            protected_path = root / "protected.wav"
            write_wav(clean_path, np.zeros(160, dtype=np.float32))
            write_wav(protected_path, np.zeros(160, dtype=np.float32))

            class FakeSpeaker:
                def score(self, reference_audio: Path, candidate_audio: Path) -> float:
                    return 0.7

            result = metrics.compute_direct_speaker_metrics(clean_path, protected_path, speaker_model=FakeSpeaker())

        self.assertEqual(result["simAfter"], 0.7)
        self.assertEqual(result["embeddingDistanceAfter"], 0.30000000000000004)
        self.assertEqual(result["simOriginalProtected"], 0.7)

    def test_clone_eval_scores_against_original_audio_not_protected_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original.wav"
            protected = root / "protected.wav"
            original_clone = root / "original_clone.wav"
            protected_clone = root / "protected_clone.wav"
            write_wav(original, np.zeros(160, dtype=np.float32))
            write_wav(protected, np.zeros(160, dtype=np.float32))
            write_wav(original_clone, np.zeros(160, dtype=np.float32))
            write_wav(protected_clone, np.zeros(160, dtype=np.float32))

            calls: list[tuple[Path, Path]] = []

            class FakeSpeaker:
                def score(self, reference_audio: Path, candidate_audio: Path) -> float:
                    calls.append((Path(reference_audio), Path(candidate_audio)))
                    if Path(candidate_audio) == original_clone:
                        return 0.8
                    if Path(candidate_audio) == protected_clone:
                        return 0.4
                    if Path(candidate_audio) == protected:
                        return 0.6
                    return 0.0

            clone_result = {
                "request": {"model": "fake-tts", "text": "hello"},
                "originalCloneAudio": {"filename": "original_clone.wav"},
                "protectedCloneAudio": {"filename": "protected_clone.wav"},
            }
            result = metrics.compute_clone_eval(original, original_clone, protected_clone, clone_result, protected_audio_path=protected, speaker_model=FakeSpeaker())

            self.assertEqual(calls, [(original, original_clone), (original, protected_clone), (original, protected)])
            self.assertEqual(result["directSimilarity"], 0.6)
            self.assertEqual(result["originalSimilarity"], 0.8)
            self.assertEqual(result["protectedSimilarity"], 0.4)
            self.assertTrue(math.isclose(result["similarityDropRate"], 0.5, rel_tol=1e-6))
            self.assertTrue(math.isclose(result["embeddingDistanceBefore"], 0.2, rel_tol=1e-6))
            self.assertTrue(math.isclose(result["embeddingDistanceAfter"], 0.6, rel_tol=1e-6))
            self.assertTrue(math.isclose(result["embeddingDistanceIncreaseRate"], 2.0, rel_tol=1e-6))
            self.assertIsNone(result["cloneConfidenceBefore"])
            self.assertIsNone(result["cloneConfidenceAfter"])
            self.assertIsNone(result["cloneConfidenceDropRate"])
            self.assertIsNone(result["cloneTrend"])
            self.assertEqual([item["name"] for item in result["cloneRadar"]], ["直接声纹偏移", "相似度下降", "嵌入距离增加", "保护后克隆防护"])
            self.assertTrue(math.isclose(result["cloneRadar"][0]["value"], 40.0, rel_tol=1e-6))
            self.assertEqual(result["cloneRadar"][1]["rawMetricKeys"], ["originalSimilarity", "protectedSimilarity", "similarityDropRate"])
            self.assertEqual(result["cloneRadar"][1]["formula"], "100*clip(similarityDropRate/0.5,0,1)")

    def test_clone_eval_keeps_negative_similarity_and_distance_above_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original.wav"
            original_clone = root / "original_clone.wav"
            protected_clone = root / "protected_clone.wav"
            write_wav(original, np.zeros(160, dtype=np.float32))
            write_wav(original_clone, np.zeros(160, dtype=np.float32))
            write_wav(protected_clone, np.zeros(160, dtype=np.float32))

            class FakeSpeaker:
                def score(self, reference_audio: Path, candidate_audio: Path) -> float:
                    return 0.458 if Path(candidate_audio) == original_clone else -0.035

            result = metrics.compute_clone_eval(
                original,
                original_clone,
                protected_clone,
                {"request": {"model": "fake-tts"}},
                speaker_model=FakeSpeaker(),
            )

        self.assertTrue(math.isclose(result["originalSimilarity"], 0.458, rel_tol=1e-6))
        self.assertTrue(math.isclose(result["protectedSimilarity"], -0.035, rel_tol=1e-6))
        self.assertTrue(math.isclose(result["similarityDropRate"], (0.458 - (-0.035)) / 0.458, rel_tol=1e-6))
        self.assertTrue(math.isclose(result["embeddingDistanceAfter"], 1.035, rel_tol=1e-6))

    def test_clone_eval_metric_sources_use_ecapa_model_without_confidence_calibrator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original.wav"
            original_clone = root / "original_clone.wav"
            protected_clone = root / "protected_clone.wav"
            write_wav(original, np.zeros(160, dtype=np.float32))
            write_wav(original_clone, np.zeros(160, dtype=np.float32))
            write_wav(protected_clone, np.zeros(160, dtype=np.float32))

            class FakeSpeaker:
                def score(self, reference_audio: Path, candidate_audio: Path) -> float:
                    return 0.9 if Path(candidate_audio) == original_clone else 0.3

            source_info = metrics.metric_source(
                "available",
                "speechbrain/spkrec-ecapa-voxceleb",
                formula="cosine(Emb(a),Emb(b))",
                metric="ECAPA-TDNN speaker embedding cosine similarity",
            )
            with mock.patch.object(metrics, "_build_speaker_scorer", return_value=(FakeSpeaker(), "speechbrain/spkrec-ecapa-voxceleb", source_info)):
                result = metrics.compute_clone_eval(
                    original,
                    original_clone,
                    protected_clone,
                    {"request": {"model": "fake-tts"}},
                )

        source = result["_metricSources"]["cloneEval.*"]
        self.assertEqual(source["status"], "available")
        self.assertEqual(source["source"], "speechbrain/spkrec-ecapa-voxceleb")
        self.assertEqual(source["metric"], "ECAPA-TDNN speaker embedding cosine similarity")
        self.assertIsNone(result["cloneConfidenceBefore"])
        self.assertIsNone(result["cloneConfidenceAfter"])
        self.assertIsNone(result["cloneConfidenceDropRate"])
        self.assertEqual(result["_metricSources"]["cloneEval.cloneConfidenceDropRate"]["status"], "unavailable")

    def test_clone_defense_score_uses_soft_mapped_distance_and_direct_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original.wav"
            protected = root / "protected.wav"
            original_clone = root / "original_clone.wav"
            protected_clone = root / "protected_clone.wav"
            for path in [original, protected, original_clone, protected_clone]:
                write_wav(path, np.zeros(160, dtype=np.float32))

            class FakeSpeaker:
                def score(self, reference_audio: Path, candidate_audio: Path) -> float:
                    if Path(candidate_audio) == original_clone:
                        return 0.8
                    if Path(candidate_audio) == protected_clone:
                        return 0.6
                    return 0.5

            result = metrics.compute_clone_eval(
                original,
                original_clone,
                protected_clone,
                {"request": {"model": "fake-tts"}},
                protected_audio_path=protected,
                speaker_model=FakeSpeaker(),
            )

        distance = 1.0 - 0.6
        mapped_distance = distance * (1.26 - 0.30 * distance)
        expected = 100.0 * (0.9 * mapped_distance + 0.1 * ((1.0 - 0.5) / 2.0))
        self.assertTrue(math.isclose(result["cloneDefenseScore"], expected, rel_tol=1e-6))
        self.assertEqual(
            result["_metricSources"]["cloneEval.cloneDefenseScore"]["formula"],
            "Delta_distance_mapped=clamp(Delta_distance*(1.26-0.30*Delta_distance),0,1); Delta_protect=100*(0.9*Delta_distance_mapped+0.1*(Delta_direct/2))",
        )

    def test_clone_distance_soft_mapping_matches_score_targets(self) -> None:
        targets = ((0.50, 49.95), (0.75, 69.8625), (0.90, 80.19))
        for distance, expected_main_score in targets:
            with self.subTest(distance=distance):
                main_score = 100.0 * 0.9 * metrics.calibrate_clone_distance(distance)
                self.assertTrue(math.isclose(main_score, expected_main_score, rel_tol=1e-6))

    def test_create_asr_eval_preserves_shared_semantic_metrics_without_recomputing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original.wav"
            protected = root / "protected.wav"
            write_wav(original, np.zeros(160, dtype=np.float32))
            write_wav(protected, np.zeros(160, dtype=np.float32))
            shared_semantic = {
                "tokenChangeRate": 0.25,
                "tokenErrorRate": 0.5,
                "tokenChangeCount": 1,
                "tokenTotal": 4,
                "semanticDrift": 0.1,
                "encoderDistances": [{"encoder": "S3", "distance": 0.1}],
                "_metricSources": {"asrEval.tokenErrorRate": {"status": "available", "source": "fake_tokenizer"}},
            }
            stored_result = {
                "details": {"semantic": shared_semantic},
                "summary": {
                    "primaryMetrics": {"tokenChangeRate": 0.25, "tokenErrorRate": 0.5, "semanticDrift": 0.1},
                    "metricSources": {"asrEval.tokenErrorRate": {"status": "available", "source": "fake_tokenizer"}},
                },
            }
            fake_asr = {"wer": 0.1, "cer": 0.2, "status": "available", "_metricSources": {"asrEval.*": {"status": "available", "source": "fake_asr"}}}

            with (
                mock.patch.object(adapter, "TASK_DIR", root / "tasks"),
                mock.patch.object(adapter, "_task_audio_paths", return_value=(original, protected, stored_result)),
                mock.patch.object(adapter, "maybe_asr_eval", return_value=fake_asr) as maybe_asr,
                mock.patch.object(adapter, "compute_semantic_token_metrics") as semantic_compute,
                mock.patch.object(adapter, "save_result"),
            ):
                response = adapter.create_asr_eval("task_test", {"model": "fake", "language": "zh-cn", "asrSubId": "asr_test"})

        semantic_compute.assert_not_called()
        self.assertNotIn("tokenChangeRate", response["asr"])
        self.assertNotIn("tokenErrorRate", response["asr"])
        self.assertNotIn("semanticDrift", response["asr"])
        self.assertNotIn("encoderDistances", response["asr"])
        self.assertIs(stored_result["details"]["semantic"], shared_semantic)
        self.assertEqual(stored_result["summary"]["primaryMetrics"]["tokenChangeRate"], 0.25)
        self.assertEqual(stored_result["summary"]["primaryMetrics"]["tokenErrorRate"], 0.5)
        self.assertEqual(stored_result["summary"]["primaryMetrics"]["semanticDrift"], 0.1)
        self.assertEqual(stored_result["summary"]["primaryMetrics"]["wer"], 0.1)
        self.assertEqual(stored_result["summary"]["primaryMetrics"]["cer"], 0.2)
        self.assertEqual(stored_result["summary"]["metricSources"]["asrEval.tokenErrorRate"]["source"], "fake_tokenizer")
        self.assertEqual(stored_result["summary"]["metricSources"]["asrEval.*"]["source"], "fake_asr")
        self.assertEqual(maybe_asr.call_args.args[2]["language"], "zh-cn")
        self.assertEqual(response["request"]["language"], "zh-cn")

    def test_create_clone_voice_writes_clone_eval_to_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_root = root / "tasks"
            task_root.mkdir()
            (task_root / "task_test").mkdir()
            original = root / "original.wav"
            protected = root / "protected.wav"
            write_wav(original, np.zeros(160, dtype=np.float32))
            write_wav(protected, np.zeros(160, dtype=np.float32))
            stored_result = {"details": {}, "summary": {"primaryMetrics": {}, "metricSources": {}}}

            generated_paths: list[Path] = []

            def fake_clone_pair(
                original_reference: Path,
                protected_reference: Path,
                original_output: Path,
                protected_output: Path,
                **kwargs: object,
            ) -> dict[str, object]:
                self.assertEqual(original_reference, original)
                self.assertEqual(protected_reference, protected)
                write_wav(original_output, np.zeros(160, dtype=np.float32))
                write_wav(protected_output, np.zeros(160, dtype=np.float32))
                generated_paths.extend([original_output, protected_output])
                return {"ok": True, "sourceModel": "fake_tts"}

            fake_clone_eval = {
                "originalSimilarity": 0.9,
                "protectedSimilarity": 0.3,
                "similarityDropRate": 2 / 3,
                "embeddingDistanceBefore": 0.1,
                "embeddingDistanceAfter": 0.7,
                "embeddingDistanceIncreaseRate": 6.0,
                "cloneConfidenceBefore": None,
                "cloneConfidenceAfter": None,
                "cloneTrend": None,
                "_metricSources": {"cloneEval.*": {"status": "available", "source": "fake_speaker"}},
            }

            with (
                mock.patch.object(adapter, "TASK_DIR", task_root),
                mock.patch.object(adapter, "_task_audio_paths", return_value=(original, protected, stored_result)),
                mock.patch.object(adapter, "_module_available", return_value=True),
                mock.patch.object(adapter, "_tts_catalog_status", return_value=("available", None, "fake-model")),
                mock.patch.object(adapter, "_coqui_tts_clone_pair", side_effect=fake_clone_pair) as clone_pair,
                mock.patch.object(adapter, "compute_clone_eval", return_value=fake_clone_eval),
            ):
                response = adapter.create_clone_voice("task_test", {"text": "hello", "model": "xtts-v2"})
                clone_pair.assert_called_once()
                self.assertEqual(clone_pair.call_args.kwargs["model"], "xtts_v2")
                self.assertEqual(len(generated_paths), 2)
                self.assertTrue(all(path.is_file() for path in generated_paths))

        self.assertEqual(response["cloneEval"]["originalSimilarity"], 0.9)
        self.assertEqual(response["cloneEval"]["protectedSimilarity"], 0.3)
        self.assertEqual(response["cloneEval"]["similarityDropRate"], 2 / 3)
        self.assertEqual(response["source"], "CoquiTTS:fake_tts")
        self.assertGreater(response["originalCloneAudio"]["sizeBytes"], 0)
        self.assertGreater(response["protectedCloneAudio"]["sizeBytes"], 0)
        self.assertIsNone(response["request"]["annotationSource"])
        self.assertIsNone(response["request"]["speakerPrompt"])
        self.assertIsNone(response["request"]["annotationAsrSubId"])
        self.assertEqual(stored_result["details"]["cloneEval"]["originalSimilarity"], 0.9)
        self.assertEqual(stored_result["summary"]["metricSources"]["cloneEval.*"]["source"], "fake_speaker")

    def test_create_clone_voice_preserves_structured_worker_error_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_root = root / "tasks"
            task_root.mkdir()
            (task_root / "task_test").mkdir()
            original = root / "original.wav"
            protected = root / "protected.wav"
            write_wav(original, np.zeros(160, dtype=np.float32))
            write_wav(protected, np.zeros(160, dtype=np.float32))
            stored_result = {"details": {}, "summary": {"primaryMetrics": {}, "metricSources": {}}}
            worker_diagnostics = {
                "returnCode": 1,
                "response": {
                    "ok": False,
                    "sourceModel": "xtts_v2",
                    "error": {
                        "code": "COQUI_TTS_WORKER_FAILED",
                        "stage": "load_model",
                        "exceptionType": "UnpicklingError",
                        "message": "worker load failed",
                        "traceback": "worker traceback",
                    },
                },
            }
            worker_error = adapter.IsolatedWorkerError(
                "worker load failed",
                diagnostics=worker_diagnostics,
            )

            with (
                mock.patch.object(adapter, "TASK_DIR", task_root),
                mock.patch.object(adapter, "_task_audio_paths", return_value=(original, protected, stored_result)),
                mock.patch.object(adapter, "_module_available", return_value=True),
                mock.patch.object(adapter, "_tts_catalog_status", return_value=("available", None, "fake-model")),
                mock.patch.object(adapter, "_coqui_tts_clone_pair", side_effect=worker_error) as clone_pair,
                mock.patch.object(adapter, "compute_clone_eval") as clone_eval,
            ):
                with self.assertRaises(adapter.CloneBackendUnavailableError) as raised:
                    adapter.create_clone_voice("task_test", {"text": "hello", "model": "xtts-v2"})

            clone_pair.assert_called_once()
            clone_eval.assert_not_called()
            self.assertEqual(raised.exception.reason, "tts_generation_failed")
            self.assertEqual(raised.exception.diagnostics["workerDiagnostics"], worker_diagnostics)
            self.assertEqual(
                raised.exception.diagnostics["workerDiagnostics"]["response"]["error"]["stage"],
                "load_model",
            )
            self.assertEqual(raised.exception.diagnostics["exceptionType"], "IsolatedWorkerError")
            self.assertFalse(any((task_root / "task_test" / "clones").rglob("*.wav")))

    def test_optimization_trace_maps_to_frontend_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_path = root / "clean.wav"
            protected_path = root / "protected.wav"
            samples = np.sin(np.linspace(0, 2 * np.pi, 800, endpoint=False)).astype(np.float32) * 0.2
            write_wav(clean_path, samples)
            write_wav(protected_path, samples + 0.001)
            payload = {"optimization": {"steps": 1, "weightL2": 0.01}, "semantic": {}, "timbre": {}, "psychoacoustic": {}}
            trace_point = {
                "step": 1,
                "Lfeat": 0.1,
                "Lsem": 0.2,
                "Lpsy": 0.3,
                "L2": 0.4,
                "total": 1.0,
                "snr": 20.0,
                "stepElapsedSec": 0.25,
            }
            with mock.patch.dict(os.environ, {"SEME2E_ENABLE_SPEAKER": "0", "SEME2E_ENABLE_TOKENIZER": "0", "SEME2E_ENABLE_MFCC": "0", "SEME2E_ENABLE_SEMANTIC_ENCODERS": "0"}, clear=False):
                result = build_task_payload(
                    "task_test",
                    payload,
                    clean_path,
                    protected_path,
                    None,
                    "2026.6.26 00:00:00",
                    "2026.6.26 00:00:01",
                    {"source": "test_guard", "optimization_trace": [trace_point]},
                )
            frontend = frontend_result(result)

        self.assertEqual(frontend["generation"]["optimizationTrace"][0]["stepElapsedSec"], 0.25)
        self.assertEqual(frontend["generation"]["optimizationTrace"][0]["snr"], 20.0)
        self.assertEqual(frontend["generation"]["optimizationTrace"][0]["Lid"], 0.1)
        self.assertEqual(frontend["lossFinal"]["total"], 1.0)
        self.assertEqual(frontend["lossFinal"]["Lid"], 0.1)
        self.assertEqual(frontend["lossFinal"]["snr"], 20.0)
        self.assertEqual(frontend["averageStepSec"], 0.25)

    def test_build_task_payload_keeps_asr_and_clone_eval_empty_until_called(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_path = root / "clean.wav"
            protected_path = root / "protected.wav"
            samples = np.sin(np.linspace(0, 2 * np.pi, 800, endpoint=False)).astype(np.float32) * 0.2
            write_wav(clean_path, samples)
            write_wav(protected_path, samples + 0.001)

            payload = {
                "mode": "standard",
                "targets": ["semantic", "timbre"],
                "semantic": {"enabled": True, "weightSemantic": 1.0},
                "timbre": {"enabled": True, "weightFeature": 1.0},
                "psychoacoustic": {"enabled": True, "weightPsy": 0.1},
                "optimization": {"epsilon": 0.01, "steps": 1, "weightL2": 0.01},
            }
            shared_semantic = {
                "tokenChangeRate": 0.25,
                "tokenErrorRate": 0.5,
                "semanticDrift": 0.1,
                "encoderDistances": [{"encoder": "S3", "distance": 0.1}],
                "status": "available",
                "_metricSources": {"asrEval.semanticDrift": {"status": "available", "source": "fake_semantic"}},
            }
            with mock.patch.dict(os.environ, {"SEME2E_ENABLE_SPEAKER": "0", "SEME2E_ENABLE_TOKENIZER": "0", "SEME2E_ENABLE_SEMANTIC_ENCODERS": "0"}):
                with mock.patch.object(adapter, "compute_mfcc_semantic", return_value=shared_semantic) as semantic_compute:
                    result = build_task_payload(
                        "task_test",
                        payload,
                        clean_path,
                        protected_path,
                        "file_test",
                        "2026.6.26 00:00:00",
                        "2026.6.26 00:00:01",
                        {"source": "test_guard", "optimizationTrace": []},
                    )
            frontend = frontend_result(result)

        semantic_compute.assert_called_once_with(clean_path, protected_path, payload["semantic"])
        self.assertIs(result["details"]["semantic"], shared_semantic)
        self.assertEqual(result["summary"]["primaryMetrics"]["tokenChangeRate"], 0.25)
        self.assertEqual(result["summary"]["primaryMetrics"]["tokenErrorRate"], 0.5)
        self.assertEqual(result["summary"]["primaryMetrics"]["semanticDrift"], 0.1)
        self.assertIsNone(frontend["asrEval"])
        self.assertIs(frontend["semanticEval"], shared_semantic)
        self.assertIsNone(frontend["cloneEval"])
        self.assertIsNone(frontend["asr"]["semanticDrift"])
        self.assertIsNotNone(frontend["perturbation"]["l2Norm"])
        self.assertIsNotNone(frontend["perturbation"]["l2Rms"])
        self.assertIsNotNone(frontend["perturbation"]["linfNorm"])
        self.assertIsNotNone(frontend["perturbation"]["clippingRate"])
        if frontend["protectionQuality"]["pesq"] is None:
            self.assertTrue(frontend["metricSources"]["protectionQuality.pesq"]["reason"])
        else:
            self.assertEqual(frontend["metricSources"]["protectionQuality.pesq"]["status"], "available")
        if frontend["protectionQuality"]["stoi"] is None:
            self.assertTrue(frontend["metricSources"]["protectionQuality.stoi"]["reason"])
        else:
            self.assertEqual(frontend["metricSources"]["protectionQuality.stoi"]["status"], "available")
        self.assertIsInstance(frontend["charts"]["psychoacoustic"], list)
        self.assertIsInstance(frontend["charts"]["optimizationTrend"], list)


if __name__ == "__main__":
    unittest.main()
