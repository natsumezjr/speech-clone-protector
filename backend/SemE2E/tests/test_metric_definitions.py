from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import metric_definitions as metrics
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

    def test_semantic_token_metrics_do_not_use_mfcc_for_token_rates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_path = root / "clean.wav"
            protected_path = root / "protected.wav"
            write_wav(clean_path, np.sin(np.linspace(0, 1, 320)).astype(np.float32))
            write_wav(protected_path, np.sin(np.linspace(0, 1, 320)).astype(np.float32) * 0.9)

            fake_librosa = SimpleNamespace(
                load=lambda path, sr=16000: (np.ones(320, dtype=np.float32), sr),
                feature=SimpleNamespace(mfcc=lambda y, sr, n_mfcc: np.tile(np.linspace(0.0, 1.0, n_mfcc)[:, None], (1, 4))),
            )
            old_env = os.environ.get("SEME2E_ENABLE_MFCC")
            try:
                os.environ["SEME2E_ENABLE_MFCC"] = "1"
                with mock.patch.dict(sys.modules, {"librosa": fake_librosa}):
                    result = metrics.compute_semantic_token_metrics(clean_path, protected_path, {})
            finally:
                if old_env is None:
                    os.environ.pop("SEME2E_ENABLE_MFCC", None)
                else:
                    os.environ["SEME2E_ENABLE_MFCC"] = old_env

        self.assertIsNotNone(result["semanticDrift"])
        self.assertIsNone(result["tokenChangeRate"])
        self.assertIsNone(result["tokenErrorRate"])
        self.assertEqual(result["_metricSources"]["asrEval.semanticDrift"]["source"], "mfcc_proxy")

    def test_clone_eval_scores_against_original_audio_not_protected_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original.wav"
            original_clone = root / "original_clone.wav"
            protected_clone = root / "protected_clone.wav"
            write_wav(original, np.zeros(160, dtype=np.float32))
            write_wav(original_clone, np.zeros(160, dtype=np.float32))
            write_wav(protected_clone, np.zeros(160, dtype=np.float32))

            calls: list[tuple[Path, Path]] = []

            class FakeSpeaker:
                def score(self, reference_audio: Path, candidate_audio: Path) -> float:
                    calls.append((Path(reference_audio), Path(candidate_audio)))
                    return 0.8 if Path(candidate_audio) == original_clone else 0.4

            clone_result = {
                "request": {"model": "fake-tts", "text": "hello"},
                "originalCloneAudio": {"filename": "original_clone.wav"},
                "protectedCloneAudio": {"filename": "protected_clone.wav"},
            }
            result = metrics.compute_clone_eval(original, original_clone, protected_clone, clone_result, speaker_model=FakeSpeaker())

            self.assertEqual(calls, [(original, original_clone), (original, protected_clone)])
            self.assertEqual(result["protectedSimilarity"], 0.4)
            self.assertIsNone(result["cloneTrend"])

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
            with mock.patch.dict(os.environ, {"SEME2E_ENABLE_SPEAKER": "0"}):
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

        self.assertIsNone(frontend["asrEval"])
        self.assertIsNone(frontend["cloneEval"])
        self.assertIsNotNone(frontend["perturbation"]["l2Norm"])
        self.assertIsNotNone(frontend["perturbation"]["l2Rms"])
        self.assertIsNotNone(frontend["perturbation"]["linfNorm"])
        self.assertIsNotNone(frontend["perturbation"]["clippingRate"])
        self.assertIsNone(frontend["protectionQuality"]["pesq"])
        self.assertIsNone(frontend["protectionQuality"]["stoi"])
        self.assertTrue(frontend["metricSources"]["protectionQuality.pesq"]["reason"])
        self.assertTrue(frontend["metricSources"]["protectionQuality.stoi"]["reason"])
        self.assertIsInstance(frontend["charts"]["psychoacoustic"], list)
        self.assertIsInstance(frontend["charts"]["optimizationTrend"], list)


if __name__ == "__main__":
    unittest.main()
