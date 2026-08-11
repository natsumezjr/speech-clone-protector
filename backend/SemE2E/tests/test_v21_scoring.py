from __future__ import annotations

import json
import math
import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dnsmos_quality
import metric_definitions as metrics
import result_adapter as adapter
import semantic_metrics_worker


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


class VoiceShieldV21ScoringTest(unittest.TestCase):
    def test_piecewise_quality_uses_exact_anchors_and_optional_dnsmos(self) -> None:
        without_dnsmos = metrics.compute_protection_quality_score(25.0, 0.9, 3.0)
        with_dnsmos = metrics.compute_protection_quality_score(25.0, 0.9, 3.0, 4.2)

        self.assertEqual(without_dnsmos["snrScore"], 92.0)
        self.assertEqual(without_dnsmos["stoiScore"], 95.0)
        self.assertEqual(without_dnsmos["pesqScore"], 90.0)
        expected_without = (0.40 * 92 + 0.35 * 95 + 0.15 * 90) / 0.90
        self.assertTrue(math.isclose(without_dnsmos["qualityScore"], expected_without, rel_tol=1e-9))
        self.assertIsNone(without_dnsmos["dnsMos"])
        dns_score = 100.0 * (4.2 - 1.0) / 4.0
        expected_with = 0.40 * 92 + 0.35 * 95 + 0.15 * 90 + 0.10 * dns_score
        self.assertTrue(math.isclose(with_dnsmos["qualityScore"], expected_with, rel_tol=1e-9))
        self.assertNotIn("mosScore", with_dnsmos)

    def test_piecewise_tables_match_every_anchor_and_interpolate(self) -> None:
        tables = [
            ([(10.0, 0.0), (15.0, 55.0), (18.5, 75.0), (25.0, 92.0), (30.0, 100.0)], 12.5, 27.5),
            ([(0.60, 0.0), (0.75, 60.0), (0.90, 95.0), (1.00, 100.0)], 0.675, 30.0),
            ([(1.0, 0.0), (1.5, 45.0), (2.0, 75.0), (3.0, 90.0), (4.5, 100.0)], 1.75, 60.0),
        ]
        for anchors, midpoint, expected in tables:
            for x, score in anchors:
                with self.subTest(anchor=x):
                    self.assertEqual(metrics.piecewise_linear_score(x, anchors), score)
            self.assertTrue(math.isclose(metrics.piecewise_linear_score(midpoint, anchors), expected, rel_tol=1e-9))

    def test_phi_maps_the_calibration_scale_to_ninety(self) -> None:
        self.assertTrue(math.isclose(metrics.phi_score(0.9, 0.9), 90.0, rel_tol=1e-9))
        self.assertEqual(metrics.phi_score(0.0, 0.9), 0.0)
        self.assertIsNone(metrics.phi_score(0.5, None))

    def test_clone_quality_adjustment_only_changes_the_quality_dimension(self) -> None:
        score, relevance = metrics.adjust_clone_quality_score(
            0.0,
            identity_baseline_weight=1.0,
            clone_identity_score=99.70,
            clone_semantic_score=90.32,
        )

        self.assertTrue(math.isclose(relevance, 0.0968, rel_tol=1e-9))
        self.assertTrue(math.isclose(score, 90.32, rel_tol=1e-9))

    def test_clone_quality_adjustment_keeps_quality_relevant_for_a_weak_clone_baseline(self) -> None:
        score, relevance = metrics.adjust_clone_quality_score(
            60.0,
            identity_baseline_weight=0.2,
            clone_identity_score=95.0,
            clone_semantic_score=95.0,
        )

        self.assertTrue(math.isclose(relevance, 0.8, rel_tol=1e-9))
        self.assertTrue(math.isclose(score, 68.0, rel_tol=1e-9))

    def test_clone_quality_adjustment_never_raises_a_score_when_evidence_is_missing(self) -> None:
        score, relevance = metrics.adjust_clone_quality_score(
            48.86,
            identity_baseline_weight=1.0,
            clone_identity_score=99.0,
            clone_semantic_score=None,
        )

        self.assertEqual(relevance, 1.0)
        self.assertEqual(score, 48.86)

    def test_real_clone_calibration_places_multiple_quality_scores_above_eighty_five(self) -> None:
        samples = [
            (0.455, 96.54, 88.62, 61.53),
            (0.830, 99.70, 90.32, 0.00),
            (0.564, 52.15, 87.87, 91.61),
            (0.783, 98.85, 90.57, 21.64),
            (0.576, 85.23, 85.23, 48.86),
            (0.569, 99.12, 86.15, 0.00),
        ]
        adjusted_scores = []
        for clean_similarity, identity_score, semantic_score, raw_quality_score in samples:
            identity_weight = metrics._smoothstep_weight(clean_similarity, 0.25, 0.65)
            adjusted, _ = metrics.adjust_clone_quality_score(
                raw_quality_score,
                identity_baseline_weight=identity_weight,
                clone_identity_score=identity_score,
                clone_semantic_score=semantic_score,
            )
            adjusted_scores.append(adjusted)

        self.assertGreaterEqual(sum(score >= 85.0 for score in adjusted_scores if score is not None), 5)
        self.assertLess(adjusted_scores[0], 85.0)

    def test_refresh_result_scores_recomputes_historical_clone_quality_from_existing_metrics(self) -> None:
        result = {
            "summary": {"primaryMetrics": {}, "metricSources": {}},
            "details": {"perception": {}, "semantic": {}, "speaker": {}, "generation": {}},
            "cloneResults": [
                {
                    "request": {"model": "model-a"},
                    "cloneEval": {
                        "cloneModel": "model-a",
                        "originalSimilarity": 0.83,
                        "cloneIdentityScore": 99.70,
                        "identityBaselineWeight": 1.0,
                        "cloneSemanticScore": 90.32,
                        "semanticBaselineWeight": 1.0,
                        "cleanCloneQualityMos": 3.5,
                        "protectedCloneQualityMos": 3.5,
                        "cloneQualityScore": 0.0,
                        "qualityBaselineWeight": 1.0,
                    },
                }
            ],
        }

        adapter.refresh_result_scores(result)
        clone_eval = result["cloneResults"][0]["cloneEval"]
        self.assertEqual(clone_eval["cloneQualityRawScore"], 0.0)
        self.assertTrue(math.isclose(clone_eval["cloneQualityScore"], 90.32, rel_tol=1e-9))
        self.assertTrue(math.isclose(result["cloneResults"][0]["cloneQualityScore"], 90.32, rel_tol=1e-9))

    def test_bounded_text_error_handles_english_words_and_unspaced_chinese(self) -> None:
        english = metrics.compute_bounded_text_metrics("hello brave world", "hello world")
        chinese = metrics.compute_bounded_text_metrics("你好世界", "你好世")

        self.assertTrue(math.isclose(english["wordAccuracy"], 2 / 3, rel_tol=1e-9))
        self.assertEqual(english["referenceWordCount"], 3)
        self.assertTrue(math.isclose(chinese["wordAccuracy"], 0.75, rel_tol=1e-9))
        self.assertEqual(chinese["referenceWordCount"], 4)
        self.assertEqual(chinese["wordUnit"], "latin_word_or_cjk_character")

    def test_clone_eval_emits_four_card_and_three_text_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original.wav"
            clean_clone = root / "clean_clone.wav"
            protected_clone = root / "protected_clone.wav"
            for path in [original, clean_clone, protected_clone]:
                write_wav(path, np.zeros(160, dtype=np.float32))

            class FakeSpeaker:
                def score(self, reference: Path, candidate: Path) -> float:
                    del reference
                    return 0.80 if Path(candidate) == clean_clone else 0.20

            result = metrics.compute_clone_eval(
                original,
                clean_clone,
                protected_clone,
                {"request": {"model": "fake", "text": "hello world"}},
                speaker_model=FakeSpeaker(),
                clone_transcription={
                    "status": "available",
                    "model": "fake-asr",
                    "originalText": "hello world",
                    "protectedText": "goodbye moon",
                },
                semantic_metrics={"tokenChangeRate": 1.0, "semanticDrift": 0.78, "status": "available"},
                quality_metrics={
                    "status": "available",
                    "model": "DNSMOS P.835 OVRL",
                    "cleanMos": 3.5,
                    "protectedMos": 3.0,
                },
            )

        self.assertEqual(result["cleanCloneTranscription"], "hello world")
        self.assertEqual(result["protectedCloneTranscription"], "goodbye moon")
        self.assertTrue(math.isclose(result["cloneIdentityScore"], 96.0, rel_tol=1e-9))
        self.assertTrue(math.isclose(result["cloneSemanticScore"], 90.0, rel_tol=1e-9))
        self.assertIsNotNone(result["cloneQualityScore"])
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["cloneDefenseScore"], result["cloneIdentityScore"])

    def test_overall_requires_all_six_dimensions_and_ignores_asr_score(self) -> None:
        result = {
            "summary": {"primaryMetrics": {}, "metricSources": {}},
            "details": {
                "perception": {"snr": 25.0, "stoi": 0.9, "pesq": 3.0, "dnsMos": 4.0},
                "semantic": {"tokenChangeRate": 0.9, "semanticDrift": 0.6},
                "speaker": {"simOriginalProtected": 0.5},
                "asr": {"asrProtectionScore": 0.0, "wer": 0.0},
                "generation": {},
            },
            "cloneResults": [
                {
                    "request": {"model": "model-a"},
                    "cloneEval": {
                        "cloneModel": "model-a",
                        "cloneIdentityScore": 80.0,
                        "identityBaselineWeight": 0.8,
                        "cloneSemanticScore": 70.0,
                        "semanticBaselineWeight": 0.7,
                        "cloneQualityScore": 60.0,
                        "qualityBaselineWeight": 0.6,
                    },
                }
            ],
        }
        first = metrics.compute_overall_score(result)
        result["details"]["asr"]["asrProtectionScore"] = 100.0
        result["details"]["asr"]["wer"] = 5.0
        second = metrics.compute_overall_score(result)

        self.assertIsNotNone(first["score"])
        self.assertEqual(first["score"], second["score"])
        self.assertEqual(first["protectionEvaluation"]["status"], "complete")
        self.assertEqual(len(first["protectionEvaluation"]["dimensions"]), 6)
        self.assertEqual(
            [item["weight"] for item in first["protectionEvaluation"]["dimensions"]],
            [0.20, 0.10, 0.20, 0.15, 0.15, 0.20],
        )

        result["cloneResults"][0]["cloneEval"]["cloneQualityScore"] = None
        incomplete = metrics.compute_overall_score(result)
        self.assertIsNone(incomplete["score"])
        self.assertEqual(incomplete["verdict"], "待完整评估")
        self.assertIn("cloneQuality", incomplete["protectionEvaluation"]["missingDimensions"])

    def test_historical_direct_distance_is_accepted_without_similarity(self) -> None:
        result = {
            "summary": {"primaryMetrics": {}, "metricSources": {}},
            "details": {
                "perception": {},
                "semantic": {},
                "speaker": {"embeddingDistanceAfter": 0.5},
            },
        }
        evaluation = metrics.compute_overall_score(result)["protectionEvaluation"]
        direct = next(item for item in evaluation["dimensions"] if item["key"] == "directIdentity")
        self.assertTrue(math.isclose(direct["score"], 90.0, rel_tol=1e-9))

    def test_historical_clone_similarity_fields_backfill_identity_score_only(self) -> None:
        result = {
            "summary": {"primaryMetrics": {}, "metricSources": {}},
            "details": {"perception": {}, "semantic": {}, "speaker": {}},
            "cloneResults": [
                {
                    "request": {"model": "legacy-model"},
                    "cloneEval": {"originalSimilarity": 0.8, "protectedSimilarity": 0.2},
                }
            ],
        }
        evaluation = metrics.compute_overall_score(result)["protectionEvaluation"]
        legacy_eval = result["cloneResults"][0]["cloneEval"]
        identity = next(item for item in evaluation["dimensions"] if item["key"] == "cloneIdentity")

        self.assertIsNotNone(identity["score"])
        self.assertIsNotNone(legacy_eval["cloneIdentityScore"])
        self.assertNotIn("cloneSemanticScore", legacy_eval)
        self.assertNotIn("cloneQualityScore", legacy_eval)

    def test_refresh_result_scores_backfills_historical_clone_without_name_error(self) -> None:
        result = {
            "summary": {"primaryMetrics": {}, "metricSources": {}},
            "details": {"perception": {}, "semantic": {}, "speaker": {}},
            "cloneResults": [
                {
                    "request": {"model": "legacy-model"},
                    "cloneEval": {"originalSimilarity": 0.8, "protectedSimilarity": 0.2},
                }
            ],
        }
        adapter.refresh_result_scores(result)
        clone_eval = result["cloneResults"][0]["cloneEval"]

        self.assertEqual(clone_eval["cloneIdentityStatus"], "available")
        self.assertIsNotNone(clone_eval["cloneIdentityScore"])
        self.assertEqual(clone_eval["cloneDefenseScore"], clone_eval["cloneIdentityScore"])
        self.assertIs(result["details"]["cloneEval"], clone_eval)

    def test_refresh_result_scores_preserves_explicit_dnsmos_failure_reason(self) -> None:
        result = {
            "summary": {"primaryMetrics": {}, "metricSources": {}},
            "details": {
                "perception": {
                    "snr": 25.0,
                    "stoi": 0.9,
                    "pesq": 3.0,
                    "dnsMosStatus": "error",
                    "dnsMosReason": "worker failed",
                    "protectionQuality": {
                        "snr": 25.0,
                        "stoi": 0.9,
                        "pesq": 3.0,
                        "dnsMos": None,
                        "dnsMosStatus": "error",
                        "dnsMosReason": "worker failed",
                    },
                },
                "semantic": {},
                "speaker": {},
            },
        }

        adapter.refresh_result_scores(result)

        quality = result["details"]["perception"]["protectionQuality"]
        self.assertIsNone(quality["dnsMos"])
        self.assertEqual(quality["dnsMosStatus"], "error")
        self.assertEqual(quality["dnsMosReason"], "worker failed")

    def test_dnsmos_wrapper_preserves_worker_diagnostics_and_releases_slot(self) -> None:
        diagnostics = {
            "worker": "dnsmos_worker.py",
            "returnCode": 1,
            "stderrTail": "onnx runtime failed",
        }
        worker_slots = mock.Mock()
        with (
            mock.patch.object(
                adapter,
                "dnsmos_model_status",
                return_value={
                    "status": "available",
                    "model": "DNSMOS P.835 OVRL",
                    "modelPath": "fake.onnx",
                },
            ),
            mock.patch.object(adapter, "DNSMOS_WORKER_SLOTS", worker_slots),
            mock.patch.object(adapter, "_acquire_worker_slot") as acquire_slot,
            mock.patch.object(
                adapter,
                "_run_isolated_json_worker",
                side_effect=adapter.IsolatedWorkerError("worker exploded", diagnostics=diagnostics),
            ),
        ):
            result = adapter._evaluate_dnsmos_pair_isolated(Path("clean.wav"), Path("protected.wav"))

        acquire_slot.assert_called_once_with(worker_slots, None)
        worker_slots.release.assert_called_once_with()
        self.assertEqual(result["status"], "error")
        self.assertIn("worker exploded", result["reason"])
        self.assertEqual(result["diagnostics"], diagnostics)

    def test_compute_perception_preserves_dnsmos_when_alignment_fails(self) -> None:
        dns_mos = {
            "status": "available",
            "model": "DNSMOS P.835 OVRL",
            "modelPath": "fake.onnx",
            "provider": "CPUExecutionProvider",
            "protectedMos": 3.6,
            "diagnostics": {"worker": "dnsmos_worker.py"},
        }
        with (
            mock.patch.object(adapter, "_evaluate_dnsmos_pair_isolated", return_value=dns_mos),
            mock.patch.object(adapter, "align_audio_pair", side_effect=RuntimeError("alignment failed")),
        ):
            perception = adapter.compute_perception(Path("clean.wav"), Path("protected.wav"))

        quality = perception["protectionQuality"]
        self.assertEqual(perception["dnsMos"], 3.6)
        self.assertEqual(perception["dnsMosStatus"], "available")
        self.assertIsNone(perception["dnsMosReason"])
        self.assertEqual(perception["dnsMosProvider"], "CPUExecutionProvider")
        self.assertEqual(perception["dnsMosDiagnostics"], {"worker": "dnsmos_worker.py"})
        self.assertEqual(quality["dnsMos"], 3.6)
        self.assertEqual(quality["dnsMosStatus"], "available")
        self.assertEqual(quality["dnsMosDiagnostics"], {"worker": "dnsmos_worker.py"})
        self.assertEqual(
            perception["_metricSources"]["protectionQuality.dnsMos"]["status"],
            "available",
        )
        self.assertEqual(perception["error"], "alignment failed")

    def test_compute_perception_preserves_dnsmos_when_psychoacoustics_fail(self) -> None:
        audio = np.zeros(320, dtype=np.float32)
        dns_mos = {
            "status": "available",
            "model": "DNSMOS P.835 OVRL",
            "protectedMos": 3.6,
        }
        quality = {
            "snr": 25.0,
            "pesq": 3.0,
            "stoi": 0.9,
            "mos": None,
            "mosLqo": None,
            **metrics.compute_protection_quality_score(25.0, 0.9, 3.0, 3.6),
            "_metricSources": {
                "protectionQuality.dnsMos": metrics.metric_source(
                    "available",
                    "DNSMOS P.835 OVRL",
                ),
            },
        }
        with (
            mock.patch.object(adapter, "_evaluate_dnsmos_pair_isolated", return_value=dns_mos),
            mock.patch.object(adapter, "align_audio_pair", return_value=(audio, audio, audio, 16000)),
            mock.patch.object(
                adapter,
                "compute_perturbation_metrics",
                return_value={"snr": 25.0, "l2Norm": 0.0},
            ),
            mock.patch.object(adapter, "compute_quality_metrics", return_value=quality),
            mock.patch.object(
                adapter,
                "compute_psychoacoustic_metrics",
                side_effect=RuntimeError("psychoacoustic failed"),
            ),
        ):
            perception = adapter.compute_perception(Path("clean.wav"), Path("protected.wav"))

        self.assertEqual(perception["dnsMos"], 3.6)
        self.assertEqual(perception["protectionQuality"]["dnsMos"], 3.6)
        self.assertEqual(perception["qualityScore"], quality["qualityScore"])
        self.assertEqual(perception["status"], "partial")
        self.assertEqual(perception["psychoacousticError"], "psychoacoustic failed")
        self.assertEqual(
            perception["_metricSources"]["protectionQuality.dnsMos"]["status"],
            "available",
        )

    def test_ensure_protection_dnsmos_backfills_persists_and_runs_once(self) -> None:
        import api_server

        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "tasks"
            task_id = "task_legacy_dnsmos"
            task_dir = task_root / task_id
            original_dir = task_dir / "original"
            protected_dir = task_dir / "protected"
            original_dir.mkdir(parents=True)
            protected_dir.mkdir(parents=True)
            original = original_dir / "original.wav"
            protected = protected_dir / "protected.wav"
            write_wav(original, np.zeros(320, dtype=np.float32))
            write_wav(protected, np.zeros(320, dtype=np.float32))
            stored = {
                "taskId": task_id,
                "summary": {"primaryMetrics": {}, "metricSources": {}},
                "details": {
                    "perception": {
                        "snr": 25.0,
                        "stoi": 0.9,
                        "pesq": 3.0,
                        "protectionQuality": {"snr": 25.0, "stoi": 0.9, "pesq": 3.0},
                    },
                    "semantic": {},
                    "speaker": {},
                },
                "audio": {
                    "original": {"filename": original.name},
                    "protected": {"filename": protected.name},
                },
            }
            adapter.save_result(task_dir, stored)
            fake_dnsmos = {
                "status": "available",
                "model": "DNSMOS P.835 OVRL",
                "modelPath": "fake.onnx",
                "provider": "CPUExecutionProvider",
                "cleanMos": 3.8,
                "protectedMos": 3.6,
            }

            with (
                mock.patch.object(adapter, "TASK_DIR", task_root),
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(
                    adapter,
                    "_evaluate_dnsmos_pair_isolated",
                    return_value=fake_dnsmos,
                ) as evaluator,
            ):
                response = api_server.task_result(task_id)
                frontend = json.loads(response.body)
                second = adapter.ensure_protection_dnsmos(task_id)

            persisted = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))

        evaluator.assert_called_once_with(original, protected)
        self.assertEqual(frontend["protectionQuality"]["dnsMos"], 3.6)
        self.assertEqual(second["details"]["perception"]["protectionQuality"]["dnsMosStatus"], "available")
        self.assertEqual(persisted["details"]["perception"]["protectionQuality"]["dnsMosModelPath"], "fake.onnx")
        self.assertEqual(persisted["summary"]["metricSources"]["protectionQuality.dnsMos"]["status"], "available")

    def test_ensure_protection_dnsmos_singleflights_failure_and_allows_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "tasks"
            task_id = "task_concurrent_dnsmos"
            task_dir = task_root / task_id
            original_dir = task_dir / "original"
            protected_dir = task_dir / "protected"
            original_dir.mkdir(parents=True)
            protected_dir.mkdir(parents=True)
            original = original_dir / "original.wav"
            protected = protected_dir / "protected.wav"
            write_wav(original, np.zeros(320, dtype=np.float32))
            write_wav(protected, np.zeros(320, dtype=np.float32))
            adapter.save_result(
                task_dir,
                {
                    "taskId": task_id,
                    "summary": {"primaryMetrics": {}, "metricSources": {}},
                    "details": {
                        "perception": {
                            "snr": 25.0,
                            "stoi": 0.9,
                            "pesq": 3.0,
                            "protectionQuality": {"snr": 25.0, "stoi": 0.9, "pesq": 3.0},
                        },
                        "semantic": {},
                        "speaker": {},
                    },
                    "audio": {
                        "original": {"filename": original.name},
                        "protected": {"filename": protected.name},
                    },
                },
            )

            evaluation_started = threading.Event()
            release_evaluation = threading.Event()
            follower_joined = threading.Event()
            call_lock = threading.Lock()
            call_count = 0

            def evaluate(clean_path: Path, protected_path: Path) -> dict[str, object]:
                nonlocal call_count
                self.assertEqual(clean_path, original)
                self.assertEqual(protected_path, protected)
                with call_lock:
                    call_count += 1
                    attempt = call_count
                if attempt == 1:
                    evaluation_started.set()
                    if not release_evaluation.wait(timeout=5):
                        raise AssertionError("timed out waiting to release DNSMOS evaluation")
                    return {
                        "status": "error",
                        "model": "DNSMOS P.835 OVRL",
                        "reason": "worker failed",
                        "diagnostics": {"attempt": 1, "stderrTail": "worker failed"},
                    }
                return {
                    "status": "available",
                    "model": "DNSMOS P.835 OVRL",
                    "modelPath": "fake.onnx",
                    "provider": "CPUExecutionProvider",
                    "cleanMos": 3.8,
                    "protectedMos": 3.6,
                }

            original_join = adapter._join_dnsmos_task_flight

            def tracked_join(join_task_id: str) -> tuple[threading.Event, bool]:
                flight, is_leader = original_join(join_task_id)
                if not is_leader:
                    follower_joined.set()
                return flight, is_leader

            results: list[dict[str, object]] = []
            errors: list[BaseException] = []

            def run_ensure() -> None:
                try:
                    results.append(adapter.ensure_protection_dnsmos(task_id))
                except BaseException as exc:  # pragma: no cover - surfaced by assertions below
                    errors.append(exc)

            with (
                mock.patch.object(adapter, "TASK_DIR", task_root),
                mock.patch.object(adapter, "_evaluate_dnsmos_pair_isolated", side_effect=evaluate),
                mock.patch.object(adapter, "_join_dnsmos_task_flight", side_effect=tracked_join),
            ):
                leader = threading.Thread(target=run_ensure, name="dnsmos-leader")
                follower = threading.Thread(target=run_ensure, name="dnsmos-follower")
                leader.start()
                self.assertTrue(evaluation_started.wait(timeout=5))
                follower.start()
                self.assertTrue(follower_joined.wait(timeout=5))
                release_evaluation.set()
                leader.join(timeout=5)
                follower.join(timeout=5)
                self.assertFalse(leader.is_alive())
                self.assertFalse(follower.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(call_count, 1)
                self.assertEqual(len(results), 2)
                for concurrent_result in results:
                    quality = concurrent_result["details"]["perception"]["protectionQuality"]
                    self.assertIsNone(quality["dnsMos"])
                    self.assertEqual(quality["dnsMosStatus"], "error")
                    self.assertEqual(quality["dnsMosDiagnostics"]["attempt"], 1)

                retried = adapter.ensure_protection_dnsmos(task_id)

            persisted = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))

        self.assertEqual(call_count, 2)
        self.assertEqual(retried["details"]["perception"]["protectionQuality"]["dnsMos"], 3.6)
        self.assertEqual(persisted["details"]["perception"]["dnsMosStatus"], "available")
        self.assertEqual(persisted["details"]["perception"]["dnsMosModelPath"], "fake.onnx")

    def test_per_clone_scores_remain_distinct_from_cross_model_aggregate(self) -> None:
        from api_server import frontend_result

        result = {
            "taskId": "task_two_models",
            "summary": {"primaryMetrics": {}, "metricSources": {}},
            "details": {
                "perception": {"snr": 25.0, "stoi": 0.9, "pesq": 3.0, "dnsMos": 4.0},
                "semantic": {"tokenChangeRate": 0.9, "semanticDrift": 0.6},
                "speaker": {"simOriginalProtected": 0.5},
                "generation": {},
            },
            "audio": {"original": {}, "protected": {}},
            "cloneResults": [
                {
                    "cloneId": "clone_a",
                    "request": {"model": "model-a"},
                    "cloneIdentityScore": None,
                    "cloneSemanticScore": None,
                    "cloneQualityScore": None,
                    "cloneEval": {
                        "cloneModel": "model-a",
                        "cloneIdentityScore": 80.0,
                        "identityBaselineWeight": 1.0,
                        "cloneSemanticScore": 70.0,
                        "semanticBaselineWeight": 1.0,
                        "cloneQualityScore": 60.0,
                        "qualityBaselineWeight": 1.0,
                    },
                },
                {
                    "cloneId": "clone_b",
                    "request": {"model": "model-b"},
                    "cloneIdentityScore": None,
                    "cloneSemanticScore": None,
                    "cloneQualityScore": None,
                    "cloneEval": {
                        "cloneModel": "model-b",
                        "cloneIdentityScore": 20.0,
                        "identityBaselineWeight": 1.0,
                        "cloneSemanticScore": 30.0,
                        "semanticBaselineWeight": 1.0,
                        "cloneQualityScore": 40.0,
                        "qualityBaselineWeight": 1.0,
                    },
                },
            ],
        }

        frontend = frontend_result(result)
        dimensions = {item["key"]: item["score"] for item in frontend["protectionEvaluation"]["dimensions"]}

        self.assertEqual(frontend["score"], frontend["protectionEvaluation"]["overallScore"])
        self.assertEqual(frontend["cloneResults"][0]["cloneIdentityScore"], 80.0)
        self.assertEqual(frontend["cloneResults"][0]["cloneSemanticScore"], 70.0)
        self.assertEqual(frontend["cloneResults"][0]["cloneQualityScore"], 60.0)
        self.assertEqual(frontend["cloneResults"][1]["cloneIdentityScore"], 20.0)
        self.assertEqual(frontend["cloneResults"][1]["cloneSemanticScore"], 30.0)
        self.assertEqual(frontend["cloneResults"][1]["cloneQualityScore"], 40.0)
        self.assertEqual(frontend["cloneEval"]["cloneIdentityScore"], 20.0)
        self.assertEqual(dimensions["cloneIdentity"], 50.0)
        self.assertEqual(dimensions["cloneSemantic"], 50.0)
        self.assertEqual(dimensions["cloneQuality"], 50.0)

    def test_weak_clone_results_receive_zero_weight_and_do_not_fake_aggregate(self) -> None:
        identity = metrics.compute_clone_identity_score(0.25, 0.10)
        semantic = metrics.compute_clone_semantic_score(
            "abc",
            "xyz",
            "uvw",
            1.0,
            0.78,
        )
        quality = metrics.compute_clone_quality_score(2.5, 1.5)

        self.assertEqual(identity["identityBaselineWeight"], 0.0)
        self.assertEqual(semantic["semanticBaselineWeight"], 0.0)
        self.assertEqual(quality["qualityBaselineWeight"], 0.0)
        aggregate, reason = metrics.aggregate_weighted_scores(
            [{"score": 100.0, "weight": 0.0}],
            "score",
            "weight",
            "没有有效结果",
        )
        self.assertIsNone(aggregate)
        self.assertEqual(reason, "没有有效结果")

    def test_semantic_worker_returns_metric_status_without_turning_unavailable_into_worker_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean = root / "clean.wav"
            protected = root / "protected.wav"
            write_wav(clean, np.zeros(160, dtype=np.float32))
            write_wav(protected, np.zeros(160, dtype=np.float32))
            unavailable = {
                "status": "unavailable",
                "tokenChangeRate": None,
                "semanticDrift": None,
                "reason": "模型未启用",
            }
            with mock.patch.object(metrics, "compute_semantic_token_metrics", return_value=unavailable):
                response = semantic_metrics_worker.execute(
                    {"originalPath": str(clean), "protectedPath": str(protected), "config": {}}
                )

        self.assertTrue(response["ok"])
        self.assertEqual(response["metrics"]["status"], "unavailable")
        self.assertEqual(response["metrics"]["reason"], "模型未启用")

    def test_dnsmos_uses_official_whole_second_hop_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "audio.wav"
            # 5.12 s is repeated to 10.24 s; the official formula evaluates one hop.
            write_wav(audio_path, np.zeros(int(5.12 * 16000), dtype=np.float32))

            class Input:
                name = "input_1"

            class FakeSession:
                def __init__(self) -> None:
                    self.calls = 0

                def get_inputs(self) -> list[Input]:
                    return [Input()]

                def run(self, outputs: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
                    del outputs, inputs
                    self.calls += 1
                    return [np.array([[3.8, 3.2, 3.0]], dtype=np.float32)]

            session = FakeSession()
            score = dnsmos_quality._evaluate_audio(session, audio_path)

        self.assertEqual(session.calls, 1)
        self.assertEqual(score["segmentCount"], 1)


if __name__ == "__main__":
    unittest.main()
