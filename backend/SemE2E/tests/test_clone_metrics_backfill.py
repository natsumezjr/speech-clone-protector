from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backfill_clone_metrics as backfill
import result_adapter as adapter


class CloneMetricsBackfillTest(unittest.TestCase):
    def _write_wav(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        adapter.write_wav_float(path, np.zeros(1600, dtype=np.float32), 16000)

    def _complete_eval(self) -> dict[str, object]:
        return {
            "cloneModel": "gpt-sovits:finetune",
            "status": "available",
            "cloneIdentityStatus": "available",
            "cloneIdentityScore": 82.0,
            "identityBaselineWeight": 1.0,
            "cloneDefenseScore": 82.0,
            "cloneAsrStatus": "available",
            "cloneAsrModel": "openai-whisper:base",
            "cleanCloneTranscription": "clean clone text",
            "protectedCloneTranscription": "protected clone text",
            "cloneSemanticStatus": "available",
            "cloneTokenChangeRate": 0.72,
            "cloneSemanticDrift": 0.61,
            "cloneSemanticScore": 78.0,
            "semanticBaselineWeight": 1.0,
            "cloneQualityStatus": "available",
            "cloneQualityModel": "DNSMOS P.835 OVRL",
            "cleanCloneQualityMos": 3.8,
            "protectedCloneQualityMos": 3.1,
            "cloneQualityScore": 65.0,
            "qualityBaselineWeight": 1.0,
            "createdAt": "2099-01-01T00:00:00+00:00",
            "_metricSources": {
                "cloneEval.cloneIdentityScore": {
                    "status": "available",
                    "source": backfill.V21_IDENTITY_SOURCE,
                },
                "cloneEval.cloneAsr": {
                    "status": "available",
                    "source": "isolated_asr_worker",
                },
                "cloneEval.cloneSemanticScore": {
                    "status": "available",
                    "source": "semantic_tokenizer + semantic_encoder + bounded_text_baseline",
                },
                "cloneEval.cloneQualityScore": {
                    "status": "available",
                    "source": "DNSMOS P.835 OVRL",
                },
            },
        }

    def _fixture(self, root: Path) -> tuple[Path, str, str, dict[str, object]]:
        task_root = root / "tasks"
        task_id = "task_dbcc4af53a80"
        clone_id = "clone_artifact_9660f2d6"
        clone_sub_id = "clone_9660f2d6"
        task_dir = task_root / task_id
        original_path = task_dir / "original" / "original.wav"
        protected_path = task_dir / "protected" / "protected.wav"
        original_clone_path = task_dir / "clones" / clone_id / f"{clone_id}_original_clone.wav"
        protected_clone_path = task_dir / "clones" / clone_id / f"{clone_id}_protected_clone.wav"
        for path in [original_path, protected_path, original_clone_path, protected_clone_path]:
            self._write_wav(path)

        created_at = "2026-08-08T00:58:32+00:00"
        clone_result: dict[str, object] = {
            "taskId": task_id,
            "cloneId": clone_id,
            "cloneSubId": clone_sub_id,
            "status": "completed",
            "createdAt": created_at,
            "request": {
                "model": "gpt-sovits:finetune",
                "text": "target text",
                "language": "en",
                "asrModel": "openai-whisper:base",
            },
            "originalCloneAudio": {
                "filename": original_clone_path.name,
                "audioUrl": f"/api/artifacts/{task_id}/clones/{clone_id}/{original_clone_path.name}",
            },
            "protectedCloneAudio": {
                "filename": protected_clone_path.name,
                "audioUrl": f"/api/artifacts/{task_id}/clones/{clone_id}/{protected_clone_path.name}",
            },
            "cloneEval": {
                "originalSimilarity": 0.76,
                "protectedSimilarity": 0.31,
                "createdAt": created_at,
                "status": "available",
            },
        }
        result = {
            "taskId": task_id,
            "createdAt": created_at,
            "updatedAt": created_at,
            "request": {"semantic": {"enabled": True}},
            "summary": {"primaryMetrics": {}, "metricSources": {}},
            "details": {
                "perception": {},
                "semantic": {},
                "speaker": {},
                "generation": {},
            },
            "audio": {
                "original": {"filename": original_path.name},
                "protected": {"filename": protected_path.name},
            },
            "cloneResults": [clone_result],
        }
        adapter.save_result(task_dir, result)
        (task_dir / "clone_result.json").write_text(
            json.dumps(clone_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        history_dir = task_dir / "clone_results"
        history_dir.mkdir(parents=True)
        (history_dir / f"{clone_sub_id}.json").write_text(
            json.dumps(clone_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        status = {
            "taskId": task_id,
            "cloneSubId": clone_sub_id,
            "cloneResult": clone_result,
            "cloneTask": {"cloneSubId": clone_sub_id, "cloneResult": clone_result},
            "cloneTasks": [{"cloneSubId": clone_sub_id, "cloneResult": clone_result}],
            "cloneBatches": [
                {
                    "batchId": "batch_clone",
                    "items": [{"cloneSubId": clone_sub_id, "cloneResult": clone_result}],
                }
            ],
        }
        (task_dir / "status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return task_root, task_id, clone_sub_id, clone_result

    def test_backfill_reuses_existing_audio_syncs_snapshots_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            task_root, task_id, clone_sub_id, _ = self._fixture(Path(temporary_name))
            task_dir = task_root / task_id
            wav_files_before = sorted(task_dir.rglob("*.wav"))
            complete_eval = self._complete_eval()
            transcription = {
                "status": "available",
                "model": "openai-whisper:base",
                "originalText": "clean clone text",
                "protectedText": "protected clone text",
            }
            semantic = {
                "status": "available",
                "tokenChangeRate": 0.72,
                "semanticDrift": 0.61,
            }
            dnsmos = {
                "status": "available",
                "model": "DNSMOS P.835 OVRL",
                "cleanMos": 3.8,
                "protectedMos": 3.1,
            }

            with (
                mock.patch.object(adapter, "TASK_DIR", task_root),
                mock.patch.object(adapter, "_transcribe_clone_pair_isolated", return_value=transcription) as asr,
                mock.patch.object(adapter, "_compute_clone_semantic_isolated", return_value=semantic) as semantic_eval,
                mock.patch.object(adapter, "_evaluate_dnsmos_pair_isolated", return_value=dnsmos) as quality_eval,
                mock.patch.object(adapter, "compute_clone_eval", return_value=complete_eval) as scorer,
                mock.patch.object(adapter, "create_clone_voice") as create_clone,
                mock.patch.object(adapter, "_coqui_tts_clone_pair") as coqui,
                mock.patch.object(adapter, "_cosyvoice_clone_pair") as cosyvoice,
                mock.patch.object(adapter, "_gpt_sovits_clone_pair") as gpt_sovits,
            ):
                first = backfill.backfill_clone_metrics(task_id, clone_sub_id=clone_sub_id)
                second = backfill.backfill_clone_metrics(task_id, clone_sub_id=clone_sub_id)

            persisted = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))
            persisted_clone = persisted["cloneResults"][0]
            clone_sidecar = json.loads((task_dir / "clone_result.json").read_text(encoding="utf-8"))
            history_sidecar = json.loads((task_dir / "clone_results" / f"{clone_sub_id}.json").read_text(encoding="utf-8"))
            status = json.loads((task_dir / "status.json").read_text(encoding="utf-8"))
            wav_files_after = sorted(task_dir.rglob("*.wav"))
            temporary_files_after = list(task_dir.rglob("*.tmp"))

        self.assertEqual(first["updated"], 1)
        self.assertEqual(first["failed"], 0)
        self.assertEqual(first["items"][0]["missingAfter"], [])
        self.assertEqual(second["skipped"], 1)
        self.assertEqual(persisted_clone["cloneEval"]["cloneSemanticScore"], 78.0)
        self.assertEqual(persisted_clone["cloneSemanticScore"], 78.0)
        self.assertEqual(persisted_clone["createdAt"], "2026-08-08T00:58:32+00:00")
        self.assertEqual(persisted_clone["cloneEval"]["createdAt"], "2026-08-08T00:58:32+00:00")
        self.assertEqual(clone_sidecar["cloneEval"]["cloneQualityScore"], 65.0)
        self.assertEqual(history_sidecar["cloneEval"]["cloneIdentityScore"], 82.0)
        self.assertEqual(status["cloneResult"]["cloneEval"]["cloneSemanticScore"], 78.0)
        self.assertEqual(status["cloneTask"]["cloneResult"]["cloneEval"]["cloneQualityScore"], 65.0)
        self.assertEqual(status["cloneTasks"][0]["cloneResult"]["cloneEval"]["cloneIdentityScore"], 82.0)
        self.assertEqual(status["cloneBatches"][0]["items"][0]["cloneResult"]["cloneEval"]["cloneSemanticScore"], 78.0)
        self.assertEqual(wav_files_after, wav_files_before)
        asr.assert_called_once()
        semantic_eval.assert_called_once()
        quality_eval.assert_called_once()
        scorer.assert_called_once()
        create_clone.assert_not_called()
        coqui.assert_not_called()
        cosyvoice.assert_not_called()
        gpt_sovits.assert_not_called()
        self.assertEqual(temporary_files_after, [])

    def test_cli_dry_run_accepts_clone_id_without_computing_or_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            task_root, task_id, _, clone_result = self._fixture(Path(temporary_name))
            task_dir = task_root / task_id
            clone_id = str(clone_result["cloneId"])
            result_before = (task_dir / "result.json").read_bytes()
            status_before = (task_dir / "status.json").read_bytes()
            output = io.StringIO()
            with (
                mock.patch.object(adapter, "TASK_DIR", task_root),
                mock.patch.object(adapter, "_transcribe_clone_pair_isolated") as asr,
                mock.patch.object(adapter, "_compute_clone_semantic_isolated") as semantic_eval,
                mock.patch.object(adapter, "_evaluate_dnsmos_pair_isolated") as quality_eval,
                mock.patch.object(adapter, "compute_clone_eval") as scorer,
                mock.patch.object(adapter, "create_clone_voice") as create_clone,
                redirect_stdout(output),
            ):
                exit_code = backfill.main(
                    [
                        "--task-id",
                        task_id,
                        "--clone-sub-id",
                        clone_id,
                        "--dry-run",
                    ]
                )

            payload = json.loads(output.getvalue())
            result_after = (task_dir / "result.json").read_bytes()
            status_after = (task_dir / "status.json").read_bytes()

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["wouldUpdate"], 1)
        self.assertEqual(payload["updated"], 0)
        self.assertEqual(result_after, result_before)
        self.assertEqual(status_after, status_before)
        asr.assert_not_called()
        semantic_eval.assert_not_called()
        quality_eval.assert_not_called()
        scorer.assert_not_called()
        create_clone.assert_not_called()


if __name__ == "__main__":
    unittest.main()
