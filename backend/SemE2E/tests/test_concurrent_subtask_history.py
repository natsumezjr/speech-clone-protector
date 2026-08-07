from __future__ import annotations

import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import api_server
import result_adapter


def minimal_result(task_id: str) -> dict[str, object]:
    return {
        "taskId": task_id,
        "status": "completed",
        "mode": "joint",
        "dataMode": "backend",
        "createdAt": "2026.6.27 00:00:00",
        "completedAt": "2026.6.27 00:00:01",
        "elapsedSec": 1,
        "summary": {"score": None, "verdict": "未完成评估", "primaryMetrics": {}, "metricSources": {}},
        "audio": {
            "original": {"filename": "original.wav", "sizeBytes": 1},
            "protected": {"filename": "protected.wav", "sizeBytes": 1},
        },
        "details": {
            "asr": {"status": "unavailable", "wer": None, "cer": None},
            "downstreamTts": {"status": "unavailable"},
            "speaker": {"status": "unavailable"},
            "perception": {"status": "available", "pesq": None},
            "generation": {"lossWeights": {}, "optimizationTrace": []},
        },
        "request": {
            "mode": "joint",
            "targets": ["semantic", "timbre"],
            "semantic": {"enabled": True},
            "timbre": {"enabled": True},
            "psychoacoustic": {"enabled": True},
            "optimization": {},
        },
        "charts": {"psychoacoustic": [], "optimizationTrend": []},
    }


class ConcurrentSubtaskHistoryTest(unittest.TestCase):
    def test_evaluation_batch_persists_expected_items_and_fixed_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp)
            task_id = "task_batch_create"
            task_dir = task_root / task_id
            task_dir.mkdir(parents=True)
            (task_dir / "result.json").write_text(json.dumps(minimal_result(task_id), ensure_ascii=False), encoding="utf-8")

            request = api_server.EvaluationBatchRequest(
                batchId="batch_asr_all",
                type="asr",
                items=[
                    {"batchItemId": "tiny", "model": "openai-whisper:tiny", "language": "en"},
                    {"batchItemId": "base", "model": "openai-whisper:base", "language": "en"},
                ],
            )
            with mock.patch.object(api_server, "TASK_DIR", task_root):
                response = api_server.create_evaluation_batch(task_id, request)
                status = api_server.read_task_status(task_id)

            self.assertEqual(response.status_code, 200)
            batch = status["asrBatches"][0]
            self.assertEqual(batch["taskId"], task_id)
            self.assertEqual(batch["batchId"], "batch_asr_all")
            self.assertEqual(batch["type"], "asr")
            self.assertEqual(batch["label"], "全模型一键测试")
            self.assertEqual(batch["status"], "queued")
            self.assertEqual(batch["progress"], 0.0)
            self.assertEqual(batch["completedCount"], 0)
            self.assertEqual(batch["failedCount"], 0)
            self.assertEqual(batch["totalCount"], 2)
            self.assertTrue(batch["createdAt"])
            self.assertTrue(batch["updatedAt"])
            self.assertEqual([item["batchItemId"] for item in batch["items"]], ["tiny", "base"])
            self.assertTrue(all(item["status"] == "queued" and item["progress"] == 0.0 for item in batch["items"]))

    def test_batch_subtasks_merge_concurrently_and_aggregate_min_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp)
            task_id = "task_batch_parallel"
            task_dir = task_root / task_id
            task_dir.mkdir(parents=True)
            (task_dir / "result.json").write_text(json.dumps(minimal_result(task_id), ensure_ascii=False), encoding="utf-8")

            request = api_server.EvaluationBatchRequest(
                batchId="batch_parallel",
                type="asr",
                items=[
                    {"batchItemId": "one", "model": "model_one"},
                    {"batchItemId": "two", "model": "model_two"},
                    {"batchItemId": "three", "model": "model_three"},
                ],
            )
            with mock.patch.object(api_server, "TASK_DIR", task_root):
                api_server.create_evaluation_batch(task_id, request)

                def start_item(item_id: str, sub_id: str, progress: float) -> None:
                    api_server.write_task_status(
                        task_id,
                        status="running",
                        progress=progress,
                        stage="asr_eval",
                        message=f"running {item_id}",
                        elapsedSec=0.5,
                        asrSubId=sub_id,
                        asrRequest={"model": f"model_{item_id}", "batchId": "batch_parallel", "batchItemId": item_id},
                        asrResult=None,
                    )

                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(start_item, "one", "asr_one", 0.6),
                        executor.submit(start_item, "two", "asr_two", 0.35),
                    ]
                    for future in futures:
                        future.result()

                status_with_missing = api_server.read_task_status(task_id)
                self.assertEqual(status_with_missing["asrBatches"][0]["progress"], 0.0)

                start_item("three", "asr_three", 0.8)
                api_server.write_task_status(
                    task_id,
                    status="completed",
                    progress=0.4,
                    stage="asr_eval",
                    message="one complete",
                    elapsedSec=1.2,
                    asrSubId="asr_one",
                    asrResult={"taskId": task_id, "asrSubId": "asr_one", "status": "available", "asr": {"wer": 0.1}},
                )
                status = api_server.read_task_status(task_id)

            batch = status["asrBatches"][0]
            self.assertEqual(batch["status"], "running")
            self.assertEqual(batch["progress"], 0.35)
            items = {item["batchItemId"]: item for item in batch["items"]}
            self.assertEqual(items["one"]["asrSubId"], "asr_one")
            self.assertEqual(items["two"]["asrSubId"], "asr_two")
            self.assertEqual(items["three"]["asrSubId"], "asr_three")
            self.assertEqual(items["one"]["asrResult"]["asr"]["wer"], 0.1)
            self.assertEqual(items["two"]["asrRequest"]["batchId"], "batch_parallel")

    def test_clone_batch_partial_failed_is_persisted_and_drives_history_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp)
            task_id = "task_batch_partial"
            task_dir = task_root / task_id
            task_dir.mkdir(parents=True)
            (task_dir / "result.json").write_text(json.dumps(minimal_result(task_id), ensure_ascii=False), encoding="utf-8")

            request = api_server.EvaluationBatchRequest(
                batchId="batch_clone_all",
                type="clone",
                items=[
                    {"batchItemId": "cosy", "model": "cosyvoice2:0.5b"},
                    {"batchItemId": "xtts", "model": "xtts-v2"},
                ],
            )
            with mock.patch.object(api_server, "TASK_DIR", task_root):
                api_server.create_evaluation_batch(task_id, request)
                for item_id, sub_id, model in [("cosy", "clone_cosy", "cosyvoice2:0.5b"), ("xtts", "clone_xtts", "xtts-v2")]:
                    api_server.write_task_status(
                        task_id,
                        status="running",
                        progress=0.4,
                        stage="downstream_tts_eval",
                        message=f"running {item_id}",
                        cloneSubId=sub_id,
                        cloneRequest={"text": "hello", "model": model, "batchId": "batch_clone_all", "batchItemId": item_id},
                        cloneResult=None,
                        elapsedSec=0.7,
                    )
                api_server.write_task_status(
                    task_id,
                    status="completed",
                    progress=1,
                    stage="downstream_tts_eval",
                    message="cosy complete",
                    cloneSubId="clone_cosy",
                    cloneResult={"taskId": task_id, "cloneSubId": "clone_cosy", "cloneId": "voice_cosy", "status": "completed"},
                    elapsedSec=2.5,
                )
                api_server.write_task_status(
                    task_id,
                    status="failed",
                    progress=1,
                    stage="downstream_tts_eval",
                    message="xtts failed",
                    cloneSubId="clone_xtts",
                    cloneResult=None,
                    elapsedSec=1.75,
                    error={"code": "CLONE_FAILED", "message": "xtts failed"},
                )
                status = api_server.read_task_status(task_id)
                rows = api_server.list_tasks()

            batch = status["cloneBatches"][0]
            self.assertEqual(batch["status"], "partial_failed")
            self.assertEqual(batch["progress"], 1.0)
            self.assertEqual(batch["completedCount"], 1)
            self.assertEqual(batch["failedCount"], 1)
            self.assertEqual(batch["totalCount"], 2)
            self.assertEqual(batch["elapsedSec"], 2.5)
            self.assertEqual(rows[0]["cloneStatus"], "partial_failed")
            self.assertEqual(rows[0]["cloneProgress"], 1.0)
            self.assertEqual(rows[0]["cloneElapsedSec"], 2.5)
            self.assertEqual(rows[0]["cloneModel"], "全模型一键测试")
            self.assertEqual(rows[0]["cloneStartedAt"], batch["createdAt"])
            self.assertEqual(rows[0]["cloneCompletedAt"], batch["updatedAt"])

    def test_asr_validation_failure_marks_expected_batch_item_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp)
            task_id = "task_batch_validation"
            task_dir = task_root / task_id
            task_dir.mkdir(parents=True)
            (task_dir / "result.json").write_text(json.dumps(minimal_result(task_id), ensure_ascii=False), encoding="utf-8")

            request = api_server.EvaluationBatchRequest(
                batchId="batch_invalid_asr",
                type="asr",
                items=[{"batchItemId": "invalid", "model": "unsupported-asr"}],
            )
            with mock.patch.object(api_server, "TASK_DIR", task_root), mock.patch.object(
                api_server,
                "runtime_config",
                return_value={"models": {"asr": [{"value": "supported-asr"}]}},
            ):
                api_server.create_evaluation_batch(task_id, request)
                response = api_server.run_asr_eval(
                    task_id,
                    api_server.AsrEvalRequest(
                        model="unsupported-asr",
                        language="en",
                        batchId="batch_invalid_asr",
                        batchItemId="invalid",
                    ),
                )
                status = api_server.read_task_status(task_id)
                rows = api_server.list_tasks()

            self.assertEqual(response.status_code, 400)
            batch = status["asrBatches"][0]
            self.assertEqual(batch["status"], "failed")
            self.assertEqual(batch["progress"], 1.0)
            self.assertEqual(batch["completedCount"], 0)
            self.assertEqual(batch["failedCount"], 1)
            self.assertEqual(batch["items"][0]["status"], "failed")
            self.assertEqual(batch["items"][0]["error"]["code"], "UNSUPPORTED_ASR_MODEL")
            self.assertEqual(rows[0]["asrStatus"], "failed")
            self.assertEqual(rows[0]["asrProgress"], 1.0)
            self.assertEqual(rows[0]["asrModel"], "全模型一键测试")

    def test_asr_and_clone_statuses_survive_top_level_stage_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp)
            task_id = "task_parallel"
            task_dir = task_root / task_id
            task_dir.mkdir(parents=True)
            (task_dir / "result.json").write_text(json.dumps(minimal_result(task_id), ensure_ascii=False), encoding="utf-8")

            with mock.patch.object(api_server, "TASK_DIR", task_root):
                api_server.write_task_status(
                    task_id,
                    status="running",
                    progress=0.15,
                    stage="asr_eval",
                    message="ASR running",
                    error=None,
                    asrSubId="asr_1",
                    asrRequest={"model": "openai-whisper:tiny", "language": "en"},
                    asrResult=None,
                    elapsedSec=1.25,
                )
                api_server.write_task_status(
                    task_id,
                    status="running",
                    progress=0.2,
                    stage="downstream_tts_eval",
                    message="Clone running",
                    error=None,
                    cloneSubId="clone_1",
                    cloneResult=None,
                )

                status = api_server.read_task_status(task_id)
                rows = api_server.list_tasks()

            self.assertEqual(status["stage"], "downstream_tts_eval")
            self.assertEqual(status["asrTask"]["status"], "running")
            self.assertEqual(status["asrTask"]["asrRequest"]["model"], "openai-whisper:tiny")
            self.assertEqual(status["asrTask"]["elapsedSec"], 1.25)
            self.assertEqual(status["cloneTask"]["status"], "running")
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["asrStatus"], "running")
            self.assertEqual(row["asrProgress"], 0.15)
            self.assertEqual(row["asrMessage"], "ASR running")
            self.assertEqual(row["asrElapsedSec"], 1.25)
            self.assertEqual(row["cloneStatus"], "running")
            self.assertEqual(row["protectionCompletedAt"], "2026.6.27 00:00:01")
            self.assertTrue(row["hasAsrResult"])
            self.assertTrue(row["hasCloneResult"])

    def test_multiple_asr_and_clone_subtasks_keep_distinct_histories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp)
            task_id = "task_many"
            task_dir = task_root / task_id
            task_dir.mkdir(parents=True)
            (task_dir / "result.json").write_text(json.dumps(minimal_result(task_id), ensure_ascii=False), encoding="utf-8")

            with mock.patch.object(api_server, "TASK_DIR", task_root):
                for sub_id in ["asr_early", "asr_late"]:
                    api_server.write_task_status(task_id, status="running", progress=0.2, stage="asr_eval", message=sub_id, asrSubId=sub_id, asrRequest={"model": f"model_{sub_id}"}, asrResult=None)
                for sub_id in ["clone_early", "clone_late"]:
                    api_server.write_task_status(task_id, status="running", progress=0.2, stage="downstream_tts_eval", message=sub_id, cloneSubId=sub_id, cloneResult=None)
                api_server.write_task_status(
                    task_id,
                    status="completed",
                    progress=1,
                    stage="asr_eval",
                    message="first ASR done",
                    asrSubId="asr_early",
                    asrResult={"taskId": task_id, "asrSubId": "asr_early", "status": "available", "asr": {"originalText": "early"}},
                )
                api_server.write_task_status(
                    task_id,
                    status="completed",
                    progress=1,
                    stage="downstream_tts_eval",
                    message="first clone done",
                    cloneSubId="clone_early",
                    cloneResult={"taskId": task_id, "cloneSubId": "clone_early", "cloneId": "voice_early", "status": "completed"},
                )
                status = api_server.read_task_status(task_id)
                rows = api_server.list_tasks()

            self.assertEqual([item["asrSubId"] for item in status["asrTasks"]], ["asr_early", "asr_late"])
            self.assertEqual([item["cloneSubId"] for item in status["cloneTasks"]], ["clone_early", "clone_late"])
            self.assertEqual(status["asrTasks"][0]["asrResult"]["asr"]["originalText"], "early")
            self.assertEqual(status["asrTasks"][0]["asrRequest"]["model"], "model_asr_early")
            self.assertEqual(status["asrTasks"][1]["asrRequest"]["model"], "model_asr_late")
            self.assertEqual(status["cloneTasks"][0]["cloneResult"]["cloneId"], "voice_early")
            self.assertEqual(status["asrTask"]["asrSubId"], "asr_late")
            self.assertEqual(status["cloneTask"]["cloneSubId"], "clone_late")
            self.assertEqual(rows[0]["asrTaskCount"], 2)
            self.assertEqual(rows[0]["cloneTaskCount"], 2)

    def test_clone_annotation_can_reuse_requested_asr_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp)
            task_id = "task_annotation"
            task_dir = task_root / task_id
            task_dir.mkdir(parents=True)
            result = minimal_result(task_id)
            result["asrResults"] = [
                {
                    "taskId": task_id,
                    "asrSubId": "asr_old",
                    "status": "available",
                    "createdAt": "2026-08-07T01:00:00+00:00",
                    "asr": {"originalText": "人工前的旧转写", "protectedText": "保护后的旧转写", "model": "whisper:tiny"},
                },
                {
                    "taskId": task_id,
                    "asrSubId": "asr_new",
                    "status": "available",
                    "createdAt": "2026-08-07T02:00:00+00:00",
                    "asr": {"originalText": "最近一次 ASR 标注", "protectedText": "最近一次保护音频转写", "model": "whisper:base"},
                },
            ]
            (task_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

            payload = api_server.CloneVoiceRequest(
                text="测试克隆",
                model="cosyvoice2:0.5b",
                language="zh-cn",
                speakerPrompt="",
                annotationSource="asr",
                annotationAsrSubId="asr_old",
            )
            with mock.patch.object(api_server, "TASK_DIR", task_root), mock.patch.object(result_adapter, "TASK_DIR", task_root):
                resolved, error = api_server.resolve_clone_annotation(task_id, payload, "req_test")

            self.assertIsNone(error)
            self.assertIsNotNone(resolved)
            self.assertEqual(resolved["speakerPrompt"], "人工前的旧转写")
            self.assertEqual(resolved["originalSpeakerPrompt"], "人工前的旧转写")
            self.assertEqual(resolved["protectedSpeakerPrompt"], "保护后的旧转写")
            self.assertEqual(resolved["annotationAsrSubId"], "asr_old")
            self.assertEqual(resolved["annotationAsrModel"], "whisper:tiny")

    def test_clone_annotation_rejects_asr_result_without_both_transcripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp)
            task_id = "task_incomplete_annotation"
            task_dir = task_root / task_id
            task_dir.mkdir(parents=True)
            result = minimal_result(task_id)
            result["asrResults"] = [
                {
                    "taskId": task_id,
                    "asrSubId": "asr_incomplete",
                    "status": "available",
                    "createdAt": "2026-08-07T03:00:00+00:00",
                    "asr": {"originalText": "只有原始音频转写", "protectedText": "", "model": "whisper:tiny"},
                }
            ]
            (task_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            payload = api_server.CloneVoiceRequest(
                text="测试克隆",
                model="cosyvoice2:0.5b",
                language="zh-cn",
                annotationSource="asr",
                annotationAsrSubId="asr_incomplete",
            )

            with mock.patch.object(api_server, "TASK_DIR", task_root), mock.patch.object(result_adapter, "TASK_DIR", task_root):
                resolved, error = api_server.resolve_clone_annotation(task_id, payload, "req_test")

            self.assertIsNone(resolved)
            self.assertIsNotNone(error)
            self.assertEqual(error.status_code, 409)

    def test_cosyvoice_pair_passes_each_asr_transcript_to_its_reference(self) -> None:
        completed = mock.Mock(
            returncode=0,
            stdout='VOICE_SHIELD_COSYVOICE_RESULT={"ok": true}\n',
            stderr='',
        )
        with mock.patch.object(result_adapter, "_cosyvoice_model_status", return_value=("available", None, None)), mock.patch.object(result_adapter.subprocess, "run", return_value=completed) as run:
            result = result_adapter._cosyvoice_clone_pair(
                Path("original.wav"),
                Path("protected.wav"),
                Path("original_clone.wav"),
                Path("protected_clone.wav"),
                text="目标文本",
                original_prompt_text="原始音频转写",
                protected_prompt_text="保护音频转写",
                speed=1.0,
                device="cpu",
            )

        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--original-prompt-text") + 1], "原始音频转写")
        self.assertEqual(command[command.index("--protected-prompt-text") + 1], "保护音频转写")
        self.assertTrue(result["ok"])

    def test_task_artifact_names_use_uploaded_filename_not_internal_file_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upload_path = root / "file_ea3cf27b1205_original_name.wav"
            upload_path.write_bytes(b"test")
            task_root = root / "tasks"
            preprocess = {
                "source": {},
                "output": {},
                "status": "normalized",
            }

            def fake_preprocess(source_path: Path, output_path: Path, **_: object) -> dict[str, object]:
                output_path.write_bytes(b"wav")
                return preprocess

            def fake_protection(_input_path: Path, output_path: Path, *_: object, **__: object) -> dict[str, object]:
                output_path.write_bytes(b"protected")
                return {}

            with (
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "preprocess_audio", side_effect=fake_preprocess),
                mock.patch.object(result_adapter, "run_protection", side_effect=fake_protection),
                mock.patch.object(result_adapter, "build_task_payload", return_value={"taskId": "task_name", "status": "completed"}),
            ):
                result_adapter.create_task(
                    upload_path,
                    "file_ea3cf27b1205",
                    {},
                    input_filename="original_name.wav",
                    task_id="task_name",
                )

            self.assertTrue((task_root / "task_name" / "source" / "original_name.wav").exists())
            self.assertTrue((task_root / "task_name" / "original" / "original_name.wav").exists())
            self.assertTrue((task_root / "task_name" / "protected" / "original_name_protected.wav").exists())

    def test_uploaded_filename_recovery_removes_exact_internal_id_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            upload_root = Path(tmp)
            file_id = "file_ea3cf27b1205"
            stored = upload_root / f"{file_id}_2902_9006_000022_000000.wav"
            stored.write_bytes(b"wav")
            with (
                mock.patch.object(api_server, "UPLOAD_DIR", upload_root),
                mock.patch.object(api_server, "FILES", {}),
            ):
                recovered = api_server.find_uploaded_file(file_id)

            self.assertEqual(recovered["filename"], "2902_9006_000022_000000.wav")

    def test_historical_internal_prefixes_are_hidden_from_frontend_filename(self) -> None:
        result = minimal_result("task_legacy_name")
        result["audio"]["original"]["filename"] = "file_ea3cf27b1205_file_9ea577d5c2f0_2902_9006_000022_000000.wav"
        result["audio"]["protected"]["filename"] = "file_ea3cf27b1205_file_9ea577d5c2f0_2902_9006_000022_000000_protected.wav"

        frontend = api_server.frontend_result(result)

        self.assertEqual(frontend["originalAudio"]["filename"], "2902_9006_000022_000000.wav")
        self.assertEqual(frontend["protectedAudio"]["filename"], "2902_9006_000022_000000_protected.wav")


if __name__ == "__main__":
    unittest.main()
