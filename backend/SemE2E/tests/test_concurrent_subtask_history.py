from __future__ import annotations

import json
import sys
import tempfile
import threading
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


def automatic_medium_result(task_id: str, *, status: str = "available", error: str | None = None) -> dict[str, object]:
    asr: dict[str, object] = {
        "status": status,
        "model": "openai-whisper:medium",
        "language": "en",
        "originalText": "This is the presentation of our works for the competition.",
        "protectedText": "This is a protected presentation for the competition.",
        "wer": 0.75,
        "cer": 0.65,
    }
    if error is not None:
        asr["error"] = error
    return {
        "taskId": task_id,
        "status": status,
        "asr": asr,
        "request": {"model": "openai-whisper:medium", "language": "auto"},
        "createdAt": "2026-08-13T00:00:00+00:00",
    }


class InlineThread:
    def __init__(self, *, target: object, **_: object) -> None:
        self._target = target
        self._alive = False

    def start(self) -> None:
        self._alive = True
        try:
            self._target()  # type: ignore[operator]
        finally:
            self._alive = False

    def join(self, timeout: float | None = None) -> None:
        del timeout

    def is_alive(self) -> bool:
        return self._alive


class HoldingThread:
    starts = 0
    instances: list["HoldingThread"] = []

    def __init__(self, *, target: object, **_: object) -> None:
        self._target = target
        self._alive = False
        self.join_calls = 0
        type(self).instances.append(self)

    def start(self) -> None:
        type(self).starts += 1
        self._alive = True

    def join(self, timeout: float | None = None) -> None:
        del timeout
        self.join_calls += 1

    def is_alive(self) -> bool:
        return self._alive

    def run_late(self) -> None:
        if not self._alive:
            return
        try:
            self._target()  # type: ignore[operator]
        finally:
            self._alive = False


class ConcurrentSubtaskHistoryTest(unittest.TestCase):
    def test_transcript_display_filename_uses_three_words_ellipsis_and_last_word(self) -> None:
        self.assertEqual(
            api_server._transcript_filename_stem("This is the presentation of our works for the competition."),
            "record_this_is_the……competition",
        )
        self.assertEqual(
            api_server._transcript_filename_stem("语音保护效果十分优秀"),
            "record_语音_保护_效果……优秀",
        )
        self.assertEqual(
            api_server._transcript_filename_stem(
                f"{'a' * 30} {'b' * 30} {'c' * 30} middle {'z' * 30}"
            ),
            f"record_{'a' * 20}_{'b' * 20}_{'c' * 20}……{'z' * 20}",
        )

    def test_frontend_audio_hides_uuid_before_automatic_asr_finishes(self) -> None:
        original = api_server._frontend_audio(
            {"filename": "record_ed8a1de8-cd02-4b93-9550-cf063f089764.wav"},
            "record_audio.wav",
        )
        protected = api_server._frontend_audio(
            {"filename": "record_ed8a1de8-cd02-4b93-9550-cf063f089764_protected.wav"},
            "record_audio_protected.wav",
        )
        self.assertEqual(original["filename"], "record_audio.wav")
        self.assertEqual(protected["filename"], "record_audio_protected.wav")

    def test_details_replace_stored_uuid_names_without_mutating_result(self) -> None:
        result = minimal_result("task_details_name")
        result["audio"]["original"]["filename"] = "record_ed8a1de8-cd02-4b93-9550-cf063f089764.wav"  # type: ignore[index]
        result["audio"]["protected"]["filename"] = "record_ed8a1de8-cd02-4b93-9550-cf063f089764_protected.wav"  # type: ignore[index]
        frontend = api_server._frontend_details_result(result)
        self.assertEqual(frontend["audio"]["original"]["filename"], "record_audio.wav")
        self.assertEqual(frontend["audio"]["protected"]["filename"], "record_audio_protected.wav")
        self.assertEqual(result["audio"]["original"]["filename"], "record_ed8a1de8-cd02-4b93-9550-cf063f089764.wav")  # type: ignore[index]

    def test_display_filename_is_persisted_without_renaming_audio_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp)
            task_id = "task_display_name"
            task_dir = task_root / task_id
            (task_dir / "original").mkdir(parents=True)
            (task_dir / "protected").mkdir(parents=True)
            result = minimal_result(task_id)
            result["audio"]["original"]["filename"] = "record_ed8a1de8-cd02-4b93-9550-cf063f089764.wav"  # type: ignore[index]
            result["audio"]["protected"]["filename"] = "record_ed8a1de8-cd02-4b93-9550-cf063f089764_protected.wav"  # type: ignore[index]
            (task_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
            (task_dir / "status.json").write_text(json.dumps({"taskId": task_id, "status": "completed"}), encoding="utf-8")

            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
                mock.patch.object(api_server, "DELETED_TASK_DIR", task_root / "deleted"),
            ):
                names = api_server._apply_task_display_filenames(
                    task_id,
                    "This is the presentation of our works for the competition.",
                )

            persisted = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(names["filename"], "record_this_is_the……competition.wav")
            self.assertEqual(persisted["displayFilename"], names["filename"])
            self.assertEqual(persisted["audio"]["original"]["filename"], "record_ed8a1de8-cd02-4b93-9550-cf063f089764.wav")
            self.assertEqual(persisted["audio"]["original"]["displayFilename"], names["filename"])

    def test_concurrent_equal_transcripts_receive_unique_persisted_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp)
            for task_id in ("task_one", "task_two"):
                task_dir = task_root / task_id
                task_dir.mkdir(parents=True)
                (task_dir / "result.json").write_text(json.dumps(minimal_result(task_id)), encoding="utf-8")
                (task_dir / "status.json").write_text(json.dumps({"taskId": task_id, "status": "completed"}), encoding="utf-8")
            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
                mock.patch.object(api_server, "DELETED_TASK_DIR", task_root / "deleted"),
            ):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    names = list(executor.map(
                        lambda task_id: api_server._apply_task_display_filenames(task_id, "This is the competition"),
                        ("task_one", "task_two"),
                    ))
            self.assertEqual(
                {item["filename"] for item in names},
                {"record_this_is_the……competition.wav", "record_this_is_the……competition（1）.wav"},
            )

    def test_reapplying_display_name_keeps_persisted_suffix_and_available_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp)
            for task_id in ("task_one", "task_two"):
                task_dir = task_root / task_id
                task_dir.mkdir(parents=True)
                (task_dir / "result.json").write_text(json.dumps(minimal_result(task_id)), encoding="utf-8")
                (task_dir / "status.json").write_text(json.dumps({"taskId": task_id, "status": "completed"}), encoding="utf-8")
            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
                mock.patch.object(api_server, "DELETED_TASK_DIR", task_root / "deleted"),
            ):
                api_server._apply_task_display_filenames("task_one", "This is the competition")
                allocated = api_server._apply_task_display_filenames("task_two", "This is the competition")
                (task_root / "task_one" / "result.json").unlink()
                replayed = api_server._apply_task_display_filenames("task_two", "This is the competition")
                fallback_attempt = api_server._apply_task_display_filenames("task_two", None)

            self.assertEqual(allocated["filename"], "record_this_is_the……competition（1）.wav")
            self.assertEqual(replayed, allocated)
            self.assertEqual(fallback_attempt, allocated)

    def test_automatic_filename_status_does_not_pollute_user_asr_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp)
            task_id = "task_private_asr"
            task_dir = task_root / task_id
            task_dir.mkdir(parents=True)
            initial = {"taskId": task_id, "status": "completed", "stage": "report_generation", "asrTasks": [{"asrSubId": "asr_user"}]}
            (task_dir / "status.json").write_text(json.dumps(initial), encoding="utf-8")
            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(api_server, "DELETED_TASK_DIR", task_root / "deleted"),
            ):
                updated = api_server.write_automatic_asr_status(task_id, status="failed", progress=1, message="OOM")
            self.assertEqual(updated["status"], "completed")
            self.assertEqual(updated["stage"], "report_generation")
            self.assertEqual(updated["asrTasks"], [{"asrSubId": "asr_user"}])
            self.assertEqual(updated["automaticFilenameAsr"]["status"], "failed")

    def test_automatic_filename_asr_forces_only_whisper_medium(self) -> None:
        result = minimal_result("task_medium_only")
        result["request"]["semantic"]["asrModels"] = ["openai-whisper:tiny", "openai-whisper:base"]  # type: ignore[index]
        with (
            mock.patch.object(result_adapter, "_task_audio_paths", return_value=(Path("original.wav"), Path("protected.wav"), result)),
            mock.patch.object(result_adapter, "maybe_asr_eval", return_value={"status": "available"}) as evaluator,
        ):
            result_adapter.create_automatic_filename_asr("task_medium_only")
        semantic = evaluator.call_args.args[2]["semantic"]
        self.assertEqual(semantic["asrModel"], "openai-whisper:medium")
        self.assertEqual(semantic["asrModels"], ["openai-whisper:medium"])

    def test_automatic_medium_success_syncs_normal_asr_history_without_overwriting_protection_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "tasks"
            task_root.mkdir()
            task_id = "task_auto_history_success"
            task_dir = task_root / task_id
            task_dir.mkdir()
            result = minimal_result(task_id)
            result["audio"]["original"]["filename"] = "record_ed8a1de8-cd02-4b93-9550-cf063f089764.wav"  # type: ignore[index]
            result["audio"]["protected"]["filename"] = "record_ed8a1de8-cd02-4b93-9550-cf063f089764_protected.wav"  # type: ignore[index]
            (task_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            protection_status = {
                "taskId": task_id,
                "status": "completed",
                "progress": 1,
                "stage": "report_generation",
                "message": "保护任务已完成",
                "elapsedSec": 12.34,
                "error": None,
            }
            (task_dir / "status.json").write_text(json.dumps(protection_status, ensure_ascii=False), encoding="utf-8")

            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
                mock.patch.object(api_server, "DELETED_TASK_DIR", Path(tmp) / "deleted"),
                mock.patch.object(api_server, "DELETED_TASK_IDS", set()),
                mock.patch.object(api_server, "DELETED_SUBTASK_KEYS", set()),
                mock.patch.object(api_server, "EVALUATION_COLLECTION_DELETIONS", set()),
                mock.patch.object(api_server, "AUTOMATIC_FILENAME_ACTIVE_TASK_IDS", set()),
                mock.patch.object(api_server, "TASK_CANCEL_EVENTS", {}),
                mock.patch.object(api_server, "TASK_THREADS", {}),
                mock.patch.object(api_server, "TASK_PROCESSES", {}),
                mock.patch.object(api_server.threading, "Thread", InlineThread),
                mock.patch.object(
                    api_server,
                    "create_automatic_filename_asr",
                    return_value=automatic_medium_result(task_id),
                ),
            ):
                response = api_server._submit_automatic_filename_asr(
                    task_id,
                    api_server.AsrEvalRequest(model=api_server.AUTO_FILENAME_ASR_MODEL, language="auto"),
                )
                rows = api_server.list_tasks()

            self.assertEqual(response.status_code, 200)
            stored_result = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))
            stored_status = json.loads((task_dir / "status.json").read_text(encoding="utf-8"))
            automatic_sidecar = json.loads((task_dir / "automatic_filename_asr.json").read_text(encoding="utf-8"))
            latest_sidecar = json.loads((task_dir / "asr_result.json").read_text(encoding="utf-8"))
            history_sidecar = json.loads(
                (task_dir / "asr_results" / f"{api_server.AUTO_FILENAME_ASR_SUB_ID}.json").read_text(encoding="utf-8")
            )

            self.assertEqual(len(stored_result["asrResults"]), 1)
            automatic_result = stored_result["asrResults"][0]
            self.assertEqual(automatic_result["asrSubId"], api_server.AUTO_FILENAME_ASR_SUB_ID)
            self.assertEqual(automatic_result["asr"]["model"], api_server.AUTO_FILENAME_ASR_MODEL)
            self.assertTrue(automatic_result["automatic"])
            self.assertTrue(automatic_result["automaticFilename"])
            self.assertEqual([item["asrSubId"] for item in stored_status["asrTasks"]], [api_server.AUTO_FILENAME_ASR_SUB_ID])
            self.assertEqual(stored_status["asrTask"]["asrSubId"], api_server.AUTO_FILENAME_ASR_SUB_ID)
            self.assertEqual(latest_sidecar["asrSubId"], api_server.AUTO_FILENAME_ASR_SUB_ID)
            self.assertEqual(history_sidecar, latest_sidecar)
            self.assertTrue(automatic_sidecar["historySynced"])
            self.assertEqual(automatic_sidecar["historyAsrSubId"], api_server.AUTO_FILENAME_ASR_SUB_ID)
            self.assertEqual(stored_result["displayFilename"], "record_this_is_the……competition.wav")

            for field, expected in protection_status.items():
                self.assertEqual(stored_status[field], expected, field)
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["hasAsrResult"])
            self.assertEqual(rows[0]["asrModel"], api_server.AUTO_FILENAME_ASR_MODEL)
            self.assertEqual(rows[0]["asrTaskCount"], 1)

    def test_completed_automatic_medium_sidecar_replay_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "tasks"
            task_root.mkdir()
            task_id = "task_auto_history_replay"
            task_dir = task_root / task_id
            task_dir.mkdir()
            (task_dir / "result.json").write_text(json.dumps(minimal_result(task_id), ensure_ascii=False), encoding="utf-8")
            (task_dir / "status.json").write_text(
                json.dumps({"taskId": task_id, "status": "completed", "progress": 1, "stage": "report_generation"}),
                encoding="utf-8",
            )
            completed = automatic_medium_result(task_id)
            completed["status"] = "completed"
            (task_dir / "automatic_filename_asr.json").write_text(json.dumps(completed, ensure_ascii=False), encoding="utf-8")

            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
                mock.patch.object(api_server, "DELETED_TASK_DIR", Path(tmp) / "deleted"),
                mock.patch.object(api_server, "DELETED_TASK_IDS", set()),
                mock.patch.object(api_server, "DELETED_SUBTASK_KEYS", set()),
                mock.patch.object(api_server, "EVALUATION_COLLECTION_DELETIONS", set()),
                mock.patch.object(api_server, "AUTOMATIC_FILENAME_ACTIVE_TASK_IDS", set()),
                mock.patch.object(api_server, "create_automatic_filename_asr") as evaluator,
            ):
                first = api_server._submit_automatic_filename_asr(
                    task_id,
                    api_server.AsrEvalRequest(model=api_server.AUTO_FILENAME_ASR_MODEL, language="auto"),
                )
                first_result = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))
                first_status = json.loads((task_dir / "status.json").read_text(encoding="utf-8"))
                first_sidecar = json.loads((task_dir / "automatic_filename_asr.json").read_text(encoding="utf-8"))
                first_latest_history = json.loads((task_dir / "asr_result.json").read_text(encoding="utf-8"))
                first_history = json.loads(
                    (task_dir / "asr_results" / f"{api_server.AUTO_FILENAME_ASR_SUB_ID}.json").read_text(
                        encoding="utf-8"
                    )
                )

                self.assertEqual(json.loads(first.body.decode("utf-8"))["status"], "completed")
                self.assertEqual(len(first_result["asrResults"]), 1)
                self.assertEqual(first_result["asrResults"][0], first_history)
                self.assertEqual(len(first_status["asrTasks"]), 1)
                self.assertEqual(first_status["asrTask"], first_status["asrTasks"][0])
                self.assertTrue(first_sidecar["historySynced"])
                self.assertEqual(first_sidecar["historyAsrSubId"], api_server.AUTO_FILENAME_ASR_SUB_ID)
                self.assertTrue(first_sidecar.get("historySyncedAt"))
                self.assertEqual(first_latest_history, first_history)

                second = api_server._submit_automatic_filename_asr(
                    task_id,
                    api_server.AsrEvalRequest(model=api_server.AUTO_FILENAME_ASR_MODEL, language="auto"),
                )

                second_result = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))
                second_status = json.loads((task_dir / "status.json").read_text(encoding="utf-8"))
                second_sidecar = json.loads((task_dir / "automatic_filename_asr.json").read_text(encoding="utf-8"))
                second_latest_history = json.loads((task_dir / "asr_result.json").read_text(encoding="utf-8"))
                second_history = json.loads(
                    (task_dir / "asr_results" / f"{api_server.AUTO_FILENAME_ASR_SUB_ID}.json").read_text(
                        encoding="utf-8"
                    )
                )

            evaluator.assert_not_called()
            self.assertEqual(json.loads(second.body.decode("utf-8"))["status"], "completed")
            self.assertEqual(first_sidecar["historySyncedAt"], second_sidecar["historySyncedAt"])
            self.assertEqual(first_sidecar, second_sidecar)
            self.assertEqual(len(second_result["asrResults"]), 1)
            self.assertEqual(second_result["asrResults"][0], first_result["asrResults"][0])
            self.assertEqual(len(second_status["asrTasks"]), 1)
            self.assertEqual(second_status["asrTasks"][0], first_status["asrTasks"][0])
            self.assertEqual(second_latest_history, first_latest_history)
            self.assertEqual(second_history, first_history)

    def test_deleted_automatic_medium_history_does_not_revive_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "tasks"
            task_root.mkdir()
            task_id = "task_auto_history_deleted_restart"
            task_dir = task_root / task_id
            task_dir.mkdir()
            (task_dir / "result.json").write_text(json.dumps(minimal_result(task_id), ensure_ascii=False), encoding="utf-8")
            (task_dir / "status.json").write_text(
                json.dumps({"taskId": task_id, "status": "completed", "progress": 1, "stage": "report_generation"}),
                encoding="utf-8",
            )
            completed = automatic_medium_result(task_id)
            completed["status"] = "completed"
            (task_dir / "automatic_filename_asr.json").write_text(json.dumps(completed, ensure_ascii=False), encoding="utf-8")

            deleted_subtasks: set[str] = set()
            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
                mock.patch.object(api_server, "DELETED_TASK_DIR", Path(tmp) / "deleted"),
                mock.patch.object(api_server, "DELETED_SUBTASK_KEYS", deleted_subtasks),
                mock.patch.object(api_server, "EVALUATION_COLLECTION_DELETIONS", set()),
                mock.patch.object(api_server, "AUTOMATIC_FILENAME_ACTIVE_TASK_IDS", set()),
            ):
                api_server.delete_asr_evals(task_id)
                stored_sidecar = json.loads((task_dir / "automatic_filename_asr.json").read_text(encoding="utf-8"))
                self.assertTrue(stored_sidecar["historyDeleted"])
                self.assertFalse(stored_sidecar["historySynced"])

                # In-memory tombstones disappear when the API process restarts;
                # the durable sidecar marker must still prevent resurrection.
                deleted_subtasks.clear()
                response = api_server._submit_automatic_filename_asr(
                    task_id,
                    api_server.AsrEvalRequest(model=api_server.AUTO_FILENAME_ASR_MODEL, language="auto"),
                )

            self.assertEqual(json.loads(response.body.decode("utf-8"))["status"], "deleted")
            stored_result = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))
            stored_status = json.loads((task_dir / "status.json").read_text(encoding="utf-8"))
            self.assertFalse(stored_result.get("asrResults"))
            self.assertEqual(stored_status.get("asrTasks", []), [])
            self.assertFalse((task_dir / "asr_result.json").exists())
            self.assertFalse((task_dir / "asr_results").exists())

    def test_automatic_medium_publish_tombstone_does_not_claim_history_synced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "tasks"
            task_root.mkdir()
            task_id = "task_auto_history_publish_tombstone"
            task_dir = task_root / task_id
            task_dir.mkdir()
            (task_dir / "result.json").write_text(json.dumps(minimal_result(task_id), ensure_ascii=False), encoding="utf-8")
            (task_dir / "status.json").write_text(
                json.dumps({"taskId": task_id, "status": "completed", "progress": 1, "stage": "report_generation"}),
                encoding="utf-8",
            )
            completed = automatic_medium_result(task_id)
            completed["status"] = "completed"
            (task_dir / "automatic_filename_asr.json").write_text(json.dumps(completed, ensure_ascii=False), encoding="utf-8")

            deleted_subtasks: set[str] = set()

            def tombstone_before_publish(_: str, __: str | None) -> dict[str, str]:
                api_server.mark_subtask_deleted(task_id, api_server.AUTO_FILENAME_ASR_SUB_ID)
                return {
                    "filename": "record_this_is_the……competition.wav",
                    "protectedFilename": "record_this_is_the……competition_protected.wav",
                    "status": "available",
                }

            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
                mock.patch.object(api_server, "DELETED_TASK_DIR", Path(tmp) / "deleted"),
                mock.patch.object(api_server, "DELETED_SUBTASK_KEYS", deleted_subtasks),
                mock.patch.object(api_server, "EVALUATION_COLLECTION_DELETIONS", set()),
                mock.patch.object(api_server, "AUTOMATIC_FILENAME_ACTIVE_TASK_IDS", set()),
                mock.patch.object(api_server, "_apply_task_display_filenames", side_effect=tombstone_before_publish),
            ):
                response = api_server._submit_automatic_filename_asr(
                    task_id,
                    api_server.AsrEvalRequest(model=api_server.AUTO_FILENAME_ASR_MODEL, language="auto"),
                )

            self.assertEqual(json.loads(response.body.decode("utf-8"))["status"], "completed")
            stored_sidecar = json.loads((task_dir / "automatic_filename_asr.json").read_text(encoding="utf-8"))
            stored_result = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))
            self.assertFalse(stored_sidecar["historySynced"])
            self.assertFalse(stored_result.get("asrResults"))
            self.assertFalse((task_dir / "asr_result.json").exists())
            self.assertFalse((task_dir / "asr_results").exists())

    def test_failed_automatic_medium_stays_private_and_keeps_fallback_display_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "tasks"
            task_root.mkdir()
            task_id = "task_auto_history_failed"
            task_dir = task_root / task_id
            task_dir.mkdir()
            result = minimal_result(task_id)
            original_stored = "record_ed8a1de8-cd02-4b93-9550-cf063f089764.wav"
            protected_stored = "record_ed8a1de8-cd02-4b93-9550-cf063f089764_protected.wav"
            result["audio"]["original"]["filename"] = original_stored  # type: ignore[index]
            result["audio"]["protected"]["filename"] = protected_stored  # type: ignore[index]
            (task_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            protection_status = {
                "taskId": task_id,
                "status": "completed",
                "progress": 1,
                "stage": "report_generation",
                "message": "保护任务已完成",
            }
            (task_dir / "status.json").write_text(json.dumps(protection_status, ensure_ascii=False), encoding="utf-8")

            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
                mock.patch.object(api_server, "DELETED_TASK_DIR", Path(tmp) / "deleted"),
                mock.patch.object(api_server, "DELETED_TASK_IDS", set()),
                mock.patch.object(api_server, "DELETED_SUBTASK_KEYS", set()),
                mock.patch.object(api_server, "EVALUATION_COLLECTION_DELETIONS", set()),
                mock.patch.object(api_server, "AUTOMATIC_FILENAME_ACTIVE_TASK_IDS", set()),
                mock.patch.object(api_server, "TASK_CANCEL_EVENTS", {}),
                mock.patch.object(api_server, "TASK_THREADS", {}),
                mock.patch.object(api_server, "TASK_PROCESSES", {}),
                mock.patch.object(api_server.threading, "Thread", InlineThread),
                mock.patch.object(
                    api_server,
                    "create_automatic_filename_asr",
                    return_value=automatic_medium_result(task_id, status="unavailable", error="OOM"),
                ),
            ):
                api_server._submit_automatic_filename_asr(
                    task_id,
                    api_server.AsrEvalRequest(model=api_server.AUTO_FILENAME_ASR_MODEL, language="auto"),
                )

            stored_result = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))
            stored_status = json.loads((task_dir / "status.json").read_text(encoding="utf-8"))
            automatic_sidecar = json.loads((task_dir / "automatic_filename_asr.json").read_text(encoding="utf-8"))
            self.assertFalse(stored_result.get("asrResults"))
            self.assertEqual(stored_status.get("asrTasks", []), [])
            self.assertNotIn("asrTask", stored_status)
            self.assertFalse((task_dir / "asr_result.json").exists())
            self.assertFalse((task_dir / "asr_results").exists())
            self.assertEqual(stored_status["automaticFilenameAsr"]["status"], "failed")
            self.assertEqual(stored_status["automaticFilenameAsr"]["displayFilename"], "record_audio.wav")
            self.assertEqual(automatic_sidecar["status"], "failed")
            self.assertEqual(automatic_sidecar["error"]["code"], "AUTOMATIC_FILENAME_ASR_FAILED")
            self.assertEqual(stored_result["displayFilename"], "record_audio.wav")
            self.assertEqual(stored_result["displayProtectedFilename"], "record_audio_protected.wav")
            self.assertEqual(stored_result["audio"]["original"]["filename"], original_stored)
            self.assertEqual(stored_result["audio"]["protected"]["filename"], protected_stored)
            for field, expected in protection_status.items():
                self.assertEqual(stored_status[field], expected, field)

    def test_deleting_asr_history_cancels_queued_automatic_medium_and_blocks_late_publish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "tasks"
            task_root.mkdir()
            task_id = "task_auto_history_delete_race"
            task_dir = task_root / task_id
            original_dir = task_dir / "original"
            protected_dir = task_dir / "protected"
            original_dir.mkdir(parents=True)
            protected_dir.mkdir()
            (original_dir / "original.wav").write_bytes(b"original")
            (protected_dir / "protected.wav").write_bytes(b"protected")

            result = minimal_result(task_id)
            result.update({
                "progress": 1,
                "stage": "report_generation",
                "message": "保护任务已完成",
                "error": None,
                "displayFilename": "record_saved_name.wav",
                "displayProtectedFilename": "record_saved_name_protected.wav",
            })
            result["audio"]["original"].update({  # type: ignore[index]
                "filename": "original.wav",
                "displayFilename": "record_saved_name.wav",
            })
            result["audio"]["protected"].update({  # type: ignore[index]
                "filename": "protected.wav",
                "displayFilename": "record_saved_name_protected.wav",
            })
            (task_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            protection_status = {
                "taskId": task_id,
                "status": "completed",
                "progress": 1,
                "stage": "report_generation",
                "message": "保护任务已完成",
                "elapsedSec": 1,
                "error": None,
                "displayFilename": "record_saved_name.wav",
                "displayProtectedFilename": "record_saved_name_protected.wav",
            }
            (task_dir / "status.json").write_text(json.dumps(protection_status, ensure_ascii=False), encoding="utf-8")
            naming_sidecar = {
                "taskId": task_id,
                "status": "queued",
                "displayFilename": "record_saved_name.wav",
                "displayProtectedFilename": "record_saved_name_protected.wav",
                "createdAt": "2026-08-13T00:00:00+00:00",
            }
            naming_sidecar_path = task_dir / "automatic_filename_asr.json"
            naming_sidecar_path.write_text(json.dumps(naming_sidecar, ensure_ascii=False), encoding="utf-8")

            runtime_id = f"{task_id}:{api_server.AUTO_FILENAME_ASR_SUB_ID}"
            HoldingThread.starts = 0
            HoldingThread.instances.clear()
            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
                mock.patch.object(api_server, "DELETED_TASK_DIR", Path(tmp) / "deleted"),
                mock.patch.object(api_server, "DELETED_TASK_IDS", set()),
                mock.patch.object(api_server, "DELETED_SUBTASK_KEYS", set()),
                mock.patch.object(api_server, "EVALUATION_COLLECTION_DELETIONS", set()),
                mock.patch.object(api_server, "AUTOMATIC_FILENAME_ACTIVE_TASK_IDS", set()),
                mock.patch.object(api_server, "TASK_CANCEL_EVENTS", {}),
                mock.patch.object(api_server, "TASK_THREADS", {}),
                mock.patch.object(api_server, "TASK_PROCESSES", {}),
                mock.patch.object(api_server.threading, "Thread", HoldingThread),
                mock.patch.object(api_server, "create_automatic_filename_asr") as evaluator,
            ):
                response = api_server._submit_automatic_filename_asr(
                    task_id,
                    api_server.AsrEvalRequest(model=api_server.AUTO_FILENAME_ASR_MODEL, language="auto"),
                )
                self.assertEqual(json.loads(response.body.decode("utf-8"))["status"], "queued")
                queued_status = api_server.read_task_status(task_id)
                self.assertEqual(
                    [item["asrSubId"] for item in queued_status["asrTasks"]],
                    [api_server.AUTO_FILENAME_ASR_SUB_ID],
                )
                self.assertIn(runtime_id, api_server.TASK_CANCEL_EVENTS)
                self.assertIn(runtime_id, api_server.TASK_THREADS)
                cancel_event = api_server.TASK_CANCEL_EVENTS[runtime_id]
                worker = api_server.TASK_THREADS[runtime_id]

                deleted = api_server.delete_asr_evals(task_id)

                self.assertTrue(cancel_event.is_set())
                self.assertEqual(deleted["deletedSubtaskIds"], [api_server.AUTO_FILENAME_ASR_SUB_ID])
                self.assertEqual(deleted["cancelledCount"], 1)
                self.assertNotIn(runtime_id, api_server.TASK_CANCEL_EVENTS)
                self.assertNotIn(runtime_id, api_server.TASK_THREADS)
                self.assertEqual(worker.join_calls, 1)

                worker.run_late()
                evaluator.assert_not_called()
                self.assertNotIn(task_id, api_server.AUTOMATIC_FILENAME_ACTIVE_TASK_IDS)

            stored_result = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))
            stored_status = json.loads((task_dir / "status.json").read_text(encoding="utf-8"))
            stored_naming_sidecar = json.loads(naming_sidecar_path.read_text(encoding="utf-8"))
            self.assertTrue(task_dir.exists())
            self.assertTrue((original_dir / "original.wav").exists())
            self.assertTrue((protected_dir / "protected.wav").exists())
            self.assertFalse(stored_result.get("asrResults"))
            self.assertEqual(stored_status.get("asrTasks", []), [])
            self.assertNotIn("asrTask", stored_status)
            self.assertFalse((task_dir / "asr_result.json").exists())
            self.assertFalse((task_dir / "asr_results").exists())
            self.assertEqual(
                {key: stored_naming_sidecar.get(key) for key in naming_sidecar},
                naming_sidecar,
            )
            self.assertTrue(stored_naming_sidecar["historyDeleted"])
            self.assertFalse(stored_naming_sidecar["historySynced"])
            self.assertEqual(stored_result["displayFilename"], "record_saved_name.wav")
            self.assertEqual(stored_result["displayProtectedFilename"], "record_saved_name_protected.wav")
            self.assertEqual(stored_status["displayFilename"], "record_saved_name.wav")
            self.assertEqual(stored_status["displayProtectedFilename"], "record_saved_name_protected.wav")
            for field in ("status", "progress", "stage", "message", "elapsedSec", "error"):
                self.assertEqual(stored_status[field], protection_status[field], field)

    def test_deleting_asr_history_while_automatic_medium_is_running_blocks_real_thread_late_publish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "tasks"
            task_root.mkdir()
            task_id = "task_auto_history_real_thread_delete_race"
            task_dir = task_root / task_id
            original_dir = task_dir / "original"
            protected_dir = task_dir / "protected"
            original_dir.mkdir(parents=True)
            protected_dir.mkdir()
            (original_dir / "original.wav").write_bytes(b"original")
            (protected_dir / "protected.wav").write_bytes(b"protected")

            result = minimal_result(task_id)
            result.update({
                "progress": 1,
                "stage": "report_generation",
                "message": "保护任务已完成",
                "error": None,
                "displayFilename": "record_saved_name.wav",
                "displayProtectedFilename": "record_saved_name_protected.wav",
                "displayFilenameStatus": "available",
            })
            result["audio"]["original"].update({  # type: ignore[index]
                "filename": "original.wav",
                "displayFilename": "record_saved_name.wav",
            })
            result["audio"]["protected"].update({  # type: ignore[index]
                "filename": "protected.wav",
                "displayFilename": "record_saved_name_protected.wav",
            })
            (task_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            protection_status = {
                "taskId": task_id,
                "status": "completed",
                "progress": 1,
                "stage": "report_generation",
                "message": "保护任务已完成",
                "elapsedSec": 1,
                "error": None,
                "displayFilename": "record_saved_name.wav",
                "displayProtectedFilename": "record_saved_name_protected.wav",
            }
            (task_dir / "status.json").write_text(json.dumps(protection_status, ensure_ascii=False), encoding="utf-8")

            evaluator_entered = threading.Event()
            release_evaluator = threading.Event()
            evaluator_calls: list[str] = []
            deletion_results: list[dict[str, object]] = []
            deletion_errors: list[BaseException] = []
            runtime_id = f"{task_id}:{api_server.AUTO_FILENAME_ASR_SUB_ID}"
            worker: threading.Thread | None = None
            deletion_thread: threading.Thread | None = None

            def blocking_evaluator(received_task_id: str, **_: object) -> dict[str, object]:
                evaluator_calls.append(received_task_id)
                evaluator_entered.set()
                if not release_evaluator.wait(timeout=10):
                    raise TimeoutError("test did not release automatic Medium evaluator")
                return automatic_medium_result(received_task_id)

            def delete_history() -> None:
                try:
                    deletion_results.append(api_server.delete_asr_evals(task_id))
                except BaseException as exc:
                    deletion_errors.append(exc)

            try:
                with (
                    mock.patch.object(api_server, "TASK_DIR", task_root),
                    mock.patch.object(result_adapter, "TASK_DIR", task_root),
                    mock.patch.object(api_server, "DELETED_TASK_DIR", Path(tmp) / "deleted"),
                    mock.patch.object(api_server, "DELETED_TASK_IDS", set()),
                    mock.patch.object(api_server, "DELETED_SUBTASK_KEYS", set()),
                    mock.patch.object(api_server, "EVALUATION_COLLECTION_DELETIONS", set()),
                    mock.patch.object(api_server, "AUTOMATIC_FILENAME_ACTIVE_TASK_IDS", set()),
                    mock.patch.object(api_server, "TASK_CANCEL_EVENTS", {}),
                    mock.patch.object(api_server, "TASK_THREADS", {}),
                    mock.patch.object(api_server, "TASK_PROCESSES", {}),
                    mock.patch.object(api_server, "create_automatic_filename_asr", side_effect=blocking_evaluator),
                ):
                    response = api_server._submit_automatic_filename_asr(
                        task_id,
                        api_server.AsrEvalRequest(model=api_server.AUTO_FILENAME_ASR_MODEL, language="auto"),
                    )
                    self.assertEqual(json.loads(response.body.decode("utf-8"))["status"], "queued")
                    self.assertTrue(evaluator_entered.wait(timeout=5), "automatic Medium evaluator did not start")
                    self.assertIn(runtime_id, api_server.TASK_CANCEL_EVENTS)
                    self.assertIn(runtime_id, api_server.TASK_THREADS)
                    cancel_event = api_server.TASK_CANCEL_EVENTS[runtime_id]
                    worker = api_server.TASK_THREADS[runtime_id]

                    deletion_thread = threading.Thread(target=delete_history, name="delete-auto-medium-history")
                    deletion_thread.start()
                    self.assertTrue(cancel_event.wait(timeout=5), "ASR history deletion did not signal cancellation")
                    release_evaluator.set()
                    deletion_thread.join(timeout=10)
                    self.assertFalse(deletion_thread.is_alive(), "ASR history deletion did not finish")
                    worker.join(timeout=5)
                    self.assertFalse(worker.is_alive(), "automatic Medium worker did not stop after cancellation")

                    self.assertFalse(deletion_errors)
                    self.assertEqual(len(deletion_results), 1)
                    self.assertEqual(
                        deletion_results[0]["deletedSubtaskIds"],
                        [api_server.AUTO_FILENAME_ASR_SUB_ID],
                    )
                    self.assertEqual(deletion_results[0]["cancelledCount"], 1)
                    self.assertNotIn(runtime_id, api_server.TASK_CANCEL_EVENTS)
                    self.assertNotIn(runtime_id, api_server.TASK_THREADS)
                    self.assertNotIn(task_id, api_server.AUTOMATIC_FILENAME_ACTIVE_TASK_IDS)
            finally:
                release_evaluator.set()
                if deletion_thread is not None and deletion_thread.is_alive():
                    deletion_thread.join(timeout=20)
                if worker is not None and worker.is_alive():
                    worker.join(timeout=20)

            self.assertEqual(evaluator_calls, [task_id])
            stored_result = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))
            stored_status = json.loads((task_dir / "status.json").read_text(encoding="utf-8"))
            stored_sidecar = json.loads((task_dir / "automatic_filename_asr.json").read_text(encoding="utf-8"))
            self.assertTrue(task_dir.exists())
            self.assertTrue((original_dir / "original.wav").exists())
            self.assertTrue((protected_dir / "protected.wav").exists())
            self.assertFalse(stored_result.get("asrResults"))
            self.assertEqual(stored_status.get("asrTasks", []), [])
            self.assertNotIn("asrTask", stored_status)
            self.assertFalse((task_dir / "asr_result.json").exists())
            self.assertFalse((task_dir / "asr_results").exists())
            self.assertTrue(stored_sidecar["historyDeleted"])
            self.assertFalse(stored_sidecar["historySynced"])
            self.assertEqual(stored_sidecar["historyAsrSubId"], api_server.AUTO_FILENAME_ASR_SUB_ID)
            self.assertEqual(stored_result["displayFilename"], "record_saved_name.wav")
            self.assertEqual(stored_result["displayProtectedFilename"], "record_saved_name_protected.wav")
            self.assertEqual(stored_status["displayFilename"], "record_saved_name.wav")
            self.assertEqual(stored_status["displayProtectedFilename"], "record_saved_name_protected.wav")
            for field in ("status", "progress", "stage", "message", "elapsedSec", "error"):
                self.assertEqual(stored_status[field], protection_status[field], field)

    def test_download_uses_display_name_while_reading_stored_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp)
            task_id = "task_download_name"
            protected_dir = task_root / task_id / "protected"
            protected_dir.mkdir(parents=True)
            stored_name = "record_ed8a1de8-cd02-4b93-9550-cf063f089764_protected.wav"
            (protected_dir / stored_name).write_bytes(b"wav")
            result = minimal_result(task_id)
            result["audio"]["protected"].update({  # type: ignore[index]
                "filename": stored_name,
                "displayFilename": "record_this_is_the……competition_protected.wav",
            })
            (task_root / task_id / "result.json").write_text(json.dumps(result), encoding="utf-8")
            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
            ):
                response = api_server.download_protected_audio(task_id)
            self.assertEqual(Path(response.path).name, stored_name)
            disposition = response.headers.get("content-disposition") or ""
            self.assertIn("filename*=UTF-8''record_this_is_the", disposition)
            self.assertIn("%E2%80%A6%E2%80%A6competition_protected.wav", disposition)

    def test_download_hides_internal_uuid_before_display_name_is_generated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp)
            task_id = "task_download_fallback"
            protected_dir = task_root / task_id / "protected"
            protected_dir.mkdir(parents=True)
            stored_name = "record_ed8a1de8-cd02-4b93-9550-cf063f089764_protected.wav"
            (protected_dir / stored_name).write_bytes(b"wav")
            result = minimal_result(task_id)
            result["audio"]["protected"]["filename"] = stored_name  # type: ignore[index]
            (task_root / task_id / "result.json").write_text(json.dumps(result), encoding="utf-8")
            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
            ):
                response = api_server.download_protected_audio(task_id)

            self.assertEqual(Path(response.path).name, stored_name)
            disposition = response.headers.get("content-disposition") or ""
            self.assertIn("filename*=UTF-8''record_audio_protected.wav", disposition)
            self.assertNotIn("ed8a1de8", disposition)

    def test_artifact_download_uses_display_name_but_keeps_stored_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp)
            task_id = "task_artifact_name"
            stored_name = "record_ed8a1de8-cd02-4b93-9550-cf063f089764.wav"
            original_dir = task_root / task_id / "original"
            original_dir.mkdir(parents=True)
            (original_dir / stored_name).write_bytes(b"wav")
            result = minimal_result(task_id)
            result["audio"]["original"].update({  # type: ignore[index]
                "filename": stored_name,
                "displayFilename": "record_this_is_the……competition.wav",
            })
            (task_root / task_id / "result.json").write_text(json.dumps(result), encoding="utf-8")
            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
            ):
                response = api_server.artifact_file(task_id, "original", stored_name)
            self.assertEqual(Path(response.path).name, stored_name)
            self.assertIn("%E2%80%A6%E2%80%A6competition.wav", response.headers.get("content-disposition") or "")

    def test_automatic_filename_submission_is_atomic_and_starts_one_thread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp)
            task_id = "task_atomic_auto_name"
            task_dir = task_root / task_id
            task_dir.mkdir(parents=True)
            (task_dir / "result.json").write_text(json.dumps(minimal_result(task_id)), encoding="utf-8")

            real_thread = threading.Thread
            barrier = threading.Barrier(3)
            responses: list[object] = []
            errors: list[BaseException] = []

            def submit() -> None:
                try:
                    barrier.wait(timeout=5)
                    responses.append(api_server._submit_automatic_filename_asr(
                        task_id,
                        api_server.AsrEvalRequest(model=api_server.AUTO_FILENAME_ASR_MODEL, language="auto"),
                    ))
                except BaseException as exc:
                    errors.append(exc)

            submit_threads = [real_thread(target=submit) for _ in range(2)]
            runtime_id = f"{task_id}:{api_server.AUTO_FILENAME_ASR_SUB_ID}"
            HoldingThread.starts = 0
            HoldingThread.instances.clear()
            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
                mock.patch.object(api_server, "TASK_CANCEL_EVENTS", {}),
                mock.patch.object(api_server, "TASK_THREADS", {}),
                mock.patch.object(api_server, "TASK_PROCESSES", {}),
                mock.patch.object(api_server.threading, "Thread", HoldingThread),
                mock.patch.object(api_server, "write_automatic_asr_status", return_value={}),
            ):
                api_server.AUTOMATIC_FILENAME_ACTIVE_TASK_IDS.discard(task_id)
                for thread in submit_threads:
                    thread.start()
                barrier.wait(timeout=5)
                for thread in submit_threads:
                    thread.join(timeout=5)
                    self.assertFalse(thread.is_alive())
                self.assertFalse(errors)
                self.assertIn(runtime_id, api_server.TASK_THREADS)
                api_server.AUTOMATIC_FILENAME_ACTIVE_TASK_IDS.discard(task_id)
                api_server.cleanup_task_runtime(task_id, runtime_id=runtime_id)
            self.assertEqual(HoldingThread.starts, 1)
            statuses = [json.loads(response.body.decode("utf-8"))["status"] for response in responses]  # type: ignore[attr-defined]
            self.assertCountEqual(statuses, ["queued", "running"])

    def test_positive_env_int_uses_positive_values_and_falls_back_safely(self) -> None:
        name = "SEME2E_TEST_POSITIVE_ENV_INT"
        with mock.patch.dict(api_server.os.environ, {name: "3"}):
            self.assertEqual(api_server._positive_env_int(name, 4), 3)

        for invalid_value in ("", "not-an-int", "0", "-2"):
            with self.subTest(value=invalid_value), mock.patch.dict(api_server.os.environ, {name: invalid_value}):
                self.assertEqual(api_server._positive_env_int(name, 4), 4)

    def test_reference_text_free_clone_models_clear_irrelevant_annotation_fields(self) -> None:
        for model in ("xtts-v2", "your-tts"):
            with self.subTest(model=model):
                payload = api_server.CloneVoiceRequest(
                    text="clone text",
                    model=model,
                    language="en",
                    speakerPrompt="must be ignored",
                    annotationSource="asr",
                    annotationAsrSubId="asr_must_be_ignored",
                )

                resolved, error = api_server.resolve_clone_annotation("task_reference_text_free", payload, "req_test")

                self.assertIsNone(error)
                self.assertIsNotNone(resolved)
                self.assertIsNone(api_server.CloneVoiceRequest(text="clone text", model=model).annotationSource)
                self.assertIsNone(resolved["annotationSource"])
                self.assertIsNone(resolved["speakerPrompt"])
                self.assertIsNone(resolved["originalSpeakerPrompt"])
                self.assertIsNone(resolved["protectedSpeakerPrompt"])
                self.assertIsNone(resolved["annotationAsrSubId"])
                self.assertIsNone(resolved["annotationAsrModel"])
                self.assertIsNone(resolved["annotationCreatedAt"])

    def test_prompt_required_clone_models_default_to_manual_annotation(self) -> None:
        for model in ("cosyvoice2:0.5b", "gpt-sovits:finetune"):
            with self.subTest(model=model):
                payload = api_server.CloneVoiceRequest(
                    text="clone text",
                    model=model,
                    language="zh-cn",
                    speakerPrompt="manual transcript",
                )

                resolved, error = api_server.resolve_clone_annotation("task_manual", payload, "req_test")

                self.assertIsNone(error)
                self.assertIsNotNone(resolved)
                self.assertEqual(resolved["annotationSource"], "manual")
                self.assertEqual(resolved["speakerPrompt"], "manual transcript")
                self.assertEqual(resolved["originalSpeakerPrompt"], "manual transcript")
                self.assertEqual(resolved["protectedSpeakerPrompt"], "manual transcript")

    def test_reference_text_required_models_reject_missing_annotation_immediately(self) -> None:
        for model in ("cosyvoice2:0.5b", "gpt-sovits:finetune"):
            with self.subTest(model=model):
                payload = api_server.CloneVoiceRequest(
                    text="clone text",
                    model=model,
                    language="zh-cn",
                )

                resolved, error = api_server.resolve_clone_annotation("task_missing_annotation", payload, "req_test")

                self.assertIsNone(resolved)
                self.assertIsNotNone(error)
                body = json.loads(error.body.decode("utf-8"))
                self.assertEqual(body["error"]["code"], "REFERENCE_TEXT_REQUIRED")
                self.assertIn("参考音频对应文本", body["error"]["message"])

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

    def test_active_evaluation_batch_rejects_second_batch_of_same_type(self) -> None:
        for active_status in ("queued", "running"):
            with self.subTest(status=active_status), tempfile.TemporaryDirectory() as tmp:
                task_root = Path(tmp)
                task_id = f"task_active_{active_status}"
                task_dir = task_root / task_id
                task_dir.mkdir(parents=True)
                (task_dir / "result.json").write_text(json.dumps(minimal_result(task_id), ensure_ascii=False), encoding="utf-8")
                first = api_server.EvaluationBatchRequest(
                    batchId=f"batch_first_{active_status}",
                    type="asr",
                    items=[{"batchItemId": "tiny", "model": "openai-whisper:tiny"}],
                )
                second = api_server.EvaluationBatchRequest(
                    batchId=f"batch_second_{active_status}",
                    type="asr",
                    items=[{"batchItemId": "base", "model": "openai-whisper:base"}],
                )

                with mock.patch.object(api_server, "TASK_DIR", task_root):
                    first_response = api_server.create_evaluation_batch(task_id, first)
                    if active_status == "running":
                        status_path = task_dir / "status.json"
                        status_document = json.loads(status_path.read_text(encoding="utf-8"))
                        status_document["asrBatches"][0]["status"] = "running"
                        status_path.write_text(json.dumps(status_document, ensure_ascii=False), encoding="utf-8")
                    second_response = api_server.create_evaluation_batch(task_id, second)

                error = json.loads(second_response.body.decode("utf-8"))["error"]
                self.assertEqual(first_response.status_code, 200)
                self.assertEqual(second_response.status_code, 409)
                self.assertEqual(error["code"], "EVALUATION_BATCH_ACTIVE")
                self.assertEqual(error["details"]["batchId"], first.batchId)
                self.assertEqual(error["details"]["status"], active_status)
                self.assertEqual(error["details"]["type"], "asr")

    def test_terminal_evaluation_batches_do_not_block_new_batch(self) -> None:
        for terminal_status in ("completed", "failed", "partial_failed"):
            with self.subTest(status=terminal_status), tempfile.TemporaryDirectory() as tmp:
                task_root = Path(tmp)
                task_id = f"task_terminal_{terminal_status}"
                task_dir = task_root / task_id
                task_dir.mkdir(parents=True)
                (task_dir / "result.json").write_text(json.dumps(minimal_result(task_id), ensure_ascii=False), encoding="utf-8")
                first = api_server.EvaluationBatchRequest(
                    batchId=f"batch_terminal_{terminal_status}",
                    type="clone",
                    items=[{"batchItemId": "first", "model": "xtts-v2"}],
                )
                second = api_server.EvaluationBatchRequest(
                    batchId=f"batch_new_{terminal_status}",
                    type="clone",
                    items=[{"batchItemId": "second", "model": "xtts-v2"}],
                )

                with mock.patch.object(api_server, "TASK_DIR", task_root):
                    first_response = api_server.create_evaluation_batch(task_id, first)
                    status_path = task_dir / "status.json"
                    status_document = json.loads(status_path.read_text(encoding="utf-8"))
                    status_document["cloneBatches"][0]["status"] = terminal_status
                    status_path.write_text(json.dumps(status_document, ensure_ascii=False), encoding="utf-8")
                    second_response = api_server.create_evaluation_batch(task_id, second)
                    status = api_server.read_task_status(task_id)

                self.assertEqual(first_response.status_code, 200)
                self.assertEqual(second_response.status_code, 200)
                self.assertEqual([batch["batchId"] for batch in status["cloneBatches"]], [first.batchId, second.batchId])

    def test_asr_and_clone_evaluation_batches_do_not_block_each_other(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp)
            task_id = "task_cross_type_batches"
            task_dir = task_root / task_id
            task_dir.mkdir(parents=True)
            (task_dir / "result.json").write_text(json.dumps(minimal_result(task_id), ensure_ascii=False), encoding="utf-8")
            asr_request = api_server.EvaluationBatchRequest(
                batchId="batch_asr_active",
                type="asr",
                items=[{"batchItemId": "tiny", "model": "openai-whisper:tiny"}],
            )
            clone_request = api_server.EvaluationBatchRequest(
                batchId="batch_clone_active",
                type="clone",
                items=[{"batchItemId": "xtts", "model": "xtts-v2"}],
            )

            with mock.patch.object(api_server, "TASK_DIR", task_root):
                asr_response = api_server.create_evaluation_batch(task_id, asr_request)
                clone_response = api_server.create_evaluation_batch(task_id, clone_request)
                status = api_server.read_task_status(task_id)

            self.assertEqual(asr_response.status_code, 200)
            self.assertEqual(clone_response.status_code, 200)
            self.assertEqual(status["asrBatches"][0]["status"], "queued")
            self.assertEqual(status["cloneBatches"][0]["status"], "queued")

    def test_concurrent_same_type_batch_creation_allows_only_one_active_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp)
            task_id = "task_atomic_batch_create"
            task_dir = task_root / task_id
            task_dir.mkdir(parents=True)
            (task_dir / "result.json").write_text(json.dumps(minimal_result(task_id), ensure_ascii=False), encoding="utf-8")
            requests = [
                api_server.EvaluationBatchRequest(
                    batchId=f"batch_atomic_{index}",
                    type="asr",
                    items=[{"batchItemId": f"item_{index}", "model": "openai-whisper:tiny"}],
                )
                for index in range(2)
            ]

            with mock.patch.object(api_server, "TASK_DIR", task_root), ThreadPoolExecutor(max_workers=2) as executor:
                responses = list(executor.map(lambda request: api_server.create_evaluation_batch(task_id, request), requests))
                status = api_server.read_task_status(task_id)

            self.assertEqual(sorted(response.status_code for response in responses), [200, 409])
            self.assertEqual(len(status["asrBatches"]), 1)
            rejected = next(response for response in responses if response.status_code == 409)
            error = json.loads(rejected.body.decode("utf-8"))["error"]
            self.assertEqual(error["code"], "EVALUATION_BATCH_ACTIVE")
            self.assertEqual(error["details"]["batchId"], status["asrBatches"][0]["batchId"])
            self.assertEqual(error["details"]["status"], "queued")
            self.assertEqual(error["details"]["type"], "asr")

    def test_clone_batch_clears_annotations_for_models_without_required_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp)
            task_id = "task_clone_annotation_batch"
            task_dir = task_root / task_id
            task_dir.mkdir(parents=True)
            (task_dir / "result.json").write_text(json.dumps(minimal_result(task_id), ensure_ascii=False), encoding="utf-8")
            request = api_server.EvaluationBatchRequest(
                batchId="batch_clone_annotations",
                type="clone",
                items=[
                    {
                        "batchItemId": "xtts",
                        "model": "xtts-v2",
                        "annotationSource": "manual",
                        "speakerPrompt": "legacy manual prompt",
                        "annotationAsrSubId": "legacy_xtts_asr",
                    },
                    {
                        "batchItemId": "yourtts",
                        "model": "your-tts",
                        "annotationSource": "asr",
                        "speakerPrompt": "legacy ASR prompt",
                        "annotationAsrSubId": "legacy_yourtts_asr",
                    },
                    {
                        "batchItemId": "cosy",
                        "model": "cosyvoice2:0.5b",
                        "annotationSource": "manual",
                        "speakerPrompt": "required prompt",
                    },
                ],
            )

            with mock.patch.object(api_server, "TASK_DIR", task_root):
                response = api_server.create_evaluation_batch(task_id, request)
                status = api_server.read_task_status(task_id)

            self.assertEqual(response.status_code, 200)
            items = {item["batchItemId"]: item for item in status["cloneBatches"][0]["items"]}
            for item_id in ("xtts", "yourtts"):
                self.assertIsNone(items[item_id]["annotationSource"])
                self.assertIsNone(items[item_id]["speakerPrompt"])
                self.assertIsNone(items[item_id]["annotationAsrSubId"])
                self.assertIsNone(items[item_id]["annotationAsrModel"])
            self.assertEqual(items["cosy"]["annotationSource"], "manual")
            self.assertEqual(items["cosy"]["speakerPrompt"], "required prompt")

    def test_clone_batch_cannot_publish_reference_to_deleted_asr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp)
            task_id = "task_clone_batch_deleted_asr"
            task_dir = task_root / task_id
            task_dir.mkdir(parents=True)
            (task_dir / "result.json").write_text(json.dumps(minimal_result(task_id), ensure_ascii=False), encoding="utf-8")
            request = api_server.EvaluationBatchRequest(
                batchId="batch_clone_deleted_asr",
                type="clone",
                items=[
                    {
                        "batchItemId": "gpt",
                        "model": "gpt-sovits:finetune",
                        "annotationSource": "asr",
                        "annotationAsrSubId": "asr_deleted",
                    }
                ],
            )

            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(api_server, "DELETED_SUBTASK_KEYS", {f"{task_id}:asr_deleted"}),
                mock.patch.object(api_server, "EVALUATION_COLLECTION_DELETIONS", set()),
            ):
                response = api_server.create_evaluation_batch(task_id, request)

            self.assertEqual(response.status_code, 409)
            error = json.loads(response.body.decode("utf-8"))["error"]
            self.assertEqual(error["code"], "ASR_ANNOTATION_DELETED")
            status_path = task_dir / "status.json"
            self.assertFalse(status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("cloneBatches"))

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

    def test_deleting_asr_collection_keeps_clone_references_and_removes_unreferenced_asr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "tasks"
            task_root.mkdir()
            task_id = "task_delete_asr_only"
            task_dir = task_root / task_id
            (task_dir / "original").mkdir(parents=True)
            (task_dir / "protected").mkdir()
            (task_dir / "original" / "original.wav").write_bytes(b"original")
            (task_dir / "protected" / "protected.wav").write_bytes(b"protected")
            result = minimal_result(task_id)
            result["asrResults"] = [
                {"taskId": task_id, "asrSubId": "asr_delete", "asr": {"model": "tiny", "wer": 0.8, "cer": 0.7}},
                {"taskId": task_id, "asrSubId": "asr_keep", "asr": {"model": "base", "wer": 0.4, "cer": 0.3}},
            ]
            result["cloneResults"] = [
                {"taskId": task_id, "cloneSubId": "clone_keep", "cloneId": "voice_keep", "request": {"model": "gpt-sovits", "annotationSource": "asr", "annotationAsrSubId": "asr_keep"}, "cloneEval": {"cloneIdentityScore": 90, "identityBaselineWeight": 1, "cloneSemanticScore": 90, "semanticBaselineWeight": 1, "cloneQualityScore": 90, "qualityBaselineWeight": 1}}
            ]
            (task_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            (task_dir / "asr_results").mkdir()
            for item in result["asrResults"]:
                (task_dir / "asr_results" / f"{item['asrSubId']}.json").write_text(json.dumps(item), encoding="utf-8")
            (task_dir / "asr_result.json").write_text(json.dumps(result["asrResults"][-1]), encoding="utf-8")

            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
                mock.patch.object(api_server, "DELETED_SUBTASK_KEYS", set()),
            ):
                api_server.write_task_status(task_id, status="completed", progress=1, stage="asr_eval", asrSubId="asr_delete", asrResult=result["asrResults"][0])
                api_server.write_task_status(task_id, status="completed", progress=1, stage="asr_eval", asrSubId="asr_keep", asrResult=result["asrResults"][1])
                api_server.write_task_status(task_id, status="completed", progress=1, stage="downstream_tts_eval", cloneSubId="clone_keep", cloneRequest=result["cloneResults"][0]["request"], cloneResult=result["cloneResults"][0])
                deleted = api_server._delete_evaluation_collection(task_id, "asr")
                stored = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))
                status = json.loads(api_server.task_status_path(task_id).read_text(encoding="utf-8"))

            self.assertEqual(deleted["status"], "deleted")
            self.assertEqual(deleted["deletedSubtaskIds"], ["asr_delete"])
            self.assertEqual(deleted["preservedAsrSubIds"], ["asr_keep"])
            self.assertTrue((task_dir / "original" / "original.wav").exists())
            self.assertTrue((task_dir / "protected" / "protected.wav").exists())
            self.assertEqual([item["asrSubId"] for item in stored["asrResults"]], ["asr_keep"])
            self.assertEqual([item["cloneSubId"] for item in stored["cloneResults"]], ["clone_keep"])
            self.assertEqual([item["asrSubId"] for item in status["asrTasks"]], ["asr_keep"])
            self.assertEqual(status["cloneTask"]["cloneSubId"], "clone_keep")
            self.assertFalse((task_dir / "asr_results" / "asr_delete.json").exists())
            self.assertTrue((task_dir / "asr_results" / "asr_keep.json").exists())
            self.assertEqual(json.loads((task_dir / "asr_result.json").read_text(encoding="utf-8"))["asrSubId"], "asr_keep")

    def test_deleting_asr_collection_preserves_referenced_legacy_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "tasks"
            task_root.mkdir()
            task_id = "task_delete_legacy_asr"
            task_dir = task_root / task_id
            task_dir.mkdir()
            result = minimal_result(task_id)
            result["asrResults"] = [
                {"taskId": task_id, "asrSubId": "asr_keep", "asr": {"model": "base", "wer": 0.4, "cer": 0.3}}
            ]
            result["cloneResults"] = [
                {"taskId": task_id, "cloneSubId": "clone_keep", "request": {"annotationAsrSubId": "asr_keep"}, "cloneEval": {}}
            ]
            (task_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            legacy_asr = {
                "asrSubId": "asr_keep",
                "status": "completed",
                "stage": "asr_eval",
                "asrResult": result["asrResults"][0],
            }
            legacy_clone = {
                "cloneSubId": "clone_keep",
                "status": "completed",
                "stage": "downstream_tts_eval",
                "cloneResult": result["cloneResults"][0],
            }
            status = {
                "taskId": task_id,
                "status": "completed",
                "progress": 1,
                "stage": "downstream_tts_eval",
                "asrSubId": "asr_keep",
                "asrTask": legacy_asr,
                "asrResult": result["asrResults"][0],
                "cloneSubId": "clone_keep",
                "cloneTask": legacy_clone,
                "cloneResult": result["cloneResults"][0],
            }
            (task_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")

            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
                mock.patch.object(api_server, "DELETED_SUBTASK_KEYS", set()),
            ):
                deleted = api_server._delete_evaluation_collection(task_id, "asr")
                stored_status = json.loads(api_server.task_status_path(task_id).read_text(encoding="utf-8"))

            self.assertEqual(deleted["deletedSubtaskIds"], [])
            self.assertEqual(deleted["preservedAsrSubIds"], ["asr_keep"])
            self.assertEqual([item["asrSubId"] for item in stored_status["asrTasks"]], ["asr_keep"])
            self.assertEqual(stored_status["asrTask"]["asrSubId"], "asr_keep")
            self.assertEqual(stored_status["asrResult"]["asrSubId"], "asr_keep")

    def test_deleting_asr_collection_restores_referenced_sidecar_only_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "tasks"
            task_root.mkdir()
            task_id = "task_delete_sidecar_asr"
            task_dir = task_root / task_id
            task_dir.mkdir()
            result = minimal_result(task_id)
            result["cloneResults"] = [
                {"taskId": task_id, "cloneSubId": "clone_keep", "request": {"annotationAsrSubId": "asr_keep"}, "cloneEval": {}}
            ]
            (task_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            sidecar = {
                "taskId": task_id,
                "asrSubId": "asr_keep",
                "status": "available",
                "asr": {"model": "base", "wer": 0.4, "cer": 0.3, "originalText": "clean", "protectedText": "protected"},
            }
            (task_dir / "asr_results").mkdir()
            (task_dir / "asr_results" / "asr_keep.json").write_text(json.dumps(sidecar), encoding="utf-8")
            (task_dir / "asr_result.json").write_text(json.dumps(sidecar), encoding="utf-8")
            (task_dir / "clone_result.json").write_text(json.dumps(result["cloneResults"][0]), encoding="utf-8")

            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
                mock.patch.object(api_server, "DELETED_SUBTASK_KEYS", set()),
            ):
                deleted = api_server._delete_evaluation_collection(task_id, "asr")
                stored = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))

            self.assertEqual(deleted["deletedSubtaskIds"], [])
            self.assertEqual(deleted["preservedAsrSubIds"], ["asr_keep"])
            self.assertEqual([item["asrSubId"] for item in stored["asrResults"]], ["asr_keep"])
            self.assertEqual(json.loads((task_dir / "asr_result.json").read_text(encoding="utf-8"))["asrSubId"], "asr_keep")
            self.assertTrue((task_dir / "asr_results" / "asr_keep.json").exists())

    def test_referenced_asr_can_be_deleted_after_clone_collection_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "tasks"
            task_root.mkdir()
            task_id = "task_delete_clone_then_asr"
            task_dir = task_root / task_id
            (task_dir / "original").mkdir(parents=True)
            (task_dir / "protected").mkdir()
            (task_dir / "original" / "original.wav").write_bytes(b"original")
            (task_dir / "protected" / "protected.wav").write_bytes(b"protected")
            result = minimal_result(task_id)
            asr_result = {"taskId": task_id, "asrSubId": "asr_keep", "asr": {"model": "base", "wer": 0.4, "cer": 0.3}}
            clone_result = {
                "taskId": task_id,
                "cloneSubId": "clone_delete",
                "cloneId": "voice_delete",
                "request": {"model": "gpt-sovits", "annotationSource": "asr", "annotationAsrSubId": "asr_keep"},
                "cloneEval": {},
            }
            result["asrResults"] = [asr_result]
            result["cloneResults"] = [clone_result]
            (task_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            (task_dir / "asr_results").mkdir()
            (task_dir / "asr_results" / "asr_keep.json").write_text(json.dumps(asr_result), encoding="utf-8")
            (task_dir / "asr_result.json").write_text(json.dumps(asr_result), encoding="utf-8")
            (task_dir / "clone_results").mkdir()
            (task_dir / "clone_results" / "clone_delete.json").write_text(json.dumps(clone_result), encoding="utf-8")
            (task_dir / "clone_result.json").write_text(json.dumps(clone_result), encoding="utf-8")

            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
                mock.patch.object(api_server, "DELETED_SUBTASK_KEYS", set()),
            ):
                api_server.write_task_status(task_id, status="completed", progress=1, stage="asr_eval", asrSubId="asr_keep", asrResult=asr_result)
                api_server.write_task_status(task_id, status="completed", progress=1, stage="downstream_tts_eval", cloneSubId="clone_delete", cloneRequest=clone_result["request"], cloneResult=clone_result)
                first_asr_delete = api_server._delete_evaluation_collection(task_id, "asr")
                clone_delete = api_server._delete_evaluation_collection(task_id, "clone")
                second_asr_delete = api_server._delete_evaluation_collection(task_id, "asr")
                stored = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))
                stored_status = json.loads(api_server.task_status_path(task_id).read_text(encoding="utf-8"))

            self.assertEqual(first_asr_delete["preservedAsrSubIds"], ["asr_keep"])
            self.assertEqual(clone_delete["deletedSubtaskIds"], ["clone_delete"])
            self.assertEqual(second_asr_delete["preservedAsrSubIds"], [])
            self.assertEqual(second_asr_delete["deletedSubtaskIds"], ["asr_keep"])
            self.assertEqual(stored["asrResults"], [])
            self.assertEqual(stored["cloneResults"], [])
            self.assertEqual(stored_status.get("asrTasks"), [])
            self.assertNotIn("asrTask", stored_status)
            self.assertNotIn("asrSubId", stored_status)
            self.assertFalse((task_dir / "asr_result.json").exists())
            self.assertFalse((task_dir / "asr_results").exists())
            self.assertTrue((task_dir / "result.json").exists())
            self.assertTrue((task_dir / "original" / "original.wav").exists())
            self.assertTrue((task_dir / "protected" / "protected.wav").exists())

    def test_deleting_single_referenced_asr_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "tasks"
            task_root.mkdir()
            task_id = "task_delete_referenced_asr"
            task_dir = task_root / task_id
            task_dir.mkdir()
            result = minimal_result(task_id)
            result["asrResults"] = [{"taskId": task_id, "asrSubId": "asr_keep", "asr": {"model": "base"}}]
            result["cloneResults"] = [{"taskId": task_id, "cloneSubId": "clone_keep", "request": {"annotationAsrSubId": "asr_keep"}}]
            (task_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
                mock.patch.object(api_server, "DELETED_SUBTASK_KEYS", set()),
            ):
                with self.assertRaises(api_server.HTTPException) as raised:
                    api_server._delete_evaluation_subtask(task_id, "asr", "asr_keep")

            self.assertEqual(raised.exception.status_code, 409)
            self.assertIn("still referenced", str(raised.exception.detail))
            self.assertTrue((task_dir / "result.json").exists())

    def test_deleting_asr_collection_cancels_only_unreferenced_asr_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "tasks"
            task_root.mkdir()
            task_id = "task_delete_asr_runtime"
            task_dir = task_root / task_id
            task_dir.mkdir()
            result = minimal_result(task_id)
            result["asrResults"] = [
                {"taskId": task_id, "asrSubId": "asr_delete", "asr": {"model": "tiny"}},
                {"taskId": task_id, "asrSubId": "asr_keep", "asr": {"model": "base"}},
            ]
            result["cloneResults"] = [
                {"taskId": task_id, "cloneSubId": "clone_keep", "request": {"annotationAsrSubId": "asr_keep"}}
            ]
            (task_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            protect_cancel = threading.Event()
            delete_cancel = threading.Event()
            keep_cancel = threading.Event()
            clone_cancel = threading.Event()

            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
                mock.patch.object(api_server, "TASK_CANCEL_EVENTS", {
                    task_id: protect_cancel,
                    f"{task_id}:asr_delete": delete_cancel,
                    f"{task_id}:asr_keep": keep_cancel,
                    f"{task_id}:clone_keep": clone_cancel,
                }),
                mock.patch.object(api_server, "TASK_THREADS", {}),
                mock.patch.object(api_server, "TASK_PROCESSES", {}),
                mock.patch.object(api_server, "DELETED_SUBTASK_KEYS", set()),
            ):
                api_server.write_task_status(task_id, status="running", progress=0.3, stage="asr_eval", asrSubId="asr_delete", asrResult=None)
                api_server.write_task_status(task_id, status="running", progress=0.4, stage="asr_eval", asrSubId="asr_keep", asrResult=None)
                api_server.write_task_status(task_id, status="running", progress=0.5, stage="downstream_tts_eval", cloneSubId="clone_keep", cloneRequest=result["cloneResults"][0]["request"], cloneResult=None)
                deleted = api_server._delete_evaluation_collection(task_id, "asr")

            self.assertEqual(deleted["deletedSubtaskIds"], ["asr_delete"])
            self.assertEqual(deleted["preservedAsrSubIds"], ["asr_keep"])
            self.assertEqual(deleted["cancelledCount"], 1)
            self.assertTrue(delete_cancel.is_set())
            self.assertFalse(keep_cancel.is_set())
            self.assertFalse(clone_cancel.is_set())
            self.assertFalse(protect_cancel.is_set())

    def test_evaluation_submission_is_rejected_while_collection_delete_is_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "tasks"
            task_root.mkdir()
            task_id = "task_delete_submission_guard"
            task_dir = task_root / task_id
            task_dir.mkdir()
            (task_dir / "result.json").write_text(json.dumps(minimal_result(task_id), ensure_ascii=False), encoding="utf-8")

            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
                mock.patch.object(api_server, "EVALUATION_COLLECTION_DELETIONS", {f"{task_id}:asr", f"{task_id}:clone"}),
            ):
                with self.assertRaises(api_server.HTTPException) as asr_error:
                    api_server.run_asr_eval(task_id, api_server.AsrEvalRequest(model="openai-whisper:tiny", language="en"))
                with self.assertRaises(api_server.HTTPException) as clone_error:
                    api_server.clone_voice(task_id, api_server.CloneVoiceRequest(text="hello", model="xtts-v2", language="en"))

            self.assertEqual(asr_error.exception.status_code, 409)
            self.assertEqual(clone_error.exception.status_code, 409)
            self.assertNotIn("asrTasks", json.loads((task_dir / "status.json").read_text(encoding="utf-8")) if (task_dir / "status.json").exists() else {})

    def test_clone_submission_and_single_asr_delete_share_reference_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "tasks"
            task_root.mkdir()
            task_id = "task_clone_reference_lock"
            task_dir = task_root / task_id
            task_dir.mkdir()
            result = minimal_result(task_id)
            result["asrResults"] = [
                {
                    "taskId": task_id,
                    "asrSubId": "asr_keep",
                    "status": "available",
                    "asr": {"model": "base", "originalText": "clean", "protectedText": "protected"},
                }
            ]
            (task_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            lock_entered = threading.Event()
            release_clone = threading.Event()
            original_resolver = api_server.resolve_clone_annotation

            def blocking_resolver(*args: object, **kwargs: object):
                resolved = original_resolver(*args, **kwargs)
                lock_entered.set()
                self.assertTrue(release_clone.wait(timeout=5))
                return resolved

            payload = api_server.CloneVoiceRequest(
                text="hello",
                model="gpt-sovits:finetune",
                language="en",
                annotationSource="asr",
                annotationAsrSubId="asr_keep",
            )
            clone_response: list[object] = []
            delete_error: list[BaseException] = []

            def submit_clone() -> None:
                clone_response.append(api_server.clone_voice(task_id, payload))

            def delete_asr() -> None:
                try:
                    api_server._delete_evaluation_subtask(task_id, "asr", "asr_keep")
                except BaseException as exc:
                    delete_error.append(exc)

            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
                mock.patch.object(api_server, "resolve_clone_annotation", side_effect=blocking_resolver),
                mock.patch.object(api_server, "validate_clone_config", return_value=None),
                mock.patch.object(api_server, "EVALUATION_COLLECTION_DELETIONS", set()),
                mock.patch.object(api_server, "DELETED_SUBTASK_KEYS", set()),
            ):
                clone_thread = threading.Thread(target=submit_clone)
                delete_thread = threading.Thread(target=delete_asr)
                clone_thread.start()
                self.assertTrue(lock_entered.wait(timeout=5))
                delete_thread.start()
                self.assertTrue(delete_thread.is_alive())
                release_clone.set()
                clone_thread.join(timeout=5)
                delete_thread.join(timeout=5)

            self.assertEqual(len(clone_response), 1)
            self.assertEqual(len(delete_error), 1)
            self.assertIsInstance(delete_error[0], api_server.HTTPException)
            self.assertEqual(delete_error[0].status_code, 409)

    def test_single_asr_delete_wins_before_clone_reference_is_published(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "tasks"
            task_root.mkdir()
            task_id = "task_asr_delete_wins"
            task_dir = task_root / task_id
            task_dir.mkdir()
            result = minimal_result(task_id)
            result["asrResults"] = [
                {
                    "taskId": task_id,
                    "asrSubId": "asr_delete",
                    "status": "available",
                    "asr": {"model": "base", "originalText": "clean", "protectedText": "protected"},
                }
            ]
            (task_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            delete_entered = threading.Event()
            release_delete = threading.Event()
            original_remove_status = api_server._remove_subtask_status

            def blocking_remove_status(*args: object, **kwargs: object):
                delete_entered.set()
                self.assertTrue(release_delete.wait(timeout=5))
                return original_remove_status(*args, **kwargs)

            payload = api_server.CloneVoiceRequest(
                text="hello",
                model="gpt-sovits:finetune",
                language="en",
                annotationSource="asr",
                annotationAsrSubId="asr_delete",
            )
            delete_response: list[object] = []
            clone_response: list[object] = []

            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
                mock.patch.object(api_server, "_remove_subtask_status", side_effect=blocking_remove_status),
                mock.patch.object(api_server, "validate_clone_config", return_value=None),
                mock.patch.object(api_server, "EVALUATION_COLLECTION_DELETIONS", set()),
                mock.patch.object(api_server, "DELETED_SUBTASK_KEYS", set()),
            ):
                delete_thread = threading.Thread(
                    target=lambda: delete_response.append(api_server._delete_evaluation_subtask(task_id, "asr", "asr_delete"))
                )
                clone_thread = threading.Thread(target=lambda: clone_response.append(api_server.clone_voice(task_id, payload)))
                delete_thread.start()
                self.assertTrue(delete_entered.wait(timeout=5))
                clone_thread.start()
                clone_thread.join(timeout=5)
                release_delete.set()
                delete_thread.join(timeout=5)

            self.assertEqual(len(delete_response), 1)
            self.assertEqual(len(clone_response), 1)
            self.assertEqual(clone_response[0].status_code, 409)
            clone_payload = json.loads(clone_response[0].body)
            self.assertEqual(clone_payload["error"]["code"], "ASR_ANNOTATION_DELETED")
            stored_status = json.loads((task_dir / "status.json").read_text(encoding="utf-8"))
            self.assertFalse(stored_status.get("cloneTasks"))

    def test_result_adapter_persist_guard_blocks_deleted_subtask_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "tasks"
            task_root.mkdir()
            task_id = "task_deleted_persist_guard"
            task_dir = task_root / task_id
            (task_dir / "original").mkdir(parents=True)
            (task_dir / "protected").mkdir()
            (task_dir / "original" / "original.wav").write_bytes(b"original")
            (task_dir / "protected" / "protected.wav").write_bytes(b"protected")
            (task_dir / "result.json").write_text(json.dumps(minimal_result(task_id), ensure_ascii=False), encoding="utf-8")
            fake_asr = {"status": "available", "model": "base", "wer": 0.5, "cer": 0.4}

            with (
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "maybe_asr_eval", return_value=fake_asr),
            ):
                with self.assertRaisesRegex(RuntimeError, "TASK_CANCELLED"):
                    result_adapter.create_asr_eval(
                        task_id,
                        {"model": "base", "language": "en", "asrSubId": "asr_deleted"},
                        persist_allowed=lambda: False,
                    )

            stored = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))
            self.assertFalse(stored.get("asrResults"))

    def test_clone_artifact_cleanup_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "task"
            original_dir = task_dir / "original"
            protected_dir = task_dir / "protected"
            original_dir.mkdir(parents=True)
            protected_dir.mkdir()
            (original_dir / "original.wav").write_bytes(b"original")
            (protected_dir / "protected.wav").write_bytes(b"protected")

            api_server._remove_clone_artifacts(task_dir, {"cloneId": "../original"})
            api_server._remove_clone_artifacts(task_dir, {"cloneId": "..\\protected"})

            self.assertTrue((original_dir / "original.wav").exists())
            self.assertTrue((protected_dir / "protected.wav").exists())

    def test_result_json_wins_over_stale_sidecar_for_same_asr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "tasks"
            task_root.mkdir()
            task_id = "task_result_wins_sidecar"
            task_dir = task_root / task_id
            task_dir.mkdir()
            result = minimal_result(task_id)
            result["asrResults"] = [
                {"taskId": task_id, "asrSubId": "asr_keep", "asr": {"model": "base", "wer": 0.2, "originalText": "new"}}
            ]
            result["cloneResults"] = [
                {"taskId": task_id, "cloneSubId": "clone_keep", "request": {"annotationAsrSubId": "asr_keep"}}
            ]
            (task_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            (task_dir / "asr_results").mkdir()
            stale = {"taskId": task_id, "asrSubId": "asr_keep", "asr": {"model": "base", "wer": 0.8, "originalText": "stale"}}
            (task_dir / "asr_results" / "asr_keep.json").write_text(json.dumps(stale), encoding="utf-8")
            (task_dir / "asr_result.json").write_text(json.dumps(stale), encoding="utf-8")

            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
                mock.patch.object(api_server, "EVALUATION_COLLECTION_DELETIONS", set()),
                mock.patch.object(api_server, "DELETED_SUBTASK_KEYS", set()),
            ):
                api_server._delete_evaluation_collection(task_id, "asr")
                stored = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))

            self.assertEqual(stored["asrResults"][0]["asr"]["wer"], 0.2)
            self.assertEqual(stored["asrResults"][0]["asr"]["originalText"], "new")

    def test_deleting_running_clone_only_cancels_its_runtime_and_removes_its_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / "tasks"
            task_root.mkdir()
            task_id = "task_delete_clone_only"
            task_dir = task_root / task_id
            (task_dir / "original").mkdir(parents=True)
            (task_dir / "protected").mkdir()
            (task_dir / "original" / "original.wav").write_bytes(b"original")
            (task_dir / "protected" / "protected.wav").write_bytes(b"protected")
            clone_dir = task_dir / "clones" / "clone_voice_delete"
            clone_dir.mkdir(parents=True)
            (clone_dir / "audio.wav").write_bytes(b"clone")
            result = minimal_result(task_id)
            result["asrResults"] = [{"taskId": task_id, "asrSubId": "asr_keep", "asr": {"model": "base", "wer": 0.4, "cer": 0.3}}]
            result["cloneResults"] = [{"taskId": task_id, "cloneSubId": "clone_delete", "cloneId": "clone_voice_delete", "request": {"model": "xtts-v2"}, "cloneEval": {}}]
            (task_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            clone_cancel = threading.Event()
            asr_cancel = threading.Event()
            protect_cancel = threading.Event()

            with (
                mock.patch.object(api_server, "TASK_DIR", task_root),
                mock.patch.object(result_adapter, "TASK_DIR", task_root),
                mock.patch.object(api_server, "TASK_CANCEL_EVENTS", {
                    task_id: protect_cancel,
                    f"{task_id}:asr_keep": asr_cancel,
                    f"{task_id}:clone_delete": clone_cancel,
                }),
                mock.patch.object(api_server, "TASK_THREADS", {}),
                mock.patch.object(api_server, "TASK_PROCESSES", {}),
                mock.patch.object(api_server, "DELETED_SUBTASK_KEYS", set()),
            ):
                api_server.write_task_status(task_id, status="completed", progress=1, stage="asr_eval", asrSubId="asr_keep", asrResult=result["asrResults"][0])
                api_server.write_task_status(task_id, status="running", progress=0.4, stage="downstream_tts_eval", cloneSubId="clone_delete", cloneResult=None)
                deleted = api_server._delete_evaluation_collection(task_id, "clone")
                stored = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))
                status = json.loads(api_server.task_status_path(task_id).read_text(encoding="utf-8"))

            self.assertEqual(deleted["cancelledCount"], 1)
            self.assertTrue(clone_cancel.is_set())
            self.assertFalse(asr_cancel.is_set())
            self.assertFalse(protect_cancel.is_set())
            self.assertTrue((task_dir / "original" / "original.wav").exists())
            self.assertTrue((task_dir / "protected" / "protected.wav").exists())
            self.assertFalse(clone_dir.exists())
            self.assertEqual([item["asrSubId"] for item in stored["asrResults"]], ["asr_keep"])
            self.assertEqual(stored["cloneResults"], [])
            self.assertEqual(status["asrTask"]["asrSubId"], "asr_keep")
            self.assertNotIn("cloneTask", status)
            self.assertEqual(status["stage"], "report_generation")

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

            for model in ("cosyvoice2:0.5b", "gpt-sovits:finetune"):
                with self.subTest(model=model):
                    payload = api_server.CloneVoiceRequest(
                        text="测试克隆",
                        model=model,
                        language="zh-cn",
                        speakerPrompt="",
                        annotationSource="asr",
                        annotationAsrSubId="asr_old",
                    )
                    with mock.patch.object(api_server, "TASK_DIR", task_root), mock.patch.object(result_adapter, "TASK_DIR", task_root):
                        resolved, error = api_server.resolve_clone_annotation(task_id, payload, "req_test")

                    self.assertIsNone(error)
                    self.assertIsNotNone(resolved)
                    self.assertEqual(resolved["annotationSource"], "asr")
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
        with (
            mock.patch.object(result_adapter, "_cosyvoice_model_status", return_value=("available", None, None)),
            mock.patch.object(result_adapter, "_run_cancellable_subprocess", return_value=completed) as runner,
        ):
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

        command = runner.call_args.args[0]
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
