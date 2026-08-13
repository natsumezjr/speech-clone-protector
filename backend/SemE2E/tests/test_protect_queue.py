from __future__ import annotations

import sys
import os
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import api_server


class FakeProcess:
    instances: list["FakeProcess"] = []

    def __init__(self, *, target: object, args: tuple[object, ...], daemon: bool) -> None:
        del target, daemon
        self.task_id = str(args[0])
        self.selected_gpu = args[7] if len(args) > 7 else None
        self.started = False
        self.exitcode: int | None = None
        self._released = threading.Event()
        self.__class__.instances.append(self)

    @property
    def pid(self) -> int | None:
        return 1000 + self.__class__.instances.index(self) if self.started else None

    def start(self) -> None:
        self.started = True

    def join(self, timeout: float | None = None) -> None:
        self._released.wait(timeout)
        if self._released.is_set():
            self.exitcode = 0

    def is_alive(self) -> bool:
        return self.started and not self._released.is_set()

    def release(self) -> None:
        self._released.set()


def wait_until(predicate: object, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():  # type: ignore[operator]
            return True
        time.sleep(0.01)
    return False


class ProtectQueueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = mock.patch.dict(
            os.environ,
            {
                "SEME2E_API_DEVICE": "cpu",
                "SEME2E_GPU_POOL": "",
                "SEME2E_PROTECT_GPU_POOL": "",
                "SEME2E_PROTECT_CUDA_VISIBLE_DEVICES": "",
            },
            clear=False,
        )
        self.environment.start()
        FakeProcess.instances.clear()
        with api_server.PROTECT_QUEUE_LOCK:
            api_server.PROTECT_PENDING_TASKS.clear()
            api_server.PROTECT_ACTIVE_TASK_IDS.clear()
        with api_server.TASK_REGISTRY_LOCK:
            api_server.TASK_CANCEL_EVENTS.clear()
            api_server.TASK_THREADS.clear()
            api_server.TASK_PROCESSES.clear()

    def tearDown(self) -> None:
        timer = api_server.PROTECT_DISPATCH_RETRY_TIMER
        if timer is not None:
            timer.cancel()
            api_server.PROTECT_DISPATCH_RETRY_TIMER = None
        for process in FakeProcess.instances:
            process.release()
        wait_until(lambda: not api_server.PROTECT_ACTIVE_TASK_IDS)
        with api_server.PROTECT_QUEUE_LOCK:
            api_server.PROTECT_PENDING_TASKS.clear()
            api_server.PROTECT_ACTIVE_TASK_IDS.clear()
        self.environment.stop()

    def test_protection_processes_use_spawn_context(self) -> None:
        self.assertEqual(api_server.PROTECT_PROCESS_CONTEXT.get_start_method(), "spawn")

    def test_only_two_processes_start_and_later_jobs_wait_for_a_slot(self) -> None:
        statuses: dict[str, dict[str, object]] = {}

        def write_status(task_id: str, **updates: object) -> dict[str, object]:
            statuses.setdefault(task_id, {}).update(updates)
            return statuses[task_id]

        def make_job(index: int) -> dict[str, object]:
            return {
                "task_id": f"task_{index}",
                "request_id": f"req_{index}",
                "uploaded_path": f"audio_{index}.wav",
                "uploaded_filename": f"audio_{index}.wav",
                "file_id": f"file_{index}",
                "payload": {"fileId": f"file_{index}"},
                "cancel_event": threading.Event(),
            }

        with (
            mock.patch.object(api_server.PROTECT_PROCESS_CONTEXT, "Process", FakeProcess),
            mock.patch.object(api_server, "write_task_status", side_effect=write_status),
            mock.patch.object(api_server, "read_task_status", side_effect=lambda task_id: statuses.get(task_id, {"status": "running"})),
            mock.patch.object(api_server, "is_task_deleted", return_value=False),
        ):
            for index in range(1, 5):
                api_server.enqueue_protect_job(make_job(index))

            self.assertEqual([process.task_id for process in FakeProcess.instances], ["task_1", "task_2"])
            self.assertEqual(api_server.PROTECT_ACTIVE_TASK_IDS, {"task_1", "task_2"})
            self.assertEqual([job["task_id"] for job in api_server.PROTECT_PENDING_TASKS], ["task_3", "task_4"])
            self.assertEqual(statuses["task_3"]["queuePosition"], 1)
            self.assertEqual(statuses["task_4"]["queuePosition"], 2)

            FakeProcess.instances[0].release()
            self.assertTrue(wait_until(lambda: len(FakeProcess.instances) == 3))
            self.assertEqual(FakeProcess.instances[2].task_id, "task_3")
            self.assertEqual([job["task_id"] for job in api_server.PROTECT_PENDING_TASKS], ["task_4"])

            for process in tuple(FakeProcess.instances):
                process.release()
            self.assertTrue(wait_until(lambda: len(FakeProcess.instances) == 4))
            for process in tuple(FakeProcess.instances):
                process.release()
            self.assertTrue(wait_until(lambda: not api_server.PROTECT_ACTIVE_TASK_IDS))

    def test_dynamic_protect_gpu_is_passed_to_child_and_released_after_exit(self) -> None:
        statuses: dict[str, dict[str, object]] = {}
        gpu_slot = mock.Mock()
        job = {
            "task_id": "task_dynamic",
            "request_id": "req_dynamic",
            "uploaded_path": "audio.wav",
            "uploaded_filename": "audio.wav",
            "file_id": "file_dynamic",
            "payload": {"fileId": "file_dynamic"},
            "cancel_event": threading.Event(),
        }

        with (
            mock.patch.object(api_server.PROTECT_PROCESS_CONTEXT, "Process", FakeProcess),
            mock.patch.object(api_server, "_try_acquire_protect_gpu", return_value=("5", gpu_slot, True)),
            mock.patch.object(api_server, "release_gpu_slot", side_effect=lambda slot: slot.release()),
            mock.patch.object(api_server, "write_task_status", side_effect=lambda task_id, **updates: statuses.setdefault(task_id, {}).update(updates) or statuses[task_id]),
            mock.patch.object(api_server, "read_task_status", side_effect=lambda task_id: statuses.get(task_id, {"status": "running"})),
            mock.patch.object(api_server, "is_task_deleted", return_value=False),
        ):
            api_server.enqueue_protect_job(job)
            self.assertEqual(len(FakeProcess.instances), 1)
            self.assertEqual(FakeProcess.instances[0].selected_gpu, "5")
            FakeProcess.instances[0].release()
            self.assertTrue(wait_until(lambda: not api_server.PROTECT_ACTIVE_TASK_IDS))

        gpu_slot.release.assert_called_once_with()

    def test_protect_job_remains_queued_while_all_gpu_slots_are_busy(self) -> None:
        job = {
            "task_id": "task_waiting_gpu",
            "request_id": "req_waiting_gpu",
            "uploaded_path": "audio.wav",
            "uploaded_filename": "audio.wav",
            "file_id": "file_waiting_gpu",
            "payload": {"fileId": "file_waiting_gpu"},
            "cancel_event": threading.Event(),
        }

        with (
            mock.patch.object(api_server, "_try_acquire_protect_gpu", return_value=(None, None, False)),
            mock.patch.object(api_server, "_schedule_protect_dispatch_retry_locked") as schedule_retry,
            mock.patch.object(api_server, "write_task_status", return_value={}),
            mock.patch.object(api_server, "is_task_deleted", return_value=False),
        ):
            api_server.enqueue_protect_job(job)

        self.assertEqual([queued["task_id"] for queued in api_server.PROTECT_PENDING_TASKS], ["task_waiting_gpu"])
        self.assertFalse(api_server.PROTECT_ACTIVE_TASK_IDS)
        schedule_retry.assert_called_once_with()

    def test_successful_protection_watcher_triggers_automatic_medium_once(self) -> None:
        process = mock.Mock(exitcode=0)
        cancel_event = threading.Event()
        with (
            mock.patch.object(api_server, "is_task_deleted", return_value=False),
            mock.patch.object(api_server, "read_task_status", return_value={"status": "completed"}),
            mock.patch.object(api_server, "cleanup_protect_process_runtime"),
            mock.patch.object(api_server, "_dispatch_protect_tasks_locked"),
            mock.patch.object(api_server, "_apply_task_display_filenames", return_value={}),
            mock.patch.object(api_server, "_submit_automatic_filename_asr") as submit,
        ):
            api_server._watch_protect_process("task_success", process, cancel_event)
        submit.assert_called_once()
        request = submit.call_args.args[1]
        self.assertEqual(request.model, api_server.AUTO_FILENAME_ASR_MODEL)

    def test_failed_protection_watcher_does_not_trigger_automatic_asr(self) -> None:
        process = mock.Mock(exitcode=1)
        cancel_event = threading.Event()
        with (
            mock.patch.object(api_server, "is_task_deleted", return_value=False),
            mock.patch.object(api_server, "read_task_status", return_value={"status": "failed"}),
            mock.patch.object(api_server, "cleanup_protect_process_runtime"),
            mock.patch.object(api_server, "_dispatch_protect_tasks_locked"),
            mock.patch.object(api_server, "_submit_automatic_filename_asr") as submit,
        ):
            api_server._watch_protect_process("task_failed", process, cancel_event)
        submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
