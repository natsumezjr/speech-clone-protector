from __future__ import annotations

import sys
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
        FakeProcess.instances.clear()
        with api_server.PROTECT_QUEUE_LOCK:
            api_server.PROTECT_PENDING_TASKS.clear()
            api_server.PROTECT_ACTIVE_TASK_IDS.clear()
        with api_server.TASK_REGISTRY_LOCK:
            api_server.TASK_CANCEL_EVENTS.clear()
            api_server.TASK_THREADS.clear()
            api_server.TASK_PROCESSES.clear()

    def tearDown(self) -> None:
        for process in FakeProcess.instances:
            process.release()
        wait_until(lambda: not api_server.PROTECT_ACTIVE_TASK_IDS)
        with api_server.PROTECT_QUEUE_LOCK:
            api_server.PROTECT_PENDING_TASKS.clear()
            api_server.PROTECT_ACTIVE_TASK_IDS.clear()

    def test_protection_processes_use_spawn_context(self) -> None:
        self.assertEqual(api_server.PROTECT_PROCESS_CONTEXT.get_start_method(), "spawn")

    def test_only_four_processes_start_and_fifth_waits_for_a_slot(self) -> None:
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
            for index in range(1, 7):
                api_server.enqueue_protect_job(make_job(index))

            self.assertEqual([process.task_id for process in FakeProcess.instances], ["task_1", "task_2", "task_3", "task_4"])
            self.assertEqual(api_server.PROTECT_ACTIVE_TASK_IDS, {"task_1", "task_2", "task_3", "task_4"})
            self.assertEqual([job["task_id"] for job in api_server.PROTECT_PENDING_TASKS], ["task_5", "task_6"])
            self.assertEqual(statuses["task_5"]["queuePosition"], 1)
            self.assertEqual(statuses["task_6"]["queuePosition"], 2)

            FakeProcess.instances[0].release()
            self.assertTrue(wait_until(lambda: len(FakeProcess.instances) == 5))
            self.assertEqual(FakeProcess.instances[4].task_id, "task_5")
            self.assertEqual([job["task_id"] for job in api_server.PROTECT_PENDING_TASKS], ["task_6"])

            for process in tuple(FakeProcess.instances):
                process.release()
            self.assertTrue(wait_until(lambda: len(FakeProcess.instances) == 6))
            for process in tuple(FakeProcess.instances):
                process.release()
            self.assertTrue(wait_until(lambda: not api_server.PROTECT_ACTIVE_TASK_IDS))


if __name__ == "__main__":
    unittest.main()
