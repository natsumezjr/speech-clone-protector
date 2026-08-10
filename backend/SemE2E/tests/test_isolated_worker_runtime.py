from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import api_server
import result_adapter


class IsolatedWorkerRuntimeTest(unittest.TestCase):
    def _worker(self, source: str) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        worker = Path(temporary.name) / "fixture_worker.py"
        worker.write_text(textwrap.dedent(source), encoding="utf-8")
        return temporary, worker

    def test_isolated_json_worker_returns_success_payload(self) -> None:
        temporary, worker = self._worker(
            """
            import json
            import os
            import sys

            payload = json.load(sys.stdin)
            print("fixture worker started", flush=True)
            print(json.dumps({
                "ok": True,
                "echo": payload,
                "cwd": os.getcwd(),
                "pythonExecutable": sys.executable,
                "pythonPath": os.environ.get("PYTHONPATH", ""),
            }), flush=True)
            """
        )
        self.addCleanup(temporary.cleanup)

        response = result_adapter._run_isolated_json_worker(
            worker,
            {"model": "fixture", "value": 7},
            timeout_seconds=5,
        )

        self.assertTrue(response["ok"])
        self.assertEqual(response["echo"], {"model": "fixture", "value": 7})
        self.assertEqual(Path(response["cwd"]), result_adapter.ROOT)
        self.assertEqual(Path(response["pythonExecutable"]), Path(sys.executable))
        self.assertEqual(
            Path(response["pythonPath"].split(result_adapter.os.pathsep, 1)[0]),
            result_adapter.ROOT,
        )

    def test_isolated_json_worker_preserves_nonzero_error_diagnostics(self) -> None:
        temporary, worker = self._worker(
            """
            import json
            import sys

            json.load(sys.stdin)
            print("synthetic stderr detail", file=sys.stderr, flush=True)
            print(json.dumps({
                "ok": False,
                "error": {
                    "code": "WORKER_BOOM",
                    "message": "synthetic worker failure",
                    "details": {"probe": "nonzero"},
                },
            }), flush=True)
            raise SystemExit(7)
            """
        )
        self.addCleanup(temporary.cleanup)

        with self.assertRaisesRegex(result_adapter.IsolatedWorkerError, "synthetic worker failure") as raised:
            result_adapter._run_isolated_json_worker(
                worker,
                {"value": "failure"},
                timeout_seconds=5,
            )

        diagnostics = raised.exception.diagnostics
        self.assertEqual(diagnostics["returnCode"], 7)
        self.assertEqual(diagnostics["response"]["error"]["code"], "WORKER_BOOM")
        self.assertEqual(diagnostics["response"]["error"]["details"], {"probe": "nonzero"})
        self.assertIn("synthetic stderr detail", diagnostics["stderrTail"])
        self.assertEqual(Path(diagnostics["worker"]), worker.resolve())

    def test_isolated_json_worker_timeout_terminates_process(self) -> None:
        temporary, worker = self._worker(
            """
            import json
            import sys
            import time

            json.load(sys.stdin)
            print("worker entered sleep", file=sys.stderr, flush=True)
            time.sleep(30)
            """
        )
        self.addCleanup(temporary.cleanup)

        started = time.monotonic()
        with self.assertRaisesRegex(result_adapter.IsolatedWorkerError, "timed out") as raised:
            result_adapter._run_isolated_json_worker(
                worker,
                {"value": "timeout"},
                timeout_seconds=1,
            )

        diagnostics = raised.exception.diagnostics
        self.assertEqual(diagnostics["timeoutSec"], 1)
        self.assertIsNotNone(diagnostics["returnCode"])
        self.assertIn("worker entered sleep", diagnostics["stderrTail"])
        self.assertLess(time.monotonic() - started, 10)

    def test_isolated_json_worker_cancel_event_terminates_process(self) -> None:
        temporary, worker = self._worker(
            """
            import json
            import sys
            import time

            json.load(sys.stdin)
            print("worker waiting for cancellation", file=sys.stderr, flush=True)
            time.sleep(30)
            """
        )
        self.addCleanup(temporary.cleanup)
        cancel_event = threading.Event()
        timer = threading.Timer(0.3, cancel_event.set)
        timer.start()
        self.addCleanup(timer.cancel)

        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "TASK_CANCELLED"):
            result_adapter._run_isolated_json_worker(
                worker,
                {"value": "cancel"},
                timeout_seconds=10,
                cancel_event=cancel_event,
            )

        self.assertTrue(cancel_event.is_set())
        self.assertLess(time.monotonic() - started, 10)

    def test_coqui_pair_runs_once_in_isolated_worker_with_explicit_model_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            model_dir = root / "model"
            config_path = model_dir / "config.local.json"
            original_reference = root / "original.wav"
            protected_reference = root / "protected.wav"
            original_output = root / "out" / "original.wav"
            protected_output = root / "out" / "protected.wav"
            cancel_event = threading.Event()
            worker_response = {"ok": True, "sourceModel": "xtts_v2"}
            with (
                mock.patch.object(result_adapter, "_local_tts_model_files", return_value=(model_dir, config_path)),
                mock.patch.object(result_adapter, "_run_isolated_json_worker", return_value=worker_response) as worker,
                mock.patch.object(result_adapter, "COQUI_TTS_WORKER_SLOTS", threading.BoundedSemaphore(1)),
                mock.patch.dict(result_adapter.os.environ, {"SEME2E_COQUI_TTS_WORKER_TIMEOUT_SECONDS": "33"}),
            ):
                response = result_adapter._coqui_tts_clone_pair(
                    original_reference,
                    protected_reference,
                    original_output,
                    protected_output,
                    text="hello",
                    model="xtts_v2",
                    language="en",
                    speed=1.0,
                    device="cuda:0",
                    task_id="task_test",
                    clone_sub_id="clone_test",
                    cancel_event=cancel_event,
                )

        self.assertEqual(response, worker_response)
        worker.assert_called_once()
        worker_path, payload = worker.call_args.args
        self.assertEqual(worker_path, result_adapter.ROOT / "coqui_tts_worker.py")
        self.assertEqual(payload["model"], "xtts_v2")
        self.assertEqual(payload["modelPath"], str(model_dir.resolve()))
        self.assertEqual(payload["configPath"], str(config_path.resolve()))
        self.assertEqual(payload["originalReferencePath"], str(original_reference.resolve()))
        self.assertEqual(payload["protectedReferencePath"], str(protected_reference.resolve()))
        self.assertEqual(payload["originalOutputPath"], str(original_output.resolve()))
        self.assertEqual(payload["protectedOutputPath"], str(protected_output.resolve()))
        self.assertEqual(worker.call_args.kwargs["timeout_seconds"], 33)
        self.assertIs(worker.call_args.kwargs["cancel_event"], cancel_event)


class TaskRuntimeRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        with api_server.TASK_REGISTRY_LOCK:
            self.original_cancel_events = dict(api_server.TASK_CANCEL_EVENTS)
            self.original_threads = dict(api_server.TASK_THREADS)
            self.original_processes = dict(api_server.TASK_PROCESSES)
            api_server.TASK_CANCEL_EVENTS.clear()
            api_server.TASK_THREADS.clear()
            api_server.TASK_PROCESSES.clear()

    def tearDown(self) -> None:
        with api_server.TASK_REGISTRY_LOCK:
            api_server.TASK_CANCEL_EVENTS.clear()
            api_server.TASK_CANCEL_EVENTS.update(self.original_cancel_events)
            api_server.TASK_THREADS.clear()
            api_server.TASK_THREADS.update(self.original_threads)
            api_server.TASK_PROCESSES.clear()
            api_server.TASK_PROCESSES.update(self.original_processes)

    def test_request_all_task_cancels_covers_exact_and_subtask_entries(self) -> None:
        exact_event = threading.Event()
        asr_event = threading.Event()
        clone_event = threading.Event()
        unrelated_event = threading.Event()
        exact_thread = threading.Thread(name="exact-thread")
        asr_thread = threading.Thread(name="asr-thread")
        clone_thread = threading.Thread(name="clone-thread")
        unrelated_thread = threading.Thread(name="unrelated-thread")
        exact_process = object()
        asr_process = object()
        clone_process = object()
        unrelated_process = object()

        with api_server.TASK_REGISTRY_LOCK:
            api_server.TASK_CANCEL_EVENTS.update(
                {
                    "task_target": exact_event,
                    "task_target:asr_1": asr_event,
                    "task_target:clone_1": clone_event,
                    "task_other:asr_1": unrelated_event,
                }
            )
            api_server.TASK_THREADS.update(
                {
                    "task_target": exact_thread,
                    "task_target:asr_1": asr_thread,
                    "task_target:clone_1": clone_thread,
                    "task_other:asr_1": unrelated_thread,
                }
            )
            api_server.TASK_PROCESSES.update(
                {
                    "task_target": exact_process,
                    "task_target:asr_1": asr_process,
                    "task_target:clone_1": clone_process,
                    "task_other:asr_1": unrelated_process,
                }
            )

        events, threads, processes = api_server.request_all_task_cancels("task_target")

        self.assertTrue(exact_event.is_set())
        self.assertTrue(asr_event.is_set())
        self.assertTrue(clone_event.is_set())
        self.assertFalse(unrelated_event.is_set())
        self.assertEqual({id(value) for value in events}, {id(exact_event), id(asr_event), id(clone_event)})
        self.assertEqual({id(value) for value in threads}, {id(exact_thread), id(asr_thread), id(clone_thread)})
        self.assertEqual({id(value) for value in processes}, {id(exact_process), id(asr_process), id(clone_process)})

    def test_cleanup_all_task_runtimes_removes_exact_and_subtask_entries_only(self) -> None:
        registries = (
            api_server.TASK_CANCEL_EVENTS,
            api_server.TASK_THREADS,
            api_server.TASK_PROCESSES,
        )
        for registry in registries:
            registry.update(
                {
                    "task_target": object(),
                    "task_target:asr_1": object(),
                    "task_target:clone_1": object(),
                    "task_targetish:asr_1": object(),
                    "task_other:clone_1": object(),
                }
            )

        api_server.cleanup_all_task_runtimes("task_target")

        for registry in registries:
            self.assertNotIn("task_target", registry)
            self.assertNotIn("task_target:asr_1", registry)
            self.assertNotIn("task_target:clone_1", registry)
            self.assertIn("task_targetish:asr_1", registry)
            self.assertIn("task_other:clone_1", registry)


if __name__ == "__main__":
    unittest.main()
