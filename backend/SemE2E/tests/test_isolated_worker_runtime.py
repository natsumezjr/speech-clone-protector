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

    def test_isolated_json_worker_applies_child_only_environment_overrides(self) -> None:
        temporary, worker = self._worker(
            """
            import json
            import os
            import sys

            json.load(sys.stdin)
            print(json.dumps({
                "ok": True,
                "fixtureValue": os.environ.get("VOICE_SHIELD_FIXTURE_ENV"),
                "removedValue": os.environ.get("VOICE_SHIELD_REMOVED_ENV"),
            }), flush=True)
            """
        )
        self.addCleanup(temporary.cleanup)

        with mock.patch.dict(
            result_adapter.os.environ,
            {
                "VOICE_SHIELD_FIXTURE_ENV": "parent",
                "VOICE_SHIELD_REMOVED_ENV": "parent-only",
            },
        ):
            response = result_adapter._run_isolated_json_worker(
                worker,
                {"value": "environment"},
                timeout_seconds=5,
                env_overrides={
                    "VOICE_SHIELD_FIXTURE_ENV": "child",
                    "VOICE_SHIELD_REMOVED_ENV": None,
                },
            )
            self.assertEqual(result_adapter.os.environ["VOICE_SHIELD_FIXTURE_ENV"], "parent")
            self.assertEqual(result_adapter.os.environ["VOICE_SHIELD_REMOVED_ENV"], "parent-only")

        self.assertEqual(response["fixtureValue"], "child")
        self.assertIsNone(response["removedValue"])

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

    def test_cancellable_subprocess_cancel_stops_active_process_tree(self) -> None:
        temporary, worker = self._worker(
            """
            import time

            time.sleep(30)
            """
        )
        self.addCleanup(temporary.cleanup)
        cancel_event = threading.Event()
        timer = threading.Timer(0.3, cancel_event.set)
        timer.start()
        self.addCleanup(timer.cancel)

        started = time.monotonic()
        with mock.patch.object(
            result_adapter,
            "_stop_isolated_worker",
            wraps=result_adapter._stop_isolated_worker,
        ) as stop_worker:
            with self.assertRaisesRegex(RuntimeError, "TASK_CANCELLED"):
                result_adapter._run_cancellable_subprocess(
                    [sys.executable, str(worker)],
                    cwd=worker.parent,
                    env=result_adapter.os.environ.copy(),
                    timeout_seconds=10,
                    cancel_event=cancel_event,
                )

        stop_worker.assert_called_once()
        self.assertTrue(cancel_event.is_set())
        self.assertLess(time.monotonic() - started, 10)

    def test_windows_worker_stop_uses_taskkill_process_tree(self) -> None:
        process = mock.Mock()
        process.pid = 43210
        process.poll.return_value = None
        process.wait.return_value = 0
        taskkill_result = mock.Mock(returncode=0)

        with (
            mock.patch.object(result_adapter.os, "name", "nt"),
            mock.patch.object(result_adapter.subprocess, "run", return_value=taskkill_result) as taskkill,
        ):
            result_adapter._stop_isolated_worker(process, grace_seconds=0.5)

        command = taskkill.call_args.args[0]
        self.assertEqual(command[:3], ["taskkill.exe", "/PID", "43210"])
        self.assertIn("/T", command)
        self.assertIn("/F", command)
        process.wait.assert_called_once_with(timeout=0.5)

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
                mock.patch.dict(
                    result_adapter.os.environ,
                    {
                        "SEME2E_COQUI_TTS_WORKER_TIMEOUT_SECONDS": "33",
                        "SEME2E_COQUI_TTS_CUDA_VISIBLE_DEVICES": "4",
                    },
                ),
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
        self.assertEqual(payload["device"], "cuda:0")
        self.assertEqual(worker.call_args.kwargs["timeout_seconds"], 33)
        self.assertIs(worker.call_args.kwargs["cancel_event"], cancel_event)
        self.assertEqual(worker.call_args.kwargs["env_overrides"], {"CUDA_VISIBLE_DEVICES": "4"})

    def test_asr_and_clone_asr_visible_devices_use_logical_cuda_zero(self) -> None:
        worker_response = {
            "ok": True,
            "model": "openai-whisper:base",
            "language": "en",
            "originalText": "original transcript",
            "protectedText": "protected transcript",
        }
        with (
            mock.patch.object(result_adapter, "_run_isolated_json_worker", return_value=worker_response) as worker,
            mock.patch.object(result_adapter, "ASR_WORKER_SLOTS", threading.BoundedSemaphore(1)),
            mock.patch.dict(
                result_adapter.os.environ,
                {
                    "SEME2E_API_DEVICE": "cuda:7",
                    "SEME2E_ASR_CUDA_VISIBLE_DEVICES": "2",
                },
            ),
        ):
            result_adapter.maybe_asr_eval(
                Path("original.wav"),
                Path("protected.wav"),
                {
                    "semantic": {"asrModel": "openai-whisper:base"},
                    "language": "en",
                    "forceAsrEval": True,
                },
            )

        asr_payload = worker.call_args.args[1]
        self.assertEqual(asr_payload["device"], "cuda:0")
        self.assertEqual(worker.call_args.kwargs["env_overrides"], {"CUDA_VISIBLE_DEVICES": "2"})

        with (
            mock.patch.object(result_adapter, "_run_isolated_json_worker", return_value=worker_response) as worker,
            mock.patch.object(result_adapter, "ASR_WORKER_SLOTS", threading.BoundedSemaphore(1)),
            mock.patch.dict(
                result_adapter.os.environ,
                {
                    "SEME2E_CLONE_ASR_DEVICE": "cuda:6",
                    "SEME2E_CLONE_ASR_CUDA_VISIBLE_DEVICES": "3",
                },
            ),
        ):
            response = result_adapter._transcribe_clone_pair_isolated(
                Path("original_clone.wav"),
                Path("protected_clone.wav"),
                {"asrModel": "openai-whisper:base", "language": "en", "text": "target"},
            )

        clone_payload = worker.call_args.args[1]
        self.assertEqual(clone_payload["device"], "cuda:0")
        self.assertEqual(worker.call_args.kwargs["env_overrides"], {"CUDA_VISIBLE_DEVICES": "3"})
        self.assertEqual(response["status"], "available")

    def test_cosyvoice_visible_device_uses_logical_cuda_zero(self) -> None:
        completed = mock.Mock(
            returncode=0,
            stdout='VOICE_SHIELD_COSYVOICE_RESULT={"ok": true}\n',
            stderr="",
        )
        with (
            mock.patch.object(result_adapter, "_cosyvoice_model_status", return_value=("available", None, None)),
            mock.patch.object(result_adapter, "_run_cancellable_subprocess", return_value=completed) as runner,
            mock.patch.object(result_adapter, "COSYVOICE_WORKER_SLOTS", threading.BoundedSemaphore(1)),
            mock.patch.dict(
                result_adapter.os.environ,
                {"SEME2E_COSYVOICE_CUDA_VISIBLE_DEVICES": "4"},
            ),
        ):
            response = result_adapter._cosyvoice_clone_pair(
                Path("original.wav"),
                Path("protected.wav"),
                Path("original_clone.wav"),
                Path("protected_clone.wav"),
                text="target text",
                original_prompt_text="original transcript",
                protected_prompt_text="protected transcript",
                speed=1.0,
                device="cuda:6",
            )

        command = runner.call_args.args[0]
        self.assertEqual(command[command.index("--device") + 1], "cuda:0")
        self.assertEqual(runner.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "4")
        self.assertEqual(runner.call_args.kwargs["timeout_seconds"], 900)
        self.assertTrue(response["ok"])

    def test_gpt_sovits_visible_device_routes_training_and_worker_consistently(self) -> None:
        completed = mock.Mock(
            returncode=0,
            stdout='VOICE_SHIELD_GPT_SOVITS_LIVE_RESULT={"ok": true}\n',
            stderr="",
        )
        with (
            mock.patch.object(result_adapter, "_gpt_sovits_model_status", return_value=("available", None, None)),
            mock.patch.object(result_adapter, "_run_cancellable_subprocess", return_value=completed) as runner,
            mock.patch.object(result_adapter, "GPT_SOVITS_WORKER_SLOTS", threading.BoundedSemaphore(1)),
            mock.patch.dict(
                result_adapter.os.environ,
                {"SEME2E_GPT_SOVITS_CUDA_VISIBLE_DEVICES": "5"},
            ),
        ):
            response = result_adapter._gpt_sovits_clone_pair(
                Path("original.wav"),
                Path("protected.wav"),
                Path("original_clone.wav"),
                Path("protected_clone.wav"),
                original_transcript="original transcript",
                protected_transcript="protected transcript",
                text="target text",
                language="en",
                speed=1.0,
                device="cuda:3",
            )

        command = runner.call_args.args[0]
        self.assertEqual(command[command.index("--device") + 1], "cuda:0")
        self.assertEqual(command[command.index("--gpu-numbers") + 1], "5")
        self.assertEqual(command[command.index("--cuda-visible-devices") + 1], "5")
        self.assertEqual(runner.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "5")
        self.assertEqual(runner.call_args.kwargs["timeout_seconds"], 2700)
        self.assertTrue(response["ok"])

    def test_expensive_clone_worker_concurrency_defaults_to_one(self) -> None:
        if "SEME2E_COSYVOICE_WORKER_MAX_CONCURRENCY" not in result_adapter.os.environ:
            self.assertEqual(result_adapter.COSYVOICE_WORKER_MAX_CONCURRENCY, 1)
        if "SEME2E_GPT_SOVITS_WORKER_MAX_CONCURRENCY" not in result_adapter.os.environ:
            self.assertEqual(result_adapter.GPT_SOVITS_WORKER_MAX_CONCURRENCY, 1)
        if "SEME2E_CLONE_GPU_MAX_CONCURRENCY" not in result_adapter.os.environ:
            self.assertEqual(result_adapter.CLONE_GPU_MAX_CONCURRENCY, 1)

    def test_worker_slot_wait_honors_cancellation_without_consuming_slot(self) -> None:
        semaphore = threading.BoundedSemaphore(1)
        semaphore.acquire()
        cancel_event = threading.Event()
        timer = threading.Timer(0.05, cancel_event.set)
        timer.start()
        self.addCleanup(timer.cancel)

        with self.assertRaisesRegex(RuntimeError, "TASK_CANCELLED"):
            result_adapter._acquire_worker_slot(semaphore, cancel_event)

        semaphore.release()
        self.assertTrue(semaphore.acquire(blocking=False))
        semaphore.release()

        already_cancelled = threading.Event()
        already_cancelled.set()
        with self.assertRaisesRegex(RuntimeError, "TASK_CANCELLED"):
            result_adapter._acquire_worker_slot(semaphore, already_cancelled)
        self.assertTrue(semaphore.acquire(blocking=False))
        semaphore.release()

    def test_clone_gpu_slot_keys_resolve_explicit_and_parent_visible_devices(self) -> None:
        with mock.patch.dict(
            result_adapter.os.environ,
            {
                "SEME2E_COSYVOICE_CUDA_VISIBLE_DEVICES": "5, 2, 5",
                "CUDA_VISIBLE_DEVICES": "8,7",
            },
        ):
            self.assertEqual(
                result_adapter._clone_gpu_slot_keys(
                    "cuda:1",
                    "SEME2E_COSYVOICE_CUDA_VISIBLE_DEVICES",
                ),
                ("2", "5"),
            )

        with mock.patch.dict(
            result_adapter.os.environ,
            {
                "SEME2E_COSYVOICE_CUDA_VISIBLE_DEVICES": "",
                "CUDA_VISIBLE_DEVICES": "8,7",
            },
        ):
            self.assertEqual(
                result_adapter._clone_gpu_slot_keys(
                    "cuda:1",
                    "SEME2E_COSYVOICE_CUDA_VISIBLE_DEVICES",
                ),
                ("7",),
            )
            self.assertEqual(
                result_adapter._clone_gpu_slot_keys(
                    "cpu",
                    "SEME2E_COSYVOICE_CUDA_VISIBLE_DEVICES",
                ),
                (),
            )

    def test_different_clone_models_share_one_physical_gpu_slot(self) -> None:
        first_entered = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()
        errors: list[BaseException] = []

        def occupy_first_model() -> None:
            slots: list[threading.BoundedSemaphore] = []
            try:
                slots = result_adapter._acquire_clone_worker_slots(
                    threading.BoundedSemaphore(1),
                    ("4",),
                    None,
                )
                first_entered.set()
                release_first.wait(timeout=3)
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)
            finally:
                result_adapter._release_worker_slots(slots)

        def occupy_second_model() -> None:
            slots: list[threading.BoundedSemaphore] = []
            try:
                slots = result_adapter._acquire_clone_worker_slots(
                    threading.BoundedSemaphore(1),
                    ("4",),
                    None,
                )
                second_entered.set()
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)
            finally:
                result_adapter._release_worker_slots(slots)

        with (
            mock.patch.object(result_adapter, "CLONE_GPU_MAX_CONCURRENCY", 1),
            mock.patch.object(result_adapter, "CLONE_GPU_SLOTS", {}),
        ):
            first_thread = threading.Thread(target=occupy_first_model)
            second_thread = threading.Thread(target=occupy_second_model)
            first_thread.start()
            self.assertTrue(first_entered.wait(timeout=1))
            second_thread.start()
            self.assertFalse(second_entered.wait(timeout=0.15))
            release_first.set()
            self.assertTrue(second_entered.wait(timeout=1))
            first_thread.join(timeout=1)
            second_thread.join(timeout=1)

        self.assertFalse(first_thread.is_alive())
        self.assertFalse(second_thread.is_alive())
        self.assertEqual(errors, [])

    def test_clone_workers_on_different_physical_gpus_can_enter_together(self) -> None:
        with (
            mock.patch.object(result_adapter, "CLONE_GPU_MAX_CONCURRENCY", 1),
            mock.patch.object(result_adapter, "CLONE_GPU_SLOTS", {}),
        ):
            first_slots = result_adapter._acquire_clone_worker_slots(
                threading.BoundedSemaphore(1),
                ("4",),
                None,
            )
            try:
                second_slots = result_adapter._acquire_clone_worker_slots(
                    threading.BoundedSemaphore(1),
                    ("5",),
                    None,
                )
                result_adapter._release_worker_slots(second_slots)
            finally:
                result_adapter._release_worker_slots(first_slots)

    def test_clone_gpu_slot_wait_cancellation_releases_model_slot(self) -> None:
        with (
            mock.patch.object(result_adapter, "CLONE_GPU_MAX_CONCURRENCY", 1),
            mock.patch.object(result_adapter, "CLONE_GPU_SLOTS", {}),
        ):
            first_slots = result_adapter._acquire_clone_worker_slots(
                threading.BoundedSemaphore(1),
                ("4",),
                None,
            )
            second_model_slot = threading.BoundedSemaphore(1)
            cancel_event = threading.Event()
            timer = threading.Timer(0.05, cancel_event.set)
            timer.start()
            try:
                with self.assertRaisesRegex(RuntimeError, "TASK_CANCELLED"):
                    result_adapter._acquire_clone_worker_slots(
                        second_model_slot,
                        ("4",),
                        cancel_event,
                    )
            finally:
                timer.cancel()

            self.assertTrue(second_model_slot.acquire(blocking=False))
            second_model_slot.release()
            result_adapter._release_worker_slots(first_slots)
            shared_gpu_slot = result_adapter._clone_gpu_slot("4")
            self.assertTrue(shared_gpu_slot.acquire(blocking=False))
            shared_gpu_slot.release()

    def test_cosyvoice_and_gpt_sovits_cancel_while_waiting_for_worker_slots(self) -> None:
        cosy_slots = threading.BoundedSemaphore(1)
        cosy_slots.acquire()
        cosy_cancel = threading.Event()
        cosy_timer = threading.Timer(0.05, cosy_cancel.set)
        cosy_timer.start()
        try:
            with (
                mock.patch.object(result_adapter, "_cosyvoice_model_status", return_value=("available", None, None)),
                mock.patch.object(result_adapter, "COSYVOICE_WORKER_SLOTS", cosy_slots),
                mock.patch.object(result_adapter, "_run_cancellable_subprocess") as runner,
            ):
                with self.assertRaisesRegex(RuntimeError, "TASK_CANCELLED"):
                    result_adapter._cosyvoice_clone_pair(
                        Path("original.wav"),
                        Path("protected.wav"),
                        Path("original_clone.wav"),
                        Path("protected_clone.wav"),
                        text="target text",
                        original_prompt_text="original transcript",
                        protected_prompt_text="protected transcript",
                        speed=1.0,
                        device="cuda:0",
                        cancel_event=cosy_cancel,
                    )
                runner.assert_not_called()
        finally:
            cosy_timer.cancel()
            cosy_slots.release()

        gpt_slots = threading.BoundedSemaphore(1)
        gpt_slots.acquire()
        gpt_cancel = threading.Event()
        gpt_timer = threading.Timer(0.05, gpt_cancel.set)
        gpt_timer.start()
        try:
            with (
                mock.patch.object(result_adapter, "_gpt_sovits_model_status", return_value=("available", None, None)),
                mock.patch.object(result_adapter, "GPT_SOVITS_WORKER_SLOTS", gpt_slots),
                mock.patch.object(result_adapter, "_run_cancellable_subprocess") as runner,
            ):
                with self.assertRaisesRegex(RuntimeError, "TASK_CANCELLED"):
                    result_adapter._gpt_sovits_clone_pair(
                        Path("original.wav"),
                        Path("protected.wav"),
                        Path("original_clone.wav"),
                        Path("protected_clone.wav"),
                        original_transcript="original transcript",
                        protected_transcript="protected transcript",
                        text="target text",
                        language="en",
                        speed=1.0,
                        device="cuda:0",
                        cancel_event=gpt_cancel,
                    )
                runner.assert_not_called()
        finally:
            gpt_timer.cancel()
            gpt_slots.release()


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
