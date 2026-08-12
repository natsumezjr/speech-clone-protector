from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from contextlib import contextmanager
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
        self.assertEqual(
            worker.call_args.kwargs["env_overrides"],
            {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "4"},
        )

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
        self.assertEqual(
            worker.call_args.kwargs["env_overrides"],
            {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "2"},
        )

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
        self.assertEqual(
            worker.call_args.kwargs["env_overrides"],
            {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "3"},
        )
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
        lease_condition = threading.Condition()
        leases: set[str] = set()
        with (
            mock.patch.object(result_adapter, "_gpt_sovits_model_status", return_value=("available", None, None)),
            mock.patch.object(result_adapter, "_run_cancellable_subprocess", return_value=completed) as runner,
            mock.patch.object(result_adapter, "GPT_SOVITS_WORKER_MAX_CONCURRENCY", 2),
            mock.patch.object(result_adapter, "GPT_SOVITS_WORKER_SLOTS", threading.BoundedSemaphore(2)),
            mock.patch.object(result_adapter, "CLONE_GPU_SLOTS", {}),
            mock.patch.object(
                result_adapter,
                "_nvidia_gpu_inventory",
                return_value=(("5", "2"), {"5": 48000, "2": 47000}),
            ),
            mock.patch.object(result_adapter, "GPT_SOVITS_GPU_LEASE_CONDITION", lease_condition),
            mock.patch.object(result_adapter, "GPT_SOVITS_GPU_LEASES", leases),
            mock.patch.dict(
                result_adapter.os.environ,
                {
                    "SEME2E_GPT_SOVITS_GPU_POOL": "5,2",
                    "SEME2E_GPT_SOVITS_CUDA_VISIBLE_DEVICES": "",
                },
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
        self.assertEqual(command[command.index("--min-reference-seconds") + 1], "3.0")
        self.assertEqual(command[command.index("--max-reference-seconds") + 1], "10.0")
        self.assertEqual(runner.call_args.kwargs["env"]["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(runner.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "5")
        self.assertGreater(runner.call_args.kwargs["timeout_seconds"], 0)
        self.assertLessEqual(runner.call_args.kwargs["timeout_seconds"], 900)
        self.assertEqual(leases, set())
        self.assertTrue(response["ok"])

    def test_gpu_worker_retry_switches_cards_after_cuda_oom_and_keeps_diagnostics(self) -> None:
        selected: list[str | None] = []

        def operation(
            _worker_device: str,
            _worker_env: dict[str, str] | None,
            selected_gpu: str | None,
            _attempt_timeout_seconds: float,
        ) -> dict[str, bool]:
            selected.append(selected_gpu)
            if len(selected) == 1:
                raise result_adapter.IsolatedWorkerError(
                    "CUDA out of memory",
                    diagnostics={"stderrTail": "CUDA out of memory while loading fixture"},
                )
            return {"ok": True}

        with (
            mock.patch.object(result_adapter, "CLONE_GPU_MAX_CONCURRENCY", 1),
            mock.patch.object(result_adapter, "CLONE_GPU_SLOTS", {}),
            mock.patch.object(
                result_adapter,
                "_nvidia_gpu_inventory",
                return_value=(
                    ("5", "2"),
                    {"5": 48000, "2": 47000},
                    {"5": "gpu-5", "2": "gpu-2"},
                ),
            ),
            mock.patch.dict(
                result_adapter.os.environ,
                {
                    "SEME2E_GPU_POOL": "5,2",
                    "SEME2E_GPU_RETRY_TIMEOUT_SECONDS": "1",
                    "SEME2E_GPU_WAIT_POLL_SECONDS": "0.01",
                },
            ),
        ):
            response, final_gpu, attempts = result_adapter._run_gpu_worker_with_retry(
                operation_name="fixture",
                worker_slot=threading.BoundedSemaphore(1),
                requested_device="cuda:0",
                visible_devices_env="SEME2E_FIXTURE_CUDA_VISIBLE_DEVICES",
                cancel_event=None,
                operation=operation,
                timeout_seconds=1,
            )

            self.assertEqual(selected, ["5", "2"])
            self.assertEqual(final_gpu, "2")
            self.assertTrue(response["ok"])
            self.assertEqual(attempts[0]["reason"], "gpu_memory_exhausted")
            self.assertEqual(
                attempts[0]["workerDiagnostics"]["stderrTail"],
                "CUDA out of memory while loading fixture",
            )
            for gpu in ("5", "2"):
                slot = result_adapter._clone_gpu_slot(gpu)
                self.assertTrue(slot.acquire(blocking=False))
                slot.release()

    def test_gpu_worker_retry_does_not_retry_non_resource_error(self) -> None:
        calls = 0
        expected = result_adapter.IsolatedWorkerError(
            "invalid model configuration",
            diagnostics={"stderrTail": "checkpoint shape mismatch"},
        )

        def operation(
            _worker_device: str,
            _worker_env: dict[str, str] | None,
            _selected_gpu: str | None,
            _attempt_timeout_seconds: float,
        ) -> dict[str, bool]:
            nonlocal calls
            calls += 1
            raise expected

        with (
            mock.patch.object(result_adapter, "CLONE_GPU_SLOTS", {}),
            mock.patch.object(
                result_adapter,
                "_nvidia_gpu_inventory",
                return_value=(("5", "2"), {"5": 48000, "2": 47000}),
            ),
            mock.patch.dict(result_adapter.os.environ, {"SEME2E_GPU_POOL": "5,2"}),
        ):
            with self.assertRaises(result_adapter.IsolatedWorkerError) as raised:
                result_adapter._run_gpu_worker_with_retry(
                    operation_name="fixture",
                    worker_slot=threading.BoundedSemaphore(1),
                    requested_device="cuda:0",
                    visible_devices_env="SEME2E_FIXTURE_CUDA_VISIBLE_DEVICES",
                    cancel_event=None,
                    operation=operation,
                    timeout_seconds=1,
                )

        self.assertIs(raised.exception, expected)
        self.assertEqual(calls, 1)

    def test_cpu_out_of_memory_is_not_misclassified_as_gpu_oom(self) -> None:
        self.assertIsNone(
            result_adapter._gpu_resource_error_kind(MemoryError("host allocator exhausted"))
        )
        self.assertIsNone(
            result_adapter._gpu_resource_error_kind(
                RuntimeError("out of memory while parsing a large CPU tensor")
            )
        )
        self.assertEqual(
            result_adapter._gpu_resource_error_kind(
                RuntimeError("GPU allocator failed: out of memory")
            ),
            "gpu_memory_exhausted",
        )

    def test_cosyvoice_cudnn_initialization_failure_is_retryable_on_another_gpu(self) -> None:
        error = result_adapter.IsolatedWorkerError(
            "CosyVoice2 worker failed",
            diagnostics={
                "stderrTail": (
                    "[ONNXRuntimeError] CUDNN failure 4: CUDNN_STATUS_INTERNAL_ERROR; "
                    "expr=cudnnCreate(&cudnn_handle_)"
                )
            },
        )

        self.assertEqual(
            result_adapter._gpu_resource_error_kind(error),
            "gpu_temporarily_unavailable",
        )

    def test_gpu_retry_wait_budget_defaults_to_existing_acquire_timeout(self) -> None:
        with (
            mock.patch.dict(
                result_adapter.os.environ,
                {"SEME2E_GPU_ACQUIRE_TIMEOUT_SECONDS": "0.05"},
                clear=False,
            ),
            mock.patch.object(result_adapter.time, "monotonic", return_value=100.0),
        ):
            result_adapter.os.environ.pop("SEME2E_GPU_RETRY_TIMEOUT_SECONDS", None)
            deadline, timeout = result_adapter._gpu_retry_deadline(300.0)

        self.assertEqual(timeout, 0.05)
        self.assertEqual(deadline, 100.05)

    def test_gpu_retry_attempt_timeout_is_limited_by_remaining_deadline(self) -> None:
        attempt_timeouts: list[float] = []

        @contextmanager
        def gpu_lease(*_args: object, **_kwargs: object):
            yield "cuda:0", {"CUDA_VISIBLE_DEVICES": "5"}, "5"

        def operation(
            _worker_device: str,
            _worker_env: dict[str, str] | None,
            _selected_gpu: str | None,
            attempt_timeout_seconds: float,
        ) -> dict[str, bool]:
            attempt_timeouts.append(attempt_timeout_seconds)
            return {"ok": True}

        with (
            mock.patch.object(result_adapter, "_worker_gpu_candidates", return_value=("5",)),
            mock.patch.object(result_adapter, "_gpu_retry_deadline", return_value=(110.0, 20.0)),
            mock.patch.object(result_adapter, "_isolated_worker_gpu_lease", side_effect=gpu_lease),
            mock.patch.object(result_adapter.time, "monotonic", side_effect=[100.0, 104.0]),
            mock.patch.object(
                result_adapter,
                "_nvidia_gpu_inventory",
                return_value=(("5",), {"5": 48000}, {"5": "gpu-5"}),
            ),
        ):
            response, selected_gpu, attempts = result_adapter._run_gpu_worker_with_retry(
                operation_name="fixture",
                worker_slot=threading.BoundedSemaphore(1),
                requested_device="cuda:0",
                visible_devices_env="SEME2E_FIXTURE_CUDA_VISIBLE_DEVICES",
                cancel_event=None,
                operation=operation,
                timeout_seconds=30.0,
            )

        self.assertEqual(response, {"ok": True})
        self.assertEqual(selected_gpu, "5")
        self.assertEqual(attempts, [])
        self.assertEqual(attempt_timeouts, [6.0])

    def test_wait_for_gpu_slot_change_honors_full_backoff_while_polling(self) -> None:
        clock = [100.0]
        waits: list[float] = []

        class FakeCondition:
            def __enter__(self) -> FakeCondition:
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def wait(self, timeout: float) -> bool:
                waits.append(timeout)
                clock[0] += timeout
                return False

        with (
            mock.patch.object(result_adapter, "GPU_SLOT_CONDITION", FakeCondition()),
            mock.patch.object(result_adapter.time, "monotonic", side_effect=lambda: clock[0]),
            mock.patch.dict(
                result_adapter.os.environ,
                {"SEME2E_GPU_WAIT_POLL_SECONDS": "1"},
                clear=False,
            ),
        ):
            result_adapter._wait_for_gpu_slot_change(
                None,
                110.0,
                maximum_wait_seconds=4.0,
            )

        self.assertEqual(waits, [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(clock[0], 104.0)

    def test_busy_gpu_retries_use_independent_exponential_cooldowns(self) -> None:
        clock = [100.0]
        waits: list[float] = []
        selected: list[str | None] = []

        @contextmanager
        def gpu_lease(*_args: object, **_kwargs: object):
            yield "cuda:0", {"CUDA_VISIBLE_DEVICES": "5"}, "5"

        def wait_for_slot(
            _cancel_event: object,
            _deadline: float,
            *,
            maximum_wait_seconds: float | None = None,
        ) -> None:
            self.assertIsNotNone(maximum_wait_seconds)
            waits.append(float(maximum_wait_seconds))
            clock[0] += float(maximum_wait_seconds)

        def operation(
            _worker_device: str,
            _worker_env: dict[str, str] | None,
            selected_gpu: str | None,
            _attempt_timeout_seconds: float,
        ) -> dict[str, bool]:
            selected.append(selected_gpu)
            if len(selected) < 3:
                raise RuntimeError("CUDA-capable device is busy or unavailable")
            return {"ok": True}

        with (
            mock.patch.object(result_adapter, "_worker_gpu_candidates", return_value=("5",)),
            mock.patch.object(result_adapter, "_gpu_retry_deadline", return_value=(120.0, 20.0)),
            mock.patch.object(result_adapter, "_isolated_worker_gpu_lease", side_effect=gpu_lease),
            mock.patch.object(result_adapter, "_wait_for_gpu_slot_change", side_effect=wait_for_slot),
            mock.patch.object(result_adapter.time, "monotonic", side_effect=lambda: clock[0]),
            mock.patch.object(
                result_adapter,
                "_nvidia_gpu_inventory",
                return_value=(("5",), {"5": 48000}, {"5": "gpu-5"}),
            ),
            mock.patch.dict(
                result_adapter.os.environ,
                {
                    "SEME2E_GPU_RETRY_BACKOFF_SECONDS": "1",
                    "SEME2E_GPU_RETRY_MAX_BACKOFF_SECONDS": "8",
                },
                clear=False,
            ),
        ):
            response, selected_gpu, attempts = result_adapter._run_gpu_worker_with_retry(
                operation_name="fixture",
                worker_slot=threading.BoundedSemaphore(1),
                requested_device="cuda:0",
                visible_devices_env="SEME2E_FIXTURE_CUDA_VISIBLE_DEVICES",
                cancel_event=None,
                operation=operation,
                timeout_seconds=10.0,
            )

        self.assertEqual(response, {"ok": True})
        self.assertEqual(selected_gpu, "5")
        self.assertEqual(selected, ["5", "5", "5"])
        self.assertEqual(waits, [1.0, 2.0])
        self.assertEqual(
            [attempt["retryCooldownSec"] for attempt in attempts],
            [1.0, 2.0],
        )

    def test_unknown_oom_memory_sample_waits_for_gpu_cooldown(self) -> None:
        clock = [100.0]
        failure_states = {
            "gpu-5": {"kind": "gpu_memory_exhausted", "freeMemoryMiB": None}
        }
        cooldown_deadlines = {"5": 102.0, "gpu-5": 102.0}

        with (
            mock.patch.object(result_adapter.time, "monotonic", side_effect=lambda: clock[0]),
            mock.patch.object(
                result_adapter,
                "_nvidia_gpu_inventory",
                return_value=(("5",), {}, {"5": "gpu-5"}),
            ),
        ):
            self.assertEqual(
                result_adapter._recovered_gpu_keys(
                    ("5",),
                    failure_states,
                    minimum_free_mib=0,
                    cooldown_deadlines=cooldown_deadlines,
                ),
                set(),
            )
            clock[0] = 102.0
            self.assertEqual(
                result_adapter._recovered_gpu_keys(
                    ("5",),
                    failure_states,
                    minimum_free_mib=0,
                    cooldown_deadlines=cooldown_deadlines,
                ),
                {"5", "gpu-5"},
            )

    def test_all_gpu_oom_waits_without_restart_storm_and_keeps_diagnostics(self) -> None:
        selected: list[str | None] = []

        def operation(
            _worker_device: str,
            _worker_env: dict[str, str] | None,
            selected_gpu: str | None,
            _attempt_timeout_seconds: float,
        ) -> dict[str, bool]:
            selected.append(selected_gpu)
            raise result_adapter.IsolatedWorkerError(
                "CUDA out of memory",
                diagnostics={"stderrTail": f"CUDA out of memory on {selected_gpu}"},
            )

        worker_slot = threading.BoundedSemaphore(1)
        with (
            mock.patch.object(result_adapter, "CLONE_GPU_SLOTS", {}),
            mock.patch.object(
                result_adapter,
                "_nvidia_gpu_inventory",
                return_value=(
                    ("5", "2"),
                    {"5": 48000, "2": 47000},
                    {"5": "gpu-5", "2": "gpu-2"},
                ),
            ),
            mock.patch.dict(
                result_adapter.os.environ,
                {
                    "SEME2E_GPU_POOL": "5,2",
                    "SEME2E_GPU_RETRY_TIMEOUT_SECONDS": "0.12",
                    "SEME2E_GPU_WAIT_POLL_SECONDS": "0.01",
                    "SEME2E_GPU_RETRY_BACKOFF_SECONDS": "0.05",
                },
                clear=False,
            ),
        ):
            with self.assertRaises(result_adapter.IsolatedWorkerError) as raised:
                result_adapter._run_gpu_worker_with_retry(
                    operation_name="fixture",
                    worker_slot=worker_slot,
                    requested_device="cuda:0",
                    visible_devices_env="SEME2E_FIXTURE_CUDA_VISIBLE_DEVICES",
                    cancel_event=None,
                    operation=operation,
                    timeout_seconds=10,
                )

        self.assertEqual(selected, ["5", "2"])
        self.assertEqual(raised.exception.diagnostics["gpuAttemptCount"], 2)
        self.assertEqual(len(raised.exception.diagnostics["gpuAttempts"]), 2)
        self.assertTrue(worker_slot.acquire(blocking=False))
        worker_slot.release()

    def test_gpt_lease_release_uses_key_saved_at_acquire_time(self) -> None:
        leases = {"gpu-5"}
        lease_keys = {"5": "gpu-5"}
        with (
            mock.patch.object(result_adapter, "GPT_SOVITS_GPU_LEASES", leases),
            mock.patch.object(result_adapter, "GPT_SOVITS_GPU_LEASE_KEYS", lease_keys),
            mock.patch.object(
                result_adapter,
                "_canonical_gpu_slot_key",
                side_effect=AssertionError("release must not query a new canonical key"),
            ),
            mock.patch.object(result_adapter, "_notify_gpu_slot_waiters"),
        ):
            result_adapter._release_gpt_sovits_gpu_lease("5")

        self.assertEqual(leases, set())
        self.assertEqual(lease_keys, {})

    def test_release_gpu_slot_notifies_waiter_without_waiting_for_poll_timeout(self) -> None:
        condition = threading.Condition()
        entered = threading.Event()
        errors: list[BaseException] = []
        acquired: list[str] = []

        with (
            mock.patch.object(result_adapter, "GPU_SLOT_CONDITION", condition),
            mock.patch.object(result_adapter, "CLONE_GPU_SLOTS", {}),
            mock.patch.object(
                result_adapter,
                "_nvidia_gpu_inventory",
                return_value=(("0",), {"0": 32000}, {"0": "gpu-0"}),
            ),
            mock.patch.dict(
                result_adapter.os.environ,
                {
                    "SEME2E_GPU_ACQUIRE_TIMEOUT_SECONDS": "2",
                    "SEME2E_GPU_WAIT_POLL_SECONDS": "5",
                },
            ),
        ):
            busy_slot = result_adapter._clone_gpu_slot("0")
            busy_slot.acquire()

            def wait_for_slot() -> None:
                entered.set()
                try:
                    gpu, slot = result_adapter.acquire_gpu_slot(("0",), None)
                    acquired.append(gpu)
                    result_adapter.release_gpu_slot(slot)
                except BaseException as exc:  # pragma: no cover - surfaced below
                    errors.append(exc)

            thread = threading.Thread(target=wait_for_slot)
            started = time.monotonic()
            thread.start()
            self.assertTrue(entered.wait(timeout=1))
            time.sleep(0.05)
            result_adapter.release_gpu_slot(busy_slot)
            thread.join(timeout=0.75)

        self.assertFalse(thread.is_alive())
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(acquired, ["0"])
        self.assertEqual(errors, [])

    def test_gpt_sovits_uses_independent_higher_free_memory_threshold(self) -> None:
        with (
            mock.patch.object(result_adapter, "CLONE_GPU_SLOTS", {}),
            mock.patch.object(
                result_adapter,
                "_nvidia_gpu_inventory",
                return_value=(("5",), {"5": 20000}, {"5": "gpu-5"}),
            ),
            mock.patch.dict(
                result_adapter.os.environ,
                {
                    "SEME2E_GPU_MIN_FREE_MIB": "0",
                    "SEME2E_GPT_SOVITS_GPU_MIN_FREE_MIB": "24576",
                    "SEME2E_GPU_ACQUIRE_TIMEOUT_SECONDS": "0.06",
                    "SEME2E_GPU_WAIT_POLL_SECONDS": "0.01",
                },
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "等待可用 GPU 超时"):
                result_adapter._acquire_gpt_sovits_gpu_resources(("5",), None)

    def test_gpt_sovits_retries_on_another_gpu_and_removes_fine_tune_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            clone_dir = Path(temporary)
            original_output = clone_dir / "original_clone.wav"
            protected_output = clone_dir / "protected_clone.wav"
            work_dir = clone_dir / "fine_tune"
            selected_gpus: list[str] = []
            worker_timeouts: list[float] = []

            def run_worker(command: list[str], **kwargs: object) -> mock.Mock:
                selected_gpu = str(command[command.index("--cuda-visible-devices") + 1])
                selected_gpus.append(selected_gpu)
                worker_timeouts.append(float(kwargs["timeout_seconds"]))
                self.assertFalse(work_dir.exists())
                checkpoint = work_dir / "original" / "checkpoint" / "fixture.ckpt"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"checkpoint")
                if len(selected_gpus) == 1:
                    return mock.Mock(
                        returncode=1,
                        stdout="",
                        stderr="CUDA out of memory while training GPT-SoVITS",
                    )
                payload = {
                    "mode": "live_fine_tune",
                    "workDir": str(work_dir),
                    "original": {
                        "gptCheckpoint": str(checkpoint),
                        "outputPath": str(original_output),
                    },
                    "protected": {
                        "sovitsCheckpoint": str(work_dir / "protected" / "checkpoint" / "fixture.pth"),
                        "outputPath": str(protected_output),
                    },
                }
                return mock.Mock(
                    returncode=0,
                    stdout="VOICE_SHIELD_GPT_SOVITS_LIVE_RESULT=" + json.dumps(payload) + "\n",
                    stderr="",
                )

            runtime_context: dict[str, object] = {}
            with (
                mock.patch.object(result_adapter, "_gpt_sovits_model_status", return_value=("available", None, None)),
                mock.patch.object(result_adapter, "_run_cancellable_subprocess", side_effect=run_worker),
                mock.patch.object(result_adapter, "CLONE_GPU_SLOTS", {}),
                mock.patch.object(result_adapter, "GPT_SOVITS_GPU_LEASES", set()),
                mock.patch.object(
                    result_adapter,
                    "_nvidia_gpu_inventory",
                    return_value=(
                        ("5", "2"),
                        {"5": 48000, "2": 47000},
                        {"5": "gpu-5", "2": "gpu-2"},
                    ),
                ),
                mock.patch.dict(
                    result_adapter.os.environ,
                    {
                        "SEME2E_GPT_SOVITS_GPU_POOL": "5,2",
                        "SEME2E_GPT_SOVITS_GPU_MIN_FREE_MIB": "0",
                        "SEME2E_GPU_RETRY_TIMEOUT_SECONDS": "2",
                        "SEME2E_KEEP_GPT_SOVITS_WORK_DIR": "0",
                    },
                ),
            ):
                response = result_adapter._gpt_sovits_clone_pair(
                    Path("original.wav"),
                    Path("protected.wav"),
                    original_output,
                    protected_output,
                    original_transcript="original transcript",
                    protected_transcript="protected transcript",
                    text="target text",
                    language="en",
                    speed=1.0,
                    device="cuda:0",
                    runtime_context=runtime_context,
                )

            self.assertEqual(selected_gpus, ["5", "2"])
            self.assertFalse(work_dir.exists())
            self.assertIsNone(response["workDir"])
            self.assertIsNone(response["original"]["gptCheckpoint"])
            self.assertEqual(response["original"]["outputPath"], str(original_output))
            self.assertFalse(response["workDirRetained"])
            self.assertEqual(response["gpuAttempts"][0]["reason"], "gpu_memory_exhausted")
            self.assertEqual(runtime_context["gpuKey"], "2")
            self.assertTrue(all(0 < timeout <= 2.0 for timeout in worker_timeouts))

    def test_expensive_clone_worker_concurrency_defaults(self) -> None:
        if "SEME2E_COSYVOICE_WORKER_MAX_CONCURRENCY" not in result_adapter.os.environ:
            self.assertEqual(result_adapter.COSYVOICE_WORKER_MAX_CONCURRENCY, 1)
        if "SEME2E_GPT_SOVITS_WORKER_MAX_CONCURRENCY" not in result_adapter.os.environ:
            self.assertEqual(result_adapter.GPT_SOVITS_WORKER_MAX_CONCURRENCY, 2)
        if "SEME2E_CLONE_GPU_MAX_CONCURRENCY" not in result_adapter.os.environ:
            self.assertEqual(result_adapter.CLONE_GPU_MAX_CONCURRENCY, 1)

    def test_gpt_sovits_dynamic_candidates_are_not_truncated_to_worker_limit(self) -> None:
        with (
            mock.patch.object(result_adapter, "GPT_SOVITS_WORKER_MAX_CONCURRENCY", 2),
            mock.patch.dict(
                result_adapter.os.environ,
                {
                    "SEME2E_GPT_SOVITS_GPU_POOL": "5,2,7",
                    "SEME2E_GPT_SOVITS_CUDA_VISIBLE_DEVICES": "",
                },
            ),
        ):
            self.assertEqual(
                result_adapter._gpt_sovits_gpu_candidates("cuda:0"),
                ("5", "2", "7"),
            )

    def test_gpt_sovits_gpu_pool_runs_two_distinct_leases_and_queues_the_third(self) -> None:
        lease_condition = threading.Condition()
        leases: set[str] = set()
        entered = {name: threading.Event() for name in ("first", "second", "third")}
        releases = {name: threading.Event() for name in entered}
        assignments: dict[str, str] = {}
        errors: list[BaseException] = []

        def occupy(name: str) -> None:
            try:
                with result_adapter._gpt_sovits_gpu_resource_lease("cuda:0", None) as leased_gpu:
                    assignments[name] = leased_gpu
                    entered[name].set()
                    releases[name].wait(timeout=3)
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        threads: list[threading.Thread] = []
        with (
            mock.patch.object(result_adapter, "GPT_SOVITS_WORKER_MAX_CONCURRENCY", 2),
            mock.patch.object(result_adapter, "CLONE_GPU_MAX_CONCURRENCY", 1),
            mock.patch.object(result_adapter, "CLONE_GPU_SLOTS", {}),
            mock.patch.object(result_adapter, "GPT_SOVITS_GPU_LEASE_CONDITION", lease_condition),
            mock.patch.object(result_adapter, "GPT_SOVITS_GPU_LEASES", leases),
            mock.patch.dict(
                result_adapter.os.environ,
                {
                    "SEME2E_GPT_SOVITS_GPU_POOL": "5,2",
                    "SEME2E_GPT_SOVITS_CUDA_VISIBLE_DEVICES": "",
                },
            ),
        ):
            try:
                for name in ("first", "second"):
                    thread = threading.Thread(target=occupy, args=(name,))
                    threads.append(thread)
                    thread.start()
                    self.assertTrue(entered[name].wait(timeout=1))

                self.assertEqual({assignments["first"], assignments["second"]}, {"5", "2"})
                self.assertEqual(leases, {"5", "2"})

                third_thread = threading.Thread(target=occupy, args=("third",))
                threads.append(third_thread)
                third_thread.start()
                self.assertFalse(entered["third"].wait(timeout=0.15))

                releases["first"].set()
                self.assertTrue(entered["third"].wait(timeout=1))
                self.assertEqual(assignments["third"], assignments["first"])
            finally:
                for release in releases.values():
                    release.set()
                for thread in threads:
                    thread.join(timeout=2)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(leases, set())

    def test_gpt_sovits_gpu_pool_wait_honors_cancellation_without_leaking(self) -> None:
        lease_condition = threading.Condition()
        leases: set[str] = set()
        cancel_event = threading.Event()
        timer = threading.Timer(0.05, cancel_event.set)
        with (
            mock.patch.object(result_adapter, "GPT_SOVITS_WORKER_MAX_CONCURRENCY", 2),
            mock.patch.object(result_adapter, "CLONE_GPU_MAX_CONCURRENCY", 1),
            mock.patch.object(result_adapter, "CLONE_GPU_SLOTS", {}),
            mock.patch.object(result_adapter, "GPT_SOVITS_GPU_LEASE_CONDITION", lease_condition),
            mock.patch.object(result_adapter, "GPT_SOVITS_GPU_LEASES", leases),
            mock.patch.dict(
                result_adapter.os.environ,
                {
                    "SEME2E_GPT_SOVITS_GPU_POOL": "5,2",
                    "SEME2E_GPT_SOVITS_CUDA_VISIBLE_DEVICES": "",
                },
            ),
        ):
            first, first_slot = result_adapter._acquire_gpt_sovits_gpu_resources(("5", "2"), None)
            second, second_slot = result_adapter._acquire_gpt_sovits_gpu_resources(("5", "2"), None)
            timer.start()
            try:
                with self.assertRaisesRegex(RuntimeError, "TASK_CANCELLED"):
                    result_adapter._acquire_gpt_sovits_gpu_resources(("5", "2"), cancel_event)
            finally:
                timer.cancel()
                second_slot.release()
                result_adapter._release_gpt_sovits_gpu_lease(second)
                first_slot.release()
                result_adapter._release_gpt_sovits_gpu_lease(first)

        self.assertTrue(cancel_event.is_set())
        self.assertEqual(leases, set())

    def test_gpt_sovits_shared_clone_gpu_slot_wait_honors_cancellation_without_leaking(self) -> None:
        lease_condition = threading.Condition()
        leases: set[str] = set()
        shared_gpu_slot = threading.BoundedSemaphore(1)
        shared_gpu_slot.acquire()
        cancel_event = threading.Event()
        timer = threading.Timer(0.05, cancel_event.set)
        timer.start()
        try:
            with (
                mock.patch.object(result_adapter, "_gpt_sovits_model_status", return_value=("available", None, None)),
                mock.patch.object(result_adapter, "_clone_gpu_slot", return_value=shared_gpu_slot),
                mock.patch.object(result_adapter, "_run_cancellable_subprocess") as runner,
                mock.patch.object(result_adapter, "GPT_SOVITS_WORKER_MAX_CONCURRENCY", 2),
                mock.patch.object(result_adapter, "GPT_SOVITS_GPU_LEASE_CONDITION", lease_condition),
                mock.patch.object(result_adapter, "GPT_SOVITS_GPU_LEASES", leases),
                mock.patch.dict(
                    result_adapter.os.environ,
                    {
                        "SEME2E_GPT_SOVITS_GPU_POOL": "5,2",
                        "SEME2E_GPT_SOVITS_CUDA_VISIBLE_DEVICES": "",
                    },
                ),
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
                        cancel_event=cancel_event,
                    )
                runner.assert_not_called()
        finally:
            timer.cancel()
            shared_gpu_slot.release()

        self.assertTrue(cancel_event.is_set())
        self.assertEqual(leases, set())
        self.assertTrue(shared_gpu_slot.acquire(blocking=False))
        shared_gpu_slot.release()

    def test_gpt_sovits_gpu_lease_releases_after_worker_exception_and_cancellation(self) -> None:
        lease_condition = threading.Condition()
        leases: set[str] = set()
        for message in ("synthetic worker failure", "TASK_CANCELLED"):
            with self.subTest(message=message):
                with (
                    mock.patch.object(result_adapter, "_gpt_sovits_model_status", return_value=("available", None, None)),
                    mock.patch.object(result_adapter, "_run_cancellable_subprocess", side_effect=RuntimeError(message)),
                    mock.patch.object(result_adapter, "GPT_SOVITS_WORKER_MAX_CONCURRENCY", 2),
                    mock.patch.object(result_adapter, "GPT_SOVITS_GPU_LEASE_CONDITION", lease_condition),
                    mock.patch.object(result_adapter, "GPT_SOVITS_GPU_LEASES", leases),
                    mock.patch.dict(
                        result_adapter.os.environ,
                        {
                            "SEME2E_GPT_SOVITS_GPU_POOL": "5,2",
                            "SEME2E_GPT_SOVITS_CUDA_VISIBLE_DEVICES": "",
                        },
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
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
                        )
                self.assertEqual(leases, set())

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

    def test_dynamic_worker_pool_prefers_more_free_gpu_without_pinning_post_stage(self) -> None:
        inventory = (("0", "1"), {"0": 4096, "1": 24576})
        with (
            mock.patch.object(result_adapter, "CLONE_GPU_SLOTS", {}),
            mock.patch.object(result_adapter, "_nvidia_gpu_inventory", return_value=inventory),
            mock.patch.dict(
                result_adapter.os.environ,
                {
                    "SEME2E_GPU_POOL": "0,1",
                    "SEME2E_ASR_CUDA_VISIBLE_DEVICES": "",
                },
            ),
        ):
            with result_adapter._isolated_worker_gpu_lease(
                threading.BoundedSemaphore(1),
                "cuda:0",
                "SEME2E_ASR_CUDA_VISIBLE_DEVICES",
                None,
                preferred_gpu="0",
            ) as (worker_device, worker_env, selected_gpu):
                self.assertEqual(worker_device, "cuda:0")
                self.assertEqual(selected_gpu, "1")
                self.assertEqual(
                    worker_env,
                    {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1"},
                )

    def test_dynamic_worker_wait_below_memory_threshold_can_cancel_without_slot_leak(self) -> None:
        model_slot = threading.BoundedSemaphore(1)
        cancel_event = threading.Event()
        timer = threading.Timer(0.05, cancel_event.set)
        inventory = (("0",), {"0": 1024})
        with (
            mock.patch.object(result_adapter, "CLONE_GPU_MAX_CONCURRENCY", 1),
            mock.patch.object(result_adapter, "CLONE_GPU_SLOTS", {}),
            mock.patch.object(result_adapter, "_nvidia_gpu_inventory", return_value=inventory),
            mock.patch.dict(
                result_adapter.os.environ,
                {
                    "SEME2E_GPU_POOL": "0",
                    "SEME2E_ASR_CUDA_VISIBLE_DEVICES": "",
                },
            ),
        ):
            timer.start()
            try:
                with self.assertRaisesRegex(RuntimeError, "TASK_CANCELLED"):
                    with result_adapter._isolated_worker_gpu_lease(
                        model_slot,
                        "cuda:0",
                        "SEME2E_ASR_CUDA_VISIBLE_DEVICES",
                        cancel_event,
                        minimum_free_mib=12000,
                    ):
                        self.fail("worker must not enter below the free-memory threshold")
            finally:
                timer.cancel()

            self.assertTrue(model_slot.acquire(blocking=False))
            model_slot.release()
            gpu_slot = result_adapter._clone_gpu_slot("0")
            self.assertTrue(gpu_slot.acquire(blocking=False))
            gpu_slot.release()

    def test_explicit_single_gpu_route_is_honored_even_below_dynamic_threshold(self) -> None:
        inventory = (("4", "5"), {"4": 512, "5": 46000})
        with (
            mock.patch.object(result_adapter, "CLONE_GPU_SLOTS", {}),
            mock.patch.object(result_adapter, "_nvidia_gpu_inventory", return_value=inventory),
            mock.patch.dict(
                result_adapter.os.environ,
                {
                    "SEME2E_GPU_POOL": "5",
                    "SEME2E_ASR_CUDA_VISIBLE_DEVICES": "4",
                },
            ),
        ):
            with result_adapter._isolated_worker_gpu_lease(
                threading.BoundedSemaphore(1),
                "cuda:0",
                "SEME2E_ASR_CUDA_VISIBLE_DEVICES",
                None,
                minimum_free_mib=12000,
            ) as (_, worker_env, selected_gpu):
                self.assertEqual(selected_gpu, "4")
                self.assertEqual(worker_env["CUDA_VISIBLE_DEVICES"], "4")

    def test_clone_post_asr_waits_for_the_same_shared_physical_gpu_slot(self) -> None:
        worker_response = {
            "ok": True,
            "model": "openai-whisper:base",
            "language": "en",
            "originalText": "original transcript",
            "protectedText": "protected transcript",
        }
        worker_called = threading.Event()
        response_holder: list[dict[str, object]] = []

        def fake_worker(*args: object, **kwargs: object) -> dict[str, object]:
            worker_called.set()
            return worker_response

        with (
            mock.patch.object(result_adapter, "CLONE_GPU_MAX_CONCURRENCY", 1),
            mock.patch.object(result_adapter, "CLONE_GPU_SLOTS", {}),
            mock.patch.object(result_adapter, "ASR_WORKER_SLOTS", threading.BoundedSemaphore(1)),
            mock.patch.object(result_adapter, "_run_isolated_json_worker", side_effect=fake_worker),
            mock.patch.dict(
                result_adapter.os.environ,
                {
                    "SEME2E_API_DEVICE": "cuda:0",
                    "SEME2E_CLONE_ASR_CUDA_VISIBLE_DEVICES": "4",
                },
            ),
        ):
            busy_slot = result_adapter._clone_gpu_slot("4")
            busy_slot.acquire()

            def transcribe() -> None:
                response_holder.append(
                    result_adapter._transcribe_clone_pair_isolated(
                        Path("original_clone.wav"),
                        Path("protected_clone.wav"),
                        {"asrModel": "openai-whisper:base", "language": "en", "text": "target"},
                    )
                )

            thread = threading.Thread(target=transcribe)
            thread.start()
            try:
                self.assertFalse(worker_called.wait(timeout=0.15))
            finally:
                busy_slot.release()
            self.assertTrue(worker_called.wait(timeout=1))
            thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        self.assertEqual(response_holder[0]["status"], "available")

    def test_gpt_resource_selection_skips_busy_highest_free_gpu(self) -> None:
        lease_condition = threading.Condition()
        leases: set[str] = set()
        inventory = (("5", "2"), {"5": 46000, "2": 42000})
        with (
            mock.patch.object(result_adapter, "CLONE_GPU_MAX_CONCURRENCY", 1),
            mock.patch.object(result_adapter, "CLONE_GPU_SLOTS", {}),
            mock.patch.object(result_adapter, "GPT_SOVITS_GPU_LEASE_CONDITION", lease_condition),
            mock.patch.object(result_adapter, "GPT_SOVITS_GPU_LEASES", leases),
            mock.patch.object(result_adapter, "_nvidia_gpu_inventory", return_value=inventory),
            mock.patch.dict(
                result_adapter.os.environ,
                {
                    "SEME2E_GPT_SOVITS_GPU_POOL": "5,2",
                    "SEME2E_GPU_MIN_FREE_MIB": "0",
                },
            ),
        ):
            busy_slot = result_adapter._clone_gpu_slot("5")
            busy_slot.acquire()
            try:
                selected_gpu, selected_slot = result_adapter._acquire_gpt_sovits_gpu_resources(("5", "2"), None)
                self.assertEqual(selected_gpu, "2")
                selected_slot.release()
                result_adapter._release_gpt_sovits_gpu_lease(selected_gpu)
            finally:
                busy_slot.release()

        self.assertEqual(leases, set())

    def test_clone_semantic_worker_uses_dynamic_pool_and_logical_cuda_zero(self) -> None:
        worker_response = {
            "ok": True,
            "metrics": {"status": "available", "tokenChangeRate": 0.7, "semanticDrift": 0.4},
        }
        with (
            mock.patch.object(result_adapter, "CLONE_GPU_SLOTS", {}),
            mock.patch.object(result_adapter, "SEMANTIC_WORKER_SLOTS", threading.BoundedSemaphore(1)),
            mock.patch.object(result_adapter, "_run_isolated_json_worker", return_value=worker_response) as worker,
            mock.patch.dict(
                result_adapter.os.environ,
                {
                    "SEME2E_API_DEVICE": "cuda:0",
                    "SEME2E_GPU_POOL": "3",
                    "SEME2E_SEMANTIC_CUDA_VISIBLE_DEVICES": "",
                    "SEME2E_TOKENIZER_DEVICE": "",
                    "SEME2E_SEMANTIC_ENCODER_DEVICE": "",
                },
            ),
        ):
            response = result_adapter._compute_clone_semantic_isolated(
                Path("original_clone.wav"),
                Path("protected_clone.wav"),
                {},
            )

        self.assertEqual(response["status"], "available")
        self.assertEqual(
            worker.call_args.kwargs["env_overrides"],
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "3",
                "SEME2E_API_DEVICE": "cuda:0",
                "SEME2E_SEMANTIC_DEVICE": "cuda:0",
                "SEME2E_TOKENIZER_DEVICE": "cuda:0",
                "SEME2E_SEMANTIC_ENCODER_DEVICE": "cuda:0",
            },
        )

    def test_clone_semantic_worker_retries_embedded_cuda_oom_on_another_gpu(self) -> None:
        selected_gpus: list[str] = []

        def worker_response(*_args: object, **kwargs: object) -> dict[str, object]:
            selected_gpu = str(kwargs["env_overrides"]["CUDA_VISIBLE_DEVICES"])
            selected_gpus.append(selected_gpu)
            if selected_gpu == "4":
                return {
                    "ok": True,
                    "metrics": {
                        "status": "error",
                        "tokenChangeRate": 0.9462,
                        "semanticDrift": None,
                        "error": "OutOfMemoryError: CUDA out of memory while loading the semantic encoder",
                    },
                }
            return {
                "ok": True,
                "metrics": {
                    "status": "available",
                    "tokenChangeRate": 0.9462,
                    "semanticDrift": 0.73,
                },
            }

        with (
            mock.patch.object(result_adapter, "CLONE_GPU_MAX_CONCURRENCY", 1),
            mock.patch.object(result_adapter, "CLONE_GPU_SLOTS", {}),
            mock.patch.object(result_adapter, "SEMANTIC_WORKER_SLOTS", threading.BoundedSemaphore(1)),
            mock.patch.object(result_adapter, "_run_isolated_json_worker", side_effect=worker_response),
            mock.patch.object(
                result_adapter,
                "_nvidia_gpu_inventory",
                return_value=(("4", "1"), {"4": 48000, "1": 47000}, {"4": "gpu-4", "1": "gpu-1"}),
            ),
            mock.patch.dict(
                result_adapter.os.environ,
                {
                    "SEME2E_API_DEVICE": "cuda:0",
                    "SEME2E_GPU_POOL": "4,1",
                    "SEME2E_SEMANTIC_CUDA_VISIBLE_DEVICES": "",
                    "SEME2E_SEMANTIC_DEVICE": "",
                    "SEME2E_TOKENIZER_DEVICE": "",
                    "SEME2E_SEMANTIC_ENCODER_DEVICE": "",
                    "SEME2E_GPU_RETRY_TIMEOUT_SECONDS": "1",
                    "SEME2E_GPU_WAIT_POLL_SECONDS": "0.01",
                },
            ),
        ):
            response = result_adapter._compute_clone_semantic_isolated(
                Path("original_clone.wav"),
                Path("protected_clone.wav"),
                {},
                preferred_gpu="4",
            )

        self.assertEqual(selected_gpus, ["4", "1"])
        self.assertEqual(response["status"], "available")
        self.assertEqual(response["semanticDrift"], 0.73)
        self.assertEqual(response["gpu"], "1")
        self.assertEqual(response["gpuAttempts"][0]["reason"], "gpu_memory_exhausted")

    def test_clone_semantic_worker_default_memory_floor_waits_for_four_gib(self) -> None:
        worker_response = {
            "ok": True,
            "metrics": {"status": "available", "tokenChangeRate": 0.7, "semanticDrift": 0.4},
        }
        with (
            mock.patch.object(result_adapter, "SEMANTIC_WORKER_SLOTS", threading.BoundedSemaphore(1)),
            mock.patch.object(result_adapter, "_run_gpu_worker_with_retry", return_value=(worker_response, "2", [])) as retry,
            mock.patch.dict(
                result_adapter.os.environ,
                {
                    "SEME2E_API_DEVICE": "cuda:0",
                    "SEME2E_SEMANTIC_GPU_MIN_FREE_MIB": "",
                    "SEME2E_GPU_MIN_FREE_MIB": "0",
                },
            ),
        ):
            response = result_adapter._compute_clone_semantic_isolated(
                Path("original_clone.wav"),
                Path("protected_clone.wav"),
                {},
            )

        self.assertEqual(response["status"], "available")
        self.assertEqual(retry.call_args.kwargs["minimum_free_mib"], 4096)

    def test_clone_semantic_worker_maps_explicit_cuda_devices_to_child_cuda_zero(self) -> None:
        worker_response = {
            "ok": True,
            "metrics": {"status": "available", "tokenChangeRate": 0.7, "semanticDrift": 0.4},
        }
        inventory = (
            ("0", "1", "2"),
            {"0": 1000, "1": 2000, "2": 3000},
            {"0": "gpu-0", "1": "gpu-1", "2": "gpu-2"},
        )
        with (
            mock.patch.object(result_adapter, "CLONE_GPU_SLOTS", {}),
            mock.patch.object(result_adapter, "SEMANTIC_WORKER_SLOTS", threading.BoundedSemaphore(1)),
            mock.patch.object(result_adapter, "_nvidia_gpu_inventory", return_value=inventory),
            mock.patch.object(result_adapter, "_run_isolated_json_worker", return_value=worker_response) as worker,
            mock.patch.dict(
                result_adapter.os.environ,
                {
                    "SEME2E_API_DEVICE": "cuda:0",
                    "SEME2E_GPU_POOL": "0,1,2",
                    "SEME2E_SEMANTIC_DEVICE": "cuda:2",
                    "SEME2E_SEMANTIC_CUDA_VISIBLE_DEVICES": "",
                    "SEME2E_TOKENIZER_DEVICE": "cuda:7",
                    "SEME2E_SEMANTIC_ENCODER_DEVICE": "cuda:9",
                },
            ),
        ):
            response = result_adapter._compute_clone_semantic_isolated(
                Path("original_clone.wav"),
                Path("protected_clone.wav"),
                {},
            )

        self.assertEqual(response["status"], "available")
        child_env = worker.call_args.kwargs["env_overrides"]
        self.assertEqual(child_env["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(child_env["SEME2E_API_DEVICE"], "cuda:0")
        self.assertEqual(child_env["SEME2E_SEMANTIC_DEVICE"], "cuda:0")
        self.assertEqual(child_env["SEME2E_TOKENIZER_DEVICE"], "cuda:0")
        self.assertEqual(child_env["SEME2E_SEMANTIC_ENCODER_DEVICE"], "cuda:0")

    def test_clone_semantic_worker_preserves_explicit_cpu_devices(self) -> None:
        worker_response = {
            "ok": True,
            "metrics": {"status": "available", "tokenChangeRate": 0.7, "semanticDrift": 0.4},
        }
        with (
            mock.patch.object(result_adapter, "SEMANTIC_WORKER_SLOTS", threading.BoundedSemaphore(1)),
            mock.patch.object(result_adapter, "_run_isolated_json_worker", return_value=worker_response) as worker,
            mock.patch.dict(
                result_adapter.os.environ,
                {
                    "SEME2E_API_DEVICE": "cuda:0",
                    "SEME2E_SEMANTIC_DEVICE": "cpu",
                    "SEME2E_TOKENIZER_DEVICE": "cpu",
                    "SEME2E_SEMANTIC_ENCODER_DEVICE": "cpu",
                },
            ),
        ):
            response = result_adapter._compute_clone_semantic_isolated(
                Path("original_clone.wav"),
                Path("protected_clone.wav"),
                {},
            )

        self.assertEqual(response["status"], "available")
        child_env = worker.call_args.kwargs["env_overrides"]
        self.assertNotIn("CUDA_VISIBLE_DEVICES", child_env)
        self.assertEqual(child_env["SEME2E_API_DEVICE"], "cpu")
        self.assertEqual(child_env["SEME2E_SEMANTIC_DEVICE"], "cpu")
        self.assertEqual(child_env["SEME2E_TOKENIZER_DEVICE"], "cpu")
        self.assertEqual(child_env["SEME2E_SEMANTIC_ENCODER_DEVICE"], "cpu")

    def test_gpu_index_and_uuid_share_one_physical_slot_and_capacity_key(self) -> None:
        inventory = (
            ("0", "1"),
            {"0": 32000, "GPU-AAAA": 32000, "1": 24000, "GPU-BBBB": 24000},
            {
                "0": "gpu-aaaa",
                "GPU-AAAA": "gpu-aaaa",
                "gpu-aaaa": "gpu-aaaa",
                "1": "gpu-bbbb",
                "GPU-BBBB": "gpu-bbbb",
                "gpu-bbbb": "gpu-bbbb",
            },
        )
        with (
            mock.patch.object(result_adapter, "CLONE_GPU_MAX_CONCURRENCY", 1),
            mock.patch.object(result_adapter, "CLONE_GPU_SLOTS", {}),
            mock.patch.object(result_adapter, "_nvidia_gpu_inventory", return_value=inventory),
        ):
            self.assertIs(result_adapter._clone_gpu_slot("0"), result_adapter._clone_gpu_slot("GPU-AAAA"))
            maximum = result_adapter.maximum_gpu_worker_concurrency(
                worker_limits={"asr": 1, "clone": 1},
                worker_gpu_keys={"asr": ("0",), "clone": ("GPU-AAAA",)},
                gpu_slot_limit=1,
            )
            with result_adapter._isolated_worker_gpu_lease(
                threading.BoundedSemaphore(1),
                "cuda:0",
                "SEME2E_ASR_CUDA_VISIBLE_DEVICES",
                None,
            ) as (_, worker_env, selected_gpu):
                self.assertEqual(selected_gpu, "0")
                self.assertEqual(worker_env["CUDA_VISIBLE_DEVICES"], "0")

        self.assertEqual(maximum, 1)

    def test_nvidia_inventory_slow_query_is_single_flight(self) -> None:
        query_started = threading.Event()
        release_query = threading.Event()
        query_calls = 0
        query_guard = threading.Lock()
        results: list[tuple[tuple[str, ...], dict[str, int], dict[str, str]]] = []

        def slow_query() -> tuple[tuple[str, ...], dict[str, int], dict[str, str]]:
            nonlocal query_calls
            with query_guard:
                query_calls += 1
            query_started.set()
            release_query.wait(timeout=1)
            return (("0",), {"0": 32000}, {"0": "gpu-0"})

        with (
            mock.patch.object(result_adapter, "GPU_INVENTORY_CACHE_AT", 0.0),
            mock.patch.object(result_adapter, "GPU_INVENTORY_CACHE", ((), {}, {})),
            mock.patch.object(result_adapter, "_query_nvidia_gpu_inventory", side_effect=slow_query),
            mock.patch.dict(result_adapter.os.environ, {"SEME2E_GPU_INVENTORY_CACHE_SECONDS": "2"}),
        ):
            threads = [
                threading.Thread(target=lambda: results.append(result_adapter._nvidia_gpu_inventory()))
                for _ in range(4)
            ]
            for thread in threads:
                thread.start()
            self.assertTrue(query_started.wait(timeout=1))
            time.sleep(0.03)
            release_query.set()
            for thread in threads:
                thread.join(timeout=1)

        self.assertEqual(query_calls, 1)
        self.assertEqual(len(results), 4)

    def test_nvidia_inventory_maps_index_and_uuid_to_one_canonical_key(self) -> None:
        completed = mock.Mock(
            returncode=0,
            stdout="0, GPU-AAAA, 32768\n1, GPU-BBBB, 24576\n",
            stderr="",
        )
        with mock.patch.object(result_adapter.subprocess, "run", return_value=completed):
            indices, free_memory, canonical_keys = result_adapter._query_nvidia_gpu_inventory()

        self.assertEqual(indices, ("0", "1"))
        self.assertEqual(free_memory["0"], 32768)
        self.assertEqual(free_memory["GPU-AAAA"], 32768)
        self.assertEqual(canonical_keys["0"], "gpu-aaaa")
        self.assertEqual(canonical_keys["GPU-AAAA"], "gpu-aaaa")

    def test_gpu_acquire_timeout_is_plain_and_inventory_queries_are_ttl_throttled(self) -> None:
        query_calls = 0

        def query() -> tuple[tuple[str, ...], dict[str, int], dict[str, str]]:
            nonlocal query_calls
            query_calls += 1
            return (("0",), {"0": 32000}, {"0": "gpu-0"})

        with (
            mock.patch.object(result_adapter, "CLONE_GPU_MAX_CONCURRENCY", 1),
            mock.patch.object(result_adapter, "CLONE_GPU_SLOTS", {}),
            mock.patch.object(result_adapter, "GPU_INVENTORY_CACHE_AT", 0.0),
            mock.patch.object(result_adapter, "GPU_INVENTORY_CACHE", ((), {}, {})),
            mock.patch.object(result_adapter, "_query_nvidia_gpu_inventory", side_effect=query),
            mock.patch.dict(
                result_adapter.os.environ,
                {
                    "SEME2E_GPU_ACQUIRE_TIMEOUT_SECONDS": "0.08",
                    "SEME2E_GPU_INVENTORY_CACHE_SECONDS": "1",
                },
            ),
        ):
            busy_slot = result_adapter._clone_gpu_slot("0")
            busy_slot.acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, "等待可用 GPU 超时"):
                    result_adapter._acquire_best_gpu_slot(("0",), None)
            finally:
                busy_slot.release()

        self.assertEqual(query_calls, 1)

    def test_capacity_snapshot_uses_tts_device_for_both_coqui_and_cosyvoice(self) -> None:
        with mock.patch.dict(
            result_adapter.os.environ,
            {
                "SEME2E_API_DEVICE": "cuda:0",
                "SEME2E_TTS_DEVICE": "cuda:1",
                "SEME2E_GPU_POOL": "5,2",
                "SEME2E_COQUI_TTS_CUDA_VISIBLE_DEVICES": "",
                "SEME2E_COSYVOICE_CUDA_VISIBLE_DEVICES": "",
                "SEME2E_GPT_SOVITS_GPU_POOL": "",
                "SEME2E_GPT_SOVITS_CUDA_VISIBLE_DEVICES": "",
            },
        ):
            snapshot = result_adapter.clone_worker_capacity_snapshot()

        self.assertEqual(snapshot["gpuKeys"]["coquiTts"], ["2"])
        self.assertEqual(snapshot["gpuKeys"]["cosyVoice"], ["2"])

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

    def test_cosyvoice_cancel_while_waiting_for_worker_slot(self) -> None:
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
