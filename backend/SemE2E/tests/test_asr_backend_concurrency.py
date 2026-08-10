from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asr_backends
import result_adapter


class AsrBackendConcurrencyTest(unittest.TestCase):
    def test_openai_whisper_tiny_and_base_sessions_do_not_overlap(self) -> None:
        state_lock = threading.Lock()
        start_barrier = threading.Barrier(2)
        active = 0
        max_active = 0
        events: list[str] = []

        def evaluate(model_name: str) -> None:
            nonlocal active, max_active
            start_barrier.wait(timeout=3)
            with asr_backends.openai_whisper_session(model_name):
                with state_lock:
                    active += 1
                    max_active = max(max_active, active)
                    events.append(model_name)
                time.sleep(0.02)
                with state_lock:
                    active -= 1

        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(evaluate, ["openai-whisper:tiny", "openai-whisper:base"]))

        self.assertEqual(max_active, 1)
        self.assertIn(events, [
            ["openai-whisper:tiny", "openai-whisper:base"],
            ["openai-whisper:base", "openai-whisper:tiny"],
        ])

    def test_maybe_asr_eval_uses_isolated_worker_transcriptions(self) -> None:
        clean_path = Path("clean.wav")
        protected_path = Path("protected.wav")
        worker_response = {
            "ok": True,
            "model": "openai-whisper:base",
            "language": "en",
            "originalText": "the original transcript",
            "protectedText": "the changed transcript",
        }

        with (
            mock.patch.object(result_adapter, "_run_isolated_json_worker", return_value=worker_response) as worker,
            mock.patch.object(asr_backends, "ASRTranscriber") as main_process_transcriber,
            mock.patch.dict(
                os.environ,
                {
                    "SEME2E_API_DEVICE": "cuda:7",
                    "SEME2E_ASR_WORKER_TIMEOUT_SECONDS": "17",
                },
            ),
        ):
            result = result_adapter.maybe_asr_eval(
                clean_path,
                protected_path,
                {
                    "semantic": {"asrModel": "openai-whisper:base"},
                    "language": "en",
                    "referenceText": "the original transcript",
                    "forceAsrEval": True,
                },
            )

        self.assertEqual(result["status"], "available")
        self.assertEqual(result["model"], "openai-whisper:base")
        self.assertEqual(result["originalText"], "the original transcript")
        self.assertEqual(result["protectedText"], "the changed transcript")
        main_process_transcriber.assert_not_called()
        worker.assert_called_once_with(
            result_adapter.ROOT / "asr_worker.py",
            {
                "model": "openai-whisper:base",
                "device": "cuda:7",
                "language": "en",
                "originalPath": str(clean_path.resolve()),
                "protectedPath": str(protected_path.resolve()),
            },
            timeout_seconds=17,
            cancel_event=None,
        )

    def test_concurrent_asr_requests_use_independent_workers(self) -> None:
        state_lock = threading.Lock()
        worker_barrier = threading.Barrier(2)
        active = 0
        max_active = 0
        worker_models: list[str] = []

        def fake_worker(
            worker_path: Path,
            request_payload: dict[str, object],
            *,
            timeout_seconds: int,
            cancel_event: object | None = None,
        ) -> dict[str, object]:
            nonlocal active, max_active
            self.assertEqual(worker_path, result_adapter.ROOT / "asr_worker.py")
            self.assertEqual(timeout_seconds, 600)
            self.assertIsNone(cancel_event)
            model = str(request_payload["model"])
            with state_lock:
                active += 1
                max_active = max(max_active, active)
                worker_models.append(model)
            try:
                worker_barrier.wait(timeout=3)
                time.sleep(0.02)
            finally:
                with state_lock:
                    active -= 1
            return {
                "ok": True,
                "model": model,
                "language": "en",
                "originalText": f"{model} original",
                "protectedText": f"{model} protected",
            }

        def evaluate(model_name: str) -> dict[str, object]:
            return result_adapter.maybe_asr_eval(
                Path("clean.wav"),
                Path("protected.wav"),
                {"semantic": {"asrModel": model_name}, "language": "en", "forceAsrEval": True},
            )

        tracking_lock = mock.MagicMock()
        tracking_lock.__enter__.return_value = None
        worker_slots = threading.BoundedSemaphore(2)
        with (
            mock.patch.object(result_adapter, "_run_isolated_json_worker", side_effect=fake_worker) as worker,
            mock.patch.object(result_adapter, "ASR_WORKER_SLOTS", worker_slots),
            mock.patch.object(asr_backends, "ASRTranscriber") as main_process_transcriber,
            mock.patch.object(asr_backends, "OPENAI_WHISPER_SESSION_LOCK", tracking_lock),
            mock.patch.dict(os.environ, {"SEME2E_ASR_WORKER_TIMEOUT_SECONDS": "600"}),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            results = list(executor.map(evaluate, ["openai-whisper:tiny", "openai-whisper:base"]))

        self.assertEqual(worker.call_count, 2)
        self.assertEqual(max_active, 2)
        self.assertCountEqual(worker_models, ["openai-whisper:tiny", "openai-whisper:base"])
        self.assertTrue(all(result["status"] == "available" for result in results))
        self.assertCountEqual(
            [result["originalText"] for result in results],
            ["openai-whisper:tiny original", "openai-whisper:base original"],
        )
        main_process_transcriber.assert_not_called()
        tracking_lock.__enter__.assert_not_called()
        tracking_lock.__exit__.assert_not_called()

    def test_wav2vec2_session_does_not_take_openai_whisper_lock(self) -> None:
        tracking_lock = mock.MagicMock()
        tracking_lock.__enter__.return_value = None

        with mock.patch.object(asr_backends, "OPENAI_WHISPER_SESSION_LOCK", tracking_lock):
            with asr_backends.openai_whisper_session("facebook/wav2vec2-base-960h"):
                pass

        tracking_lock.__enter__.assert_not_called()
        tracking_lock.__exit__.assert_not_called()


if __name__ == "__main__":
    unittest.main()
