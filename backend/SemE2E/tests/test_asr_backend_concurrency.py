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

        class FakeTranscriber:
            def __init__(self, model_name: str, device: str, language: str) -> None:
                self.model_name = model_name

            def transcribe(self, audio_path: str | Path) -> str:
                nonlocal active, max_active
                with state_lock:
                    active += 1
                    max_active = max(max_active, active)
                    events.append(self.model_name)
                time.sleep(0.02)
                with state_lock:
                    active -= 1
                return f"{self.model_name} transcript"

        def evaluate(model_name: str) -> dict[str, object]:
            start_barrier.wait(timeout=1)
            return result_adapter.maybe_asr_eval(
                Path("clean.wav"),
                Path("protected.wav"),
                {"semantic": {"asrModel": model_name}, "language": "en", "forceAsrEval": True},
            )

        with (
            mock.patch.object(asr_backends, "ASRTranscriber", FakeTranscriber),
            mock.patch.dict(os.environ, {"SEME2E_ENABLE_ASR": "1"}),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            results = list(executor.map(evaluate, ["openai-whisper:tiny", "openai-whisper:base"]))

        self.assertEqual(max_active, 1)
        self.assertIn(events, [
            ["openai-whisper:tiny", "openai-whisper:tiny", "openai-whisper:base", "openai-whisper:base"],
            ["openai-whisper:base", "openai-whisper:base", "openai-whisper:tiny", "openai-whisper:tiny"],
        ])
        self.assertTrue(all(result["status"] == "available" for result in results))

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
