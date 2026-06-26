from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import api_server


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
                    asrResult=None,
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
            self.assertEqual(status["cloneTask"]["status"], "running")
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["asrStatus"], "running")
            self.assertEqual(row["cloneStatus"], "running")
            self.assertTrue(row["hasAsrResult"])
            self.assertTrue(row["hasCloneResult"])


if __name__ == "__main__":
    unittest.main()
