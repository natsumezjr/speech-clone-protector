from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from capability_cache import REFRESH_FLAG_FILENAME, REFRESH_LOCK_FILENAME, get_capabilities_snapshot


class CapabilityCacheTests(unittest.TestCase):
    def test_snapshot_is_reused_and_flag_refreshes_in_background(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime_dir = Path(tmp)
            calls = 0

            def probe() -> dict[str, object]:
                nonlocal calls
                calls += 1
                return {"ok": True, "device": "cpu", "chains": {}, "config": {"probe": calls}}

            first = get_capabilities_snapshot(runtime_dir, probe)
            second = get_capabilities_snapshot(runtime_dir, probe)

            self.assertEqual(calls, 1)
            self.assertFalse(first["cache"]["hit"])
            self.assertTrue(second["cache"]["hit"])
            self.assertEqual(second["config"]["probe"], 1)

            (runtime_dir / REFRESH_FLAG_FILENAME).write_text("1\n", encoding="utf-8")
            stale = get_capabilities_snapshot(runtime_dir, probe)
            self.assertEqual(stale["config"]["probe"], 1)
            self.assertTrue(stale["cache"]["refreshing"])

            deadline = time.monotonic() + 3
            while (runtime_dir / REFRESH_LOCK_FILENAME).exists() and time.monotonic() < deadline:
                time.sleep(0.01)

            refreshed = get_capabilities_snapshot(runtime_dir, probe)
            self.assertEqual(calls, 2)
            self.assertEqual(refreshed["config"]["probe"], 2)
            self.assertEqual(refreshed["cache"]["revision"], 2)
            self.assertEqual((runtime_dir / REFRESH_FLAG_FILENAME).read_text(encoding="utf-8").strip(), "0")


if __name__ == "__main__":
    unittest.main()
