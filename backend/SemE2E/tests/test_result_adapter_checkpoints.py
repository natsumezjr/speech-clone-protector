from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from result_adapter import _torch_checkpoint_ready


class TorchCheckpointReadinessTests(unittest.TestCase):
    def test_valid_zip_checkpoint_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pth"
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("archive/data.pkl", b"x" * (1024 * 1024))
            self.assertEqual(_torch_checkpoint_ready(path), (True, None))

    def test_truncated_zip_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pth"
            path.write_bytes(b"PK\x03\x04" + b"x" * (1024 * 1024))
            ready, reason = _torch_checkpoint_ready(path)
            self.assertFalse(ready)
            self.assertIn("invalid checkpoint archive", reason or "")

    def test_tiny_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pth"
            path.write_bytes(b"not-a-checkpoint")
            ready, reason = _torch_checkpoint_ready(path)
            self.assertFalse(ready)
            self.assertIn("incomplete checkpoint", reason or "")


if __name__ == "__main__":
    unittest.main()
