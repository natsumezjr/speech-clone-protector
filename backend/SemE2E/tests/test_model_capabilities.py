from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import result_adapter
from result_adapter import runtime_config
import metric_definitions
import api_server


class ModelCapabilitiesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = runtime_config()

    def test_every_model_exposes_backend_owned_information(self) -> None:
        for branch, options in self.config["models"].items():
            with self.subTest(branch=branch):
                self.assertTrue(options)
            for option in options:
                with self.subTest(branch=branch, value=option.get("value")):
                    self.assertTrue(option.get("name"))
                    self.assertIsInstance(option.get("type"), list)
                    self.assertTrue(option.get("type"))
                    self.assertTrue(option.get("information"))
                    self.assertNotIn("primaryType", option)
                    self.assertTrue(option.get("value"))
                    self.assertTrue(option.get("backendValue"))

    def test_tts_types_are_nonexclusive_and_llm_model_is_declared(self) -> None:
        cosyvoice = next(option for option in self.config["models"]["tts"] if option["value"] == "CosyVoice2-0.5B")
        self.assertIn("zero_shot", cosyvoice["type"])
        self.assertIn("llm_based", cosyvoice["type"])
        self.assertEqual(cosyvoice["backendValue"], "cosyvoice2:0.5b")
        self.assertTrue(cosyvoice["promptRequired"])
        self.assertEqual(
            {item["value"] for item in self.config["modelTypes"]["tts"]},
            {"zero_shot", "fine_tuning", "llm_based"},
        )
        gpt_sovits = next(option for option in self.config["models"]["tts"] if option["value"] == "GPT-SoVITS")
        self.assertIn("fine_tuning", gpt_sovits["type"])
        self.assertIn("llm_based", gpt_sovits["type"])
        self.assertEqual(gpt_sovits["fineTuneMode"], "live_fine_tune")
        self.assertTrue(gpt_sovits["promptRequired"])
        self.assertNotIn("fineTuneDatasetSeconds", gpt_sovits)

    def test_gpt_sovits_status_accepts_live_fine_tune_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            python_path = root / "runtime" / ".venv" / "bin" / "python"
            tts_path = root / "runtime" / "repo" / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py"
            cnhubert = root / "checkpoints" / "hubert"
            bert = root / "checkpoints" / "bert"
            worker = root / "backend" / "gpt_sovits_live_finetune.py"
            infer_worker = root / "backend" / "gpt_sovits_worker.py"
            pretrained_s1 = root / "pretrained" / "s1.ckpt"
            pretrained_s2g = root / "pretrained" / "s2G.pth"
            pretrained_s2d = root / "pretrained" / "s2D.pth"
            for path in (python_path, tts_path, worker, infer_worker, pretrained_s1, pretrained_s2g, pretrained_s2d):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("ready", encoding="utf-8")
            cnhubert.mkdir(parents=True)
            bert.mkdir(parents=True)

            with patch.multiple(
                result_adapter,
                ROOT=worker.parent,
                GPT_SOVITS_PYTHON=python_path,
                GPT_SOVITS_RUNTIME_DIR=root / "runtime",
                GPT_SOVITS_REPO_DIR=root / "runtime" / "repo",
                GPT_SOVITS_CNHUBERT=cnhubert,
                GPT_SOVITS_BERT=bert,
                GPT_SOVITS_PRETRAINED_S1=pretrained_s1,
                GPT_SOVITS_PRETRAINED_S2G=pretrained_s2g,
                GPT_SOVITS_PRETRAINED_S2D=pretrained_s2d,
            ):
                status, reason, local_path = result_adapter._gpt_sovits_model_status()

            self.assertEqual(status, "available")
            self.assertIsNone(reason)
            self.assertEqual(local_path, str(root / "runtime"))

    def test_asr_and_independent_evaluator_are_declared(self) -> None:
        asr_values = {option["value"] for option in self.config["models"]["asr"]}
        self.assertIn("openai-whisper:medium", asr_values)
        self.assertIn("facebook/wav2vec2-base-960h", asr_values)
        evaluation = self.config["models"]["evaluation"]
        self.assertEqual(evaluation[0]["value"], "speechbrain/spkrec-ecapa-voxceleb")
        self.assertIn("evaluation_model", evaluation[0]["type"])
        self.assertNotEqual(evaluation[0]["branch"], "timbre")

    def test_capabilities_overlay_fresh_runtime_config_on_cached_probe(self) -> None:
        cached = {
            "ok": True,
            "modelTypes": {"asr": [{"value": "old"}]},
            "config": {"defaults": {"optimization": {"steps": 100}}},
            "cache": {"hit": True, "revision": 1},
        }
        fresh = {
            "modelTypes": {"asr": [{"value": "generative_asr"}]},
            "defaults": {"optimization": {"steps": 200}},
        }

        with patch.object(api_server, "cached_capabilities", return_value=cached), patch.object(
            api_server,
            "runtime_config",
            return_value=fresh,
        ):
            payload = api_server.capabilities()

        self.assertEqual(payload["config"]["defaults"]["optimization"]["steps"], 200)
        self.assertEqual(payload["modelTypes"], fresh["modelTypes"])
        self.assertEqual(payload["cache"]["revision"], 1)

    def test_hugging_face_model_ids_resolve_to_project_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "checkpoints" / "hf" / "facebook" / "hubert-large-ll60k"
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            with patch.object(metric_definitions, "ROOT", Path(temp_dir)):
                resolved = metric_definitions._resolve_local_model_path("facebook/hubert-large-ll60k")
            self.assertEqual(Path(resolved), checkpoint.resolve())


if __name__ == "__main__":
    unittest.main()
