from __future__ import annotations

import json
import os
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

    def test_tts_reference_text_capability_matches_real_backend_contract(self) -> None:
        options = {option["value"]: option for option in self.config["models"]["tts"]}
        expected = {
            "XTTS-v2": False,
            "YourTTS": False,
            "CosyVoice2-0.5B": True,
            "GPT-SoVITS": True,
        }

        self.assertEqual(set(options), set(expected))
        for model, requires_reference_text in expected.items():
            with self.subTest(model=model):
                option = options[model]
                self.assertIs(option["requiresReferenceText"], requires_reference_text)
                self.assertIs(option["promptRequired"], requires_reference_text)
                self.assertEqual(option["annotationSources"], ["manual", "asr"] if requires_reference_text else [])
                self.assertIs(
                    result_adapter.tts_model_requires_reference_text(option["backendValue"]),
                    requires_reference_text,
                )

        self.assertFalse(result_adapter.tts_model_requires_reference_text("XTTS-v1.1"))

    def test_xtts_v11_is_not_advertised_for_new_clone_tasks(self) -> None:
        tts_values = {option["value"] for option in self.config["models"]["tts"]}
        tts_backend_values = {option["backendValue"] for option in self.config["models"]["tts"]}

        self.assertIn("XTTS-v2", tts_values)
        self.assertIn("xtts_v2", tts_backend_values)
        self.assertNotIn("XTTS-v1.1", tts_values)
        self.assertNotIn("tts_models/multilingual/multi-dataset/xtts_v1.1", tts_backend_values)
        self.assertEqual(
            result_adapter.normalize_tts_model("XTTS-v1.1"),
            "tts_models/multilingual/multi-dataset/xtts_v1.1",
        )
        self.assertEqual(result_adapter._tts_catalog_entry("XTTS-v1.1")["value"], "XTTS-v1.1")

    def test_visible_tts_language_catalog_matches_supported_model_matrix(self) -> None:
        languages = {
            option["value"]: set(option.get("languages") or [])
            for option in self.config["models"]["tts"]
        }

        self.assertEqual(languages["XTTS-v2"], {"en", "zh-cn"})
        self.assertEqual(languages["YourTTS"], {"en"})
        self.assertEqual(languages["CosyVoice2-0.5B"], {"en", "zh-cn"})
        self.assertEqual(languages["GPT-SoVITS"], {"en", "zh-cn"})

    def test_hidden_tts_environment_default_falls_back_to_visible_available_model(self) -> None:
        def catalog_status(item: dict[str, object], *, coqui_available: bool) -> tuple[str, str | None, str | None]:
            del coqui_available
            if item["value"] == "YourTTS":
                return "available", None, "your-tts-cache"
            return "unavailable", "test unavailable", None

        with patch.dict(os.environ, {"SEME2E_API_DEFAULT_TTS_MODEL": "XTTS-v1.1"}), patch.object(
            result_adapter,
            "_tts_catalog_status",
            side_effect=catalog_status,
        ):
            config = runtime_config()

        visible_options = config["models"]["tts"]
        form_options = config["formSchema"]["modelOptions"]["ttsModels"]
        visible_backend_values = {option["backendValue"] for option in visible_options}
        default_backend = config["clone"]["defaults"]["backendValue"]

        self.assertEqual(default_backend, "tts_models/multilingual/multi-dataset/your_tts")
        self.assertEqual(config["clone"]["defaults"]["model"], default_backend)
        self.assertIn(default_backend, visible_backend_values)
        self.assertEqual(form_options, visible_options)
        self.assertNotIn("tts_models/multilingual/multi-dataset/xtts_v1.1", visible_backend_values)

    def test_hidden_tts_environment_default_falls_back_to_first_visible_when_none_are_available(self) -> None:
        with patch.dict(os.environ, {"SEME2E_API_DEFAULT_TTS_MODEL": "XTTS-v1.1"}), patch.object(
            result_adapter,
            "_tts_catalog_status",
            return_value=("unavailable", "test unavailable", None),
        ):
            config = runtime_config()

        first_visible = config["models"]["tts"][0]
        self.assertEqual(config["clone"]["defaults"]["backendValue"], first_visible["backendValue"])
        self.assertEqual(config["clone"]["defaults"]["model"], first_visible["backendValue"])

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
            "config": {
                "defaults": {"optimization": {"steps": 100}},
                "models": {"tts": [{"value": "XTTS-v1.1"}]},
            },
            "cache": {"hit": True, "revision": 1},
        }
        fresh = {
            "modelTypes": {"asr": [{"value": "generative_asr"}]},
            "defaults": {"optimization": {"steps": 200}},
            "models": {"tts": [{"value": "XTTS-v2"}]},
        }

        with patch.object(api_server, "cached_capabilities", return_value=cached), patch.object(
            api_server,
            "runtime_config",
            return_value=fresh,
        ):
            payload = api_server.capabilities()

        self.assertEqual(payload["config"]["defaults"]["optimization"]["steps"], 200)
        self.assertEqual(payload["config"]["models"]["tts"], [{"value": "XTTS-v2"}])
        self.assertEqual(payload["modelTypes"], fresh["modelTypes"])
        self.assertEqual(payload["cache"]["revision"], 1)

    def test_runtime_concurrency_snapshot_counts_training_and_inference_workers(self) -> None:
        with patch.multiple(
            api_server,
            PROTECT_MAX_CONCURRENCY=3,
            ASR_WORKER_MAX_CONCURRENCY=2,
            clone_worker_capacity_snapshot=lambda: {
                "maxConcurrency": 3,
                "asrCloneMaxConcurrency": 4,
                "protectSharesWorkerGpu": False,
                "backendLimits": {"coquiTts": 1, "cosyVoice": 1, "gptSoVits": 2},
                "gpuSlotLimit": 1,
                "gpuKeys": {"coquiTts": ["4"], "cosyVoice": ["4"], "gptSoVits": ["0", "1"]},
            },
        ):
            payload = api_server.runtime_concurrency_snapshot()

        self.assertEqual(payload["protect"], 3)
        self.assertEqual(payload["asr"], 2)
        self.assertEqual(payload["clone"], 3)
        self.assertEqual(payload["asrCloneShared"], 4)
        self.assertFalse(payload["protectSharesWorkerGpu"])
        self.assertEqual(payload["total"], 7)
        self.assertEqual(payload["unit"], "worker")
        self.assertIn("HTTP", payload["definition"])

    def test_runtime_concurrency_does_not_add_protect_when_single_gpu_is_shared(self) -> None:
        namespace = dict(api_server.runtime_concurrency_snapshot.__globals__)
        namespace.update(
            PROTECT_MAX_CONCURRENCY=1,
            ASR_WORKER_MAX_CONCURRENCY=1,
            clone_worker_capacity_snapshot=lambda: {
                "maxConcurrency": 1,
                "asrCloneMaxConcurrency": 1,
                "protectSharesWorkerGpu": True,
                "backendLimits": {"coquiTts": 1, "cosyVoice": 1, "gptSoVits": 1},
                "gpuSlotLimit": 1,
                "gpuKeys": {"coquiTts": ["0"], "cosyVoice": ["0"], "gptSoVits": ["0"]},
                "asrGpuKeys": ["0"],
            },
        )
        snapshot = type(api_server.runtime_concurrency_snapshot)(
            api_server.runtime_concurrency_snapshot.__code__,
            namespace,
        )()

        self.assertEqual(snapshot["total"], 1)
        self.assertTrue(snapshot["protectSharesWorkerGpu"])
        self.assertIn("共享同一 GPU", snapshot["definition"])

    def test_clone_concurrency_respects_shared_gpu_slots(self) -> None:
        maximum = result_adapter.maximum_clone_worker_concurrency(
            coqui_limit=1,
            cosyvoice_limit=1,
            gpt_sovits_limit=2,
            clone_gpu_limit=1,
            coqui_gpu_keys=("4",),
            cosyvoice_gpu_keys=("4",),
            gpt_sovits_gpu_keys=("0", "1"),
        )
        self.assertEqual(maximum, 3)

    def test_clone_concurrency_counts_separate_backend_gpus(self) -> None:
        maximum = result_adapter.maximum_clone_worker_concurrency(
            coqui_limit=1,
            cosyvoice_limit=1,
            gpt_sovits_limit=2,
            clone_gpu_limit=1,
            coqui_gpu_keys=("2",),
            cosyvoice_gpu_keys=("3",),
            gpt_sovits_gpu_keys=("0", "1"),
        )
        self.assertEqual(maximum, 4)

    def test_clone_concurrency_avoids_double_counting_overlapping_gpt_gpu(self) -> None:
        maximum = result_adapter.maximum_clone_worker_concurrency(
            coqui_limit=1,
            cosyvoice_limit=1,
            gpt_sovits_limit=2,
            clone_gpu_limit=1,
            coqui_gpu_keys=("0",),
            cosyvoice_gpu_keys=("0",),
            gpt_sovits_gpu_keys=("0", "1"),
        )
        self.assertEqual(maximum, 2)

    def test_asr_and_clone_dynamic_pool_capacity_is_bounded_by_physical_slots(self) -> None:
        maximum = result_adapter.maximum_gpu_worker_concurrency(
            worker_limits={"asr": 2, "coqui": 1, "cosy": 1, "gpt": 2},
            worker_gpu_keys={
                "asr": ("0", "1", "2"),
                "coqui": ("0", "1", "2"),
                "cosy": ("0", "1", "2"),
                "gpt": ("0", "1", "2"),
            },
            gpu_slot_limit=1,
        )

        self.assertEqual(maximum, 3)

    def test_capacity_snapshot_detects_index_uuid_alias_as_shared_protect_gpu(self) -> None:
        inventory = (
            ("0",),
            {"0": 32000, "GPU-AAAA": 32000},
            {"0": "gpu-aaaa", "GPU-AAAA": "gpu-aaaa", "gpu-aaaa": "gpu-aaaa"},
        )
        with (
            patch.object(result_adapter, "_nvidia_gpu_inventory", return_value=inventory),
            patch.dict(
                result_adapter.os.environ,
                {
                    "SEME2E_API_DEVICE": "cuda:0",
                    "CUDA_VISIBLE_DEVICES": "GPU-AAAA",
                    "SEME2E_GPU_POOL": "0",
                    "SEME2E_PROTECT_GPU_SHARED_WITH_WORKERS": "",
                    "SEME2E_TTS_DEVICE": "",
                    "SEME2E_ASR_DEVICE": "",
                    "SEME2E_COQUI_TTS_CUDA_VISIBLE_DEVICES": "",
                    "SEME2E_COSYVOICE_CUDA_VISIBLE_DEVICES": "",
                    "SEME2E_GPT_SOVITS_GPU_POOL": "",
                    "SEME2E_GPT_SOVITS_CUDA_VISIBLE_DEVICES": "",
                    "SEME2E_ASR_CUDA_VISIBLE_DEVICES": "",
                },
            ),
        ):
            snapshot = result_adapter.clone_worker_capacity_snapshot()

        self.assertTrue(snapshot["protectSharesWorkerGpu"])
        self.assertEqual(snapshot["protectGpuKeys"], ["GPU-AAAA"])

    def test_remote_setup_uses_shared_dynamic_worker_pool_instead_of_role_pins(self) -> None:
        script = (ROOT.parents[1] / "remote-setup.ps1").read_text(encoding="utf-8")

        self.assertIn('export SEME2E_GPU_POOL="$worker_gpu_pool"', script)
        self.assertIn("unset SEME2E_ASR_CUDA_VISIBLE_DEVICES", script)
        self.assertIn("unset SEME2E_CLONE_ASR_CUDA_VISIBLE_DEVICES", script)
        self.assertIn("unset SEME2E_COQUI_TTS_CUDA_VISIBLE_DEVICES", script)
        self.assertIn("unset SEME2E_COSYVOICE_CUDA_VISIBLE_DEVICES", script)
        self.assertIn("export SEME2E_PROTECT_GPU_SHARED_WITH_WORKERS=1", script)
        self.assertIn("export SEME2E_PROTECT_GPU_SHARED_WITH_WORKERS=0", script)
        self.assertIn("export SEME2E_PROTECT_CUDA_VISIBLE_DEVICES=\"$protect_gpu\"", script)
        self.assertIn("export SEME2E_TOKENIZER_DEVICE='cuda:0'", script)
        self.assertIn("export SEME2E_SEMANTIC_ENCODER_DEVICE='cuda:0'", script)

    def test_latest_runtime_performance_uses_latest_completed_real_average(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            def write_result(name: str, modified_at: int, payload: dict[str, object]) -> None:
                task_dir = root / name
                task_dir.mkdir()
                (task_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
                os.utime(task_dir, (modified_at, modified_at))

            write_result(
                "task_older",
                100,
                {"taskId": "task_older", "status": "completed", "details": {"generation": {"averageStepSec": 0.45}}},
            )
            write_result(
                "task_latest_failed",
                300,
                {"taskId": "task_latest_failed", "status": "failed", "details": {"generation": {"averageStepSec": 0.1}}},
            )
            write_result(
                "task_latest_completed",
                200,
                {"taskId": "task_latest_completed", "status": "completed", "details": {"generation": {"averageStepSec": 0.37}}},
            )

            with patch.object(api_server, "TASK_DIR", root):
                payload = api_server.latest_runtime_performance_snapshot()

        self.assertEqual(payload["averageStepSec"], 0.37)
        self.assertEqual(payload["sourceTaskId"], "task_latest_completed")
        self.assertEqual(payload["source"], "latest_completed_protection_result")

    def test_config_exposes_runtime_concurrency_and_measured_timing(self) -> None:
        concurrency = {"protect": 1, "asr": 1, "clone": 2, "total": 4, "unit": "worker"}
        performance = {"averageStepSec": 0.37, "sourceTaskId": "task_demo"}
        with patch.object(api_server, "runtime_config", return_value={"modelTypes": {}}), patch.object(
            api_server,
            "runtime_concurrency_snapshot",
            return_value=concurrency,
        ), patch.object(
            api_server,
            "latest_runtime_performance_snapshot",
            return_value=performance,
        ):
            payload = api_server.config()

        self.assertEqual(payload["runtimeConcurrency"], concurrency)
        self.assertEqual(payload["runtimePerformance"], performance)

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
