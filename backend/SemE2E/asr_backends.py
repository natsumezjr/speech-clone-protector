from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import torchaudio.functional as ta_functional

from runtime import resolve_torch_device

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parent
LOCAL_ASR_MODELS = {
    "openai/whisper-small": ROOT / "checkpoints" / "asr" / "openai-whisper-small",
}
LOCAL_FUNASR_MODELS = {
    "paraformer-zh": ROOT / "checkpoints" / "modelscope" / "damo" / "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
}


def resolve_asr_model_path(model_name: str) -> str:
    local_path = LOCAL_ASR_MODELS.get(model_name)
    if local_path and (local_path / "config.json").exists():
        return str(local_path)
    return model_name


class ASRTranscriber:
    """Thin wrapper over the ASR backends used by the evaluation scripts."""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = resolve_torch_device(device)

        if model_name.startswith("funasr:"):
            self.backend = "funasr"
            self._init_funasr(model_name.split(":", 1)[1])
        elif model_name.startswith("openai-whisper:"):
            self.backend = "openai-whisper"
            self._init_openai_whisper(model_name.split(":", 1)[1])
        else:
            self.backend = "transformers"
            self._init_transformers(model_name)

    def _init_funasr(self, model_id: str) -> None:
        from funasr import AutoModel as FunASRAutoModel

        local_path = LOCAL_FUNASR_MODELS.get(model_id)
        resolved_model = str(local_path) if local_path and (local_path / "model.pt").exists() else model_id
        self.model = FunASRAutoModel(model=resolved_model, disable_update=True)

    def _init_openai_whisper(self, model_id: str) -> None:
        import whisper

        self.model = whisper.load_model(model_id, device=str(self.device))

    def _init_transformers(self, model_name: str) -> None:
        from transformers import pipeline

        self.is_whisper = "whisper" in model_name.lower()
        device_id = (self.device.index or 0) if self.device.type == "cuda" else -1
        dtype = torch.float16 if device_id >= 0 and self.is_whisper else torch.float32
        resolved_model = resolve_asr_model_path(model_name)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=resolved_model,
            device=device_id,
            torch_dtype=dtype,
        )

    @staticmethod
    def _load_audio_array(audio_path: str | Path, target_sr: int = 16000) -> np.ndarray:
        audio_np, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        audio = torch.from_numpy(audio_np.T)
        if audio.size(0) != 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sample_rate != target_sr:
            audio = ta_functional.resample(audio, sample_rate, target_sr)
        return audio.squeeze(0).contiguous().cpu().numpy()

    def transcribe(self, audio_path: str | Path) -> str:
        if self.backend == "funasr":
            result = self.model.generate(input=str(audio_path))
            if isinstance(result, list) and result:
                return str(result[0].get("text", "")).strip()
            return str(result).strip()

        if self.backend == "openai-whisper":
            audio = self._load_audio_array(audio_path)
            result = self.model.transcribe(
                audio,
                language="en",
                task="transcribe",
                fp16=self.device.type == "cuda",
            )
            return result["text"].strip()

        kwargs = {}
        if self.is_whisper:
            kwargs["generate_kwargs"] = {"language": "en", "task": "transcribe"}
        audio = self._load_audio_array(audio_path)
        result = self.pipe({"array": audio, "sampling_rate": 16000}, **kwargs)
        return result["text"].strip()
