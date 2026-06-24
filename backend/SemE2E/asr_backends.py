from __future__ import annotations

import os
from pathlib import Path

import torch

from runtime import resolve_torch_device

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


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

        self.model = FunASRAutoModel(model=model_id, disable_update=True)

    def _init_openai_whisper(self, model_id: str) -> None:
        import whisper

        self.model = whisper.load_model(model_id, device=str(self.device))

    def _init_transformers(self, model_name: str) -> None:
        from transformers import pipeline

        self.is_whisper = "whisper" in model_name.lower()
        device_id = (self.device.index or 0) if self.device.type == "cuda" else -1
        dtype = torch.float16 if device_id >= 0 and self.is_whisper else torch.float32
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device_id,
            torch_dtype=dtype,
        )

    def transcribe(self, audio_path: str | Path) -> str:
        if self.backend == "funasr":
            result = self.model.generate(input=str(audio_path))
            if isinstance(result, list) and result:
                return str(result[0].get("text", "")).strip()
            return str(result).strip()

        if self.backend == "openai-whisper":
            result = self.model.transcribe(
                str(audio_path),
                language="en",
                task="transcribe",
                fp16=self.device.type == "cuda",
            )
            return result["text"].strip()

        kwargs = {}
        if self.is_whisper:
            kwargs["generate_kwargs"] = {"language": "en", "task": "transcribe"}
        result = self.pipe(str(audio_path), **kwargs)
        return result["text"].strip()
