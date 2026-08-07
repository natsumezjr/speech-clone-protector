from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import soundfile as sf
import torchaudio.functional as ta_functional

from runtime import resolve_torch_device

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parent
LOCAL_ASR_MODELS = {
    "openai/whisper-small": ROOT / "checkpoints" / "asr" / "openai-whisper-small",
    "facebook/wav2vec2-base-960h": ROOT / "checkpoints" / "hf" / "facebook" / "wav2vec2-base-960h",
}
LOCAL_FUNASR_MODELS = {
    "paraformer-zh": ROOT / "checkpoints" / "modelscope" / "damo" / "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
}

# openai-whisper performs model loading and decoding through process-global
# PyTorch/CUDA state.  Keep one complete OpenAI Whisper evaluation session at
# a time so concurrently queued Tiny/Base jobs cannot interleave model dtype
# and device work.  Other backends (especially the direct Wav2Vec2 CTC path)
# deliberately do not use this lock.
OPENAI_WHISPER_SESSION_LOCK = threading.RLock()


@contextmanager
def openai_whisper_session(model_name: str) -> Iterator[None]:
    if model_name.startswith("openai-whisper:"):
        with OPENAI_WHISPER_SESSION_LOCK:
            yield
        return
    yield


def resolve_asr_model_path(model_name: str) -> str:
    local_path = LOCAL_ASR_MODELS.get(model_name)
    if local_path and (local_path / "config.json").exists():
        return str(local_path)
    return model_name


class ASRTranscriber:
    """Thin wrapper over the ASR backends used by the evaluation scripts."""

    def __init__(self, model_name: str, device: str = "cuda", language: str = "en"):
        self.model_name = model_name
        self.device = resolve_torch_device(device)
        self.language = "zh" if str(language).lower().startswith("zh") else "en"

        if model_name.startswith("funasr:"):
            self.backend = "funasr"
            self._init_funasr(model_name.split(":", 1)[1])
        elif model_name.startswith("openai-whisper:"):
            self.backend = "openai-whisper"
            self._init_openai_whisper(model_name.split(":", 1)[1])
        elif "wav2vec2" in model_name.lower():
            self.backend = "wav2vec2-ctc"
            self._init_wav2vec2(model_name)
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

    def _init_wav2vec2(self, model_name: str) -> None:
        from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForCTC
        from transformers.models.wav2vec2.processing_wav2vec2 import Wav2Vec2Processor

        resolved_model = resolve_asr_model_path(model_name)
        local_only = Path(resolved_model).is_dir()
        self.processor = Wav2Vec2Processor.from_pretrained(resolved_model, local_files_only=local_only)
        self.model = Wav2Vec2ForCTC.from_pretrained(resolved_model, local_files_only=local_only)
        self.model.to(self.device).eval()

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
                language=self.language,
                task="transcribe",
                fp16=self.device.type == "cuda",
            )
            return result["text"].strip()

        if self.backend == "wav2vec2-ctc":
            audio = self._load_audio_array(audio_path)
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            model_inputs = {name: value.to(self.device) for name, value in inputs.items() if isinstance(value, torch.Tensor)}
            with torch.inference_mode():
                logits = self.model(**model_inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            return self.processor.batch_decode(predicted_ids)[0].strip()

        kwargs = {}
        if self.is_whisper:
            kwargs["generate_kwargs"] = {"language": self.language, "task": "transcribe"}
        audio = self._load_audio_array(audio_path)
        result = self.pipe({"array": audio, "sampling_rate": 16000}, **kwargs)
        return result["text"].strip()
