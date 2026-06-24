from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from audio_utils import load_mono
from runtime import resolve_torch_device

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(__file__).resolve().parent


class WavLMSpeakerSimilarity:
    """Fast speaker-similarity proxy based on mean-pooled WavLM states."""

    def __init__(self, model_path: str | Path, device: str = "cuda"):
        self.device = resolve_torch_device(device)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device).eval()

    def embed(self, audio_path: str | Path) -> torch.Tensor:
        wav, _ = load_mono(audio_path, sr=16000)
        if self.feature_extractor.do_normalize:
            wav = (wav - wav.mean()) / (wav.std() + 1.0e-7)

        inputs = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            hidden = self.model(input_values=inputs).last_hidden_state
        return hidden.mean(dim=1)

    def score(self, reference_audio: str | Path, candidate_audio: str | Path) -> float:
        ref = self.embed(reference_audio)
        cand = self.embed(candidate_audio)
        return float(F.cosine_similarity(ref, cand, dim=-1).item())


class ECAPASpeakerSimilarity:
    """Speaker similarity from SpeechBrain ECAPA-TDNN embeddings."""

    def __init__(self, model_path: str | Path, device: str = "cuda"):
        self.device = resolve_torch_device(device)
        self.device_name = str(self.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device_name = "cuda:0"
        self.model_path = str(model_path)

        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier

        savedir = ROOT / "checkpoints" / "ecapa"
        self.model = EncoderClassifier.from_hparams(
            source=self.model_path,
            savedir=str(savedir),
            run_opts={"device": self.device_name},
        )

    def embed(self, audio_path: str | Path) -> torch.Tensor:
        wav, _ = load_mono(audio_path, sr=16000)
        inputs = torch.from_numpy(wav).float().unsqueeze(0).to(self.device_name)
        lengths = torch.ones(inputs.shape[0], device=self.device_name)
        with torch.no_grad():
            embedding = self.model.encode_batch(inputs, lengths).to(self.device_name)
        if embedding.dim() == 3 and embedding.size(1) == 1:
            embedding = embedding.squeeze(1)
        if embedding.dim() > 2:
            embedding = embedding.reshape(embedding.size(0), -1)
        return F.normalize(embedding, dim=-1)

    def score(self, reference_audio: str | Path, candidate_audio: str | Path) -> float:
        ref = self.embed(reference_audio)
        cand = self.embed(candidate_audio)
        return float(F.cosine_similarity(ref, cand, dim=-1).item())


def build_speaker_similarity(metric: str, model_path: str | Path, device: str = "cuda"):
    metric = metric.lower().strip()
    if metric == "ecapa":
        return ECAPASpeakerSimilarity(model_path, device)
    if metric == "wavlm":
        return WavLMSpeakerSimilarity(model_path, device)
    raise ValueError(f"unsupported speaker similarity metric: {metric}")
