import os
from pathlib import Path

import torch
import torch.nn.functional as F
import librosa
import torchaudio
import s3tokenizer

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import (
    HubertModel,
    Wav2Vec2FeatureExtractor,
    WhisperFeatureExtractor,
    WhisperModel,
)


class DifferentiableWhisperMel(torch.nn.Module):
    def __init__(self, feature_extractor: WhisperFeatureExtractor, device):
        super().__init__()
        self.n_fft = feature_extractor.n_fft
        self.hop_length = feature_extractor.hop_length
        self.n_mels = feature_extractor.feature_size
        self.sampling_rate = feature_extractor.sampling_rate
        self.n_samples = feature_extractor.n_samples
        self.max_frames = self.n_samples // self.hop_length

        mel_filters = librosa.filters.mel(
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
        )
        self.register_buffer("mel_filters", torch.from_numpy(mel_filters).float())
        self.register_buffer("window", torch.hann_window(self.n_fft))
        self.to(device)

    def forward(self, wave_16k):
        wav = wave_16k.squeeze(0)
        target_len = self.n_samples
        if wav.shape[-1] < target_len:
            wav = F.pad(wav, (0, target_len - wav.shape[-1]))
        else:
            wav = wav[..., :target_len]

        stft = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            center=True,
        )
        magnitudes = stft.abs() ** 2
        mel_spec = self.mel_filters @ magnitudes
        log_mel = torch.clamp(mel_spec, min=1e-10).log10()
        log_mel = torch.clamp((log_mel + 4.0) / 4.0, min=0.0)

        if log_mel.dim() == 2:
            log_mel = log_mel.unsqueeze(0)
        if log_mel.shape[-1] < self.max_frames:
            log_mel = F.pad(log_mel, (0, self.max_frames - log_mel.shape[-1]))
        else:
            log_mel = log_mel[..., :self.max_frames]

        return log_mel


class DifferentiableOpenAIWhisperMel(torch.nn.Module):
    def __init__(self, n_mels: int, device):
        super().__init__()
        import whisper

        self.n_fft = whisper.audio.N_FFT
        self.hop_length = whisper.audio.HOP_LENGTH
        self.n_mels = n_mels
        self.sampling_rate = whisper.audio.SAMPLE_RATE
        self.n_samples = whisper.audio.N_SAMPLES
        self.max_frames = whisper.audio.N_FRAMES

        mel_filters = whisper.audio.mel_filters(torch.device("cpu"), n_mels=n_mels)
        self.register_buffer("mel_filters", mel_filters.float())
        self.register_buffer("window", torch.hann_window(self.n_fft))
        self.to(device)

    def forward(self, wave_16k):
        wav = wave_16k.squeeze(0)
        if wav.shape[-1] < self.n_samples:
            wav = F.pad(wav, (0, self.n_samples - wav.shape[-1]))
        else:
            wav = wav[..., : self.n_samples]

        stft = torch.stft(
            wav,
            self.n_fft,
            self.hop_length,
            window=self.window,
            return_complex=True,
            center=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = self.mel_filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        if log_spec.dim() == 2:
            log_spec = log_spec.unsqueeze(0)
        if log_spec.shape[-1] < self.max_frames:
            log_spec = F.pad(log_spec, (0, self.max_frames - log_spec.shape[-1]))
        else:
            log_spec = log_spec[..., : self.max_frames]
        return log_spec


class SemanticEncoderEnsemble:
    """T-SemAttack semantic surrogate encoders used as Lsem."""

    def __init__(
        self,
        device,
        tokenizer_path,
        hubert_path="facebook/hubert-large-ll60k",
        whisper_path="openai/whisper-large-v3",
        sample_rate=16000,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.criterion = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.whisper_valid_frames = None
        self.vector_names = []

        self.speech_tokenizer = s3tokenizer.load_model(tokenizer_path).to(self.device)
        self.encoder_s3 = self.speech_tokenizer.encoder
        self._freeze(self.encoder_s3)
        self.vector_names.append("s3")

        self.hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_path)
        self.encoder_hubert = HubertModel.from_pretrained(
            hubert_path,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.encoder_hubert.eval()
        self._freeze(self.encoder_hubert)
        self.hubert_do_normalize = self.hubert_feature_extractor.do_normalize
        self.vector_names.append("hubert")

        self.use_whisper = str(whisper_path).lower() not in {"none", "off", "disabled"}
        if self.use_whisper:
            if str(whisper_path).startswith("openai-whisper:"):
                import whisper

                self.whisper_backend = "openai"
                model_name = str(whisper_path).split(":", 1)[1]
                download_root = os.environ.get("WHISPER_CACHE_DIR") or str(Path.home() / ".cache" / "whisper")
                self.openai_whisper = whisper.load_model(
                    model_name,
                    device=str(self.device),
                    download_root=download_root,
                ).eval()
                self.whisper_mel = DifferentiableOpenAIWhisperMel(
                    self.openai_whisper.dims.n_mels,
                    self.device,
                )
                self.encoder_whisper = self.openai_whisper.encoder
                self._freeze(self.openai_whisper)
            else:
                self.whisper_backend = "transformers"
                whisper_fe = WhisperFeatureExtractor.from_pretrained(whisper_path)
                self.whisper_mel = DifferentiableWhisperMel(whisper_fe, self.device)
                self.encoder_whisper = WhisperModel.from_pretrained(
                    whisper_path,
                    torch_dtype=torch.float32,
                ).to(self.device).encoder
                self.encoder_whisper.eval()
                self._freeze(self.encoder_whisper)
            self.vector_names.append("whisper")

        self.encoder_mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=40,
        ).to(self.device)
        self.vector_names.append("mfcc")

    @staticmethod
    def _freeze(model):
        for param in model.parameters():
            param.requires_grad = False

    def _extract_hubert(self, wave_16k):
        wav = wave_16k.squeeze(0)
        if self.hubert_do_normalize:
            wav = (wav - wav.mean()) / (wav.std() + 1e-7)
        outputs = self.encoder_hubert(input_values=wav.unsqueeze(0))
        return outputs.last_hidden_state

    def _extract_whisper(self, wave_16k):
        actual_samples = wave_16k.shape[-1]
        actual_frames = actual_samples // self.whisper_mel.hop_length
        self.whisper_valid_frames = min(actual_frames, self.whisper_mel.max_frames)

        input_features = self.whisper_mel(wave_16k)
        if self.whisper_backend == "openai":
            return self.encoder_whisper(input_features)
        outputs = self.encoder_whisper(input_features=input_features)
        return outputs.last_hidden_state

    def _extract_s3(self, wave_16k):
        mel = s3tokenizer.log_mel_spectrogram(wave_16k.squeeze(0)).to(self.device)
        mel, mel_len = s3tokenizer.padding([mel])
        hidden, _ = self.encoder_s3(mel.to(self.device), mel_len.to(self.device))
        return hidden.transpose(2, 1)

    def _extract_mfcc(self, wave_16k):
        return self.encoder_mfcc(wave_16k)

    def get_vectors(self, wave_16k):
        vectors = [
            self._extract_s3(wave_16k),
            self._extract_hubert(wave_16k),
        ]
        if self.use_whisper:
            vectors.append(self._extract_whisper(wave_16k))
        vectors.append(self._extract_mfcc(wave_16k))
        return vectors

    def compute_loss(self, clean_vectors, adv_vectors):
        loss = 0.0

        for name, clean_vec, adv_vec in zip(self.vector_names, clean_vectors, adv_vectors):
            if clean_vec.dim() == 2:
                clean_vec = clean_vec.unsqueeze(0)
            if adv_vec.dim() == 2:
                adv_vec = adv_vec.unsqueeze(0)

            if name == "whisper" and self.whisper_valid_frames is not None:
                n_valid_enc = max(1, self.whisper_valid_frames // 2)
                clean_vec = clean_vec[:, :n_valid_enc, :]
                adv_vec = adv_vec[:, :n_valid_enc, :]

            try:
                score = self.criterion(clean_vec, adv_vec).mean()
            except RuntimeError:
                min_len = min(clean_vec.shape[-1], adv_vec.shape[-1])
                min_seq = min(clean_vec.shape[-2], adv_vec.shape[-2])
                clean_vec = clean_vec[..., :min_seq, :min_len]
                adv_vec = adv_vec[..., :min_seq, :min_len]
                score = self.criterion(clean_vec, adv_vec).mean()

            loss = loss + score

        return loss
