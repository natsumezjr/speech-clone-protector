import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import librosa
import numpy as np
import onnx
import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torch.nn.functional as F

multi_head_attention_forward = F.multi_head_attention_forward

from onnx2torch import convert
from tqdm import tqdm
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "tts_models" / "vits"))
import tts_models.vits.commons as commons
from tts_models.vits.mel_processing import spectrogram_torch

sys.path.append(str(ROOT / "tts_models" / "gsv"))
sys.path.append(str(ROOT / "tts_models" / "gsv" / "GPT_SoVITS"))
from tts_models.gsv.GPT_SoVITS.module.models import SynthesizerTrn as GSVSynthesizerTrn

from core.encoders import SemanticEncoderEnsemble
from core.masking import Masker
from core.modeling import Config, build_models_vits, load_config
from core.utils import legacy_weight_norm_for_transformers_audio, resolve_torch_device

F.multi_head_attention_forward = multi_head_attention_forward


@dataclass(frozen=True)
class CheckpointPaths:
    vits_config: Path = ROOT / "tts_models" / "configs" / "LibriTTS_VITS.json"
    vits_checkpoint: Path = ROOT / "checkpoints" / "VITS" / "pretrained_ljs.pth"
    gsv_checkpoint: Path = ROOT / "checkpoints" / "GSV" / "base_models" / "gsv-v2final-pretrained" / "s2G2333k.pth"
    wavlm_dir: Path = ROOT / "checkpoints" / "wavlm"
    cosyvoice_campplus: Path = ROOT / "checkpoints" / "CosyVoice" / "base_models" / "CosyVoice-300M" / "campplus.onnx"


def build_gsv_sovits_encoder(sovits_path, device):
    checkpoint = torch.load(str(sovits_path), map_location="cpu")
    hps = Config(checkpoint["config"])
    hps.model.semantic_frame_rate = "25hz"
    hps.model.version = "v2"
    model = GSVSynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    model.load_state_dict(checkpoint["weight"], strict=False)
    model.eval()
    return model.to(device)


class VoiceSheild:
    """E2E-VGuard with Lasr replaced by T-SemAttack Lsem."""

    def __init__(
        self,
        epsilon=8 / 255,
        max_items=500,
        device=None,
        tokenizer_path=None,
        hubert_path="facebook/hubert-large-ls960-ft",
        whisper_path="openai/whisper-large-v3",
        use_vits=True,
        use_gsv=True,
        use_mfcc_timbre=True,
        use_wavlm=True,
        use_cosyvoice=True,
        weight_feature=500.0,
        weight_semantic=100.0,
        weight_psy=1.0e-5,
        weight_l2=0.1,
        l2_reduction="rms",
        init_noise="zero",
        step_size=None,
        weight_stft=0.0,
        weight_snr=0.0,
        target_snr_db=18.0,
        selection_snr_db=None,
    ):
        self.device = resolve_torch_device(device)
        self.epsilon = float(epsilon)
        self.max_items = int(max_items)
        self.lr_optim = 1.0e-3
        self.sampling_rate = 16000
        self.paths = CheckpointPaths()

        self.weight_feature = float(weight_feature)
        self.weight_semantic = float(weight_semantic)
        self.weight_psy = float(weight_psy)
        self.weight_l2 = float(weight_l2)
        self.l2_reduction = l2_reduction
        self.init_noise = init_noise
        self.step_size = float(step_size) if step_size is not None else None
        self.weight_stft = float(weight_stft)
        self.weight_snr = float(weight_snr)
        self.target_snr_db = float(target_snr_db)
        self.selection_snr_db = (
            float(selection_snr_db) if selection_snr_db is not None else None
        )

        if self.l2_reduction not in {"rms", "norm"}:
            raise ValueError(f"Unsupported l2_reduction: {self.l2_reduction}")
        if self.init_noise not in {"zero", "random"}:
            raise ValueError(f"Unsupported init_noise: {self.init_noise}")

        if tokenizer_path is None:
            tokenizer_path = ROOT / "checkpoints" / "CosyVoice" / "speech_tokenizer_v1.onnx"
        self.tokenizer_path = str(tokenizer_path)
        self.hubert_path = hubert_path
        self.whisper_path = whisper_path
        self.use_vits = use_vits
        self.use_gsv = use_gsv
        self.use_mfcc_timbre = use_mfcc_timbre
        self.use_wavlm = use_wavlm
        self.use_cosyvoice = use_cosyvoice

        self.criterion = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        self._init_scheduler_state()

        self.hps = load_config(self.paths.vits_config)

        self.build_timbre_encoders()
        self.semantic_encoders = SemanticEncoderEnsemble(
            device=self.device,
            tokenizer_path=self.tokenizer_path,
            hubert_path=self.hubert_path,
            whisper_path=self.whisper_path,
        )

    def _init_scheduler_state(self):
        self.plateau_length = 5
        default_step = 2.0 * self.epsilon / max(self.max_items, 1)
        self.max_lr = self.step_size if self.step_size is not None else min(0.001, default_step)
        self.min_lr = 1.0e-6
        self.plateau_drop = 2.0
        self.last_losses = []
        self.lr = self.max_lr

    @staticmethod
    def _freeze(model):
        for param in model.parameters():
            param.requires_grad = False

    def build_timbre_encoders(self):
        self.active_timbre_encoders = []

        if self.use_vits:
            self.vits = build_models_vits(
                self.hps,
                checkpoint_path=str(self.paths.vits_checkpoint),
            )
            self.vits.to(self.device)
            self.encoder_vits = self.vits.enc_q
            self._freeze(self.vits)
            self.active_timbre_encoders.append("vits")

        if self.use_gsv:
            self.gsv = build_gsv_sovits_encoder(self.paths.gsv_checkpoint, self.device)
            self.encoder_gsv = self.gsv.enc_q
            self._freeze(self.gsv)
            self.active_timbre_encoders.append("gsv")

        if self.use_mfcc_timbre:
            self.encoder_mfcc = torchaudio.transforms.MFCC(
                sample_rate=16000,
                n_mfcc=40,
            ).to(self.device)
            self.active_timbre_encoders.append("mfcc")

        if self.use_wavlm:
            with legacy_weight_norm_for_transformers_audio():
                self.encoder_wavlm = AutoModel.from_pretrained(self.paths.wavlm_dir)
            self.encoder_wavlm.to(self.device)
            self._freeze(self.encoder_wavlm)
            self.active_timbre_encoders.append("wavlm")

        if self.use_cosyvoice:
            # onnx2torch 1.5.15 opens a NamedTemporaryFile twice when a path is
            # passed, which fails under Windows file-locking semantics. Loading
            # the protobuf first keeps the conversion identical and portable.
            cosyvoice_onnx = onnx.load(str(self.paths.cosyvoice_campplus))
            self.encoder_cosyvoice = convert(cosyvoice_onnx).to(self.device).eval()
            self._freeze(self.encoder_cosyvoice)
            self.active_timbre_encoders.append("cosyvoice")

    def get_vits_emb(self, wave, sr):
        spec = spectrogram_torch(wave, sampling_rate=sr)
        spec_len = torch.tensor([spec.shape[2]]).to(self.device)
        speaker_id = torch.tensor([0]).to(self.device)

        g = self.vits.emb_g(speaker_id).unsqueeze(-1)
        latent, _, _, spec_mask = self.encoder_vits(spec, spec_len, g=g)
        return self.vits.flow(latent, spec_mask, g=g)

    def get_gsv_emb(self, wave, sr):
        trans_32k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=32000).to(self.device)
        wave = trans_32k(wave)
        spec = spectrogram_torch(
            wave,
            n_fft=2048,
            sampling_rate=32000,
            hop_size=640,
            win_size=2048,
            center=False,
        )
        spec_len = torch.tensor([spec.shape[2]]).to(self.device)

        spec_mask = torch.unsqueeze(commons.sequence_mask(spec_len, spec.size(2)), 1).to(spec.dtype)
        ge = self.gsv.ref_enc(spec[:, :704] * spec_mask, spec_mask)
        latent, _, _, spec_mask = self.encoder_gsv(spec, spec_len, g=ge)
        return self.gsv.flow(latent, spec_mask, g=ge)

    def get_timbre_embeddings(self, wave, sr):
        embeddings = []

        if "vits" in self.active_timbre_encoders:
            embeddings.append(self.get_vits_emb(wave, sr))

        if "gsv" in self.active_timbre_encoders:
            embeddings.append(self.get_gsv_emb(wave, sr))

        if "mfcc" in self.active_timbre_encoders:
            embeddings.append(self.encoder_mfcc(wave))

        if "wavlm" in self.active_timbre_encoders:
            emb_wavlm = self.encoder_wavlm(
                input_values=wave,
                output_hidden_states=True,
            ).last_hidden_state
            embeddings.append(emb_wavlm.transpose(2, 1))

        if "cosyvoice" in self.active_timbre_encoders:
            feature = kaldi.fbank(
                wave,
                num_mel_bins=80,
                dither=0,
                sample_frequency=sr,
            )
            feature = feature - feature.mean(dim=0, keepdim=True)
            embeddings.append(self.encoder_cosyvoice(feature.unsqueeze(0)))

        return embeddings

    def compute_timbre_loss(self, clean_embeddings, adv_embeddings):
        loss = 0.0
        for name, clean_emb, adv_emb in zip(self.active_timbre_encoders, clean_embeddings, adv_embeddings):
            score = self.criterion(clean_emb, adv_emb).mean()

            if name == "cosyvoice":
                score = score * 0.5
            loss = loss + score
        return loss

    @staticmethod
    def calculate_snr(wave, adv_wave):
        if isinstance(wave, np.ndarray):
            wave = torch.from_numpy(wave)
        if isinstance(adv_wave, np.ndarray):
            adv_wave = torch.from_numpy(adv_wave)
        noise = adv_wave - wave
        signal_power = torch.sum(wave ** 2)
        noise_power = torch.sum(noise ** 2)
        return 10 * torch.log10(signal_power / (noise_power + 1.0e-8))

    @staticmethod
    def differentiable_snr(wave, adv_wave):
        signal_power = torch.mean(wave ** 2)
        noise_power = torch.mean((adv_wave - wave) ** 2)
        return 10.0 * torch.log10((signal_power + 1.0e-12) / (noise_power + 1.0e-12))

    def prepare_stft_targets(self, wave):
        targets = []
        for n_fft in (256, 512, 1024):
            hop_length = n_fft // 4
            window = torch.hann_window(n_fft, device=self.device, dtype=wave.dtype)
            magnitude = torch.stft(
                wave,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                return_complex=True,
            ).abs()
            targets.append((n_fft, hop_length, window, magnitude.detach()))
        return targets

    @staticmethod
    def compute_stft_loss(adv_wave, targets):
        losses = []
        for n_fft, hop_length, window, clean_magnitude in targets:
            adv_magnitude = torch.stft(
                adv_wave,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                return_complex=True,
            ).abs()
            spectral_convergence = torch.linalg.vector_norm(adv_magnitude - clean_magnitude) / (
                torch.linalg.vector_norm(clean_magnitude) + 1.0e-8
            )
            log_magnitude = F.l1_loss(
                torch.log1p(adv_magnitude),
                torch.log1p(clean_magnitude),
            )
            losses.append(spectral_convergence + log_magnitude)
        return torch.stack(losses).mean()

    @staticmethod
    def save_audio(output_path, wave, sr):
        if isinstance(wave, torch.Tensor):
            wave = wave.detach().cpu().numpy()
        sf.write(str(output_path), wave, samplerate=sr)

    @staticmethod
    def _check_cancelled(cancel_event):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("TASK_CANCELLED")

    def effective_config(self):
        return {
            "epsilon": self.epsilon,
            "max_steps": self.max_items,
            "weight_feature": self.weight_feature,
            "weight_semantic": self.weight_semantic,
            "weight_psy": self.weight_psy,
            "weight_l2": self.weight_l2,
            "weight_stft": self.weight_stft,
            "weight_snr": self.weight_snr,
            "target_snr_db": self.target_snr_db,
            "selection_snr_db": self.selection_snr_db,
            "step_size": self.max_lr,
            "init_noise": self.init_noise,
            "l2_reduction": self.l2_reduction,
            "min_lr": self.min_lr,
            "semantic_encoders": list(self.semantic_encoders.vector_names),
            "timbre_encoders": list(self.active_timbre_encoders),
            "models": {
                "tokenizer": self.tokenizer_path,
                "hubert": str(self.hubert_path),
                "whisper": str(self.whisper_path),
            },
            "checkpoints": {
                "vits": str(self.paths.vits_checkpoint),
                "gpt_sovits": str(self.paths.gsv_checkpoint),
                "wavlm": str(self.paths.wavlm_dir),
                "cosyvoice_campplus": str(self.paths.cosyvoice_campplus),
            },
        }

    def protect(
        self,
        input_wav,
        output_wav=None,
        verbose=False,
        progress_callback=None,
        cancel_event=None,
    ):
        input_wav = Path(input_wav).resolve()
        if output_wav is None:
            output_wav = input_wav.with_name(f"{input_wav.stem}_semantic.wav")
        output_wav = Path(output_wav).resolve()

        self._check_cancelled(cancel_event)
        wave_np, sr = sf.read(
            str(input_wav),
            dtype="float32",
            always_2d=True,
        )
        wave_np = wave_np.mean(axis=1, dtype=np.float32)
        wave = torch.from_numpy(np.ascontiguousarray(wave_np)).unsqueeze(0)
        original_length = wave.shape[1]
        if original_length < sr:
            wave = torch.cat((wave, torch.zeros((1, sr - original_length))), dim=1)
        wave = wave.to(self.device)

        if self.init_noise == "random":
            perturbation = torch.randn_like(wave)
            perturbation = perturbation / max(perturbation.abs().max().item(), 1.0e-8)
            perturbation = perturbation * self.epsilon
        else:
            perturbation = torch.zeros_like(wave)
        perturbation.requires_grad = True

        transform_16k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(self.device)
        wave_16k = transform_16k(wave)

        # The clean branches are fixed reference features; keeping their
        # forward graphs provides no gradient and wastes memory, especially
        # with HuBERT-large and Whisper-large-v3.
        with torch.no_grad():
            clean_timbre = self.get_timbre_embeddings(wave_16k, 16000)
            clean_semantic = self.semantic_encoders.get_vectors(wave_16k)

        masker = Masker(device=self.device, sample_rate=sr)
        theta, original_max_psd = masker._compute_masking_threshold(wave[0].detach().cpu().numpy())
        theta_batch = torch.FloatTensor(theta.transpose(1, 0)).unsqueeze(0).to(self.device)
        original_max_psd_batch = torch.FloatTensor([original_max_psd]).unsqueeze(0).to(self.device)

        stft_targets = self.prepare_stft_targets(wave) if self.weight_stft > 0.0 else []
        self._init_scheduler_state()
        best_wave = None
        best_attack_loss = float("inf")
        best_step = None
        best_loss_items = None
        loss_items = {}
        optimization_trace = []
        step_times = []
        for step in tqdm(range(self.max_items)):
            self._check_cancelled(cancel_event)
            step_started = time.perf_counter()
            step_number = step + 1
            adv_wave = torch.clamp(wave + perturbation, min=-1.0, max=1.0)
            adv_wave_16k = transform_16k(adv_wave)

            adv_timbre = self.get_timbre_embeddings(adv_wave_16k, 16000)
            loss_timbre = self.compute_timbre_loss(clean_timbre, adv_timbre)

            adv_semantic = self.semantic_encoders.get_vectors(adv_wave_16k)
            loss_semantic = self.semantic_encoders.compute_loss(clean_semantic, adv_semantic)

            loss_psy = masker.batch_forward_2nd_stage(
                local_delta_rescale=(adv_wave - wave).squeeze(1),
                theta_batch=theta_batch,
                original_max_psd_batch=original_max_psd_batch,
            )
            loss_psy_max = 1.0e7
            if torch.isnan(loss_psy) or torch.isinf(loss_psy) or loss_psy > loss_psy_max:
                loss_psy = torch.FloatTensor([loss_psy_max]).to(self.device)

            if self.l2_reduction == "rms":
                loss_l2 = torch.sqrt(torch.mean((adv_wave - wave) ** 2) + 1.0e-12)
            else:
                loss_l2 = torch.norm(adv_wave - wave, p=2)
            loss_stft = (
                self.compute_stft_loss(adv_wave, stft_targets)
                if stft_targets
                else torch.zeros((), device=self.device)
            )
            snr_db = self.differentiable_snr(wave, adv_wave)
            loss_snr = F.relu(self.target_snr_db - snr_db) ** 2
            attack_loss = (
                loss_timbre * self.weight_feature
                + loss_semantic * self.weight_semantic
            )
            loss = (
                attack_loss
                + loss_psy * self.weight_psy
                + loss_l2 * self.weight_l2
                + loss_stft * self.weight_stft
                + loss_snr * self.weight_snr
            )
            loss_items = {
                "loss_timbre": f"{loss_timbre.item():.6f}",
                "loss_semantic": f"{loss_semantic.item():.6f}",
                "loss_psy": f"{loss_psy.item():.6f}",
                "loss_l2": f"{loss_l2.item():.6f}",
                "loss_stft": f"{loss_stft.item():.6f}",
                "loss_snr": f"{loss_snr.item():.6f}",
                "snr_db": f"{snr_db.item():.6f}",
            }

            if self.selection_snr_db is not None and snr_db.item() >= self.selection_snr_db:
                attack_value = attack_loss.item()
                if attack_value < best_attack_loss:
                    best_attack_loss = attack_value
                    best_wave = adv_wave.detach().clone()
                    best_step = step_number
                    best_loss_items = dict(loss_items)

            if verbose and step % 50 == 0:
                print(step, loss_items, self.lr)

            loss.backward()

            self.last_losses.append(loss.item())
            self.last_losses = self.last_losses[-self.plateau_length:]
            if len(self.last_losses) == self.plateau_length and self.last_losses[-1] > self.last_losses[0]:
                if self.lr > self.min_lr:
                    self.lr = max(self.lr / self.plateau_drop, self.min_lr)
                self.last_losses = []

            with torch.no_grad():
                perturbation -= self.lr * torch.sign(perturbation.grad)
                perturbation.clamp_(min=-self.epsilon, max=self.epsilon)
                perturbation.copy_(torch.clamp(wave + perturbation, -1.0, 1.0) - wave)
            perturbation.grad = None

            step_elapsed_sec = time.perf_counter() - step_started
            step_times.append(step_elapsed_sec)
            trace_point = {
                "step": step_number,
                "total_steps": self.max_items,
                "progress": step_number / max(self.max_items, 1),
                "current_lr": float(self.lr),
                "total_loss": float(loss.item()),
                "feature_loss": float(loss_timbre.item()),
                "semantic_loss": float(loss_semantic.item()),
                "psychoacoustic_loss": float(loss_psy.item()),
                "l2_loss": float(loss_l2.item()),
                "stft_loss": float(loss_stft.item()),
                "snr_loss": float(loss_snr.item()),
                "current_snr_db": float(snr_db.item()),
                "Lid": float(loss_timbre.item()),
                "Lfeat": float(loss_timbre.item()),
                "Lsem": float(loss_semantic.item()),
                "Lpsy": float(loss_psy.item()),
                "L2": float(loss_l2.item()),
                "total": float(loss.item()),
                "snr": float(snr_db.item()),
                "stepElapsedSec": step_elapsed_sec,
            }
            optimization_trace.append(trace_point)
            if progress_callback is not None:
                progress_callback(
                    step=step_number,
                    total_steps=self.max_items,
                    total=self.max_items,
                    progress=trace_point["progress"],
                    current_lr=trace_point["current_lr"],
                    total_loss=trace_point["total_loss"],
                    feature_loss=trace_point["feature_loss"],
                    semantic_loss=trace_point["semantic_loss"],
                    psychoacoustic_loss=trace_point["psychoacoustic_loss"],
                    l2_loss=trace_point["l2_loss"],
                    stft_loss=trace_point["stft_loss"],
                    snr_loss=trace_point["snr_loss"],
                    current_snr_db=trace_point["current_snr_db"],
                    loss_items=dict(loss_items),
                    trace=trace_point,
                )
            self._check_cancelled(cancel_event)

        final_wave = torch.clamp(wave + perturbation, min=-1.0, max=1.0).detach()
        if self.selection_snr_db is not None and best_wave is not None:
            final_wave = best_wave
            loss_items = best_loss_items
        elif best_step is None:
            best_step = self.max_items

        snr = self.calculate_snr(wave, final_wave).item()
        final_wave = final_wave[:, :original_length]
        result = final_wave.cpu().numpy()[0]
        self.save_audio(output_wav, result, sr)

        effective_config = self.effective_config()

        return {
            "output_wav": str(output_wav),
            "snr": snr,
            "snr_db": snr,
            "selected_step": best_step,
            "loss_items": loss_items,
            "optimization_trace": optimization_trace,
            "average_step_sec": sum(step_times) / len(step_times) if step_times else None,
            "epsilon": self.epsilon,
            "max_steps": self.max_items,
            "effective_config": effective_config,
            "models": effective_config["models"],
            "checkpoints": effective_config["checkpoints"],
        }
