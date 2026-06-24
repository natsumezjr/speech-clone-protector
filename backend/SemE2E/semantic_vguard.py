import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
warnings.filterwarnings("ignore")

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.optim as optim
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torch.nn.functional as F

multi_head_attention_forward = F.multi_head_attention_forward

from onnx2torch import convert
from tqdm import tqdm
from transformers import AutoModel

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

sys.path.append(str(ROOT / "tts_models" / "vits"))
from tts_models.vits.mel_processing import spectrogram_torch
import tts_models.vits.commons as commons
import tts_models.vits.utils as utils

sys.path.append(str(ROOT / "tts_models" / "gsv"))
sys.path.append(str(ROOT / "tts_models" / "gsv" / "GPT_SoVITS"))
from tts_models.gsv.GPT_SoVITS.module.models import SynthesizerTrn as GSVSynthesizerTrn

from masker import Masker
from runtime import resolve_torch_device
from semantic_encoders import SemanticEncoderEnsemble
from toolbox import build_models_styletts2, build_models_vits

F.multi_head_attention_forward = multi_head_attention_forward


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(f"Attribute {item} not found") from exc


@dataclass(frozen=True)
class CheckpointPaths:
    vits_config: Path = ROOT / "tts_models" / "configs" / "LibriTTS_VITS.json"
    vits_checkpoint: Path = ROOT / "checkpoints" / "VITS" / "pretrained_ljs.pth"
    gsv_checkpoint: Path = ROOT / "checkpoints" / "GSV" / "base_models" / "gsv-v2final-pretrained" / "s2G2333k.pth"
    wavlm_dir: Path = ROOT / "checkpoints" / "wavlm"
    cosyvoice_campplus: Path = ROOT / "checkpoints" / "CosyVoice" / "base_models" / "CosyVoice-300M" / "campplus.onnx"
    style_config: Path = ROOT / "checkpoints" / "StyleTTS2" / "base_models" / "config.yml"
    style_checkpoint: Path = ROOT / "checkpoints" / "StyleTTS2" / "base_models" / "epochs_2nd_00020.pth"
    speaker_database: Path = ROOT / "data" / "speakers_database"


def build_gsv_sovits_encoder(sovits_path, device):
    checkpoint = torch.load(str(sovits_path), map_location="cpu")
    hps = DictToAttrRecursive(checkpoint["config"])
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


class SemanticE2EVGuard:
    """E2E-VGuard with Lasr replaced by T-SemAttack Lsem."""

    def __init__(
        self,
        epsilon=8 / 255,
        max_items=500,
        device=None,
        timbre_mode="untargeted",
        tokenizer_path=None,
        hubert_path="facebook/hubert-large-ls960-ft",
        whisper_path="openai/whisper-large-v3",
        use_vits=True,
        use_gsv=True,
        use_mfcc_timbre=True,
        use_wavlm=True,
        use_cosyvoice=True,
        use_style=True,
        weight_feature=500.0,
        weight_semantic=100.0,
        weight_psy=1.0e-5,
        weight_l2=0.1,
    ):
        self.device = resolve_torch_device(device)
        self.epsilon = float(epsilon)
        self.max_items = int(max_items)
        self.timbre_mode = timbre_mode
        self.lr_optim = 1.0e-3
        self.sampling_rate = 16000
        self.paths = CheckpointPaths()

        self.weight_feature = float(weight_feature)
        self.weight_semantic = float(weight_semantic)
        self.weight_psy = float(weight_psy)
        self.weight_l2 = float(weight_l2)

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
        self.use_style = use_style

        self.criterion = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        self._init_scheduler_state()

        self.hps = utils.get_hparams_from_file(config_path=str(self.paths.vits_config))

        self.build_timbre_encoders()
        self.semantic_encoders = SemanticEncoderEnsemble(
            device=self.device,
            tokenizer_path=self.tokenizer_path,
            hubert_path=self.hubert_path,
            whisper_path=self.whisper_path,
        )

    def _init_scheduler_state(self):
        self.plateau_length = 5
        self.max_lr = 0.001
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

        if self.use_vits and self.paths.vits_checkpoint.exists():
            self.vits = build_models_vits(
                self.hps,
                checkpoint_path=str(self.paths.vits_checkpoint),
            )[0]
            self.vits.to(self.device)
            self.encoder_vits = self.vits.enc_q
            self.active_timbre_encoders.append("vits")
        else:
            print(f"Skip VITS timbre encoder: missing {self.paths.vits_checkpoint}")

        if self.use_gsv and self.paths.gsv_checkpoint.exists():
            self.gsv = build_gsv_sovits_encoder(self.paths.gsv_checkpoint, self.device)
            self.encoder_gsv = self.gsv.enc_q
            self.active_timbre_encoders.append("gsv")
        else:
            print(f"Skip GPT-SoVITS timbre encoder: missing {self.paths.gsv_checkpoint}")

        if self.use_mfcc_timbre:
            self.encoder_mfcc = torchaudio.transforms.MFCC(
                sample_rate=16000,
                n_mfcc=40,
            ).to(self.device)
            self.active_timbre_encoders.append("mfcc")

        if self.use_wavlm:
            wavlm_model_id = (
                str(self.paths.wavlm_dir)
                if (self.paths.wavlm_dir / "pytorch_model.bin").exists()
                else "microsoft/wavlm-base-plus"
            )
            self.encoder_wavlm = AutoModel.from_pretrained(wavlm_model_id)
            self.encoder_wavlm.to(self.device)
            self.active_timbre_encoders.append("wavlm")

        if self.use_cosyvoice and self.paths.cosyvoice_campplus.exists():
            try:
                self.encoder_cosyvoice = convert(str(self.paths.cosyvoice_campplus)).to(self.device)
                self.encoder_cosyvoice.eval()
                self.active_timbre_encoders.append("cosyvoice")
            except Exception as exc:
                print(f"Skip CosyVoice timbre encoder: ONNX conversion failed: {exc}")
        else:
            print(f"Skip CosyVoice timbre encoder: missing {self.paths.cosyvoice_campplus}")

        if self.use_style and self.paths.style_config.exists() and self.paths.style_checkpoint.exists():
            styletts2 = build_models_styletts2(
                str(self.paths.style_config),
                str(self.paths.style_checkpoint),
                device=self.device,
            )[0]
            self.encoder_style = styletts2.style_encoder
            self.active_timbre_encoders.append("style")
        else:
            print(f"Skip StyleTTS2 style encoder: missing {self.paths.style_checkpoint}")

        for attr in [
            "vits",
            "encoder_vits",
            "gsv",
            "encoder_gsv",
            "encoder_wavlm",
            "encoder_cosyvoice",
            "encoder_style",
        ]:
            if hasattr(self, attr):
                self._freeze(getattr(self, attr))

    def get_vits_emb(self, wave, sr):
        spec = spectrogram_torch(wave, sampling_rate=sr)
        spec_len = torch.tensor([spec.shape[2]]).to(self.device)
        speaker_id = torch.tensor([0]).to(self.device)

        try:
            g = self.vits.emb_g(speaker_id).unsqueeze(-1)
        except Exception:
            g = None
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

        if "style" in self.active_timbre_encoders:
            to_mel = torchaudio.transforms.MelSpectrogram(
                n_mels=80,
                n_fft=2048,
                win_length=1200,
                hop_length=300,
            ).to(self.device)
            wave_24k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000).to(self.device)(wave)
            mel = to_mel(wave_24k)
            mel = (torch.log(1.0e-5 + mel.unsqueeze(0)) + 4) / 4
            embeddings.append(self.encoder_style(mel))

        return embeddings

    def compute_timbre_loss(self, clean_embeddings, adv_embeddings):
        loss = 0.0
        for name, clean_emb, adv_emb in zip(self.active_timbre_encoders, clean_embeddings, adv_embeddings):
            try:
                score = self.criterion(clean_emb, adv_emb).mean()
            except RuntimeError:
                min_len = min(clean_emb.shape[2], adv_emb.shape[2])
                clean_emb = clean_emb[:, :, :min_len]
                adv_emb = adv_emb[:, :, :min_len]
                score = self.criterion(clean_emb, adv_emb).mean()

            if name == "cosyvoice":
                score = score * 0.5
            loss = loss + score
        return loss

    def select_target_speaker(self, clean_embeddings):
        min_loss = float("inf")
        best_path = None
        best_embeddings = None

        for speaker_path in self.paths.speaker_database.glob("*.wav"):
            wave, sr = librosa.load(str(speaker_path), sr=16000)
            wave = torch.from_numpy(wave).unsqueeze(0).to(self.device)
            embeddings = self.get_timbre_embeddings(wave, sr)
            loss = self.compute_timbre_loss(clean_embeddings, embeddings)
            if loss < min_loss:
                min_loss = loss
                best_path = speaker_path
                best_embeddings = embeddings

        return best_path, best_embeddings

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
    def save_audio(output_path, wave, sr):
        if isinstance(wave, torch.Tensor):
            wave = wave.detach().cpu().numpy()
        sf.write(str(output_path), wave, samplerate=sr)

    def protect(self, input_wav, output_wav=None, verbose=False):
        input_wav = Path(input_wav).resolve()
        if output_wav is None:
            output_wav = input_wav.with_name(f"{input_wav.stem}_semantic.wav")
        output_wav = Path(output_wav).resolve()

        wave, sr = torchaudio.load(str(input_wav))
        original_length = wave.shape[1]
        if original_length < sr:
            wave = torch.cat((wave, torch.zeros((1, sr - original_length))), dim=1)
        wave = wave.to(self.device)

        noise = torch.randn_like(wave)
        noise = noise / max(noise.abs().max().item(), 1.0e-8) * self.epsilon
        adv_wave = torch.clamp(wave + noise, min=-1.0, max=1.0)
        adv_wave.requires_grad = True

        transform_16k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(self.device)
        wave_16k = transform_16k(wave)

        clean_timbre = self.get_timbre_embeddings(wave_16k, 16000)
        clean_semantic = self.semantic_encoders.get_vectors(wave_16k)

        target_timbre = None
        target_speaker = None
        if self.timbre_mode == "targeted":
            target_speaker, target_timbre = self.select_target_speaker(clean_timbre)
            print(f"Selected target speaker: {target_speaker}")

        optimizer = optim.SGD([adv_wave], lr=self.lr_optim, weight_decay=0.95)

        masker = Masker(device=self.device, sample_rate=sr)
        theta, original_max_psd = masker._compute_masking_threshold(wave[0].detach().cpu().numpy())
        theta_batch = torch.FloatTensor(theta.transpose(1, 0)).unsqueeze(0).to(self.device)
        original_max_psd_batch = torch.FloatTensor([original_max_psd]).unsqueeze(0).to(self.device)

        loss_items = {}
        optimization_trace = []
        trace_every = max(1, self.max_items // 20)
        for step in tqdm(range(self.max_items)):
            adv_wave_16k = transform_16k(adv_wave)

            adv_timbre = self.get_timbre_embeddings(adv_wave_16k, 16000)
            if self.timbre_mode == "targeted":
                loss_timbre = self.compute_timbre_loss(target_timbre, adv_timbre) * -1.0
            else:
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

            loss_l2 = torch.norm(adv_wave - wave, p=2)
            loss = (
                loss_timbre * self.weight_feature
                + loss_semantic * self.weight_semantic
                + loss_psy * self.weight_psy
                + loss_l2 * self.weight_l2
            )
            loss_items = {
                "loss_timbre": f"{loss_timbre.item():.6f}",
                "loss_semantic": f"{loss_semantic.item():.6f}",
                "loss_psy": f"{loss_psy.item():.6f}",
                "loss_l2": f"{loss_l2.item():.6f}",
                "loss_total": f"{loss.item():.6f}",
            }

            if step == 0 or step == self.max_items - 1 or step % trace_every == 0:
                optimization_trace.append(
                    {
                        "step": step,
                        "Lfea": float(loss_timbre.item()),
                        "Lsem": float(loss_semantic.item()),
                        "Lpsy": float(loss_psy.item()),
                        "L2": float(loss_l2.item()),
                        "total": float(loss.item()),
                        "snr": float(self.calculate_snr(wave, adv_wave.detach()).item()),
                    }
                )

            if verbose and step % 50 == 0:
                print(step, loss_items, self.lr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.last_losses.append(loss.detach())
            self.last_losses = self.last_losses[-self.plateau_length:]
            if len(self.last_losses) == self.plateau_length and self.last_losses[-1] > self.last_losses[0]:
                if self.lr > self.min_lr:
                    self.lr = max(self.lr / self.plateau_drop, self.min_lr)
                self.last_losses = []

            delta = self.lr * torch.sign(adv_wave.grad) * -1.0
            adv_wave = delta + adv_wave
            noise = torch.clamp(adv_wave.data - wave, min=-self.epsilon, max=self.epsilon)
            adv_wave = torch.clamp(wave + noise, min=-1.0, max=1.0)
            adv_wave.requires_grad = True

        snr = self.calculate_snr(wave, adv_wave.detach()).item()
        adv_wave = adv_wave[:, :original_length]
        result = adv_wave.detach().cpu().numpy()[0]
        self.save_audio(output_wav, result, sr)

        return {
            "output_wav": str(output_wav),
            "snr": snr,
            "loss_items": loss_items,
            "optimization_trace": optimization_trace,
            "target_speaker": str(target_speaker) if target_speaker else None,
        }
