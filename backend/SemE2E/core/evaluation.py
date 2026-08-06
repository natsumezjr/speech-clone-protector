from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy import signal

from core.utils import parse_model_list, read_csv_rows, resolve_torch_device, write_csv_rows, write_json_csv_results


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_WAVLM_MODEL = str(ROOT / "checkpoints" / "wavlm")
QUALITY_METRIC_SR = 16000


def load_mono(path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    y, in_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr is not None and in_sr != sr:
        y = librosa.resample(y, orig_sr=in_sr, target_sr=sr)
        in_sr = sr
    return y, in_sr


def perceptual_quality_metrics(clean_path: str | Path, audio_path: str | Path) -> dict[str, float]:
    from pesq import pesq
    from pystoi import stoi

    clean, _ = load_mono(clean_path, QUALITY_METRIC_SR)
    audio, _ = load_mono(audio_path, QUALITY_METRIC_SR)

    n = min(len(clean), len(audio))
    clean = clean[:n]
    audio = audio[:n]
    return {
        "pesq_wb": float(pesq(QUALITY_METRIC_SR, clean, audio, "wb")),
        "stoi": float(stoi(clean, audio, QUALITY_METRIC_SR, extended=False)),
    }


def audio_metrics(clean_path: str | Path, audio_path: str | Path) -> dict[str, float]:
    clean, sr_clean = load_mono(clean_path)
    audio, _ = load_mono(audio_path, sr_clean)

    n = min(len(clean), len(audio))
    clean = clean[:n]
    audio = audio[:n]
    noise = audio - clean

    signal_power = float(np.sum(clean * clean) + 1.0e-12)
    noise_power = float(np.sum(noise * noise) + 1.0e-12)
    corr = float(np.corrcoef(clean, audio)[0, 1]) if n > 1 else 0.0

    metrics = {
        "duration_s": round(n / sr_clean, 4),
        "snr_db": 10.0 * np.log10(signal_power / noise_power),
        "noise_linf": float(np.max(np.abs(noise))) if n else 0.0,
        "noise_rms": float(np.sqrt(np.mean(noise * noise))) if n else 0.0,
        "wave_corr": corr,
    }
    metrics.update(perceptual_quality_metrics(clean_path, audio_path))
    return metrics


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(reference: Sequence[object], hypothesis: Sequence[object]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for index, reference_item in enumerate(reference, start=1):
        current = [index]
        for offset, hypothesis_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[offset] + 1,
                    current[offset - 1] + 1,
                    previous[offset - 1] + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def wer(reference: str, hypothesis: str) -> float:
    reference_words = normalize_text(reference).split()
    hypothesis_words = normalize_text(hypothesis).split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return edit_distance(reference_words, hypothesis_words) / len(reference_words)


def cer(reference: str, hypothesis: str) -> float:
    reference_chars = list(normalize_text(reference).replace(" ", ""))
    hypothesis_chars = list(normalize_text(hypothesis).replace(" ", ""))
    if not reference_chars:
        return 0.0 if not hypothesis_chars else 1.0
    return edit_distance(reference_chars, hypothesis_chars) / len(reference_chars)


class ASRTranscriber:
    """Wrapper over the ASR backends used by the evaluation pipeline."""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = resolve_torch_device(device)
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        if model_name.startswith("openai-whisper:"):
            self.backend = "openai-whisper"
            import whisper

            self.model = whisper.load_model(model_name.split(":", 1)[1], device=str(self.device))
        else:
            self.backend = "transformers"
            self._init_transformers(model_name)

    def _init_transformers(self, model_name: str) -> None:
        from contextlib import nullcontext
        from transformers import pipeline

        from core.utils import legacy_weight_norm_for_transformers_audio

        self.is_whisper = "whisper" in model_name.lower()
        device_id = (self.device.index or 0) if self.device.type == "cuda" else -1
        dtype = torch.float16 if device_id >= 0 and self.is_whisper else torch.float32
        weight_norm_context = (
            legacy_weight_norm_for_transformers_audio()
            if "wav2vec2" in model_name.lower()
            else nullcontext()
        )
        with weight_norm_context:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                device=device_id,
                torch_dtype=dtype,
            )

    def transcribe(self, audio_path: str | Path) -> str:
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
        return self.pipe(str(audio_path), **kwargs)["text"].strip()


def cosine_score(reference_embedding: torch.Tensor, candidate_embedding: torch.Tensor) -> float:
    return float(F.cosine_similarity(reference_embedding, candidate_embedding, dim=-1).item())


class WavLMSpeakerSimilarity:
    def __init__(self, model_path: str | Path, device: str = "cuda"):
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

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
        return cosine_score(self.embed(reference_audio), self.embed(candidate_audio))


class ECAPASpeakerSimilarity:
    def __init__(self, model_path: str | Path, device: str = "cuda"):
        self.device = resolve_torch_device(device)
        self.device_name = str(self.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device_name = "cuda:0"
        from speechbrain.inference.speaker import EncoderClassifier
        self.model = EncoderClassifier.from_hparams(
            source=str(model_path),
            savedir=str(ROOT / "checkpoints" / "ecapa"),
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
        return cosine_score(self.embed(reference_audio), self.embed(candidate_audio))


def build_speaker_similarity(metric: str, model_path: str | Path, device: str = "cuda"):
    if metric == "ecapa":
        return ECAPASpeakerSimilarity(model_path, device)
    if metric == "wavlm":
        return WavLMSpeakerSimilarity(model_path, device)
    raise ValueError(f"unsupported speaker similarity metric: {metric}")


def default_speaker_model(metric: str) -> str:
    if metric == "ecapa":
        return DEFAULT_ECAPA_MODEL
    if metric == "wavlm":
        return DEFAULT_WAVLM_MODEL
    raise ValueError(f"unsupported speaker similarity metric: {metric}")


def _cached_audio_metrics(cache, clean_path: Path, audio_path: Path) -> dict[str, float]:
    key = (clean_path, audio_path)
    if key not in cache:
        cache[key] = audio_metrics(clean_path, audio_path)
    return cache[key]


def run_asr_manifest(args) -> None:
    model_names = parse_model_list(args.asr_models)
    transcribers = [ASRTranscriber(model_name, args.device) for model_name in model_names]
    rows = []
    metric_cache = {}
    manifest_rows = read_csv_rows(
        args.manifest,
        required={"id", "condition", "audio", "reference_text"},
    )
    for item in manifest_rows:
        audio_path = Path(item["audio"]).resolve()
        clean_path = Path(item.get("clean_audio") or item["audio"]).resolve()
        for transcriber in transcribers:
            hypothesis = transcriber.transcribe(audio_path)
            row = {
                "task": "asr",
                "sample_id": item["id"],
                "model": transcriber.model_name,
                "condition": item["condition"],
                "audio": str(audio_path),
                "clean_audio": str(clean_path),
                "reference": item["reference_text"],
                "hypothesis": hypothesis,
                "wer": wer(item["reference_text"], hypothesis),
                "cer": cer(item["reference_text"], hypothesis),
            }
            row.update(_cached_audio_metrics(metric_cache, clean_path, audio_path))
            rows.append(row)
    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "reference_text_source": "manifest",
        "models": model_names,
        "rows": rows,
    }
    write_json_csv_results(args.output_dir, "asr_manifest_results", rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _shard_manifest_rows(rows, shard_index: int, num_shards: int):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards) and num_shards must be positive")
    sample_ids = list(dict.fromkeys(row.get("sample_id") or row.get("id") or "" for row in rows))
    assigned_ids = set(sample_ids[shard_index::num_shards])
    return [row for row in rows if (row.get("sample_id") or row.get("id") or "") in assigned_ids]


def run_tts(args) -> None:
    model_names = parse_model_list(args.asr_models)
    transcribers = [ASRTranscriber(model_name, args.device) for model_name in model_names]
    speaker_model = args.speaker_model or default_speaker_model(args.speaker_metric)
    speaker = build_speaker_similarity(args.speaker_metric, speaker_model, args.device)
    manifest_rows = read_csv_rows(
        args.manifest,
        required={
            "sample_id",
            "condition",
            "synth_audio",
            "target_text",
            "reference_audio",
            "similarity_reference_audio",
        },
    )
    manifest_rows = _shard_manifest_rows(manifest_rows, args.shard_index, args.num_shards)
    reference_cache = {}
    rows = []
    for item in manifest_rows:
        synth_audio = Path(item["synth_audio"]).resolve()
        reference_audio = Path(item["reference_audio"]).resolve()
        similarity_audio = Path(item["similarity_reference_audio"]).resolve()
        if similarity_audio not in reference_cache:
            reference_cache[similarity_audio] = speaker.embed(similarity_audio)
        speaker_score = cosine_score(reference_cache[similarity_audio], speaker.embed(synth_audio))
        for transcriber in transcribers:
            hypothesis = transcriber.transcribe(synth_audio)
            rows.append(
                {
                    "task": "tts",
                    "sample_id": item["sample_id"],
                    "model": transcriber.model_name,
                    "condition": item["condition"],
                    "reference_audio": str(reference_audio),
                    "similarity_reference_audio": str(similarity_audio),
                    "similarity_reference_source": "manifest",
                    "synth_audio": str(synth_audio),
                    "target_text": item["target_text"],
                    "hypothesis": hypothesis,
                    "wer": wer(item["target_text"], hypothesis),
                    "cer": cer(item["target_text"], hypothesis),
                    "speaker_metric": args.speaker_metric,
                    "speaker_similarity": speaker_score,
                }
            )
    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "speaker_metric": args.speaker_metric,
        "speaker_model": speaker_model,
        "models": model_names,
        "rows": rows,
    }
    write_json_csv_results(args.output_dir, "tts_results", rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_speaker_manifest(args) -> None:
    speaker_model = args.speaker_model or default_speaker_model(args.speaker_metric)
    scorer = build_speaker_similarity(args.speaker_metric, speaker_model, args.device)
    rows = []
    for item in read_csv_rows(
        args.manifest,
        required={"id", "condition", "audio", "clean_audio"},
    ):
        row = {
            "task": "speaker_manifest",
            "sample_id": item["id"],
            "condition": item["condition"],
            "audio": str(Path(item["audio"]).resolve()),
            "clean_audio": str(Path(item["clean_audio"]).resolve()),
            "speaker_metric": args.speaker_metric,
            "speaker_model": speaker_model,
            "speaker_similarity": scorer.score(item["clean_audio"], item["audio"]),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "speaker_metric": args.speaker_metric,
        "speaker_model": speaker_model,
        "rows": rows,
    }
    write_json_csv_results(args.output_dir, "speaker_manifest_results", rows, summary)


def _write_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(waveform, -1.0, 1.0), sample_rate)


def _transform_audio(waveform: np.ndarray, sample_rate: int, condition: str, seed: int):
    if condition.startswith("mp3_"):
        bitrate = condition.removeprefix("mp3_")
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_dir = Path(temporary_dir)
            source = temporary_dir / "source.wav"
            compressed = temporary_dir / "compressed.mp3"
            restored = temporary_dir / "restored.wav"
            sf.write(str(source), waveform, sample_rate)
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-b:a", bitrate, str(compressed)],
                check=True,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(compressed), "-ar", str(sample_rate), "-ac", "1", str(restored)],
                check=True,
            )
            return load_mono(restored, sr=sample_rate)
    if condition == "resample_8k":
        downsampled = librosa.resample(waveform, orig_sr=sample_rate, target_sr=8000)
        return librosa.resample(downsampled, orig_sr=8000, target_sr=sample_rate).astype(np.float32), sample_rate
    if condition == "lowpass_4k":
        sos = signal.butter(8, 4000.0, btype="lowpass", fs=sample_rate, output="sos")
        return signal.sosfiltfilt(sos, waveform).astype(np.float32), sample_rate
    if condition == "noise_20db":
        random_generator = np.random.default_rng(seed)
        noise = random_generator.standard_normal(len(waveform)).astype(np.float32)
        signal_power = float(np.mean(waveform * waveform) + 1.0e-12)
        noise_power = float(np.mean(noise * noise) + 1.0e-12)
        return waveform + noise * (signal_power / (100.0 * noise_power)) ** 0.5, sample_rate
    raise ValueError(f"unknown transform condition: {condition}")


def run_robustness_manifest(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = args.output_manifest or (args.output_dir / "robust_manifest.csv")
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    source_rows = [
        row
        for row in read_csv_rows(
            args.manifest,
            required={"id", "condition", "audio", "clean_audio", "reference_text"},
        )
        if row["condition"] == args.source_condition
    ]
    if args.limit:
        source_rows = source_rows[: args.limit]
    rows = list(source_rows) if args.include_source else []
    for index, row in enumerate(source_rows):
        waveform, sample_rate = load_mono(row["audio"], sr=16000)
        for condition in conditions:
            output_audio = args.output_dir / condition / row["id"] / f"{row['id']}_{condition}.wav"
            transformed, output_rate = _transform_audio(waveform, sample_rate, condition, seed=index)
            _write_wav(output_audio, transformed, output_rate)
            rows.append({**row, "condition": condition, "audio": str(output_audio.resolve())})
            print(f"{row['id']} {condition} -> {output_audio}", flush=True)
    write_csv_rows(
        rows,
        output_manifest,
        ["id", "condition", "clean_audio", "audio", "reference_text", "duration_s", "source_split", "source_path"],
    )
    print(f"Wrote {output_manifest} ({len(rows)} rows)")
