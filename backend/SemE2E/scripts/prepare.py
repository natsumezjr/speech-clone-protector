from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import shutil
from pathlib import Path

import soundfile as sf
from huggingface_hub import hf_hub_download, snapshot_download

from core.utils import read_csv_rows, write_csv_rows


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def _download_files(repo_id: str, files: list[str], download_path: Path) -> None:
    download_path.mkdir(parents=True, exist_ok=True)
    for filename in files:
        target = download_path / filename
        if target.is_file() and target.stat().st_size > 0:
            print(f"Reusing {target}")
            continue
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(download_path),
            local_dir_use_symlinks=False,
        )
        print(f"Downloaded {filename}")


def _model_directory_ready(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    return any(next(path.glob(pattern), None) is not None for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth"))


def _ensure_hf_snapshot(repo_id: str, target: Path) -> Path:
    if _model_directory_ready(target):
        print(f"Reusing {target}")
        return target
    try:
        cached = Path(snapshot_download(repo_id=repo_id, local_files_only=True))
        if _model_directory_ready(cached):
            print(f"Reusing Hugging Face cache {cached}")
            return cached
    except Exception:
        pass
    print(f"Downloading missing snapshot {repo_id} -> {target}")
    return Path(snapshot_download(repo_id=repo_id, local_dir=str(target), local_dir_use_symlinks=False))


def download_models(_args: argparse.Namespace) -> None:
    print("Downloading final semantic surrogate models...")
    for repo_id, target in (
        ("facebook/hubert-large-ll60k", ROOT / "checkpoints" / "hf" / "facebook" / "hubert-large-ll60k"),
        ("openai/whisper-large-v3", ROOT / "checkpoints" / "hf" / "openai" / "whisper-large-v3"),
    ):
        _ensure_hf_snapshot(repo_id, target)

    print("Downloading GPT-SoVITS SoVITS checkpoint...")
    _download_files(
        "lj1995/GPT-SoVITS",
        ["gsv-v2final-pretrained/s2G2333k.pth"],
        ROOT / "checkpoints" / "GSV" / "base_models",
    )
    print("Downloading WavLM...")
    _ensure_hf_snapshot("microsoft/wavlm-base-plus", ROOT / "checkpoints" / "wavlm")
    print("Downloading CosyVoice encoders...")
    cosyvoice_dir = ROOT / "checkpoints" / "CosyVoice" / "base_models" / "CosyVoice-300M"
    _download_files(
        "FunAudioLLM/CosyVoice-300M",
        ["campplus.onnx", "speech_tokenizer_v1.onnx"],
        cosyvoice_dir,
    )
    tokenizer_source = cosyvoice_dir / "speech_tokenizer_v1.onnx"
    tokenizer_target = ROOT / "checkpoints" / "CosyVoice" / "speech_tokenizer_v1.onnx"
    tokenizer_target.parent.mkdir(parents=True, exist_ok=True)
    if not tokenizer_target.is_file() or tokenizer_target.stat().st_size == 0:
        shutil.copy2(tokenizer_source, tokenizer_target)
    else:
        print(f"Reusing {tokenizer_target}")

    print("Downloading VITS checkpoint...")
    _download_files(
        "csukuangfj/vits-ljs",
        ["pretrained_ljs.pth"],
        ROOT / "checkpoints" / "VITS",
    )


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip()).strip("_") or "item"


def _iter_libritts_rows(args: argparse.Namespace):
    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    os.environ.setdefault("HF_DATASETS_OFFLINE", "0")
    from datasets import Audio, load_dataset

    dataset = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    for row_index, row in enumerate(dataset):
        if row_index < args.offset:
            continue
        if row_index >= args.max_scan:
            break
        yield row_index, row


def fetch_dataset(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.resolve()
    selected = []
    for row_index, row in _iter_libritts_rows(args):
        text = row["text_normalized"].strip()
        if not args.min_text_chars <= len(text) <= args.max_text_chars:
            continue
        wav_path = args.output_dir / "wavs" / f"{_slug(row['id'])}.wav"
        audio_bytes = (row.get("audio") or {}).get("bytes")
        if not audio_bytes:
            raise ValueError(f"row {row.get('id', '<unknown>')} has no audio bytes")
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.write_bytes(audio_bytes)
        info = sf.info(str(wav_path))
        duration = float(info.frames) / float(info.samplerate)
        if not args.min_duration <= duration <= args.max_duration:
            continue
        selected.append(
            {
                "row_idx": row_index,
                "id": row["id"],
                "speaker_id": row["speaker_id"],
                "chapter_id": row["chapter_id"],
                "split": args.split,
                "audio": str(wav_path.resolve()),
                "duration_s": f"{duration:.4f}",
                "text_normalized": text,
                "text_original": row["text_original"].strip(),
                "source_path": row["path"],
            }
        )
        print(f"selected {row['id']}: {duration:.2f}s | {text}", flush=True)
        if len(selected) >= args.max_items:
            break
    if not selected:
        raise SystemExit("no rows selected")
    manifest = args.manifest or (args.output_dir / "manifest.csv")
    write_csv_rows(
        selected,
        manifest,
        ["row_idx", "id", "speaker_id", "chapter_id", "split", "audio", "duration_s", "text_normalized", "text_original", "source_path"],
    )
    print(f"Wrote {manifest}")


def prepare_listening_test(args: argparse.Namespace) -> None:
    rows = read_csv_rows(args.manifest.resolve(), required={"id", "condition", "audio"})
    by_sample = {}
    for row in rows:
        by_sample.setdefault(row["id"], {})[row["condition"]] = row
    eligible = sorted(
        sample_id
        for sample_id, condition_rows in by_sample.items()
        if all(condition in condition_rows for condition in args.conditions)
    )
    if args.sample_count < 1 or len(eligible) < args.sample_count:
        raise ValueError(f"requested {args.sample_count} samples but only {len(eligible)} are eligible")

    random_generator = random.Random(args.seed)
    selected = random_generator.sample(eligible, args.sample_count)
    trials = [(sample_id, condition) for sample_id in selected for condition in args.conditions]
    random_generator.shuffle(trials)
    output_dir = args.output_dir.resolve()
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    reference_paths = {}
    for index, sample_id in enumerate(selected, start=1):
        clean_row = by_sample[sample_id].get("clean")
        if clean_row is None:
            raise ValueError(f"sample {sample_id} has no clean reference")
        source = Path(clean_row.get("clean_audio") or clean_row["audio"]).resolve()
        destination = audio_dir / f"R{index:03d}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        reference_paths[sample_id] = destination

    response_rows = []
    key_rows = []
    for index, (sample_id, condition) in enumerate(trials, start=1):
        row = by_sample[sample_id][condition]
        source = Path(row["audio"]).resolve()
        trial_id = f"T{index:03d}"
        destination = audio_dir / f"{trial_id}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        reference = reference_paths[sample_id].relative_to(output_dir).as_posix()
        candidate = destination.relative_to(output_dir).as_posix()
        response_rows.append(
            {
                "trial_id": trial_id,
                "reference_audio": reference,
                "candidate_audio": candidate,
                "quality_mos_1_5": "",
                "content_consistency_1_5": "",
                "timbre_similarity_1_5": "",
                "notes": "",
            }
        )
        key_rows.append(
            {
                "trial_id": trial_id,
                "sample_id": sample_id,
                "condition": condition,
                "reference_audio": reference,
                "candidate_audio": candidate,
                "reference_text": row.get("reference_text", ""),
            }
        )
    write_csv_rows(
        response_rows,
        output_dir / "response_sheet.csv",
        [
            "trial_id",
            "reference_audio",
            "candidate_audio",
            "quality_mos_1_5",
            "content_consistency_1_5",
            "timbre_similarity_1_5",
            "notes",
        ],
    )
    write_csv_rows(
        key_rows,
        output_dir / "answer_key.csv",
        [
            "trial_id",
            "sample_id",
            "condition",
            "reference_audio",
            "candidate_audio",
            "reference_text",
        ],
    )
    with (output_dir / "protocol.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "manifest": str(args.manifest.resolve()),
                "conditions": args.conditions,
                "sample_count": args.sample_count,
                "trial_count": len(trials),
                "seed": args.seed,
                "selected_sample_ids": selected,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Wrote {len(trials)} blinded trials to {output_dir}")


def _cached_hf_snapshot(repo_id: str, project_path: Path | None = None) -> Path | None:
    if project_path is not None and _model_directory_ready(project_path):
        return project_path.resolve()
    try:
        path = Path(snapshot_download(repo_id=repo_id, local_files_only=True))
        return path.resolve() if _model_directory_ready(path) else None
    except Exception:
        return None


def _directory_size(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _huggingface_revision(path: Path | None) -> str | None:
    if path is None:
        return None
    base = path if path.is_dir() else path.parent
    for candidate_base in (base, base.parent):
        metadata_root = candidate_base / ".cache" / "huggingface" / "download"
        if not metadata_root.is_dir():
            continue
        for metadata_path in sorted(metadata_root.rglob("*.metadata")):
            try:
                first_line = metadata_path.read_text(encoding="utf-8").splitlines()[0].strip()
            except (IndexError, OSError, UnicodeError):
                continue
            if re.fullmatch(r"[0-9a-f]{40}", first_line):
                return first_line
    return None


def _manifest_entry(
    *,
    feature: str,
    model_name: str,
    source: str,
    expected_path: Path | None,
    found_path: Path | None,
    required_files: list[str],
    module: str | None = None,
) -> dict[str, object]:
    resolved = found_path.resolve() if found_path is not None and found_path.exists() else None
    missing_files: list[str] = []
    if resolved is not None:
        base = resolved if resolved.is_dir() else resolved.parent
        missing_files = [item for item in required_files if not (base / item).is_file()]
    module_ready = module is None or importlib.util.find_spec(module) is not None
    if resolved is None:
        status = "missing"
    elif missing_files or not module_ready:
        status = "incomplete"
    else:
        status = "ready"
    revision = None
    if resolved is not None and resolved.parent.name == "snapshots":
        revision = resolved.name
    if revision is None:
        revision = _huggingface_revision(resolved)
    return {
        "feature": feature,
        "model_name": model_name,
        "source": source,
        "revision": revision,
        "expected_path": str(expected_path.resolve()) if expected_path is not None else None,
        "found_path": str(resolved) if resolved is not None else None,
        "status": status,
        "required_files": required_files,
        "missing_files": missing_files,
        "module": module,
        "module_available": module_ready,
        "size_bytes": _directory_size(resolved),
        "load_test": "not_run",
        "load_error": None,
    }


def build_checkpoint_manifest() -> list[dict[str, object]]:
    checkpoint_root = ROOT / "checkpoints"
    tts_root = checkpoint_root / "tts"
    whisper_cache = Path(os.getenv("WHISPER_CACHE_DIR") or (Path.home() / ".cache" / "whisper"))
    modelscope_root = Path(
        os.getenv("MODELSCOPE_CACHE") or (Path.home() / ".cache" / "modelscope")
    )
    paraformer = (
        modelscope_root
        / "hub"
        / "models"
        / "iic"
        / "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    )
    hubert_large = _cached_hf_snapshot(
        "facebook/hubert-large-ll60k",
        checkpoint_root / "hf" / "facebook" / "hubert-large-ll60k",
    )
    whisper_large = _cached_hf_snapshot(
        "openai/whisper-large-v3",
        checkpoint_root / "hf" / "openai" / "whisper-large-v3",
    )
    ecapa = checkpoint_root / "ecapa"
    entries = [
        _manifest_entry(feature="protection", model_name="VITS", source="csukuangfj/vits-ljs", expected_path=checkpoint_root / "VITS" / "pretrained_ljs.pth", found_path=checkpoint_root / "VITS" / "pretrained_ljs.pth", required_files=["pretrained_ljs.pth"]),
        _manifest_entry(feature="protection", model_name="GPT-SoVITS", source="lj1995/GPT-SoVITS", expected_path=checkpoint_root / "GSV" / "base_models" / "gsv-v2final-pretrained", found_path=checkpoint_root / "GSV" / "base_models" / "gsv-v2final-pretrained", required_files=["s2G2333k.pth"]),
        _manifest_entry(feature="protection", model_name="WavLM", source="microsoft/wavlm-base-plus", expected_path=checkpoint_root / "wavlm", found_path=checkpoint_root / "wavlm", required_files=["config.json", "preprocessor_config.json", "pytorch_model.bin"]),
        _manifest_entry(feature="protection", model_name="CosyVoice CAM++", source="FunAudioLLM/CosyVoice-300M", expected_path=checkpoint_root / "CosyVoice" / "base_models" / "CosyVoice-300M", found_path=checkpoint_root / "CosyVoice" / "base_models" / "CosyVoice-300M", required_files=["campplus.onnx"]),
        _manifest_entry(feature="tokenizer", model_name="S3 Tokenizer encoder", source="FunAudioLLM/CosyVoice-300M", expected_path=checkpoint_root / "CosyVoice" / "speech_tokenizer_v1.onnx", found_path=checkpoint_root / "CosyVoice" / "speech_tokenizer_v1.onnx", required_files=["speech_tokenizer_v1.onnx"], module="s3tokenizer"),
        _manifest_entry(feature="protection", model_name="HuBERT-large", source="facebook/hubert-large-ll60k", expected_path=checkpoint_root / "hf" / "facebook" / "hubert-large-ll60k", found_path=hubert_large, required_files=["config.json", "preprocessor_config.json"], module="transformers"),
        _manifest_entry(feature="protection", model_name="Whisper-large-v3", source="openai/whisper-large-v3", expected_path=checkpoint_root / "hf" / "openai" / "whisper-large-v3", found_path=whisper_large, required_files=["config.json", "preprocessor_config.json"], module="transformers"),
        _manifest_entry(feature="protection", model_name="MFCC", source="torchaudio.transforms.MFCC", expected_path=None, found_path=Path(importlib.util.find_spec("torchaudio").origin).parent if importlib.util.find_spec("torchaudio") else None, required_files=[], module="torchaudio"),
        _manifest_entry(feature="asr", model_name="Whisper Small (Transformers)", source="openai/whisper-small", expected_path=checkpoint_root / "asr" / "openai-whisper-small", found_path=checkpoint_root / "asr" / "openai-whisper-small", required_files=["config.json", "preprocessor_config.json", "tokenizer.json", "model.safetensors"], module="transformers"),
        _manifest_entry(feature="asr", model_name="OpenAI Whisper Tiny", source="openai-whisper:tiny", expected_path=whisper_cache / "tiny.pt", found_path=whisper_cache / "tiny.pt", required_files=["tiny.pt"], module="whisper"),
        _manifest_entry(feature="asr", model_name="OpenAI Whisper Base", source="openai-whisper:base", expected_path=whisper_cache / "base.pt", found_path=whisper_cache / "base.pt", required_files=["base.pt"], module="whisper"),
        _manifest_entry(feature="asr", model_name="FunASR Paraformer", source="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", expected_path=paraformer, found_path=paraformer, required_files=["configuration.json", "config.yaml", "model.pt", "tokens.json", "am.mvn"], module="funasr"),
        _manifest_entry(feature="similarity", model_name="ECAPA-TDNN", source="speechbrain/spkrec-ecapa-voxceleb", expected_path=ecapa, found_path=ecapa, required_files=["hyperparams.yaml", "embedding_model.ckpt", "classifier.ckpt", "label_encoder.ckpt", "mean_var_norm_emb.ckpt"], module="speechbrain"),
        _manifest_entry(feature="cloning", model_name="XTTS-v2", source="tts_models/multilingual/multi-dataset/xtts_v2", expected_path=tts_root / "tts_models--multilingual--multi-dataset--xtts_v2", found_path=tts_root / "tts_models--multilingual--multi-dataset--xtts_v2", required_files=["config.json", "model.pth", "vocab.json", "speakers_xtts.pth"], module="TTS"),
        _manifest_entry(feature="cloning", model_name="XTTS-v1.1", source="tts_models/multilingual/multi-dataset/xtts_v1.1", expected_path=tts_root / "tts_models--multilingual--multi-dataset--xtts_v1.1", found_path=tts_root / "tts_models--multilingual--multi-dataset--xtts_v1.1", required_files=["config.json", "model.pth", "vocab.json"], module="TTS"),
        _manifest_entry(feature="cloning", model_name="YourTTS", source="tts_models/multilingual/multi-dataset/your_tts", expected_path=tts_root / "tts_models--multilingual--multi-dataset--your_tts", found_path=tts_root / "tts_models--multilingual--multi-dataset--your_tts", required_files=["config.json", "model_file.pth", "model_se.pth", "speakers.json", "language_ids.json"], module="TTS"),
        _manifest_entry(feature="perception", model_name="PESQ", source="pesq package", expected_path=None, found_path=Path(importlib.util.find_spec("pesq").origin).parent if importlib.util.find_spec("pesq") else None, required_files=[], module="pesq"),
        _manifest_entry(feature="perception", model_name="STOI", source="pystoi package", expected_path=None, found_path=Path(importlib.util.find_spec("pystoi").origin).parent if importlib.util.find_spec("pystoi") else None, required_files=[], module="pystoi"),
    ]
    return entries


def write_checkpoint_manifest(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = build_checkpoint_manifest()
    load_results_path = args.load_results.resolve()
    if load_results_path.is_file():
        load_payload = json.loads(load_results_path.read_text(encoding="utf-8-sig"))
        load_results = {
            str(item.get("model_name")): item
            for item in load_payload.get("results", [])
            if isinstance(item, dict) and item.get("model_name")
        }
        for entry in entries:
            load_result = load_results.get(str(entry["model_name"]))
            if load_result is None:
                continue
            entry["load_test"] = load_result.get("status", "not_run")
            entry["load_error"] = load_result.get("error")
            entry["load_evidence"] = load_result.get("evidence")
            entry["load_elapsed_sec"] = load_result.get("elapsed_sec")
            if entry["status"] == "ready" and entry["load_test"] not in {
                "load_ok",
                "inference_ok",
                "inference_and_backward_ok",
            }:
                entry["status"] = "load_failed"
    payload = {
        "generated_by": "python -m scripts.prepare manifest",
        "root": str(ROOT),
        "load_results": str(load_results_path) if load_results_path.is_file() else None,
        "entries": entries,
        "summary": {
            status: sum(1 for item in entries if item["status"] == status)
            for status in ("ready", "missing", "incomplete", "load_failed")
        },
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {output}")


def inspect_runtime(_args: argparse.Namespace) -> None:
    from audio_preprocess import audio_preprocess_capabilities

    capabilities = audio_preprocess_capabilities()
    print(json.dumps({"audioPreprocessing": capabilities}, ensure_ascii=False, indent=2))
    if not capabilities["recordingSupported"]:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model, dataset, and listening-test preparation.")
    commands = parser.add_subparsers(dest="command", required=True)

    models = commands.add_parser("models", help="Download project model weights.")
    models.set_defaults(func=download_models)

    dataset = commands.add_parser("dataset", help="Fetch a small LibriTTS subset.")
    dataset.add_argument("--dataset", default="mythicinfinity/libritts")
    dataset.add_argument("--split", default="dev.clean")
    dataset.add_argument("--config", default="all")
    dataset.add_argument("--hf_endpoint", default=DEFAULT_HF_ENDPOINT)
    dataset.add_argument("--offset", type=int, default=0)
    dataset.add_argument("--max_scan", type=int, default=500)
    dataset.add_argument("--max_items", type=int, default=5)
    dataset.add_argument("--min_duration", type=float, default=2.0)
    dataset.add_argument("--max_duration", type=float, default=8.0)
    dataset.add_argument("--min_text_chars", type=int, default=25)
    dataset.add_argument("--max_text_chars", type=int, default=180)
    dataset.add_argument("--output_dir", type=Path, default=ROOT / "data" / "libritts_subset")
    dataset.add_argument("--manifest", type=Path, default=None)
    dataset.set_defaults(func=fetch_dataset)

    listening = commands.add_parser("listening", help="Create a blinded paired listening set.")
    listening.add_argument("--manifest", type=Path, required=True)
    listening.add_argument("--conditions", nargs="+", required=True)
    listening.add_argument("--sample_count", type=int, default=20)
    listening.add_argument("--seed", type=int, default=20260804)
    listening.add_argument("--output_dir", type=Path, required=True)
    listening.set_defaults(func=prepare_listening_test)

    manifest = commands.add_parser("manifest", help="Inspect local model/checkpoint completeness without downloading.")
    manifest.add_argument(
        "--output",
        type=Path,
        default=ROOT.parents[1] / "seme2e-runtime" / "diagnostics" / "checkpoint-manifest.json",
    )
    manifest.add_argument(
        "--load-results",
        type=Path,
        default=ROOT.parents[1] / "seme2e-runtime" / "diagnostics" / "model-load-results.json",
    )
    manifest.set_defaults(func=write_checkpoint_manifest)

    runtime = commands.add_parser("runtime", help="Verify runtime dependencies required for browser recordings.")
    runtime.set_defaults(func=inspect_runtime)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
