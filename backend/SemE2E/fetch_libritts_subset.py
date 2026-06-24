import argparse
import csv
import os
import re
import sys
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parent
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def slug(text):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "item"


def wav_duration(path):
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def iter_libritts_rows(args):
    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    os.environ.setdefault("HF_DATASETS_OFFLINE", "0")

    from datasets import Audio, load_dataset

    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=True,
    )
    dataset = dataset.cast_column("audio", Audio(decode=False))
    for row_idx, row in enumerate(dataset):
        if row_idx < args.offset:
            continue
        if row_idx >= args.max_scan:
            break
        yield row_idx, row


def write_audio(row, path):
    audio = row.get("audio") or {}
    audio_bytes = audio.get("bytes")
    if not audio_bytes:
        raise ValueError(f"row {row.get('id', '<unknown>')} has no audio bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size <= 44:
        path.write_bytes(audio_bytes)


def choose_items(args):
    selected = []
    for row_idx, row in iter_libritts_rows(args):
        text = row["text_normalized"].strip()
        if not (args.min_text_chars <= len(text) <= args.max_text_chars):
            continue

        audio_id = row["id"]
        wav_path = args.output_dir / "wavs" / f"{slug(audio_id)}.wav"
        write_audio(row, wav_path)
        duration = wav_duration(wav_path)
        if not (args.min_duration <= duration <= args.max_duration):
            continue

        selected.append(
            {
                "row_idx": row_idx,
                "id": audio_id,
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
        print(f"selected {audio_id}: {duration:.2f}s | {text}", flush=True)
        if len(selected) >= args.max_items:
            break
    return selected


def write_manifest(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "row_idx",
        "id",
        "speaker_id",
        "chapter_id",
        "split",
        "audio",
        "duration_s",
        "text_normalized",
        "text_original",
        "source_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch a small known-transcript LibriTTS subset.")
    parser.add_argument("--dataset", default="mythicinfinity/libritts")
    parser.add_argument("--split", default="dev.clean")
    parser.add_argument("--config", default="all")
    parser.add_argument("--hf_endpoint", default=DEFAULT_HF_ENDPOINT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max_scan", type=int, default=500)
    parser.add_argument("--max_items", type=int, default=5)
    parser.add_argument("--min_duration", type=float, default=2.0)
    parser.add_argument("--max_duration", type=float, default=8.0)
    parser.add_argument("--min_text_chars", type=int, default=25)
    parser.add_argument("--max_text_chars", type=int, default=180)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "data" / "libritts_subset")
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    manifest = args.manifest or (args.output_dir / "manifest.csv")
    rows = choose_items(args)
    if not rows:
        raise SystemExit("no rows selected")
    write_manifest(rows, manifest)
    print(f"Wrote {manifest}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
