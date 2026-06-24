from __future__ import annotations

import argparse
import json
from pathlib import Path

from asr_backends import ASRTranscriber
from audio_utils import audio_metrics
from evaluation_common import parse_model_list
from io_utils import read_csv_rows, write_json_csv_results
from text_metrics import cer, wer


ROOT = Path(__file__).resolve().parent


def run_asr(args: argparse.Namespace) -> None:
    clean = Path(args.clean).resolve()
    audios = [clean] + [Path(path).resolve() for path in args.audios]
    labels = args.labels if args.labels else ["clean"] + [Path(path).stem for path in args.audios]
    if len(labels) != len(audios):
        raise ValueError("labels count must match clean + audios count")

    model_names = parse_model_list(args.asr_models)
    rows = []
    summary = {
        "clean": str(clean),
        "reference_text_source": "provided" if args.reference_text else "clean_transcription",
        "models": model_names,
        "rows": rows,
    }

    for model_name in model_names:
        transcriber = ASRTranscriber(model_name, args.device)
        reference_text = args.reference_text or transcriber.transcribe(clean)
        for label, audio_path in zip(labels, audios):
            hypothesis = (
                reference_text
                if args.reference_text is None and audio_path == clean
                else transcriber.transcribe(audio_path)
            )
            row = {
                "task": "asr",
                "model": model_name,
                "condition": label,
                "audio": str(audio_path),
                "reference": reference_text,
                "hypothesis": hypothesis,
                "wer": wer(reference_text, hypothesis),
                "cer": cer(reference_text, hypothesis),
            }
            row.update(audio_metrics(clean, audio_path))
            rows.append(row)

    write_json_csv_results(args.output_dir, "asr_results", rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_asr_manifest(args: argparse.Namespace) -> None:
    model_names = parse_model_list(args.asr_models)
    transcribers = [ASRTranscriber(model_name, args.device) for model_name in model_names]

    rows = []
    manifest_rows = read_csv_rows(args.manifest, required={"id", "condition", "audio", "reference_text"})
    for item in manifest_rows:
        audio_path = Path(item["audio"]).resolve()
        clean_path = Path(item.get("clean_audio") or item["audio"]).resolve()
        reference_text = item["reference_text"]
        for transcriber in transcribers:
            hypothesis = transcriber.transcribe(audio_path)
            row = {
                "task": "asr",
                "sample_id": item["id"],
                "model": transcriber.model_name,
                "condition": item["condition"],
                "audio": str(audio_path),
                "clean_audio": str(clean_path),
                "reference": reference_text,
                "hypothesis": hypothesis,
                "wer": wer(reference_text, hypothesis),
                "cer": cer(reference_text, hypothesis),
            }
            row.update(audio_metrics(clean_path, audio_path))
            rows.append(row)

    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "reference_text_source": "manifest",
        "models": model_names,
        "rows": rows,
    }
    write_json_csv_results(args.output_dir, "asr_manifest_results", rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASR threat-model evaluation for Semantic E2E-VGuard.")
    sub = parser.add_subparsers(dest="command", required=True)

    single = sub.add_parser("single", aliases=["asr"], help="Evaluate protected audio against downstream ASR.")
    single.add_argument("--clean", required=True, help="Clean source audio.")
    single.add_argument("--audios", nargs="+", required=True, help="Protected audios to evaluate.")
    single.add_argument("--labels", nargs="+", default=None, help="Labels for clean plus protected audios.")
    single.add_argument("--reference_text", default=None, help="Ground-truth transcript. If omitted, clean ASR is used.")
    single.add_argument("--asr_models", default="openai/whisper-small")
    single.add_argument("--device", default="cuda")
    single.add_argument("--output_dir", default=str(ROOT / "outputs" / "eval"))
    single.set_defaults(func=run_asr)

    manifest = sub.add_parser(
        "manifest",
        aliases=["asr_manifest"],
        help="Evaluate multiple audios with per-sample transcripts.",
    )
    manifest.add_argument("--manifest", required=True, help="CSV with id,condition,audio,reference_text[,clean_audio].")
    manifest.add_argument("--asr_models", default="openai/whisper-small")
    manifest.add_argument("--device", default="cuda")
    manifest.add_argument("--output_dir", default=str(ROOT / "outputs" / "eval"))
    manifest.set_defaults(func=run_asr_manifest)

    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
