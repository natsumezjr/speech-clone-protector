from __future__ import annotations

import argparse
from pathlib import Path

from core.evaluation import (
    run_asr_manifest,
    run_robustness_manifest,
    run_speaker_manifest,
    run_tts,
)


ROOT = Path(__file__).resolve().parents[1]


def add_asr_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--asr_models", default="openai/whisper-small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default=str(ROOT / "outputs" / "eval"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic E2E-VGuard evaluation tools.")
    commands = parser.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser("asr", help="Evaluate an audio manifest with ASR and quality metrics.")
    manifest.add_argument("--manifest", required=True)
    add_asr_options(manifest)
    manifest.set_defaults(func=run_asr_manifest)

    tts = commands.add_parser("tts", help="Evaluate TTS output with ASR and speaker similarity.")
    tts.add_argument("--manifest", required=True)
    tts.add_argument("--asr_models", default="openai/whisper-small")
    tts.add_argument("--speaker_metric", choices=["ecapa", "wavlm"], default="ecapa")
    tts.add_argument("--speaker_model", default=None)
    tts.add_argument("--device", default="cuda")
    tts.add_argument("--num_shards", type=int, default=1)
    tts.add_argument("--shard_index", type=int, default=0)
    tts.add_argument("--output_dir", default=str(ROOT / "outputs" / "eval"))
    tts.set_defaults(func=run_tts)

    speaker = commands.add_parser("speaker", help="Evaluate speaker similarity for a manifest.")
    speaker.add_argument("--manifest", type=Path, required=True)
    speaker.add_argument("--speaker_metric", choices=["ecapa", "wavlm"], default="ecapa")
    speaker.add_argument("--speaker_model", default=None)
    speaker.add_argument("--device", default="cuda")
    speaker.add_argument("--output_dir", type=Path, default=ROOT / "outputs" / "speaker_eval")
    speaker.set_defaults(func=run_speaker_manifest)

    robustness = commands.add_parser("robustness", help="Create transformed robustness audio and manifest.")
    robustness.add_argument("--manifest", type=Path, required=True)
    robustness.add_argument("--output_dir", type=Path, required=True)
    robustness.add_argument("--output_manifest", type=Path, default=None)
    robustness.add_argument(
        "--conditions",
        default="mp3_128k,resample_8k,lowpass_4k,noise_20db",
    )
    robustness.add_argument("--source_condition", default="semantic_e50")
    robustness.add_argument("--include_source", action="store_true")
    robustness.add_argument("--limit", type=int, default=None)
    robustness.set_defaults(func=run_robustness_manifest)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
