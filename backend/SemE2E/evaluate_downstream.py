from __future__ import annotations

import argparse

from evaluate_asr import run_asr, run_asr_manifest
from evaluate_tts import run_tts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compatibility CLI for ASR/TTS downstream evaluation. "
        "Prefer evaluate_asr.py and evaluate_tts.py for new experiments."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    asr = sub.add_parser("asr", help="Evaluate protected audio against downstream ASR.")
    asr.add_argument("--clean", required=True, help="Clean source audio.")
    asr.add_argument("--audios", nargs="+", required=True, help="Protected audios to evaluate.")
    asr.add_argument("--labels", nargs="+", default=None, help="Labels for clean plus protected audios.")
    asr.add_argument("--reference_text", default=None, help="Ground-truth transcript. If omitted, clean ASR is used.")
    asr.add_argument("--asr_models", default="openai/whisper-small")
    asr.add_argument("--device", default="cuda")
    asr.add_argument("--output_dir", default="outputs/eval")
    asr.set_defaults(func=run_asr)

    asr_manifest = sub.add_parser("asr_manifest", help="Evaluate multiple audios with per-sample transcripts.")
    asr_manifest.add_argument("--manifest", required=True, help="CSV with id,condition,audio,reference_text[,clean_audio].")
    asr_manifest.add_argument("--asr_models", default="openai/whisper-small")
    asr_manifest.add_argument("--device", default="cuda")
    asr_manifest.add_argument("--output_dir", default="outputs/eval")
    asr_manifest.set_defaults(func=run_asr_manifest)

    tts = sub.add_parser("tts", help="Evaluate TTS outputs generated from clean/protected references.")
    tts.add_argument("--manifest", required=True, help="CSV with condition,synth_audio,target_text[,reference_audio].")
    tts.add_argument("--clean_reference", default=None, help="Fallback speaker reference audio for a single-sample manifest.")
    tts.add_argument("--asr_models", default="openai/whisper-small")
    tts.add_argument("--similarity_reference_mode", choices=["original_clean", "tts_reference"], default="original_clean")
    tts.add_argument("--similarity_reference_manifest", default=None)
    tts.add_argument("--speaker_metric", choices=["ecapa", "wavlm"], default="ecapa")
    tts.add_argument("--speaker_model", default=None)
    tts.add_argument("--device", default="cuda")
    tts.add_argument("--output_dir", default="outputs/eval")
    tts.set_defaults(func=run_tts)

    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
