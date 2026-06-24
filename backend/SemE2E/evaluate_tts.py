from __future__ import annotations

import argparse
import json
from pathlib import Path

from asr_backends import ASRTranscriber
from evaluation_common import parse_model_list
from io_utils import read_csv_rows, write_json_csv_results
from speaker_similarity import build_speaker_similarity
from text_metrics import cer, wer


ROOT = Path(__file__).resolve().parent
DEFAULT_ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_WAVLM_MODEL = str(ROOT / "checkpoints" / "wavlm")


def load_original_reference_map(path: str | Path | None) -> dict[str, str]:
    """Load sample_id -> original clean speaker audio for speaker-sim scoring."""
    if not path:
        return {}

    mapping = {}
    for row in read_csv_rows(path):
        sample_id = row.get("sample_id") or row.get("id")
        if not sample_id:
            continue
        clean_audio = row.get("clean_audio")
        if clean_audio:
            mapping[sample_id] = clean_audio
            continue
        if row.get("condition") == "clean":
            reference = row.get("audio") or row.get("reference_audio")
            if reference:
                mapping[sample_id] = reference
    return mapping


def resolve_similarity_reference(
    item: dict[str, str],
    args: argparse.Namespace,
    original_refs: dict[str, str],
    tts_reference_audio: Path,
    distinct_sample_ids: set[str],
) -> tuple[Path, str]:
    if args.similarity_reference_mode == "tts_reference":
        return tts_reference_audio, "tts_reference"

    for column in ("similarity_reference_audio", "clean_audio"):
        value = item.get(column)
        if value:
            return Path(value).resolve(), column

    sample_id = item.get("sample_id") or item.get("id")
    if sample_id and sample_id in original_refs:
        return Path(original_refs[sample_id]).resolve(), "similarity_reference_manifest"

    if args.clean_reference and len(distinct_sample_ids) <= 1:
        return Path(args.clean_reference).resolve(), "single_sample_clean_reference"

    raise ValueError(
        "cannot resolve original clean speaker reference for ECAPA-SIM. "
        "Pass --similarity_reference_manifest with id/sample_id and clean_audio columns, "
        "or include similarity_reference_audio in the TTS manifest."
    )


def run_tts(args: argparse.Namespace) -> None:
    model_names = parse_model_list(args.asr_models)
    asr_transcribers = [ASRTranscriber(model_name, args.device) for model_name in model_names]
    speaker_model = args.speaker_model or default_speaker_model(args.speaker_metric)
    speaker = build_speaker_similarity(args.speaker_metric, speaker_model, args.device)

    rows = []
    manifest_rows = read_csv_rows(args.manifest, required={"condition", "synth_audio", "target_text"})
    distinct_sample_ids = {row.get("sample_id") or row.get("id") or "" for row in manifest_rows}
    distinct_sample_ids.discard("")
    original_refs = load_original_reference_map(args.similarity_reference_manifest)
    for item in manifest_rows:
        condition = item["condition"]
        synth_audio = Path(item["synth_audio"]).resolve()
        target_text = item["target_text"]
        ref_audio = Path(item.get("reference_audio") or args.clean_reference).resolve()
        sim_ref_audio, sim_ref_source = resolve_similarity_reference(
            item,
            args,
            original_refs,
            ref_audio,
            distinct_sample_ids,
        )

        speaker_score = speaker.score(sim_ref_audio, synth_audio)
        for transcriber in asr_transcribers:
            hypothesis = transcriber.transcribe(synth_audio)
            rows.append(
                {
                    "task": "tts",
                    "model": transcriber.model_name,
                    "condition": condition,
                    "reference_audio": str(ref_audio),
                    "similarity_reference_audio": str(sim_ref_audio),
                    "similarity_reference_source": sim_ref_source,
                    "synth_audio": str(synth_audio),
                    "target_text": target_text,
                    "hypothesis": hypothesis,
                    "wer": wer(target_text, hypothesis),
                    "cer": cer(target_text, hypothesis),
                    "speaker_metric": args.speaker_metric,
                    "speaker_similarity": speaker_score,
                }
            )

    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "similarity_reference_mode": args.similarity_reference_mode,
        "similarity_reference_manifest": (
            str(Path(args.similarity_reference_manifest).resolve())
            if args.similarity_reference_manifest
            else None
        ),
        "speaker_metric": args.speaker_metric,
        "speaker_model": speaker_model,
        "models": model_names,
        "rows": rows,
    }
    write_json_csv_results(args.output_dir, "tts_results", rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TTS downstream evaluation for Semantic E2E-VGuard.")
    parser.add_argument("--manifest", required=True, help="CSV with condition,synth_audio,target_text[,reference_audio].")
    parser.add_argument("--clean_reference", default=None, help="Fallback speaker reference audio for a single-sample manifest.")
    parser.add_argument("--asr_models", default="openai/whisper-small")
    parser.add_argument(
        "--similarity_reference_mode",
        choices=["original_clean", "tts_reference"],
        default="original_clean",
        help="Use original clean speaker audio for SIM by default; tts_reference reproduces the legacy prompt-reference SIM.",
    )
    parser.add_argument(
        "--similarity_reference_manifest",
        default=None,
        help="CSV with id/sample_id and clean_audio columns, such as protected_e50.csv.",
    )
    parser.add_argument("--speaker_metric", choices=["ecapa", "wavlm"], default="ecapa")
    parser.add_argument("--speaker_model", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default=str(ROOT / "outputs" / "eval"))
    parser.set_defaults(func=run_tts)
    return parser


def default_speaker_model(metric: str) -> str:
    if metric == "ecapa":
        return DEFAULT_ECAPA_MODEL
    if metric == "wavlm":
        return DEFAULT_WAVLM_MODEL
    raise ValueError(f"unsupported speaker similarity metric: {metric}")


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
