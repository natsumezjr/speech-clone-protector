import argparse
from pathlib import Path

import soundfile as sf

from io_utils import read_csv_rows, write_csv_rows


ROOT = Path(__file__).resolve().parent


def read_rows(path, limit=None):
    rows = read_csv_rows(path)
    return rows[:limit] if limit else rows


def write_rows(rows, path):
    columns = [
        "condition",
        "sample_id",
        "synth_audio",
        "target_text",
        "reference_audio",
        "similarity_reference_audio",
        "prompt_text",
        "tts_backend",
    ]
    write_csv_rows(rows, path, columns)


def patch_torch_load_for_coqui_xtts():
    import torch

    if getattr(torch.load, "_semantic_vguard_patched", False):
        return

    original_load = torch.load

    def load_with_legacy_default(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    load_with_legacy_default._semantic_vguard_patched = True
    torch.load = load_with_legacy_default


def parse_args():
    parser = argparse.ArgumentParser(description="Generate TTS samples with XTTS-v2 speaker references.")
    parser.add_argument("--references", type=Path, required=True, help="CSV from run_semantic_batch.py.")
    parser.add_argument("--target_text", required=True)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs" / "tts_xtts")
    parser.add_argument("--output_manifest", type=Path, default=None)
    parser.add_argument("--model_name", default="tts_models/multilingual/multi-dataset/xtts_v2")
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    out_manifest = args.output_manifest or (args.output_dir / "tts_manifest.csv")

    patch_torch_load_for_coqui_xtts()
    from TTS.api import TTS

    tts = TTS(args.model_name)
    if args.device.startswith("cuda"):
        tts = tts.to("cuda")

    rows = []
    for item in read_rows(args.references.resolve(), args.limit):
        ref_audio = Path(item["audio"]).resolve()
        out_dir = args.output_dir / item["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        synth_audio = out_dir / f"{item['id']}_{item['condition']}_xtts.wav"
        print(f"XTTS {item['id']} {item['condition']} -> {synth_audio}")
        wav = tts.tts(
            text=args.target_text,
            speaker_wav=str(ref_audio),
            language=args.language,
        )
        sf.write(str(synth_audio), wav, 24000)
        rows.append(
            {
                "condition": item["condition"],
                "sample_id": item["id"],
                "synth_audio": str(synth_audio.resolve()),
                "target_text": args.target_text,
                "reference_audio": str(ref_audio),
                "similarity_reference_audio": str(Path(item.get("clean_audio") or item["audio"]).resolve()),
                "prompt_text": item.get("reference_text", ""),
                "tts_backend": args.model_name,
            }
        )

    write_rows(rows, out_manifest)
    print(f"Wrote {out_manifest}")


if __name__ == "__main__":
    main()
