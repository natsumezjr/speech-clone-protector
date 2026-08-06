from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from core.utils import read_csv_rows, write_csv_rows


ROOT = Path(__file__).resolve().parents[1]
QUALITY_PRESETS = {
    "lq25_large_balanced": (
        "--hubert_path", str(ROOT / "checkpoints" / "hf" / "facebook" / "hubert-large-ll60k"),
        "--whisper_path", str(ROOT / "checkpoints" / "hf" / "openai" / "whisper-large-v3"),
        "--epsilon", "0.01568627450980392",
        "--init_noise", "zero",
        "--l2_reduction", "rms",
        "--step_size", "0.00012",
        "--weight_feature", "150",
        "--weight_semantic", "300",
        "--weight_psy", "0.001",
        "--weight_stft", "150",
        "--weight_snr", "20",
        "--target_snr_db", "25",
        "--selection_snr_db", "25",
    ),
    "q18_perceptual": (
        "--epsilon", "0.01568627450980392",
        "--init_noise", "zero",
        "--l2_reduction", "rms",
        "--weight_semantic", "100",
        "--weight_l2", "1",
        "--step_size", "0.00035",
        "--weight_feature", "400",
        "--weight_psy", "0.0005",
        "--weight_stft", "50",
        "--weight_snr", "10",
        "--target_snr_db", "18",
        "--selection_snr_db", "18",
    ),
    "q24_perceptual": (
        "--epsilon", "0.01568627450980392",
        "--init_noise", "zero",
        "--l2_reduction", "rms",
        "--weight_semantic", "100",
        "--weight_l2", "1",
        "--step_size", "0.00025",
        "--weight_feature", "300",
        "--weight_psy", "0.001",
        "--weight_stft", "100",
        "--weight_snr", "10",
        "--target_snr_db", "24",
        "--selection_snr_db", "24",
    ),
}


def add_guard_arguments(parser: argparse.ArgumentParser, require_input: bool = True) -> None:
    parser.add_argument("--input_wav", required=require_input)
    parser.add_argument("--output_wav", default=None)
    parser.add_argument("--epsilon", type=float, default=4 / 255)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--tokenizer_path",
        default=str(ROOT / "checkpoints" / "CosyVoice" / "speech_tokenizer_v1.onnx"),
    )
    parser.add_argument("--hubert_path", default="facebook/hubert-large-ll60k")
    parser.add_argument("--whisper_path", default="openai/whisper-large-v3")
    parser.add_argument("--no_vits", action="store_true")
    parser.add_argument("--no_gsv", action="store_true")
    parser.add_argument("--no_mfcc_timbre", action="store_true")
    parser.add_argument("--no_wavlm", action="store_true")
    parser.add_argument("--no_cosyvoice", action="store_true")
    parser.add_argument("--weight_feature", type=float, default=150.0)
    parser.add_argument("--weight_semantic", type=float, default=300.0)
    parser.add_argument("--weight_psy", type=float, default=0.001)
    parser.add_argument("--weight_l2", type=float, default=0.1)
    parser.add_argument("--l2_reduction", choices=["rms", "norm"], default="rms")
    parser.add_argument("--init_noise", choices=["zero", "random"], default="zero")
    parser.add_argument("--step_size", type=float, default=0.00012)
    parser.add_argument("--weight_stft", type=float, default=150.0)
    parser.add_argument("--weight_snr", type=float, default=20.0)
    parser.add_argument("--target_snr_db", type=float, default=25.0)
    parser.add_argument("--selection_snr_db", type=float, default=25.0)
    parser.add_argument("--verbose", action="store_true")


def parse_guard_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    add_guard_arguments(parser)
    return parser.parse_args(argv)


def create_guard(args: argparse.Namespace):
    import torch

    from core.guard import VoiceShield

    device = torch.device(args.device)
    return VoiceShield(
        epsilon=args.epsilon,
        max_items=args.epochs,
        device=device,
        tokenizer_path=args.tokenizer_path,
        hubert_path=args.hubert_path,
        whisper_path=args.whisper_path,
        use_vits=not args.no_vits,
        use_gsv=not args.no_gsv,
        use_mfcc_timbre=not args.no_mfcc_timbre,
        use_wavlm=not args.no_wavlm,
        use_cosyvoice=not args.no_cosyvoice,
        weight_feature=args.weight_feature,
        weight_semantic=args.weight_semantic,
        weight_psy=args.weight_psy,
        weight_l2=args.weight_l2,
        l2_reduction=args.l2_reduction,
        init_noise=args.init_noise,
        step_size=args.step_size,
        weight_stft=args.weight_stft,
        weight_snr=args.weight_snr,
        target_snr_db=args.target_snr_db,
        selection_snr_db=args.selection_snr_db,
    )


def run_single(args: argparse.Namespace) -> None:
    result = create_guard(args).protect(
        input_wav=Path(args.input_wav).resolve(),
        output_wav=Path(args.output_wav).resolve() if args.output_wav else None,
        verbose=args.verbose,
    )
    print("Protected audio:", result["output_wav"])
    print(f"SNR: {result['snr']:.3f}")
    print("Selected step:", result["selected_step"])
    print("Loss:", result["loss_items"])


def _batch_rows(path: Path, limit: int | None, shard_index: int, num_shards: int):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards) and num_shards must be positive")
    rows = read_csv_rows(path)[shard_index::num_shards]
    return rows[:limit] if limit is not None else rows


def _protect_batch_item(row, args, guard, guard_args) -> Path:
    input_wav = Path(row["audio"]).resolve()
    output_dir = args.output_dir / row["id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_wav = output_dir / f"{row['id']}_{args.condition}.wav"
    if guard is not None:
        print(f"Protecting {input_wav} -> {output_wav}")
        guard.protect(input_wav, output_wav, verbose=guard_args.verbose)
        return output_wav

    command = [
        sys.executable,
        "-B",
        "-m",
        "scripts.protect",
        "single",
        "--input_wav",
        str(input_wav),
        "--output_wav",
        str(output_wav),
        "--epochs",
        str(args.epochs),
        "--device",
        args.device,
        *args.protect_args,
    ]
    print(" ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)
    return output_wav


def run_batch(args: argparse.Namespace) -> None:
    if args.protect_args and args.protect_args[0] == "--":
        args.protect_args = args.protect_args[1:]
    args.protect_args = [*QUALITY_PRESETS[args.quality_preset], *args.protect_args]
    args.epochs = args.epochs if args.epochs is not None else (100 if args.quality_preset == "lq25_large_balanced" else 40)
    args.condition = args.condition or args.quality_preset or f"semantic_e{args.epochs}"
    args.condition = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.condition).strip("._-")
    if not args.condition:
        raise ValueError("condition must contain at least one filename-safe character")
    args.output_dir = args.output_dir.resolve()
    if args.output_manifest:
        output_manifest = args.output_manifest
    elif args.num_shards > 1:
        suffix = f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
        output_manifest = args.output_dir / f"protected_{args.condition}.{suffix}.csv"
    else:
        output_manifest = args.output_dir / f"protected_{args.condition}.csv"

    rows = _batch_rows(args.manifest.resolve(), args.limit, args.shard_index, args.num_shards)
    guard_args = parse_guard_args(
        [
            "--input_wav",
            str(Path(rows[0]["audio"]).resolve()),
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
            *args.protect_args,
        ]
    )
    guard = create_guard(guard_args) if args.reuse_model else None
    result_rows = []
    for row in rows:
        clean_audio = str(Path(row["audio"]).resolve())
        shared = {
            "id": row["id"],
            "clean_audio": clean_audio,
            "reference_text": row["text_normalized"],
            "duration_s": row.get("duration_s", ""),
            "source_split": row.get("split", ""),
            "source_path": row.get("source_path", ""),
        }
        result_rows.append({**shared, "condition": "clean", "audio": clean_audio})
        protected = _protect_batch_item(row, args, guard, guard_args)
        result_rows.append({**shared, "condition": args.condition, "audio": str(protected.resolve())})

    write_csv_rows(
        result_rows,
        output_manifest,
        ["id", "condition", "clean_audio", "audio", "reference_text", "duration_s", "source_split", "source_path"],
    )
    print(f"Wrote {output_manifest}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic E2E-VGuard protection tools.")
    commands = parser.add_subparsers(dest="command", required=True)

    single = commands.add_parser("single", help="Protect one WAV file.")
    add_guard_arguments(single)
    single.set_defaults(func=run_single)

    batch = commands.add_parser("batch", help="Protect all rows in a dataset manifest.")
    batch.add_argument("--manifest", type=Path, required=True)
    batch.add_argument("--output_dir", type=Path, default=ROOT / "outputs" / "libritts_subset")
    batch.add_argument("--output_manifest", type=Path, default=None)
    batch.add_argument("--quality_preset", choices=sorted(QUALITY_PRESETS), default="lq25_large_balanced")
    batch.add_argument("--epochs", type=int, default=None)
    batch.add_argument("--condition", default=None)
    batch.add_argument("--limit", type=int, default=None)
    batch.add_argument("--num_shards", type=int, default=1)
    batch.add_argument("--shard_index", type=int, default=0)
    batch.add_argument("--device", default="cuda")
    batch.add_argument("--no_reuse_model", action="store_false", dest="reuse_model")
    batch.set_defaults(reuse_model=True)
    batch.add_argument("protect_args", nargs=argparse.REMAINDER)
    batch.set_defaults(func=run_batch)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
