import argparse
import subprocess
import sys
from pathlib import Path

from io_utils import read_csv_rows, write_csv_rows

ROOT = Path(__file__).resolve().parent


def read_rows(path, limit=None):
    rows = read_csv_rows(path)
    return rows[:limit] if limit else rows


def write_rows(rows, path):
    columns = [
        "id",
        "condition",
        "clean_audio",
        "audio",
        "reference_text",
        "duration_s",
        "source_split",
        "source_path",
    ]
    write_csv_rows(rows, path, columns)


def protect(row, args):
    input_wav = Path(row["audio"]).resolve()
    out_dir = args.output_dir / row["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    output_wav = out_dir / f"{row['id']}_semantic_e{args.epochs}.wav"
    log_path = out_dir / f"{row['id']}_semantic_e{args.epochs}.log"

    cmd = [
        sys.executable,
        "-B",
        str(ROOT / "protect_semantic.py"),
        "--input_wav",
        str(input_wav),
        "--output_wav",
        str(output_wav),
        "--epochs",
        str(args.epochs),
        "--device",
        args.device,
    ]
    cmd.extend(args.protect_args)
    print(" ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, check=True)
    return output_wav


def parse_args():
    parser = argparse.ArgumentParser(description="Run semantic protection for a LibriTTS manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs" / "libritts_subset")
    parser.add_argument("--output_manifest", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("protect_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.protect_args and args.protect_args[0] == "--":
        args.protect_args = args.protect_args[1:]
    args.output_dir = args.output_dir.resolve()
    out_manifest = args.output_manifest or (args.output_dir / f"protected_e{args.epochs}.csv")
    rows = read_rows(args.manifest.resolve(), args.limit)

    result_rows = []
    for row in rows:
        clean_audio = str(Path(row["audio"]).resolve())
        reference_text = row["text_normalized"]
        result_rows.append(
            {
                "id": row["id"],
                "condition": "clean",
                "clean_audio": clean_audio,
                "audio": clean_audio,
                "reference_text": reference_text,
                "duration_s": row.get("duration_s", ""),
                "source_split": row.get("split", ""),
                "source_path": row.get("source_path", ""),
            }
        )
        protected = protect(row, args)
        result_rows.append(
            {
                "id": row["id"],
                "condition": f"semantic_e{args.epochs}",
                "clean_audio": clean_audio,
                "audio": str(protected.resolve()),
                "reference_text": reference_text,
                "duration_s": row.get("duration_s", ""),
                "source_split": row.get("split", ""),
                "source_path": row.get("source_path", ""),
            }
        )

    write_rows(result_rows, out_manifest)
    print(f"Wrote {out_manifest}")


if __name__ == "__main__":
    main()
