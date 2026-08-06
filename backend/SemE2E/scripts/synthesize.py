import argparse
import hashlib
import os
import random
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import numpy as np
import soundfile as sf

from core.utils import read_csv_rows, write_csv_rows


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path, limit=None, shard_index=0, num_shards=1):
    rows = read_csv_rows(path)
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    sample_ids = list(dict.fromkeys(row["id"] for row in rows))
    assigned_ids = set(sample_ids[shard_index::num_shards])
    rows = [row for row in rows if row["id"] in assigned_ids]
    return rows[:limit] if limit is not None else rows


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
        "generation_seed",
    ]
    write_csv_rows(rows, path, columns)


def paired_generation_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2**31)


def seed_inference(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patch_torch_load_for_coqui_xtts():
    import torch

    original_load = torch.load

    def load_with_legacy_default(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = load_with_legacy_default


def patch_hf_download_endpoint():
    import requests

    endpoint = os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    os.environ.setdefault("XDG_DATA_HOME", str((ROOT / "checkpoints" / "coqui_tts").resolve()))
    mirror = urlparse(endpoint)

    def mirror_url(url):
        if isinstance(url, str):
            original = urlparse(url)
            if original.scheme in {"http", "https"} and original.netloc.lower() == "huggingface.co":
                return urlunparse(
                    (mirror.scheme, mirror.netloc, original.path, original.params, original.query, original.fragment)
                )
        return url

    original_request = requests.sessions.Session.request
    original_send = requests.sessions.Session.send
    original_adapter_send = requests.adapters.HTTPAdapter.send
    original_get_redirect_target = requests.sessions.SessionRedirectMixin.get_redirect_target

    def request_with_hf_mirror(self, method, url, *args, **kwargs):
        url = mirror_url(url)
        return original_request(self, method, url, *args, **kwargs)

    def send_with_hf_mirror(self, request, **kwargs):
        request.url = mirror_url(request.url)
        return original_send(self, request, **kwargs)

    def adapter_send_with_hf_mirror(self, request, **kwargs):
        request.url = mirror_url(request.url)
        return original_adapter_send(self, request, **kwargs)

    def get_redirect_target_with_hf_mirror(self, response):
        return mirror_url(original_get_redirect_target(self, response))

    requests.sessions.Session.request = request_with_hf_mirror
    requests.sessions.Session.send = send_with_hf_mirror
    requests.adapters.HTTPAdapter.send = adapter_send_with_hf_mirror
    requests.sessions.SessionRedirectMixin.get_redirect_target = get_redirect_target_with_hf_mirror


def parse_args():
    parser = argparse.ArgumentParser(description="Generate TTS samples with XTTS-v2 speaker references.")
    parser.add_argument("--references", type=Path, required=True, help="Protection manifest CSV.")
    parser.add_argument("--target_text", required=True)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs" / "tts_xtts")
    parser.add_argument("--output_manifest", type=Path, default=None)
    parser.add_argument("--model_name", default="tts_models/multilingual/multi-dataset/xtts_v2")
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=20260804,
        help="Base seed; rows with the same sample id receive the same paired generation seed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.output_manifest:
        out_manifest = args.output_manifest
    elif args.num_shards > 1:
        shard_suffix = f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
        out_manifest = args.output_dir / f"tts_manifest.{shard_suffix}.csv"
    else:
        out_manifest = args.output_dir / "tts_manifest.csv"

    patch_torch_load_for_coqui_xtts()
    patch_hf_download_endpoint()
    from TTS.api import TTS

    tts = TTS(args.model_name)
    if args.device.startswith("cuda"):
        tts = tts.to("cuda")

    rows = []
    for item in read_rows(
        args.references.resolve(),
        args.limit,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    ):
        ref_audio = Path(item["audio"]).resolve()
        out_dir = args.output_dir / item["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        synth_audio = out_dir / f"{item['id']}_{item['condition']}_xtts.wav"
        generation_seed = paired_generation_seed(args.seed, item["id"])
        seed_inference(generation_seed)
        print(
            f"XTTS {item['id']} {item['condition']} seed={generation_seed} "
            f"-> {synth_audio}"
        )
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
                "generation_seed": generation_seed,
            }
        )

    write_rows(rows, out_manifest)
    print(f"Wrote {out_manifest}")


if __name__ == "__main__":
    main()
