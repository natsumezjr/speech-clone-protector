from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two CosyVoice2 zero-shot clones in one isolated process.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cosyvoice-repo", required=True)
    parser.add_argument("--original-reference", required=True)
    parser.add_argument("--protected-reference", required=True)
    parser.add_argument("--original-output", required=True)
    parser.add_argument("--protected-output", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--original-prompt-text", required=True)
    parser.add_argument("--protected-prompt-text", required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _save_generation(model: object, reference: str, output: Path, text: str, prompt_text: str, speed: float) -> dict[str, float]:
    import torch
    import torchaudio

    started = time.perf_counter()
    chunks = [item["tts_speech"].detach().cpu() for item in model.inference_zero_shot(
        text,
        prompt_text,
        reference,
        stream=False,
        speed=speed,
    )]
    if not chunks:
        raise RuntimeError("CosyVoice2 returned no audio chunks")
    speech = torch.cat(chunks, dim=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output), speech, int(model.sample_rate))
    return {
        "inferenceSec": round(time.perf_counter() - started, 4),
        "durationSec": round(speech.shape[1] / float(model.sample_rate), 4),
    }


def main() -> int:
    args = _parse_args()
    repo = Path(args.cosyvoice_repo).resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "third_party" / "Matcha-TTS"))
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    import torch
    from cosyvoice.cli.cosyvoice import CosyVoice2

    requested = str(args.device).strip().lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {args.device}, but CUDA is unavailable")
    if requested.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    load_started = time.perf_counter()
    model = CosyVoice2(args.model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=requested.startswith("cuda"))
    load_sec = time.perf_counter() - load_started
    original = _save_generation(
        model,
        args.original_reference,
        Path(args.original_output),
        args.text,
        args.original_prompt_text,
        args.speed,
    )
    protected = _save_generation(
        model,
        args.protected_reference,
        Path(args.protected_output),
        args.text,
        args.protected_prompt_text,
        args.speed,
    )
    peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024) if requested.startswith("cuda") else 0.0
    payload = {
        "ok": True,
        "backend": "CosyVoice2",
        "modelDir": str(Path(args.model_dir).resolve()),
        "device": args.device,
        "loadSec": round(load_sec, 4),
        "peakAllocatedMiB": round(peak_mib, 2),
        "original": original,
        "protected": protected,
    }
    print("VOICE_SHIELD_COSYVOICE_RESULT=" + json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
