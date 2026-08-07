from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


INFER_RESULT_MARKER = "VOICE_SHIELD_GPT_SOVITS_INFER_RESULT="


def _language(value: str, *, fallback: str = "en") -> str:
    normalized = (value or fallback).strip().lower().replace("_", "-")
    if normalized in {"zh", "zh-cn", "chinese"}:
        return "zh"
    if normalized in {"en", "en-us", "en-gb", "english"}:
        return "en"
    return fallback


def infer(args: argparse.Namespace) -> dict[str, object]:
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "GPT_SoVITS"))
    sys.path.insert(0, str(repo))

    import numpy as np
    import soundfile as sf
    import torch
    from TTS_infer_pack.TTS import TTS, TTS_Config

    config = TTS_Config(
        {
            "custom": {
                "device": args.device,
                "is_half": True,
                "version": "v2",
                "t2s_weights_path": str(args.gpt_checkpoint),
                "vits_weights_path": str(args.sovits_checkpoint),
                "cnhuhbert_base_path": str(args.cnhubert),
                "bert_base_path": str(args.bert),
            }
        }
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tts = TTS(config)
    load_seconds = time.perf_counter() - started
    inputs = {
        "text": args.text,
        "text_lang": _language(args.text_language),
        "ref_audio_path": str(args.reference),
        "prompt_text": args.prompt_text,
        "prompt_lang": _language(args.prompt_language),
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": "cut5",
        "batch_size": 1,
        "speed_factor": args.speed,
        "seed": 1234,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
    }
    started = time.perf_counter()
    generated = list(tts.run(inputs))
    inference_seconds = time.perf_counter() - started
    if not generated:
        raise RuntimeError("GPT-SoVITS returned no audio")
    sample_rate, audio = generated[-1]
    audio_array = np.asarray(audio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio_array, sample_rate)
    return {
        "ok": True,
        "loadSec": round(load_seconds, 4),
        "inferenceSec": round(inference_seconds, 4),
        "peakAllocatedMiB": round(torch.cuda.max_memory_allocated() / 1048576, 2),
        "sampleRate": int(sample_rate),
        "samples": int(audio_array.size),
        "durationSec": round(audio_array.size / float(sample_rate), 4),
        "output": str(args.output),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="VoiceShield GPT-SoVITS live-checkpoint inference worker")
    result.add_argument("--repo", type=Path, required=True)
    result.add_argument("--text", required=True)
    result.add_argument("--speed", type=float, default=1.0)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--cnhubert", type=Path, required=True)
    result.add_argument("--bert", type=Path, required=True)
    result.add_argument("--gpt-checkpoint", type=Path, required=True)
    result.add_argument("--sovits-checkpoint", type=Path, required=True)
    result.add_argument("--reference", type=Path, required=True)
    result.add_argument("--prompt-text", default="")
    result.add_argument("--prompt-language", default="en")
    result.add_argument("--text-language", default="en")
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        payload = infer(args)
        print(INFER_RESULT_MARKER + json.dumps(payload, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print(f"GPT-SoVITS worker failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
