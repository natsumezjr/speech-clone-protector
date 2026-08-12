from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

import yaml


RESULT_MARKER = "VOICE_SHIELD_GPT_SOVITS_LIVE_RESULT="


def _language(value: str) -> str:
    normalized = (value or "en").strip().lower().replace("_", "-")
    return "zh" if normalized in {"zh", "zh-cn", "chinese"} else "en"


def _reference_language(transcript: str, configured: str = "auto", *, fallback: str = "en") -> str:
    normalized = (configured or "auto").strip().lower().replace("_", "-")
    if normalized not in {"", "auto", "default"}:
        return _language(normalized)
    if any("\u4e00" <= character <= "\u9fff" for character in transcript or ""):
        return "zh"
    if any(character.isascii() and character.isalpha() for character in transcript or ""):
        return "en"
    return _language(fallback)


def _prepared_key(value: str) -> str:
    return Path(value.strip().replace("\\", "/")).name


def _prepared_keys(path: Path, *, semantic: bool) -> set[str]:
    keys: set[str] = set()
    for index, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines()):
        if not line.strip():
            continue
        columns = line.split("\t")
        if semantic and index == 0 and columns[0].strip() == "item_name":
            continue
        expected_columns = 2 if semantic else 4
        if len(columns) < expected_columns:
            continue
        key = _prepared_key(columns[0])
        if key:
            keys.add(key)
    return keys


def _normalize_prepared_keys(path: Path, *, semantic: bool) -> None:
    normalized_lines: list[str] = []
    for index, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines()):
        if not line.strip():
            continue
        columns = line.split("\t")
        if semantic and index == 0 and columns[0].strip() == "item_name":
            normalized_lines.append(line)
            continue
        if columns:
            columns[0] = _prepared_key(columns[0])
        normalized_lines.append("\t".join(columns))
    path.write_text("\n".join(normalized_lines) + ("\n" if normalized_lines else ""), encoding="utf-8")


def _validate_prepared_dataset(
    phoneme_path: Path,
    semantic_path: Path,
    *,
    condition: str,
    transcript_language: str,
) -> None:
    phoneme_keys = _prepared_keys(phoneme_path, semantic=False)
    semantic_keys = _prepared_keys(semantic_path, semantic=True)
    if not phoneme_keys:
        raise RuntimeError(
            "GPT-SoVITS preprocessing produced no phoneme entries for "
            f"the {condition} reference transcript (detected language={transcript_language}); "
            "verify that the transcript language matches the reference audio"
        )
    if not semantic_keys:
        raise RuntimeError(
            f"GPT-SoVITS preprocessing produced no semantic entries for the {condition} reference audio"
        )
    if phoneme_keys != semantic_keys:
        raise RuntimeError(
            "GPT-SoVITS preprocessing key mismatch for "
            f"{condition}: phoneme keys={sorted(phoneme_keys)[:3]}, "
            f"semantic keys={sorted(semantic_keys)[:3]}"
        )


def _run(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    timeout: int,
) -> float:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        completed.stdout + ("\n" if completed.stdout and completed.stderr else "") + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        details = completed.stderr[-4000:].strip() or completed.stdout[-3000:].strip()
        raise RuntimeError(f"command failed (exit={completed.returncode}): {details}")
    return elapsed


def _copy_stage_output(prepared_dir: Path, prefix: str, destination: str) -> Path:
    candidates = sorted(prepared_dir.glob(f"{prefix}-*.{'tsv' if prefix.startswith('6-') else 'txt'}"))
    if not candidates:
        raise RuntimeError(f"GPT-SoVITS preprocessing did not create {prefix} output")
    output = prepared_dir / destination
    source = candidates[-1]
    if prefix.startswith("6-"):
        content = source.read_text(encoding="utf-8")
        header = "item_name\tsemantic_audio"
        if not content.startswith(header):
            content = f"{header}\n{content.lstrip()}"
        output.write_text(content, encoding="utf-8")
    else:
        shutil.copyfile(source, output)
    return output


def _training_audio(audio_path: Path, condition_dir: Path, max_seconds: float) -> tuple[Path, float, float]:
    with wave.open(str(audio_path), "rb") as source:
        sample_rate = source.getframerate()
        frame_count = source.getnframes()
        duration = frame_count / sample_rate if sample_rate else 0.0
        if duration <= max_seconds:
            return audio_path, duration, duration
        target_frames = max(1, int(max_seconds * sample_rate))
        frames = source.readframes(target_frames)
        trimmed_path = condition_dir / f"{audio_path.stem}_training.wav"
        with wave.open(str(trimmed_path), "wb") as destination:
            destination.setparams(source.getparams())
            destination.setnframes(target_frames)
            destination.writeframes(frames)
    return trimmed_path, duration, max_seconds


def _reference_audio(
    audio_path: Path,
    condition_dir: Path,
    *,
    min_seconds: float,
    max_seconds: float,
) -> tuple[Path, float, float]:
    if min_seconds <= 0 or max_seconds < min_seconds:
        raise ValueError("Invalid GPT-SoVITS reference-audio duration limits")
    with wave.open(str(audio_path), "rb") as source:
        sample_rate = source.getframerate()
        frame_count = source.getnframes()
        duration = frame_count / sample_rate if sample_rate else 0.0
        if duration < min_seconds:
            raise ValueError(
                f"GPT-SoVITS reference audio must be at least {min_seconds:.2f} seconds; "
                f"got {duration:.2f} seconds"
            )
        if duration <= max_seconds:
            return audio_path, duration, duration
        target_frames = max(1, int(max_seconds * sample_rate))
        frames = source.readframes(target_frames)
        trimmed_path = condition_dir / f"{audio_path.stem}_reference.wav"
        with wave.open(str(trimmed_path), "wb") as destination:
            destination.setparams(source.getparams())
            destination.setnframes(target_frames)
            destination.writeframes(frames)
    return trimmed_path, duration, max_seconds


def _prepare(
    *,
    args: argparse.Namespace,
    condition: str,
    audio_path: Path,
    transcript: str,
    transcript_language: str,
    condition_dir: Path,
    environment: dict[str, str],
) -> tuple[dict[str, float], Path, Path]:
    prepared_dir = condition_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = condition_dir / "manifest.list"
    manifest_path.write_text(
        f"{audio_path.resolve()}|voice_shield_{condition}|{_language(transcript_language)}|{transcript.strip()}\n",
        encoding="utf-8",
    )
    common = environment.copy()
    common.update(
        {
            "inp_text": str(manifest_path),
            "inp_wav_dir": "",
            "exp_name": f"voice_shield_{condition}",
            "i_part": "0",
            "all_parts": "1",
            "opt_dir": str(prepared_dir),
            "is_half": "True",
            "version": "v2",
            "bert_pretrained_dir": str(args.bert),
            # GPT-SoVITS v2's chinese2/G2PW frontend reads ``bert_path``
            # while 1-get-text.py itself reads ``bert_pretrained_dir``.
            # Both must point at the same absolute local checkpoint.
            "bert_path": str(args.bert),
            "cnhubert_base_dir": str(args.cnhubert),
            "pretrained_s2G": str(args.pretrained_s2g),
            "s2config_path": str(args.repo / "GPT_SoVITS" / "configs" / "s2.json"),
        }
    )
    timings: dict[str, float] = {}
    timings["textSec"] = _run(
        [str(args.python), "GPT_SoVITS/prepare_datasets/1-get-text.py"],
        cwd=args.repo,
        environment=common,
        log_path=condition_dir / "text.log",
        timeout=args.timeout,
    )
    timings["hubertSec"] = _run(
        [str(args.python), "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"],
        cwd=args.repo,
        environment=common,
        log_path=condition_dir / "hubert.log",
        timeout=args.timeout,
    )
    timings["semanticSec"] = _run(
        [str(args.python), "GPT_SoVITS/prepare_datasets/3-get-semantic.py"],
        cwd=args.repo,
        environment=common,
        log_path=condition_dir / "semantic.log",
        timeout=args.timeout,
    )
    phoneme_path = _copy_stage_output(prepared_dir, "2-name2text", "2-name2text.txt")
    semantic_path = _copy_stage_output(prepared_dir, "6-name2semantic", "6-name2semantic.tsv")
    _normalize_prepared_keys(phoneme_path, semantic=False)
    _normalize_prepared_keys(semantic_path, semantic=True)
    _validate_prepared_dataset(
        phoneme_path,
        semantic_path,
        condition=condition,
        transcript_language=_language(transcript_language),
    )
    return timings, phoneme_path, semantic_path


def _train(
    *,
    args: argparse.Namespace,
    condition: str,
    condition_dir: Path,
    phoneme_path: Path,
    semantic_path: Path,
    environment: dict[str, str],
) -> tuple[dict[str, float], Path, Path]:
    checkpoint_dir = condition_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = f"voice_shield_{condition}"

    s1_config = yaml.safe_load((args.repo / "GPT_SoVITS" / "configs" / "s1longer-v2.yaml").read_text(encoding="utf-8"))
    s1_config["train"].update(
        {
            "epochs": 1,
            "batch_size": 1,
            "save_every_n_epoch": 1,
            "precision": "16-mixed",
            "if_save_every_weights": True,
            "if_save_latest": False,
            "half_weights_save_dir": str(checkpoint_dir),
            "exp_name": experiment_name,
        }
    )
    s1_config["pretrained_s1"] = str(args.pretrained_s1)
    s1_config["train_semantic_path"] = str(semantic_path)
    s1_config["train_phoneme_path"] = str(phoneme_path)
    s1_config["output_dir"] = str(condition_dir / "logs_s1_v2")
    s1_config_path = condition_dir / "s1_config.yaml"
    s1_config_path.write_text(yaml.safe_dump(s1_config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    s2_config = json.loads((args.repo / "GPT_SoVITS" / "configs" / "s2.json").read_text(encoding="utf-8"))
    s2_config["train"].update(
        {
            "epochs": 1,
            "batch_size": 1,
            "fp16_run": True,
            "pretrained_s2G": str(args.pretrained_s2g),
            "pretrained_s2D": str(args.pretrained_s2d),
            "if_save_latest": False,
            "if_save_every_weights": True,
            "save_every_epoch": 1,
            "gpu_numbers": args.gpu_numbers,
        }
    )
    s2_config["data"]["exp_dir"] = str(condition_dir / "prepared")
    s2_config["model"]["version"] = "v2"
    s2_config["s2_ckpt_dir"] = str(condition_dir / "prepared")
    s2_config["save_weight_dir"] = str(checkpoint_dir)
    s2_config["name"] = experiment_name
    s2_config["version"] = "v2"
    s2_config["train"]["lora_rank"] = 0
    s2_config_path = condition_dir / "s2_config.json"
    s2_config_path.write_text(json.dumps(s2_config, ensure_ascii=False, indent=2), encoding="utf-8")
    (condition_dir / "prepared" / "logs_s2_v2").mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    training_environment = environment.copy()
    training_environment["MASTER_PORT"] = str(20000 + ((os.getpid() + int(time.time() * 1000)) % 30000))
    timings["s1TrainSec"] = _run(
        [str(args.python), "GPT_SoVITS/s1_train.py", "-c", str(s1_config_path)],
        cwd=args.repo,
        environment=training_environment,
        log_path=condition_dir / "s1_train.log",
        timeout=args.timeout,
    )
    timings["s2TrainSec"] = _run(
        [str(args.python), "GPT_SoVITS/s2_train.py", "-c", str(s2_config_path)],
        cwd=args.repo,
        environment=training_environment,
        log_path=condition_dir / "s2_train.log",
        timeout=args.timeout,
    )

    gpt_candidates = sorted(checkpoint_dir.glob("*.ckpt"), key=lambda item: item.stat().st_mtime)
    sovits_candidates = sorted(checkpoint_dir.glob("*.pth"), key=lambda item: item.stat().st_mtime)
    if not gpt_candidates or not sovits_candidates:
        raise RuntimeError(f"GPT-SoVITS training did not create a checkpoint pair in {checkpoint_dir}")
    return timings, gpt_candidates[-1], sovits_candidates[-1]


def _infer(
    *,
    args: argparse.Namespace,
    condition: str,
    audio_path: Path,
    transcript: str,
    prompt_language: str,
    text_language: str,
    gpt_checkpoint: Path,
    sovits_checkpoint: Path,
    output_path: Path,
    condition_dir: Path,
    environment: dict[str, str],
) -> float:
    worker = Path(__file__).resolve().with_name("gpt_sovits_worker.py")
    return _run(
        [
            str(args.python),
            str(worker),
            "--repo",
            str(args.repo),
            "--gpt-checkpoint",
            str(gpt_checkpoint),
            "--sovits-checkpoint",
            str(sovits_checkpoint),
            "--reference",
            str(audio_path),
            "--prompt-text",
            transcript,
            "--prompt-language",
            _language(prompt_language),
            "--text",
            args.text,
            "--text-language",
            _language(text_language),
            "--speed",
            str(args.speed),
            "--output",
            str(output_path),
            "--device",
            args.device,
            "--cnhubert",
            str(args.cnhubert),
            "--bert",
            str(args.bert),
        ],
        cwd=args.repo,
        environment=environment,
        log_path=condition_dir / "inference.log",
        timeout=args.timeout,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.work_dir.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(args.repo / "GPT_SoVITS"), str(args.repo), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    if args.cuda_visible_devices:
        environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    started = time.perf_counter()
    results: dict[str, Any] = {}
    text_language = _language(getattr(args, "language", "en"))
    conditions = (
        (
            "original",
            args.original_audio,
            args.original_transcript,
            args.original_output,
            _reference_language(
                args.original_transcript,
                getattr(args, "original_prompt_language", "auto"),
                fallback=text_language,
            ),
        ),
        (
            "protected",
            args.protected_audio,
            args.protected_transcript,
            args.protected_output,
            _reference_language(
                args.protected_transcript,
                getattr(args, "protected_prompt_language", "auto"),
                fallback=text_language,
            ),
        ),
    )
    for condition, audio_path, transcript, output_path, prompt_language in conditions:
        condition_started = time.perf_counter()
        condition_dir = args.work_dir / condition
        condition_dir.mkdir(parents=True)
        training_audio, source_duration, training_duration = _training_audio(audio_path, condition_dir, args.max_training_seconds)
        reference_audio, _, reference_duration = _reference_audio(
            audio_path,
            condition_dir,
            min_seconds=args.min_reference_seconds,
            max_seconds=args.max_reference_seconds,
        )
        prepare_timings, phoneme_path, semantic_path = _prepare(
            args=args,
            condition=condition,
            audio_path=training_audio,
            transcript=transcript,
            transcript_language=prompt_language,
            condition_dir=condition_dir,
            environment=environment,
        )
        train_timings, gpt_checkpoint, sovits_checkpoint = _train(
            args=args,
            condition=condition,
            condition_dir=condition_dir,
            phoneme_path=phoneme_path,
            semantic_path=semantic_path,
            environment=environment,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        inference_sec = _infer(
            args=args,
            condition=condition,
            audio_path=reference_audio,
            transcript=transcript,
            prompt_language=prompt_language,
            text_language=text_language,
            gpt_checkpoint=gpt_checkpoint,
            sovits_checkpoint=sovits_checkpoint,
            output_path=output_path,
            condition_dir=condition_dir,
            environment=environment,
        )
        results[condition] = {
            **prepare_timings,
            **train_timings,
            "inferenceWallSec": round(inference_sec, 4),
            "totalWallSec": round(time.perf_counter() - condition_started, 4),
            "gptCheckpoint": str(gpt_checkpoint),
            "sovitsCheckpoint": str(sovits_checkpoint),
            "referencePath": str(reference_audio),
            "sourceReferencePath": str(audio_path),
            "trainingAudioPath": str(training_audio),
            "sourceDurationSec": round(source_duration, 4),
            "trainingDurationSec": round(training_duration, 4),
            "referenceDurationSec": round(reference_duration, 4),
            "promptLanguage": prompt_language,
            "textLanguage": text_language,
            "outputPath": str(output_path),
        }
    return {
        "mode": "live_fine_tune",
        "model": "GPT-SoVITS-v2",
        "workDir": str(args.work_dir),
        "pairWallSec": round(time.perf_counter() - started, 4),
        **results,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Train GPT-SoVITS on the current original/protected audio pair")
    value.add_argument("--repo", type=Path, required=True)
    value.add_argument("--python", type=Path, required=True)
    value.add_argument("--work-dir", type=Path, required=True)
    value.add_argument("--original-audio", type=Path, required=True)
    value.add_argument("--protected-audio", type=Path, required=True)
    value.add_argument("--original-transcript", required=True)
    value.add_argument("--protected-transcript", required=True)
    value.add_argument("--original-prompt-language", default="auto")
    value.add_argument("--protected-prompt-language", default="auto")
    value.add_argument("--text", required=True)
    value.add_argument("--language", default="en")
    value.add_argument("--speed", type=float, default=1.0)
    value.add_argument("--original-output", type=Path, required=True)
    value.add_argument("--protected-output", type=Path, required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--cuda-visible-devices", default="")
    value.add_argument("--gpu-numbers", default="5")
    value.add_argument("--cnhubert", type=Path, required=True)
    value.add_argument("--bert", type=Path, required=True)
    value.add_argument("--pretrained-s1", type=Path, required=True)
    value.add_argument("--pretrained-s2g", type=Path, required=True)
    value.add_argument("--pretrained-s2d", type=Path, required=True)
    value.add_argument("--timeout", type=int, default=900)
    value.add_argument("--max-training-seconds", type=float, default=54.0)
    value.add_argument("--min-reference-seconds", type=float, default=3.0)
    value.add_argument("--max-reference-seconds", type=float, default=10.0)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        payload = run(args)
        print(RESULT_MARKER + json.dumps(payload, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print(f"GPT-SoVITS live fine-tuning failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
