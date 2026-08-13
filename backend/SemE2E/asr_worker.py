"""Isolated two-audio ASR worker using one UTF-8 JSON document per stream."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _text_field(request: dict[str, Any], name: str, *, default: str | None = None) -> str:
    value = request.get(name, default)
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} is required")
    return text


def _audio_path(request: dict[str, Any], name: str) -> Path:
    path = Path(_text_field(request, name)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist or is not a file: {path}")
    return path


def _configure_utf8_stream(stream: Any) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


@contextmanager
def _worker_output_to_stderr() -> Iterator[None]:
    """Keep dependency diagnostics away from the single JSON stdout response."""

    stdout_fd: int | None = None
    saved_stdout_fd: int | None = None
    try:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
        sys.stdout.flush()
        saved_stdout_fd = os.dup(stdout_fd)
        os.dup2(stderr_fd, stdout_fd)
    except (AttributeError, OSError):
        if saved_stdout_fd is not None:
            if stdout_fd is not None:
                try:
                    os.dup2(saved_stdout_fd, stdout_fd)
                except OSError:
                    pass
            os.close(saved_stdout_fd)
        with redirect_stdout(sys.stderr):
            yield
        return

    try:
        with redirect_stdout(sys.stderr):
            yield
    finally:
        try:
            sys.stderr.flush()
        finally:
            saved_fd = saved_stdout_fd
            target_fd = stdout_fd
            if saved_fd is None or target_fd is None:
                raise RuntimeError("stdout redirection state was not initialized")
            os.dup2(saved_fd, target_fd)
            os.close(saved_fd)


def execute(request: dict[str, Any]) -> dict[str, Any]:
    model = _text_field(request, "model")
    requested_language = _text_field(request, "language", default="en")
    requested_device = _text_field(
        request,
        "device",
        default=os.getenv("SEME2E_API_DEVICE", "cpu"),
    )
    original_path = _audio_path(request, "originalPath")
    protected_path = _audio_path(request, "protectedPath")

    from asr_backends import ASRTranscriber, openai_whisper_session

    started = time.perf_counter()
    with openai_whisper_session(model):
        load_started = time.perf_counter()
        transcriber = ASRTranscriber(model, requested_device, requested_language)
        load_sec = time.perf_counter() - load_started

        original_started = time.perf_counter()
        original_text = transcriber.transcribe(original_path)
        original_sec = time.perf_counter() - original_started

        protected_started = time.perf_counter()
        protected_text = transcriber.transcribe(protected_path)
        protected_sec = time.perf_counter() - protected_started

    return {
        "ok": True,
        "model": str(getattr(transcriber, "model_name", model)),
        "language": str(
            getattr(transcriber, "detected_language", None)
            or getattr(transcriber, "language", None)
            or requested_language
        ),
        "requestedLanguage": requested_language,
        "device": str(getattr(transcriber, "device", requested_device)),
        "requestedDevice": requested_device,
        "originalPath": str(original_path),
        "protectedPath": str(protected_path),
        "originalText": str(original_text),
        "protectedText": str(protected_text),
        "loadSec": round(load_sec, 4),
        "originalSec": round(original_sec, 4),
        "protectedSec": round(protected_sec, 4),
        "elapsedSec": round(time.perf_counter() - started, 4),
    }


def _request_metadata(request: Any) -> tuple[str | None, str | None, str | None]:
    if not isinstance(request, dict):
        return None, None, None
    model = str(request.get("model") or "").strip() or None
    language = str(request.get("language") or "").strip() or None
    device = str(request.get("device") or "").strip() or None
    return model, language, device


def main() -> int:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        _configure_utf8_stream(stream)

    request: Any = None
    exit_code = 0
    try:
        raw_request = sys.stdin.read()
        if not raw_request.strip():
            raise ValueError("stdin must contain one JSON request object")
        request = json.loads(raw_request)
        if not isinstance(request, dict):
            raise TypeError("ASR worker request must be a JSON object")
        with _worker_output_to_stderr():
            response = execute(request)
    except Exception as exc:
        model, language, device = _request_metadata(request)
        response = {
            "ok": False,
            "model": model,
            "language": language,
            "device": device,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        exit_code = 1

    sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
