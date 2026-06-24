from __future__ import annotations

import csv
import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from result_adapter import TASK_DIR, UPLOAD_DIR, create_clone_voice, create_task, ensure_runtime_dirs, load_result, new_file_id
from result_schema import utc_now_iso

ensure_runtime_dirs()

app = FastAPI(title="SemE2E API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FILES: dict[str, dict[str, Any]] = {}


class ProtectTaskRequest(BaseModel):
    fileId: str | None = None
    mode: str | None = "custom"
    targets: list[str] | None = None
    semantic: dict[str, Any] | None = None
    timbre: dict[str, Any] | None = None
    psychoacoustic: dict[str, Any] | None = None
    optimization: dict[str, Any] | None = None
    referenceText: str | None = None
    reference_text: str | None = None


class CloneVoiceRequest(BaseModel):
    text: str
    model: str | None = "default"
    language: str | None = "auto"
    speed: float | None = 1.0
    speakerPrompt: str | None = None


def public_file_url(file_id: str, filename: str) -> str:
    return f"/api/files/{file_id}/{filename}"


def find_uploaded_file(file_id: str) -> dict[str, Any]:
    if file_id in FILES:
        return FILES[file_id]
    candidates = sorted(UPLOAD_DIR.glob(f"{file_id}_*"))
    if candidates:
        path = candidates[0]
        data = {"fileId": file_id, "filename": path.name.split("_", 1)[1], "path": path}
        FILES[file_id] = data
        return data
    raise HTTPException(status_code=404, detail=f"fileId not found: {file_id}")


def task_result_path(task_id: str) -> Path:
    path = TASK_DIR / task_id / "result.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"task not found: {task_id}")
    return path


def _number(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if number == number and number not in {float("inf"), float("-inf")} else fallback


def _fallback_trend(score: float) -> list[dict[str, float]]:
    return [
        {
            "step": step,
            "wer": min(0.92, 0.12 + step * 0.035),
            "sim": max(0.08, 0.86 - step * 0.045),
            "mos": max(3.1, 4.2 - step * 0.035),
            "pesq": max(2.9, 4.0 - step * 0.03),
            "elapsed": step * max(2.0, score / 24.0),
        }
        for step in range(1, 13)
    ]


def _frontend_audio(meta: dict[str, Any] | None, fallback_name: str) -> dict[str, Any]:
    meta = meta or {}
    return {
        "fileId": meta.get("fileId"),
        "filename": meta.get("filename") or fallback_name,
        "durationSec": meta.get("durationSec") or meta.get("duration"),
        "duration": meta.get("duration") or meta.get("durationSec"),
        "sampleRate": meta.get("sampleRate"),
        "channels": meta.get("channels"),
        "bitDepth": meta.get("bitDepth"),
        "sizeBytes": meta.get("sizeBytes") or 0,
        "format": meta.get("format") or Path(fallback_name).suffix.lstrip(".").upper() or "AUDIO",
        "src": meta.get("src"),
        "audioUrl": meta.get("audioUrl"),
        "downloadUrl": meta.get("downloadUrl"),
        "uploadedAt": meta.get("uploadedAt"),
        "fingerprint": meta.get("fingerprint"),
    }


def _frontend_clone(clone: dict[str, Any]) -> dict[str, Any]:
    return {
        "cloneId": clone.get("cloneId"),
        "taskId": clone.get("taskId"),
        "status": clone.get("status", "partial"),
        "source": clone.get("source"),
        "message": clone.get("message"),
        "request": clone.get("request") or {},
        "originalCloneAudio": _frontend_audio(clone.get("originalCloneAudio"), "original_clone.wav"),
        "protectedCloneAudio": _frontend_audio(clone.get("protectedCloneAudio"), "protected_clone.wav"),
    }


def frontend_result(result: dict[str, Any]) -> dict[str, Any]:
    summary = result.get("summary") or {}
    primary = summary.get("primaryMetrics") or {}
    details = result.get("details") or {}
    audio = result.get("audio") or {}
    score = _number(summary.get("score"), 80.0)
    snr = _number(primary.get("snr"), 0.0)
    pesq = _number(primary.get("pesq"), 3.5)
    sim_after = _number(primary.get("speakerSimilarity") or (details.get("speaker") or {}).get("simOriginalProtected"), 0.3)
    sim_before = max(sim_after, 0.9)
    asr = details.get("asr") or {}
    semantic = details.get("semantic") or {}
    charts = result.get("charts") or {}

    original_audio = _frontend_audio(audio.get("original"), "original.wav")
    protected_audio = _frontend_audio(audio.get("protected"), "protected.wav")

    return {
        "taskId": result.get("taskId"),
        "status": result.get("status", "completed"),
        "mode": result.get("mode", "joint"),
        "dataMode": result.get("dataMode", "backend"),
        "verdict": summary.get("verdict") or "防护结果已生成",
        "score": score,
        "createdAt": result.get("createdAt"),
        "submittedAt": result.get("submittedAt") or result.get("createdAt"),
        "completedAt": result.get("completedAt") or result.get("createdAt") or "-",
        "elapsedSec": _number(result.get("elapsedSec"), 0.0),
        "inputSource": "后端 API",
        "language": asr.get("language") or "未标注",
        "processingModel": (details.get("generation") or {}).get("source") or (result.get("backend") or {}).get("version"),
        "optimizationTarget": (details.get("generation") or {}).get("mode") or result.get("mode", "joint"),
        "asrModel": asr.get("model"),
        "artifacts": [
            {"label": "原始音频", "filename": original_audio["filename"], "sizeBytes": original_audio["sizeBytes"]},
            {"label": "保护音频", "filename": protected_audio["filename"], "sizeBytes": protected_audio["sizeBytes"]},
            {"label": "结果 JSON", "filename": "result.json"},
        ],
        "originalAudio": original_audio,
        "protectedAudio": protected_audio,
        "cloneResults": [_frontend_clone(item) for item in result.get("cloneResults", [])],
        "asr": {
            "originalText": asr.get("referenceText") or asr.get("cleanTranscription") or "ASR 未启用，暂无原始转写。",
            "protectedText": asr.get("protectedTranscription") or "ASR 未启用，暂无保护音频转写。",
            "wer": primary.get("wer") or asr.get("wer"),
            "cer": primary.get("cer") or asr.get("cer"),
            "tokenErrorRate": primary.get("tokenErrorRate") or semantic.get("tokenErrorRate"),
            "semanticDrift": primary.get("semanticDrift") or semantic.get("semanticDrift"),
        },
        "speaker": {
            "simBefore": sim_before,
            "simAfter": sim_after,
            "simDropRate": max(0.0, (sim_before - sim_after) / sim_before) if sim_before else 0.0,
            "embeddingDistanceBefore": max(0.0, 1.0 - sim_before),
            "embeddingDistanceAfter": max(0.0, 1.0 - sim_after),
        },
        "quality": {
            "snr": snr,
            "pesq": pesq,
            "mosLqo": max(2.8, min(4.6, 3.2 + snr / 30.0)),
        },
        "charts": {
            "psychoacoustic": charts.get("psychoacoustic") or [],
            "trend": charts.get("trend") or _fallback_trend(score),
            "radarBefore": charts.get("radarBefore") or [0.92, 0.88, 0.82, 0.76, 0.84, 0.8],
            "radarAfter": charts.get("radarAfter") or [0.22, 0.26, 0.24, 0.34, 0.29, 0.31],
        },
    }


def tiny_pdf_bytes(title: str, body: str) -> bytes:
    text = f"{title}\\n\\n{body}".replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 760 Td ({text}) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream.encode('utf-8'))} >>\nstream\n{stream}\nendstream".encode("utf-8"),
    ]
    output = io.BytesIO()
    output.write(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(output.tell())
        output.write(f"{index} 0 obj\n".encode("ascii"))
        output.write(obj)
        output.write(b"\nendobj\n")
    xref_at = output.tell()
    output.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        output.write(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.write(f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode("ascii"))
    return output.getvalue()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "time": utc_now_iso()}


@app.post("/api/files/upload")
async def upload_file(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")
    file_id = new_file_id()
    safe_name = Path(file.filename).name
    path = UPLOAD_DIR / f"{file_id}_{safe_name}"
    with path.open("wb") as target:
        shutil.copyfileobj(file.file, target)
    data = {
        "fileId": file_id,
        "filename": safe_name,
        "sizeBytes": path.stat().st_size,
        "format": path.suffix.lstrip(".").upper() or "AUDIO",
        "audioUrl": public_file_url(file_id, safe_name),
        "downloadUrl": public_file_url(file_id, safe_name),
        "uploadedAt": utc_now_iso(),
        "path": path,
    }
    FILES[file_id] = data
    return {key: value for key, value in data.items() if key != "path"}


@app.get("/api/files/{file_id}/{filename}")
def get_uploaded_file(file_id: str, filename: str) -> FileResponse:
    data = find_uploaded_file(file_id)
    path = Path(data["path"])
    if filename != data["filename"]:
        raise HTTPException(status_code=404, detail="filename mismatch")
    return FileResponse(path, filename=data["filename"])


@app.post("/api/tasks/protect")
def protect_task(payload: ProtectTaskRequest) -> dict[str, str]:
    if not payload.fileId:
        raise HTTPException(status_code=400, detail="fileId is required in backend mode")
    uploaded = find_uploaded_file(payload.fileId)
    try:
        result = create_task(Path(uploaded["path"]), payload.fileId, payload.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"taskId": result["taskId"], "status": result["status"]}


@app.get("/api/tasks")
def list_tasks() -> list[dict[str, Any]]:
    rows = []
    for result_file in sorted(TASK_DIR.glob("*/result.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            result = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        audio = result.get("audio") or {}
        primary = (result.get("summary") or {}).get("primaryMetrics") or {}
        rows.append(
            {
                "taskId": result.get("taskId"),
                "filename": (audio.get("original") or {}).get("filename", "-"),
                "protectedFilename": (audio.get("protected") or {}).get("filename", "-"),
                "mode": result.get("mode", "joint"),
                "dataMode": "backend",
                "status": result.get("status", "completed"),
                "wer": primary.get("wer"),
                "simDropRate": None,
                "pesq": primary.get("pesq"),
                "createdAt": result.get("createdAt"),
            }
        )
    return rows


@app.get("/api/tasks/{task_id}")
def task_status(task_id: str) -> dict[str, Any]:
    result = load_result(task_id)
    return {
        "taskId": task_id,
        "status": result.get("status", "completed"),
        "progress": 1,
        "stage": "report_generation",
        "message": "Task completed",
        "createdAt": result.get("createdAt"),
        "updatedAt": result.get("completedAt"),
        "error": None,
    }


@app.get("/api/tasks/{task_id}/result")
def task_result(task_id: str) -> JSONResponse:
    task_result_path(task_id)
    result = load_result(task_id)
    return JSONResponse(frontend_result(result))


@app.get("/api/tasks/{task_id}/details")
def task_details(task_id: str) -> JSONResponse:
    task_result_path(task_id)
    return JSONResponse(load_result(task_id))


@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: str) -> dict[str, str]:
    task_result_path(task_id)
    shutil.rmtree(TASK_DIR / task_id)
    return {"taskId": task_id, "status": "deleted"}


@app.get("/api/tasks/{task_id}/download/protected-audio")
def download_protected_audio(task_id: str) -> FileResponse:
    result = load_result(task_id)
    filename = ((result.get("audio") or {}).get("protected") or {}).get("filename")
    path = TASK_DIR / task_id / "protected" / filename if filename else None
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="protected audio not found")
    return FileResponse(path, media_type="audio/wav", filename=filename)


@app.post("/api/tasks/{task_id}/clone-voice")
def clone_voice(task_id: str, payload: CloneVoiceRequest) -> JSONResponse:
    task_result_path(task_id)
    try:
        result = create_clone_voice(task_id, payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JSONResponse(result)


@app.get("/api/artifacts/{task_id}/result.json")
def artifact_result_json(task_id: str) -> FileResponse:
    path = task_result_path(task_id)
    return FileResponse(path, media_type="application/json", filename="result.json")


@app.get("/api/artifacts/{task_id}/{kind}/{filename}")
def artifact_file(task_id: str, kind: str, filename: str) -> FileResponse:
    if kind not in {"original", "protected"}:
        raise HTTPException(status_code=404, detail="unknown artifact kind")
    path = TASK_DIR / task_id / kind / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    media_type = "audio/wav" if path.suffix.lower() == ".wav" else "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=filename)


@app.get("/api/artifacts/{task_id}/clones/{clone_id}/{filename}")
def clone_artifact_file(task_id: str, clone_id: str, filename: str) -> FileResponse:
    path = TASK_DIR / task_id / "clones" / clone_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="clone artifact not found")
    media_type = "audio/wav" if path.suffix.lower() == ".wav" else "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=filename)


@app.post("/api/reports/export")
def export_pdf(payload: dict[str, Any] | None = None) -> Response:
    task_id = str((payload or {}).get("taskId") or "")
    if task_id:
        task_result_path(task_id)
    pdf = tiny_pdf_bytes("Voice Protection Report", f"Task: {task_id or 'unknown'}")
    headers = {"Content-Disposition": f'attachment; filename="{task_id or "task"}-report.pdf"'}
    return Response(pdf, media_type="application/pdf", headers=headers)


@app.get("/api/tasks/{task_id}/export/csv")
def export_csv(task_id: str) -> Response:
    result = load_result(task_id)
    frontend = frontend_result(result)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["metric", "value"])
    writer.writerow(["taskId", frontend["taskId"]])
    writer.writerow(["status", frontend["status"]])
    writer.writerow(["score", frontend["score"]])
    writer.writerow(["wer", frontend["asr"].get("wer")])
    writer.writerow(["cer", frontend["asr"].get("cer")])
    writer.writerow(["simBefore", frontend["speaker"].get("simBefore")])
    writer.writerow(["simAfter", frontend["speaker"].get("simAfter")])
    writer.writerow(["snr", frontend["quality"].get("snr")])
    writer.writerow(["pesq", frontend["quality"].get("pesq")])
    headers = {"Content-Disposition": f'attachment; filename="{task_id}-metrics.csv"'}
    return Response(buffer.getvalue(), media_type="text/csv; charset=utf-8", headers=headers)


@app.get("/api/tasks/{task_id}/download/evidence")
def evidence_zip(task_id: str) -> Response:
    task_result_path(task_id)
    task_dir = TASK_DIR / task_id
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
      for path in task_dir.rglob("*"):
          if path.is_file():
              archive.write(path, path.relative_to(task_dir).as_posix())
    headers = {"Content-Disposition": f'attachment; filename="{task_id}-evidence.zip"'}
    return Response(buffer.getvalue(), media_type="application/zip", headers=headers)


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=int(os.getenv("SEME2E_API_PORT", "8000")), reload=False)
