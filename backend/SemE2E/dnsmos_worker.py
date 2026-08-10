"""Isolated DNSMOS P.835 pair evaluator using one UTF-8 JSON response."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def execute(request: dict[str, Any]) -> dict[str, Any]:
    from dnsmos_quality import evaluate_dnsmos_pair

    original_path = Path(str(request.get("originalPath") or "")).expanduser().resolve()
    protected_path = Path(str(request.get("protectedPath") or "")).expanduser().resolve()
    if not original_path.is_file():
        raise FileNotFoundError(f"originalPath does not exist: {original_path}")
    if not protected_path.is_file():
        raise FileNotFoundError(f"protectedPath does not exist: {protected_path}")
    result = evaluate_dnsmos_pair(
        original_path,
        protected_path,
        request.get("modelPath"),
    )
    if result.get("status") != "available":
        return {"ok": False, **result}
    return {"ok": True, **result}


def main() -> int:
    request: Any = None
    exit_code = 0
    try:
        raw = sys.stdin.read()
        request = json.loads(raw)
        if not isinstance(request, dict):
            raise TypeError("DNSMOS worker request must be a JSON object")
        response = execute(request)
        if response.get("ok") is not True:
            exit_code = 1
    except Exception as exc:
        response = {
            "ok": False,
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
