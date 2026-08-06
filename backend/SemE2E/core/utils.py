from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import torch


ROOT = Path(__file__).resolve().parents[1]


def parse_model_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_torch_device(requested: str | torch.device | None = "cuda") -> torch.device:
    import torch

    return requested if isinstance(requested, torch.device) else torch.device(requested or "cuda")


@contextmanager
def legacy_weight_norm_for_transformers_audio():
    """Load legacy weight_g/weight_v checkpoints on recent PyTorch releases."""
    import torch

    parametrizations = torch.nn.utils.parametrizations
    parametrized_weight_norm = parametrizations.weight_norm
    delattr(parametrizations, "weight_norm")
    try:
        yield
    finally:
        setattr(parametrizations, "weight_norm", parametrized_weight_norm)


def read_csv_rows(path: str | Path, required: Iterable[str] | None = None) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = set(required or [])
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        return list(reader)


def write_csv_rows(
    rows: list[dict[str, object]],
    path: str | Path,
    columns: Iterable[str] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(columns) if columns else sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json_csv_results(
    out_dir: str | Path,
    stem: str,
    rows: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    write_csv_rows(rows, csv_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
