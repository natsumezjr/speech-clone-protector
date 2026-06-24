from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


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
