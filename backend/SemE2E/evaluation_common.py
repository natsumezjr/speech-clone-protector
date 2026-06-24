from __future__ import annotations


def parse_model_list(value: str) -> list[str]:
    """Parse comma-separated model identifiers used by evaluation CLIs."""
    return [item.strip() for item in value.split(",") if item.strip()]
