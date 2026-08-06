"""Compatibility entrypoint for the missing-only model preparation workflow."""

from __future__ import annotations

import argparse

from scripts.prepare import download_models


def main() -> None:
    download_models(argparse.Namespace())


if __name__ == "__main__":
    main()
