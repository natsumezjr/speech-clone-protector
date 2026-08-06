"""Compatibility CLI forwarding legacy single-file calls to ``scripts.protect``."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.protect import add_guard_arguments, create_guard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic E2E-VGuard compatibility entrypoint.")
    add_guard_arguments(parser)
    parser.add_argument("--timbre_mode", choices=["untargeted"], default="untargeted")
    parser.add_argument("--no_style", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--weight_identity", type=float, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.weight_identity is not None:
        args.weight_feature = args.weight_identity
    result = create_guard(args).protect(
        input_wav=Path(args.input_wav).resolve(),
        output_wav=Path(args.output_wav).resolve() if args.output_wav else None,
        verbose=args.verbose,
    )
    print("Protected audio:", result["output_wav"])
    print(f"SNR: {result['snr']:.3f}")
    print("Selected step:", result["selected_step"])
    print("Loss:", result["loss_items"])


if __name__ == "__main__":
    main()
