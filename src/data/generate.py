"""CLI for deterministic synthetic dataset generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .synthetic import generate_bundle, git_commit, load_config, tiny_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Generate only 2 F, 4 R and 3 N_rel units for implementation testing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.tiny:
        config = tiny_config(config)
    manifest = generate_bundle(config, args.output_dir, source_git_commit=git_commit())
    print(json.dumps(manifest["validation"], indent=2, sort_keys=True))
    print(f"Dataset bundle written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
