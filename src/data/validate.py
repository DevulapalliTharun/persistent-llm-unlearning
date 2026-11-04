"""Verify checksums and validation status for a generated dataset bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .synthetic import verify_checksums


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.bundle / "manifest.json"
    validation_path = args.bundle / "reports/validation.json"
    errors = verify_checksums(args.bundle)
    if not manifest_path.exists():
        errors.append("Missing manifest.json")
    if not validation_path.exists():
        errors.append("Missing reports/validation.json")
    else:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if not validation.get("passed"):
            errors.append("Saved validation report did not pass")
    if errors:
        print(json.dumps({"passed": False, "errors": errors}, indent=2))
        return 1
    print(json.dumps({"passed": True, "errors": []}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
