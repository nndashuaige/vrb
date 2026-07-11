#!/usr/bin/env python3
"""E20 export driver.

The current E20 implementation emits progress per subaction through the reused
E18b runner. This wrapper preserves the planned command shape.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vrbreproduction.e20.pipeline import run_export


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E20 export for the first N subactions.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/e20"))
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_export(args.output_root, limit=args.limit)


if __name__ == "__main__":
    main()

