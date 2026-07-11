#!/usr/bin/env python3
"""CLI entrypoint for E20."""

from __future__ import annotations

import argparse
from pathlib import Path

from vrbreproduction.e20.pipeline import run_export, run_finalize, run_ingest, run_pack, run_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E20 hand-inpaint label-redraw stages.")
    parser.add_argument("--stage", choices=["export", "pack", "ingest", "finalize", "report"], required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/e20"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--results-tar", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "export":
        run_export(args.output_root, limit=args.limit)
    elif args.stage == "pack":
        run_pack(args.output_root)
    elif args.stage == "ingest":
        run_ingest(args.output_root, results_dir=args.results_dir, results_tar=args.results_tar)
    elif args.stage == "finalize":
        run_finalize(args.output_root)
    elif args.stage == "report":
        run_report(args.output_root)


if __name__ == "__main__":
    main()

