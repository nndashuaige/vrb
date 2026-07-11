#!/usr/bin/env python3
"""CLI entrypoint for E20-B."""

from __future__ import annotations

import argparse
from pathlib import Path

from vrbreproduction.e20.pipeline import run_finalize
from vrbreproduction.e20.pipeline_b import DEFAULT_A_ROOT, run_ingest, run_pack, run_prepare, run_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E20-B hand-inpaint label-redraw stages.")
    parser.add_argument("--stage", choices=["prepare", "pack", "ingest", "finalize", "report"], required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/e20-b"))
    parser.add_argument("--source-root", type=Path, default=DEFAULT_A_ROOT, help="E20-A root to reuse export from")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--results-tar", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "prepare":
        run_prepare(args.output_root, source_root=args.source_root)
    elif args.stage == "pack":
        run_pack(args.output_root)
    elif args.stage == "ingest":
        run_ingest(args.output_root, results_dir=args.results_dir, results_tar=args.results_tar)
    elif args.stage == "finalize":
        run_finalize(args.output_root)
    elif args.stage == "report":
        run_report(args.output_root, e20a_root=args.source_root)


if __name__ == "__main__":
    main()
