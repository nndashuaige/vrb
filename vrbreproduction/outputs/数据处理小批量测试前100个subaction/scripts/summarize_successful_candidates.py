#!/usr/bin/env python3
"""
Summarize successful VRB data-processing candidates.

Reads candidate_diagnostics.csv from a diagnostic experiment output directory
and writes compact tables that answer:
- which experiment passed how many frames;
- which subactions passed;
- exactly which frame/hand/reference pairs passed;
- where the per-sample visual outputs live.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "数据处理小批量测试前100个subaction"
)

SUCCESS_COLUMNS = [
    "experiment_id",
    "subaction_index",
    "narration_id",
    "narration",
    "frame_0_based",
    "frame_1_based",
    "hand",
    "ref_idx",
    "ref_frame_1_based",
    "contact_points",
    "trajectory_points",
    "sample_score",
    "sample_dir",
    "reference_frame",
    "label_heatmap_png",
    "label_heatmap_overlay",
    "vrb_style_affordance",
    "problem3_full_overlay",
    "problem3_crop",
]


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "None\n"
    view = df.copy().where(pd.notna(df), "")
    lines = [
        "| " + " | ".join(map(str, view.columns)) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in view.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def make_relative_to(path_value: object, base: Path) -> str:
    if pd.isna(path_value):
        return ""
    path = Path(str(path_value))
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def summarize(output_root: Path, out_dir: Path) -> dict[str, Path]:
    candidate_path = output_root / "candidate_diagnostics.csv"
    overview_path = output_root / "experiment_overview.csv"
    if not candidate_path.exists():
        raise FileNotFoundError(f"missing candidate diagnostics: {candidate_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(candidate_path)
    overview = pd.read_csv(overview_path) if overview_path.exists() else pd.DataFrame()
    successful = candidates[candidates["status"] == "keep"].copy()
    successful = successful.sort_values(
        ["experiment_id", "subaction_index", "frame_0_based", "hand"],
        kind="stable",
    )

    keep_cols = [col for col in SUCCESS_COLUMNS if col in successful.columns]
    success_table = successful[keep_cols].copy()
    for col in [
        "sample_dir",
        "reference_frame",
        "label_heatmap_png",
        "label_heatmap_overlay",
        "vrb_style_affordance",
        "problem3_full_overlay",
        "problem3_crop",
    ]:
        if col in success_table.columns:
            success_table[col] = success_table[col].map(lambda value: make_relative_to(value, output_root))

    counts = (
        successful.groupby("experiment_id")
        .agg(
            successful_samples=("frame_0_based", "count"),
            successful_subactions=("subaction_index", "nunique"),
        )
        .reset_index()
    )
    if not overview.empty:
        counts = overview[
            [
                "experiment_id",
                "subactions_total",
                "candidates_total",
                "cell2_pass",
                "ref_found",
                "homography_available",
                "geometry_pass",
                "heatmap_pass",
            ]
        ].merge(counts, on="experiment_id", how="left")
        counts[["successful_samples", "successful_subactions"]] = counts[
            ["successful_samples", "successful_subactions"]
        ].fillna(0).astype(int)

    per_subaction = (
        successful.groupby(["experiment_id", "subaction_index", "narration_id", "narration"])
        .agg(
            successful_samples=("frame_0_based", "count"),
            frames_0_based=("frame_0_based", lambda s: ", ".join(str(int(v)) for v in s)),
            frames_1_based=("frame_1_based", lambda s: ", ".join(str(int(v)) for v in s)),
            hands=("hand", lambda s: ", ".join(map(str, s))),
            refs_0_based=("ref_idx", lambda s: ", ".join(str(int(v)) for v in s if pd.notna(v))),
        )
        .reset_index()
        .sort_values(["experiment_id", "subaction_index"], kind="stable")
    )

    counts_path = out_dir / "successful_counts_by_experiment.csv"
    subaction_path = out_dir / "successful_subactions.csv"
    all_frames_path = out_dir / "successful_frames_all.csv"
    counts.to_csv(counts_path, index=False)
    per_subaction.to_csv(subaction_path, index=False)
    success_table.to_csv(all_frames_path, index=False)

    experiment_paths: dict[str, Path] = {}
    for experiment_id, exp_df in success_table.groupby("experiment_id", sort=True):
        path = out_dir / f"successful_frames_{experiment_id}.csv"
        exp_df.to_csv(path, index=False)
        experiment_paths[str(experiment_id)] = path

    readme_path = out_dir / "README.md"
    e2_preview = success_table[success_table["experiment_id"] == "E2"].head(30)
    e3_preview = success_table[success_table["experiment_id"] == "E3"].head(30)
    lines = [
        "# Successful Frames Summary",
        "",
        f"- source: `{candidate_path}`",
        f"- total successful samples: `{len(success_table)}`",
        f"- output directory: `{out_dir}`",
        "",
        "## Counts By Experiment",
        "",
        markdown_table(counts),
        "",
        "## Files",
        "",
        "- `successful_counts_by_experiment.csv`: each experiment's pass counts.",
        "- `successful_subactions.csv`: one row per successful subaction per experiment.",
        "- `successful_frames_all.csv`: one row per successful frame/sample.",
        "- `successful_frames_E0.csv` ... `successful_frames_E3.csv`: per-experiment successful frames.",
        "",
        "## How To Read A Successful Frame Row",
        "",
        "- `frame_0_based`: frame index used by the pipeline.",
        "- `frame_1_based`: image filename index; frame 2143 means `frame_0000002143.jpg`.",
        "- `ref_idx` / `ref_frame_1_based`: selected human-less reference frame.",
        "- `sample_dir`: directory containing the full visual output for that successful sample.",
        "",
        "## E2 Preview",
        "",
        markdown_table(
            e2_preview[
                [
                    "subaction_index",
                    "narration_id",
                    "narration",
                    "frame_0_based",
                    "frame_1_based",
                    "hand",
                    "ref_idx",
                    "sample_dir",
                ]
            ]
        ),
        "",
        "## E3 Preview",
        "",
        markdown_table(
            e3_preview[
                [
                    "subaction_index",
                    "narration_id",
                    "narration",
                    "frame_0_based",
                    "frame_1_based",
                    "hand",
                    "ref_idx",
                    "sample_dir",
                ]
            ]
        ),
    ]
    readme_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "counts": counts_path,
        "subactions": subaction_path,
        "all_frames": all_frames_path,
        "readme": readme_path,
        **{f"frames_{key}": value for key, value in experiment_paths.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Experiment output root containing candidate_diagnostics.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for summary outputs. Defaults to OUTPUT_ROOT/successful_frames.",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else output_root / "successful_frames"
    paths = summarize(output_root, out_dir)

    print(f"Successful-frame summary written to: {out_dir}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
