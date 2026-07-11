from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def bool_count(df: pd.DataFrame, column: str) -> int:
    if column not in df:
        return 0
    return int(df[column].fillna(False).astype(bool).sum())


def median_value(df: pd.DataFrame, column: str):
    if column not in df:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def summarize(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    keep = df[df["status"] == "keep"] if "status" in df else pd.DataFrame()
    inpaint_attempted = df.get("inpaint_attempted", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    boundary = pd.to_numeric(df.get("inpaint_boundary_delta", pd.Series(index=df.index, dtype=float)), errors="coerce")
    mask_area = pd.to_numeric(df.get("inpaint_mask_area_ratio", pd.Series(index=df.index, dtype=float)), errors="coerce")
    status = df.get("inpaint_status", pd.Series(index=df.index, dtype=object)).fillna("")
    manual_review = inpaint_attempted & (
        (boundary > 25)
        | (mask_area > 0.20)
        | (status != "applied")
    )
    return pd.Series(
        {
            "total_candidates": int(len(df)),
            "keep_candidates": int(len(keep)),
            "active_hand_visible_at_reference": bool_count(df, "active_hand_visible_at_ref"),
            "inpaint_required": bool_count(df, "inpaint_required"),
            "inpaint_attempted": bool_count(df, "inpaint_attempted"),
            "inpaint_applied": bool_count(df, "inpaint_applied"),
            "failed_kept_original": int((status == "failed_kept_original").sum()),
            "discarded_by_inpaint_gate": int((status == "discarded").sum()),
            "median_mask_area_ratio": median_value(df, "inpaint_mask_area_ratio"),
            "median_boundary_delta": median_value(df, "inpaint_boundary_delta"),
            "manual_review_needed_count": int(manual_review.sum()),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize E19 hand inpainting quality fields.")
    parser.add_argument("candidate_diagnostics_csv", type=Path)
    args = parser.parse_args()
    summary = summarize(args.candidate_diagnostics_csv)
    print(summary.to_string())


if __name__ == "__main__":
    main()
