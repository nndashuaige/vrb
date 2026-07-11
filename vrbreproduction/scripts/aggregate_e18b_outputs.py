"""Aggregate E18B per-subaction pickles into the standard experiment outputs."""
import json, pickle, sys
from pathlib import Path

sys.path.insert(0, str(Path("~/mnt/vrbreproduction/src").expanduser()))

import pandas as pd

from vrbreproduction.e18b_paper_faithful_cell2_retry import (
    E18BConfig, EXPERIMENT, VRBREPRODUCTION_ROOT,
    get_subactions, compute_subaction_frame_coverage,
    summarize_subactions, build_overview,
    save_success_manifests, write_comparison_files,
    save_funnel_chart, save_bar_chart_failure_reasons, save_timeline_chart,
    save_ref_gap_histogram, save_reference_anchor_gap_histogram,
    save_candidate_offset_histogram, save_gap_length_histogram,
    save_reference_mode_ref_gap_histogram, save_paginated_success_contact_sheets,
    ensure_placeholder_png, write_markdown_report, _json_safe,
)

STATE_DIR = Path("/tmp/e18b_state")
config = E18BConfig()
charts_dir = config.output_root / "charts"
charts_dir.mkdir(parents=True, exist_ok=True)

subactions = get_subactions(config)
frame_coverage = compute_subaction_frame_coverage(subactions)

all_records, plan_rows = [], []
for f in sorted(STATE_DIR.glob("sub_*.pkl")):
    with open(f, "rb") as fh:
        blob = pickle.load(fh)
    plan_rows.append(blob["plan"])
    all_records.extend(blob["records"])

candidate_df = pd.DataFrame(all_records)
candidate_plan_df = pd.DataFrame(plan_rows)
subaction_summary_df = summarize_subactions(candidate_df, subactions, candidate_plan_df)
overview_df = build_overview(candidate_df, subaction_summary_df, candidate_plan_df)

experiment_json = config.output_root / "experiment_results.json"
experiment_json.write_text(
    json.dumps(_json_safe({
        "video_id": config.video_id,
        "num_subactions": config.num_subactions,
        "max_ref_backtrack": config.max_ref_backtrack,
        "early_candidate_offsets": list(config.early_candidate_offsets),
        "crop_size": config.crop_size,
        "cell2_retry_max_offset": config.cell2_retry_max_offset,
        "crop_hand_overlap_max": config.crop_hand_overlap_max,
        "hand_object_iou_max": config.hand_object_iou_max,
        "min_hand_object_center_distance_px": config.min_hand_object_center_distance_px,
        "frame_coverage": frame_coverage,
        "experiment": EXPERIMENT,
        "overview": overview_df.to_dict(orient="records"),
        "subaction_summary": subaction_summary_df.to_dict(orient="records"),
        "candidate_diagnostics": candidate_df.to_dict(orient="records"),
        "candidate_plan": candidate_plan_df.to_dict(orient="records"),
    }), ensure_ascii=False, indent=2),
    encoding="utf-8",
)

overview_df.to_csv(config.output_root / "experiment_overview.csv", index=False)
subaction_summary_df.to_csv(config.output_root / "subaction_summary.csv", index=False)
candidate_df.to_csv(config.output_root / "candidate_diagnostics.csv", index=False)
candidate_plan_df.to_csv(config.output_root / "candidate_plan.csv", index=False)

save_success_manifests(candidate_df, config.output_root)
write_comparison_files(candidate_df, config.output_root)
save_funnel_chart(overview_df, charts_dir / "e18b_pipeline_funnel.png")
save_bar_chart_failure_reasons(candidate_df, charts_dir / "e18b_failure_reasons.png")
save_timeline_chart(candidate_df, subactions, charts_dir / "e18b_timeline.png")
success_df = candidate_df[candidate_df["status"] == "keep"].copy()
save_ref_gap_histogram(success_df, charts_dir / "e18b_ref_gap_success_hist.png")
save_reference_anchor_gap_histogram(success_df, charts_dir / "e18b_reference_anchor_gap_hist.png")
save_candidate_offset_histogram(success_df, charts_dir / "e18b_candidate_offset_success_hist.png")
save_gap_length_histogram(candidate_df, charts_dir / "e18b_gap_length_hist.png")
save_reference_mode_ref_gap_histogram(success_df, charts_dir / "e18b_ref_gap_success_by_reference_mode_hist.png")
save_paginated_success_contact_sheets(success_df, charts_dir, "e18b_success_contact_sheet", "E18B success samples", per_page=30)
ensure_placeholder_png(charts_dir / "e18b_success_contact_sheet_page_001.png", "E18B success samples")
save_paginated_success_contact_sheets(success_df, charts_dir, "e18b_success_heatmap_trajectory_contact_sheet", "E18B success heatmap + trajectory overlays", per_page=30)
ensure_placeholder_png(charts_dir / "e18b_success_heatmap_trajectory_contact_sheet_page_001.png", "E18B success heatmap + trajectory overlays")
summary_path = write_markdown_report(
    config=config, overview_df=overview_df, subaction_summary_df=subaction_summary_df,
    candidate_df=candidate_df, candidate_plan_df=candidate_plan_df, frame_coverage=frame_coverage,
)
cols = ["subactions_success","cell2_pass","ref_found","homography_available","geometry_pass","heatmap_pass",
        "cell2_retry_rescued","cell2_retry_rescued_keep","shadow_geometry_fail_keep","shadow_homography_low_quality_keep","main_failure"]
print(overview_df[cols].to_string())
print("summary:", summary_path)
