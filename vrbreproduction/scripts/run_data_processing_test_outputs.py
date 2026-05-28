from collections import Counter
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from epic_kitchens.hoa import load_detections

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vrbreproduction.contact_point_utils import ContactExtractionConfig
from vrbreproduction.label_heatmap_utils import (
    build_label_heatmaps,
    draw_vrb_style_affordance_overlay,
    merge_label_heatmaps,
    save_label_heatmap_outputs,
    transform_covariances_by_homography,
)
from vrbreproduction.pipeline_retention import fit_cell2_contact_gmm, scan_contact_point_frames, _json_safe
from vrbreproduction.problem3_runner import run_problem3_cell3


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

SCAN_LIMIT = 500
HOA_PKL = PROJECT_ROOT / "data/P01_109.pkl"
IMAGE_DIR = PROJECT_ROOT / "data/P01_109_frames"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "数据处理测试"


def _silent(func, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _make_heatmap_artifacts(sample, cell2_result, cell3_result, sample_output_dir):
    ref_idx = int(cell3_result["ref_idx"])
    ref_path = IMAGE_DIR / f"frame_{ref_idx + 1:010d}.jpg"
    ref_img_bgr = cv2.imread(str(ref_path))
    if ref_img_bgr is None:
        raise FileNotFoundError(f"Missing reference frame: {ref_path}")

    ref_img_rgb = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = ref_img_rgb.shape[:2]

    contact_means = cell2_result["contact_means"]
    contact_weights = cell2_result["contact_weights"]
    contact_covariances = cell2_result["contact_covariances"]
    mu_transformed = np.asarray(cell3_result["mu_transformed"], dtype=np.float32)
    H_contact_to_ref = cell3_result["H_contact_to_ref"]

    heatmap_mode = "covariance"
    covariances_ref = transform_covariances_by_homography(
        contact_covariances,
        contact_means,
        H_contact_to_ref,
    )
    per_mode_heatmaps = build_label_heatmaps(
        image_shape=(h_img, w_img),
        centers_xy=mu_transformed,
        sigma_px=12.0,
        weights=contact_weights,
        normalize_each=True,
        covariances_xy=covariances_ref,
        covariance_scale=6.0,
        min_sigma_px=7.0,
        max_sigma_px=40.0,
    )
    merged_heatmap = merge_label_heatmaps(
        per_mode_heatmaps=per_mode_heatmaps,
        merge_method="sum",
        normalize_output=True,
    )

    heatmap_paths = _silent(
        save_label_heatmap_outputs,
        merged_heatmap=merged_heatmap,
        ref_image=ref_img_rgb,
        output_dir=sample_output_dir,
        prefix="label_heatmap",
        sigma_px=12.0,
        merge_method="sum",
        use_weights=True,
        heatmap_mode=heatmap_mode,
        t_contact=sample["frame"],
        active_hand=sample["hand"],
        ref_idx=ref_idx,
    )

    reference_path = sample_output_dir / "reference_frame.png"
    cv2.imwrite(str(reference_path), ref_img_bgr)

    arrow_anchor_xy = cell3_result.get("contact_centroid")
    if arrow_anchor_xy is None:
        arrow_anchor_xy = mu_transformed.mean(axis=0)
    vrb_style_img, arrow_info = draw_vrb_style_affordance_overlay(
        ref_img_rgb,
        merged_heatmap,
        cell3_result.get("tau_transformed", []),
        arrow_anchor_xy,
        heatmap_alpha=0.45,
        colormap=cv2.COLORMAP_JET,
        arrow_length_px=None,
        arrow_length_heatmap_ratio=1.2,
        arrow_width_px=None,
        source_step="last",
    )
    vrb_style_path = sample_output_dir / "vrb_style_affordance.png"
    cv2.imwrite(str(vrb_style_path), cv2.cvtColor(vrb_style_img, cv2.COLOR_RGB2BGR))

    return {
        "reference_frame": str(reference_path),
        "label_heatmap_npy": heatmap_paths["npy_path"],
        "label_heatmap_png": heatmap_paths["png_path"],
        "label_heatmap_overlay": heatmap_paths["overlay_path"],
        "vrb_style_affordance": str(vrb_style_path),
        "arrow_info": arrow_info,
        "heatmap_mode": heatmap_mode,
        "heatmap_shape": tuple(int(v) for v in merged_heatmap.shape),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detections = load_detections(str(HOA_PKL))
    contact_config = ContactExtractionConfig(
        use_object_mask=True,
        use_object_boundary=True,
        boundary_distance_px=6.0,
        project_points_to_object_boundary=True,
    )

    valid_cell1, rejected_cell1, cell1_reasons = scan_contact_point_frames(
        detections=detections,
        scan_limit=SCAN_LIMIT,
        image_dir=IMAGE_DIR,
        contact_extraction_config=contact_config,
    )

    records = []
    cell2_passed = []
    cell2_reasons = Counter()
    for sample in valid_cell1:
        cell2_result = fit_cell2_contact_gmm(
            detections=detections,
            sample=sample,
            image_dir=IMAGE_DIR,
            contact_extraction_config=contact_config,
        )
        if cell2_result["passed"]:
            cell2_passed.append((sample, cell2_result))
            continue
        cell2_reasons[cell2_result["reason"]] += 1
        records.append({
            "frame": sample["frame"],
            "hand": sample["hand"],
            "status": "discard",
            "failed_cell": "Cell 2",
            "discard_reason": cell2_result["reason"],
            "contact_points": cell2_result.get("contact_points", 0),
        })

    cell3_passed = []
    cell3_reasons = Counter()
    diagnostics_dir = OUTPUT_DIR / "diagnostics"
    for sample, cell2_result in cell2_passed:
        cell3_result = _silent(
            run_problem3_cell3,
            detections=detections,
            t_contact=sample["frame"],
            active_hand=sample["hand"],
            contact_means=cell2_result["contact_means"],
            discard=False,
            discard_reason=None,
            image_dir=IMAGE_DIR,
            output_dir=diagnostics_dir,
        )
        if not cell3_result.get("discard") and cell3_result.get("status") == "KEEP":
            cell3_passed.append((sample, cell2_result, cell3_result))
            continue
        reason = cell3_result.get("discard_reason") or "cell3_discard"
        cell3_reasons[reason] += 1
        records.append({
            "frame": sample["frame"],
            "hand": sample["hand"],
            "status": "discard",
            "failed_cell": "Cell 3",
            "discard_reason": reason,
            "contact_points": cell2_result.get("contact_points", 0),
            "ref_idx": cell3_result.get("ref_idx"),
        })

    cell4_reasons = Counter()
    keep_records = []
    for sample, cell2_result, cell3_result in cell3_passed:
        sample_dir = OUTPUT_DIR / f"frame_{int(sample['frame']):04d}_{sample['hand']}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        try:
            artifacts = _make_heatmap_artifacts(
                sample=sample,
                cell2_result=cell2_result,
                cell3_result=cell3_result,
                sample_output_dir=sample_dir,
            )
        except Exception as exc:
            reason = f"heatmap_save_failed:{type(exc).__name__}"
            cell4_reasons[reason] += 1
            records.append({
                "frame": sample["frame"],
                "hand": sample["hand"],
                "status": "discard",
                "failed_cell": "Cell 4",
                "discard_reason": reason,
                "contact_points": cell2_result.get("contact_points", 0),
                "ref_idx": cell3_result.get("ref_idx"),
            })
            continue

        record = {
            "frame": int(sample["frame"]),
            "frame_1_based": int(sample["frame"]) + 1,
            "hand": sample["hand"],
            "status": "keep",
            "failed_cell": None,
            "discard_reason": None,
            "contact_points": int(cell2_result.get("contact_points", 0)),
            "ref_idx": int(cell3_result["ref_idx"]),
            "ref_frame_1_based": int(cell3_result["ref_idx"]) + 1,
            "heatmap_mode": artifacts["heatmap_mode"],
            "output_dir": str(sample_dir),
            "artifacts": artifacts,
        }
        records.append(record)
        keep_records.append(record)

    for item in rejected_cell1:
        records.append({
            "frame": item["frame"],
            "hand": item.get("hand"),
            "status": "discard",
            "failed_cell": "Cell 1-test",
            "discard_reason": item.get("discard_reason", item.get("status", "cell1_rejected")),
        })

    summary = [
        {
            "cell": "Cell 1-test",
            "stage": "contact_point_precheck",
            "entered_frames": int(cell1_reasons.get("scanned_frames", SCAN_LIMIT)),
            "kept_frames": len(valid_cell1),
            "discarded_frames": int(cell1_reasons.get("scanned_frames", SCAN_LIMIT)) - len(valid_cell1),
        },
        {
            "cell": "Cell 2",
            "stage": "contact_gmm",
            "entered_frames": len(valid_cell1),
            "kept_frames": len(cell2_passed),
            "discarded_frames": len(valid_cell1) - len(cell2_passed),
        },
        {
            "cell": "Cell 3",
            "stage": "homography_geometry",
            "entered_frames": len(cell2_passed),
            "kept_frames": len(cell3_passed),
            "discarded_frames": len(cell2_passed) - len(cell3_passed),
        },
        {
            "cell": "Cell 4",
            "stage": "label_heatmap",
            "entered_frames": len(cell3_passed),
            "kept_frames": len(keep_records),
            "discarded_frames": len(cell3_passed) - len(keep_records),
        },
    ]

    result = {
        "notebook": "数据处理测试.ipynb",
        "scan_limit": SCAN_LIMIT,
        "hoa_pkl": str(HOA_PKL),
        "image_dir": str(IMAGE_DIR),
        "output_dir": str(OUTPUT_DIR),
        "full_pipeline_frames": len(keep_records),
        "summary": summary,
        "reason_counts": {
            "Cell 1-test": {
                k: int(v)
                for k, v in cell1_reasons.items()
                if k != "scanned_frames"
            },
            "Cell 2": {k: int(v) for k, v in cell2_reasons.items()},
            "Cell 3": {k: int(v) for k, v in cell3_reasons.items()},
            "Cell 4": {k: int(v) for k, v in cell4_reasons.items()},
        },
        "kept_frames": keep_records,
        "records": sorted(records, key=lambda r: (int(r["frame"]), str(r.get("hand")))),
    }

    json_path = OUTPUT_DIR / "pipeline_outputs_500.json"
    json_path.write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 数据处理测试 - 前 500 帧最终输出",
        "",
        f"- notebook: `数据处理测试.ipynb`",
        f"- scan limit: `{SCAN_LIMIT}`",
        f"- full pipeline frames: `{len(keep_records)}`",
        "",
        "## Stage Summary",
        "",
        "| cell | input | kept | discarded |",
        "|---|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['cell']} | {row['entered_frames']} | "
            f"{row['kept_frames']} | {row['discarded_frames']} |"
        )
    lines.extend([
        "",
        "## Kept Frames",
        "",
        "| frame_0_based | frame_1_based | hand | ref_frame_0_based | contact_points | output_dir |",
        "|---:|---:|---|---:|---:|---|",
    ])
    for record in keep_records:
        lines.append(
            f"| {record['frame']} | {record['frame_1_based']} | {record['hand']} | "
            f"{record['ref_idx']} | {record['contact_points']} | `{record['output_dir']}` |"
        )
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("=" * 80)
    print("前 500 帧完整 pipeline 最终输出")
    print("=" * 80)
    for row in summary:
        print(
            "{cell} 进入流程帧数: {entered_frames}, 保留可用: {kept_frames}, 不能用: {discarded_frames}".format(
                **row
            )
        )
    print(f"完整 pipeline 出结果帧数: {len(keep_records)}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"统计 JSON: {json_path}")


if __name__ == "__main__":
    main()
