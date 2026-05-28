"""Build E11 outputs by dehanding an E10 output copy.

This is intentionally a postprocess step. It does not rerun candidate
selection, CoTracker, projection, part-aware gates, top-k export, heatmap
coordinate generation, or trajectory projection. It only replaces the exported
reference training image with an inpainted hand-less version and records
diagnostics.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from epic_kitchens.hoa import load_detections


REMOTE_E10_ROOT = "/root/workspace/vrb/vrbreproduction/outputs/100subaction-e10"
LOCAL_E10_ROOT = "/Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/outputs/e10"
LOCAL_E11_ROOT = "/Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/outputs/e11"


def localize_path(value: Any, e10_root: Path, e11_root: Path) -> Any:
    if not isinstance(value, str) or not value:
        return value
    out = value
    out = out.replace(REMOTE_E10_ROOT, str(e11_root))
    out = out.replace(str(e10_root), str(e11_root))
    out = out.replace("/experiments/E10/", "/experiments/E11/")
    out = out.replace("100subaction-e10", "e11")
    out = out.replace("E10", "E11")
    return out


def clip_bbox(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def hand_mask_from_detections(frame_det: Any, shape: tuple[int, int, int], score_threshold: float, expand_px: int, dilate_px: int) -> tuple[np.ndarray, int]:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    count = 0
    for hand in getattr(frame_det, "hands", []):
        if float(hand.score) < float(score_threshold):
            continue
        x1 = float(hand.bbox.left) * w - expand_px
        y1 = float(hand.bbox.top) * h - expand_px
        x2 = float(hand.bbox.right) * w + expand_px
        y2 = float(hand.bbox.bottom) * h + expand_px
        ix1, iy1, ix2, iy2 = clip_bbox((x1, y1, x2, y2), w, h)
        mask[iy1:iy2, ix1:ix2] = 255
        count += 1
    if dilate_px > 0 and int(np.sum(mask > 0)) > 0:
        k = 2 * int(dilate_px) + 1
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1)
    return mask, count


def binary_mask(value: Any, shape_hw: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape_hw, dtype=np.uint8)
    arr = np.asarray(value, dtype=np.uint8)
    if arr.shape[:2] != shape_hw:
        arr = cv2.resize(arr, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (arr > 0).astype(np.uint8) * 255


def bbox_overlap(mask: np.ndarray, bbox: Any) -> float:
    if bbox is None:
        return 0.0
    if isinstance(bbox, str):
        try:
            bbox = json.loads(bbox)
        except Exception:
            return 0.0
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0.0
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = clip_bbox(tuple(float(v) for v in bbox), w, h)
    area = max(1, (x2 - x1) * (y2 - y1))
    return float(np.sum(mask[y1:y2, x1:x2] > 0) / area)


def mask_overlap_ratio(hand_mask: np.ndarray, target_mask: np.ndarray) -> float:
    denom = int(np.sum(target_mask > 0))
    if denom <= 0:
        return 0.0
    return float(np.sum((hand_mask > 0) & (target_mask > 0)) / denom)


def infer_crop_bbox(record: dict[str, Any], shape: tuple[int, int, int], crop_size: int = 150) -> tuple[int, int, int, int] | None:
    bbox = record.get("crop_bbox")
    if bbox is not None:
        if isinstance(bbox, str):
            try:
                bbox = json.loads(bbox)
            except Exception:
                bbox = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return clip_bbox(tuple(float(v) for v in bbox), shape[1], shape[0])
    centers = record.get("projected_contact_means")
    if not centers:
        return None
    arr = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    cx, cy = arr.mean(axis=0)
    half = crop_size / 2.0
    x1 = int(round(float(cx) - half))
    y1 = int(round(float(cy) - half))
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > shape[1]:
        x1 -= x2 - shape[1]
        x2 = shape[1]
    if y2 > shape[0]:
        y1 -= y2 - shape[0]
        y2 = shape[0]
    return clip_bbox((x1, y1, x2, y2), shape[1], shape[0])


def dehand_sample(sample_dir: Path, record: dict[str, Any], detections: Any, args: argparse.Namespace) -> dict[str, Any]:
    original_ref = sample_dir / "reference_frame.png"
    if not original_ref.exists():
        raise FileNotFoundError(str(original_ref))
    ref_bgr = cv2.imread(str(original_ref))
    if ref_bgr is None:
        raise RuntimeError(f"failed to read {original_ref}")

    ref_idx = int(record["ref_idx"])
    hand_mask, hand_bbox_count = hand_mask_from_detections(
        detections[ref_idx],
        ref_bgr.shape,
        score_threshold=float(args.hand_score_threshold),
        expand_px=int(args.expand_px),
        dilate_px=int(args.dilate_px),
    )
    h, w = ref_bgr.shape[:2]
    crop_bbox = infer_crop_bbox(record, ref_bgr.shape, crop_size=int(args.crop_size))
    object_mask = binary_mask(record.get("reference_object_mask"), (h, w))
    part_mask = binary_mask(record.get("reference_part_mask"), (h, w))

    mode = "strict_no_hand"
    backend = "none"
    status = "not_needed"
    error = ""
    dehanded = ref_bgr.copy()
    if hand_bbox_count > 0:
        crop_overlap = bbox_overlap(hand_mask, crop_bbox)
        mode = "local_clean" if crop_overlap <= float(args.hand_overlap_max) else "inpainted_bbox"
        backend = "opencv_telea"
        try:
            dehanded = cv2.inpaint(ref_bgr, hand_mask, float(args.inpaint_radius), cv2.INPAINT_TELEA)
            status = "success"
        except Exception as exc:
            dehanded = ref_bgr.copy()
            mode = "unresolved"
            backend = "opencv_telea"
            status = "failed"
            error = f"{type(exc).__name__}:{exc}"

    original_out = sample_dir / "reference_frame_original.png"
    dehanded_out = sample_dir / "reference_frame_dehanded.png"
    hand_mask_out = sample_dir / "hand_mask.png"
    if not original_out.exists():
        shutil.copy2(original_ref, original_out)
    cv2.imwrite(str(dehanded_out), dehanded)
    cv2.imwrite(str(hand_mask_out), hand_mask)
    cv2.imwrite(str(original_ref), dehanded)

    diagnostics = {
        "reference_handless_mode": mode,
        "reference_hand_mask_area_ratio": float(np.sum(hand_mask > 0) / max(1, h * w)),
        "reference_object_hand_overlap_ratio": mask_overlap_ratio(hand_mask, object_mask),
        "reference_contact_crop_hand_overlap_ratio": bbox_overlap(hand_mask, crop_bbox),
        "reference_contact_part_hand_overlap_ratio": mask_overlap_ratio(hand_mask, part_mask),
        "inpaint_backend": backend,
        "inpaint_status": status,
        "inpaint_error": error,
        "reference_frame_original": str(original_out),
        "reference_frame_dehanded": str(dehanded_out),
        "hand_mask": str(hand_mask_out),
        "reference_frame": str(dehanded_out),
        "reference_hand_bbox_count": int(hand_bbox_count),
    }
    record.update(diagnostics)
    (sample_dir / "candidate_result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return diagnostics


def rewrite_csv_paths(df: pd.DataFrame, e10_root: Path, e11_root: Path) -> pd.DataFrame:
    out = df.copy()
    if "experiment_id" in out.columns:
        out["experiment_id"] = out["experiment_id"].replace({"E10": "E11"})
    else:
        out["experiment_id"] = "E11"
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(lambda v: localize_path(v, e10_root, e11_root))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e10-root", type=Path, default=Path(LOCAL_E10_ROOT))
    parser.add_argument("--e11-root", type=Path, default=Path(LOCAL_E11_ROOT))
    parser.add_argument("--hoa-pkl", type=Path, default=Path("/Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/data/P01_109.pkl"))
    parser.add_argument("--hand-score-threshold", type=float, default=0.5)
    parser.add_argument("--hand-overlap-max", type=float, default=0.08)
    parser.add_argument("--expand-px", type=int, default=14)
    parser.add_argument("--dilate-px", type=int, default=9)
    parser.add_argument("--inpaint-radius", type=float, default=5.0)
    parser.add_argument("--crop-size", type=int, default=150)
    args = parser.parse_args()

    e10_root = args.e10_root.resolve()
    e11_root = args.e11_root.resolve()
    manifest_path = e11_root / "successful_sample_manifest.csv"
    candidate_path = e11_root / "candidate_diagnostics.csv"
    subaction_path = e11_root / "subaction_summary.csv"
    overview_path = e11_root / "experiment_overview.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing copied manifest: {manifest_path}")

    print(f"[phase] loading detections from {args.hoa_pkl}", flush=True)
    detections = load_detections(str(args.hoa_pkl))
    manifest = rewrite_csv_paths(pd.read_csv(manifest_path), e10_root, e11_root)
    candidate = rewrite_csv_paths(pd.read_csv(candidate_path), e10_root, e11_root) if candidate_path.exists() else pd.DataFrame()
    subaction = rewrite_csv_paths(pd.read_csv(subaction_path), e10_root, e11_root) if subaction_path.exists() else pd.DataFrame()
    overview = rewrite_csv_paths(pd.read_csv(overview_path), e10_root, e11_root) if overview_path.exists() else pd.DataFrame()

    for col in [
        "reference_handless_mode",
        "reference_hand_mask_area_ratio",
        "reference_object_hand_overlap_ratio",
        "reference_contact_crop_hand_overlap_ratio",
        "reference_contact_part_hand_overlap_ratio",
        "inpaint_backend",
        "inpaint_status",
        "inpaint_error",
        "reference_frame_original",
        "reference_frame_dehanded",
        "hand_mask",
        "reference_hand_bbox_count",
    ]:
        if col not in manifest.columns:
            manifest[col] = np.nan
        if not candidate.empty and col not in candidate.columns:
            candidate[col] = np.nan

    total = len(manifest)
    print(f"[phase] dehanding {total} final E10 samples into E11 copy", flush=True)
    stats: list[dict[str, Any]] = []
    for pos, row in manifest.iterrows():
        sample_dir = Path(str(row["sample_dir"]))
        if not sample_dir.exists():
            sample_dir = Path(localize_path(str(row["sample_dir"]), e10_root, e11_root))
        record_path = sample_dir / "candidate_result.json"
        record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else row.dropna().to_dict()
        record["experiment_id"] = "E11"
        for key, value in list(record.items()):
            if isinstance(value, str):
                record[key] = localize_path(value, e10_root, e11_root)
        diag = dehand_sample(sample_dir, record, detections, args)
        stats.append(diag)
        for key, value in diag.items():
            manifest.at[pos, key] = value
        manifest.at[pos, "experiment_id"] = "E11"
        manifest.at[pos, "reference_frame"] = diag["reference_frame"]
        if (pos + 1) % 10 == 0 or pos + 1 == total:
            print(f"[progress] {pos + 1:03d}/{total:03d} samples dehanded", flush=True)

    if not candidate.empty:
        final_rows = candidate.get("is_final_tuple", False) == True
        key_cols = ["subaction_index", "frame_0_based", "hand", "ref_idx"]
        manifest_key = {
            tuple(str(row.get(c)) for c in key_cols): row.to_dict()
            for _, row in manifest.iterrows()
        }
        for idx, row in candidate[final_rows].iterrows():
            key = tuple(str(row.get(c)) for c in key_cols)
            match = manifest_key.get(key)
            if not match:
                continue
            for col in [
                "reference_handless_mode",
                "reference_hand_mask_area_ratio",
                "reference_object_hand_overlap_ratio",
                "reference_contact_crop_hand_overlap_ratio",
                "reference_contact_part_hand_overlap_ratio",
                "inpaint_backend",
                "inpaint_status",
                "inpaint_error",
                "reference_frame_original",
                "reference_frame_dehanded",
                "hand_mask",
                "reference_hand_bbox_count",
                "reference_frame",
            ]:
                candidate.at[idx, col] = match.get(col)
            candidate.at[idx, "experiment_id"] = "E11"

    if not subaction.empty:
        subaction["experiment_id"] = "E11"
        for idx, row in subaction.iterrows():
            matches = manifest[manifest["subaction_index"] == row["subaction_index"]]
            if not matches.empty:
                best = matches.iloc[0]
                for col in ["reference_handless_mode", "inpaint_status", "reference_hand_mask_area_ratio"]:
                    subaction.at[idx, f"best_{col}"] = best.get(col)

    if not overview.empty:
        overview["experiment_id"] = "E11"
        mode_counts = manifest["reference_handless_mode"].fillna("unknown").value_counts()
        backend_counts = manifest["inpaint_backend"].fillna("unknown").value_counts()
        status_counts = manifest["inpaint_status"].fillna("unknown").value_counts()
        overview["strict_no_hand_count"] = int(mode_counts.get("strict_no_hand", 0))
        overview["local_clean_count"] = int(mode_counts.get("local_clean", 0))
        overview["inpainted_bbox_count"] = int(mode_counts.get("inpainted_bbox", 0))
        overview["inpainted_sam2_count"] = int(mode_counts.get("inpainted_sam2", 0))
        overview["unresolved_count"] = int(mode_counts.get("unresolved", 0))
        overview["opencv_telea_count"] = int(backend_counts.get("opencv_telea", 0))
        overview["inpaint_success_count"] = int(status_counts.get("success", 0))
        overview["inpaint_failed_count"] = int(status_counts.get("failed", 0))
        overview["inpaint_not_needed_count"] = int(status_counts.get("not_needed", 0))

    manifest.to_csv(manifest_path, index=False)
    if not candidate.empty:
        candidate.to_csv(candidate_path, index=False)
    if not subaction.empty:
        subaction.to_csv(subaction_path, index=False)
    if not overview.empty:
        overview.to_csv(overview_path, index=False)

    mode_counts = manifest["reference_handless_mode"].fillna("unknown").value_counts()
    backend_counts = manifest["inpaint_backend"].fillna("unknown").value_counts()
    status_counts = manifest["inpaint_status"].fillna("unknown").value_counts()
    e10_overview = pd.read_csv(e10_root / "experiment_overview.csv").iloc[0].to_dict()
    e11_overview = overview.iloc[0].to_dict() if not overview.empty else e10_overview
    summary_lines = [
        "# E11 handless-reference inpainting postprocess",
        "",
        f"- Source E10 output: `{e10_root}`",
        f"- E11 output: `{e11_root}`",
        "- Projection/candidate/reference selection: unchanged from E10; this run only dehands the exported reference image.",
        f"- Subactions covered: `{int(e11_overview.get('subactions_success', 0))}`",
        f"- Final tuples: `{int(e11_overview.get('final_tuple_count', 0))}`",
        f"- E10 baseline subactions/final tuples: `{int(e10_overview.get('subactions_success', 0))}` / `{int(e10_overview.get('final_tuple_count', 0))}`",
        f"- CoTracker success count: `{int(e11_overview.get('cotracker_success', 0))}`",
        f"- fallback tracker success count: `{int(e11_overview.get('fallback_tracker_success', 0))}`",
        "",
        "## reference_handless_mode",
        "",
        "| mode | count |",
        "| --- | ---: |",
    ]
    summary_lines.extend(f"| {k} | {int(v)} |" for k, v in mode_counts.items())
    summary_lines.extend(["", "## inpaint_backend", "", "| backend | count |", "| --- | ---: |"])
    summary_lines.extend(f"| {k} | {int(v)} |" for k, v in backend_counts.items())
    summary_lines.extend(["", "## inpaint_status", "", "| status | count |", "| --- | ---: |"])
    summary_lines.extend(f"| {k} | {int(v)} |" for k, v in status_counts.items())
    summary_lines.extend(
        [
            "",
            "## Output files",
            "",
            "- `experiment_overview.csv`",
            "- `subaction_summary.csv`",
            "- `candidate_diagnostics.csv`",
            "- `successful_sample_manifest.csv`",
            "- Per-sample: `reference_frame_original.png`, `reference_frame_dehanded.png`, `hand_mask.png`",
        ]
    )
    (e11_root / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("[done] E11 postprocess complete", flush=True)
    print(mode_counts.to_string(), flush=True)
    print(status_counts.to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
