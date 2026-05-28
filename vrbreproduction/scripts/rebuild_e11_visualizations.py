"""Rebuild E11 visualizations on dehanded reference frames.

E11 is an E10 postprocess: the coordinates, heatmaps, projected contacts, and
trajectories are unchanged, but all visual summaries should use the dehanded
reference image rather than the copied E10 reference image.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vrbreproduction.label_heatmap_utils import draw_vrb_style_affordance_overlay, overlay_heatmap_on_rgb


def binary_mask(value: Any, shape_hw: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape_hw, dtype=np.uint8)
    arr = np.asarray(value, dtype=np.uint8)
    if arr.shape[:2] != shape_hw:
        arr = cv2.resize(arr, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (arr > 0).astype(np.uint8) * 255


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(np.asarray(mask) > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def draw_mask_and_points(
    image_rgb: np.ndarray,
    record: dict[str, Any],
    centers: np.ndarray,
) -> np.ndarray:
    out = image_rgb.copy()
    h, w = out.shape[:2]
    mask = binary_mask(record.get("reference_object_mask"), (h, w))
    if int(np.sum(mask > 0)) > 0:
        green = np.zeros_like(out)
        green[..., 1] = mask
        out = cv2.addWeighted(out, 1.0, green, 0.35, 0.0)
        bbox = mask_bbox(mask)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    tracked_points = record.get("tracked_reference_points")
    tracked_vis = record.get("tracked_reference_visibility")
    if tracked_points is not None:
        pts = np.asarray(tracked_points, dtype=np.float32).reshape(-1, 2)
        vis = np.asarray(tracked_vis, dtype=bool).reshape(-1) if tracked_vis is not None else np.ones(len(pts), dtype=bool)
        for pt, is_vis in zip(pts, vis):
            color = (0, 255, 255) if bool(is_vis) else (128, 128, 128)
            cv2.circle(out, (int(round(pt[0])), int(round(pt[1]))), 2, color, -1)

    for pt in centers:
        cv2.circle(out, (int(round(pt[0])), int(round(pt[1]))), 3, (255, 255, 255), -1)
    return out


def rebuild_sample_visuals(sample_dir: Path) -> dict[str, Any]:
    record_path = sample_dir / "candidate_result.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    dehanded_path = Path(record.get("reference_frame_dehanded") or sample_dir / "reference_frame_dehanded.png")
    if not dehanded_path.exists():
        dehanded_path = sample_dir / "reference_frame.png"
    ref_bgr = cv2.imread(str(dehanded_path))
    if ref_bgr is None:
        raise RuntimeError(f"failed to read {dehanded_path}")
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)

    heatmap_path = Path(record.get("label_heatmap_npy") or sample_dir / "label_heatmap_merged.npy")
    merged = np.load(heatmap_path)
    overlay = overlay_heatmap_on_rgb(ref_rgb, merged, alpha=0.45)
    overlay_path = sample_dir / "label_heatmap_merged_overlay.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    centers = np.asarray(record.get("projected_contact_means"), dtype=np.float32).reshape(-1, 2)
    tau = [np.asarray(pt, dtype=np.float32).reshape(2) for pt in record.get("trajectory_projected", [])]
    anchor = centers.mean(axis=0)
    vrb_img, arrow_info = draw_vrb_style_affordance_overlay(ref_rgb, merged, tau, anchor)
    vrb_img = draw_mask_and_points(vrb_img, record, centers)
    vrb_path = sample_dir / "vrb_style_affordance.png"
    mask_overlay_path = sample_dir / "mask_projection_overlay.png"
    cv2.imwrite(str(vrb_path), cv2.cvtColor(vrb_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_overlay_path), cv2.cvtColor(vrb_img, cv2.COLOR_RGB2BGR))

    record.update(
        {
            "reference_frame": str(dehanded_path),
            "label_heatmap_overlay": str(overlay_path),
            "vrb_style_affordance": str(vrb_path),
            "mask_projection_overlay": str(mask_overlay_path),
            "arrow_generated": arrow_info is not None,
            "visualization_background": "reference_frame_dehanded",
        }
    )
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "reference_frame": str(dehanded_path),
        "label_heatmap_overlay": str(overlay_path),
        "vrb_style_affordance": str(vrb_path),
        "mask_projection_overlay": str(mask_overlay_path),
        "arrow_generated": arrow_info is not None,
    }


def save_funnel_chart(overview_df: pd.DataFrame, path: Path) -> None:
    row = overview_df.iloc[0]
    stages = [
        ("raw candidate", "raw_candidate_total"),
        ("deep run", "deep_run_candidate_total"),
        ("contact GMM", "contact_gmm_pass"),
        ("active object", "active_object_selected"),
        ("object track", "object_track_success"),
        ("reference", "reference_found"),
        ("projection", "projection_success"),
        ("trajectory", "trajectory_success"),
        ("final tuple", "final_tuple_count"),
    ]
    labels = [s[0] for s in stages]
    y = [int(row[s[1]]) for s in stages]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    ax.plot(x, y, marker="o", color="#1a73e8")
    for xi, yi in zip(x, y):
        ax.text(xi, yi, str(yi), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("count")
    ax.set_title("E11 pipeline funnel")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_failure_chart(candidate_df: pd.DataFrame, path: Path) -> None:
    failed = candidate_df[candidate_df["status"] != "keep"]
    if failed.empty:
        return
    counts = failed["fail_reason"].fillna("unknown").value_counts()
    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(counts))))
    counts.sort_values().plot(kind="barh", ax=ax, color="#c5221f")
    ax.set_title("E11 failure reasons")
    ax.set_xlabel("candidate count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_projection_source_counts(candidate_df: pd.DataFrame, path: Path) -> None:
    success = candidate_df[candidate_df["status"] == "keep"]
    if success.empty:
        return
    counts = success["projection_source"].fillna("unknown").value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax, color="#f29900")
    ax.set_title("E11 projection source counts")
    ax.set_xlabel("projection_source")
    ax.set_ylabel("success candidate count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_success_by_subaction(subaction_df: pd.DataFrame, path: Path) -> None:
    y = (subaction_df["final_status"] == "keep").astype(int)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(subaction_df["subaction_index"], y, color=["#188038" if v else "#dadce0" for v in y])
    ax.set_title("E11 success by subaction")
    ax.set_xlabel("subaction_index")
    ax.set_ylabel("success")
    ax.set_ylim(0, 1.2)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_track_confidence_hist(candidate_df: pd.DataFrame, path: Path) -> None:
    values = candidate_df["object_track_confidence"].dropna().astype(float) if "object_track_confidence" in candidate_df else pd.Series(dtype=float)
    if values.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=20, color="#188038", alpha=0.85)
    ax.set_title("E11 track confidence")
    ax.set_xlabel("confidence")
    ax.set_ylabel("candidate count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_contact_sheet(final_df: pd.DataFrame, path: Path, title: str, page_size: int = 36) -> list[Path]:
    if final_df.empty:
        return []
    pages: list[Path] = []
    rows = final_df.reset_index(drop=True)
    for page_idx in range(int(math.ceil(len(rows) / page_size))):
        view = rows.iloc[page_idx * page_size : (page_idx + 1) * page_size]
        images = []
        labels = []
        for _, row in view.iterrows():
            img_path = row.get("mask_projection_overlay") or row.get("vrb_style_affordance") or row.get("label_heatmap_overlay")
            if not isinstance(img_path, str) or not Path(img_path).exists():
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            labels.append(
                f"s{int(row['subaction_index'])} f{int(row['frame_0_based'])} r{int(row['ref_idx'])}\n"
                f"{row.get('verb')} {row.get('noun')} | {row.get('reference_handless_mode')}"
            )
        if not images:
            continue
        cols = min(4, len(images))
        grid_rows = int(math.ceil(len(images) / cols))
        fig, axes = plt.subplots(grid_rows, cols, figsize=(cols * 3.4, grid_rows * 2.8))
        axes_arr = np.asarray(axes).reshape(-1)
        for ax in axes_arr:
            ax.axis("off")
        for ax, img, label in zip(axes_arr, images, labels):
            thumb = cv2.resize(img, (260, 190), interpolation=cv2.INTER_AREA)
            ax.imshow(thumb)
            ax.set_title(label, fontsize=8)
        fig.suptitle(title if page_idx == 0 else f"{title} page {page_idx + 1}", fontsize=12)
        plt.tight_layout()
        out = path if page_idx == 0 and len(rows) <= page_size else path.with_name(f"{path.stem}_page_{page_idx + 1:03d}{path.suffix}")
        plt.savefig(out, dpi=160)
        plt.close()
        pages.append(out)
    return pages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e11-root", type=Path, default=PROJECT_ROOT / "outputs" / "e11")
    args = parser.parse_args()
    root = args.e11_root.resolve()
    charts = root / "charts"
    if charts.exists():
        shutil.rmtree(charts)
    charts.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "successful_sample_manifest.csv"
    candidate_path = root / "candidate_diagnostics.csv"
    subaction_path = root / "subaction_summary.csv"
    overview_path = root / "experiment_overview.csv"
    manifest = pd.read_csv(manifest_path)
    candidate = pd.read_csv(candidate_path)
    subaction = pd.read_csv(subaction_path)
    overview = pd.read_csv(overview_path)

    print(f"[phase] rebuilding E11 per-sample visualizations for {len(manifest)} final tuples", flush=True)
    for pos, row in manifest.iterrows():
        sample_dir = Path(str(row["sample_dir"]))
        artifacts = rebuild_sample_visuals(sample_dir)
        for key, value in artifacts.items():
            manifest.at[pos, key] = value
        if (pos + 1) % 10 == 0 or pos + 1 == len(manifest):
            print(f"[progress] {pos + 1:03d}/{len(manifest):03d} sample visuals rebuilt", flush=True)

    final_keys = {
        (int(row["subaction_index"]), int(row["frame_0_based"]), str(row["hand"]), int(row["ref_idx"])): row.to_dict()
        for _, row in manifest.iterrows()
    }
    final_mask = candidate.get("is_final_tuple", False) == True
    for idx, row in candidate[final_mask].iterrows():
        key = (int(row["subaction_index"]), int(row["frame_0_based"]), str(row["hand"]), int(row["ref_idx"]))
        item = final_keys.get(key)
        if not item:
            continue
        for col in ["reference_frame", "label_heatmap_overlay", "vrb_style_affordance", "mask_projection_overlay"]:
            candidate.at[idx, col] = item.get(col)

    manifest.to_csv(manifest_path, index=False)
    candidate.to_csv(candidate_path, index=False)

    print("[phase] rebuilding E11 charts from dehanded visualizations", flush=True)
    save_funnel_chart(overview, charts / "e11_pipeline_funnel.png")
    save_failure_chart(candidate, charts / "e11_failure_reasons.png")
    save_track_confidence_hist(candidate, charts / "e11_track_confidence_hist.png")
    save_projection_source_counts(candidate, charts / "e11_projection_source_counts.png")
    save_success_by_subaction(subaction, charts / "e11_success_by_subaction.png")
    make_contact_sheet(manifest, charts / "e11_mask_projection_contact_sheet.png", "E11 dehanded mask projection samples", page_size=1000)
    contact_pages = make_contact_sheet(manifest, charts / "e11_success_contact_sheet_page_001.png", "E11 final dehanded samples", page_size=16)

    summary = root / "summary.md"
    if summary.exists():
        text = summary.read_text(encoding="utf-8").rstrip()
    else:
        text = "# E11 handless-reference inpainting postprocess"
    text += "\n\n## Rebuilt Charts\n\n"
    chart_lines = [
        "- `charts/e11_pipeline_funnel.png`",
        "- `charts/e11_failure_reasons.png`",
        "- `charts/e11_track_confidence_hist.png`",
        "- `charts/e11_projection_source_counts.png`",
        "- `charts/e11_success_by_subaction.png`",
        "- `charts/e11_mask_projection_contact_sheet.png`",
    ]
    chart_lines.extend(f"- `{p.relative_to(root)}`" for p in contact_pages)
    summary.write_text(text + "\n".join(chart_lines) + "\n", encoding="utf-8")
    print("[done] rebuilt E11 visualizations and charts", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
