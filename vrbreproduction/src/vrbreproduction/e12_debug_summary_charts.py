"""Generate E12 debug summary contact sheets from existing outputs.

This script does not rerun E12 projection. It reads the downloaded E12 CSVs,
HOA detections, and frame images, then writes object/hand/contact/reference
diagnostic sheets under outputs/e12/charts/排查汇总.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

VRBREPRODUCTION_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(VRBREPRODUCTION_ROOT / "src"))
    __package__ = "vrbreproduction"

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLCONFIGDIR", str(VRBREPRODUCTION_ROOT / ".mplconfig"))

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from epic_kitchens.hoa import load_detections
from sklearn.mixture import GaussianMixture

from .contact_point_utils import (
    ContactExtractionConfig,
    extract_contact_points,
    get_active_hand_bbox,
    get_valid_object_bboxes,
    normalized_bbox_to_pixels,
    select_active_object_bbox,
)


OUTPUT_ROOT = VRBREPRODUCTION_ROOT / "outputs" / "e12"
SUMMARY_DIR = OUTPUT_ROOT / "charts" / "排查汇总"
IMAGE_DIR = VRBREPRODUCTION_ROOT / "data" / "P01_109_frames"
HOA_PKL = VRBREPRODUCTION_ROOT / "data" / "P01_109.pkl"
REMOTE_OUTPUT_ROOT = "/root/workspace/vrb/vrbreproduction/outputs/100subaction-e12"
LOCAL_OUTPUT_ROOT = str(OUTPUT_ROOT)


def to_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def parse_bbox(value: Any) -> Optional[List[float]]:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return [float(v) for v in value]
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return None
    if isinstance(parsed, (list, tuple)) and len(parsed) == 4:
        return [float(v) for v in parsed]
    return None


def localize_output_path(value: Any) -> Optional[Path]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith(REMOTE_OUTPUT_ROOT):
        text = text.replace(REMOTE_OUTPUT_ROOT, LOCAL_OUTPUT_ROOT, 1)
    elif text.startswith("/root/workspace/vrb/vrbreproduction"):
        text = text.replace("/root/workspace/vrb/vrbreproduction", str(VRBREPRODUCTION_ROOT), 1)
    return Path(text)


def frame_path(frame_0_based: int) -> Path:
    return IMAGE_DIR / f"frame_{frame_0_based + 1:010d}.jpg"


def read_rgb(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def bbox_to_xyxy_px(bbox_norm: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = normalized_bbox_to_pixels(bbox_norm, width, height)
    return int(x1), int(y1), int(x2), int(y2)


def draw_rect(
    ax: plt.Axes,
    xyxy: Sequence[float],
    color: str,
    label: str,
    linewidth: float = 2.0,
    linestyle: str = "-",
) -> None:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    rect = plt.Rectangle(
        (x1, y1),
        max(1.0, x2 - x1),
        max(1.0, y2 - y1),
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    ax.add_patch(rect)
    ax.text(
        x1,
        max(0.0, y1 - 3.0),
        label,
        color="black",
        fontsize=6,
        bbox=dict(facecolor=color, alpha=0.78, edgecolor="none", pad=1.0),
    )


def detect_active_object_bbox(frame_det: Any, active_hand: str, width: int, height: int) -> Tuple[Optional[List[float]], Optional[List[float]], List[List[float]]]:
    hand_bbox = get_active_hand_bbox(frame_det, active_hand, score_threshold=0.5)
    object_bboxes = get_valid_object_bboxes(frame_det, score_threshold=0.01)
    object_bboxes_high = get_valid_object_bboxes(frame_det, score_threshold=0.5)
    active_object = None
    if hand_bbox is not None:
        active_object = select_active_object_bbox(hand_bbox, object_bboxes_high or object_bboxes)
    return hand_bbox, active_object, object_bboxes


def fit_contact_debug(
    img_bgr: np.ndarray,
    frame_det: Any,
    active_hand: str,
) -> Dict[str, Any]:
    config = ContactExtractionConfig()
    active_hand_bbox = get_active_hand_bbox(frame_det, active_hand, score_threshold=config.hand_score_threshold)
    object_bboxes = get_valid_object_bboxes(frame_det, score_threshold=config.object_score_threshold)
    if active_hand_bbox is None or not object_bboxes:
        return {"passed": False, "reason": "missing_hand_or_object_bbox"}
    extraction = extract_contact_points(img_bgr, active_hand_bbox, object_bboxes, config=config)
    contact_points = extraction["contact_points"]
    result: Dict[str, Any] = {
        "passed": False,
        "reason": None,
        "extraction": extraction,
        "contact_points": contact_points,
        "means": np.empty((0, 2), dtype=np.float32),
    }
    if len(contact_points) < 5:
        result["reason"] = "contact_points_lt5_after_object_boundary_filter"
        return result
    try:
        gmm = GaussianMixture(n_components=5, random_state=42)
        gmm.fit(contact_points)
    except Exception as exc:
        result["reason"] = f"gmm_failed:{type(exc).__name__}"
        return result
    result["passed"] = True
    result["means"] = np.asarray(gmm.means_, dtype=np.float32)
    return result


def annotate_common(ax: plt.Axes, row: pd.Series) -> None:
    sub = to_int(row.get("subaction_index"))
    frame = to_int(row.get("frame_0_based"))
    ref = row.get("ref_idx", "")
    hand = str(row.get("hand", ""))
    clean = str(row.get("reference_clean_class", ""))
    final = bool(row.get("is_final_tuple") is True or str(row.get("is_final_tuple")).lower() == "true")
    title = f"S{sub:02d} f{frame} {hand}"
    if ref != "":
        title += f" ref{to_int(ref)}"
    if final:
        title += " final"
    if clean:
        title += f"\\n{clean}"
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def render_object_panel(row: pd.Series, detections: Sequence[Any]) -> Optional[np.ndarray]:
    frame_idx = to_int(row.get("frame_0_based"), -1)
    if frame_idx < 0:
        return None
    img_rgb = read_rgb(frame_path(frame_idx))
    if img_rgb is None:
        return None
    h, w = img_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(3.2, 2.0), dpi=120)
    ax.imshow(img_rgb)
    frame_det = detections[frame_idx]
    _, active_object, object_bboxes = detect_active_object_bbox(frame_det, str(row.get("hand", "")).lower(), w, h)
    active_px = parse_bbox(row.get("active_object_bbox"))
    for obj_idx, bbox_norm in enumerate(object_bboxes):
        xyxy = bbox_to_xyxy_px(bbox_norm, w, h)
        is_active = active_object is not None and np.allclose(np.asarray(bbox_norm), np.asarray(active_object), atol=1e-4)
        draw_rect(ax, xyxy, "#39d353" if is_active else "#ffd33d", "active obj" if is_active else f"obj {obj_idx}", 2.2 if is_active else 1.2)
    if active_px is not None:
        draw_rect(ax, active_px, "#00ffff", "E12 active_object", 1.4, "--")
    annotate_common(ax, row)
    return fig_to_rgb(fig)


def render_hand_panel(row: pd.Series, detections: Sequence[Any]) -> Optional[np.ndarray]:
    frame_idx = to_int(row.get("frame_0_based"), -1)
    if frame_idx < 0:
        return None
    img_rgb = read_rgb(frame_path(frame_idx))
    if img_rgb is None:
        return None
    h, w = img_rgb.shape[:2]
    active_hand = str(row.get("hand", "")).lower()
    fig, ax = plt.subplots(figsize=(3.2, 2.0), dpi=120)
    ax.imshow(img_rgb)
    frame_det = detections[frame_idx]
    for hand in getattr(frame_det, "hands", []):
        if hand.score <= 0.01:
            continue
        side = hand.side.name.lower()
        xyxy = bbox_to_xyxy_px([hand.bbox.left, hand.bbox.top, hand.bbox.right, hand.bbox.bottom], w, h)
        is_active = side == active_hand and hand.score > 0.5
        label = f"{'active ' if is_active else ''}{side} {hand.score:.2f}"
        draw_rect(ax, xyxy, "#2f81f7" if is_active else "#d2a8ff", label, 2.2 if is_active else 1.2)
    annotate_common(ax, row)
    return fig_to_rgb(fig)


def render_contact_panel(row: pd.Series, detections: Sequence[Any]) -> Optional[np.ndarray]:
    frame_idx = to_int(row.get("frame_0_based"), -1)
    if frame_idx < 0:
        return None
    path = frame_path(frame_idx)
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    active_hand = str(row.get("hand", "")).lower()
    frame_det = detections[frame_idx]
    fig, ax = plt.subplots(figsize=(3.2, 2.0), dpi=120)
    ax.imshow(img_rgb)
    hand_bbox, active_object, _ = detect_active_object_bbox(frame_det, active_hand, w, h)
    if active_object is not None:
        draw_rect(ax, bbox_to_xyxy_px(active_object, w, h), "#39d353", "active obj", 1.6)
    if hand_bbox is not None:
        draw_rect(ax, bbox_to_xyxy_px(hand_bbox, w, h), "#2f81f7", "active hand", 1.6)
    contact = fit_contact_debug(img_bgr, frame_det, active_hand)
    extraction = contact.get("extraction")
    if extraction is not None:
        raw = extraction.get("raw_bbox_contact_points", np.empty((0, 2)))
        pts = extraction.get("contact_points", np.empty((0, 2)))
        if len(raw):
            ax.scatter(raw[:, 0], raw[:, 1], s=2, c="#ff7eb6", alpha=0.45, label="raw")
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=4, c="#ffd33d", alpha=0.85, label="contact")
    means = contact.get("means", np.empty((0, 2)))
    if len(means):
        ax.scatter(means[:, 0], means[:, 1], s=34, c="#ff0000", marker="x", linewidths=1.7, label="GMM")
    ax.text(
        2,
        h - 4,
        f"contact_points={to_int(row.get('contact_points'))} pass={contact.get('passed')}",
        fontsize=6,
        color="white",
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.0),
    )
    annotate_common(ax, row)
    return fig_to_rgb(fig)


def render_reference_panel(row: pd.Series) -> Optional[np.ndarray]:
    ref_path = localize_output_path(row.get("reference_frame"))
    if ref_path is None or not ref_path.exists():
        sample_dir = localize_output_path(row.get("sample_dir"))
        if sample_dir is not None:
            ref_path = sample_dir / "reference_frame.png"
    img_rgb = read_rgb(ref_path) if ref_path is not None else None
    if img_rgb is None:
        frame_idx = to_int(row.get("ref_idx"), -1)
        img_rgb = read_rgb(frame_path(frame_idx)) if frame_idx >= 0 else None
    if img_rgb is None:
        return None
    fig, ax = plt.subplots(figsize=(3.2, 2.0), dpi=120)
    ax.imshow(img_rgb)
    tracked = parse_bbox(row.get("tracked_reference_object_bbox"))
    if tracked is not None:
        draw_rect(ax, tracked, "#00ffff", "tracked ref obj", 1.5)
    text = (
        f"gap={to_int(row.get('reference_gap'))} "
        f"overlap={to_float(row.get('reference_hand_overlap_ratio'), 0):.2f}\\n"
        f"{row.get('reference_selection_tier', '')}"
    )
    ax.text(
        2,
        img_rgb.shape[0] - 4,
        text,
        fontsize=6,
        color="white",
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.0),
    )
    annotate_common(ax, row)
    return fig_to_rgb(fig)


def fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.tight_layout(pad=0.15)
    fig.canvas.draw()
    arr_rgba = np.asarray(fig.canvas.buffer_rgba())
    arr = arr_rgba[:, :, :3].copy()
    plt.close(fig)
    return arr


def blank_panel(text: str, width: int = 384, height: int = 240) -> np.ndarray:
    arr = np.full((height, width, 3), 245, dtype=np.uint8)
    fig, ax = plt.subplots(figsize=(width / 120, height / 120), dpi=120)
    ax.imshow(arr)
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=8, transform=ax.transAxes)
    ax.axis("off")
    return fig_to_rgb(fig)


def save_contact_sheets(
    rows: pd.DataFrame,
    prefix: str,
    renderer,
    detections: Optional[Sequence[Any]] = None,
    per_page: int = 32,
    cols: int = 4,
) -> List[Path]:
    paths: List[Path] = []
    total = len(rows)
    pages = max(1, math.ceil(total / per_page))
    for page_idx in range(pages):
        print(f"[E12 debug] {prefix}: page {page_idx + 1}/{pages}", flush=True)
        page = rows.iloc[page_idx * per_page : (page_idx + 1) * per_page]
        panels: List[np.ndarray] = []
        for _, row in page.iterrows():
            try:
                panel = renderer(row, detections) if detections is not None else renderer(row)
            except Exception as exc:
                panel = blank_panel(f"render failed\\n{type(exc).__name__}")
            panels.append(panel if panel is not None else blank_panel("missing image"))
        if not panels:
            continue
        cell_h = max(p.shape[0] for p in panels)
        cell_w = max(p.shape[1] for p in panels)
        rows_count = math.ceil(len(panels) / cols)
        canvas = np.full((rows_count * cell_h, cols * cell_w, 3), 255, dtype=np.uint8)
        for i, panel in enumerate(panels):
            resized = cv2.resize(panel, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            r, c = divmod(i, cols)
            canvas[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = resized
        out = SUMMARY_DIR / f"{prefix}_page_{page_idx + 1:03d}.png"
        cv2.imwrite(str(out), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        paths.append(out)
    return paths


def write_funnel_table(candidate_df: pd.DataFrame, overview_df: pd.DataFrame) -> List[Tuple[str, int]]:
    overview = overview_df.iloc[0].to_dict()
    new_contact = int(candidate_df.loc[candidate_df["candidate_source"].astype(str) == "new_contact_episode", "subaction_index"].nunique())
    final_keep = int(candidate_df.loc[candidate_df["is_final_tuple"].astype(str).str.lower() == "true", "subaction_index"].nunique())
    table = [
        ("100 subactions", int(to_int(overview.get("subactions_total"), 100))),
        ("有 new-contact candidate", new_contact),
        ("Cell2 contact/GMM pass", int(to_int(overview.get("contact_gmm_pass")))),
        ("reference found", int(to_int(overview.get("reference_found")))),
        ("homography available", int(to_int(overview.get("projection_success")))),
        ("final keep", final_keep),
    ]
    with (SUMMARY_DIR / "e12_debug_funnel_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "remaining"])
        writer.writerows(table)
    md_lines = ["| 阶段 | 剩余 |", "| --- | ---: |"] + [f"| {k} | {v} |" for k, v in table]
    (SUMMARY_DIR / "e12_debug_funnel_table.md").write_text("\\n".join(md_lines) + "\\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6.0, 2.2), dpi=160)
    ax.axis("off")
    tbl = ax.table(cellText=[[k, v] for k, v in table], colLabels=["阶段", "剩余"], loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.25)
    fig.tight_layout()
    fig.savefig(SUMMARY_DIR / "e12_debug_funnel_table.png")
    plt.close(fig)
    return table


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    candidate_df = pd.read_csv(OUTPUT_ROOT / "candidate_diagnostics.csv")
    overview_df = pd.read_csv(OUTPUT_ROOT / "experiment_overview.csv")
    final_df = candidate_df[candidate_df["is_final_tuple"].astype(str).str.lower() == "true"].copy()
    final_df = final_df.sort_values(["subaction_index", "candidate_index", "frame_0_based"]).reset_index(drop=True)

    print(f"[E12 debug] final tuples: {len(final_df)}")
    print("[E12 debug] loading HOA detections")
    detections = load_detections(HOA_PKL)

    table = write_funnel_table(candidate_df, overview_df)
    save_contact_sheets(final_df, "e12_detected_objects", render_object_panel, detections)
    save_contact_sheets(final_df, "e12_detected_hands", render_hand_panel, detections)
    save_contact_sheets(final_df, "e12_contact_frame_contact_points", render_contact_panel, detections)
    save_contact_sheets(final_df, "e12_reference_frames", render_reference_panel, None)

    readme = [
        "# E12 排查汇总",
        "",
        "内容：",
        "- `e12_detected_objects_page_*.png`: contact frame 上的 object detection bbox，绿色为 HOA active object，青色虚线为 E12 记录的 active object。",
        "- `e12_detected_hands_page_*.png`: contact frame 上的 hand detection bbox，蓝色为 active hand。",
        "- `e12_contact_frame_contact_points_page_*.png`: contact frame 上的 active hand、active object、raw/contact points 和重新拟合的 5 个 GMM means。",
        "- `e12_reference_frames_page_*.png`: E12 最终选择的 reference frame，青色框为 tracked reference object bbox。",
        "- `e12_debug_funnel_table.png/csv/md`: E12 漏斗表。",
        "",
        "漏斗：",
        "",
        "| 阶段 | 剩余 |",
        "| --- | ---: |",
    ]
    readme.extend([f"| {k} | {v} |" for k, v in table])
    readme.extend(
        [
            "",
            "说明：",
            "- 本脚本只生成排查图，不重跑 E12 CoTracker 或投影。",
            "- contact point 图中的 GMM means 是按 E12 同一 Cell2 逻辑在 contact frame 上重新拟合，用于排查节点可视化。",
        ]
    )
    (SUMMARY_DIR / "README.md").write_text("\\n".join(readme) + "\\n", encoding="utf-8")
    print(f"[E12 debug] wrote {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
