from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from epic_kitchens.hoa import load_detections
from sklearn.mixture import GaussianMixture

from vrbreproduction.contact_point_utils import (
    ContactExtractionConfig,
    extract_contact_points,
    get_active_hand_bbox,
    get_valid_object_bboxes,
    normalized_bbox_to_pixels,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "e17"
SUMMARY_DIR = OUTPUT_ROOT / "charts" / "排查汇总"
IMAGE_DIR = ROOT / "data" / "P01_109_frames"
HOA_PKL = ROOT / "data" / "P01_109.pkl"


def draw_label(img: np.ndarray, text: str, y: int = 18) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], min(58, img.shape[0])), (0, 0, 0), -1)
    for idx, line in enumerate(text.split("\n")[:3]):
        cv2.putText(
            img,
            line[:72],
            (6, y + idx * 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def bbox_px_from_norm(bbox: Sequence[float], image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    h, w = image_shape[:2]
    return normalized_bbox_to_pixels(bbox, w, h)


def draw_bbox(img: np.ndarray, bbox_px: Sequence[int], color: Tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = [int(v) for v in bbox_px]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img,
        label,
        (x1, max(14, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def load_frame(frame_idx: int) -> Optional[np.ndarray]:
    path = IMAGE_DIR / f"frame_{int(frame_idx) + 1:010d}.jpg"
    return cv2.imread(str(path))


def fit_gmm_means(contact_points: np.ndarray, n_components: int = 5) -> Optional[np.ndarray]:
    if len(contact_points) < n_components:
        return None
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(contact_points)
    return np.asarray(gmm.means_, dtype=np.float32)


def render_tile(
    row: pd.Series,
    detections: Any,
    mode: str,
    contact_config: ContactExtractionConfig,
    tile_size: Tuple[int, int] = (360, 260),
) -> np.ndarray:
    tile_w, tile_h = tile_size
    canvas = np.full((tile_h, tile_w, 3), 245, dtype=np.uint8)
    sub_idx = int(row["subaction_index"])
    narration_id = str(row.get("narration_id", ""))
    narration = str(row.get("narration", ""))
    frame_value = row.get("frame_0_based")
    hand = row.get("hand")
    status = str(row.get("status", ""))
    fail_reason = str(row.get("fail_reason", ""))

    if pd.isna(frame_value) or pd.isna(hand):
        draw_label(canvas, f"s{sub_idx} {narration_id}\nno candidate\n{narration}")
        return canvas

    frame_idx = int(frame_value)
    hand = str(hand)
    img = load_frame(frame_idx)
    if img is None:
        draw_label(canvas, f"s{sub_idx} {narration_id}\nmissing frame {frame_idx}\n{narration}")
        return canvas

    frame_det = detections[frame_idx]
    active_hand_bbox = get_active_hand_bbox(frame_det, hand, score_threshold=0.5)
    object_bboxes = get_valid_object_bboxes(frame_det, score_threshold=0.5)
    extraction: Dict[str, Any] = {}
    if active_hand_bbox is not None and object_bboxes:
        extraction = extract_contact_points(img, active_hand_bbox, object_bboxes, config=contact_config)

    view = img.copy()
    if mode == "objects":
        for idx, bbox in enumerate(object_bboxes):
            color = (0, 190, 255)
            label = f"obj{idx + 1}"
            if extraction.get("active_object_bbox") is not None and list(bbox) == list(extraction["active_object_bbox"]):
                color = (0, 255, 0)
                label = "active obj"
            draw_bbox(view, bbox_px_from_norm(bbox, img.shape), color, label)
    elif mode == "hands":
        if hasattr(frame_det, "hands"):
            for hand_det in frame_det.hands:
                if hand_det.score < 0.5:
                    continue
                side = hand_det.side.name.lower()
                color = (255, 0, 0) if side == str(hand).lower() else (180, 80, 255)
                bbox = [hand_det.bbox.left, hand_det.bbox.top, hand_det.bbox.right, hand_det.bbox.bottom]
                draw_bbox(view, bbox_px_from_norm(bbox, img.shape), color, f"{side} {hand_det.score:.2f}")
    elif mode == "contact":
        for bbox in object_bboxes:
            draw_bbox(view, bbox_px_from_norm(bbox, img.shape), (0, 190, 255), "obj")
        if active_hand_bbox is not None:
            draw_bbox(view, bbox_px_from_norm(active_hand_bbox, img.shape), (255, 0, 0), f"{hand} hand")
        active_obj = extraction.get("active_object_bbox")
        if active_obj is not None:
            draw_bbox(view, bbox_px_from_norm(active_obj, img.shape), (0, 255, 0), "active obj")
        raw = np.asarray(extraction.get("raw_bbox_contact_points", np.empty((0, 2))), dtype=np.float32)
        pts = np.asarray(extraction.get("contact_points", np.empty((0, 2))), dtype=np.float32)
        if len(raw) > 0:
            sample = raw[:: max(1, len(raw) // 120)]
            for x, y in sample:
                cv2.circle(view, (int(round(x)), int(round(y))), 1, (80, 80, 255), -1)
        if len(pts) > 0:
            sample = pts[:: max(1, len(pts) // 160)]
            for x, y in sample:
                cv2.circle(view, (int(round(x)), int(round(y))), 2, (0, 255, 255), -1)
            means = fit_gmm_means(pts)
            if means is not None:
                for x, y in means:
                    cv2.circle(view, (int(round(x)), int(round(y))), 6, (0, 0, 255), -1)
                    cv2.circle(view, (int(round(x)), int(round(y))), 8, (255, 255, 255), 1)
    elif mode == "reference":
        ref_value = row.get("ref_idx")
        if pd.isna(ref_value):
            draw_label(canvas, f"s{sub_idx} {narration_id}\nno reference\n{fail_reason[:48]}")
            return canvas
        ref_idx = int(ref_value)
        ref_img = load_frame(ref_idx)
        if ref_img is None:
            draw_label(canvas, f"s{sub_idx} {narration_id}\nmissing ref frame {ref_idx}\n{fail_reason[:48]}")
            return canvas
        view = ref_img.copy()
        if hasattr(detections[ref_idx], "hands"):
            for hand_det in detections[ref_idx].hands:
                if hand_det.score < 0.5:
                    continue
                side = hand_det.side.name.lower()
                color = (255, 0, 0) if side == str(hand).lower() else (180, 80, 255)
                bbox = [hand_det.bbox.left, hand_det.bbox.top, hand_det.bbox.right, hand_det.bbox.bottom]
                draw_bbox(view, bbox_px_from_norm(bbox, ref_img.shape), color, f"{side} {hand_det.score:.2f}")
        for idx, bbox in enumerate(get_valid_object_bboxes(detections[ref_idx], score_threshold=0.5)):
            draw_bbox(view, bbox_px_from_norm(bbox, ref_img.shape), (0, 190, 255), f"obj{idx + 1}")
        view = cv2.resize(view, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        ref_gap = row.get("ref_gap")
        anchor_gap = row.get("reference_anchor_gap")
        header = (
            f"s{sub_idx} {narration_id} ref{ref_idx} cf{frame_idx}\n"
            f"off {row.get('reference_offset')} gap {ref_gap} anchor {anchor_gap}\n"
            f"{str(row.get('reference_mode', ''))[:28]} {fail_reason[:18]}"
        )
        draw_label(view, header)
        return view

    view = cv2.resize(view, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    header = (
        f"s{sub_idx} {narration_id} f{frame_idx} {hand}\n"
        f"{narration[:44]}\n"
        f"{status} {fail_reason[:42]}"
    )
    draw_label(view, header)
    return view


def save_contact_sheet(
    rows: pd.DataFrame,
    detections: Any,
    mode: str,
    out_prefix: Path,
    contact_config: ContactExtractionConfig,
    per_page: int = 20,
    cols: int = 4,
) -> List[Path]:
    out_paths: List[Path] = []
    rows = rows.sort_values("subaction_index").reset_index(drop=True)
    tile_w, tile_h = 360, 260
    for page_idx, start in enumerate(range(0, len(rows), per_page), start=1):
        page = rows.iloc[start : start + per_page]
        page_rows = int(math.ceil(len(page) / cols))
        sheet = np.full((page_rows * tile_h, cols * tile_w, 3), 245, dtype=np.uint8)
        for local_idx, (_, row) in enumerate(page.iterrows()):
            r = local_idx // cols
            c = local_idx % cols
            tile = render_tile(row, detections, mode, contact_config, tile_size=(tile_w, tile_h))
            sheet[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tile
        out_path = out_prefix.parent / f"{out_prefix.name}_page_{page_idx:03d}.png"
        cv2.imwrite(str(out_path), sheet)
        out_paths.append(out_path)
    return out_paths


def save_funnel_table(overview: pd.Series, out_path: Path) -> None:
    stages = [
        ("100 subactions", int(overview["subactions_total"])),
        ("有 new-contact candidate", int(overview["episode_filtered_candidate_total"])),
        ("Cell2 contact/GMM pass", int(overview["cell2_pass"])),
        ("near-contact ref found", int(overview["near_contact_ref_found"])),
        ("homography available", int(overview["homography_available"])),
        ("final keep", int(overview["heatmap_pass"])),
    ]
    table_df = pd.DataFrame(stages, columns=["stage", "remaining"])
    table_df.to_csv(out_path.with_suffix(".csv"), index=False)
    md_lines = [
        "| stage | remaining |",
        "| --- | ---: |",
    ]
    for _, row in table_df.iterrows():
        md_lines.append(f"| {row['stage']} | {int(row['remaining'])} |")
    out_path.with_suffix(".md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=["阶段", "剩余"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    ax.set_title("E17 排查漏斗", fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    candidate_df = pd.read_csv(OUTPUT_ROOT / "candidate_diagnostics.csv")
    overview = pd.read_csv(OUTPUT_ROOT / "experiment_overview.csv").iloc[0]
    detections = load_detections(str(HOA_PKL))
    contact_config = ContactExtractionConfig(
        use_object_mask=True,
        use_object_boundary=True,
        boundary_distance_px=6.0,
        project_points_to_object_boundary=True,
    )

    save_funnel_table(overview, SUMMARY_DIR / "e17_debug_funnel_table.png")
    save_contact_sheet(candidate_df, detections, "objects", SUMMARY_DIR / "e17_detected_objects", contact_config)
    save_contact_sheet(candidate_df, detections, "hands", SUMMARY_DIR / "e17_detected_hands", contact_config)
    save_contact_sheet(candidate_df, detections, "contact", SUMMARY_DIR / "e17_contact_frame_contact_points", contact_config)
    save_contact_sheet(candidate_df, detections, "reference", SUMMARY_DIR / "e17_reference_frames", contact_config)

    readme = SUMMARY_DIR / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# E17 排查汇总",
                "",
                "内容：",
                "- `e17_detected_objects_page_*.png`: contact frame 上的 object detection bbox，绿色为 active object。",
                "- `e17_detected_hands_page_*.png`: contact frame 上的 hand detection bbox，蓝色为 active hand。",
                "- `e17_contact_frame_contact_points_page_*.png`: contact frame 上的 active hand、active object、raw/contact points 和 5 个 GMM means。",
                "- `e17_reference_frames_page_*.png`: E17 找到的 reference frame；没有 ref 的样本显示 no reference。",
                "- `e17_debug_funnel_table.png/csv/md`: E17 漏斗表。",
                "",
                "图例：",
                "- object 图：绿色 active object，黄色其它 object。",
                "- hand 图：蓝色 active hand，紫色其它 hand。",
                "- contact 图：黄色小点为 filtered contact points，红色大点为 Cell2 GMM means，粉色小点为 raw bbox-intersection hand-edge points。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote debug summary to {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
