"""
Contact point extraction utilities for the VRB reproduction notebook.

The original VRB paper describes weak contact labels as hand-skin boundary
points that intersect with the contacted object bounding box. In practice,
loose object boxes can include a lot of hand pixels. These helpers keep the
paper's weak-supervision structure, but add an optional visible-object mask
and boundary-distance filter so contact candidates stay close to the object
surface instead of drifting onto the hand interior.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class ContactExtractionConfig:
    hand_score_threshold: float = 0.5
    object_score_threshold: float = 0.5
    lower_skin_ycrcb: Tuple[int, int, int] = (0, 133, 77)
    upper_skin_ycrcb: Tuple[int, int, int] = (255, 173, 127)
    use_object_mask: bool = True
    use_object_boundary: bool = True
    boundary_distance_px: float = 8.0
    object_bbox_shrink_px: int = 2
    grabcut_iterations: int = 3
    grabcut_bg_margin_px: int = 6
    hand_exclusion_dilate_px: int = 3
    project_points_to_object_boundary: bool = True
    rng_seed: Optional[int] = 42


def normalized_bbox_to_pixels(
    bbox: Sequence[float],
    width: int,
    height: int,
    clip: bool = True,
) -> Tuple[int, int, int, int]:
    x1 = int(round(float(bbox[0]) * width))
    y1 = int(round(float(bbox[1]) * height))
    x2 = int(round(float(bbox[2]) * width))
    y2 = int(round(float(bbox[3]) * height))
    if clip:
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def hand_matches_side(hand: Any, active_hand: str) -> bool:
    return hand.side.name.lower() == active_hand.lower()


def get_active_hand_bbox(
    frame_det: Any,
    active_hand: str,
    score_threshold: float = 0.5,
) -> Optional[List[float]]:
    for hand in frame_det.hands:
        if hand.score >= score_threshold and hand_matches_side(hand, active_hand):
            return [hand.bbox.left, hand.bbox.top, hand.bbox.right, hand.bbox.bottom]
    return None


def get_valid_object_bboxes(
    frame_det: Any,
    score_threshold: float = 0.5,
) -> List[List[float]]:
    bboxes = []
    for obj in frame_det.objects:
        if obj.score >= score_threshold:
            bboxes.append([obj.bbox.left, obj.bbox.top, obj.bbox.right, obj.bbox.bottom])
    return bboxes


def select_active_object_bbox(
    hand_bbox: Sequence[float],
    object_bboxes: Sequence[Sequence[float]],
) -> Optional[List[float]]:
    """
    Select the object box most plausibly associated with the active hand.

    The HOA annotations expose hand.object_offset but not a direct object id in
    this package. For multiple objects, prefer the box with the largest
    hand/object intersection, then shortest center distance.
    """
    if not object_bboxes:
        return None

    hx1, hy1, hx2, hy2 = hand_bbox
    hc = np.array([(hx1 + hx2) * 0.5, (hy1 + hy2) * 0.5], dtype=np.float32)

    best_bbox = None
    best_key = None
    for bbox in object_bboxes:
        ox1, oy1, ox2, oy2 = bbox
        ix1 = max(hx1, ox1)
        iy1 = max(hy1, oy1)
        ix2 = min(hx2, ox2)
        iy2 = min(hy2, oy2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        oc = np.array([(ox1 + ox2) * 0.5, (oy1 + oy2) * 0.5], dtype=np.float32)
        center_dist = float(np.linalg.norm(hc - oc))
        area = max(1e-6, (ox2 - ox1) * (oy2 - oy1))
        key = (intersection, -center_dist, -area)
        if best_key is None or key > best_key:
            best_key = key
            best_bbox = list(bbox)

    return best_bbox


def build_hand_skin_edge_mask(
    img_bgr: np.ndarray,
    hand_bbox: Sequence[float],
    config: ContactExtractionConfig = ContactExtractionConfig(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    h, w = img_bgr.shape[:2]
    x1_h, y1_h, x2_h, y2_h = normalized_bbox_to_pixels(hand_bbox, w, h)

    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array(config.lower_skin_ycrcb, dtype=np.uint8)
    upper_skin = np.array(config.upper_skin_ycrcb, dtype=np.uint8)
    skin_mask = cv2.inRange(img_ycrcb, lower_skin, upper_skin)

    hand_skin_mask = np.zeros_like(skin_mask)
    hand_skin_mask[y1_h:y2_h, x1_h:x2_h] = skin_mask[y1_h:y2_h, x1_h:x2_h]
    skin_count = int(np.sum(hand_skin_mask > 0))

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(hand_skin_mask, kernel, iterations=1)
    hand_edge = cv2.subtract(hand_skin_mask, eroded)
    edge_count = int(np.sum(hand_edge > 0))

    return skin_mask, hand_skin_mask, hand_edge, skin_count, edge_count


def _safe_rect_from_bbox(
    bbox_px: Sequence[int],
    width: int,
    height: int,
    shrink_px: int = 0,
) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bbox_px
    x1 += shrink_px
    y1 += shrink_px
    x2 -= shrink_px
    y2 -= shrink_px
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return None
    return x1, y1, x2 - x1, y2 - y1


def build_object_visible_mask(
    img_bgr: np.ndarray,
    object_bbox: Sequence[float],
    hand_skin_mask: Optional[np.ndarray] = None,
    config: ContactExtractionConfig = ContactExtractionConfig(),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate visible object region from its bbox using GrabCut.

    This is intentionally heuristic: the dataset provides boxes, not masks.
    The hand skin mask is marked as probable background to avoid labeling the
    visible hand as object.
    """
    h, w = img_bgr.shape[:2]
    object_px = normalized_bbox_to_pixels(object_bbox, w, h)
    object_mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = object_px

    if not config.use_object_mask:
        object_mask[y1:y2, x1:x2] = 255
        return object_mask, object_mask.copy()

    gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    margin = int(config.grabcut_bg_margin_px)
    rx1 = max(0, x1 - margin)
    ry1 = max(0, y1 - margin)
    rx2 = min(w, x2 + margin)
    ry2 = min(h, y2 + margin)
    gc_mask[ry1:ry2, rx1:rx2] = cv2.GC_PR_BGD
    gc_mask[y1:y2, x1:x2] = cv2.GC_PR_FGD

    rect = _safe_rect_from_bbox(
        object_px,
        w,
        h,
        shrink_px=int(config.object_bbox_shrink_px),
    )
    if rect is None:
        object_mask[y1:y2, x1:x2] = 255
        return object_mask, object_mask.copy()

    if hand_skin_mask is not None and config.hand_exclusion_dilate_px > 0:
        k = 2 * int(config.hand_exclusion_dilate_px) + 1
        hand_kernel = np.ones((k, k), np.uint8)
        hand_bg = cv2.dilate((hand_skin_mask > 0).astype(np.uint8), hand_kernel, iterations=1)
        gc_mask[hand_bg > 0] = cv2.GC_PR_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        if config.rng_seed is not None:
            cv2.setRNGSeed(int(config.rng_seed))
        cv2.grabCut(
            img_bgr,
            gc_mask,
            rect,
            bgd_model,
            fgd_model,
            int(config.grabcut_iterations),
            cv2.GC_INIT_WITH_MASK,
        )
        object_mask = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)
    except cv2.error:
        object_mask[y1:y2, x1:x2] = 255

    # Keep the estimate within the detection box. We want visible object pixels,
    # not unrelated foreground from the surrounding image.
    bbox_mask = np.zeros((h, w), dtype=np.uint8)
    bbox_mask[y1:y2, x1:x2] = 255
    object_mask = cv2.bitwise_and(object_mask, bbox_mask)
    if int(np.sum(object_mask > 0)) < 20:
        object_mask = bbox_mask

    return object_mask, bbox_mask


def build_object_boundary_band(
    object_mask: np.ndarray,
    radius_px: float,
) -> np.ndarray:
    radius = max(1, int(round(radius_px)))
    object_binary = (object_mask > 0).astype(np.uint8)
    eroded = cv2.erode(object_binary, np.ones((3, 3), np.uint8), iterations=1)
    boundary = cv2.subtract(object_binary, eroded)
    if radius <= 1:
        return (boundary > 0).astype(np.uint8) * 255
    k = 2 * radius + 1
    band = cv2.dilate(boundary, np.ones((k, k), np.uint8), iterations=1)
    return (band > 0).astype(np.uint8) * 255


def build_object_boundary_mask(object_mask: np.ndarray) -> np.ndarray:
    object_binary = (object_mask > 0).astype(np.uint8)
    eroded = cv2.erode(object_binary, np.ones((3, 3), np.uint8), iterations=1)
    boundary = cv2.subtract(object_binary, eroded)
    return (boundary > 0).astype(np.uint8) * 255


def project_points_to_boundary(
    points_xy: np.ndarray,
    boundary_mask: np.ndarray,
) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if len(points_xy) == 0:
        return points_xy

    boundary_yx = np.argwhere(boundary_mask > 0)
    if len(boundary_yx) == 0:
        return points_xy

    boundary_xy = boundary_yx[:, ::-1].astype(np.float32)
    projected = np.zeros_like(points_xy, dtype=np.float32)
    for i, pt in enumerate(points_xy):
        dist2 = np.sum((boundary_xy - pt[None, :]) ** 2, axis=1)
        projected[i] = boundary_xy[int(np.argmin(dist2))]
    return projected


def extract_contact_points(
    img_bgr: np.ndarray,
    hand_bbox: Sequence[float],
    object_bboxes: Sequence[Sequence[float]],
    config: ContactExtractionConfig = ContactExtractionConfig(),
) -> Dict[str, Any]:
    h, w = img_bgr.shape[:2]
    hand_px = normalized_bbox_to_pixels(hand_bbox, w, h)
    active_object_bbox = select_active_object_bbox(hand_bbox, object_bboxes)

    result: Dict[str, Any] = {
        "status": "missing_object_bbox" if active_object_bbox is None else "ok",
        "hand_bbox": list(hand_bbox),
        "active_object_bbox": active_object_bbox,
        "hand_bbox_px": hand_px,
        "active_object_bbox_px": None,
        "object_bboxes": [list(b) for b in object_bboxes],
        "skin_pixels": 0,
        "edge_pixels": 0,
        "raw_bbox_contact_points": np.empty((0, 2), dtype=np.float32),
        "candidate_hand_edge_points": np.empty((0, 2), dtype=np.float32),
        "contact_points": np.empty((0, 2), dtype=np.float32),
        "object_mask": np.zeros((h, w), dtype=np.uint8),
        "object_bbox_mask": np.zeros((h, w), dtype=np.uint8),
        "object_boundary_mask": np.zeros((h, w), dtype=np.uint8),
        "object_boundary_band": np.zeros((h, w), dtype=np.uint8),
        "hand_skin_mask": np.zeros((h, w), dtype=np.uint8),
        "hand_edge_mask": np.zeros((h, w), dtype=np.uint8),
        "filter_mode": "object_mask_boundary",
        "boundary_distance_px": float(config.boundary_distance_px),
    }

    if active_object_bbox is None:
        return result

    object_px = normalized_bbox_to_pixels(active_object_bbox, w, h)
    result["active_object_bbox_px"] = object_px
    x1_h, y1_h, x2_h, y2_h = hand_px
    x1_o, y1_o, x2_o, y2_o = object_px

    _, hand_skin_mask, hand_edge, skin_count, edge_count = build_hand_skin_edge_mask(
        img_bgr,
        hand_bbox,
        config,
    )
    result["skin_pixels"] = skin_count
    result["edge_pixels"] = edge_count
    result["hand_skin_mask"] = hand_skin_mask
    result["hand_edge_mask"] = hand_edge

    object_mask, bbox_mask = build_object_visible_mask(
        img_bgr,
        active_object_bbox,
        hand_skin_mask=hand_skin_mask,
        config=config,
    )
    boundary_mask = build_object_boundary_mask(object_mask)
    boundary_band = build_object_boundary_band(object_mask, config.boundary_distance_px)
    result["object_mask"] = object_mask
    result["object_bbox_mask"] = bbox_mask
    result["object_boundary_mask"] = boundary_mask
    result["object_boundary_band"] = boundary_band

    ix1 = max(x1_h, x1_o)
    iy1 = max(y1_h, y1_o)
    ix2 = min(x2_h, x2_o)
    iy2 = min(y2_h, y2_o)
    result["bbox_intersection_px"] = (ix1, iy1, ix2, iy2) if ix2 > ix1 and iy2 > iy1 else None

    edge_points_yx = np.argwhere(hand_edge > 0)
    raw_points: List[List[float]] = []
    candidate_points: List[List[float]] = []
    for y, x in edge_points_yx:
        if not (x1_o <= x <= x2_o and y1_o <= y <= y2_o):
            continue
        raw_points.append([float(x), float(y)])

        # Contact candidates are still hand-edge pixels. Do not require the
        # candidate itself to be inside the object mask; require it to be close
        # to the visible object boundary estimated from that mask.
        if config.use_object_boundary and boundary_band[y, x] == 0:
            continue
        if not config.use_object_boundary and config.use_object_mask and object_mask[y, x] == 0:
            continue
        candidate_points.append([float(x), float(y)])

    result["raw_bbox_contact_points"] = np.asarray(raw_points, dtype=np.float32).reshape(-1, 2)
    candidate_points_arr = np.asarray(candidate_points, dtype=np.float32).reshape(-1, 2)
    result["candidate_hand_edge_points"] = candidate_points_arr
    if config.project_points_to_object_boundary:
        result["contact_points"] = project_points_to_boundary(candidate_points_arr, boundary_mask)
    else:
        result["contact_points"] = candidate_points_arr
    return result
