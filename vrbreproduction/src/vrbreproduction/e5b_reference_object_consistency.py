"""Reference-frame object consistency checks for the E5b experiment."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


ARTICULATED_OBJECT_TOKENS = {
    "door",
    "drawer",
    "freezer",
    "fridge",
    "refrigerator",
    "cupboard",
    "cabinet",
}


def bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def polygon_to_bbox(polygon: Optional[np.ndarray], image_shape: Tuple[int, int]) -> Optional[Tuple[float, float, float, float]]:
    if polygon is None:
        return None
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0 or not np.isfinite(pts).all():
        return None
    h, w = image_shape[:2]
    x1 = float(np.clip(np.min(pts[:, 0]), 0, w))
    y1 = float(np.clip(np.min(pts[:, 1]), 0, h))
    x2 = float(np.clip(np.max(pts[:, 0]), 0, w))
    y2 = float(np.clip(np.max(pts[:, 1]), 0, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def point_to_bbox_distance(point_xy: Sequence[float], bbox: Sequence[float]) -> float:
    x, y = float(point_xy[0]), float(point_xy[1])
    x1, y1, x2, y2 = [float(v) for v in bbox]
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float((dx * dx + dy * dy) ** 0.5)


def count_points_near_bbox(points_xy: np.ndarray, bbox: Sequence[float], tolerance_px: float) -> int:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    return int(sum(point_to_bbox_distance(pt, bbox) <= float(tolerance_px) for pt in points))


def bbox_center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ac = np.array([(ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5], dtype=np.float32)
    bc = np.array([(bx1 + bx2) * 0.5, (by1 + by2) * 0.5], dtype=np.float32)
    return float(np.linalg.norm(ac - bc))


def reference_object_bboxes(frame_det: Any, image_shape: Tuple[int, int], score_threshold: float = 0.5):
    h, w = image_shape[:2]
    boxes = []
    if not hasattr(frame_det, "objects"):
        return boxes
    for obj in frame_det.objects:
        if obj.score < score_threshold:
            continue
        bbox = (
            float(obj.bbox.left) * w,
            float(obj.bbox.top) * h,
            float(obj.bbox.right) * w,
            float(obj.bbox.bottom) * h,
        )
        if bbox_area(bbox) <= 0:
            continue
        boxes.append({"bbox": bbox, "score": float(obj.score)})
    return boxes


def is_articulated_object_candidate(noun: Any, all_nouns: Any = None, narration: Any = None) -> bool:
    text_parts = [str(noun or ""), str(all_nouns or ""), str(narration or "")]
    text = " ".join(text_parts).lower()
    return any(token in text for token in ARTICULATED_OBJECT_TOKENS)


def evaluate_reference_object_consistency(
    *,
    detections: Sequence[Any],
    ref_idx: int,
    image_shape: Tuple[int, int],
    projected_object_polygon: Optional[np.ndarray],
    projected_contact_means: np.ndarray,
    projected_contact_centroid: Sequence[float],
    noun: Any = None,
    all_nouns: Any = None,
    narration: Any = None,
    iou_min: float = 0.10,
    contact_bbox_tolerance_px: float = 16.0,
    center_distance_max_px: float = 80.0,
    min_contact_means_near_bbox: int = 4,
    object_score_threshold: float = 0.5,
) -> Dict[str, Any]:
    projected_bbox = polygon_to_bbox(projected_object_polygon, image_shape)
    projected_area = float(cv2.contourArea(np.asarray(projected_object_polygon, dtype=np.float32))) if projected_object_polygon is not None else 0.0
    articulated = is_articulated_object_candidate(noun, all_nouns, narration)
    base = {
        "reference_object_consistency_pass": False,
        "reference_object_consistency_reason": None,
        "reference_object_bbox": None,
        "reference_object_bbox_score": None,
        "reference_object_iou": None,
        "reference_object_center_distance": None,
        "projected_object_bbox": tuple(float(v) for v in projected_bbox) if projected_bbox is not None else None,
        "projected_object_area": projected_area,
        "projected_contact_to_reference_object_distance": None,
        "projected_contact_means_inside_reference_object_count": 0,
        "reference_object_candidate_count": 0,
        "is_articulated_object_candidate": bool(articulated),
        "articulated_object_policy": "same_thresholds_record_only" if articulated else "not_articulated",
    }
    if projected_bbox is None:
        base["reference_object_consistency_reason"] = "missing_projected_object_bbox"
        return base

    ref_objects = reference_object_bboxes(detections[int(ref_idx)], image_shape, score_threshold=object_score_threshold)
    base["reference_object_candidate_count"] = int(len(ref_objects))
    if not ref_objects:
        base["reference_object_consistency_reason"] = "no_reference_object_bbox"
        return base

    best = None
    best_key = None
    for item in ref_objects:
        ref_bbox = item["bbox"]
        iou = bbox_iou(projected_bbox, ref_bbox)
        center_distance = bbox_center_distance(projected_bbox, ref_bbox)
        contact_distance = point_to_bbox_distance(projected_contact_centroid, ref_bbox)
        inside_count = count_points_near_bbox(
            np.asarray(projected_contact_means, dtype=np.float32),
            ref_bbox,
            tolerance_px=contact_bbox_tolerance_px,
        )
        key = (
            iou,
            inside_count,
            -contact_distance,
            -center_distance,
            item["score"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "bbox": ref_bbox,
                "score": item["score"],
                "iou": iou,
                "center_distance": center_distance,
                "contact_distance": contact_distance,
                "inside_count": inside_count,
            }

    assert best is not None
    base.update(
        {
            "reference_object_bbox": tuple(float(v) for v in best["bbox"]),
            "reference_object_bbox_score": float(best["score"]),
            "reference_object_iou": float(best["iou"]),
            "reference_object_center_distance": float(best["center_distance"]),
            "projected_contact_to_reference_object_distance": float(best["contact_distance"]),
            "projected_contact_means_inside_reference_object_count": int(best["inside_count"]),
        }
    )

    iou_or_contacts_ok = best["iou"] >= iou_min or best["inside_count"] >= min_contact_means_near_bbox
    contact_ok = best["contact_distance"] <= contact_bbox_tolerance_px
    center_ok = best["center_distance"] <= center_distance_max_px
    if iou_or_contacts_ok and contact_ok and center_ok:
        base["reference_object_consistency_pass"] = True
        base["reference_object_consistency_reason"] = "matched_reference_object_bbox"
    elif not iou_or_contacts_ok:
        base["reference_object_consistency_reason"] = "reference_object_iou_and_contact_count_too_low"
    elif not contact_ok:
        base["reference_object_consistency_reason"] = "projected_contact_too_far_from_reference_object"
    else:
        base["reference_object_consistency_reason"] = "projected_object_center_too_far_from_reference_object"
    return base
