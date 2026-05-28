"""E13 experiment: engineered short-window homography projection."""

from __future__ import annotations

import ast
import io
import json
import os
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

VRBREPRODUCTION_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLCONFIGDIR", str(VRBREPRODUCTION_ROOT / ".mplconfig"))

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from epic_kitchens.hoa import load_detections
from epic_kitchens.hoa.types import HandState
from scipy.signal import savgol_filter

from .contact_point_utils import (
    ContactExtractionConfig,
    build_object_visible_mask,
    get_active_hand_bbox,
    get_valid_object_bboxes,
    normalized_bbox_to_pixels,
    select_active_object_bbox,
)
from .e5b_reference_object_consistency import (
    bbox_center_distance,
    bbox_iou,
    polygon_to_bbox,
    reference_object_bboxes,
)
from .label_heatmap_utils import (
    build_label_heatmaps,
    draw_vrb_style_affordance_overlay,
    merge_label_heatmaps,
    save_label_heatmap_outputs,
    transform_covariances_by_homography,
)
from .pipeline_retention import _json_safe, fit_cell2_contact_gmm, run_cell4_heatmap_gate
from .problem3_utils import (
    compute_centered_crop_bbox,
    compute_crop_hand_overlap_ratio,
    count_points_near_polygon,
    draw_problem3_crop_overlay,
    draw_problem3_full_overlay,
    polygon_area,
    transform_bbox_to_polygon,
    transform_points,
)


ARTICULATED_OBJECT_TOKENS = (
    "door",
    "drawer",
    "freezer",
    "fridge",
    "refrigerator",
    "cupboard",
    "cabinet",
)


@dataclass(frozen=True)
class E13Config:
    video_id: str = "P01_109"
    num_subactions: int = int(os.environ.get("E13_NUM_SUBACTIONS", "100"))
    max_ref_backtrack: int = int(os.environ.get("E13_MAX_REF_BACKTRACK", "60"))
    early_candidate_offsets: Tuple[int, ...] = (0, 1, 2)
    crop_size: int = 150
    crop_hand_overlap_max: float = 0.05
    hand_score_threshold: float = float(os.environ.get("E13_HAND_SCORE_THRESHOLD", "0.5"))
    object_score_threshold: float = float(os.environ.get("E13_OBJECT_SCORE_THRESHOLD", "0.5"))
    homography_ratio_test: float = float(os.environ.get("E13_HOMOGRAPHY_RATIO_TEST", "0.7"))
    homography_ransac_reproj_threshold: float = float(os.environ.get("E13_HOMOGRAPHY_RANSAC_REPROJ_THRESHOLD", "4.0"))
    homography_min_good_matches: int = int(os.environ.get("E13_HOMOGRAPHY_MIN_GOOD_MATCHES", "12"))
    homography_min_inliers: int = int(os.environ.get("E13_HOMOGRAPHY_MIN_INLIERS", "8"))
    homography_min_inlier_ratio: float = float(os.environ.get("E13_HOMOGRAPHY_MIN_INLIER_RATIO", "0.25"))
    dynamic_bbox_expand_ratio: float = float(os.environ.get("E13_DYNAMIC_BBOX_EXPAND_RATIO", "0.15"))
    snap_max_distance_px: float = float(os.environ.get("E13_SNAP_MAX_DISTANCE_PX", "20.0"))
    output_root: Path = Path(
        os.environ.get(
            "E13_OUTPUT_ROOT",
            str(VRBREPRODUCTION_ROOT / "outputs" / "e13_engineered_homography_100subaction"),
        )
    )
    annotation_csv: Path = VRBREPRODUCTION_ROOT / "data" / "annotations" / "epic-kitchens-100-annotations" / "EPIC_100_train.csv"
    hoa_pkl: Path = VRBREPRODUCTION_ROOT / "data" / "P01_109.pkl"
    image_dir: Path = VRBREPRODUCTION_ROOT / "data" / "P01_109_frames"


EXPERIMENT = {
    "experiment_id": "E13",
    "name": "E13_engineered_homography_projection",
    "contact_strategy": "new_contact_episode_early_frames",
    "reference_strategy": "short_window_pre_contact_nearest_no_contact",
    "reference_fallback": "none",
    "description": (
        "使用 smoothed contact 的 0->1 new-contact episode 作为样本单位；"
        "reference 只从 contact 前短窗口 pre-contact gap 里选最近帧；"
        "pairwise H 使用 SIFT/AKAZE/ORB + Lowe ratio + RANSAC，并 mask hands/portable objects；"
        "contact GMM 投影到 reference 后必须通过 reference object mask snapping gate。"
    ),
}


def silent_call(func, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def smooth_contact_binary(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) < 7:
        return values.astype(int)
    return (savgol_filter(values, window_length=7, polyorder=2) > 0.75).astype(int)


def build_contact_arrays(detections) -> Tuple[np.ndarray, np.ndarray]:
    left = np.zeros(len(detections), dtype=np.float32)
    right = np.zeros(len(detections), dtype=np.float32)
    for frame_idx, frame_det in enumerate(detections):
        if not hasattr(frame_det, "hands"):
            continue
        for hand in frame_det.hands:
            if hand.score <= 0.5:
                continue
            if hand.state not in [HandState.PORTABLE_OBJECT, HandState.STATIONARY_OBJECT]:
                continue
            side = hand.side.name.lower()
            if side == "left":
                left[frame_idx] = 1.0
            elif side == "right":
                right[frame_idx] = 1.0
    return smooth_contact_binary(left), smooth_contact_binary(right)


def get_subactions(config: E13Config) -> pd.DataFrame:
    df = pd.read_csv(config.annotation_csv)
    subactions = df[df["video_id"] == config.video_id].copy()
    subactions = subactions.sort_values(["start_frame", "stop_frame", "narration_id"]).head(config.num_subactions)
    return subactions.reset_index(drop=True)


def compute_subaction_frame_coverage(subactions: pd.DataFrame) -> Dict[str, int]:
    total_annotated_frames = int((subactions["stop_frame"] - subactions["start_frame"] + 1).sum())
    covered = set()
    for _, item in subactions.iterrows():
        start_0 = max(0, int(item["start_frame"]) - 1)
        stop_0 = int(item["stop_frame"]) - 1
        if start_0 <= stop_0:
            covered.update(range(start_0, stop_0 + 1))
    return {
        "total_annotated_frames": total_annotated_frames,
        "unique_covered_frames": int(len(covered)),
    }


def subaction_bounds(row: pd.Series, num_frames: int) -> Tuple[int, int]:
    start_0 = max(0, int(row["start_frame"]) - 1)
    stop_0 = min(num_frames - 1, int(row["stop_frame"]) - 1)
    return start_0, stop_0


def parse_all_nouns_field(value: Any) -> List[Any]:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = [text]
    if isinstance(parsed, (list, tuple, set)):
        return list(parsed)
    return [parsed]


def is_articulated_object_candidate(noun: Any, all_nouns: Any = None, narration: Any = None) -> bool:
    text = " ".join([str(noun or ""), str(all_nouns or ""), str(narration or "")]).lower()
    return any(token in text for token in ARTICULATED_OBJECT_TOKENS)


def contact_frame_candidates_for_subaction(
    row: pd.Series,
    left_binary: np.ndarray,
    right_binary: np.ndarray,
    num_frames: int,
) -> List[Dict[str, Any]]:
    start_0, stop_0 = subaction_bounds(row, num_frames)
    if start_0 > stop_0:
        return []
    candidates: List[Dict[str, Any]] = []
    for frame_idx in range(start_0, stop_0 + 1):
        if left_binary[frame_idx] == 1:
            candidates.append({"frame": int(frame_idx), "hand": "left"})
        if right_binary[frame_idx] == 1:
            candidates.append({"frame": int(frame_idx), "hand": "right"})
    return sorted(candidates, key=lambda item: (item["frame"], item["hand"]))


def segment_contact_runs(binary: np.ndarray, hand: str) -> List[Dict[str, Any]]:
    arr = np.asarray(binary, dtype=int)
    runs: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(arr):
        if arr[idx] != 1 or (idx > 0 and arr[idx - 1] == 1):
            idx += 1
            continue
        end_idx = idx
        while end_idx + 1 < len(arr) and arr[end_idx + 1] == 1:
            end_idx += 1
        runs.append(
            {
                "hand": hand,
                "start_frame_0_based": int(idx),
                "end_frame_0_based": int(end_idx),
                "run_length": int(end_idx - idx + 1),
                "start_reason": "smoothed_contact_0_to_1",
            }
        )
        idx = end_idx + 1
    return runs


def build_new_contact_runs(left_binary: np.ndarray, right_binary: np.ndarray) -> List[Dict[str, Any]]:
    runs = segment_contact_runs(left_binary, "left") + segment_contact_runs(right_binary, "right")
    return sorted(runs, key=lambda item: (item["start_frame_0_based"], item["hand"]))


def compute_pre_contact_gap_start(
    *,
    run_start: int,
    left_binary: np.ndarray,
    right_binary: np.ndarray,
) -> int:
    """Return the start of the contiguous no-contact gap immediately before t0."""
    idx = int(run_start) - 1
    if idx < 0:
        return int(run_start)
    contact_any = (np.asarray(left_binary, dtype=int) == 1) | (np.asarray(right_binary, dtype=int) == 1)
    if contact_any[idx]:
        return int(run_start)
    while idx - 1 >= 0 and not contact_any[idx - 1]:
        idx -= 1
    return int(idx)


def runs_for_subaction(
    row: pd.Series,
    all_runs: Sequence[Dict[str, Any]],
    left_binary: np.ndarray,
    right_binary: np.ndarray,
    num_frames: int,
    early_offsets: Sequence[int],
    max_ref_backtrack: int,
) -> Tuple[List[Dict[str, Any]], str, str]:
    start_0, stop_0 = subaction_bounds(row, num_frames)
    raw_contact_present = False
    runs_in_subaction = []
    for run in all_runs:
        run_start = int(run["start_frame_0_based"])
        run_end = int(run["end_frame_0_based"])
        if run_end < start_0 or run_start > stop_0:
            continue
        raw_contact_present = True
        if start_0 <= run_start <= stop_0:
            runs_in_subaction.append(dict(run))

    if runs_in_subaction:
        decision = "new_contact_episode"
        reason = "smoothed_contact_run_start_within_subaction"
    elif raw_contact_present:
        decision = "continuation_contact_run"
        reason = "contact_present_but_run_started_before_subaction"
    else:
        decision = "no_contact_candidate"
        reason = "no_smoothed_contact_in_subaction"

    enriched_runs: List[Dict[str, Any]] = []
    for local_idx, run in enumerate(sorted(runs_in_subaction, key=lambda item: (item["start_frame_0_based"], item["hand"])), start=1):
        run_start = int(run["start_frame_0_based"])
        run_end = int(run["end_frame_0_based"])
        pre_gap_start = compute_pre_contact_gap_start(
            run_start=run_start,
            left_binary=left_binary,
            right_binary=right_binary,
        )
        pre_gap_end = run_start - 1
        gap_length = max(0, pre_gap_end - pre_gap_start + 1)
        raw_start = int(pre_gap_start)
        raw_end = int(pre_gap_end)
        effective_start = max(raw_start, run_start - int(max_ref_backtrack))
        effective_end = raw_end
        window_truncated = bool(effective_start > raw_start)
        candidates = []
        for offset in early_offsets:
            candidate_frame = run_start + int(offset)
            if candidate_frame > run_end or candidate_frame > stop_0:
                continue
            candidates.append(
                {
                    "frame": int(candidate_frame),
                    "hand": run["hand"],
                    "candidate_frame_offset_from_episode_start": int(offset),
                    "episode_contact_start_frame_0_based": run_start,
                    "episode_contact_end_frame_0_based": run_end,
                    "episode_contact_run_length": int(run["run_length"]),
                    "episode_id_within_subaction": int(local_idx),
                    "episode_start_hand": run["hand"],
                    "episode_start_reason": run["start_reason"],
                    "candidate_is_episode_representative": bool(int(offset) == 0),
                    "pre_contact_gap_start_frame_0_based": int(pre_gap_start),
                    "pre_contact_gap_end_frame_0_based": int(pre_gap_end),
                    "gap_length_before_episode_start": int(gap_length),
                    "raw_semantic_window_start_0_based": int(raw_start),
                    "raw_semantic_window_end_0_based": int(raw_end),
                    "effective_reference_window_start_0_based": int(effective_start),
                    "effective_reference_window_end_0_based": int(effective_end),
                    "max_ref_backtrack": int(max_ref_backtrack),
                    "window_truncated_by_max_backtrack": bool(window_truncated),
                    "reference_search_window_start_0_based": int(effective_start),
                    "reference_search_window_end_0_based": int(effective_end),
                }
            )
        run_copy = dict(run)
        run_copy["episode_id_within_subaction"] = int(local_idx)
        run_copy["pre_contact_gap_start_frame_0_based"] = int(pre_gap_start)
        run_copy["pre_contact_gap_end_frame_0_based"] = int(pre_gap_end)
        run_copy["gap_length_before_episode_start"] = int(gap_length)
        run_copy["raw_semantic_window_start_0_based"] = int(raw_start)
        run_copy["raw_semantic_window_end_0_based"] = int(raw_end)
        run_copy["effective_reference_window_start_0_based"] = int(effective_start)
        run_copy["effective_reference_window_end_0_based"] = int(effective_end)
        run_copy["max_ref_backtrack"] = int(max_ref_backtrack)
        run_copy["window_truncated_by_max_backtrack"] = bool(window_truncated)
        run_copy["reference_search_window_start_0_based"] = int(effective_start)
        run_copy["reference_search_window_end_0_based"] = int(effective_end)
        run_copy["candidates"] = candidates
        enriched_runs.append(run_copy)
    return enriched_runs, decision, reason


def _expand_bbox_xyxy(
    bbox: Sequence[float],
    image_shape: Tuple[int, int],
    expand_ratio: float,
) -> Tuple[int, int, int, int]:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * float(expand_ratio)
    pad_y = bh * float(expand_ratio)
    return (
        int(max(0, np.floor(x1 - pad_x))),
        int(max(0, np.floor(y1 - pad_y))),
        int(min(w, np.ceil(x2 + pad_x))),
        int(min(h, np.ceil(y2 + pad_y))),
    )


def build_engineered_dynamic_mask(
    frame_det,
    image_shape: Tuple[int, int],
    *,
    hand_score_threshold: float,
    object_score_threshold: float,
    expand_ratio: float,
) -> np.ndarray:
    """Return uint8 feature mask: 255 static, 0 hands and portable/contact objects."""
    h, w = image_shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)

    object_bboxes = get_valid_object_bboxes(frame_det, score_threshold=object_score_threshold)
    portable_object_bboxes: List[Sequence[float]] = []
    for hand in getattr(frame_det, "hands", []):
        if hand.score < hand_score_threshold:
            continue
        hand_bbox_norm = [hand.bbox.left, hand.bbox.top, hand.bbox.right, hand.bbox.bottom]
        hand_bbox_px = normalized_bbox_to_pixels(hand_bbox_norm, w, h)
        x1, y1, x2, y2 = _expand_bbox_xyxy(hand_bbox_px, image_shape, expand_ratio)
        mask[y1:y2, x1:x2] = 0

        if hand.state == HandState.PORTABLE_OBJECT:
            active_obj = select_active_object_bbox(hand_bbox_norm, object_bboxes)
            if active_obj is not None:
                portable_object_bboxes.append(active_obj)

    for object_bbox_norm in portable_object_bboxes:
        object_bbox_px = normalized_bbox_to_pixels(object_bbox_norm, w, h)
        x1, y1, x2, y2 = _expand_bbox_xyxy(object_bbox_px, image_shape, expand_ratio)
        mask[y1:y2, x1:x2] = 0
    return mask


def _detector_candidates() -> List[Tuple[str, Any, int]]:
    candidates: List[Tuple[str, Any, int]] = []
    if hasattr(cv2, "SIFT_create"):
        try:
            candidates.append(("SIFT", cv2.SIFT_create(nfeatures=2500), cv2.NORM_L2))
        except cv2.error:
            pass
    try:
        candidates.append(("AKAZE", cv2.AKAZE_create(), cv2.NORM_HAMMING))
    except cv2.error:
        pass
    candidates.append(("ORB", cv2.ORB_create(nfeatures=2500), cv2.NORM_HAMMING))
    return candidates


def compute_engineered_pairwise_homography(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    prev_mask: np.ndarray,
    cur_mask: np.ndarray,
    *,
    ratio_test: float,
    ransac_reproj_threshold: float,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    last_failure = "no_detector_attempted"
    for method_name, detector, norm_type in _detector_candidates():
        kp_prev, des_prev = detector.detectAndCompute(prev_gray, prev_mask)
        kp_cur, des_cur = detector.detectAndCompute(cur_gray, cur_mask)
        if des_prev is None or des_cur is None or len(kp_prev) < 8 or len(kp_cur) < 8:
            last_failure = f"insufficient_features_{method_name.lower()}"
            continue

        matcher = cv2.BFMatcher(norm_type, crossCheck=False)
        try:
            raw_matches = matcher.knnMatch(des_prev, des_cur, k=2)
        except cv2.error:
            last_failure = f"matcher_failed_{method_name.lower()}"
            continue

        good = []
        for item in raw_matches:
            if len(item) == 2 and item[0].distance < float(ratio_test) * item[1].distance:
                good.append(item[0])
        if len(good) < 4:
            return None, {
                "feature_method": method_name,
                "keypoints_prev": int(len(kp_prev)),
                "keypoints_cur": int(len(kp_cur)),
                "good_matches": int(len(good)),
                "inliers": 0,
                "inlier_ratio": 0.0,
                "median_reprojection_error": None,
                "fail_reason": "insufficient_good_matches",
                "passed_quality_gate": False,
            }

        src_pts = np.float32([kp_cur[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_prev[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, float(ransac_reproj_threshold))
        if H is None or inlier_mask is None:
            return None, {
                "feature_method": method_name,
                "keypoints_prev": int(len(kp_prev)),
                "keypoints_cur": int(len(kp_cur)),
                "good_matches": int(len(good)),
                "inliers": 0,
                "inlier_ratio": 0.0,
                "median_reprojection_error": None,
                "fail_reason": "ransac_failed",
                "passed_quality_gate": False,
            }

        inlier_mask_flat = inlier_mask.ravel().astype(bool)
        projected = cv2.perspectiveTransform(src_pts, H).reshape(-1, 2)
        target = dst_pts.reshape(-1, 2)
        errors = np.linalg.norm(projected - target, axis=1)
        inlier_errors = errors[inlier_mask_flat] if np.any(inlier_mask_flat) else errors
        inliers = int(np.sum(inlier_mask_flat))
        return H.astype(np.float64), {
            "feature_method": method_name,
            "keypoints_prev": int(len(kp_prev)),
            "keypoints_cur": int(len(kp_cur)),
            "good_matches": int(len(good)),
            "inliers": inliers,
            "inlier_ratio": float(inliers / max(1, len(good))),
            "median_reprojection_error": float(np.median(inlier_errors)) if len(inlier_errors) else None,
            "fail_reason": None,
            "passed_quality_gate": True,
        }

    return None, {
        "feature_method": None,
        "keypoints_prev": 0,
        "keypoints_cur": 0,
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "median_reprojection_error": None,
        "fail_reason": last_failure,
        "passed_quality_gate": False,
    }


def snap_points_to_mask(
    points_xy: np.ndarray,
    object_mask: np.ndarray,
    max_distance_px: float,
) -> Dict[str, Any]:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    mask_yx = np.argwhere(object_mask > 0)
    if len(mask_yx) == 0:
        return {
            "passed": False,
            "reason": "empty_reference_object_mask",
            "snapped_points": points,
            "snap_distances": [None for _ in points],
            "inside_count": 0,
            "max_snap_distance": None,
        }
    mask_xy = mask_yx[:, ::-1].astype(np.float32)
    h, w = object_mask.shape[:2]
    snapped = np.zeros_like(points, dtype=np.float32)
    distances: List[float] = []
    inside_count = 0
    for idx, pt in enumerate(points):
        x = int(round(float(pt[0])))
        y = int(round(float(pt[1])))
        if 0 <= x < w and 0 <= y < h and object_mask[y, x] > 0:
            snapped[idx] = pt
            distances.append(0.0)
            inside_count += 1
            continue
        dist2 = np.sum((mask_xy - pt[None, :]) ** 2, axis=1)
        nearest_idx = int(np.argmin(dist2))
        dist = float(np.sqrt(dist2[nearest_idx]))
        snapped[idx] = mask_xy[nearest_idx]
        distances.append(dist)

    max_dist = max(distances) if distances else None
    passed = bool(max_dist is not None and max_dist <= float(max_distance_px))
    return {
        "passed": passed,
        "reason": None if passed else "projected_contact_too_far_from_reference_object_mask",
        "snapped_points": snapped,
        "snap_distances": distances,
        "inside_count": int(inside_count),
        "max_snap_distance": float(max_dist) if max_dist is not None else None,
    }


class EngineeredHomographyRunnerE13:
    def __init__(self, detections, image_dir: Path, config: E13Config):
        self.detections = detections
        self.image_dir = Path(image_dir)
        self.config = config
        self.gray_cache: Dict[int, np.ndarray] = {}
        self.mask_cache: Dict[int, np.ndarray] = {}
        self.pair_cache: Dict[Tuple[int, int], Tuple[Optional[np.ndarray], Dict[str, Any]]] = {}
        self.cache_stats = Counter()

    def load_gray(self, frame_idx: int) -> np.ndarray:
        frame_idx = int(frame_idx)
        if frame_idx not in self.gray_cache:
            img_path = self.image_dir / f"frame_{frame_idx + 1:010d}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"missing frame image: {img_path}")
            self.gray_cache[frame_idx] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.gray_cache[frame_idx]

    def load_bgr(self, frame_idx: int) -> np.ndarray:
        img_path = self.image_dir / f"frame_{int(frame_idx) + 1:010d}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"missing frame image: {img_path}")
        return img

    def dynamic_mask(self, frame_idx: int):
        key = int(frame_idx)
        if key not in self.mask_cache:
            gray = self.load_gray(frame_idx)
            self.mask_cache[key] = build_engineered_dynamic_mask(
                self.detections[int(frame_idx)],
                gray.shape,
                hand_score_threshold=float(self.config.hand_score_threshold),
                object_score_threshold=float(self.config.object_score_threshold),
                expand_ratio=float(self.config.dynamic_bbox_expand_ratio),
            )
        return self.mask_cache[key]

    def pairwise_homography(self, cur_idx: int, prev_idx: int):
        key = (int(cur_idx), int(prev_idx))
        if key in self.pair_cache:
            self.cache_stats["pair_cache_hits"] += 1
            H, stats = self.pair_cache[key]
            return H.copy() if H is not None else None, dict(stats)

        cur_gray = self.load_gray(cur_idx)
        prev_gray = self.load_gray(prev_idx)
        cur_mask = self.dynamic_mask(cur_idx)
        prev_mask = self.dynamic_mask(prev_idx)
        H, stats = compute_engineered_pairwise_homography(
            prev_gray,
            cur_gray,
            prev_mask,
            cur_mask,
            ratio_test=float(self.config.homography_ratio_test),
            ransac_reproj_threshold=float(self.config.homography_ransac_reproj_threshold),
        )
        self.cache_stats["pair_cache_misses"] += 1
        self.pair_cache[key] = (H.copy() if H is not None else None, dict(stats))
        return H, dict(stats)

    def accumulate_homography_to_ref(self, ref_idx: int, target_idx: int):
        ref_idx = int(ref_idx)
        target_idx = int(target_idx)
        if target_idx == ref_idx:
            return np.eye(3, dtype=np.float64), [], None
        if target_idx < ref_idx:
            return None, [], "target_idx_less_than_ref_idx"

        H_total = np.eye(3, dtype=np.float64)
        pair_stats = []
        for cur_idx in range(target_idx, ref_idx, -1):
            prev_idx = cur_idx - 1
            H_cur_to_prev, stats = self.pairwise_homography(cur_idx, prev_idx)
            stats["cur"] = cur_idx
            stats["prev"] = prev_idx
            pair_stats.append(stats)
            if H_cur_to_prev is None:
                stats["passed_quality_gate"] = False
                return None, pair_stats, f"pairwise_homography_failed_f{cur_idx}_to_f{prev_idx}"
            if (
                stats["good_matches"] < int(self.config.homography_min_good_matches)
                or stats["inliers"] < int(self.config.homography_min_inliers)
                or stats["inlier_ratio"] < float(self.config.homography_min_inlier_ratio)
            ):
                stats["passed_quality_gate"] = False
                return None, pair_stats, f"pairwise_homography_low_quality_f{cur_idx}_to_f{prev_idx}"
            stats["passed_quality_gate"] = True
            H_total = H_cur_to_prev @ H_total
        return H_total, pair_stats, None

    def _trajectory_pixels_for_ref_shape(self, t_contact: int, active_hand: str, image_shape: Tuple[int, int]):
        h_img, w_img = image_shape[:2]
        trajectory_pixels = []
        trajectory_missing_offsets = []
        for offset in range(6):
            idx = int(t_contact) + offset
            if idx >= len(self.detections):
                break
            frame_det = self.detections[idx]
            found_active_hand = False
            for hand in frame_det.hands:
                if hand.score > 0.5 and hand.side.name.lower() == active_hand:
                    bbox = [hand.bbox.left, hand.bbox.top, hand.bbox.right, hand.bbox.bottom]
                    cx = (bbox[0] + bbox[2]) / 2 * w_img
                    cy = (bbox[1] + bbox[3]) / 2 * h_img
                    trajectory_pixels.append(np.array([cx, cy], dtype=np.float32))
                    found_active_hand = True
                    break
            if not found_active_hand:
                trajectory_missing_offsets.append(offset)
        return trajectory_pixels, trajectory_missing_offsets

    def build_reference_object_gate(
        self,
        *,
        ref_idx: int,
        ref_img: np.ndarray,
        object_polygon_ref: np.ndarray,
        raw_projected_points: np.ndarray,
    ) -> Dict[str, Any]:
        projected_bbox = polygon_to_bbox(object_polygon_ref, ref_img.shape)
        base: Dict[str, Any] = {
            "reference_object_consistency_pass": False,
            "reference_object_consistency_reason": None,
            "reference_object_bbox": None,
            "reference_object_bbox_score": None,
            "reference_object_iou": None,
            "reference_object_center_distance": None,
            "projected_contact_to_reference_object_distance": None,
            "projected_contact_means_inside_reference_object_count": 0,
            "reference_object_candidate_count": 0,
            "projected_object_bbox": tuple(float(v) for v in projected_bbox) if projected_bbox is not None else None,
            "object_mask_snap_inside_count": 0,
            "object_mask_snap_max_distance": None,
            "object_mask_snap_distances": None,
            "mu_transformed_raw": np.asarray(raw_projected_points, dtype=np.float32),
            "mu_transformed_snapped": None,
            "reference_object_mask": None,
        }
        if projected_bbox is None:
            base["reference_object_consistency_reason"] = "missing_projected_object_bbox"
            return base

        ref_objects = reference_object_bboxes(
            self.detections[int(ref_idx)],
            ref_img.shape,
            score_threshold=float(self.config.object_score_threshold),
        )
        base["reference_object_candidate_count"] = int(len(ref_objects))
        if not ref_objects:
            base["reference_object_consistency_reason"] = "no_reference_object_bbox"
            return base

        best = None
        best_key = None
        contact_centroid = np.asarray(raw_projected_points, dtype=np.float32).reshape(-1, 2).mean(axis=0)
        for item in ref_objects:
            bbox = item["bbox"]
            iou = bbox_iou(projected_bbox, bbox)
            center_dist = bbox_center_distance(projected_bbox, bbox)
            contact_dist = float(np.linalg.norm(contact_centroid - np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5])))
            key = (iou, -contact_dist, -center_dist, item["score"])
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    "bbox": bbox,
                    "score": float(item["score"]),
                    "iou": float(iou),
                    "center_distance": float(center_dist),
                    "contact_distance": float(contact_dist),
                }
        assert best is not None

        h, w = ref_img.shape[:2]
        ref_bbox_norm = [
            best["bbox"][0] / w,
            best["bbox"][1] / h,
            best["bbox"][2] / w,
            best["bbox"][3] / h,
        ]
        object_mask, _ = build_object_visible_mask(
            ref_img,
            ref_bbox_norm,
            hand_skin_mask=None,
            config=ContactExtractionConfig(
                object_score_threshold=float(self.config.object_score_threshold),
                use_object_mask=True,
                use_object_boundary=False,
                project_points_to_object_boundary=False,
            ),
        )
        snap = snap_points_to_mask(
            np.asarray(raw_projected_points, dtype=np.float32),
            object_mask,
            max_distance_px=float(self.config.snap_max_distance_px),
        )
        base.update(
            {
                "reference_object_bbox": tuple(float(v) for v in best["bbox"]),
                "reference_object_bbox_score": float(best["score"]),
                "reference_object_iou": float(best["iou"]),
                "reference_object_center_distance": float(best["center_distance"]),
                "projected_contact_to_reference_object_distance": float(best["contact_distance"]),
                "reference_object_mask": object_mask,
                "mu_transformed_snapped": snap["snapped_points"],
                "object_mask_snap_inside_count": int(snap["inside_count"]),
                "object_mask_snap_max_distance": snap["max_snap_distance"],
                "object_mask_snap_distances": snap["snap_distances"],
                "projected_contact_means_inside_reference_object_count": int(snap["inside_count"]),
            }
        )
        if snap["passed"]:
            base["reference_object_consistency_pass"] = True
            base["reference_object_consistency_reason"] = "snapped_to_reference_object_mask"
        else:
            base["reference_object_consistency_reason"] = snap["reason"]
        return base

    def run_cell3(
        self,
        *,
        t_contact: int,
        active_hand: str,
        contact_means: np.ndarray,
        output_dir: Path,
        episode_start_frame_0_based: int,
        reference_search_window_start_0_based: int,
        reference_search_window_end_0_based: Optional[int] = None,
        crop_size: int = 150,
        crop_hand_overlap_max: float = 0.05,
        discard: bool = False,
        discard_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        episode_start = int(episode_start_frame_0_based)
        search_start = max(0, int(reference_search_window_start_0_based))
        search_end = int(reference_search_window_end_0_based) if reference_search_window_end_0_based is not None else episode_start - 1
        candidate_offset = int(t_contact) - episode_start
        result: Dict[str, Any] = {
            "status": "KEEP",
            "discard": bool(discard),
            "discard_reason": discard_reason,
            "ref_idx": None,
            "trajectory_pixels": [],
            "trajectory_missing_offsets": [],
            "all_pair_stats": [],
            "H_contact_to_ref": None,
            "mu_transformed": None,
            "tau_transformed": [],
            "object_polygon_ref": None,
            "hand_polygon_ref": None,
            "contact_centroid": None,
            "projected_area": None,
            "area_ratio": None,
            "inside_count": None,
            "centroid_dist": None,
            "full_overlay_path": None,
            "crop_path": None,
            "crop_bbox": None,
            "reference_mode": None,
            "reference_reason": None,
            "strict_reference_available": False,
            "fallback_reference_used": False,
            "fallback_reference_attempted": False,
            "fallback_reference_in_pre_contact_gap": False,
            "fallback_ref_before_episode_start": False,
            "fallback_rescued_from_no_strict_humanless": False,
            "no_pre_episode_reference_window_not_rescued": False,
            "rescued_from_no_strict_humanless": False,
            "crop_bbox_150": None,
            "crop_hand_overlap_ratio": None,
            "object_hand_overlap_ratio": None,
            "fallback_attempted_refs": 0,
            "fallback_reject_reasons": {},
            "fallback_precheck_homography_min_inlier_ratio": None,
            "fallback_precheck_homography_min_inliers": None,
            "fallback_precheck_homography_min_good_matches": None,
            "episode_contact_start_frame_0_based": int(episode_start),
            "episode_contact_start_frame_1_based": int(episode_start) + 1,
            "candidate_frame_offset_from_episode_start": int(candidate_offset),
            "reference_anchor_frame_0_based": int(episode_start),
            "reference_search_window_start_0_based": int(search_start),
            "reference_search_window_end_0_based": int(search_end),
            "ref_before_episode_start": None,
            "is_candidate_at_episode_start": bool(candidate_offset == 0),
            "is_candidate_near_episode_start": bool(candidate_offset <= 2),
            "reference_anchor_gap": None,
        }
        if result["discard"]:
            result["status"] = "DISCARD"
            return result
        if contact_means is None:
            result.update(discard=True, discard_reason="missing_contact_means", status="DISCARD")
            return result
        if search_end < search_start:
            result.update(
                discard=True,
                discard_reason="no_pre_episode_reference_window",
                status="DISCARD",
                no_pre_episode_reference_window_not_rescued=True,
                reference_reason="empty_pre_contact_reference_window",
            )
            return result

        ref_idx = int(search_end)
        if ref_idx >= episode_start:
            ref_idx = episode_start - 1
        if ref_idx < search_start or ref_idx < 0:
            result.update(
                discard=True,
                discard_reason="no_short_window_pre_contact_reference",
                status="DISCARD",
                reference_reason="nearest_pre_contact_ref_outside_window",
            )
            return result
        result.update(
            ref_idx=ref_idx,
            reference_mode="short_window_pre_contact",
            reference_reason="nearest_frame_before_new_contact_episode",
            strict_reference_available=False,
            fallback_reference_used=False,
            ref_before_episode_start=bool(ref_idx < episode_start),
            reference_anchor_gap=int(episode_start - ref_idx),
            reference_gap=int(t_contact) - int(ref_idx),
        )

        ref_img = self.load_bgr(ref_idx)
        trajectory_pixels, trajectory_missing_offsets = self._trajectory_pixels_for_ref_shape(int(t_contact), active_hand, ref_img.shape)
        result["trajectory_pixels"] = trajectory_pixels
        result["trajectory_missing_offsets"] = trajectory_missing_offsets
        if len(trajectory_pixels) == 0:
            result.update(discard=True, discard_reason="missing_trajectory_points", status="DISCARD")
            return result

        cumulative_H_list: List[Optional[np.ndarray]] = []
        all_pair_stats = []
        fail_reason = None
        H_contact_to_ref, pair_stats, fail_reason = self.accumulate_homography_to_ref(ref_idx, int(t_contact))
        cumulative_H_list.append(H_contact_to_ref)
        all_pair_stats.extend(pair_stats)
        result["all_pair_stats"] = all_pair_stats
        if H_contact_to_ref is None:
            result.update(discard=True, discard_reason=fail_reason, status="DISCARD")
            return result

        H_contact = H_contact_to_ref
        result["H_contact_to_ref"] = H_contact
        mu_transformed_raw = transform_points(contact_means.astype(np.float32), H_contact)
        if mu_transformed_raw is None or not np.isfinite(mu_transformed_raw).all():
            result.update(discard=True, discard_reason="missing_transformed_contact_points", status="DISCARD")
            return result

        tau_transformed = []
        for offset in range(len(trajectory_pixels)):
            target_idx = int(t_contact) + offset
            if offset == 0:
                H = H_contact
                pair_stats = []
            else:
                H, pair_stats, _ = self.accumulate_homography_to_ref(ref_idx, target_idx)
                all_pair_stats.extend(pair_stats)
            if H is None:
                tau_transformed.append(None)
                continue
            pt = transform_points(trajectory_pixels[offset].reshape(1, -1).astype(np.float32), H)
            tau_transformed.append(pt[0] if pt is not None else None)
        result["all_pair_stats"] = all_pair_stats
        result["tau_transformed"] = tau_transformed

        h_img, w_img = ref_img.shape[:2]
        frame_det_contact = self.detections[int(t_contact)]
        active_hand_bbox_norm = get_active_hand_bbox(frame_det_contact, active_hand, score_threshold=0.5)
        object_bboxes_norm = get_valid_object_bboxes(frame_det_contact, score_threshold=0.5)
        active_object_bbox_norm = (
            select_active_object_bbox(active_hand_bbox_norm, object_bboxes_norm)
            if active_hand_bbox_norm is not None
            else None
        )
        if active_object_bbox_norm is None:
            result.update(discard=True, discard_reason="no_active_object_bbox", status="DISCARD")
            return result
        active_object_bbox = np.array(
            [
                active_object_bbox_norm[0] * w_img,
                active_object_bbox_norm[1] * h_img,
                active_object_bbox_norm[2] * w_img,
                active_object_bbox_norm[3] * h_img,
            ]
        )
        object_polygon_ref = transform_bbox_to_polygon(active_object_bbox, H_contact)
        result["object_polygon_ref"] = object_polygon_ref
        original_area = (active_object_bbox[2] - active_object_bbox[0]) * (active_object_bbox[3] - active_object_bbox[1])
        projected_area = polygon_area(object_polygon_ref)
        area_ratio = projected_area / original_area if original_area > 0 else 0.0
        result["projected_area"] = projected_area
        result["area_ratio"] = area_ratio
        if projected_area <= 0 or area_ratio < 0.15:
            result.update(discard=True, discard_reason="projected_object_polygon_invalid", status="DISCARD")
            return result

        object_gate = self.build_reference_object_gate(
            ref_idx=ref_idx,
            ref_img=ref_img,
            object_polygon_ref=object_polygon_ref,
            raw_projected_points=mu_transformed_raw,
        )
        for key, value in object_gate.items():
            if key != "reference_object_mask":
                result[key] = value
        result["reference_object_mask"] = object_gate.get("reference_object_mask")
        if not object_gate.get("reference_object_consistency_pass"):
            result.update(
                discard=True,
                discard_reason=object_gate.get("reference_object_consistency_reason") or "reference_object_mask_gate_failed",
                status="DISCARD",
            )
            return result

        mu_transformed = np.asarray(object_gate["mu_transformed_snapped"], dtype=np.float32)
        result["mu_transformed_raw"] = np.asarray(mu_transformed_raw, dtype=np.float32)
        result["mu_transformed"] = mu_transformed
        result["contact_centroid_raw"] = mu_transformed_raw.mean(axis=0)
        contact_centroid = mu_transformed.mean(axis=0)
        result["contact_centroid"] = contact_centroid
        result["inside_count"] = int(object_gate.get("object_mask_snap_inside_count") or 0)
        result["centroid_dist"] = None

        for offset, pt in enumerate(tau_transformed):
            if pt is None:
                continue
            if pt[0] < 0 or pt[1] < 0 or pt[0] > w_img or pt[1] > h_img:
                result.update(discard=True, discard_reason=f"trajectory_out_of_bounds_t+{offset}", status="DISCARD")
                return result
        for mu_idx, mu in enumerate(mu_transformed):
            if mu[0] < 0 or mu[1] < 0 or mu[0] > w_img or mu[1] > h_img:
                result.update(discard=True, discard_reason=f"contact_point_out_of_bounds_mu_{mu_idx + 1}", status="DISCARD")
                return result

        hand_polygon_ref = None
        for hand in frame_det_contact.hands:
            if hand.score > 0.5 and hand.side.name.lower() == active_hand:
                hand_bbox = np.array(
                    [
                        hand.bbox.left * w_img,
                        hand.bbox.top * h_img,
                        hand.bbox.right * w_img,
                        hand.bbox.bottom * h_img,
                    ]
                )
                hand_polygon_ref = transform_bbox_to_polygon(hand_bbox, H_contact)
                break
        result["hand_polygon_ref"] = hand_polygon_ref

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        full_overlay = draw_problem3_full_overlay(
            ref_img,
            object_polygon_ref,
            hand_polygon_ref,
            mu_transformed,
            tau_transformed,
            int(t_contact),
            active_hand,
            ref_idx,
            result["discard"],
            result["discard_reason"],
            contact_centroid,
        )
        full_overlay_path = output_dir / "problem3_ref_full_overlay.png"
        cv2.imwrite(str(full_overlay_path), full_overlay)
        result["full_overlay_path"] = str(full_overlay_path)

        crop_overlay, crop_bbox = draw_problem3_crop_overlay(
            ref_img,
            mu_transformed,
            tau_transformed,
            object_polygon_ref,
            crop_size=150,
        )
        crop_path = output_dir / "problem3_ref_crop.png"
        cv2.imwrite(str(crop_path), cv2.cvtColor(crop_overlay, cv2.COLOR_RGB2BGR))
        result["crop_path"] = str(crop_path)
        result["crop_bbox"] = crop_bbox

        crop_bbox_150 = compute_centered_crop_bbox(contact_centroid, ref_img.shape, crop_size=150)
        result["crop_bbox_150"] = tuple(int(v) for v in crop_bbox_150)
        result["crop_hand_overlap_ratio"] = float(
            compute_crop_hand_overlap_ratio(
                self.detections[int(ref_idx)],
                ref_img.shape,
                crop_bbox_150,
                score_threshold=0.5,
            )
        )
        result["status"] = "KEEP"
        return result


def compact_pair_stats(cell3_result: Dict[str, Any]) -> Dict[str, Any]:
    stats = cell3_result.get("all_pair_stats") or []
    ratios = []
    inliers = []
    good_matches = []
    median_errors = []
    methods = []
    fail_reasons = []
    for item in stats:
        if item.get("inlier_ratio") is not None:
            ratios.append(float(item.get("inlier_ratio", 0.0)))
        if item.get("inliers") is not None:
            inliers.append(int(item.get("inliers", 0)))
        if item.get("good_matches") is not None:
            good_matches.append(int(item.get("good_matches", 0)))
        if item.get("median_reprojection_error") is not None:
            median_errors.append(float(item.get("median_reprojection_error", 0.0)))
        if item.get("feature_method"):
            methods.append(str(item.get("feature_method")))
        if item.get("fail_reason"):
            fail_reasons.append(str(item.get("fail_reason")))
    return {
        "homography_pairs": int(len(stats)),
        "homography_min_inlier_ratio": min(ratios) if ratios else None,
        "homography_min_inliers": min(inliers) if inliers else None,
        "homography_min_good_matches": min(good_matches) if good_matches else None,
        "homography_max_median_reprojection_error": max(median_errors) if median_errors else None,
        "homography_feature_methods": ";".join(sorted(set(methods))) if methods else None,
        "homography_fail_reasons": ";".join(sorted(set(fail_reasons))) if fail_reasons else None,
    }


def score_success_candidate(record: Dict[str, Any]) -> float:
    ratio = record.get("homography_min_inlier_ratio")
    ratio = 0.0 if ratio is None or pd.isna(ratio) else float(ratio)
    contact_points = int(record.get("contact_points") or 0)
    traj_pts = int(record.get("trajectory_points") or 0)
    ref_gap = int(record.get("ref_gap") or 0)
    missing_traj = int(record.get("missing_trajectory_count") or 0)
    candidate_offset = int(record.get("candidate_frame_offset_from_episode_start") or 0)
    return 2.0 * ratio + 0.02 * contact_points + 0.5 * traj_pts - 0.01 * ref_gap - 0.5 * missing_traj - 0.2 * candidate_offset


def write_cell4_pipeline_outputs(
    *,
    config: E13Config,
    record: Dict[str, Any],
    cell2_result: Dict[str, Any],
    cell3_result: Dict[str, Any],
    sample_dir: Path,
) -> Dict[str, Any]:
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    ref_idx = int(cell3_result["ref_idx"])
    ref_path = config.image_dir / f"frame_{ref_idx + 1:010d}.jpg"
    ref_img_bgr = cv2.imread(str(ref_path))
    if ref_img_bgr is None:
        raise FileNotFoundError(f"missing reference frame: {ref_path}")
    ref_img_rgb = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = ref_img_rgb.shape[:2]

    contact_means = cell2_result["contact_means"]
    contact_weights = cell2_result["contact_weights"]
    contact_covariances = cell2_result["contact_covariances"]
    mu_transformed = np.asarray(cell3_result["mu_transformed"], dtype=np.float32)
    H_contact_to_ref = cell3_result["H_contact_to_ref"]

    heatmap_mode = "covariance"
    covariances_ref = transform_covariances_by_homography(contact_covariances, contact_means, H_contact_to_ref)
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
    merged_heatmap = merge_label_heatmaps(per_mode_heatmaps=per_mode_heatmaps, merge_method="sum", normalize_output=True)

    heatmap_paths = silent_call(
        save_label_heatmap_outputs,
        merged_heatmap=merged_heatmap,
        ref_image=ref_img_rgb,
        output_dir=sample_dir,
        prefix="label_heatmap",
        sigma_px=12.0,
        merge_method="sum",
        use_weights=True,
        heatmap_mode=heatmap_mode,
        t_contact=record["frame_0_based"],
        active_hand=record["hand"],
        ref_idx=ref_idx,
    )

    reference_path = sample_dir / "reference_frame.png"
    cv2.imwrite(str(reference_path), ref_img_bgr)

    contact_frame_idx = int(record["frame_0_based"])
    contact_path = config.image_dir / f"frame_{contact_frame_idx + 1:010d}.jpg"
    contact_img_bgr = cv2.imread(str(contact_path))
    if contact_img_bgr is None:
        raise FileNotFoundError(f"missing contact frame: {contact_path}")
    contact_frame_path = sample_dir / "contact_frame.png"
    cv2.imwrite(str(contact_frame_path), contact_img_bgr)

    contact_overlay = contact_img_bgr.copy()
    for idx, point in enumerate(np.asarray(contact_means, dtype=np.float32).reshape(-1, 2)):
        cv2.circle(contact_overlay, (int(round(point[0])), int(round(point[1]))), 4, (0, 0, 255), -1)
        cv2.putText(
            contact_overlay,
            str(idx + 1),
            (int(round(point[0])) + 5, int(round(point[1])) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
    cv2.putText(
        contact_overlay,
        f"contact f{contact_frame_idx} {record['hand']}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    contact_overlay_path = sample_dir / "contact_overlay.png"
    cv2.imwrite(str(contact_overlay_path), contact_overlay)

    projection_overlay = ref_img_bgr.copy()
    reference_object_mask = cell3_result.get("reference_object_mask")
    if isinstance(reference_object_mask, np.ndarray) and np.any(reference_object_mask > 0):
        contours, _ = cv2.findContours((reference_object_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(projection_overlay, contours, -1, (255, 255, 255), 1)
    ref_bbox = cell3_result.get("reference_object_bbox")
    if ref_bbox is not None:
        x1, y1, x2, y2 = [int(round(float(v))) for v in ref_bbox]
        cv2.rectangle(projection_overlay, (x1, y1), (x2, y2), (255, 0, 0), 1)
    raw_points = np.asarray(cell3_result.get("mu_transformed_raw", mu_transformed), dtype=np.float32).reshape(-1, 2)
    snapped_points = np.asarray(mu_transformed, dtype=np.float32).reshape(-1, 2)
    for raw, snapped in zip(raw_points, snapped_points):
        raw_pt = (int(round(float(raw[0]))), int(round(float(raw[1]))))
        snap_pt = (int(round(float(snapped[0]))), int(round(float(snapped[1]))))
        cv2.circle(projection_overlay, raw_pt, 4, (0, 0, 255), -1)
        cv2.circle(projection_overlay, snap_pt, 4, (0, 255, 0), 1)
        cv2.line(projection_overlay, raw_pt, snap_pt, (0, 255, 255), 1)
    cv2.putText(
        projection_overlay,
        f"ref f{ref_idx} raw(red) snapped(green)",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    projection_overlay_path = sample_dir / "projection_overlay_raw_vs_snapped.png"
    cv2.imwrite(str(projection_overlay_path), projection_overlay)

    hcat_h = max(contact_overlay.shape[0], projection_overlay.shape[0])
    def pad_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
        if img.shape[0] == target_h:
            return img
        pad = np.zeros((target_h - img.shape[0], img.shape[1], 3), dtype=img.dtype)
        return np.vstack([img, pad])

    match_sheet = cv2.hconcat([pad_to_height(contact_overlay, hcat_h), pad_to_height(projection_overlay, hcat_h)])
    cv2.putText(
        match_sheet,
        f"min good={record.get('homography_min_good_matches')} min inliers={record.get('homography_min_inliers')} min ratio={record.get('homography_min_inlier_ratio')}",
        (8, hcat_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    homography_matches_path = sample_dir / "homography_matches.png"
    cv2.imwrite(str(homography_matches_path), match_sheet)

    contact_anchor = cell3_result.get("contact_centroid")
    if contact_anchor is None:
        contact_anchor = mu_transformed.mean(axis=0)
    vrb_style_img, arrow_info = draw_vrb_style_affordance_overlay(
        ref_img_rgb,
        merged_heatmap,
        cell3_result.get("tau_transformed", []),
        contact_anchor,
        heatmap_alpha=0.45,
        colormap=cv2.COLORMAP_JET,
        arrow_length_px=None,
        arrow_length_heatmap_ratio=1.2,
        arrow_width_px=None,
        source_step="last",
    )
    vrb_style_path = sample_dir / "vrb_style_affordance.png"
    cv2.imwrite(str(vrb_style_path), cv2.cvtColor(vrb_style_img, cv2.COLOR_RGB2BGR))

    return {
        "sample_dir": str(sample_dir),
        "contact_frame": str(contact_frame_path),
        "contact_overlay": str(contact_overlay_path),
        "projection_overlay_raw_vs_snapped": str(projection_overlay_path),
        "homography_matches": str(homography_matches_path),
        "reference_frame": str(reference_path),
        "label_heatmap_npy": heatmap_paths["npy_path"],
        "label_heatmap_png": heatmap_paths["png_path"],
        "label_heatmap_overlay": heatmap_paths["overlay_path"],
        "vrb_style_affordance": str(vrb_style_path),
        "heatmap_mode": heatmap_mode,
        "heatmap_shape": tuple(int(v) for v in merged_heatmap.shape),
        "arrow_generated": arrow_info is not None,
        "problem3_full_overlay": str(sample_dir / "diagnostics" / "problem3_ref_full_overlay.png"),
        "problem3_crop": str(sample_dir / "diagnostics" / "problem3_ref_crop.png"),
    }


def base_candidate_record(
    *,
    row: pd.Series,
    candidate: Dict[str, Any],
    start_0: int,
    stop_0: int,
    candidate_index: int,
    config: E13Config,
    subaction_new_contact_count: int,
    raw_candidate_count: int,
) -> Dict[str, Any]:
    episode_start = int(candidate["episode_contact_start_frame_0_based"])
    search_start = int(candidate["reference_search_window_start_0_based"])
    search_end = int(candidate["reference_search_window_end_0_based"])
    pre_contact_window_nonempty = bool(search_start <= search_end)
    return {
        "experiment_id": EXPERIMENT["experiment_id"],
        "experiment_name": EXPERIMENT["name"],
        "contact_strategy": EXPERIMENT["contact_strategy"],
        "reference_strategy": EXPERIMENT["reference_strategy"],
        "subaction_index": int(row.name),
        "narration_id": row["narration_id"],
        "video_id": row["video_id"],
        "narration": row["narration"],
        "verb": row["verb"],
        "noun": row["noun"],
        "all_nouns": row.get("all_nouns"),
        "start_frame_1_based": int(row["start_frame"]),
        "stop_frame_1_based": int(row["stop_frame"]),
        "clipped_start_frame_0_based": int(start_0),
        "clipped_stop_frame_0_based": int(stop_0),
        "candidate_index": int(candidate_index),
        "frame_0_based": int(candidate["frame"]),
        "frame_1_based": int(candidate["frame"]) + 1,
        "hand": candidate["hand"],
        "status": "pending",
        "passed_stage": "candidate",
        "failed_stage": None,
        "fail_reason": None,
        "contact_points": 0,
        "ref_idx": None,
        "ref_frame_1_based": None,
        "ref_gap": None,
        "reference_gap": None,
        "reference_anchor_gap": None,
        "trajectory_points": 0,
        "missing_trajectory_count": None,
        "heatmap_mode": None,
        "sample_score": None,
        "sample_dir": None,
        "reference_mode": None,
        "reference_reason": None,
        "strict_reference_available": False,
        "fallback_reference_used": False,
        "fallback_reference_attempted": False,
        "fallback_reference_in_pre_contact_gap": False,
        "fallback_ref_before_episode_start": False,
        "fallback_rescued_from_no_strict_humanless": False,
        "no_pre_episode_reference_window_not_rescued": bool(not pre_contact_window_nonempty),
        "rescued_from_no_strict_humanless": False,
        "crop_bbox_150": None,
        "crop_hand_overlap_ratio": None,
        "object_hand_overlap_ratio": None,
        "fallback_attempted_refs": 0,
        "fallback_reject_reasons": None,
        "fallback_precheck_homography_min_inlier_ratio": None,
        "fallback_precheck_homography_min_inliers": None,
        "fallback_precheck_homography_min_good_matches": None,
        "pre_contact_window_nonempty": bool(pre_contact_window_nonempty),
        "episode_contact_start_frame_0_based": episode_start,
        "episode_contact_start_frame_1_based": episode_start + 1,
        "episode_contact_end_frame_0_based": int(candidate["episode_contact_end_frame_0_based"]),
        "candidate_frame_offset_from_episode_start": int(candidate["candidate_frame_offset_from_episode_start"]),
        "reference_anchor_frame_0_based": episode_start,
        "pre_contact_gap_start_frame_0_based": int(candidate["pre_contact_gap_start_frame_0_based"]),
        "pre_contact_gap_end_frame_0_based": int(candidate["pre_contact_gap_end_frame_0_based"]),
        "gap_length_before_episode_start": int(candidate["gap_length_before_episode_start"]),
        "raw_semantic_window_start_0_based": int(candidate["raw_semantic_window_start_0_based"]),
        "raw_semantic_window_end_0_based": int(candidate["raw_semantic_window_end_0_based"]),
        "effective_reference_window_start_0_based": int(candidate["effective_reference_window_start_0_based"]),
        "effective_reference_window_end_0_based": int(candidate["effective_reference_window_end_0_based"]),
        "max_ref_backtrack": int(candidate["max_ref_backtrack"]),
        "window_truncated_by_max_backtrack": bool(candidate["window_truncated_by_max_backtrack"]),
        "reference_search_window_start_0_based": int(search_start),
        "reference_search_window_end_0_based": int(search_end),
        "ref_before_episode_start": None,
        "is_candidate_at_episode_start": bool(candidate["candidate_frame_offset_from_episode_start"] == 0),
        "is_candidate_near_episode_start": bool(candidate["candidate_frame_offset_from_episode_start"] <= 2),
        "episode_contact_run_length": int(candidate["episode_contact_run_length"]),
        "subaction_new_contact_count": int(subaction_new_contact_count),
        "candidate_is_episode_representative": bool(candidate.get("candidate_is_episode_representative", False)),
        "episode_id_within_subaction": int(candidate["episode_id_within_subaction"]),
        "episode_start_hand": candidate["episode_start_hand"],
        "episode_start_reason": candidate["episode_start_reason"],
        "reference_object_consistency_pass": None,
        "reference_object_consistency_reason": None,
        "reference_object_bbox": None,
        "reference_object_bbox_score": None,
        "reference_object_iou": None,
        "reference_object_center_distance": None,
        "projected_object_bbox": None,
        "projected_object_area": None,
        "projected_contact_to_reference_object_distance": None,
        "projected_contact_means_inside_reference_object_count": None,
        "reference_object_candidate_count": None,
        "object_mask_snap_inside_count": None,
        "object_mask_snap_max_distance": None,
        "object_mask_snap_distances": None,
        "mu_transformed_raw": None,
        "mu_transformed_snapped": None,
        "is_articulated_object_candidate": bool(is_articulated_object_candidate(row["noun"], row.get("all_nouns"), row["narration"])),
        "articulated_object_policy": "record_only",
        "episode_decision": "new_contact_episode",
        "episode_final_label": "new_contact_episode",
        "episode_id": f"{row['video_id']}_sub{int(row.name):03d}_ep{int(candidate['episode_id_within_subaction']):02d}_{candidate['hand']}",
        "episode_reason": "smoothed_contact_run_start_within_subaction",
        "episode_object_key": str(row["noun"]).strip().lower(),
        "episode_noun_tokens": str(parse_all_nouns_field(row.get("all_nouns"))),
        "episode_prev_subaction_index": None,
        "episode_prev_narration_id": None,
        "episode_prev_last_contact_frame": None,
        "episode_gap_start_frame": None,
        "episode_gap_stop_frame": None,
        "episode_no_contact_streak": None,
        "episode_no_hand_streak": None,
        "episode_raw_candidate_count": int(raw_candidate_count),
        "episode_filtered_candidate_count": int(subaction_new_contact_count),
        "min_ref_frame_idx": int(search_start),
        "reference_frame": None,
        "contact_frame": None,
        "contact_overlay": None,
        "projection_overlay_raw_vs_snapped": None,
        "homography_matches": None,
        "label_heatmap_npy": None,
        "label_heatmap_png": None,
        "label_heatmap_overlay": None,
        "vrb_style_affordance": None,
        "heatmap_shape": None,
        "arrow_generated": None,
        "problem3_full_overlay": None,
        "problem3_crop": None,
        "homography_pairs": None,
        "homography_min_inlier_ratio": None,
        "homography_min_inliers": None,
        "homography_min_good_matches": None,
        "homography_max_median_reprojection_error": None,
        "homography_feature_methods": None,
        "homography_fail_reasons": None,
    }


def write_candidate_json(record: Dict[str, Any], sample_dir: Path) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    candidate_json = sample_dir / "candidate_result.json"
    candidate_json.write_text(json.dumps(_json_safe(record), ensure_ascii=False, indent=2), encoding="utf-8")


def run_one_candidate(
    *,
    row: pd.Series,
    candidate: Dict[str, Any],
    candidate_index: int,
    detections,
    contact_config: ContactExtractionConfig,
    problem3_runner: EngineeredHomographyRunnerE13,
    config: E13Config,
    subaction_new_contact_count: int,
    raw_candidate_count: int,
) -> Dict[str, Any]:
    start_0, stop_0 = subaction_bounds(row, len(detections))
    record = base_candidate_record(
        row=row,
        candidate=candidate,
        start_0=start_0,
        stop_0=stop_0,
        candidate_index=candidate_index,
        config=config,
        subaction_new_contact_count=subaction_new_contact_count,
        raw_candidate_count=raw_candidate_count,
    )
    sample_dir = (
        config.output_root
        / "experiments"
        / EXPERIMENT["experiment_id"]
        / "pipeline_outputs"
        / f"subaction_{int(row.name):02d}_{row['narration_id']}"
        / f"episode_{int(candidate['episode_id_within_subaction']):02d}_{candidate['hand']}"
        / f"frame_{int(candidate['frame']):06d}_{candidate['hand']}"
    )
    diagnostics_dir = sample_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    record["sample_dir"] = str(sample_dir)

    sample = {
        "frame": int(candidate["frame"]),
        "hand": candidate["hand"],
        "source": f"{EXPERIMENT['experiment_id']}_{EXPERIMENT['contact_strategy']}",
        "narration_id": row["narration_id"],
    }
    cell2 = fit_cell2_contact_gmm(
        detections=detections,
        sample=sample,
        image_dir=config.image_dir,
        contact_extraction_config=contact_config,
    )
    record["contact_points"] = int(cell2.get("contact_points", 0))
    if not cell2["passed"]:
        record.update(status="discard", failed_stage="Cell2_contact_gmm", fail_reason=cell2["reason"])
        write_candidate_json(record, sample_dir)
        return record

    record["passed_stage"] = "Cell2_contact_gmm"
    cell3 = problem3_runner.run_cell3(
        t_contact=int(candidate["frame"]),
        active_hand=candidate["hand"],
        contact_means=cell2["contact_means"],
        output_dir=diagnostics_dir,
        episode_start_frame_0_based=int(candidate["episode_contact_start_frame_0_based"]),
        reference_search_window_start_0_based=int(record["reference_search_window_start_0_based"]),
        reference_search_window_end_0_based=int(record["reference_search_window_end_0_based"]),
        crop_size=int(config.crop_size),
        crop_hand_overlap_max=float(config.crop_hand_overlap_max),
        discard=False,
        discard_reason=None,
    )
    record["ref_idx"] = cell3.get("ref_idx")
    if record["ref_idx"] is not None:
        record["ref_frame_1_based"] = int(record["ref_idx"]) + 1
        record["ref_gap"] = int(candidate["frame"]) - int(record["ref_idx"])
        record["reference_gap"] = record["ref_gap"]
        record["reference_anchor_gap"] = int(candidate["episode_contact_start_frame_0_based"]) - int(record["ref_idx"])

    for field in [
        "reference_mode",
        "reference_reason",
        "strict_reference_available",
        "fallback_reference_used",
        "fallback_reference_attempted",
        "fallback_reference_in_pre_contact_gap",
        "fallback_ref_before_episode_start",
        "fallback_rescued_from_no_strict_humanless",
        "no_pre_episode_reference_window_not_rescued",
        "rescued_from_no_strict_humanless",
        "crop_bbox_150",
        "crop_hand_overlap_ratio",
        "object_hand_overlap_ratio",
        "fallback_attempted_refs",
        "fallback_reject_reasons",
        "fallback_precheck_homography_min_inlier_ratio",
        "fallback_precheck_homography_min_inliers",
        "fallback_precheck_homography_min_good_matches",
        "ref_before_episode_start",
        "reference_gap",
        "reference_anchor_gap",
        "reference_object_consistency_pass",
        "reference_object_consistency_reason",
        "reference_object_bbox",
        "reference_object_bbox_score",
        "reference_object_iou",
        "reference_object_center_distance",
        "projected_object_bbox",
        "projected_object_area",
        "projected_contact_to_reference_object_distance",
        "projected_contact_means_inside_reference_object_count",
        "reference_object_candidate_count",
        "object_mask_snap_inside_count",
        "object_mask_snap_max_distance",
        "object_mask_snap_distances",
        "mu_transformed_raw",
        "mu_transformed_snapped",
    ]:
        record[field] = cell3.get(field, record.get(field))

    record["trajectory_points"] = len(cell3.get("trajectory_pixels") or [])
    record["missing_trajectory_count"] = len(cell3.get("trajectory_missing_offsets") or [])
    record.update(compact_pair_stats(cell3))

    if cell3.get("H_contact_to_ref") is not None:
        record["passed_stage"] = "Cell3_homography_available"
    if cell3.get("discard") or cell3.get("status") != "KEEP":
        reason = cell3.get("discard_reason") or "cell3_discard"
        record.update(status="discard", failed_stage="Cell3_homography_geometry", fail_reason=reason)
        write_candidate_json(record, sample_dir)
        return record

    record["passed_stage"] = "Cell3_homography_geometry"
    cell4 = run_cell4_heatmap_gate(
        cell3_result=cell3,
        contact_means=cell2["contact_means"],
        contact_weights=cell2["contact_weights"],
        contact_covariances=cell2["contact_covariances"],
        image_dir=config.image_dir,
    )
    if not cell4["passed"]:
        record.update(status="discard", failed_stage="Cell4_label_heatmap", fail_reason=cell4["reason"])
        write_candidate_json(record, sample_dir)
        return record

    artifacts = write_cell4_pipeline_outputs(
        config=config,
        record=record,
        cell2_result=cell2,
        cell3_result=cell3,
        sample_dir=sample_dir,
    )
    record.update(artifacts)
    record["status"] = "keep"
    record["passed_stage"] = "Cell4_label_heatmap"
    record["failed_stage"] = None
    record["fail_reason"] = None
    record["heatmap_mode"] = cell4.get("heatmap_mode")
    record["sample_score"] = score_success_candidate(record)
    write_candidate_json(record, sample_dir)
    return record


def summarize_subactions(candidate_df: pd.DataFrame, subactions: pd.DataFrame, candidate_plan_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in subactions.iterrows():
        sub_df = candidate_df[candidate_df["subaction_index"] == idx].copy()
        plan_row = candidate_plan_df[candidate_plan_df["subaction_index"] == idx]
        plan = plan_row.iloc[0] if not plan_row.empty else None
        success_df = sub_df[sub_df["status"] == "keep"].copy()
        best = None
        if not success_df.empty:
            success_df = success_df.sort_values("sample_score", ascending=False)
            best = success_df.iloc[0]
        if not sub_df.empty:
            failed = sub_df[sub_df["status"] != "keep"]
            fail_reason = failed["fail_reason"].value_counts(dropna=True).index[0] if not failed.empty else None
        elif plan is not None and int(plan["raw_candidate_count"]) > 0:
            fail_reason = "continuation_contact_run"
        else:
            fail_reason = "no_contact_candidate"

        rows.append(
            {
                "experiment_id": EXPERIMENT["experiment_id"],
                "experiment_name": EXPERIMENT["name"],
                "subaction_index": int(idx),
                "narration_id": row["narration_id"],
                "narration": row["narration"],
                "verb": row["verb"],
                "noun": row["noun"],
                "all_nouns": row.get("all_nouns"),
                "frames_1_based": f"{int(row['start_frame'])}-{int(row['stop_frame'])}",
                "subaction_new_contact_count": int(plan["subaction_new_contact_count"]) if plan is not None else 0,
                "candidates": int(len(sub_df)),
                "cell2_pass": int((sub_df["passed_stage"].isin(["Cell2_contact_gmm", "Cell3_homography_available", "Cell3_homography_geometry", "Cell4_label_heatmap"])).sum()) if not sub_df.empty else 0,
                "ref_found": int(sub_df["ref_idx"].notna().sum()) if not sub_df.empty else 0,
                "homography_available": int((sub_df["passed_stage"].isin(["Cell3_homography_available", "Cell3_homography_geometry", "Cell4_label_heatmap"])).sum()) if not sub_df.empty else 0,
                "geometry_pass": int((sub_df["passed_stage"].isin(["Cell3_homography_geometry", "Cell4_label_heatmap"])).sum()) if not sub_df.empty else 0,
                "heatmap_pass": int((sub_df["status"] == "keep").sum()) if not sub_df.empty else 0,
                "final_samples": int((sub_df["status"] == "keep").sum()) if not sub_df.empty else 0,
                "best_frame_0_based": int(best["frame_0_based"]) if best is not None else None,
                "best_ref_idx": int(best["ref_idx"]) if best is not None and pd.notna(best["ref_idx"]) else None,
                "best_score": float(best["sample_score"]) if best is not None else None,
                "best_sample_dir": best["sample_dir"] if best is not None else None,
                "best_episode_contact_start_frame_0_based": int(best["episode_contact_start_frame_0_based"]) if best is not None else None,
                "best_candidate_frame_offset_from_episode_start": int(best["candidate_frame_offset_from_episode_start"]) if best is not None else None,
                "main_failure": fail_reason,
                "episode_decision": str(plan["episode_decision"]) if plan is not None and pd.notna(plan["episode_decision"]) else None,
                "episode_final_label": "new_contact_episode" if best is not None else (str(plan["episode_decision"]) if plan is not None and pd.notna(plan["episode_decision"]) else None),
                "episode_id": None,
                "episode_reason": str(plan["episode_reason"]) if plan is not None and pd.notna(plan["episode_reason"]) else None,
            }
        )
    return pd.DataFrame(rows)


def build_overview(candidate_df: pd.DataFrame, subaction_summary_df: pd.DataFrame, candidate_plan_df: pd.DataFrame) -> pd.DataFrame:
    failure_counts = candidate_df[candidate_df["status"] != "keep"]["fail_reason"].value_counts(dropna=True)
    main_failure = failure_counts.index[0] if len(failure_counts) else None
    return pd.DataFrame(
        [
            {
                "experiment_id": EXPERIMENT["experiment_id"],
                "experiment": EXPERIMENT["name"],
                "contact_strategy": EXPERIMENT["contact_strategy"],
                "reference_strategy": EXPERIMENT["reference_strategy"],
                "subactions_total": int(len(subaction_summary_df)),
                "subactions_success": int((subaction_summary_df["final_samples"] > 0).sum()),
                "candidates_total": int(len(candidate_df)),
                "raw_candidate_total": int(candidate_plan_df["raw_candidate_count"].sum()),
                "episode_filtered_candidate_total": int(candidate_plan_df["episode_filtered_candidate_count"].sum()),
                "deep_run_candidate_total": int(candidate_plan_df["deep_run_candidate_count"].sum()),
                "episode_new_contact_subactions": int((candidate_plan_df["episode_decision"] == "new_contact_episode").sum()),
                "episode_continuation_subactions": int((candidate_plan_df["episode_decision"] == "continuation_contact_run").sum()),
                "episode_no_contact_subactions": int((candidate_plan_df["episode_decision"] == "no_contact_candidate").sum()),
                "episode_pipeline_discarded_subactions": 0,
                "cell2_pass": int((candidate_df["passed_stage"].isin(["Cell2_contact_gmm", "Cell3_homography_available", "Cell3_homography_geometry", "Cell4_label_heatmap"])).sum()),
                "ref_found": int(candidate_df["ref_idx"].notna().sum()),
                "strict_ref_found": int((candidate_df.get("reference_mode") == "strict_global_no_hand").sum()) if "reference_mode" in candidate_df else 0,
                "fallback_ref_found": int((candidate_df.get("reference_mode") == "crop_level_humanless").sum()) if "reference_mode" in candidate_df else 0,
                "fallback_ref_used": int(candidate_df.get("fallback_reference_used", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "rescued_from_no_strict_humanless": int(candidate_df.get("rescued_from_no_strict_humanless", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "fallback_rescued_from_no_strict_humanless": int(candidate_df.get("fallback_rescued_from_no_strict_humanless", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "fallback_attempted": int(candidate_df.get("fallback_reference_attempted", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "no_pre_episode_reference_window_not_rescued": int(candidate_df.get("no_pre_episode_reference_window_not_rescued", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "max_ref_backtrack": int(candidate_df["max_ref_backtrack"].dropna().iloc[0]) if "max_ref_backtrack" in candidate_df and candidate_df["max_ref_backtrack"].notna().any() else None,
                "window_truncated_by_max_backtrack_count": int(candidate_df.get("window_truncated_by_max_backtrack", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "window_truncated_by_max_backtrack_rate": float(candidate_df.get("window_truncated_by_max_backtrack", pd.Series(dtype=bool)).fillna(False).astype(bool).mean()) if len(candidate_df) else 0.0,
                "homography_available": int((candidate_df["passed_stage"].isin(["Cell3_homography_available", "Cell3_homography_geometry", "Cell4_label_heatmap"])).sum()),
                "object_consistency_pass": 0,
                "object_consistency_fail": 0,
                "geometry_pass": int((candidate_df["passed_stage"].isin(["Cell3_homography_geometry", "Cell4_label_heatmap"])).sum()),
                "heatmap_pass": int((candidate_df["status"] == "keep").sum()),
                "main_failure": main_failure,
                "description": EXPERIMENT["description"],
            }
        ]
    )


def save_bar_chart_failure_reasons(candidate_df: pd.DataFrame, path: Path) -> None:
    failed = candidate_df[candidate_df["status"] != "keep"]
    if failed.empty:
        return
    counts = failed["fail_reason"].value_counts(dropna=True)
    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(counts))))
    counts.sort_values().plot(kind="barh", ax=ax, color="#c5221f")
    ax.set_title("E13 failure reasons")
    ax.set_xlabel("candidate count")
    ax.set_ylabel("failure reason")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_funnel_chart(overview_df: pd.DataFrame, path: Path) -> None:
    row = overview_df.iloc[0]
    stages = [
        "raw_candidate_total",
        "episode_filtered_candidate_total",
        "deep_run_candidate_total",
        "cell2_pass",
        "ref_found",
        "homography_available",
        "geometry_pass",
        "heatmap_pass",
    ]
    labels = [
        "raw candidates",
        "new-contact episodes",
        "deep-run frames",
        "Cell2",
        "reference",
        "homography",
        "geometry",
        "heatmap",
    ]
    y = [int(row[stage]) for stage in stages]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    ax.plot(x, y, marker="o", color="#1a73e8")
    for xi, yi in zip(x, y):
        ax.text(xi, yi, str(yi), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("count")
    ax.set_title("E13 pipeline funnel")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_timeline_chart(candidate_df: pd.DataFrame, subactions: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, max(5, 0.5 * len(subactions))))
    y_ticks = []
    y_labels = []
    for idx, row in subactions.iterrows():
        y = len(subactions) - idx
        y_ticks.append(y)
        y_labels.append(f"{idx}: {row['narration_id']} {row['narration']}")
        start = int(row["start_frame"]) - 1
        stop = int(row["stop_frame"]) - 1
        ax.hlines(y, start, stop, color="#9aa0a6", linewidth=7, alpha=0.45)
        sub_df = candidate_df[candidate_df["subaction_index"] == idx]
        failed = sub_df[sub_df["status"] != "keep"]
        kept = sub_df[sub_df["status"] == "keep"]
        ax.scatter(failed["frame_0_based"], [y] * len(failed), s=14, color="#d93025", alpha=0.55, label="failed candidates" if idx == 0 else None)
        ax.scatter(kept["frame_0_based"], [y] * len(kept), s=42, color="#188038", marker="o", label="success samples" if idx == 0 else None)
        for _, item in kept.iterrows():
            if pd.notna(item.get("ref_idx")):
                ax.annotate(
                    "",
                    xy=(item["frame_0_based"], y + 0.08),
                    xytext=(item["ref_idx"], y + 0.08),
                    arrowprops=dict(arrowstyle="->", color="#1a73e8", lw=1.0, alpha=0.75),
                )
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("0-based frame index")
    ax.set_title("Timeline: E13 candidates, successes, and reference arrows")
    ax.grid(True, axis="x", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_ref_gap_histogram(success_df: pd.DataFrame, path: Path) -> None:
    if success_df.empty or "ref_gap" not in success_df:
        return
    values = success_df["ref_gap"].dropna().astype(float)
    if values.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=min(20, max(5, int(values.nunique()))), color="#1a73e8", alpha=0.85)
    ax.set_title("E13 success ref_gap distribution")
    ax.set_xlabel("candidate frame - reference frame")
    ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_candidate_offset_histogram(success_df: pd.DataFrame, path: Path) -> None:
    if success_df.empty or "candidate_frame_offset_from_episode_start" not in success_df:
        return
    counts = success_df["candidate_frame_offset_from_episode_start"].fillna(-1).astype(int).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color="#188038")
    ax.set_title("E13 success candidate offset distribution")
    ax.set_xlabel("candidate offset from episode start")
    ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_gap_length_histogram(candidate_df: pd.DataFrame, path: Path) -> None:
    if candidate_df.empty or "gap_length_before_episode_start" not in candidate_df:
        return
    values = candidate_df["gap_length_before_episode_start"].dropna().astype(float)
    if values.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=min(30, max(5, int(values.nunique()))), color="#5f6368", alpha=0.85)
    ax.set_title("E13 pre-contact gap length distribution")
    ax.set_xlabel("gap length before episode start")
    ax.set_ylabel("candidate count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_reference_mode_ref_gap_histogram(success_df: pd.DataFrame, path: Path) -> None:
    if success_df.empty or "ref_gap" not in success_df or "reference_mode" not in success_df:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for mode, color in [("strict_global_no_hand", "#1a73e8"), ("crop_level_humanless", "#f29900")]:
        values = success_df[success_df["reference_mode"] == mode]["ref_gap"].dropna().astype(float)
        if not values.empty:
            ax.hist(values, bins=min(20, max(5, int(values.nunique()))), alpha=0.65, label=mode, color=color)
    ax.set_title("E13 success ref_gap by reference mode")
    ax.set_xlabel("candidate frame - reference frame")
    ax.set_ylabel("count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_overlay_contact_sheet(df: pd.DataFrame, path: Path, title: str, max_items: int = 36) -> None:
    view = df[df["status"] == "keep"].copy()
    if view.empty or "vrb_style_affordance" not in view:
        return
    view = view.head(max_items)
    images = []
    labels = []
    for _, row in view.iterrows():
        img_path = row.get("vrb_style_affordance")
        if not isinstance(img_path, str) or not Path(img_path).exists():
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(
            f"s{int(row['subaction_index'])} f{int(row['frame_0_based'])} "
            f"{row.get('reference_mode', '')}"
        )
    if not images:
        return

    thumb_w, thumb_h = 240, 180
    cols = min(4, len(images))
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.7))
    axes_arr = np.asarray(axes).reshape(-1)
    for ax in axes_arr:
        ax.axis("off")
    for ax, img, label in zip(axes_arr, images, labels):
        thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        ax.imshow(thumb)
        ax.set_title(label, fontsize=8)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_image_sheet(
    df: pd.DataFrame,
    path: Path,
    *,
    image_column: str,
    title: str,
    max_items: int = 36,
) -> None:
    view = df[df["status"] == "keep"].copy()
    if view.empty or image_column not in view:
        return
    view = view.head(max_items)
    images = []
    labels = []
    for _, row in view.iterrows():
        img_path = row.get(image_column)
        if not isinstance(img_path, str) or not Path(img_path).exists():
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        labels.append(
            f"s{int(row['subaction_index'])} f{int(row['frame_0_based'])} "
            f"gap={int(row['ref_gap']) if pd.notna(row.get('ref_gap')) else ''}"
        )
    if not images:
        return
    thumb_w, thumb_h = 320, 180
    cols = min(3, len(images))
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 2.6))
    axes_arr = np.asarray(axes).reshape(-1)
    for ax in axes_arr:
        ax.axis("off")
    for ax, img, label in zip(axes_arr, images, labels):
        thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        ax.imshow(thumb)
        ax.set_title(label, fontsize=8)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: Optional[int] = None) -> str:
    if df.empty:
        return "None\n"
    view = df[list(columns)].copy()
    if max_rows is not None:
        view = view.head(max_rows)
    view = view.where(pd.notna(view), "")
    lines = [
        "| " + " | ".join(str(col) for col in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("\n", " ") for col in columns) + " |")
    return "\n".join(lines)


def distribution_stats(series: pd.Series) -> Dict[str, Any]:
    values = series.dropna().astype(float)
    if values.empty:
        return {"count": 0, "mean": None, "q1": None, "median": None, "q3": None, "min": None, "max": None}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "q1": float(values.quantile(0.25)),
        "median": float(values.quantile(0.5)),
        "q3": float(values.quantile(0.75)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def build_sample_key_df(df: pd.DataFrame) -> pd.DataFrame:
    keep = df[df["status"] == "keep"].copy()
    if keep.empty:
        keep["sample_key"] = []
        return keep
    keep["sample_key"] = keep.apply(lambda row: f"{int(row['subaction_index'])}:{int(row['frame_0_based'])}:{row['hand']}", axis=1)
    return keep


def save_success_manifests(candidate_df: pd.DataFrame, output_root: Path) -> None:
    keep = candidate_df[candidate_df["status"] == "keep"].copy()
    manifest_cols = [
        "subaction_index",
        "narration_id",
        "narration",
        "verb",
        "noun",
        "episode_id_within_subaction",
        "episode_contact_start_frame_0_based",
        "frame_0_based",
        "candidate_frame_offset_from_episode_start",
        "hand",
        "reference_mode",
        "ref_idx",
        "ref_gap",
        "reference_gap",
        "reference_anchor_gap",
        "object_mask_snap_inside_count",
        "object_mask_snap_max_distance",
        "sample_dir",
        "contact_frame",
        "reference_frame",
        "contact_overlay",
        "projection_overlay_raw_vs_snapped",
        "homography_matches",
        "label_heatmap_overlay",
        "vrb_style_affordance",
        "problem3_full_overlay",
        "problem3_crop",
    ]
    keep[manifest_cols].to_csv(output_root / "successful_sample_manifest.csv", index=False)


def resolve_output_dir(root_name: str) -> Optional[Path]:
    candidates = [
        VRBREPRODUCTION_ROOT / "Outputs" / root_name,
        VRBREPRODUCTION_ROOT / "outputs" / root_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def write_comparison_files(candidate_df: pd.DataFrame, output_root: Path) -> None:
    e13_keep = build_sample_key_df(candidate_df)
    fallback_success = candidate_df[
        (candidate_df["status"] == "keep") & (candidate_df.get("reference_mode") == "crop_level_humanless")
    ].copy()
    for baseline_exp, baseline_root_name, out_name in [
        ("E4", "数据处理小批量测试前100个subaction_E4_episode_filter", "e13_new_success_samples_vs_e4.csv"),
        ("E5b", "数据处理小批量测试前100个subaction_E5b_reference_object_consistency", "e13_new_success_samples_vs_e5b.csv"),
        ("E13_x", "数据处理小批量测试前100个subaction_E13_x_new_contact_anchor_pre_contact_gap", "e13_new_success_samples_vs_e13_x.csv"),
    ]:
        baseline_root = resolve_output_dir(baseline_root_name)
        baseline_path = baseline_root / "candidate_diagnostics.csv" if baseline_root is not None else None
        if baseline_path is None or not baseline_path.exists():
            if baseline_exp == "E13_x":
                fallback_success.to_csv(output_root / out_name, index=False)
            else:
                pd.DataFrame().to_csv(output_root / out_name, index=False)
            continue
        baseline_df = pd.read_csv(baseline_path)
        if "experiment_id" in baseline_df.columns:
            baseline_df = baseline_df[baseline_df["experiment_id"] == baseline_exp]
        baseline_keep = build_sample_key_df(baseline_df)
        baseline_keys = set(baseline_keep.get("sample_key", []))
        added = e13_keep[~e13_keep["sample_key"].isin(baseline_keys)].copy()
        added.to_csv(output_root / out_name, index=False)

    e13_root = resolve_output_dir("数据处理小批量测试前100个subaction_E13_x_new_contact_anchor_pre_contact_gap")
    e13_path = e13_root / "candidate_diagnostics.csv" if e13_root is not None else None
    if e13_path is not None and e13_path.exists():
        e13_df = pd.read_csv(e13_path)
        if "experiment_id" in e13_df.columns:
            e13_df = e13_df[e13_df["experiment_id"] == "E13_x"]
        e13_keep = build_sample_key_df(e13_df)
        e13_keep_keys = set(e13_keep.get("sample_key", []))
        changed = e13_keep[~e13_keep["sample_key"].isin(set(e13_keep.get("sample_key", [])))].copy()
        changed.to_csv(output_root / "e13_removed_or_changed_samples_vs_e13_x.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_root / "e13_removed_or_changed_samples_vs_e13_x.csv", index=False)

    fallback_success.to_csv(output_root / "e13_fallback_success_samples.csv", index=False)

    fallback_fail = candidate_df[
        (candidate_df["status"] != "keep") & (candidate_df.get("fallback_reference_attempted").fillna(False))
    ].copy()
    if fallback_fail.empty:
        pd.DataFrame(columns=["fail_reason", "count"]).to_csv(output_root / "e13_fallback_failure_reasons.csv", index=False)
    else:
        fallback_fail["fallback_fail_bucket"] = fallback_fail["fail_reason"].fillna("unknown")
        fallback_fail["fallback_fail_bucket"].value_counts().rename_axis("fail_reason").reset_index(name="count").to_csv(
            output_root / "e13_fallback_failure_reasons.csv", index=False
        )


def write_markdown_report(
    *,
    config: E13Config,
    overview_df: pd.DataFrame,
    subaction_summary_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    candidate_plan_df: pd.DataFrame,
    frame_coverage: Dict[str, int],
) -> Path:
    output_root = config.output_root
    success_df = candidate_df[candidate_df["status"] == "keep"].copy()
    strict_success_df = success_df[success_df["reference_mode"] == "strict_global_no_hand"].copy()
    fallback_success_df = success_df[success_df["reference_mode"] == "crop_level_humanless"].copy()

    compare_rows = []
    baseline_specs = [
        ("E4", "数据处理小批量测试前100个subaction_E4_episode_filter"),
        ("E5b", "数据处理小批量测试前100个subaction_E5b_reference_object_consistency"),
        ("E13_x", "数据处理小批量测试前100个subaction_E13_x_new_contact_anchor_pre_contact_gap"),
    ]
    for experiment_id, root_name in baseline_specs:
        baseline_root = resolve_output_dir(root_name)
        overview_path = baseline_root / "experiment_overview.csv" if baseline_root is not None else None
        if overview_path is not None and overview_path.exists():
            ext_df = pd.read_csv(overview_path)
            if "experiment_id" in ext_df.columns:
                ext_df = ext_df[ext_df["experiment_id"] == experiment_id]
            if not ext_df.empty:
                compare_rows.append(ext_df.iloc[0].to_dict())
    if not any(row.get("experiment_id") == "E13_x" for row in compare_rows):
        compare_rows.append(
            {
                "experiment_id": "E13_x",
                "subactions_success": 9,
                "heatmap_pass": 25,
                "strict_ref_found": 46,
                "fallback_ref_found": 0,
                "homography_available": 34,
                "geometry_pass": 25,
                "main_failure": "no_pre_episode_reference_window",
                "description": "fallback to recorded results in learning note",
            }
        )
    compare_rows.append(overview_df.iloc[0].to_dict())
    compare_overview = pd.DataFrame(compare_rows)
    for column in [
        "strict_ref_found",
        "fallback_ref_found",
        "fallback_ref_used",
        "rescued_from_no_strict_humanless",
        "heatmap_pass",
        "subactions_success",
        "main_failure",
    ]:
        if column not in compare_overview.columns:
            compare_overview[column] = 0

    top_failures = (
        candidate_df[candidate_df["status"] != "keep"]["fail_reason"]
        .fillna("unknown")
        .value_counts()
        .rename_axis("fail_reason")
        .reset_index(name="count")
    )
    candidate_offset_dist = (
        success_df["candidate_frame_offset_from_episode_start"].value_counts(dropna=False).sort_index().reset_index()
        if not success_df.empty
        else pd.DataFrame(columns=["candidate_frame_offset_from_episode_start", "count"])
    )
    if not candidate_offset_dist.empty:
        candidate_offset_dist.columns = ["candidate_frame_offset_from_episode_start", "count"]

    focus_df = candidate_df[candidate_df["narration"].isin(["open freezer", "open drawer", "close freezer", "close door", "close cupboard"])].copy()
    focus_df = focus_df[
        [
            "subaction_index",
            "narration_id",
            "narration",
            "candidate_index",
            "episode_id_within_subaction",
            "frame_0_based",
            "candidate_frame_offset_from_episode_start",
            "hand",
            "status",
            "fail_reason",
            "reference_mode",
            "ref_idx",
            "reference_anchor_gap",
            "gap_length_before_episode_start",
        ]
    ]

    lines = [
        "# 数据处理小批量测试前100个subaction E13 pre-contact-gap crop fallback - 诊断实验报告",
        "",
        f"- video_id: `{config.video_id}`",
        f"- subactions: first `{config.num_subactions}` annotations of this video",
        f"- output root: `{output_root}`",
        f"- max ref backtrack: `{config.max_ref_backtrack}` frames",
        f"- candidate offsets from episode start: `{list(config.early_candidate_offsets)}`",
        f"- crop fallback: `{config.crop_size}x{config.crop_size}`, hand overlap <= `{config.crop_hand_overlap_max}`",
        f"- annotated frame intervals total: `{frame_coverage['total_annotated_frames']}` frames",
        f"- unique covered frames after overlap removal: `{frame_coverage['unique_covered_frames']}` frames",
        "",
        "## E13 定义",
        "",
        "- 数据单位保持为 `new contact episode`。",
        "- candidate 只允许 `t0`, `t0+1`, `t0+2`。",
        "- reference 搜索锚定 `t0`，语义窗口为当前 episode 前的 pre-contact gap。",
        "- strict global no-hand 找不到时，才允许在同一 effective pre-contact window 内做 crop-level humanless fallback。",
        "- `no_pre_episode_reference_window` 不允许被救回。",
        "",
        "## Funnel",
        "",
        markdown_table(
            overview_df,
            [
                "experiment_id",
                "raw_candidate_total",
                "episode_filtered_candidate_total",
                "deep_run_candidate_total",
                "cell2_pass",
                "strict_ref_found",
                "fallback_ref_found",
                "homography_available",
                "geometry_pass",
                "heatmap_pass",
                "subactions_success",
                "main_failure",
            ],
        ),
        "",
        "## E4 / E5b / E13-x / E13 对比",
        "",
        markdown_table(
            compare_overview,
            [
                "experiment_id",
                "subactions_success",
                "strict_ref_found",
                "fallback_ref_found",
                "heatmap_pass",
                "homography_available",
                "geometry_pass",
                "main_failure",
            ],
        ),
        "",
        "## 核心回答",
        "",
        f"- strict success count: `{len(strict_success_df)}`",
        f"- fallback success count: `{len(fallback_success_df)}`",
        f"- total success count: `{len(success_df)}`",
        f"- no_pre_episode_reference_window_not_rescued: `{int(candidate_df.get('no_pre_episode_reference_window_not_rescued', pd.Series(dtype=bool)).fillna(False).astype(bool).sum())}`",
        f"- fallback_reference_in_pre_contact_gap all true: `{bool(fallback_success_df['fallback_reference_in_pre_contact_gap'].fillna(False).all()) if not fallback_success_df.empty else True}`",
        f"- fallback_ref_before_episode_start all true: `{bool(fallback_success_df['fallback_ref_before_episode_start'].fillna(False).all()) if not fallback_success_df.empty else True}`",
        "",
        "## success 分布",
        "",
        f"- strict ref_gap stats: `{distribution_stats(strict_success_df['ref_gap']) if 'ref_gap' in strict_success_df else distribution_stats(pd.Series(dtype=float))}`",
        f"- fallback ref_gap stats: `{distribution_stats(fallback_success_df['ref_gap']) if 'ref_gap' in fallback_success_df else distribution_stats(pd.Series(dtype=float))}`",
        f"- strict reference_anchor_gap stats: `{distribution_stats(strict_success_df['reference_anchor_gap']) if 'reference_anchor_gap' in strict_success_df else distribution_stats(pd.Series(dtype=float))}`",
        f"- fallback reference_anchor_gap stats: `{distribution_stats(fallback_success_df['reference_anchor_gap']) if 'reference_anchor_gap' in fallback_success_df else distribution_stats(pd.Series(dtype=float))}`",
        f"- gap_length_before_episode_start stats: `{distribution_stats(candidate_df['gap_length_before_episode_start']) if 'gap_length_before_episode_start' in candidate_df else distribution_stats(pd.Series(dtype=float))}`",
        "",
        markdown_table(candidate_offset_dist, ["candidate_frame_offset_from_episode_start", "count"]),
        "",
        "## 主要失败原因",
        "",
        markdown_table(top_failures, ["fail_reason", "count"], max_rows=20),
        "",
        "## 重点动作类别状态",
        "",
        markdown_table(
            focus_df,
            [
                "subaction_index",
                "narration_id",
                "narration",
                "candidate_index",
                "episode_id_within_subaction",
                "frame_0_based",
                "candidate_frame_offset_from_episode_start",
                "hand",
                "status",
                "fail_reason",
                "reference_mode",
                "ref_idx",
                "reference_anchor_gap",
                "gap_length_before_episode_start",
            ],
            max_rows=120,
        ),
        "",
        "## 图表",
        "",
        "- pipeline funnel: `charts/pipeline_funnel.png`",
        "- failure reasons: `charts/failure_reasons.png`",
        "- timeline: `charts/timeline_E13.png`",
        "- success ref_gap histogram: `charts/ref_gap_success_hist.png`",
        "- success candidate offset histogram: `charts/candidate_offset_success_hist.png`",
        "- gap length histogram: `charts/gap_length_hist.png`",
        "- success overlay contact sheet: `charts/success_overlay_all_contact_sheet.png`",
        "- fallback success overlay contact sheet: `charts/fallback_success_overlay_contact_sheet.png`",
        "",
        "## 明细文件",
        "",
        "- experiment JSON: `experiment_results.json`",
        "- overview CSV: `experiment_overview.csv`",
        "- subaction summary CSV: `subaction_summary.csv`",
        "- candidate diagnostics CSV: `candidate_diagnostics.csv`",
        "- candidate plan CSV: `candidate_plan.csv`",
        "- successful sample manifest: `successful_sample_manifest.csv`",
        "- new success vs E13-x: `e13_new_success_samples_vs_e13_x.csv`",
        "- removed/changed vs E13-x: `e13_removed_or_changed_samples_vs_e13_x.csv`",
        "- fallback success samples: `e13_fallback_success_samples.csv`",
        "- fallback failure reasons: `e13_fallback_failure_reasons.csv`",
        "- pipeline outputs: `experiments/E13/pipeline_outputs/`",
    ]
    report_path = output_root / "summary.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def save_manual_audit_plan(candidate_df: pd.DataFrame, output_root: Path) -> Path:
    rng = np.random.default_rng(13)
    success_df = candidate_df[candidate_df["status"] == "keep"].copy()
    failure_df = candidate_df[candidate_df["status"] != "keep"].copy()
    rows = []
    for split_name, df, n in [("success", success_df, 30), ("failure", failure_df, 20)]:
        if df.empty:
            continue
        take = min(n, len(df))
        sampled_idx = rng.choice(df.index.to_numpy(), size=take, replace=False)
        for _, row in df.loc[sampled_idx].iterrows():
            rows.append(
                {
                    "audit_split": split_name,
                    "subaction_index": row.get("subaction_index"),
                    "narration_id": row.get("narration_id"),
                    "narration": row.get("narration"),
                    "frame_0_based": row.get("frame_0_based"),
                    "hand": row.get("hand"),
                    "status": row.get("status"),
                    "fail_reason": row.get("fail_reason"),
                    "sample_dir": row.get("sample_dir"),
                    "projection_overlay_raw_vs_snapped": row.get("projection_overlay_raw_vs_snapped"),
                    "vrb_style_affordance": row.get("vrb_style_affordance"),
                    "manual_label": "",
                    "allowed_labels": "accurate_same_part|same_object_but_shifted|wrong_object|background_or_empty|unclear",
                    "notes": "",
                }
            )
    path = output_root / "manual_audit_sample_plan.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_e13_markdown_report(
    *,
    config: E13Config,
    overview_df: pd.DataFrame,
    subaction_summary_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    candidate_plan_df: pd.DataFrame,
    frame_coverage: Dict[str, int],
    manual_audit_plan_path: Path,
) -> Path:
    del candidate_plan_df, frame_coverage
    output_root = config.output_root
    overview = overview_df.iloc[0].to_dict() if not overview_df.empty else {}
    success_df = candidate_df[candidate_df["status"] == "keep"].copy()
    failed_df = candidate_df[candidate_df["status"] != "keep"].copy()
    top_failures = (
        failed_df["fail_reason"].fillna("unknown").value_counts().rename_axis("fail_reason").reset_index(name="count")
        if not failed_df.empty
        else pd.DataFrame(columns=["fail_reason", "count"])
    )
    methods = (
        candidate_df["homography_feature_methods"].fillna("none").value_counts().rename_axis("feature_methods").reset_index(name="count")
        if "homography_feature_methods" in candidate_df
        else pd.DataFrame(columns=["feature_methods", "count"])
    )
    baselines = pd.DataFrame(
        [
            {"experiment": "E6b homography", "reported_success": "18/100", "note": "task baseline"},
            {"experiment": "E6c conservative homography", "reported_success": "11/100", "note": "task baseline"},
            {"experiment": "E9 part-aware LK", "reported_success": "74/100", "note": "non-CoTracker upper reference"},
            {"experiment": "E10 CoTracker", "reported_success": "79/100", "note": "not used by E13"},
            {"experiment": "E12 clean reference + CoTracker", "reported_success": "79/100", "note": "not used by E13"},
            {
                "experiment": "E13 engineered homography",
                "reported_success": f"{int(overview.get('subactions_success', 0))}/{int(overview.get('subactions_total', len(subaction_summary_df)))}",
                "note": "automatic gate, before human audit",
            },
        ]
    )
    audit_status = "manual_audit_pending"
    if (output_root / "manual_audit_completed.csv").exists():
        audit_status = "manual_audit_file_present"

    lines = [
        "# E13 engineered homography projection",
        "",
        "## Result",
        "",
        f"- subactions evaluated: {int(overview.get('subactions_total', len(subaction_summary_df)))}",
        f"- automatic gate successes: {int(overview.get('subactions_success', 0))}/{int(overview.get('subactions_total', len(subaction_summary_df)))}",
        f"- successful candidates / heatmap pass: {int(overview.get('heatmap_pass', 0))}",
        f"- reference policy: nearest pre-contact frame within max_ref_backtrack={config.max_ref_backtrack}",
        f"- homography gate: good_matches>={config.homography_min_good_matches}, inliers>={config.homography_min_inliers}, inlier_ratio>={config.homography_min_inlier_ratio}",
        f"- object gate: raw projected GMM means snapped to reference object mask, max snap distance <= {config.snap_max_distance_px}px",
        f"- audit status: {audit_status}",
        "",
        "E13 does not use CoTracker. The count above is automatic gate success unless `manual_audit_completed.csv` is filled later.",
        "",
        "## Baseline comparison",
        "",
        markdown_table(baselines, ["experiment", "reported_success", "note"]),
        "",
        "Interpretation rule: 50+/100 supports homography as a cheap first pass; 20-30/100 points toward a cascade with LK/CoTracker fallback.",
        "",
        "## Pipeline overview",
        "",
        markdown_table(
            overview_df,
            [
                "subactions_total",
                "subactions_success",
                "raw_candidate_total",
                "deep_run_candidate_total",
                "cell2_pass",
                "ref_found",
                "homography_available",
                "geometry_pass",
                "heatmap_pass",
                "main_failure",
            ],
        ),
        "",
        "## Homography diagnostics",
        "",
        markdown_table(methods, ["feature_methods", "count"], max_rows=10),
        "",
        "## Top failure reasons",
        "",
        markdown_table(top_failures, ["fail_reason", "count"], max_rows=15),
        "",
        "## Manual audit",
        "",
        f"A random audit template was written to `{manual_audit_plan_path.name}`: up to 30 automatic successes and 20 failures.",
        "Allowed labels: `accurate_same_part`, `same_object_but_shifted`, `wrong_object`, `background_or_empty`, `unclear`.",
        "",
        "Until that template is manually filled, E13's reported rate is automatic gate success, not human-usable success.",
        "",
        "## Required artifacts",
        "",
        "- `experiment_overview.csv`",
        "- `subaction_summary.csv`",
        "- `candidate_diagnostics.csv`",
        "- `candidate_plan.csv`",
        "- `successful_sample_manifest.csv`",
        "- `experiment_results.json`",
        "- `charts/e13_pipeline_funnel.png`",
        "- `charts/e13_failure_reasons.png`",
        "- `charts/e13_success_contact_sheet_page_001.png`",
        "- `charts/e13_homography_match_sheet.png`",
    ]
    path = output_root / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def empty_subaction_record(
    *,
    row: pd.Series,
    start_0: int,
    stop_0: int,
    raw_candidate_count: int,
    subaction_new_contact_count: int,
    episode_decision: str,
    episode_reason: str,
) -> Dict[str, Any]:
    fail_reason = "continuation_contact_run" if episode_decision == "continuation_contact_run" else "no_smoothed_contact_in_subaction"
    return {
        "experiment_id": EXPERIMENT["experiment_id"],
        "experiment_name": EXPERIMENT["name"],
        "contact_strategy": EXPERIMENT["contact_strategy"],
        "reference_strategy": EXPERIMENT["reference_strategy"],
        "subaction_index": int(row.name),
        "narration_id": row["narration_id"],
        "video_id": row["video_id"],
        "narration": row["narration"],
        "verb": row["verb"],
        "noun": row["noun"],
        "all_nouns": row.get("all_nouns"),
        "start_frame_1_based": int(row["start_frame"]),
        "stop_frame_1_based": int(row["stop_frame"]),
        "clipped_start_frame_0_based": int(start_0),
        "clipped_stop_frame_0_based": int(stop_0),
        "candidate_index": None,
        "frame_0_based": None,
        "frame_1_based": None,
        "hand": None,
        "status": "discard",
        "passed_stage": "episode_filter" if episode_decision == "continuation_contact_run" else "no_candidate",
        "failed_stage": "episode_filter" if episode_decision == "continuation_contact_run" else "subaction_contact",
        "fail_reason": fail_reason,
        "contact_points": 0,
        "ref_idx": None,
        "ref_frame_1_based": None,
        "ref_gap": None,
        "reference_anchor_gap": None,
        "trajectory_points": 0,
        "missing_trajectory_count": None,
        "heatmap_mode": None,
        "sample_score": None,
        "sample_dir": None,
        "reference_mode": None,
        "reference_reason": None,
        "strict_reference_available": False,
        "fallback_reference_used": False,
        "fallback_reference_attempted": False,
        "fallback_reference_in_pre_contact_gap": False,
        "fallback_ref_before_episode_start": False,
        "fallback_rescued_from_no_strict_humanless": False,
        "no_pre_episode_reference_window_not_rescued": False,
        "rescued_from_no_strict_humanless": False,
        "crop_bbox_150": None,
        "crop_hand_overlap_ratio": None,
        "object_hand_overlap_ratio": None,
        "fallback_attempted_refs": 0,
        "fallback_reject_reasons": None,
        "fallback_precheck_homography_min_inlier_ratio": None,
        "fallback_precheck_homography_min_inliers": None,
        "fallback_precheck_homography_min_good_matches": None,
        "pre_contact_window_nonempty": None,
        "episode_contact_start_frame_0_based": None,
        "episode_contact_start_frame_1_based": None,
        "episode_contact_end_frame_0_based": None,
        "candidate_frame_offset_from_episode_start": None,
        "reference_anchor_frame_0_based": None,
        "pre_contact_gap_start_frame_0_based": None,
        "pre_contact_gap_end_frame_0_based": None,
        "gap_length_before_episode_start": None,
        "raw_semantic_window_start_0_based": None,
        "raw_semantic_window_end_0_based": None,
        "effective_reference_window_start_0_based": None,
        "effective_reference_window_end_0_based": None,
        "max_ref_backtrack": None,
        "window_truncated_by_max_backtrack": None,
        "reference_search_window_start_0_based": None,
        "reference_search_window_end_0_based": None,
        "ref_before_episode_start": None,
        "is_candidate_at_episode_start": None,
        "is_candidate_near_episode_start": None,
        "episode_contact_run_length": None,
        "subaction_new_contact_count": int(subaction_new_contact_count),
        "candidate_is_episode_representative": None,
        "episode_id_within_subaction": None,
        "episode_start_hand": None,
        "episode_start_reason": None,
        "reference_object_consistency_pass": None,
        "reference_object_consistency_reason": None,
        "reference_object_bbox": None,
        "reference_object_bbox_score": None,
        "reference_object_iou": None,
        "reference_object_center_distance": None,
        "projected_object_bbox": None,
        "projected_object_area": None,
        "projected_contact_to_reference_object_distance": None,
        "projected_contact_means_inside_reference_object_count": None,
        "reference_object_candidate_count": None,
        "is_articulated_object_candidate": bool(is_articulated_object_candidate(row["noun"], row.get("all_nouns"), row["narration"])),
        "articulated_object_policy": "record_only",
        "episode_decision": episode_decision,
        "episode_final_label": episode_decision,
        "episode_id": None,
        "episode_reason": episode_reason,
        "episode_object_key": str(row["noun"]).strip().lower(),
        "episode_noun_tokens": str(parse_all_nouns_field(row.get("all_nouns"))),
        "episode_prev_subaction_index": None,
        "episode_prev_narration_id": None,
        "episode_prev_last_contact_frame": None,
        "episode_gap_start_frame": None,
        "episode_gap_stop_frame": None,
        "episode_no_contact_streak": None,
        "episode_no_hand_streak": None,
        "episode_raw_candidate_count": int(raw_candidate_count),
        "episode_filtered_candidate_count": int(subaction_new_contact_count),
        "min_ref_frame_idx": None,
        "reference_frame": None,
        "label_heatmap_npy": None,
        "label_heatmap_png": None,
        "label_heatmap_overlay": None,
        "vrb_style_affordance": None,
        "heatmap_shape": None,
        "arrow_generated": None,
        "problem3_full_overlay": None,
        "problem3_crop": None,
        "homography_pairs": None,
        "homography_min_inlier_ratio": None,
        "homography_min_inliers": None,
        "homography_min_good_matches": None,
        "homography_fail_reasons": None,
    }


def run_e13_experiment(config: Optional[E13Config] = None) -> Dict[str, Any]:
    config = config or E13Config()
    config.output_root.mkdir(parents=True, exist_ok=True)
    (VRBREPRODUCTION_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
    charts_dir = config.output_root / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    detections = load_detections(str(config.hoa_pkl))
    subactions = get_subactions(config)
    frame_coverage = compute_subaction_frame_coverage(subactions)
    left_binary, right_binary = build_contact_arrays(detections)
    all_runs = build_new_contact_runs(left_binary, right_binary)

    contact_config = ContactExtractionConfig(
        use_object_mask=True,
        use_object_boundary=True,
        boundary_distance_px=6.0,
        project_points_to_object_boundary=True,
    )
    problem3_runner = EngineeredHomographyRunnerE13(detections, config.image_dir, config)

    all_records: List[Dict[str, Any]] = []
    candidate_plan_rows: List[Dict[str, Any]] = []

    print(f"Running {EXPERIMENT['experiment_id']}: {EXPERIMENT['description']}")
    print(f"Total frames: {len(detections)}")
    print(f"Selected subactions: {len(subactions)}")
    print(f"Annotated frame intervals total: {frame_coverage['total_annotated_frames']}")
    print(f"Unique covered frames: {frame_coverage['unique_covered_frames']}")

    for idx, row in subactions.iterrows():
        start_0, stop_0 = subaction_bounds(row, len(detections))
        raw_candidates = contact_frame_candidates_for_subaction(row, left_binary, right_binary, len(detections))
        sub_runs, episode_decision, episode_reason = runs_for_subaction(
            row,
            all_runs,
            left_binary,
            right_binary,
            len(detections),
            config.early_candidate_offsets,
            config.max_ref_backtrack,
        )
        flattened_candidates = []
        for run in sub_runs:
            flattened_candidates.extend(run["candidates"])

        candidate_plan_rows.append(
            {
                "experiment_id": EXPERIMENT["experiment_id"],
                "subaction_index": int(idx),
                "narration_id": row["narration_id"],
                "narration": row["narration"],
                "verb": row["verb"],
                "noun": row["noun"],
                "all_nouns": row.get("all_nouns"),
                "start_frame_0_based": int(start_0),
                "stop_frame_0_based": int(stop_0),
                "raw_candidate_count": int(len(raw_candidates)),
                "episode_filtered_candidate_count": int(len(sub_runs)),
                "deep_run_candidate_count": int(len(flattened_candidates)),
                "episode_decision": episode_decision,
                "episode_reason": episode_reason,
                "subaction_new_contact_count": int(len(sub_runs)),
                "episode_start_frames_0_based": ";".join(str(int(run["start_frame_0_based"])) for run in sub_runs),
                "episode_hands": ";".join(str(run["hand"]) for run in sub_runs),
            }
        )

        print(
            f"E13 subaction {idx:02d} {row['narration_id']} | {row['narration']} | "
            f"raw_candidates={len(raw_candidates)} | new_contact_runs={len(sub_runs)} | "
            f"deep_run={len(flattened_candidates)} | episode={episode_decision}"
        )

        if not flattened_candidates:
            all_records.append(
                empty_subaction_record(
                    row=row,
                    start_0=start_0,
                    stop_0=stop_0,
                    raw_candidate_count=len(raw_candidates),
                    subaction_new_contact_count=len(sub_runs),
                    episode_decision=episode_decision,
                    episode_reason=episode_reason,
                )
            )
            continue

        for candidate_index, candidate in enumerate(flattened_candidates):
            record = run_one_candidate(
                row=row,
                candidate=candidate,
                candidate_index=candidate_index,
                detections=detections,
                contact_config=contact_config,
                problem3_runner=problem3_runner,
                config=config,
                subaction_new_contact_count=len(sub_runs),
                raw_candidate_count=len(raw_candidates),
            )
            all_records.append(record)

    candidate_df = pd.DataFrame(all_records)
    candidate_plan_df = pd.DataFrame(candidate_plan_rows)
    subaction_summary_df = summarize_subactions(candidate_df, subactions, candidate_plan_df)
    overview_df = build_overview(candidate_df, subaction_summary_df, candidate_plan_df)

    experiment_json = config.output_root / "experiment_results.json"
    experiment_json.write_text(
        json.dumps(
            _json_safe(
                {
                    "video_id": config.video_id,
                    "num_subactions": config.num_subactions,
                    "max_ref_backtrack": config.max_ref_backtrack,
                    "early_candidate_offsets": list(config.early_candidate_offsets),
                    "crop_size": config.crop_size,
                    "crop_hand_overlap_max": config.crop_hand_overlap_max,
                    "homography_ratio_test": config.homography_ratio_test,
                    "homography_ransac_reproj_threshold": config.homography_ransac_reproj_threshold,
                    "homography_min_good_matches": config.homography_min_good_matches,
                    "homography_min_inliers": config.homography_min_inliers,
                    "homography_min_inlier_ratio": config.homography_min_inlier_ratio,
                    "dynamic_bbox_expand_ratio": config.dynamic_bbox_expand_ratio,
                    "snap_max_distance_px": config.snap_max_distance_px,
                    "frame_coverage": frame_coverage,
                    "experiment": EXPERIMENT,
                    "overview": overview_df.to_dict(orient="records"),
                    "subaction_summary": subaction_summary_df.to_dict(orient="records"),
                    "candidate_diagnostics": candidate_df.to_dict(orient="records"),
                    "candidate_plan": candidate_plan_df.to_dict(orient="records"),
                    "problem3_cache_stats": dict(problem3_runner.cache_stats),
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    overview_df.to_csv(config.output_root / "experiment_overview.csv", index=False)
    subaction_summary_df.to_csv(config.output_root / "subaction_summary.csv", index=False)
    candidate_df.to_csv(config.output_root / "candidate_diagnostics.csv", index=False)
    candidate_plan_df.to_csv(config.output_root / "candidate_plan.csv", index=False)

    save_success_manifests(candidate_df, config.output_root)
    write_comparison_files(candidate_df, config.output_root)
    save_funnel_chart(overview_df, charts_dir / "e13_pipeline_funnel.png")
    save_bar_chart_failure_reasons(candidate_df, charts_dir / "e13_failure_reasons.png")
    save_timeline_chart(candidate_df, subactions, charts_dir / "timeline_E13.png")
    success_df = candidate_df[candidate_df["status"] == "keep"].copy()
    save_ref_gap_histogram(success_df, charts_dir / "ref_gap_success_hist.png")
    save_candidate_offset_histogram(success_df, charts_dir / "candidate_offset_success_hist.png")
    save_gap_length_histogram(candidate_df, charts_dir / "gap_length_hist.png")
    save_reference_mode_ref_gap_histogram(success_df, charts_dir / "ref_gap_success_by_reference_mode_hist.png")
    save_overlay_contact_sheet(success_df, charts_dir / "e13_success_contact_sheet_page_001.png", "E13 success overlays")
    save_image_sheet(
        success_df,
        charts_dir / "e13_homography_match_sheet.png",
        image_column="homography_matches",
        title="E13 homography raw/snapped match sheet",
    )
    manual_audit_plan_path = save_manual_audit_plan(candidate_df, config.output_root)
    summary_path = write_e13_markdown_report(
        config=config,
        overview_df=overview_df,
        subaction_summary_df=subaction_summary_df,
        candidate_df=candidate_df,
        candidate_plan_df=candidate_plan_df,
        frame_coverage=frame_coverage,
        manual_audit_plan_path=manual_audit_plan_path,
    )

    print("\nExperiment overview")
    print(
        overview_df[
            [
                "experiment_id",
                "subactions_success",
                "raw_candidate_total",
                "episode_filtered_candidate_total",
                "deep_run_candidate_total",
                "cell2_pass",
                "ref_found",
                "homography_available",
                "geometry_pass",
                "heatmap_pass",
                "main_failure",
            ]
        ]
    )
    print(f"Summary: {summary_path}")
    print(f"Candidate diagnostics CSV: {config.output_root / 'candidate_diagnostics.csv'}")
    print(f"Pipeline outputs: {config.output_root / 'experiments'}")
    print(f"Problem3 cache stats: {dict(problem3_runner.cache_stats)}")

    return {
        "config": config,
        "overview": overview_df,
        "subaction_summary": subaction_summary_df,
        "candidate_diagnostics": candidate_df,
        "candidate_plan": candidate_plan_df,
        "frame_coverage": frame_coverage,
        "output_root": config.output_root,
        "summary_path": summary_path,
    }


if __name__ == "__main__":
    run_e13_experiment()
