"""E17 experiment: near-contact reference diagnostic baseline."""

from __future__ import annotations

import ast
import argparse
import io
import json
import os
import sys
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from epic_kitchens.hoa.types import HandState
from scipy.signal import savgol_filter

from .contact_point_utils import (
    ContactExtractionConfig,
    get_active_hand_bbox,
    get_valid_object_bboxes,
    normalized_bbox_to_pixels,
    select_active_object_bbox,
)
from .e14_clean_pre_contact_reference import Problem3CachedRunnerE14
from .e5b_reference_object_consistency import bbox_center_distance, bbox_iou, polygon_to_bbox
from .label_heatmap_utils import (
    build_label_heatmaps,
    draw_vrb_style_affordance_overlay,
    merge_label_heatmaps,
    save_label_heatmap_outputs,
    transform_covariances_by_homography,
)
from .pipeline_retention import _json_safe, fit_cell2_contact_gmm, run_cell4_heatmap_gate
from .problem3_utils import (
    build_dynamic_mask,
    compute_centered_crop_bbox,
    compute_crop_hand_overlap_ratio,
    compute_pairwise_homography,
    count_points_near_polygon,
    draw_problem3_crop_overlay,
    draw_problem3_full_overlay,
    find_strict_humanless_frame,
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
class E17Config:
    video_id: str = "P01_109"
    num_subactions: int = 100
    reference_offsets: Tuple[int, ...] = (5, 6, 7, 8, 9, 10)
    reference_min_gap: int = 5
    reference_max_gap: int = 10
    max_ref_backtrack: int = 10
    early_candidate_offsets: Tuple[int, ...] = (0,)
    crop_size: int = 150
    crop_hand_overlap_max: Optional[float] = None
    hand_object_iou_max: Optional[float] = None
    min_hand_object_center_distance_px: Optional[float] = None
    output_root: Path = VRBREPRODUCTION_ROOT / "outputs" / "e17"
    annotation_csv: Path = VRBREPRODUCTION_ROOT / "data" / "annotations" / "epic-kitchens-100-annotations" / "EPIC_100_train.csv"
    hoa_pkl: Path = VRBREPRODUCTION_ROOT / "data" / "P01_109.pkl"
    image_dir: Path = VRBREPRODUCTION_ROOT / "data" / "P01_109_frames"


EXPERIMENT = {
    "experiment_id": "E17",
    "name": "E17_near_contact_reference",
    "contact_strategy": "first_new_contact_episode_first_timestep",
    "reference_strategy": "near_contact_pre_frame_offset_5_to_10",
    "reference_fallback": "none",
    "description": (
        "每个 subaction 只取第一个 smoothed contact 0->1 episode 的 first contact timestep；"
        "reference 直接使用 first contact frame 前 5-10 帧内最近可用帧；"
        "humanless、active-hand-invisible、E14 clean gate 只作为 shadow diagnostics，不进入主筛选。"
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


def get_subactions(config: E17Config) -> pd.DataFrame:
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
    hand: str,
    left_binary: np.ndarray,
    right_binary: np.ndarray,
) -> int:
    """Return the active-hand contiguous no-contact gap immediately before t0."""
    idx = int(run_start) - 1
    if idx < 0:
        return int(run_start)
    if str(hand).lower() == "left":
        contact_active = np.asarray(left_binary, dtype=int) == 1
    else:
        contact_active = np.asarray(right_binary, dtype=int) == 1
    if contact_active[idx]:
        return int(run_start)
    while idx - 1 >= 0 and not contact_active[idx - 1]:
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
        runs_in_subaction = [sorted(runs_in_subaction, key=lambda item: (item["start_frame_0_based"], item["hand"]))[0]]
        decision = "new_contact_episode"
        reason = "first_smoothed_contact_run_start_within_subaction"
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
            hand=run["hand"],
            left_binary=left_binary,
            right_binary=right_binary,
        )
        pre_gap_end = run_start - 1
        gap_length = max(0, pre_gap_end - pre_gap_start + 1)
        raw_start = int(run_start) - 10
        raw_end = int(run_start) - 5
        effective_start = raw_start
        effective_end = raw_end
        window_truncated = False
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


def hand_state_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    return getattr(value, "name", str(value))


def hand_bbox_and_meta(
    frame_det: Any,
    active_hand: str,
    image_shape: Tuple[int, int],
    score_threshold: float = 0.5,
) -> Dict[str, Any]:
    h, w = image_shape[:2]
    meta = {
        "visible": False,
        "score": None,
        "state": None,
        "bbox_norm": None,
        "bbox_px": None,
    }
    if not hasattr(frame_det, "hands"):
        return meta
    for hand in frame_det.hands:
        if hand.side.name.lower() != active_hand.lower():
            continue
        meta["score"] = float(hand.score)
        meta["state"] = hand_state_name(hand.state)
        if hand.score >= score_threshold:
            bbox_norm = [hand.bbox.left, hand.bbox.top, hand.bbox.right, hand.bbox.bottom]
            meta["visible"] = True
            meta["bbox_norm"] = bbox_norm
            meta["bbox_px"] = normalized_bbox_to_pixels(bbox_norm, w, h)
        break
    return meta


def preselect_near_contact_reference(
    *,
    detections,
    image_dir: Path,
    episode_start: int,
    t_contact: int,
    active_hand: str,
    left_binary: np.ndarray,
    right_binary: np.ndarray,
    reference_offsets: Sequence[int],
    search_start: int,
    search_end: int,
) -> Dict[str, Any]:
    search_start = int(search_start)
    search_end = int(search_end)
    episode_start = int(episode_start)
    active_hand = str(active_hand).lower()
    offsets = tuple(int(offset) for offset in reference_offsets)
    shadow_start = max(0, search_start)
    shadow_end = int(search_end)

    strict_ref_idx = None
    strict_debug: Dict[str, Any] = {
        "rejected_isolated_frames": [],
        "streak_lengths": {},
        "final_streak_start": None,
        "final_streak_length": 0,
    }
    if shadow_end >= shadow_start:
        strict_ref_idx, strict_debug = find_strict_humanless_frame(
            detections,
            shadow_end + 1,
            score_threshold=0.5,
            min_no_hand_streak=3,
            min_frame_idx=shadow_start,
        )

    active_hand_invisible_ref_idx = None
    if shadow_end >= shadow_start:
        for ref_candidate in range(shadow_end, shadow_start - 1, -1):
            ref_path = Path(image_dir) / f"frame_{ref_candidate + 1:010d}.jpg"
            ref_img = cv2.imread(str(ref_path))
            if ref_img is None:
                continue
            hand_meta = hand_bbox_and_meta(detections[int(ref_candidate)], active_hand, ref_img.shape, score_threshold=0.5)
            if not bool(hand_meta["visible"]):
                active_hand_invisible_ref_idx = int(ref_candidate)
                break

    reject_reasons: Counter = Counter()
    selected_ref_idx = None
    selected_offset = None
    attempted = 0
    for offset in offsets:
        ref_candidate = episode_start - int(offset)
        if ref_candidate < 0:
            reject_reasons["ref_idx_lt_zero"] += 1
            continue
        if ref_candidate >= len(detections):
            reject_reasons["ref_idx_out_of_detection_range"] += 1
            continue
        ref_path = Path(image_dir) / f"frame_{ref_candidate + 1:010d}.jpg"
        if not ref_path.exists():
            reject_reasons["reference_frame_missing"] += 1
            continue
        attempted += 1
        selected_ref_idx = int(ref_candidate)
        selected_offset = int(offset)
        break

    detail: Dict[str, Any] = {
        "ref_idx": selected_ref_idx,
        "ref_frame_1_based": int(selected_ref_idx) + 1 if selected_ref_idx is not None else None,
        "ref_gap": int(t_contact) - int(selected_ref_idx) if selected_ref_idx is not None else None,
        "reference_anchor_gap": int(episode_start) - int(selected_ref_idx) if selected_ref_idx is not None else None,
        "reference_offset": selected_offset,
        "reference_offsets": ",".join(str(offset) for offset in offsets),
        "reference_mode": "near_contact_pre_frame" if selected_ref_idx is not None else "no_near_contact_reference",
        "reference_reason": (
            "closest_pre_contact_offset_in_5_10" if selected_ref_idx is not None else "no_near_contact_reference"
        ),
        "reference_attempted_refs": int(attempted),
        "reference_reject_reasons": dict(reject_reasons),
        "strict_ref_idx_shadow": int(strict_ref_idx) if strict_ref_idx is not None else None,
        "strict_reference_anchor_gap_shadow": (
            int(episode_start - int(strict_ref_idx)) if strict_ref_idx is not None else None
        ),
        "active_hand_invisible_ref_idx_shadow": (
            int(active_hand_invisible_ref_idx) if active_hand_invisible_ref_idx is not None else None
        ),
        "active_hand_invisible_reference_anchor_gap_shadow": (
            int(episode_start - int(active_hand_invisible_ref_idx))
            if active_hand_invisible_ref_idx is not None
            else None
        ),
        "strict_reference_available": strict_ref_idx is not None,
        "strict_reference_debug": strict_debug,
    }
    if selected_ref_idx is None:
        reject_reasons["no_near_contact_reference"] += 1
        detail["reference_reject_reasons"] = dict(reject_reasons)
        return detail

    ref_path = Path(image_dir) / f"frame_{selected_ref_idx + 1:010d}.jpg"
    ref_img = cv2.imread(str(ref_path))
    if ref_img is not None:
        hand_meta = hand_bbox_and_meta(detections[int(selected_ref_idx)], active_hand, ref_img.shape, score_threshold=0.5)
        detail.update(
            active_hand_visible_at_ref=bool(hand_meta["visible"]),
            active_hand_score_at_ref=hand_meta["score"],
            active_hand_state_at_ref=hand_meta["state"],
            active_hand_contact_binary_at_ref=int(left_binary[selected_ref_idx])
            if active_hand == "left"
            else int(right_binary[selected_ref_idx]),
            any_contact_binary_at_ref=int(left_binary[selected_ref_idx] or right_binary[selected_ref_idx]),
            ref_before_episode_start=bool(int(selected_ref_idx) < int(episode_start)),
        )
    return detail


class Problem3CachedRunnerE17:
    def __init__(
        self,
        detections,
        image_dir: Path,
        left_binary: np.ndarray,
        right_binary: np.ndarray,
        hand_object_iou_max: Optional[float] = None,
        min_hand_object_center_distance_px: Optional[float] = None,
    ):
        self.detections = detections
        self.image_dir = Path(image_dir)
        self.left_binary = np.asarray(left_binary, dtype=int)
        self.right_binary = np.asarray(right_binary, dtype=int)
        self.hand_object_iou_max = float(hand_object_iou_max) if hand_object_iou_max is not None else None
        self.min_hand_object_center_distance_px = min_hand_object_center_distance_px
        self.gray_cache: Dict[int, np.ndarray] = {}
        self.mask_cache: Dict[Tuple[int, float, float], np.ndarray] = {}
        self.pair_cache: Dict[Tuple[int, int, float, float], Tuple[Optional[np.ndarray], Dict[str, Any]]] = {}
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

    def dynamic_mask(self, frame_idx: int, hand_score_threshold: float = 0.5, object_score_threshold: float = 0.5):
        key = (int(frame_idx), float(hand_score_threshold), float(object_score_threshold))
        if key not in self.mask_cache:
            gray = self.load_gray(frame_idx)
            self.mask_cache[key] = build_dynamic_mask(
                self.detections[int(frame_idx)],
                gray.shape,
                hand_score_threshold,
                object_score_threshold,
            )
        return self.mask_cache[key]

    def pairwise_homography(self, cur_idx: int, prev_idx: int, hand_score_threshold: float = 0.5, object_score_threshold: float = 0.5):
        key = (int(cur_idx), int(prev_idx), float(hand_score_threshold), float(object_score_threshold))
        if key in self.pair_cache:
            self.cache_stats["pair_cache_hits"] += 1
            H, stats = self.pair_cache[key]
            return H.copy() if H is not None else None, dict(stats)

        cur_gray = self.load_gray(cur_idx)
        prev_gray = self.load_gray(prev_idx)
        cur_mask = self.dynamic_mask(cur_idx, hand_score_threshold, object_score_threshold)
        prev_mask = self.dynamic_mask(prev_idx, hand_score_threshold, object_score_threshold)
        H, stats = compute_pairwise_homography(
            prev_gray,
            cur_gray,
            prev_mask,
            cur_mask,
            nfeatures=2000,
            ratio_test=0.75,
            ransac_reproj_threshold=5.0,
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
            if stats["good_matches"] < 60 or stats["inliers"] < 40 or stats["inlier_ratio"] < 0.45:
                stats["passed_quality_gate"] = False
                return None, pair_stats, f"pairwise_homography_low_quality_f{cur_idx}_to_f{prev_idx}"
            stats["passed_quality_gate"] = True
            H_total = H_cur_to_prev @ H_total
        return H_total, pair_stats, None

    def find_near_contact_reference(
        self,
        *,
        t_contact: int,
        active_hand: str,
        contact_means: np.ndarray,
        projected_object_bbox_contact: Optional[Tuple[float, float, float, float]],
        left_binary: np.ndarray,
        right_binary: np.ndarray,
        search_start: int,
        search_end: int,
        episode_start: int,
        reference_offsets: Sequence[int],
        crop_size: int,
        crop_hand_overlap_max: Optional[float],
        hand_object_iou_max: Optional[float],
        min_hand_object_center_distance_px: Optional[float],
    ) -> Tuple[Optional[int], Dict[str, Any]]:
        raw_search_start = int(search_start)
        raw_search_end = int(search_end)
        shadow_search_start = max(0, raw_search_start)
        shadow_search_end = raw_search_end
        strict_ref_idx = None
        strict_debug: Dict[str, Any] = {
            "rejected_isolated_frames": [],
            "streak_lengths": {},
            "final_streak_start": None,
            "final_streak_length": 0,
        }
        if shadow_search_end >= shadow_search_start:
            strict_ref_idx, strict_debug = find_strict_humanless_frame(
                self.detections,
                shadow_search_end + 1,
                score_threshold=0.5,
                min_no_hand_streak=3,
                min_frame_idx=shadow_search_start,
            )

        active_hand_invisible_ref_idx = None
        if shadow_search_end >= shadow_search_start:
            for ref_candidate in range(shadow_search_end, shadow_search_start - 1, -1):
                try:
                    ref_img = self.load_bgr(ref_candidate)
                except FileNotFoundError:
                    continue
                hand_meta = hand_bbox_and_meta(
                    self.detections[int(ref_candidate)],
                    active_hand,
                    ref_img.shape,
                    score_threshold=0.5,
                )
                if not bool(hand_meta["visible"]):
                    active_hand_invisible_ref_idx = int(ref_candidate)
                    break

        clean_ref_idx = None
        if shadow_search_end >= shadow_search_start and contact_means is not None:
            try:
                clean_candidate, _clean_detail = Problem3CachedRunnerE14.find_closest_clean_pre_contact_reference(
                    self,
                    t_contact=int(t_contact),
                    active_hand=active_hand,
                    contact_means=contact_means,
                    projected_object_bbox_contact=projected_object_bbox_contact,
                    left_binary=left_binary,
                    right_binary=right_binary,
                    search_start=shadow_search_start,
                    search_end=shadow_search_end,
                    episode_start=episode_start,
                    crop_size=int(crop_size),
                    crop_hand_overlap_max=0.05,
                    hand_object_iou_max=0.05,
                    min_hand_object_center_distance_px=min_hand_object_center_distance_px,
                )
                clean_ref_idx = int(clean_candidate) if clean_candidate is not None else None
            except Exception:
                clean_ref_idx = None

        detail: Dict[str, Any] = {
            "reference_attempted_refs": 0,
            "reference_reject_reasons": {},
            "strict_ref_idx_shadow": int(strict_ref_idx) if strict_ref_idx is not None else None,
            "strict_reference_anchor_gap_shadow": int(episode_start - int(strict_ref_idx)) if strict_ref_idx is not None else None,
            "active_hand_invisible_ref_idx_shadow": int(active_hand_invisible_ref_idx) if active_hand_invisible_ref_idx is not None else None,
            "active_hand_invisible_reference_anchor_gap_shadow": (
                int(episode_start - int(active_hand_invisible_ref_idx))
                if active_hand_invisible_ref_idx is not None
                else None
            ),
            "clean_ref_idx_shadow": int(clean_ref_idx) if clean_ref_idx is not None else None,
            "clean_reference_anchor_gap_shadow": int(episode_start - int(clean_ref_idx)) if clean_ref_idx is not None else None,
            "strict_ref_idx": None,
            "strict_reference_anchor_gap": None,
            "reference_anchor_gap_delta_vs_strict": None,
            "crop_hand_overlap_ratio": None,
            "reference_hand_object_iou": None,
            "reference_hand_object_center_distance": None,
            "projected_contact_centroid_ref": None,
            "projected_contact_centroid_in_bounds": None,
            "active_hand_visible_at_ref": False,
            "active_hand_score_at_ref": None,
            "active_hand_state_at_ref": None,
            "active_hand_contact_binary_at_ref": None,
            "any_contact_binary_at_ref": None,
            "reference_object_bbox": None,
            "reference_object_bbox_score": None,
            "reference_object_iou": None,
            "reference_object_center_distance": None,
            "projected_object_bbox": None,
            "fallback_precheck_homography_min_inlier_ratio": None,
            "fallback_precheck_homography_min_inliers": None,
            "fallback_precheck_homography_min_good_matches": None,
            "closest_active_hand_no_contact_frame_shadow": None,
            "closest_any_contact_zero_frame_shadow": None,
            "e14_clean_ref_idx_shadow": int(clean_ref_idx) if clean_ref_idx is not None else None,
            "strict_reference_debug": strict_debug,
            "reference_offset": None,
            "reference_offsets": ",".join(str(int(offset)) for offset in reference_offsets),
            "reference_search_window_start_0_based": int(raw_search_start),
            "reference_search_window_end_0_based": int(raw_search_end),
        }

        reject_reasons: Counter = Counter()
        selected_ref_idx: Optional[int] = None
        selected_offset: Optional[int] = None
        for offset in reference_offsets:
            offset = int(offset)
            ref_candidate = int(episode_start) - offset
            if ref_candidate < 0:
                reject_reasons["ref_idx_lt_zero"] += 1
                continue
            if ref_candidate >= len(self.detections):
                reject_reasons["ref_idx_out_of_detection_range"] += 1
                continue
            ref_path = self.image_dir / f"frame_{ref_candidate + 1:010d}.jpg"
            if not ref_path.exists():
                reject_reasons["reference_frame_missing"] += 1
                continue
            detail["reference_attempted_refs"] += 1
            selected_ref_idx = int(ref_candidate)
            selected_offset = int(offset)
            break

        if selected_ref_idx is None:
            reject_reasons["no_near_contact_reference"] += 1
            detail["reference_reject_reasons"] = dict(reject_reasons)
            return None, detail

        ref_idx = int(selected_ref_idx)
        active_contact = int(left_binary[ref_idx]) if active_hand == "left" else int(right_binary[ref_idx])
        any_contact = int(left_binary[ref_idx] or right_binary[ref_idx])
        ref_img = self.load_bgr(ref_idx)
        hand_meta = hand_bbox_and_meta(self.detections[ref_idx], active_hand, ref_img.shape, score_threshold=0.5)
        detail.update(
            reference_mode="near_contact_pre_frame",
            reference_reason="closest_pre_contact_offset_in_5_10",
            reference_offset=int(selected_offset) if selected_offset is not None else None,
            reference_reject_reasons=dict(reject_reasons),
            active_hand_visible_at_ref=bool(hand_meta["visible"]),
            active_hand_score_at_ref=hand_meta["score"],
            active_hand_state_at_ref=hand_meta["state"],
            active_hand_contact_binary_at_ref=int(active_contact),
            any_contact_binary_at_ref=int(any_contact),
            ref_before_episode_start=bool(ref_idx < episode_start),
        )
        if strict_ref_idx is not None:
            detail["reference_anchor_gap_delta_vs_strict"] = int(
                (episode_start - ref_idx) - (episode_start - int(strict_ref_idx))
            )
        return ref_idx, detail

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
        reference_offsets: Sequence[int] = (5, 6, 7, 8, 9, 10),
        crop_size: int = 150,
        crop_hand_overlap_max: Optional[float] = None,
        hand_object_iou_max: Optional[float] = None,
        min_hand_object_center_distance_px: Optional[float] = None,
        discard: bool = False,
        discard_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        episode_start = int(episode_start_frame_0_based)
        search_start = int(reference_search_window_start_0_based)
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
            "reference_offset": None,
            "reference_offsets": ",".join(str(int(offset)) for offset in reference_offsets),
            "reference_attempted_refs": 0,
            "reference_reject_reasons": {},
            "active_hand_visible_at_ref": None,
            "active_hand_score_at_ref": None,
            "active_hand_state_at_ref": None,
            "active_hand_contact_binary_at_ref": None,
            "any_contact_binary_at_ref": None,
            "reference_hand_object_iou": None,
            "reference_hand_object_center_distance": None,
            "projected_contact_centroid_ref": None,
            "projected_contact_centroid_in_bounds": None,
            "strict_ref_idx_shadow": None,
            "strict_ref_idx": None,
            "strict_reference_anchor_gap_shadow": None,
            "strict_reference_anchor_gap": None,
            "active_hand_invisible_ref_idx_shadow": None,
            "active_hand_invisible_reference_anchor_gap_shadow": None,
            "clean_ref_idx_shadow": None,
            "clean_reference_anchor_gap_shadow": None,
            "reference_anchor_gap_delta_vs_strict": None,
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
            "closest_active_hand_no_contact_frame_shadow": None,
            "closest_any_contact_zero_frame_shadow": None,
            "e14_clean_ref_idx_shadow": None,
            "strict_reference_debug": None,
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
            "homography_available_without_geometry_gate": False,
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
                discard_reason="no_near_contact_reference",
                status="DISCARD",
                no_pre_episode_reference_window_not_rescued=True,
                reference_mode="no_near_contact_reference",
                reference_reason="no_near_contact_reference",
            )
            return result

        frame_det_contact = self.detections[int(t_contact)]
        contact_img = self.load_bgr(int(t_contact))
        h_contact, w_contact = contact_img.shape[:2]
        active_hand_bbox_norm = get_active_hand_bbox(frame_det_contact, active_hand, score_threshold=0.5)
        object_bboxes_norm = get_valid_object_bboxes(frame_det_contact, score_threshold=0.5)
        active_object_bbox_norm = (
            select_active_object_bbox(active_hand_bbox_norm, object_bboxes_norm)
            if active_hand_bbox_norm is not None
            else None
        )
        projected_object_bbox_contact = None
        if active_object_bbox_norm is not None:
            projected_object_bbox_contact = (
                float(active_object_bbox_norm[0] * w_contact),
                float(active_object_bbox_norm[1] * h_contact),
                float(active_object_bbox_norm[2] * w_contact),
                float(active_object_bbox_norm[3] * h_contact),
            )

        result["fallback_reference_attempted"] = False
        ref_idx, detail = self.find_near_contact_reference(
            t_contact=int(t_contact),
            active_hand=active_hand,
            contact_means=contact_means,
            projected_object_bbox_contact=projected_object_bbox_contact,
            left_binary=self.left_binary,
            right_binary=self.right_binary,
            search_start=search_start,
            search_end=search_end,
            episode_start=episode_start,
            reference_offsets=reference_offsets,
            crop_size=int(crop_size),
            crop_hand_overlap_max=crop_hand_overlap_max,
            hand_object_iou_max=(hand_object_iou_max if hand_object_iou_max is not None else self.hand_object_iou_max),
            min_hand_object_center_distance_px=(
                min_hand_object_center_distance_px
                if min_hand_object_center_distance_px is not None
                else self.min_hand_object_center_distance_px
            ),
        )
        result.update(detail)
        result["strict_reference_available"] = result.get("strict_ref_idx_shadow") is not None
        result["ref_idx"] = ref_idx
        if ref_idx is None:
            result.update(
                discard=True,
                discard_reason="no_near_contact_reference",
                status="DISCARD",
                reference_mode="no_near_contact_reference",
                reference_reason="no_near_contact_reference",
            )
            return result
        result.update(
            fallback_reference_used=False,
            rescued_from_no_strict_humanless=False,
            fallback_rescued_from_no_strict_humanless=False,
            ref_before_episode_start=bool(int(ref_idx) < episode_start),
            reference_anchor_gap=int(episode_start - int(ref_idx)),
        )

        ref_img = self.load_bgr(ref_idx)
        trajectory_pixels, trajectory_missing_offsets = self._trajectory_pixels_for_ref_shape(int(t_contact), active_hand, ref_img.shape)
        result["trajectory_pixels"] = trajectory_pixels
        result["trajectory_missing_offsets"] = trajectory_missing_offsets
        if len(trajectory_pixels) == 0:
            result.update(discard=True, discard_reason="missing_trajectory_points", status="DISCARD")
            return result

        cumulative_H_list = []
        all_pair_stats = []
        fail_reason = None
        homography_failed = False
        for offset in range(len(trajectory_pixels)):
            target_idx = int(t_contact) + offset
            H_target_to_ref, pair_stats, fail_reason = self.accumulate_homography_to_ref(ref_idx, target_idx)
            cumulative_H_list.append(H_target_to_ref)
            all_pair_stats.extend(pair_stats)
            if H_target_to_ref is None:
                homography_failed = True
        result["all_pair_stats"] = all_pair_stats
        if homography_failed:
            result.update(discard=True, discard_reason=fail_reason, status="DISCARD")
            return result

        H_contact = cumulative_H_list[0]
        result["H_contact_to_ref"] = H_contact
        result["homography_available_without_geometry_gate"] = H_contact is not None
        mu_transformed = transform_points(contact_means.astype(np.float32), H_contact)
        if mu_transformed is None:
            result.update(discard=True, discard_reason="missing_transformed_contact_points", status="DISCARD")
            return result
        result["mu_transformed"] = mu_transformed
        contact_centroid = mu_transformed.mean(axis=0)
        result["contact_centroid"] = contact_centroid

        tau_transformed = []
        for offset, H in enumerate(cumulative_H_list):
            if H is not None:
                pt = transform_points(trajectory_pixels[offset].reshape(1, -1).astype(np.float32), H)
                tau_transformed.append(pt[0])
            else:
                tau_transformed.append(None)
        result["tau_transformed"] = tau_transformed

        h_img, w_img = ref_img.shape[:2]
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

        inside_count = count_points_near_polygon(mu_transformed, object_polygon_ref, tolerance_px=8.0)
        result["inside_count"] = inside_count
        if inside_count < 4:
            result.update(discard=True, discard_reason="transformed_contact_points_off_object", status="DISCARD")
            return result

        centroid_dist = cv2.pointPolygonTest(object_polygon_ref.astype(np.float32), tuple(contact_centroid), True)
        result["centroid_dist"] = centroid_dist
        if centroid_dist < -8.0:
            result.update(discard=True, discard_reason="transformed_contact_centroid_off_object", status="DISCARD")
            return result

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
    fail_reasons = []
    for item in stats:
        if item.get("inlier_ratio") is not None:
            ratios.append(float(item.get("inlier_ratio", 0.0)))
        if item.get("inliers") is not None:
            inliers.append(int(item.get("inliers", 0)))
        if item.get("good_matches") is not None:
            good_matches.append(int(item.get("good_matches", 0)))
        if item.get("fail_reason"):
            fail_reasons.append(str(item.get("fail_reason")))
    return {
        "homography_pairs": int(len(stats)),
        "homography_min_inlier_ratio": min(ratios) if ratios else None,
        "homography_min_inliers": min(inliers) if inliers else None,
        "homography_min_good_matches": min(good_matches) if good_matches else None,
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
    config: E17Config,
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
    config: E17Config,
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
        "raw_contact_points_count": 0,
        "cell2_raw_contact_available": False,
        "ref_idx": None,
        "ref_frame_1_based": None,
        "ref_gap": None,
        "reference_anchor_gap": None,
        "reference_offset": None,
        "reference_offsets": ",".join(str(int(offset)) for offset in config.reference_offsets),
        "trajectory_points": 0,
        "missing_trajectory_count": None,
        "heatmap_mode": None,
        "sample_score": None,
        "sample_dir": None,
        "reference_mode": None,
        "reference_reason": None,
        "reference_attempted_refs": 0,
        "reference_reject_reasons": None,
        "active_hand_visible_at_ref": None,
        "active_hand_score_at_ref": None,
        "active_hand_state_at_ref": None,
        "active_hand_contact_binary_at_ref": None,
        "any_contact_binary_at_ref": None,
        "reference_hand_object_iou": None,
        "reference_hand_object_center_distance": None,
        "projected_contact_centroid_ref": None,
        "projected_contact_centroid_in_bounds": None,
        "strict_ref_idx_shadow": None,
        "strict_ref_idx": None,
        "strict_reference_anchor_gap_shadow": None,
        "strict_reference_anchor_gap": None,
        "active_hand_invisible_ref_idx_shadow": None,
        "active_hand_invisible_reference_anchor_gap_shadow": None,
        "clean_ref_idx_shadow": None,
        "clean_reference_anchor_gap_shadow": None,
        "reference_anchor_gap_delta_vs_strict": None,
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
        "closest_active_hand_no_contact_frame_shadow": None,
        "closest_any_contact_zero_frame_shadow": None,
        "e14_clean_ref_idx_shadow": None,
        "strict_reference_debug": None,
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
        "homography_available_without_geometry_gate": False,
        "projected_area": None,
        "area_ratio": None,
        "inside_count": None,
        "centroid_dist": None,
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
    problem3_runner: Problem3CachedRunnerE17,
    config: E17Config,
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

    reference_pre = preselect_near_contact_reference(
        detections=detections,
        image_dir=config.image_dir,
        episode_start=int(candidate["episode_contact_start_frame_0_based"]),
        t_contact=int(candidate["frame"]),
        active_hand=candidate["hand"],
        left_binary=problem3_runner.left_binary,
        right_binary=problem3_runner.right_binary,
        reference_offsets=config.reference_offsets,
        search_start=int(record["reference_search_window_start_0_based"]),
        search_end=int(record["reference_search_window_end_0_based"]),
    )
    for field, value in reference_pre.items():
        record[field] = value

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
    record["raw_contact_points_count"] = int(cell2.get("contact_points", 0))
    record["cell2_raw_contact_available"] = bool(int(cell2.get("contact_points", 0)) > 0)
    if record.get("ref_idx") is None:
        record.update(status="discard", failed_stage="reference_selection", fail_reason="no_near_contact_reference")
        write_candidate_json(record, sample_dir)
        return record
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
        reference_offsets=config.reference_offsets,
        crop_size=int(config.crop_size),
        crop_hand_overlap_max=config.crop_hand_overlap_max,
        hand_object_iou_max=config.hand_object_iou_max,
        min_hand_object_center_distance_px=config.min_hand_object_center_distance_px,
        discard=False,
        discard_reason=None,
    )
    record["ref_idx"] = cell3.get("ref_idx")
    if record["ref_idx"] is not None:
        record["ref_frame_1_based"] = int(record["ref_idx"]) + 1
        record["ref_gap"] = int(candidate["frame"]) - int(record["ref_idx"])
        record["reference_anchor_gap"] = int(candidate["episode_contact_start_frame_0_based"]) - int(record["ref_idx"])
        record["reference_offset"] = int(record["reference_anchor_gap"])

    for field in [
        "reference_mode",
        "reference_reason",
        "reference_offset",
        "reference_offsets",
        "reference_attempted_refs",
        "reference_reject_reasons",
        "active_hand_visible_at_ref",
        "active_hand_score_at_ref",
        "active_hand_state_at_ref",
        "active_hand_contact_binary_at_ref",
        "any_contact_binary_at_ref",
        "reference_hand_object_iou",
        "reference_hand_object_center_distance",
        "projected_contact_centroid_ref",
        "projected_contact_centroid_in_bounds",
        "strict_ref_idx_shadow",
        "strict_ref_idx",
        "strict_reference_anchor_gap_shadow",
        "strict_reference_anchor_gap",
        "active_hand_invisible_ref_idx_shadow",
        "active_hand_invisible_reference_anchor_gap_shadow",
        "clean_ref_idx_shadow",
        "clean_reference_anchor_gap_shadow",
        "reference_anchor_gap_delta_vs_strict",
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
        "closest_active_hand_no_contact_frame_shadow",
        "closest_any_contact_zero_frame_shadow",
        "e14_clean_ref_idx_shadow",
        "strict_reference_debug",
        "ref_before_episode_start",
        "reference_anchor_gap",
        "homography_available_without_geometry_gate",
        "projected_area",
        "area_ratio",
        "inside_count",
        "centroid_dist",
    ]:
        record[field] = cell3.get(field, record.get(field))

    record["trajectory_points"] = len(cell3.get("trajectory_pixels") or [])
    record["missing_trajectory_count"] = len(cell3.get("trajectory_missing_offsets") or [])
    record.update(compact_pair_stats(cell3))

    if cell3.get("H_contact_to_ref") is not None:
        record["passed_stage"] = "Cell3_homography_available"
    if cell3.get("discard") or cell3.get("status") != "KEEP":
        reason = cell3.get("discard_reason") or "cell3_discard"
        failed_stage = "reference_selection" if reason == "no_near_contact_reference" else "Cell3_homography_geometry"
        record.update(status="discard", failed_stage=failed_stage, fail_reason=reason)
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
    keep_anchor_gap = (
        candidate_df.loc[candidate_df["status"] == "keep", "reference_anchor_gap"].dropna().astype(float)
        if "reference_anchor_gap" in candidate_df
        else pd.Series(dtype=float)
    )
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
                "strict_ref_found": int(candidate_df.get("strict_ref_idx_shadow", pd.Series(dtype=object)).notna().sum()) if "strict_ref_idx_shadow" in candidate_df else 0,
                "active_hand_invisible_ref_found": int(candidate_df.get("active_hand_invisible_ref_idx_shadow", pd.Series(dtype=object)).notna().sum()) if "active_hand_invisible_ref_idx_shadow" in candidate_df else 0,
                "near_contact_ref_found": int(candidate_df["ref_idx"].notna().sum()),
                "fallback_ref_found": 0,
                "clean_ref_found": int(candidate_df.get("clean_ref_idx_shadow", pd.Series(dtype=object)).notna().sum()) if "clean_ref_idx_shadow" in candidate_df else 0,
                "active_hand_invisible_ref_keep": 0,
                "near_contact_ref_keep": int(((candidate_df["status"] == "keep") & (candidate_df.get("reference_mode") == "near_contact_pre_frame")).sum()) if "reference_mode" in candidate_df else 0,
                "fallback_ref_used": 0,
                "rescued_from_no_strict_humanless": 0,
                "fallback_rescued_from_no_strict_humanless": 0,
                "fallback_attempted": 0,
                "no_pre_episode_reference_window_not_rescued": int(candidate_df.get("no_pre_episode_reference_window_not_rescued", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "max_ref_backtrack": int(candidate_df["max_ref_backtrack"].dropna().iloc[0]) if "max_ref_backtrack" in candidate_df and candidate_df["max_ref_backtrack"].notna().any() else None,
                "window_truncated_by_max_backtrack_count": int(candidate_df.get("window_truncated_by_max_backtrack", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "window_truncated_by_max_backtrack_rate": float(candidate_df.get("window_truncated_by_max_backtrack", pd.Series(dtype=bool)).fillna(False).astype(bool).mean()) if len(candidate_df) else 0.0,
                "homography_available": int((candidate_df["passed_stage"].isin(["Cell3_homography_available", "Cell3_homography_geometry", "Cell4_label_heatmap"])).sum()),
                "homography_available_without_geometry_gate": int(candidate_df.get("homography_available_without_geometry_gate", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                "object_consistency_pass": 0,
                "object_consistency_fail": 0,
                "geometry_pass": int((candidate_df["passed_stage"].isin(["Cell3_homography_geometry", "Cell4_label_heatmap"])).sum()),
                "heatmap_pass": int((candidate_df["status"] == "keep").sum()),
                "reference_anchor_gap_median_keep": float(keep_anchor_gap.median()) if not keep_anchor_gap.empty else None,
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
    ax.set_title("E17 failure reasons")
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
    ax.set_title("E17 pipeline funnel")
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
    ax.set_title("Timeline: E17 first-contact samples and reference arrows")
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
    ax.set_title("E17 success ref_gap distribution")
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
    ax.set_title("E17 success candidate offset distribution")
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
    ax.set_title("E17 pre-contact gap length distribution")
    ax.set_xlabel("gap length before episode start")
    ax.set_ylabel("candidate count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_reference_mode_ref_gap_histogram(success_df: pd.DataFrame, path: Path) -> None:
    if success_df.empty or "ref_gap" not in success_df or "reference_mode" not in success_df:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for mode, color in [
        ("near_contact_pre_frame", "#1a73e8"),
    ]:
        values = success_df[success_df["reference_mode"] == mode]["ref_gap"].dropna().astype(float)
        if not values.empty:
            ax.hist(values, bins=min(20, max(5, int(values.nunique()))), alpha=0.65, label=mode, color=color)
    ax.set_title("E17 success ref_gap by reference mode")
    ax.set_xlabel("candidate frame - reference frame")
    ax.set_ylabel("count")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_overlay_contact_sheet(df: pd.DataFrame, path: Path, title: str, max_items: int = 30) -> None:
    view = df[df["status"] == "keep"].copy()
    if view.empty or "vrb_style_affordance" not in view:
        return
    view = view.head(max_items)
    images = []
    labels = []
    for _, row in view.iterrows():
        img_path = None
        for col in ["vrb_style_affordance", "label_heatmap_overlay", "problem3_full_overlay"]:
            candidate_path = row.get(col)
            if isinstance(candidate_path, str) and Path(candidate_path).exists():
                img_path = candidate_path
                break
        if img_path is None:
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(
            f"s{int(row['subaction_index'])} {row.get('narration_id', '')}\n"
            f"{str(row.get('narration', ''))[:34]}\n"
            f"cf{int(row['frame_0_based'])} ref{int(row['ref_idx'])} off{int(row.get('reference_offset', row['ref_gap']))} "
            f"gap/anchor {int(row['ref_gap'])}/{int(row['reference_anchor_gap'])}\n"
            f"{row.get('reference_mode', '')}"
        )
    if not images:
        return

    thumb_w, thumb_h = 320, 200
    cols = min(5, len(images))
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.2))
    axes_arr = np.asarray(axes).reshape(-1)
    for ax in axes_arr:
        ax.axis("off")
    for ax, img, label in zip(axes_arr, images, labels):
        thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        ax.imshow(thumb)
        ax.set_title(label, fontsize=7)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_paginated_success_contact_sheets(
    success_df: pd.DataFrame,
    charts_dir: Path,
    prefix: str,
    title: str,
    per_page: int = 30,
) -> List[Path]:
    paths = []
    if success_df.empty:
        return paths
    view = success_df.sort_values(["subaction_index", "frame_0_based"]).copy()
    for page_idx, start in enumerate(range(0, len(view), int(per_page)), start=1):
        page = view.iloc[start:start + int(per_page)].copy()
        path = charts_dir / f"{prefix}_page_{page_idx:03d}.png"
        save_overlay_contact_sheet(page, path, f"{title} page {page_idx}", max_items=int(per_page))
        if path.exists():
            paths.append(path)
    return paths


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


def ensure_placeholder_png(path: Path, title: str, message: str = "No keep samples") -> None:
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


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
        "reference_offset",
        "ref_gap",
        "reference_anchor_gap",
        "sample_dir",
        "reference_frame",
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
    keep = build_sample_key_df(candidate_df)
    for baseline_exp, baseline_root_name, out_name in [
        ("E6b", "数据处理小批量测试前100个subaction_E6b_pre_contact_gap_crop_fallback", "e17_success_vs_e6b.csv"),
        ("E6c", "数据处理小批量测试前100个subaction_E6c_object_consistency", "e17_success_vs_e6c.csv"),
    ]:
        baseline_root = resolve_output_dir(baseline_root_name)
        baseline_path = baseline_root / "candidate_diagnostics.csv" if baseline_root is not None else None
        if baseline_path is None or not baseline_path.exists():
            pd.DataFrame().to_csv(output_root / out_name, index=False)
            continue
        baseline_df = pd.read_csv(baseline_path)
        if "experiment_id" in baseline_df.columns:
            baseline_df = baseline_df[baseline_df["experiment_id"] == baseline_exp]
        baseline_keep = build_sample_key_df(baseline_df)
        baseline_keys = set(baseline_keep.get("sample_key", []))
        keep.assign(in_baseline=keep["sample_key"].isin(baseline_keys)).to_csv(output_root / out_name, index=False)


def write_markdown_report(
    *,
    config: E17Config,
    overview_df: pd.DataFrame,
    subaction_summary_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    candidate_plan_df: pd.DataFrame,
    frame_coverage: Dict[str, int],
) -> Path:
    output_root = config.output_root
    success_df = candidate_df[candidate_df["status"] == "keep"].copy()

    compare_rows = []
    baseline_specs = [
        ("E16", "e16"),
        ("E15", "e15"),
        ("E14", "e14"),
        ("E6b", "数据处理小批量测试前100个subaction_E6b_pre_contact_gap_crop_fallback"),
        ("E6c", "数据处理小批量测试前100个subaction_E6c_object_consistency"),
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
    compare_rows.append(overview_df.iloc[0].to_dict())
    compare_overview = pd.DataFrame(compare_rows)
    for column in [
        "strict_ref_found",
        "clean_ref_found",
        "ref_found",
        "heatmap_pass",
        "candidates_total",
        "episode_filtered_candidate_total",
        "subactions_success",
        "active_hand_invisible_ref_found",
        "near_contact_ref_found",
        "homography_available",
        "homography_available_without_geometry_gate",
        "geometry_pass",
        "reference_anchor_gap_median_keep",
        "main_failure",
    ]:
        if column not in compare_overview.columns:
            compare_overview[column] = 0
    for idx, row in compare_overview.iterrows():
        if not row.get("reference_anchor_gap_median_keep"):
            exp_id = str(row.get("experiment_id", ""))
            baseline_root = resolve_output_dir("e16") if exp_id == "E16" else None
            if exp_id == "E15":
                baseline_root = resolve_output_dir("e15")
            if exp_id == "E14":
                baseline_root = resolve_output_dir("e14")
            if exp_id == "E6b":
                baseline_root = resolve_output_dir("数据处理小批量测试前100个subaction_E6b_pre_contact_gap_crop_fallback")
            elif exp_id == "E6c":
                baseline_root = resolve_output_dir("数据处理小批量测试前100个subaction_E6c_object_consistency")
            median = None
            if exp_id == "E17":
                median = distribution_stats(success_df["reference_anchor_gap"]).get("median") if "reference_anchor_gap" in success_df else None
            elif baseline_root is not None and (baseline_root / "candidate_diagnostics.csv").exists():
                baseline_candidates = pd.read_csv(baseline_root / "candidate_diagnostics.csv")
                if "status" in baseline_candidates and "reference_anchor_gap" in baseline_candidates:
                    baseline_keep = baseline_candidates[baseline_candidates["status"] == "keep"]
                    median = distribution_stats(baseline_keep["reference_anchor_gap"]).get("median")
            compare_overview.loc[idx, "reference_anchor_gap_median_keep"] = median

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
        "# E17 near-contact reference baseline - 诊断实验报告",
        "",
        f"- video_id: `{config.video_id}`",
        f"- subactions: first `{config.num_subactions}` annotations of this video",
        f"- output root: `{output_root}`",
        f"- reference offsets: `{list(config.reference_offsets)}`",
        f"- candidate offsets from episode start: `{list(config.early_candidate_offsets)}`",
        "- reference strategy: closest available pre-contact frame at offsets 5-10; humanless/active-hand-invisible/E14 clean gates are shadow-only.",
        f"- annotated frame intervals total: `{frame_coverage['total_annotated_frames']}` frames",
        f"- unique covered frames after overlap removal: `{frame_coverage['unique_covered_frames']}` frames",
        "",
        "## E17 定义",
        "",
        "- 一个 subaction 视作一个 training sequence，最多输出一个 final tuple。",
        "- first contact timestep 是该 subaction 内最早 new contact run 的 start frame，active hand 使用该 run 的 hand。",
        "- reference 从 episode start 前 5-10 帧里选最近可用帧，不再要求 humanless 或 active-hand-invisible。",
        "- E14 clean gate、strict humanless、active-hand-invisible 只记录 shadow diagnostics，不进入主筛选。",
        "- E17 是 near-contact diagnostic baseline，不是 paper-faithful humanless baseline。",
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
                "active_hand_invisible_ref_found",
                "near_contact_ref_found",
                "clean_ref_found",
                "homography_available",
                "geometry_pass",
                "heatmap_pass",
                "subactions_success",
                "main_failure",
            ],
        ),
        "",
        "## E17 vs E16 vs E15 vs E14 vs E6b vs E6c 对比",
        "",
        "E6b/E6c 是 multi-candidate per subaction，candidate-level heatmap_pass 不能直接和 E17 的 candidate count 对齐；`subactions_success` 是更公平口径。E14/E15/E16/E17 中，E16 和 E17 都是 one candidate per subaction，最可比。",
        "",
        markdown_table(
            compare_overview,
            [
                "experiment_id",
                "subactions_success",
                "heatmap_pass",
                "candidates_total",
                "episode_filtered_candidate_total",
                "ref_found",
                "strict_ref_found",
                "active_hand_invisible_ref_found",
                "near_contact_ref_found",
                "clean_ref_found",
                "homography_available",
                "geometry_pass",
                "homography_available_without_geometry_gate",
                "reference_anchor_gap_median_keep",
                "main_failure",
            ],
        ),
        "",
        "## 核心回答",
        "",
        f"- total success count: `{len(success_df)}`",
        f"- candidates_total: `{int(overview_df.iloc[0]['candidates_total'])}`",
        f"- episode_filtered_candidate_total: `{int(overview_df.iloc[0]['episode_filtered_candidate_total'])}`",
        f"- median reference_anchor_gap: `{distribution_stats(success_df['reference_anchor_gap']).get('median') if 'reference_anchor_gap' in success_df else None}`",
        f"- median ref_gap: `{distribution_stats(success_df['ref_gap']).get('median') if 'ref_gap' in success_df else None}`",
        f"- homography_available / geometry_pass / heatmap_pass: `{int(overview_df.iloc[0]['homography_available'])}` / `{int(overview_df.iloc[0]['geometry_pass'])}` / `{int(overview_df.iloc[0]['heatmap_pass'])}`",
        f"- homography_available_without_geometry_gate: `{int(overview_df.iloc[0]['homography_available_without_geometry_gate'])}`",
        "",
        "## success 分布",
        "",
        f"- ref_gap stats: `{distribution_stats(success_df['ref_gap']) if 'ref_gap' in success_df else distribution_stats(pd.Series(dtype=float))}`",
        f"- success reference_anchor_gap stats: `{distribution_stats(success_df['reference_anchor_gap']) if 'reference_anchor_gap' in success_df else distribution_stats(pd.Series(dtype=float))}`",
        f"- strict shadow reference anchor gap stats: `{distribution_stats(candidate_df['strict_reference_anchor_gap_shadow']) if 'strict_reference_anchor_gap_shadow' in candidate_df else distribution_stats(pd.Series(dtype=float))}`",
        f"- active-hand-invisible shadow anchor gap stats: `{distribution_stats(candidate_df['active_hand_invisible_reference_anchor_gap_shadow']) if 'active_hand_invisible_reference_anchor_gap_shadow' in candidate_df else distribution_stats(pd.Series(dtype=float))}`",
        f"- clean shadow anchor gap stats: `{distribution_stats(candidate_df['clean_reference_anchor_gap_shadow']) if 'clean_reference_anchor_gap_shadow' in candidate_df else distribution_stats(pd.Series(dtype=float))}`",
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
        "- pipeline funnel: `charts/e17_pipeline_funnel.png`",
        "- failure reasons: `charts/e17_failure_reasons.png`",
        "- success ref_gap histogram: `charts/e17_ref_gap_success_hist.png`",
        "- success reference_anchor_gap histogram: `charts/e17_reference_anchor_gap_hist.png`",
        "- success contact sheet: `charts/e17_success_contact_sheet_page_001.png`",
        "- success heatmap+trajectory contact sheet: `charts/e17_success_heatmap_trajectory_contact_sheet_page_001.png`",
        "- 排查汇总: `charts/排查汇总/`",
        "",
        "## 明细文件",
        "",
        "- experiment JSON: `experiment_results.json`",
        "- overview CSV: `experiment_overview.csv`",
        "- subaction summary CSV: `subaction_summary.csv`",
        "- candidate diagnostics CSV: `candidate_diagnostics.csv`",
        "- candidate plan CSV: `candidate_plan.csv`",
        "- successful sample manifest: `successful_sample_manifest.csv`",
        "- E17 vs E6b success table: `e17_success_vs_e6b.csv`",
        "- E17 vs E6c success table: `e17_success_vs_e6c.csv`",
        "- pipeline outputs: `experiments/E17/pipeline_outputs/`",
        "",
        "## Interpretation",
        "",
        "- E17 是 near-contact diagnostic baseline，不是 paper-faithful humanless baseline。",
        "- 如果 E17 仍沿用当前 Cell2，那么 100 个 subaction 的成功上限大概率仍被 Cell2 限制在约 59，而不是 80。",
    ]
    report_path = output_root / "summary.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


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
    fail_reason = "continuation_contact_run" if episode_decision == "continuation_contact_run" else "no_contact_candidate"
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
        "raw_contact_points_count": 0,
        "cell2_raw_contact_available": False,
        "ref_idx": None,
        "ref_frame_1_based": None,
        "ref_gap": None,
        "reference_anchor_gap": None,
        "reference_offset": None,
        "reference_offsets": "5,6,7,8,9,10",
        "trajectory_points": 0,
        "missing_trajectory_count": None,
        "heatmap_mode": None,
        "sample_score": None,
        "sample_dir": None,
        "reference_mode": None,
        "reference_reason": None,
        "reference_attempted_refs": 0,
        "reference_reject_reasons": None,
        "active_hand_visible_at_ref": None,
        "active_hand_score_at_ref": None,
        "active_hand_state_at_ref": None,
        "active_hand_contact_binary_at_ref": None,
        "any_contact_binary_at_ref": None,
        "reference_hand_object_iou": None,
        "reference_hand_object_center_distance": None,
        "projected_contact_centroid_ref": None,
        "projected_contact_centroid_in_bounds": None,
        "strict_ref_idx_shadow": None,
        "strict_ref_idx": None,
        "strict_reference_anchor_gap_shadow": None,
        "strict_reference_anchor_gap": None,
        "active_hand_invisible_ref_idx_shadow": None,
        "active_hand_invisible_reference_anchor_gap_shadow": None,
        "clean_ref_idx_shadow": None,
        "clean_reference_anchor_gap_shadow": None,
        "reference_anchor_gap_delta_vs_strict": None,
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
        "closest_active_hand_no_contact_frame_shadow": None,
        "closest_any_contact_zero_frame_shadow": None,
        "e14_clean_ref_idx_shadow": None,
        "strict_reference_debug": None,
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
        "homography_available_without_geometry_gate": False,
        "projected_area": None,
        "area_ratio": None,
        "inside_count": None,
        "centroid_dist": None,
    }


def save_reference_anchor_gap_histogram(success_df: pd.DataFrame, path: Path) -> None:
    if success_df.empty or "reference_anchor_gap" not in success_df:
        return
    values = success_df["reference_anchor_gap"].dropna().astype(float)
    if values.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=min(20, max(5, int(values.nunique()))), color="#188038", alpha=0.85)
    ax.set_title("E17 success reference_anchor_gap distribution")
    ax.set_xlabel("episode start frame - reference frame")
    ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def run_e17_experiment(config: Optional[E17Config] = None) -> Dict[str, Any]:
    config = config or E17Config()
    if not config.reference_offsets:
        raise ValueError("reference_offsets must be non-empty")
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
    problem3_runner = Problem3CachedRunnerE17(
        detections,
        config.image_dir,
        left_binary,
        right_binary,
        hand_object_iou_max=config.hand_object_iou_max,
        min_hand_object_center_distance_px=config.min_hand_object_center_distance_px,
    )

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
            f"E17 subaction {idx:02d} {row['narration_id']} | {row['narration']} | "
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
                    "reference_offsets": list(config.reference_offsets),
                    "reference_min_gap": config.reference_min_gap,
                    "reference_max_gap": config.reference_max_gap,
                    "max_ref_backtrack": config.max_ref_backtrack,
                    "reference_offsets": list(config.reference_offsets),
                    "early_candidate_offsets": list(config.early_candidate_offsets),
                    "crop_size": config.crop_size,
                    "crop_hand_overlap_max": config.crop_hand_overlap_max,
                    "hand_object_iou_max": config.hand_object_iou_max,
                    "min_hand_object_center_distance_px": config.min_hand_object_center_distance_px,
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
    save_funnel_chart(overview_df, charts_dir / "e17_pipeline_funnel.png")
    save_bar_chart_failure_reasons(candidate_df, charts_dir / "e17_failure_reasons.png")
    save_timeline_chart(candidate_df, subactions, charts_dir / "e17_timeline.png")
    success_df = candidate_df[candidate_df["status"] == "keep"].copy()
    save_ref_gap_histogram(success_df, charts_dir / "e17_ref_gap_success_hist.png")
    save_reference_anchor_gap_histogram(success_df, charts_dir / "e17_reference_anchor_gap_hist.png")
    save_candidate_offset_histogram(success_df, charts_dir / "e17_candidate_offset_success_hist.png")
    save_gap_length_histogram(candidate_df, charts_dir / "e17_gap_length_hist.png")
    save_reference_mode_ref_gap_histogram(success_df, charts_dir / "e17_ref_gap_success_by_reference_mode_hist.png")
    save_paginated_success_contact_sheets(
        success_df,
        charts_dir,
        "e17_success_contact_sheet",
        "E17 success samples",
        per_page=30,
    )
    ensure_placeholder_png(
        charts_dir / "e17_success_contact_sheet_page_001.png",
        "E17 success samples",
    )
    save_paginated_success_contact_sheets(
        success_df,
        charts_dir,
        "e17_success_heatmap_trajectory_contact_sheet",
        "E17 success heatmap + trajectory overlays",
        per_page=30,
    )
    ensure_placeholder_png(
        charts_dir / "e17_success_heatmap_trajectory_contact_sheet_page_001.png",
        "E17 success heatmap + trajectory overlays",
    )
    summary_path = write_markdown_report(
        config=config,
        overview_df=overview_df,
        subaction_summary_df=subaction_summary_df,
        candidate_df=candidate_df,
        candidate_plan_df=candidate_plan_df,
        frame_coverage=frame_coverage,
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
                "strict_ref_found",
                "active_hand_invisible_ref_found",
                "near_contact_ref_found",
                "clean_ref_found",
                "homography_available",
                "homography_available_without_geometry_gate",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E17 near-contact reference baseline.")
    parser.add_argument("--video-id", default=E17Config.video_id)
    parser.add_argument("--num-subactions", type=int, default=E17Config.num_subactions)
    parser.add_argument("--reference-offsets", type=str, default=",".join(str(v) for v in E17Config.reference_offsets))
    parser.add_argument("--max-ref-backtrack", type=int, default=E17Config.max_ref_backtrack)
    parser.add_argument("--crop-size", type=int, default=E17Config.crop_size)
    parser.add_argument("--output-root", type=Path, default=E17Config.output_root)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    reference_offsets = tuple(int(item.strip()) for item in str(args.reference_offsets).split(",") if str(item).strip())
    run_e17_experiment(
        E17Config(
            video_id=args.video_id,
            num_subactions=args.num_subactions,
            reference_offsets=reference_offsets,
            reference_min_gap=min(reference_offsets) if reference_offsets else E17Config.reference_min_gap,
            reference_max_gap=max(reference_offsets) if reference_offsets else E17Config.reference_max_gap,
            max_ref_backtrack=args.max_ref_backtrack,
            crop_size=args.crop_size,
            output_root=args.output_root,
        )
    )
