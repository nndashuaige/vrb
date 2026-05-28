"""E12 experiment: target-aware clean reference selection and projection.

This script keeps E10's CoTracker / contact-part projection gates, then
changes reference-frame selection to prefer target-visible frames where hands
do not occlude the future contact object or contact part.
"""

from __future__ import annotations

import ast
import io
import json
import math
import os
import sys
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

VRBREPRODUCTION_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLCONFIGDIR", str(VRBREPRODUCTION_ROOT / ".mplconfig"))
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from epic_kitchens.hoa import load_detections
from scipy.signal import savgol_filter
from sklearn.mixture import GaussianMixture

from .contact_point_utils import (
    ContactExtractionConfig,
    build_object_boundary_band,
    build_object_boundary_mask,
    build_object_visible_mask,
    extract_contact_points,
    get_active_hand_bbox,
    get_valid_object_bboxes,
    normalized_bbox_to_pixels,
)
from .e5b_reference_object_consistency import (
    ARTICULATED_OBJECT_TOKENS,
    bbox_iou,
    count_points_near_bbox,
    point_to_bbox_distance,
)
from .label_heatmap_utils import (
    build_label_heatmaps,
    draw_vrb_style_affordance_overlay,
    merge_label_heatmaps,
    save_label_heatmap_outputs,
)
from .pipeline_retention import _json_safe
from .problem3_utils import compute_centered_crop_bbox, compute_crop_hand_overlap_ratio


EXPERIMENT = {
    "experiment_id": "E12",
    "name": "E12_target_aware_clean_reference_projection",
    "description": (
        "继承 E10 的 CoTracker/contact-part local projection 与 part-aware gate，"
        "但在所有投影 gate 通过的 reference candidate 中优先选择目标物体和未来接触部件"
        "未被手遮挡的 clean reference frame。"
    ),
}


@dataclass(frozen=True)
class E12Config:
    video_id: str = "P01_109"
    num_subactions: int = 100
    max_ref_backtrack: int = 360
    early_candidate_offsets: Tuple[int, ...] = (0, 1, 2)
    continuation_rescue_offsets: Tuple[int, ...] = (0, 1)
    max_candidates_per_subaction: int = 12
    object_score_threshold: float = 0.5
    hand_score_threshold: float = 0.5
    min_contact_points: int = 5
    gmm_components: int = 5
    contact_inside_tolerance_px: float = 16.0
    crop_size: int = 150
    reference_hand_overlap_max: float = 0.08
    min_track_confidence: float = 0.18
    sam2_enabled: bool = os.environ.get("E12_SAM2_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
    sam2_model_id: str = os.environ.get("E12_SAM2_MODEL_ID", "facebook/sam2-hiera-tiny")
    sam2_device: str = os.environ.get("E12_SAM2_DEVICE", "cuda")
    sam2_multimask_output: bool = os.environ.get("E12_SAM2_MULTIMASK", "0").strip().lower() in {"1", "true", "yes", "on"}
    use_visor_mask: bool = os.environ.get("E12_USE_VISOR_MASK", "1").strip().lower() not in {"0", "false", "no", "off"}
    cotracker_enabled: bool = os.environ.get("E12_COTRACKER_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
    cotracker_repo: str = os.environ.get("E12_COTRACKER_REPO", "facebookresearch/co-tracker")
    cotracker_model: str = os.environ.get("E12_COTRACKER_MODEL", "cotracker3_online")
    cotracker_max_frames: int = int(os.environ.get("E12_COTRACKER_MAX_FRAMES", "16"))
    cotracker_max_side_px: int = int(os.environ.get("E12_COTRACKER_MAX_SIDE_PX", "384"))
    min_tracking_points: int = int(os.environ.get("E12_MIN_TRACKING_POINTS", "20"))
    max_tracking_points: int = int(os.environ.get("E12_MAX_TRACKING_POINTS", "36"))
    hand_exclusion_pad_px: int = int(os.environ.get("E12_HAND_EXCLUSION_PAD_PX", "6"))
    point_boundary_band_px: int = int(os.environ.get("E12_POINT_BOUNDARY_BAND_PX", "10"))
    point_contact_band_px: int = int(os.environ.get("E12_POINT_CONTACT_BAND_PX", "28"))
    part_boundary_band_px: int = int(os.environ.get("E12_PART_BOUNDARY_BAND_PX", "14"))
    part_contact_dilate_px: int = int(os.environ.get("E12_PART_CONTACT_DILATE_PX", "24"))
    part_hand_dilate_px: int = int(os.environ.get("E12_PART_HAND_DILATE_PX", "18"))
    part_edge_strip_px: int = int(os.environ.get("E12_PART_EDGE_STRIP_PX", "18"))
    part_min_tracking_points: int = int(os.environ.get("E12_PART_MIN_TRACKING_POINTS", "14"))
    part_gate_min_inside_count: int = int(os.environ.get("E12_PART_GATE_MIN_INSIDE_COUNT", "3"))
    part_gate_max_distance_px: float = float(os.environ.get("E12_PART_GATE_MAX_DISTANCE_PX", "20.0"))
    local_projection_max_median_residual: float = float(os.environ.get("E12_LOCAL_PROJECTION_MAX_MEDIAN_RESIDUAL", "10.0"))
    local_projection_min_inlier_ratio: float = float(os.environ.get("E12_LOCAL_PROJECTION_MIN_INLIER_RATIO", "0.4"))
    track_fb_max: float = float(os.environ.get("E12_TRACK_FB_MAX", "3.0"))
    track_min_visible_points: int = int(os.environ.get("E12_TRACK_MIN_VISIBLE_POINTS", "8"))
    affine_ransac_reproj_threshold: float = float(os.environ.get("E12_AFFINE_RANSAC_REPROJ_THRESHOLD", "4.0"))
    affine_min_inliers: int = int(os.environ.get("E12_AFFINE_MIN_INLIERS", "6"))
    affine_min_inlier_ratio: float = float(os.environ.get("E12_AFFINE_MIN_INLIER_RATIO", "0.35"))
    affine_max_median_residual: float = float(os.environ.get("E12_AFFINE_MAX_MEDIAN_RESIDUAL", "12.0"))
    mask_gate_min_inside_count: int = int(os.environ.get("E12_MASK_GATE_MIN_INSIDE_COUNT", "3"))
    mask_gate_boundary_tolerance_px: float = float(os.environ.get("E12_MASK_GATE_BOUNDARY_TOLERANCE_PX", "14.0"))
    mask_gate_centroid_max_distance_px: float = float(os.environ.get("E12_MASK_GATE_CENTROID_MAX_DISTANCE_PX", "18.0"))
    mask_gate_min_visible_points: int = int(os.environ.get("E12_MASK_GATE_MIN_VISIBLE_POINTS", "8"))
    mask_gate_min_inlier_ratio: float = float(os.environ.get("E12_MASK_GATE_MIN_INLIER_RATIO", "0.35"))
    final_quality_keep_threshold: float = float(os.environ.get("E12_FINAL_QUALITY_KEEP_THRESHOLD", "0.34"))
    target_clean_object_overlap_max: float = float(os.environ.get("E12_TARGET_CLEAN_OBJECT_OVERLAP_MAX", "0.01"))
    target_clean_part_overlap_max: float = float(os.environ.get("E12_TARGET_CLEAN_PART_OVERLAP_MAX", "0.01"))
    target_clean_crop_overlap_max: float = float(os.environ.get("E12_TARGET_CLEAN_CROP_OVERLAP_MAX", "0.04"))
    target_visible_min_score: float = float(os.environ.get("E12_TARGET_VISIBLE_MIN_SCORE", "0.45"))
    clean_score_weight: float = float(os.environ.get("E12_CLEAN_SCORE_WEIGHT", "1.0"))
    local_flow_contact_radius_px: int = int(os.environ.get("E7_LOCAL_FLOW_CONTACT_RADIUS_PX", "52"))
    local_flow_object_bbox_pad_px: int = int(os.environ.get("E7_LOCAL_FLOW_OBJECT_BBOX_PAD_PX", "6"))
    local_flow_hand_exclusion_px: int = int(os.environ.get("E7_LOCAL_FLOW_HAND_EXCLUSION_PX", "2"))
    local_flow_min_points: int = int(os.environ.get("E7_LOCAL_FLOW_MIN_POINTS", "8"))
    local_flow_min_inliers: int = int(os.environ.get("E7_LOCAL_FLOW_MIN_INLIERS", "6"))
    local_flow_min_inlier_ratio: float = float(os.environ.get("E7_LOCAL_FLOW_MIN_INLIER_RATIO", "0.42"))
    local_flow_max_step_px: float = float(os.environ.get("E7_LOCAL_FLOW_MAX_STEP_PX", "48"))
    reference_gap_penalty: float = float(os.environ.get("E7_REFERENCE_GAP_PENALTY", "0.006"))
    final_max_tuples_per_subaction: int = int(os.environ.get("E7_FINAL_MAX_TUPLES_PER_SUBACTION", "3"))
    final_frame_dedup_gap: int = int(os.environ.get("E7_FINAL_FRAME_DEDUP_GAP", "2"))
    gpu_flow_enabled: bool = os.environ.get("E7_GPU_FLOW", "1").strip().lower() not in {"0", "false", "no", "off"}
    gpu_flow_device: str = os.environ.get("E7_GPU_FLOW_DEVICE", "cuda")
    gpu_flow_patch_radius: int = 5
    gpu_flow_search_radius: int = 12
    gpu_gray_cache_enabled: bool = os.environ.get("E7_GPU_GRAY_CACHE", "1").strip().lower() not in {"0", "false", "no", "off"}
    gpu_gray_cache_max_frames: int = int(os.environ.get("E7_GPU_GRAY_CACHE_MAX_FRAMES", "20000"))
    gpu_preload_gray_frames: bool = os.environ.get("E7_GPU_PRELOAD_GRAY", "0").strip().lower() in {"1", "true", "yes", "on"}
    gpu_preload_max_frames: int = int(os.environ.get("E7_GPU_PRELOAD_MAX_FRAMES", "20000"))
    gpu_workspace_gb: float = float(os.environ.get("E7_GPU_WORKSPACE_GB", "0"))
    output_root: Path = Path(os.environ.get("E12_OUTPUT_ROOT", str(VRBREPRODUCTION_ROOT / "outputs" / "100subaction-e12")))
    annotation_csv: Path = (
        VRBREPRODUCTION_ROOT
        / "data"
        / "annotations"
        / "epic-kitchens-100-annotations"
        / "EPIC_100_train.csv"
    )
    hoa_pkl: Path = VRBREPRODUCTION_ROOT / "data" / "P01_109.pkl"
    image_dir: Path = VRBREPRODUCTION_ROOT / "data" / "P01_109_frames"


FLOW_RUNTIME_STATS: Counter = Counter()
_TORCH_MODULE = None
_TORCH_IMPORT_ERROR: Optional[str] = None
_GPU_FLOW_STATUS_PRINTED = False
_GPU_WORKSPACE = None
_SAM2_IMAGE_PREDICTOR = None
_SAM2_INIT_ERROR: Optional[str] = None
_COTRACKER_MODEL = None
_COTRACKER_INIT_STATUS: Optional[str] = None
_COTRACKER_INIT_ERROR: Optional[str] = None


def silent_call(func, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def smooth_contact_binary(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) < 7:
        return values.astype(int)
    return (savgol_filter(values, window_length=7, polyorder=2) > 0.75).astype(int)


def build_contact_arrays(detections) -> Tuple[np.ndarray, np.ndarray]:
    from epic_kitchens.hoa.types import HandState

    left = np.zeros(len(detections), dtype=np.float32)
    right = np.zeros(len(detections), dtype=np.float32)
    for frame_idx, frame_det in enumerate(detections):
        for hand in getattr(frame_det, "hands", []):
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


def get_subactions(config: E12Config) -> pd.DataFrame:
    df = pd.read_csv(config.annotation_csv)
    subactions = df[df["video_id"] == config.video_id].copy()
    subactions = subactions.sort_values(["start_frame", "stop_frame", "narration_id"]).head(config.num_subactions)
    return subactions.reset_index(drop=True)


def subaction_bounds(row: pd.Series, num_frames: int) -> Tuple[int, int]:
    start_0 = max(0, int(row["start_frame"]) - 1)
    stop_0 = min(num_frames - 1, int(row["stop_frame"]) - 1)
    return start_0, stop_0


def compute_subaction_frame_coverage(subactions: pd.DataFrame) -> Dict[str, int]:
    total_annotated_frames = int((subactions["stop_frame"] - subactions["start_frame"] + 1).sum())
    covered = set()
    for _, item in subactions.iterrows():
        start_0 = max(0, int(item["start_frame"]) - 1)
        stop_0 = int(item["stop_frame"]) - 1
        if start_0 <= stop_0:
            covered.update(range(start_0, stop_0 + 1))
    return {"total_annotated_frames": total_annotated_frames, "unique_covered_frames": int(len(covered))}


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


def compute_pre_contact_gap_start(run_start: int, left_binary: np.ndarray, right_binary: np.ndarray) -> int:
    idx = int(run_start) - 1
    if idx < 0:
        return int(run_start)
    contact_any = (np.asarray(left_binary, dtype=int) == 1) | (np.asarray(right_binary, dtype=int) == 1)
    if contact_any[idx]:
        return int(run_start)
    while idx - 1 >= 0 and not contact_any[idx - 1]:
        idx -= 1
    return int(idx)


def contact_frame_candidates_for_subaction(
    row: pd.Series,
    left_binary: np.ndarray,
    right_binary: np.ndarray,
    num_frames: int,
) -> List[Dict[str, Any]]:
    start_0, stop_0 = subaction_bounds(row, num_frames)
    candidates: List[Dict[str, Any]] = []
    for frame_idx in range(start_0, stop_0 + 1):
        if left_binary[frame_idx] == 1:
            candidates.append({"frame": int(frame_idx), "hand": "left"})
        if right_binary[frame_idx] == 1:
            candidates.append({"frame": int(frame_idx), "hand": "right"})
    return candidates


def build_candidates_for_subaction(
    row: pd.Series,
    all_runs: Sequence[Dict[str, Any]],
    left_binary: np.ndarray,
    right_binary: np.ndarray,
    num_frames: int,
    config: E12Config,
) -> Tuple[List[Dict[str, Any]], str, str, int]:
    start_0, stop_0 = subaction_bounds(row, num_frames)
    raw_candidates = contact_frame_candidates_for_subaction(row, left_binary, right_binary, num_frames)
    candidates: List[Dict[str, Any]] = []
    runs_in_subaction = []
    raw_contact_present = False
    for run in all_runs:
        run_start = int(run["start_frame_0_based"])
        run_end = int(run["end_frame_0_based"])
        if run_end < start_0 or run_start > stop_0:
            continue
        raw_contact_present = True
        if start_0 <= run_start <= stop_0:
            runs_in_subaction.append(dict(run))

    for episode_id, run in enumerate(runs_in_subaction, start=1):
        run_start = int(run["start_frame_0_based"])
        run_end = int(run["end_frame_0_based"])
        pre_gap_start = compute_pre_contact_gap_start(run_start, left_binary, right_binary)
        pre_gap_end = run_start - 1
        search_start = max(0, run_start - int(config.max_ref_backtrack), pre_gap_start)
        search_end = pre_gap_end
        for offset in config.early_candidate_offsets:
            frame_idx = run_start + int(offset)
            if frame_idx > run_end or frame_idx > stop_0:
                continue
            candidates.append(
                {
                    "frame": int(frame_idx),
                    "hand": run["hand"],
                    "candidate_source": "new_contact_episode",
                    "episode_id_within_subaction": int(episode_id),
                    "episode_contact_start_frame_0_based": int(run_start),
                    "episode_contact_end_frame_0_based": int(run_end),
                    "candidate_frame_offset_from_episode_start": int(offset),
                    "pre_contact_gap_start_frame_0_based": int(pre_gap_start),
                    "pre_contact_gap_end_frame_0_based": int(pre_gap_end),
                    "reference_search_window_start_0_based": int(search_start),
                    "reference_search_window_end_0_based": int(search_end),
                }
            )

    if candidates:
        return candidates[: config.max_candidates_per_subaction], "new_contact_episode", "smoothed_contact_run_start_within_subaction", len(raw_candidates)

    continuation_runs = []
    for run in all_runs:
        run_start = int(run["start_frame_0_based"])
        run_end = int(run["end_frame_0_based"])
        if run_end < start_0 or run_start > stop_0:
            continue
        if run_start < start_0 <= run_end:
            continuation_runs.append(dict(run))

    rescue_candidates: List[Dict[str, Any]] = []
    for episode_id, run in enumerate(sorted(continuation_runs, key=lambda item: (int(item["start_frame_0_based"]), item["hand"])), start=1):
        run_start = int(run["start_frame_0_based"])
        run_end = int(run["end_frame_0_based"])
        pre_gap_start = compute_pre_contact_gap_start(run_start, left_binary, right_binary)
        pre_gap_end = run_start - 1
        search_start = max(0, run_start - int(config.max_ref_backtrack), pre_gap_start)
        search_end = pre_gap_end
        continuation_entry = max(start_0, run_start)
        for offset in config.continuation_rescue_offsets:
            frame_idx = continuation_entry + int(offset)
            if frame_idx > run_end or frame_idx > stop_0:
                continue
            rescue_candidates.append(
                {
                    "frame": int(frame_idx),
                    "hand": run["hand"],
                    "candidate_source": "continuation_entry_rescue",
                    "episode_id_within_subaction": int(episode_id),
                    "episode_contact_start_frame_0_based": int(run_start),
                    "episode_contact_end_frame_0_based": int(run_end),
                    "candidate_frame_offset_from_episode_start": int(frame_idx - run_start),
                    "pre_contact_gap_start_frame_0_based": int(pre_gap_start),
                    "pre_contact_gap_end_frame_0_based": int(pre_gap_end),
                    "reference_search_window_start_0_based": int(search_start),
                    "reference_search_window_end_0_based": int(search_end),
                }
            )
    if rescue_candidates:
        return rescue_candidates[: config.max_candidates_per_subaction], "continuation_entry_rescue", "continuation_contact_run_rescue_at_subaction_entry", len(raw_candidates)

    reason = "continuation_contact_run" if raw_contact_present else "no_smoothed_contact_in_subaction"
    return [], "no_candidate", reason, len(raw_candidates)


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


def bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_center(bbox: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


def clip_bbox(bbox: Sequence[float], width: int, height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = float(np.clip(x1, 0, width - 1))
    y1 = float(np.clip(y1, 0, height - 1))
    x2 = float(np.clip(x2, x1 + 1, width))
    y2 = float(np.clip(y2, y1 + 1, height))
    return x1, y1, x2, y2


def norm_bbox_to_px(bbox: Sequence[float], width: int, height: int) -> Tuple[float, float, float, float]:
    return (
        float(bbox[0]) * width,
        float(bbox[1]) * height,
        float(bbox[2]) * width,
        float(bbox[3]) * height,
    )


def px_bbox_to_norm(bbox: Sequence[float], width: int, height: int) -> List[float]:
    return [float(bbox[0]) / width, float(bbox[1]) / height, float(bbox[2]) / width, float(bbox[3]) / height]


def expand_bbox(bbox: Sequence[float], pad: float, width: int, height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return clip_bbox((x1 - pad, y1 - pad, x2 + pad, y2 + pad), width, height)


def hand_overlap_ratio(hand_bbox: Sequence[float], object_bbox: Sequence[float]) -> float:
    hx1, hy1, hx2, hy2 = [float(v) for v in hand_bbox]
    ox1, oy1, ox2, oy2 = [float(v) for v in object_bbox]
    ix1, iy1 = max(hx1, ox1), max(hy1, oy1)
    ix2, iy2 = min(hx2, ox2), min(hy2, oy2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return float(inter / max(1e-6, bbox_area(object_bbox)))


class E12FrameCache:
    def __init__(self, image_dir: Path, config: Optional[E12Config] = None):
        self.image_dir = Path(image_dir)
        self.config = config
        self.bgr_cache: Dict[int, np.ndarray] = {}
        self.gray_cache: Dict[int, np.ndarray] = {}
        self.gpu_gray_cache: Dict[int, Any] = {}
        self.mask_cache: Dict[Tuple[str, int, Tuple[int, int, int, int]], Dict[str, Any]] = {}

    def load_bgr(self, frame_idx: int) -> np.ndarray:
        frame_idx = int(frame_idx)
        if frame_idx not in self.bgr_cache:
            path = self.image_dir / f"frame_{frame_idx + 1:010d}.jpg"
            img = cv2.imread(str(path))
            if img is None:
                raise FileNotFoundError(f"missing frame image: {path}")
            self.bgr_cache[frame_idx] = img
        return self.bgr_cache[frame_idx]

    def load_gray(self, frame_idx: int) -> np.ndarray:
        frame_idx = int(frame_idx)
        if frame_idx not in self.gray_cache:
            self.gray_cache[frame_idx] = cv2.cvtColor(self.load_bgr(frame_idx), cv2.COLOR_BGR2GRAY)
        return self.gray_cache[frame_idx]

    def load_gray_torch(self, frame_idx: int, config: E12Config):
        torch = get_torch_for_gpu_flow(config)
        if torch is None or not config.gpu_gray_cache_enabled:
            return None
        frame_idx = int(frame_idx)
        cached = self.gpu_gray_cache.get(frame_idx)
        if cached is not None:
            FLOW_RUNTIME_STATS["gpu_gray_cache_hit"] += 1
            return cached
        if len(self.gpu_gray_cache) >= int(config.gpu_gray_cache_max_frames):
            FLOW_RUNTIME_STATS["gpu_gray_cache_limit_reached"] += 1
            return None
        gray = self.load_gray(frame_idx)
        tensor = torch.as_tensor(gray, dtype=torch.float32, device=torch.device(config.gpu_flow_device))[None, None] / 255.0
        self.gpu_gray_cache[frame_idx] = tensor
        FLOW_RUNTIME_STATS["gpu_gray_cache_miss"] += 1
        FLOW_RUNTIME_STATS["gpu_gray_cache_frames"] = len(self.gpu_gray_cache)
        return tensor

    def available_frame_indices(self, max_frames: Optional[int] = None) -> List[int]:
        items = []
        for path in sorted(self.image_dir.glob("frame_*.jpg")):
            try:
                idx = int(path.stem.split("_")[-1]) - 1
            except ValueError:
                continue
            items.append(idx)
            if max_frames is not None and len(items) >= int(max_frames):
                break
        return items


def is_articulated_object_candidate(noun: Any, all_nouns: Any = None, narration: Any = None) -> bool:
    text = " ".join([str(noun or ""), str(all_nouns or ""), str(narration or "")]).lower()
    return any(token in text for token in ARTICULATED_OBJECT_TOKENS)


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(np.asarray(mask) > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bbox_to_int_tuple(bbox: Sequence[float]) -> Tuple[int, int, int, int]:
    return tuple(int(round(float(v))) for v in bbox)  # type: ignore[return-value]


def _clip_int_bbox(bbox: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def _parse_box_like(value: Any) -> Optional[Tuple[float, float, float, float]]:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 4:
        return tuple(float(v) for v in value)
    return None


def _mask_to_bbox_px(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    bbox = _mask_bbox(mask)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return float(x1), float(y1), float(x2), float(y2)


def _make_binary_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=np.uint8)
    arr = np.asarray(mask)
    if arr.shape[:2] != shape:
        arr = cv2.resize(arr.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (arr > 0).astype(np.uint8) * 255


def _candidates_from_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask > 0)
    if len(pts) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return pts[:, ::-1].astype(np.float32)


def _sample_points_from_coords(coords_xy: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    coords = np.asarray(coords_xy, dtype=np.float32).reshape(-1, 2)
    if len(coords) == 0 or count <= 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(coords) <= count:
        return coords.astype(np.float32)
    idx = rng.choice(len(coords), size=int(count), replace=False)
    return coords[idx].astype(np.float32)


def _distance_to_bbox(points_xy: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if len(pts) == 0:
        return np.empty((0,), dtype=np.float32)
    return np.asarray([point_to_bbox_distance(pt, bbox) for pt in pts], dtype=np.float32)


SMALL_WHOLE_OBJECT_TOKENS = {
    "potato",
    "slice:potato",
    "bit:potato",
    "egg",
    "onion",
}
HANDLE_LIKE_TOKENS = {
    "pan",
    "pot",
    "knife",
    "peeler",
    "blade:peeler",
}
PANEL_EDGE_TOKENS = {
    "drawer",
    "door",
    "cupboard",
    "freezer",
    "fridge",
}
TAP_TOKENS = {"tap"}


def _noun_text(noun: Any, all_nouns: Any = None, narration: Any = None) -> str:
    return " ".join(str(v or "") for v in (noun, all_nouns, narration)).lower()


def classify_contact_part_prior(noun: Any, all_nouns: Any = None, narration: Any = None) -> str:
    text = _noun_text(noun, all_nouns, narration)
    if any(token in text for token in SMALL_WHOLE_OBJECT_TOKENS):
        return "small_object_whole_mask_allowed"
    if any(token in text for token in HANDLE_LIKE_TOKENS):
        return "handle_like_edge_or_endpoint"
    if any(token in text for token in PANEL_EDGE_TOKENS):
        return "panel_edge_or_handle"
    if any(token in text for token in TAP_TOKENS):
        return "tap_knob_or_spout_edge"
    return "contact_local_boundary"


def _points_to_mask(points_xy: np.ndarray, shape: Tuple[int, int], radius_px: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return mask
    radius = max(1, int(radius_px))
    for x, y in points:
        xi, yi = int(round(float(x))), int(round(float(y)))
        if 0 <= xi < shape[1] and 0 <= yi < shape[0]:
            cv2.circle(mask, (xi, yi), radius, 255, -1)
    return mask


def _bbox_region_mask(bbox: Optional[Sequence[float]], shape: Tuple[int, int], pad_px: int = 0) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if bbox is None:
        return mask
    h, w = shape
    x1, y1, x2, y2 = _clip_int_bbox(bbox, w, h)
    pad = max(0, int(pad_px))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    mask[y1:y2, x1:x2] = 255
    return mask


def _extreme_boundary_mask(object_mask: np.ndarray, contact_center: np.ndarray, band_px: int) -> np.ndarray:
    mask = (np.asarray(object_mask) > 0).astype(np.uint8)
    h, w = mask.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    bbox = _mask_bbox(mask)
    if bbox is None:
        return out
    x1, y1, x2, y2 = bbox
    band = max(3, int(band_px))
    cx, cy = float(contact_center[0]), float(contact_center[1])
    distances = {
        "left": abs(cx - x1),
        "right": abs(cx - x2),
        "top": abs(cy - y1),
        "bottom": abs(cy - y2),
    }
    ordered = [name for name, _ in sorted(distances.items(), key=lambda kv: kv[1])[:2]]
    if "left" in ordered:
        out[y1:y2, x1:min(x2, x1 + band)] = 255
    if "right" in ordered:
        out[y1:y2, max(x1, x2 - band):x2] = 255
    if "top" in ordered:
        out[y1:min(y2, y1 + band), x1:x2] = 255
    if "bottom" in ordered:
        out[max(y1, y2 - band):y2, x1:x2] = 255
    return cv2.bitwise_and(out, mask * 255)


def build_contact_part_mask(
    *,
    object_mask: np.ndarray,
    contact_means: np.ndarray,
    raw_contact_points: Optional[np.ndarray],
    hand_bbox: Optional[Sequence[float]],
    noun: Any,
    all_nouns: Any = None,
    narration: Any = None,
    config: E12Config,
) -> Dict[str, Any]:
    object_binary = (np.asarray(object_mask) > 0).astype(np.uint8)
    h, w = object_binary.shape[:2]
    part_prior = classify_contact_part_prior(noun, all_nouns, narration)
    object_area = int(object_binary.sum())
    if object_area == 0:
        return {
            "mask": np.zeros((h, w), dtype=np.uint8),
            "prior": part_prior,
            "area": 0,
            "area_ratio": 0.0,
            "source": "empty_object_mask",
        }

    means = np.asarray(contact_means, dtype=np.float32).reshape(-1, 2)
    raw = np.asarray(raw_contact_points if raw_contact_points is not None else np.empty((0, 2)), dtype=np.float32).reshape(-1, 2)
    contact_center = means.mean(axis=0) if len(means) else (raw.mean(axis=0) if len(raw) else np.array([w * 0.5, h * 0.5], dtype=np.float32))

    boundary_band = build_object_boundary_band(object_binary * 255, float(config.part_boundary_band_px))
    mean_mask = _points_to_mask(means, (h, w), int(config.part_contact_dilate_px))
    raw_mask = _points_to_mask(raw, (h, w), max(6, int(config.part_contact_dilate_px * 0.65)))
    hand_region = _bbox_region_mask(hand_bbox, (h, w), pad_px=int(config.part_hand_dilate_px))
    hand_boundary_local = cv2.bitwise_and(boundary_band, hand_region)
    contact_local = cv2.bitwise_or(mean_mask, raw_mask)
    contact_local = cv2.bitwise_and(contact_local, object_binary * 255)
    contact_boundary = cv2.bitwise_and(cv2.bitwise_or(contact_local, hand_boundary_local), boundary_band)

    prior_mask = np.zeros((h, w), dtype=np.uint8)
    if part_prior == "small_object_whole_mask_allowed":
        prior_mask = object_binary * 255
    elif part_prior in {"handle_like_edge_or_endpoint", "panel_edge_or_handle", "tap_knob_or_spout_edge"}:
        extreme = _extreme_boundary_mask(object_binary * 255, contact_center, int(config.part_edge_strip_px))
        prior_mask = cv2.bitwise_or(extreme, boundary_band)
        prior_mask = cv2.bitwise_and(prior_mask, cv2.bitwise_or(mean_mask, _bbox_region_mask(
            (contact_center[0] - 2.5 * config.part_contact_dilate_px, contact_center[1] - 2.5 * config.part_contact_dilate_px,
             contact_center[0] + 2.5 * config.part_contact_dilate_px, contact_center[1] + 2.5 * config.part_contact_dilate_px),
            (h, w),
        )))
    else:
        prior_mask = boundary_band

    part_mask = contact_local.copy()
    part_mask = cv2.bitwise_or(part_mask, contact_boundary)
    if int(np.sum(prior_mask > 0)) > 0:
        part_mask = cv2.bitwise_or(part_mask, cv2.bitwise_and(boundary_band, prior_mask))
        part_mask = cv2.bitwise_or(part_mask, cv2.bitwise_and(contact_local, prior_mask))
    if int(np.sum(part_mask > 0)) < int(config.part_min_tracking_points):
        part_mask = cv2.bitwise_or(part_mask, cv2.bitwise_and(contact_local, object_binary * 255))
    if int(np.sum(part_mask > 0)) < int(config.part_min_tracking_points) and part_prior != "small_object_whole_mask_allowed":
        local_bbox = (
            contact_center[0] - 2.0 * config.part_contact_dilate_px,
            contact_center[1] - 2.0 * config.part_contact_dilate_px,
            contact_center[0] + 2.0 * config.part_contact_dilate_px,
            contact_center[1] + 2.0 * config.part_contact_dilate_px,
        )
        part_mask = cv2.bitwise_or(part_mask, cv2.bitwise_and(boundary_band, _bbox_region_mask(local_bbox, (h, w))))
    if int(np.sum(part_mask > 0)) < int(config.part_min_tracking_points):
        part_mask = object_binary * 255 if part_prior == "small_object_whole_mask_allowed" else cv2.bitwise_and(boundary_band, object_binary * 255)
    part_mask = cv2.bitwise_and(part_mask, object_binary * 255)
    area = int(np.sum(part_mask > 0))
    return {
        "mask": part_mask.astype(np.uint8),
        "prior": part_prior,
        "area": area,
        "area_ratio": float(area / max(1, object_area)),
        "source": "contact_gmm_raw_boundary_prior",
        "object_area": object_area,
        "boundary_area": int(np.sum(boundary_band > 0)),
        "contact_local_area": int(np.sum(contact_local > 0)),
        "hand_boundary_area": int(np.sum(hand_boundary_local > 0)),
    }


def _ensure_sam2_predictor(config: E12Config):
    global _SAM2_IMAGE_PREDICTOR, _SAM2_INIT_ERROR
    if not config.sam2_enabled:
        return None
    if _SAM2_IMAGE_PREDICTOR is not None:
        return _SAM2_IMAGE_PREDICTOR
    if _SAM2_INIT_ERROR is not None:
        return None
    try:
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = config.sam2_device
        torch_mod = get_torch_for_gpu_flow(config)
        if torch_mod is None and device.startswith("cuda"):
            device = "cpu"
        model = build_sam2_hf(config.sam2_model_id, device=device, apply_postprocessing=True)
        _SAM2_IMAGE_PREDICTOR = SAM2ImagePredictor(model)
        return _SAM2_IMAGE_PREDICTOR
    except Exception as exc:
        _SAM2_INIT_ERROR = f"{type(exc).__name__}:{exc}"
        return None


def _ensure_cotracker_model(config: E12Config):
    global _COTRACKER_MODEL, _COTRACKER_INIT_ERROR, _COTRACKER_INIT_STATUS
    if not config.cotracker_enabled:
        _COTRACKER_INIT_STATUS = "disabled"
        return None
    if _COTRACKER_MODEL is not None:
        _COTRACKER_INIT_STATUS = "ready"
        return _COTRACKER_MODEL
    if _COTRACKER_INIT_ERROR is not None:
        _COTRACKER_INIT_STATUS = "failed"
        return None
    try:
        torch_mod = get_torch_for_gpu_flow(config)
        if torch_mod is None:
            torch_mod = _TORCH_MODULE
        if torch_mod is None:
            import torch as torch_mod  # type: ignore
        _COTRACKER_MODEL = torch_mod.hub.load(config.cotracker_repo, config.cotracker_model, trust_repo=True)
        device = torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
        if hasattr(_COTRACKER_MODEL, "to"):
            _COTRACKER_MODEL = _COTRACKER_MODEL.to(device)
        if hasattr(_COTRACKER_MODEL, "eval"):
            _COTRACKER_MODEL.eval()
        _COTRACKER_INIT_STATUS = "ready"
        return _COTRACKER_MODEL
    except Exception as exc:
        _COTRACKER_INIT_STATUS = "failed"
        _COTRACKER_INIT_ERROR = f"{type(exc).__name__}:{exc}"
        return None


def _load_visor_mask(frame_idx: int, image_bgr: np.ndarray, active_object_bbox: Sequence[float], config: E12Config) -> Dict[str, Any]:
    h, w = image_bgr.shape[:2]
    candidates: List[Path] = []
    visor_root = os.environ.get("E12_VISOR_MASK_ROOT")
    if visor_root:
        candidates.append(Path(visor_root))
    candidates.extend([
        VRBREPRODUCTION_ROOT / "data" / "visor_masks" / config.video_id,
        VRBREPRODUCTION_ROOT / "data" / "VISOR" / config.video_id,
        VRBREPRODUCTION_ROOT / "data" / "visor" / config.video_id,
    ])
    for root in candidates:
        if not root.exists():
            continue
        patterns = [
            root / f"frame_{frame_idx + 1:010d}.png",
            root / f"frame_{frame_idx + 1:010d}.jpg",
            root / f"{frame_idx + 1:010d}.png",
            root / f"{frame_idx + 1:010d}.jpg",
        ]
        for path in patterns:
            if not path.exists():
                continue
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = _make_binary_mask(mask, (h, w))
            if int(mask.sum()) > 0:
                return {"mask": mask, "mask_source": f"visor:{path}"}
    return {"mask": None, "mask_source": None}


def get_object_mask(frame_idx: int, image_bgr: np.ndarray, active_object_bbox: Sequence[float], noun: Any, config: E12Config) -> Dict[str, Any]:
    h, w = image_bgr.shape[:2]
    active_bbox = _parse_box_like(active_object_bbox) or (0.0, 0.0, float(w), float(h))
    bbox_px = _clip_int_bbox(active_bbox, w, h)

    if config.use_visor_mask:
        visor = _load_visor_mask(frame_idx, image_bgr, bbox_px, config)
        if visor["mask"] is not None:
            return {
                "mask": visor["mask"],
                "mask_source": visor["mask_source"],
                "mask_bbox": _mask_to_bbox_px(visor["mask"]),
                "backend": "visor",
            }

    predictor = _ensure_sam2_predictor(config)
    if predictor is not None:
        try:
            predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            masks, scores, _ = predictor.predict(
                box=np.asarray(bbox_px, dtype=np.float32),
                multimask_output=bool(config.sam2_multimask_output),
                normalize_coords=False,
            )
            masks = np.asarray(masks)
            if masks.ndim == 2:
                masks = masks[None, ...]
            scores = np.asarray(scores).reshape(-1) if scores is not None else np.ones(len(masks), dtype=np.float32)
            best_idx = int(np.argmax(scores)) if len(scores) else 0
            mask = _make_binary_mask(masks[best_idx], (h, w))
            if int(mask.sum()) > 0:
                return {
                    "mask": mask,
                    "mask_source": f"sam2:{config.sam2_model_id}",
                    "mask_bbox": _mask_to_bbox_px(mask),
                    "backend": "sam2",
                    "sam2_score": float(scores[best_idx]) if len(scores) else None,
                }
        except Exception as exc:
            return {
                "mask": None,
                "mask_source": f"sam2_failed:{type(exc).__name__}",
                "mask_bbox": None,
                "backend": "sam2_failed",
                "error": str(exc),
            }

    hand_skin_config = ContactExtractionConfig(
        hand_score_threshold=config.hand_score_threshold,
        object_score_threshold=config.object_score_threshold,
        use_object_mask=True,
        use_object_boundary=True,
        boundary_distance_px=8.0,
        object_bbox_shrink_px=2,
        grabcut_iterations=3,
        grabcut_bg_margin_px=6,
        hand_exclusion_dilate_px=max(1, int(config.hand_exclusion_pad_px)),
        project_points_to_object_boundary=True,
    )
    hand_skin_mask = np.zeros((h, w), dtype=np.uint8)
    fallback_mask, bbox_mask = build_object_visible_mask(
        image_bgr,
        [active_bbox[0] / w, active_bbox[1] / h, active_bbox[2] / w, active_bbox[3] / h],
        hand_skin_mask=hand_skin_mask,
        config=hand_skin_config,
    )
    if int(fallback_mask.sum()) > 0:
        return {
            "mask": fallback_mask,
            "mask_source": "grabcut_fallback",
            "mask_bbox": _mask_to_bbox_px(fallback_mask),
            "backend": "grabcut_fallback",
        }
    return {
        "mask": bbox_mask,
        "mask_source": "bbox_fallback",
        "mask_bbox": _mask_to_bbox_px(bbox_mask),
        "backend": "bbox_fallback",
    }


def sample_object_points_for_tracking(
    object_mask: np.ndarray,
    contact_means: np.ndarray,
    hand_bbox: Optional[Sequence[float]],
    config: E12Config,
) -> Dict[str, Any]:
    mask = (np.asarray(object_mask) > 0).astype(np.uint8)
    h, w = mask.shape[:2]
    if hand_bbox is not None:
        hx1, hy1, hx2, hy2 = _clip_int_bbox(hand_bbox, w, h)
        hand_mask = np.zeros_like(mask, dtype=np.uint8)
        hand_mask[max(0, hy1 - int(config.hand_exclusion_pad_px)):min(h, hy2 + int(config.hand_exclusion_pad_px)), max(0, hx1 - int(config.hand_exclusion_pad_px)):min(w, hx2 + int(config.hand_exclusion_pad_px))] = 1
        mask = np.where(hand_mask > 0, 0, mask)
    boundary_band = build_object_boundary_band(mask * 255, config.point_boundary_band_px)
    interior_mask = np.where((mask > 0) & (boundary_band == 0), 255, 0).astype(np.uint8)

    rng = np.random.default_rng(42)
    contact_centroid = np.asarray(contact_means, dtype=np.float32).reshape(-1, 2).mean(axis=0) if len(contact_means) else np.array([w * 0.5, h * 0.5], dtype=np.float32)

    object_coords = _candidates_from_mask(mask)
    boundary_coords = _candidates_from_mask(boundary_band)
    interior_coords = _candidates_from_mask(interior_mask)

    dist_to_contact = _distance_to_bbox(object_coords, [contact_centroid[0] - config.point_contact_band_px, contact_centroid[1] - config.point_contact_band_px, contact_centroid[0] + config.point_contact_band_px, contact_centroid[1] + config.point_contact_band_px]) if len(object_coords) else np.empty((0,), dtype=np.float32)
    contact_near_coords = object_coords[dist_to_contact <= float(config.point_contact_band_px)] if len(object_coords) else np.empty((0, 2), dtype=np.float32)

    hand_exclusion_mask = np.zeros_like(mask, dtype=np.uint8)
    if hand_bbox is not None:
        hx1, hy1, hx2, hy2 = _clip_int_bbox(hand_bbox, w, h)
        hand_exclusion_mask[max(0, hy1 - int(config.hand_exclusion_pad_px)):min(h, hy2 + int(config.hand_exclusion_pad_px)), max(0, hx1 - int(config.hand_exclusion_pad_px)):min(w, hx2 + int(config.hand_exclusion_pad_px))] = 1
    valid_object_coords = object_coords[hand_exclusion_mask[object_coords[:, 1].astype(int), object_coords[:, 0].astype(int)] == 0] if len(object_coords) else object_coords
    contact_near_coords = valid_object_coords[_distance_to_bbox(valid_object_coords, [contact_centroid[0] - config.point_contact_band_px, contact_centroid[1] - config.point_contact_band_px, contact_centroid[0] + config.point_contact_band_px, contact_centroid[1] + config.point_contact_band_px]) <= float(config.point_contact_band_px)] if len(valid_object_coords) else np.empty((0, 2), dtype=np.float32)
    boundary_coords = boundary_coords[hand_exclusion_mask[boundary_coords[:, 1].astype(int), boundary_coords[:, 0].astype(int)] == 0] if len(boundary_coords) else boundary_coords
    interior_coords = interior_coords[hand_exclusion_mask[interior_coords[:, 1].astype(int), interior_coords[:, 0].astype(int)] == 0] if len(interior_coords) else interior_coords

    target_n = max(int(config.min_tracking_points), min(int(config.max_tracking_points), 36))
    n_contact = min(max(8, target_n // 2), len(contact_near_coords))
    n_boundary = min(max(6, target_n // 3), len(boundary_coords))
    n_interior = max(0, target_n - n_contact - n_boundary)
    sampled = []
    sources = []
    if n_contact > 0:
        pts = _sample_points_from_coords(contact_near_coords, n_contact, rng)
        sampled.append(pts)
        sources.extend(["contact_near_mask"] * len(pts))
    if n_boundary > 0:
        pts = _sample_points_from_coords(boundary_coords, n_boundary, rng)
        sampled.append(pts)
        sources.extend(["boundary"] * len(pts))
    if n_interior > 0:
        pts = _sample_points_from_coords(interior_coords, n_interior, rng)
        sampled.append(pts)
        sources.extend(["interior"] * len(pts))
    points = np.concatenate(sampled, axis=0) if sampled else np.empty((0, 2), dtype=np.float32)
    if len(points) > target_n:
        idx = rng.choice(len(points), size=target_n, replace=False)
        points = points[idx]
        sources = [sources[i] for i in idx.tolist()]
    return {
        "points_xy": points.astype(np.float32),
        "sources": sources,
        "count": int(len(points)),
        "boundary_count": int(len(boundary_coords)),
        "contact_near_count": int(len(contact_near_coords)),
        "interior_count": int(len(interior_coords)),
        "object_mask_area": int(mask.sum()),
        "boundary_mask_area": int(boundary_band.sum() // 255),
    }


def sample_contact_part_points_for_tracking(
    object_mask: np.ndarray,
    contact_part_mask: np.ndarray,
    contact_means: np.ndarray,
    hand_bbox: Optional[Sequence[float]],
    config: E12Config,
) -> Dict[str, Any]:
    object_binary = (np.asarray(object_mask) > 0).astype(np.uint8)
    part_binary = (np.asarray(contact_part_mask) > 0).astype(np.uint8)
    h, w = object_binary.shape[:2]
    part_binary = np.where(object_binary > 0, part_binary, 0).astype(np.uint8)

    hand_exclusion_mask = np.zeros_like(object_binary, dtype=np.uint8)
    if hand_bbox is not None:
        hx1, hy1, hx2, hy2 = _clip_int_bbox(hand_bbox, w, h)
        pad = int(config.hand_exclusion_pad_px)
        hand_exclusion_mask[max(0, hy1 - pad):min(h, hy2 + pad), max(0, hx1 - pad):min(w, hx2 + pad)] = 1
    part_no_hand = np.where(hand_exclusion_mask > 0, 0, part_binary).astype(np.uint8)

    boundary_band = (build_object_boundary_band(object_binary * 255, config.point_boundary_band_px) > 0).astype(np.uint8)
    boundary_no_hand = np.where((boundary_band > 0) & (hand_exclusion_mask == 0) & (object_binary > 0), 1, 0).astype(np.uint8)

    means = np.asarray(contact_means, dtype=np.float32).reshape(-1, 2)
    contact_centroid = means.mean(axis=0) if len(means) else np.array([w * 0.5, h * 0.5], dtype=np.float32)
    local_bbox = [
        contact_centroid[0] - 1.5 * config.point_contact_band_px,
        contact_centroid[1] - 1.5 * config.point_contact_band_px,
        contact_centroid[0] + 1.5 * config.point_contact_band_px,
        contact_centroid[1] + 1.5 * config.point_contact_band_px,
    ]
    local_boundary = np.where((_bbox_region_mask(local_bbox, (h, w)) > 0) & (boundary_no_hand > 0), 1, 0).astype(np.uint8)

    rng = np.random.default_rng(43)
    target_n = max(int(config.min_tracking_points), min(int(config.max_tracking_points), 36))
    part_coords = _candidates_from_mask(part_no_hand * 255)
    local_boundary_coords = _candidates_from_mask(local_boundary * 255)
    boundary_coords = _candidates_from_mask(boundary_no_hand * 255)

    sampled: List[np.ndarray] = []
    sources: List[str] = []
    n_part = min(len(part_coords), max(int(config.part_min_tracking_points), int(round(target_n * 0.72))))
    if n_part > 0:
        pts = _sample_points_from_coords(part_coords, n_part, rng)
        sampled.append(pts)
        sources.extend(["contact_part"] * len(pts))

    remaining = target_n - sum(len(x) for x in sampled)
    if remaining > 0 and len(local_boundary_coords) > 0:
        pts = _sample_points_from_coords(local_boundary_coords, min(remaining, len(local_boundary_coords)), rng)
        sampled.append(pts)
        sources.extend(["contact_local_boundary"] * len(pts))

    remaining = target_n - sum(len(x) for x in sampled)
    if remaining > 0 and len(boundary_coords) > 0:
        pts = _sample_points_from_coords(boundary_coords, min(remaining, len(boundary_coords)), rng)
        sampled.append(pts)
        sources.extend(["whole_object_boundary_fallback"] * len(pts))

    points = np.concatenate(sampled, axis=0) if sampled else np.empty((0, 2), dtype=np.float32)
    if len(points) > target_n:
        idx = rng.choice(len(points), size=target_n, replace=False)
        points = points[idx]
        sources = [sources[i] for i in idx.tolist()]

    return {
        "points_xy": points.astype(np.float32),
        "sources": sources,
        "count": int(len(points)),
        "part_point_count": int(sum(1 for s in sources if s == "contact_part")),
        "local_boundary_point_count": int(sum(1 for s in sources if s == "contact_local_boundary")),
        "whole_boundary_fallback_point_count": int(sum(1 for s in sources if s == "whole_object_boundary_fallback")),
        "part_mask_area": int(part_binary.sum()),
        "part_no_hand_area": int(part_no_hand.sum()),
        "boundary_mask_area": int(boundary_no_hand.sum()),
        "object_mask_area": int(object_binary.sum()),
    }


def track_object_points_to_reference(
    frame_indices: Sequence[int],
    points_xy: np.ndarray,
    config: E12Config,
    cache: E12FrameCache,
) -> Dict[str, Any]:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    frame_indices = [int(i) for i in frame_indices]
    if len(pts) == 0 or len(frame_indices) < 2:
        return {
            "passed": False,
            "reason": "insufficient_tracking_setup",
            "backend": "fallback_lk",
            "cotracker_init_status": _COTRACKER_INIT_STATUS or ("disabled" if not config.cotracker_enabled else "not_attempted"),
            "cotracker_init_error": _COTRACKER_INIT_ERROR,
            "ref_points_xy": np.empty((0, 2), dtype=np.float32),
            "visibility": np.empty((0,), dtype=bool),
            "confidence": np.empty((0,), dtype=np.float32),
            "track_point_count": int(len(pts)),
            "visible_point_count": 0,
        }
    backend = "fallback_lk"
    model = _ensure_cotracker_model(config)
    if model is not None:
        try:
            torch_mod = _TORCH_MODULE
            if torch_mod is None:
                import torch as torch_mod  # type: ignore
            chronological_indices = list(reversed(frame_indices))
            cotracker_frame_limit = int(config.cotracker_max_frames)
            if str(config.cotracker_model).endswith("_online"):
                cotracker_frame_limit = min(cotracker_frame_limit, 16)
            if len(chronological_indices) > cotracker_frame_limit:
                sampled_idx = np.linspace(
                    0,
                    len(chronological_indices) - 1,
                    num=cotracker_frame_limit,
                ).round().astype(int)
                chronological_indices = [chronological_indices[i] for i in sampled_idx.tolist()]
            frames = [cache.load_bgr(idx) for idx in chronological_indices]
            query_frame = len(frames) - 1
            model_params = list(model.parameters()) if hasattr(model, "parameters") else []
            device = model_params[0].device if model_params else torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
            rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
            h0, w0 = rgb_frames[0].shape[:2]
            scale = min(1.0, float(config.cotracker_max_side_px) / float(max(h0, w0)))
            if scale < 1.0:
                scaled_w = max(16, int(round(w0 * scale)))
                scaled_h = max(16, int(round(h0 * scale)))
                rgb_frames = [
                    cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                    for frame in rgb_frames
                ]
            else:
                scaled_h, scaled_w = h0, w0
            video_np = np.stack(rgb_frames, axis=0)
            video = torch_mod.from_numpy(video_np).permute(0, 3, 1, 2)[None].float().to(device)
            query_pts = pts.astype(np.float32).copy()
            if scale < 1.0:
                query_pts[:, 0] *= float(scaled_w) / float(w0)
                query_pts[:, 1] *= float(scaled_h) / float(h0)
            query_t = np.full((len(query_pts), 1), float(query_frame), dtype=np.float32)
            query_np = np.concatenate([query_t, query_pts], axis=1)
            queries = torch_mod.as_tensor(query_np[None, :, :], dtype=torch_mod.float32, device=device)
            with torch_mod.no_grad():
                if str(config.cotracker_model).endswith("_online"):
                    _ = model(video_chunk=video, is_first_step=True, queries=queries, grid_size=0)
                    output = model(video_chunk=video, is_first_step=False, queries=queries, grid_size=0)
                else:
                    output = model(video, queries=queries, backward_tracking=True)
            if output is not None:
                if isinstance(output, tuple) and len(output) >= 2:
                    pred_tracks, pred_visibility = output[:2]
                    pred_tracks = _as_numpy_points(pred_tracks)
                    pred_visibility = pred_visibility.detach().cpu().numpy()
                    if pred_tracks.ndim == 4:
                        pred_tracks = pred_tracks[0]
                    if pred_visibility.ndim == 4:
                        pred_visibility = pred_visibility[0]
                    if pred_visibility.ndim == 3:
                        pred_visibility = pred_visibility[0]
                    if pred_tracks.ndim == 3:
                        ref_points = pred_tracks[0]
                    elif pred_tracks.ndim == 2:
                        ref_points = pred_tracks
                    else:
                        ref_points = None
                    if ref_points is not None and len(ref_points) == len(pts):
                        if scale < 1.0:
                            ref_points = np.asarray(ref_points, dtype=np.float32).copy()
                            ref_points[:, 0] *= float(w0) / float(scaled_w)
                            ref_points[:, 1] *= float(h0) / float(scaled_h)
                        vis = pred_visibility[0] if pred_visibility.ndim >= 2 else pred_visibility
                        vis = np.asarray(vis).reshape(-1).astype(bool)
                        if vis.shape[0] == len(pts):
                            backend = "cotracker"
                            confidence = np.clip(np.where(vis, 1.0, 0.0), 0.0, 1.0).astype(np.float32)
                            return {
                                "passed": int(vis.sum()) >= int(config.track_min_visible_points),
                                "reason": None if int(vis.sum()) >= int(config.track_min_visible_points) else "track_visibility_below_threshold",
                                "backend": backend,
                                "cotracker_init_status": _COTRACKER_INIT_STATUS or "ready",
                                "cotracker_init_error": _COTRACKER_INIT_ERROR,
                                "ref_points_xy": np.asarray(ref_points, dtype=np.float32).reshape(-1, 2),
                                "visibility": vis,
                                "confidence": confidence,
                                "track_point_count": int(len(pts)),
                                "visible_point_count": int(vis.sum()),
                                "median_fb_error": None,
                            }
        except Exception as exc:
            print(f"E12 CoTracker runtime fallback: {type(exc).__name__}:{exc}", flush=True)
            backend = "fallback_lk"

    current_points = pts.copy()
    visible = np.ones(len(pts), dtype=bool)
    confidence = np.ones(len(pts), dtype=np.float32)
    median_fb_errors: List[float] = []
    for cur_idx, prev_idx in zip(frame_indices[:-1], frame_indices[1:]):
        cur_gray = cache.load_gray(cur_idx)
        prev_gray = cache.load_gray(prev_idx)
        active = np.where(visible)[0]
        if len(active) == 0:
            break
        cur_pts = current_points[active].reshape(-1, 1, 2).astype(np.float32)
        next_pts, st, fb_err = cv2.calcOpticalFlowPyrLK(
            cur_gray,
            prev_gray,
            cur_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if next_pts is None or st is None:
            visible[active] = False
            continue
        back_pts, st_back, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            cur_gray,
            next_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if back_pts is None or st_back is None:
            visible[active] = False
            continue
        cur_arr = cur_pts.reshape(-1, 2)
        next_arr = next_pts.reshape(-1, 2)
        back_arr = back_pts.reshape(-1, 2)
        fb = np.linalg.norm(cur_arr - back_arr, axis=1)
        good = (st.reshape(-1) == 1) & (st_back.reshape(-1) == 1) & np.isfinite(fb) & (fb <= float(config.track_fb_max))
        confidence_active = np.clip(1.0 / (1.0 + fb), 0.0, 1.0)
        current_points[active[good]] = next_arr[good]
        visible[active[~good]] = False
        confidence[active[good]] = 0.5 * confidence[active[good]] + 0.5 * confidence_active[good].astype(np.float32)
        if np.any(good):
            median_fb_errors.append(float(np.median(fb[good])))

    visible_count = int(visible.sum())
    return {
        "passed": visible_count >= int(config.track_min_visible_points),
        "reason": None if visible_count >= int(config.track_min_visible_points) else "track_visibility_below_threshold",
        "backend": backend,
        "cotracker_init_status": _COTRACKER_INIT_STATUS or ("disabled" if not config.cotracker_enabled else "failed"),
        "cotracker_init_error": _COTRACKER_INIT_ERROR,
        "ref_points_xy": current_points.astype(np.float32),
        "visibility": visible,
        "confidence": confidence.astype(np.float32),
        "track_point_count": int(len(pts)),
        "visible_point_count": visible_count,
        "median_fb_error": float(np.median(median_fb_errors)) if median_fb_errors else None,
    }


def estimate_contact_projection_from_tracks(
    contact_means: np.ndarray,
    src_points: np.ndarray,
    ref_points: np.ndarray,
    visibility: np.ndarray,
    config: E12Config,
    projection_method: str = "cotracker_weighted_affine",
) -> Dict[str, Any]:
    means = np.asarray(contact_means, dtype=np.float32).reshape(-1, 2)
    src = np.asarray(src_points, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(ref_points, dtype=np.float32).reshape(-1, 2)
    vis = np.asarray(visibility).reshape(-1).astype(bool)
    if len(src) != len(dst) or len(src) != len(vis):
        return {"passed": False, "reason": "track_shape_mismatch"}
    if len(src) < 4 or int(vis.sum()) < int(config.affine_min_inliers):
        return {"passed": False, "reason": "insufficient_track_points"}
    src_v = src[vis]
    dst_v = dst[vis]
    if len(src_v) < 4:
        return {"passed": False, "reason": "insufficient_visible_track_points"}
    centroid = src_v.mean(axis=0)
    dists = np.linalg.norm(src_v - centroid[None, :], axis=1)
    weights = np.exp(-0.5 * (dists / max(8.0, float(config.local_flow_contact_radius_px) * 0.5)) ** 2).astype(np.float32)
    try:
        affine, inliers = cv2.estimateAffinePartial2D(
            src_v.reshape(-1, 1, 2),
            dst_v.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=float(config.affine_ransac_reproj_threshold),
            maxIters=1000,
            confidence=0.98,
        )
    except cv2.error:
        affine, inliers = None, None
    if affine is None or inliers is None:
        return {"passed": False, "reason": "affine_ransac_failed", "track_point_count": int(len(src)), "visible_point_count": int(len(src_v))}
    inlier_mask = inliers.reshape(-1) > 0
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = float(inlier_count / max(1, len(src_v)))
    projected = cv2.transform(means.reshape(-1, 1, 2), affine.astype(np.float32)).reshape(-1, 2)
    residuals = np.linalg.norm(cv2.transform(src_v.reshape(-1, 1, 2), affine.astype(np.float32)).reshape(-1, 2) - dst_v, axis=1)
    median_residual = float(np.median(residuals[inlier_mask])) if np.any(inlier_mask) else float(np.median(residuals))
    if inlier_count < int(config.affine_min_inliers) or inlier_ratio < float(config.affine_min_inlier_ratio) or median_residual > float(config.affine_max_median_residual):
        return {
            "passed": False,
            "reason": "affine_unreliable",
            "track_point_count": int(len(src)),
            "visible_point_count": int(len(src_v)),
            "affine_inlier_count": int(inlier_count),
            "affine_inlier_ratio": float(inlier_ratio),
            "affine_median_residual": float(median_residual),
        }
    return {
        "passed": True,
        "reason": None,
        "projected_means": projected.astype(np.float32),
        "projected_covariances": None,
        "projection_method": projection_method,
        "track_point_count": int(len(src)),
        "visible_point_count": int(len(src_v)),
        "affine_inlier_count": int(inlier_count),
        "affine_inlier_ratio": float(inlier_ratio),
        "affine_median_residual": float(median_residual),
    }


def get_reference_object_mask(
    ref_idx: int,
    ref_image: np.ndarray,
    tracked_points: np.ndarray,
    noun: Any,
    config: E12Config,
    tracked_bbox: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    h, w = ref_image.shape[:2]
    bbox = _parse_box_like(tracked_bbox)
    if bbox is None and len(tracked_points) > 0:
        pts = np.asarray(tracked_points, dtype=np.float32).reshape(-1, 2)
        if len(pts) > 0:
            bbox = (
                float(np.min(pts[:, 0]) - 12.0),
                float(np.min(pts[:, 1]) - 12.0),
                float(np.max(pts[:, 0]) + 12.0),
                float(np.max(pts[:, 1]) + 12.0),
            )
    if bbox is None:
        bbox = (0.0, 0.0, float(w), float(h))
    predictor = _ensure_sam2_predictor(config)
    if predictor is not None:
        try:
            predictor.set_image(cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB))
            point_coords = None
            point_labels = None
            if tracked_points is not None:
                pts = np.asarray(tracked_points, dtype=np.float32).reshape(-1, 2)
                if len(pts) > 0:
                    limit = min(len(pts), max(8, min(16, len(pts))))
                    pts = pts[:limit]
                    point_coords = pts
                    point_labels = np.ones(len(pts), dtype=np.int32)
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=np.asarray(_clip_int_bbox(bbox, w, h), dtype=np.float32),
                multimask_output=bool(config.sam2_multimask_output),
                normalize_coords=False,
            )
            masks = np.asarray(masks)
            if masks.ndim == 2:
                masks = masks[None, ...]
            scores = np.asarray(scores).reshape(-1) if scores is not None else np.ones(len(masks), dtype=np.float32)
            best_idx = int(np.argmax(scores)) if len(scores) else 0
            mask = _make_binary_mask(masks[best_idx], (h, w))
            if int(mask.sum()) > 0:
                return {
                    "mask": mask,
                    "mask_source": f"sam2_ref:{config.sam2_model_id}",
                    "mask_bbox": _mask_to_bbox_px(mask),
                    "backend": "sam2",
                    "sam2_score": float(scores[best_idx]) if len(scores) else None,
                }
        except Exception as exc:
            return {
                "mask": None,
                "mask_source": f"sam2_ref_failed:{type(exc).__name__}",
                "mask_bbox": None,
                "backend": "sam2_failed",
                "error": str(exc),
            }
    bbox_mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = _clip_int_bbox(bbox, w, h)
    bbox_mask[y1:y2, x1:x2] = 255
    return {
        "mask": bbox_mask,
        "mask_source": "bbox_fallback",
        "mask_bbox": _mask_to_bbox_px(bbox_mask),
        "backend": "bbox_fallback",
    }


def mask_based_projection_gate(
    projected_contact_means: np.ndarray,
    reference_object_mask: np.ndarray,
    visibility: np.ndarray,
    affine_inlier_ratio: float,
    config: E12Config,
) -> Dict[str, Any]:
    projected = np.asarray(projected_contact_means, dtype=np.float32).reshape(-1, 2)
    mask = (np.asarray(reference_object_mask) > 0).astype(np.uint8)
    if len(projected) == 0 or mask.size == 0 or int(mask.sum()) == 0:
        return {"passed": False, "reason": "empty_projection_gate"}
    inside = [0 <= int(round(x)) < mask.shape[1] and 0 <= int(round(y)) < mask.shape[0] and mask[int(round(y)), int(round(x))] > 0 for x, y in projected]
    inside_count = int(sum(inside))
    boundary_dist_map = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
    centroid = projected.mean(axis=0)
    cx, cy = int(round(centroid[0])), int(round(centroid[1]))
    if 0 <= cx < mask.shape[1] and 0 <= cy < mask.shape[0]:
        centroid_dist = float(boundary_dist_map[cy, cx])
    else:
        centroid_dist = float("inf")
    boundary_distances = []
    for x, y in projected:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < mask.shape[1] and 0 <= yi < mask.shape[0]:
            boundary_distances.append(float(boundary_dist_map[yi, xi]))
    projected_contact_to_mask_distance = float(np.median(boundary_distances)) if boundary_distances else float("inf")
    projected_contact_inside_mask_count = inside_count
    visible_count = int(np.asarray(visibility).reshape(-1).astype(bool).sum())
    passed = (
        inside_count >= int(config.mask_gate_min_inside_count)
        and centroid_dist <= float(config.mask_gate_centroid_max_distance_px)
        and visible_count >= int(config.mask_gate_min_visible_points)
        and affine_inlier_ratio >= float(config.mask_gate_min_inlier_ratio)
    )
    quality = (
        0.35 * (inside_count / max(1, len(projected)))
        + 0.25 * max(0.0, 1.0 - centroid_dist / max(1.0, float(config.mask_gate_centroid_max_distance_px)))
        + 0.20 * max(0.0, 1.0 - projected_contact_to_mask_distance / max(1.0, float(config.mask_gate_boundary_tolerance_px)))
        + 0.20 * max(0.0, affine_inlier_ratio)
    )
    return {
        "passed": bool(passed),
        "reason": None if passed else "mask_gate_failed",
        "projected_contact_inside_mask_count": int(projected_contact_inside_mask_count),
        "projected_contact_to_mask_distance": float(projected_contact_to_mask_distance),
        "projected_contact_centroid_distance": float(centroid_dist),
        "reference_mask_area": int(mask.sum()),
        "final_quality_score": float(quality),
    }


def part_aware_projection_gate(
    projected_contact_means: np.ndarray,
    reference_part_mask: np.ndarray,
    reference_object_mask: np.ndarray,
    visibility: np.ndarray,
    affine_inlier_ratio: float,
    config: E12Config,
    allow_whole_object_gate: bool = False,
) -> Dict[str, Any]:
    whole_gate = mask_based_projection_gate(
        projected_contact_means=projected_contact_means,
        reference_object_mask=reference_object_mask,
        visibility=visibility,
        affine_inlier_ratio=affine_inlier_ratio,
        config=config,
    )
    projected = np.asarray(projected_contact_means, dtype=np.float32).reshape(-1, 2)
    part_mask = (np.asarray(reference_part_mask) > 0).astype(np.uint8)
    if len(projected) == 0 or part_mask.size == 0 or int(part_mask.sum()) == 0:
        if allow_whole_object_gate and whole_gate["passed"]:
            return {
                **whole_gate,
                "part_gate_passed": False,
                "part_inside_count": 0,
                "part_distance": float("inf"),
                "gate_source": "whole_object_only",
            }
        return {
            **whole_gate,
            "passed": False,
            "reason": "part_mask_empty",
            "part_gate_passed": False,
            "part_inside_count": 0,
            "part_distance": float("inf"),
            "gate_source": "part_empty",
        }

    part_dist_map = cv2.distanceTransform((part_mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
    inside = []
    distances = []
    for x, y in projected:
        xi, yi = int(round(float(x))), int(round(float(y)))
        if 0 <= xi < part_mask.shape[1] and 0 <= yi < part_mask.shape[0]:
            inside.append(bool(part_mask[yi, xi] > 0))
            distances.append(float(part_dist_map[yi, xi]))
        else:
            inside.append(False)
    part_inside_count = int(sum(inside))
    part_distance = float(np.median(distances)) if distances else float("inf")
    part_passed = (
        part_inside_count >= int(config.part_gate_min_inside_count)
        and part_distance <= float(config.part_gate_max_distance_px)
        and affine_inlier_ratio >= float(config.local_projection_min_inlier_ratio)
    )
    if part_passed:
        quality = 0.65 * float(whole_gate["final_quality_score"]) + 0.35 * (
            0.6 * (part_inside_count / max(1, len(projected)))
            + 0.4 * max(0.0, 1.0 - part_distance / max(1.0, float(config.part_gate_max_distance_px)))
        )
        return {
            **whole_gate,
            "passed": True,
            "reason": None,
            "final_quality_score": float(quality),
            "part_gate_passed": True,
            "part_inside_count": part_inside_count,
            "part_distance": float(part_distance),
            "gate_source": "contact_part_mask",
        }
    if allow_whole_object_gate and whole_gate["passed"]:
        quality = 0.92 * float(whole_gate["final_quality_score"])
        return {
            **whole_gate,
            "passed": True,
            "reason": None,
            "final_quality_score": float(quality),
            "part_gate_passed": False,
            "part_inside_count": part_inside_count,
            "part_distance": float(part_distance),
            "gate_source": "whole_object_mask_fallback",
        }
    return {
        **whole_gate,
        "passed": False,
        "reason": "part_gate_failed",
        "part_gate_passed": False,
        "part_inside_count": part_inside_count,
        "part_distance": float(part_distance),
        "gate_source": "contact_part_mask",
    }


def detection_object_bboxes_px(frame_det: Any, image_shape: Tuple[int, int], score_threshold: float) -> List[Dict[str, Any]]:
    h, w = image_shape[:2]
    boxes = []
    for obj in getattr(frame_det, "objects", []):
        if obj.score < score_threshold:
            continue
        bbox = norm_bbox_to_px([obj.bbox.left, obj.bbox.top, obj.bbox.right, obj.bbox.bottom], w, h)
        bbox = clip_bbox(bbox, w, h)
        if bbox_area(bbox) <= 4:
            continue
        boxes.append({"bbox": bbox, "score": float(obj.score)})
    return boxes


def associate_detection(
    predicted_bbox: Sequence[float],
    detections_px: Sequence[Dict[str, Any]],
    image_shape: Tuple[int, int],
) -> Tuple[Optional[Dict[str, Any]], float]:
    if not detections_px:
        return None, 0.0
    h, w = image_shape[:2]
    pred_center = bbox_center(predicted_bbox)
    pred_diag = math.sqrt(w * w + h * h)
    pred_area = max(1.0, bbox_area(predicted_bbox))
    best = None
    best_score = -1e12
    for item in detections_px:
        det_bbox = item["bbox"]
        iou = bbox_iou(predicted_bbox, det_bbox)
        center_dist = float(np.linalg.norm(pred_center - bbox_center(det_bbox)))
        center_score = max(0.0, 1.0 - center_dist / max(1.0, 0.18 * pred_diag))
        size_ratio = min(pred_area, bbox_area(det_bbox)) / max(pred_area, bbox_area(det_bbox), 1.0)
        score = 1.8 * iou + 0.8 * center_score + 0.4 * size_ratio + 0.15 * float(item.get("score", 0.0))
        if score > best_score:
            best_score = score
            best = {**item, "assoc_iou": float(iou), "assoc_center_dist": center_dist, "assoc_score": float(score)}
    if best is None:
        return None, 0.0
    if best["assoc_iou"] >= 0.08 or best["assoc_center_dist"] <= 0.16 * pred_diag:
        return best, float(best_score)
    return None, float(best_score)


def corners_in_bbox(gray: np.ndarray, bbox: Sequence[float], max_corners: int = 80) -> Optional[np.ndarray]:
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in expand_bbox(bbox, 4, w, h)]
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return None
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=int(max_corners),
        qualityLevel=0.01,
        minDistance=4,
        mask=mask,
        blockSize=5,
    )
    return pts


def bbox_from_points(points_xy: np.ndarray, width: int, height: int, pad: float = 0.0) -> Tuple[float, float, float, float]:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if len(pts) == 0:
        return clip_bbox((0.0, 0.0, 1.0, 1.0), width, height)
    x1 = float(np.min(pts[:, 0]) - pad)
    y1 = float(np.min(pts[:, 1]) - pad)
    x2 = float(np.max(pts[:, 0]) + pad)
    y2 = float(np.max(pts[:, 1]) + pad)
    return clip_bbox((x1, y1, x2, y2), width, height)


def intersection_bbox(
    bbox_a: Sequence[float],
    bbox_b: Sequence[float],
    width: int,
    height: int,
) -> Optional[Tuple[float, float, float, float]]:
    ax1, ay1, ax2, ay2 = [float(v) for v in bbox_a]
    bx1, by1, bx2, by2 = [float(v) for v in bbox_b]
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 + 3 or y2 <= y1 + 3:
        return None
    return clip_bbox((x1, y1, x2, y2), width, height)


def hand_bbox_px_for_frame(
    detections,
    frame_idx: int,
    active_hand: str,
    image_shape: Tuple[int, int],
    score_threshold: float,
) -> Optional[Tuple[float, float, float, float]]:
    h, w = image_shape[:2]
    hand_norm = get_active_hand_bbox(detections[int(frame_idx)], active_hand, score_threshold=score_threshold)
    if hand_norm is None:
        return None
    return clip_bbox(norm_bbox_to_px(hand_norm, w, h), w, h)


def hand_bboxes_px_for_frame(
    frame_det: Any,
    image_shape: Tuple[int, int],
    score_threshold: float,
) -> List[Tuple[float, float, float, float]]:
    h, w = image_shape[:2]
    boxes: List[Tuple[float, float, float, float]] = []
    for hand in getattr(frame_det, "hands", []):
        if getattr(hand, "score", 0.0) < score_threshold:
            continue
        bbox = norm_bbox_to_px([hand.bbox.left, hand.bbox.top, hand.bbox.right, hand.bbox.bottom], w, h)
        bbox = clip_bbox(bbox, w, h)
        if bbox_area(bbox) > 4:
            boxes.append(tuple(float(v) for v in bbox))
    return boxes


def hand_mask_from_bboxes(
    hand_bboxes: Sequence[Sequence[float]],
    shape: Tuple[int, int],
    pad_px: int = 0,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    h, w = shape[:2]
    for bbox in hand_bboxes:
        x1, y1, x2, y2 = _clip_int_bbox(bbox, w, h)
        pad = max(0, int(pad_px))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        mask[y1:y2, x1:x2] = 255
    return mask


def mask_overlap_ratio(numerator_mask: np.ndarray, denominator_mask: np.ndarray) -> float:
    numerator = np.asarray(numerator_mask) > 0
    denominator = np.asarray(denominator_mask) > 0
    denom = int(denominator.sum())
    if denom <= 0:
        return 0.0
    return float(np.logical_and(numerator, denominator).sum() / max(1, denom))


def bbox_to_bbox_distance(bbox_a: Sequence[float], bbox_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in bbox_a]
    bx1, by1, bx2, by2 = [float(v) for v in bbox_b]
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return float(math.hypot(dx, dy))


def min_hand_distance_to_bbox(
    hand_bboxes: Sequence[Sequence[float]],
    target_bbox: Optional[Sequence[float]],
    image_shape: Tuple[int, int],
) -> float:
    h, w = image_shape[:2]
    if target_bbox is None:
        return float(math.hypot(w, h))
    if not hand_bboxes:
        return float(math.hypot(w, h))
    return float(min(bbox_to_bbox_distance(hand_bbox, target_bbox) for hand_bbox in hand_bboxes))


def classify_reference_cleanliness(
    *,
    global_hand_count: int,
    object_overlap_ratio: float,
    contact_crop_overlap_ratio: float,
    contact_part_overlap_ratio: float,
    hand_to_object_distance: float,
    hand_to_contact_part_distance: float,
    target_visible_score: float,
    config: E12Config,
) -> Tuple[str, float, str]:
    visible = target_visible_score >= float(config.target_visible_min_score)
    if global_hand_count == 0 and visible:
        return "strict_target_handless", 8.0, "tier_0_strict_target_handless"
    if (
        visible
        and object_overlap_ratio <= float(config.target_clean_object_overlap_max)
        and contact_part_overlap_ratio <= float(config.target_clean_part_overlap_max)
        and contact_crop_overlap_ratio <= float(config.target_clean_crop_overlap_max)
        and hand_to_object_distance >= 8.0
        and hand_to_contact_part_distance >= 8.0
    ):
        return "local_target_clean", 6.3, "tier_1_local_target_clean"
    if (
        visible
        and object_overlap_ratio <= 0.05
        and contact_part_overlap_ratio <= 0.03
        and contact_crop_overlap_ratio <= 0.16
    ):
        return "hand_visible_target_not_occluded", 4.6, "tier_2_hand_visible_target_not_occluded"
    if visible and object_overlap_ratio <= 0.20 and contact_part_overlap_ratio <= 0.15:
        return "target_partially_occluded", 2.0, "tier_3_target_partially_occluded"
    return "target_occluded_by_hand", -1.5, "tier_4_target_occluded_by_hand"


def evaluate_reference_cleanliness(
    *,
    frame_det: Any,
    image_shape: Tuple[int, int],
    reference_object_mask: np.ndarray,
    reference_part_mask: np.ndarray,
    crop_bbox: Sequence[int],
    tracked_points: np.ndarray,
    tracked_visibility: np.ndarray,
    config: E12Config,
) -> Dict[str, Any]:
    h, w = image_shape[:2]
    hand_bboxes = hand_bboxes_px_for_frame(frame_det, image_shape, config.hand_score_threshold)
    hand_mask = hand_mask_from_bboxes(hand_bboxes, (h, w), pad_px=0)
    hand_area_ratio = float((hand_mask > 0).sum() / max(1, h * w))
    object_mask = _make_binary_mask(reference_object_mask, (h, w))
    part_mask = _make_binary_mask(reference_part_mask, (h, w))
    object_overlap = mask_overlap_ratio(hand_mask, object_mask)
    part_overlap = mask_overlap_ratio(hand_mask, part_mask)
    crop_mask = _bbox_region_mask(crop_bbox, (h, w), pad_px=0)
    crop_overlap = mask_overlap_ratio(hand_mask, crop_mask)
    object_bbox = _mask_to_bbox_px(object_mask)
    part_bbox = _mask_to_bbox_px(part_mask)
    hand_to_object = min_hand_distance_to_bbox(hand_bboxes, object_bbox, image_shape)
    hand_to_part = min_hand_distance_to_bbox(hand_bboxes, part_bbox, image_shape)

    pts = np.asarray(tracked_points, dtype=np.float32).reshape(-1, 2)
    vis = np.asarray(tracked_visibility).reshape(-1).astype(bool) if tracked_visibility is not None else np.ones(len(pts), dtype=bool)
    pts = pts[vis] if len(pts) == len(vis) else pts
    inside_count = 0
    if len(pts) > 0 and int((object_mask > 0).sum()) > 0:
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
        inside_count = int((object_mask[ys, xs] > 0).sum())
    tracked_inside_ratio = float(inside_count / max(1, len(pts)))
    object_area_ratio = float((object_mask > 0).sum() / max(1, h * w))
    target_visible_score = float(
        np.clip(
            0.50 * min(1.0, object_area_ratio / 0.015)
            + 0.30 * tracked_inside_ratio
            + 0.20 * (1.0 - min(1.0, object_overlap * 3.0))
            - 0.35 * min(1.0, part_overlap * 4.0),
            0.0,
            1.0,
        )
    )
    clean_class, clean_score, tier = classify_reference_cleanliness(
        global_hand_count=len(hand_bboxes),
        object_overlap_ratio=object_overlap,
        contact_crop_overlap_ratio=crop_overlap,
        contact_part_overlap_ratio=part_overlap,
        hand_to_object_distance=hand_to_object,
        hand_to_contact_part_distance=hand_to_part,
        target_visible_score=target_visible_score,
        config=config,
    )
    clean_score = float(clean_score - 1.2 * object_overlap - 1.8 * part_overlap - 0.25 * min(1.0, hand_area_ratio / 0.12))
    return {
        "reference_global_hand_count": int(len(hand_bboxes)),
        "reference_hand_area_ratio": float(hand_area_ratio),
        "reference_object_hand_overlap_ratio": float(object_overlap),
        "reference_contact_crop_hand_overlap_ratio": float(crop_overlap),
        "reference_contact_part_hand_overlap_ratio": float(part_overlap),
        "reference_hand_to_object_distance": float(hand_to_object),
        "reference_hand_to_contact_part_distance": float(hand_to_part),
        "reference_target_visible_score": float(target_visible_score),
        "reference_clean_class": clean_class,
        "reference_clean_score": float(clean_score),
        "reference_selection_tier": tier,
    }


def corners_in_mask(gray: np.ndarray, mask: np.ndarray, max_corners: int = 80) -> Optional[np.ndarray]:
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if int(mask_u8.sum()) == 0:
        return None
    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=int(max_corners),
        qualityLevel=0.01,
        minDistance=4,
        mask=mask_u8,
        blockSize=5,
    )
    return pts


def build_local_flow_feature_mask(
    *,
    image_shape: Tuple[int, int],
    object_bbox: Sequence[float],
    contact_points: np.ndarray,
    hand_bbox: Optional[Sequence[float]],
    config: E12Config,
) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    object_region = expand_bbox(object_bbox, float(config.local_flow_object_bbox_pad_px), w, h)
    ox1, oy1, ox2, oy2 = [int(round(v)) for v in object_region]
    mask[oy1:oy2, ox1:ox2] = 255

    contact_pts = np.asarray(contact_points, dtype=np.float32).reshape(-1, 2)
    if len(contact_pts) > 0:
        contact_region = bbox_from_points(contact_pts, w, h, pad=float(config.local_flow_contact_radius_px))
        contact_region = intersection_bbox(contact_region, object_region, w, h) or contact_region
        cx1, cy1, cx2, cy2 = [int(round(v)) for v in contact_region]
        local_mask = np.zeros_like(mask)
        local_mask[cy1:cy2, cx1:cx2] = 255
        mask = cv2.bitwise_and(mask, local_mask)

    if hand_bbox is not None:
        hx1, hy1, hx2, hy2 = [int(round(v)) for v in expand_bbox(hand_bbox, float(config.local_flow_hand_exclusion_px), w, h)]
        mask[hy1:hy2, hx1:hx2] = 0

    return mask


def select_local_flow_points(
    *,
    gray: np.ndarray,
    object_bbox: Sequence[float],
    contact_points: np.ndarray,
    hand_bbox: Optional[Sequence[float]],
    config: E12Config,
) -> Optional[np.ndarray]:
    feature_mask = build_local_flow_feature_mask(
        image_shape=gray.shape,
        object_bbox=object_bbox,
        contact_points=contact_points,
        hand_bbox=hand_bbox,
        config=config,
    )
    pts = corners_in_mask(gray, feature_mask, max_corners=90)
    if pts is not None and len(pts) >= int(config.local_flow_min_points):
        return pts

    h, w = gray.shape[:2]
    contact_bbox = bbox_from_points(contact_points, w, h, pad=float(config.local_flow_contact_radius_px))
    object_local = intersection_bbox(contact_bbox, object_bbox, w, h)
    if object_local is not None:
        pts = corners_in_bbox(gray, object_local, max_corners=90)
        if pts is not None and len(pts) >= int(config.local_flow_min_points):
            return pts

    object_no_hand_mask = build_local_flow_feature_mask(
        image_shape=gray.shape,
        object_bbox=object_bbox,
        contact_points=np.empty((0, 2), dtype=np.float32),
        hand_bbox=hand_bbox,
        config=config,
    )
    pts = corners_in_mask(gray, object_no_hand_mask, max_corners=90)
    if pts is not None and len(pts) >= int(config.local_flow_min_points):
        return pts

    pts = corners_in_bbox(gray, object_bbox, max_corners=90)
    return pts


def estimate_weighted_affine_from_points(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    contact_center_xy: np.ndarray,
    config: E12Config,
) -> Dict[str, Any]:
    pts_src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 2)
    pts_dst = np.asarray(dst_pts, dtype=np.float32).reshape(-1, 2)
    if len(pts_src) < int(config.local_flow_min_points) or len(pts_dst) < int(config.local_flow_min_points):
        return {"passed": False, "reason": "insufficient_contact_local_flow_points"}

    center = np.asarray(contact_center_xy, dtype=np.float32).reshape(2)
    distances = np.linalg.norm(pts_src - center[None, :], axis=1)
    sigma = max(8.0, float(config.local_flow_contact_radius_px) * 0.5)
    weights = np.exp(-0.5 * (distances / sigma) ** 2).astype(np.float32)
    try:
        affine, inliers = cv2.estimateAffinePartial2D(
            pts_src.reshape(-1, 1, 2),
            pts_dst.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=1000,
            confidence=0.98,
        )
    except cv2.error:
        affine, inliers = None, None

    if affine is None or inliers is None:
        return {"passed": False, "reason": "local_affine_ransac_failed"}

    inlier_mask = (inliers.reshape(-1) > 0)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < int(config.local_flow_min_inliers):
        return {"passed": False, "reason": "local_affine_insufficient_inliers"}

    inlier_ratio = float(inlier_count / max(1, len(pts_src)))
    if inlier_ratio < float(config.local_flow_min_inlier_ratio):
        return {"passed": False, "reason": "local_affine_low_inlier_ratio"}

    affine = affine.astype(np.float32)
    projected_src = cv2.transform(pts_src.reshape(-1, 1, 2), affine).reshape(-1, 2)
    residuals = np.linalg.norm(projected_src - pts_dst, axis=1)
    weighted_residual = float(np.sum(residuals * weights) / max(1e-6, float(np.sum(weights))))

    return {
        "passed": True,
        "affine": affine,
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "weighted_residual": weighted_residual,
        "median_residual": float(np.median(residuals[inlier_mask])) if np.any(inlier_mask) else float(np.median(residuals)),
        "point_count": int(len(pts_src)),
    }


def get_torch_for_gpu_flow(config: E12Config):
    global _TORCH_MODULE, _TORCH_IMPORT_ERROR
    if not config.gpu_flow_enabled:
        FLOW_RUNTIME_STATS["disabled"] += 1
        return None
    if _TORCH_IMPORT_ERROR is not None:
        FLOW_RUNTIME_STATS["torch_unavailable"] += 1
        return None
    if _TORCH_MODULE is None:
        try:
            import torch

            _TORCH_MODULE = torch
        except Exception as exc:
            _TORCH_IMPORT_ERROR = f"{type(exc).__name__}:{exc}"
            FLOW_RUNTIME_STATS["torch_import_error"] += 1
            return None
    if config.gpu_flow_device.startswith("cuda") and not _TORCH_MODULE.cuda.is_available():
        FLOW_RUNTIME_STATS["cuda_unavailable"] += 1
        return None
    return _TORCH_MODULE


def describe_gpu_flow_backend(config: E12Config) -> str:
    torch = get_torch_for_gpu_flow(config)
    if torch is None:
        if not config.gpu_flow_enabled:
            return "disabled_by_E7_GPU_FLOW"
        return f"cpu_opencv_fallback({_TORCH_IMPORT_ERROR or 'cuda_unavailable'})"
    device_name = torch.cuda.get_device_name(0) if config.gpu_flow_device.startswith("cuda") else config.gpu_flow_device
    return f"torch_patch_cuda(device={config.gpu_flow_device}, name={device_name})"


def reserve_gpu_workspace(config: E12Config) -> None:
    global _GPU_WORKSPACE
    if float(config.gpu_workspace_gb) <= 0 or _GPU_WORKSPACE is not None:
        return
    torch = get_torch_for_gpu_flow(config)
    if torch is None:
        return
    numel = int(float(config.gpu_workspace_gb) * (1024 ** 3) / 4)
    if numel <= 0:
        return
    _GPU_WORKSPACE = torch.empty(numel, dtype=torch.float32, device=torch.device(config.gpu_flow_device))
    _GPU_WORKSPACE.fill_(0.0)
    if config.gpu_flow_device.startswith("cuda"):
        torch.cuda.synchronize()
    FLOW_RUNTIME_STATS["gpu_workspace_gb"] = float(config.gpu_workspace_gb)


def preload_gpu_gray_cache(cache: E12FrameCache, config: E12Config) -> None:
    if not config.gpu_preload_gray_frames:
        return
    torch = get_torch_for_gpu_flow(config)
    if torch is None:
        return
    max_frames = min(int(config.gpu_preload_max_frames), int(config.gpu_gray_cache_max_frames))
    frame_indices = cache.available_frame_indices(max_frames=max_frames)
    print(f"Preloading {len(frame_indices)} gray frames to GPU cache...", flush=True)
    for idx, frame_idx in enumerate(frame_indices, start=1):
        cache.load_gray_torch(frame_idx, config)
        if idx % 250 == 0 or idx == len(frame_indices):
            pct = idx / max(1, len(frame_indices)) * 100.0
            bar_width = 28
            filled = int(round(bar_width * idx / max(1, len(frame_indices))))
            bar = "#" * filled + "-" * (bar_width - filled)
            print(f"    preload [{bar}] {idx}/{len(frame_indices)} ({pct:5.1f}%)", flush=True)
    if config.gpu_flow_device.startswith("cuda"):
        torch.cuda.synchronize()
    FLOW_RUNTIME_STATS["gpu_gray_preloaded_frames"] = len(cache.gpu_gray_cache)


def _as_numpy_points(points_tensor: Any) -> np.ndarray:
    return points_tensor.detach().cpu().numpy().astype(np.float32)


def track_points_backward_torch_patch(
    cur_gray: np.ndarray,
    prev_gray: np.ndarray,
    pts_cur: np.ndarray,
    config: E12Config,
    cache: Optional[E12FrameCache] = None,
    cur_frame_idx: Optional[int] = None,
    prev_frame_idx: Optional[int] = None,
    fb_max: float = 2.5,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    torch = get_torch_for_gpu_flow(config)
    if torch is None or pts_cur is None or len(pts_cur) == 0:
        return None

    try:
        device = torch.device(config.gpu_flow_device)
        patch_radius = int(config.gpu_flow_patch_radius)
        search_radius = int(config.gpu_flow_search_radius)
        if patch_radius < 1 or search_radius < 1:
            return None
        h, w = cur_gray.shape[:2]
        pts = np.asarray(pts_cur, dtype=np.float32).reshape(-1, 2)
        margin = patch_radius + search_radius + 2
        inside = (
            (pts[:, 0] >= margin)
            & (pts[:, 0] <= (w - 1 - margin))
            & (pts[:, 1] >= margin)
            & (pts[:, 1] <= (h - 1 - margin))
        )
        if not np.any(inside):
            FLOW_RUNTIME_STATS["gpu_patch_no_points_inside"] += 1
            return None
        pts = pts[inside]

        cur = None
        prev = None
        if cache is not None and cur_frame_idx is not None and prev_frame_idx is not None:
            cur = cache.load_gray_torch(int(cur_frame_idx), config)
            prev = cache.load_gray_torch(int(prev_frame_idx), config)
        if cur is None:
            cur = torch.as_tensor(cur_gray, dtype=torch.float32, device=device)[None, None] / 255.0
            FLOW_RUNTIME_STATS["gpu_gray_direct_upload"] += 1
        if prev is None:
            prev = torch.as_tensor(prev_gray, dtype=torch.float32, device=device)[None, None] / 255.0
            FLOW_RUNTIME_STATS["gpu_gray_direct_upload"] += 1
        pts_t = torch.as_tensor(pts, dtype=torch.float32, device=device)

        offsets_1d = torch.arange(-patch_radius, patch_radius + 1, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(offsets_1d, offsets_1d, indexing="ij")
        patch_offsets = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)

        search_1d = torch.arange(-search_radius, search_radius + 1, device=device, dtype=torch.float32)
        sy, sx = torch.meshgrid(search_1d, search_1d, indexing="ij")
        search_offsets = torch.stack((sx.reshape(-1), sy.reshape(-1)), dim=1)

        def sample_at(points_xy):
            grid_xy = points_xy[:, None, :] + patch_offsets[None, :, :]
            grid = grid_xy.clone()
            grid[..., 0] = grid[..., 0] / max(1, w - 1) * 2.0 - 1.0
            grid[..., 1] = grid[..., 1] / max(1, h - 1) * 2.0 - 1.0
            grid = grid.view(1, points_xy.shape[0], patch_offsets.shape[0], 2)
            values = torch.nn.functional.grid_sample(
                cur if sample_at.source == "cur" else prev,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            return values.view(points_xy.shape[0], patch_offsets.shape[0])

        sample_at.source = "cur"
        cur_patch = sample_at(pts_t)
        sample_at.source = "prev"
        candidates = pts_t[:, None, :] + search_offsets[None, :, :]
        candidate_grid = candidates[:, :, None, :] + patch_offsets[None, None, :, :]
        grid = candidate_grid.clone()
        grid[..., 0] = grid[..., 0] / max(1, w - 1) * 2.0 - 1.0
        grid[..., 1] = grid[..., 1] / max(1, h - 1) * 2.0 - 1.0
        grid = grid.view(1, pts_t.shape[0] * search_offsets.shape[0], patch_offsets.shape[0], 2)
        prev_patches = torch.nn.functional.grid_sample(
            prev,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).view(pts_t.shape[0], search_offsets.shape[0], patch_offsets.shape[0])
        ssd = ((prev_patches - cur_patch[:, None, :]) ** 2).mean(dim=2)
        best = torch.argmin(ssd, dim=1)
        pts_prev = pts_t + search_offsets[best]

        sample_at.source = "prev"
        prev_patch = sample_at(pts_prev)
        back_candidates = pts_prev[:, None, :] + search_offsets[None, :, :]
        back_grid = back_candidates[:, :, None, :] + patch_offsets[None, None, :, :]
        grid = back_grid.clone()
        grid[..., 0] = grid[..., 0] / max(1, w - 1) * 2.0 - 1.0
        grid[..., 1] = grid[..., 1] / max(1, h - 1) * 2.0 - 1.0
        grid = grid.view(1, pts_t.shape[0] * search_offsets.shape[0], patch_offsets.shape[0], 2)
        cur_patches = torch.nn.functional.grid_sample(
            cur,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).view(pts_t.shape[0], search_offsets.shape[0], patch_offsets.shape[0])
        back_ssd = ((cur_patches - prev_patch[:, None, :]) ** 2).mean(dim=2)
        back_best = torch.argmin(back_ssd, dim=1)
        pts_back = pts_prev + search_offsets[back_best]

        fb = torch.linalg.norm(pts_t - pts_back, dim=1)
        good = torch.isfinite(fb) & (fb <= float(fb_max))
        if not bool(good.any()):
            FLOW_RUNTIME_STATS["gpu_patch_no_fb_good"] += 1
            return None
        FLOW_RUNTIME_STATS["gpu_patch_success"] += 1
        return _as_numpy_points(pts_t[good]), _as_numpy_points(pts_prev[good]), float(torch.median(fb[good]).detach().cpu().item())
    except Exception as exc:
        FLOW_RUNTIME_STATS[f"gpu_patch_exception_{type(exc).__name__}"] += 1
        return None


def track_points_backward(
    cur_gray: np.ndarray,
    prev_gray: np.ndarray,
    pts_cur: np.ndarray,
    config: Optional[E12Config] = None,
    cache: Optional[E12FrameCache] = None,
    cur_frame_idx: Optional[int] = None,
    prev_frame_idx: Optional[int] = None,
    fb_max: float = 2.5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if pts_cur is None or len(pts_cur) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), float("inf")
    if config is not None:
        gpu_result = track_points_backward_torch_patch(
            cur_gray,
            prev_gray,
            pts_cur,
            config,
            cache=cache,
            cur_frame_idx=cur_frame_idx,
            prev_frame_idx=prev_frame_idx,
            fb_max=fb_max,
        )
        if gpu_result is not None:
            return gpu_result
        FLOW_RUNTIME_STATS["opencv_cpu_fallback"] += 1
    next_pts, st, _ = cv2.calcOpticalFlowPyrLK(
        cur_gray,
        prev_gray,
        pts_cur,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    if next_pts is None or st is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), float("inf")
    back_pts, st_back, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        cur_gray,
        next_pts,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    good = (st.reshape(-1) == 1) & (st_back.reshape(-1) == 1)
    pts_a = pts_cur.reshape(-1, 2)
    pts_b = next_pts.reshape(-1, 2)
    pts_back = back_pts.reshape(-1, 2)
    fb = np.linalg.norm(pts_a - pts_back, axis=1)
    good = good & np.isfinite(fb) & (fb <= float(fb_max))
    if not np.any(good):
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), float("inf")
    return pts_a[good].astype(np.float32), pts_b[good].astype(np.float32), float(np.median(fb[good]))


def transform_bbox_by_affine(bbox: Sequence[float], affine: np.ndarray, width: int, height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.transform(corners, affine).reshape(-1, 2)
    return clip_bbox((float(transformed[:, 0].min()), float(transformed[:, 1].min()), float(transformed[:, 0].max()), float(transformed[:, 1].max())), width, height)


def points_near_bbox_count(points_xy: np.ndarray, bbox: Sequence[float], tolerance_px: float) -> int:
    return count_points_near_bbox(np.asarray(points_xy, dtype=np.float32), bbox, float(tolerance_px))


def bbox_relative_project(points_xy: np.ndarray, src_bbox: Sequence[float], dst_bbox: Sequence[float]) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    sx1, sy1, sx2, sy2 = [float(v) for v in src_bbox]
    dx1, dy1, dx2, dy2 = [float(v) for v in dst_bbox]
    sw = max(1.0, sx2 - sx1)
    sh = max(1.0, sy2 - sy1)
    u = (pts[:, 0] - sx1) / sw
    v = (pts[:, 1] - sy1) / sh
    return np.stack([dx1 + u * (dx2 - dx1), dy1 + v * (dy2 - dy1)], axis=1).astype(np.float32)


def bbox_relative_covariance_project(
    covariances_xy: Optional[np.ndarray],
    src_bbox: Sequence[float],
    dst_bbox: Sequence[float],
) -> Optional[np.ndarray]:
    if covariances_xy is None:
        return None
    cov = np.asarray(covariances_xy, dtype=np.float32)
    if cov.ndim != 3 or cov.shape[1:] != (2, 2):
        return None
    sx1, sy1, sx2, sy2 = [float(v) for v in src_bbox]
    dx1, dy1, dx2, dy2 = [float(v) for v in dst_bbox]
    scale_x = (dx2 - dx1) / max(1.0, sx2 - sx1)
    scale_y = (dy2 - dy1) / max(1.0, sy2 - sy1)
    jacobian = np.array([[scale_x, 0.0], [0.0, scale_y]], dtype=np.float32)
    return np.asarray([jacobian @ item @ jacobian.T for item in cov], dtype=np.float32)


def transform_covariances_by_affine(covariances_xy: Optional[np.ndarray], affine: np.ndarray) -> Optional[np.ndarray]:
    if covariances_xy is None:
        return None
    cov = np.asarray(covariances_xy, dtype=np.float32)
    if cov.ndim != 3 or cov.shape[1:] != (2, 2):
        return None
    linear = np.asarray(affine, dtype=np.float32)[:2, :2]
    return np.asarray([linear @ item @ linear.T for item in cov], dtype=np.float32)


def temporal_active_object_selection(
    *,
    detections,
    cache: E12FrameCache,
    frame_idx: int,
    active_hand: str,
    contact_points: np.ndarray,
    config: E12Config,
    window: int = 3,
) -> Dict[str, Any]:
    img = cache.load_bgr(frame_idx)
    h, w = img.shape[:2]
    frame_det = detections[int(frame_idx)]
    hand_norm = get_active_hand_bbox(frame_det, active_hand, score_threshold=config.hand_score_threshold)
    if hand_norm is None:
        return {"passed": False, "reason": "missing_active_hand_bbox"}
    hand_px = norm_bbox_to_px(hand_norm, w, h)
    object_bboxes = detection_object_bboxes_px(frame_det, img.shape, config.object_score_threshold)
    if not object_bboxes:
        return {"passed": False, "reason": "missing_object_bbox"}

    neighboring = []
    for j in range(max(0, frame_idx - window), min(len(detections), frame_idx + window + 1)):
        neighboring.extend(detection_object_bboxes_px(detections[j], img.shape, config.object_score_threshold))

    best = None
    best_score = -1e12
    for item in object_bboxes:
        bbox = item["bbox"]
        overlap = hand_overlap_ratio(hand_px, bbox)
        center_dist = float(np.linalg.norm(bbox_center(hand_px) - bbox_center(bbox)))
        diag = math.sqrt(w * w + h * h)
        dist_score = max(0.0, 1.0 - center_dist / max(1.0, 0.25 * diag))
        stability_matches = [bbox_iou(bbox, other["bbox"]) for other in neighboring]
        stability = float(np.mean(sorted(stability_matches, reverse=True)[: min(5, len(stability_matches))])) if stability_matches else 0.0
        boundary_count = points_near_bbox_count(contact_points, bbox, tolerance_px=10.0)
        boundary_score = min(1.0, boundary_count / max(1.0, len(contact_points)))
        score = 2.0 * overlap + 1.0 * dist_score + 0.7 * stability + 1.5 * boundary_score + 0.2 * float(item["score"])
        if score > best_score:
            best_score = score
            best = {
                "bbox": bbox,
                "bbox_norm": px_bbox_to_norm(bbox, w, h),
                "score": float(score),
                "detector_score": float(item["score"]),
                "hand_overlap": float(overlap),
                "center_distance": float(center_dist),
                "stability": float(stability),
                "contact_boundary_count": int(boundary_count),
            }
    if best is None:
        return {"passed": False, "reason": "no_active_object_selection"}
    return {"passed": True, **best}


def fit_contact_gmm_with_object(
    *,
    detections,
    cache: E12FrameCache,
    frame_idx: int,
    active_hand: str,
    object_bbox_norm: Sequence[float],
    contact_config: ContactExtractionConfig,
    config: E12Config,
) -> Dict[str, Any]:
    img = cache.load_bgr(frame_idx)
    frame_det = detections[int(frame_idx)]
    hand_bbox = get_active_hand_bbox(frame_det, active_hand, score_threshold=config.hand_score_threshold)
    if hand_bbox is None:
        return {"passed": False, "reason": "missing_active_hand_bbox", "contact_points": 0}
    extraction = extract_contact_points(img, hand_bbox, [object_bbox_norm], config=contact_config)
    points = np.asarray(extraction["contact_points"], dtype=np.float32).reshape(-1, 2)
    if len(points) < int(config.min_contact_points):
        return {"passed": False, "reason": "contact_points_lt5_after_object_boundary_filter", "contact_points": int(len(points))}
    try:
        with np.errstate(all="ignore"):
            gmm = GaussianMixture(n_components=int(config.gmm_components), random_state=42)
            gmm.fit(points)
    except Exception as exc:
        return {"passed": False, "reason": f"gmm_failed:{type(exc).__name__}", "contact_points": int(len(points))}
    return {
        "passed": True,
        "reason": None,
        "contact_points": int(len(points)),
        "contact_means": np.asarray(gmm.means_, dtype=np.float32),
        "contact_weights": np.asarray(gmm.weights_, dtype=np.float32),
        "contact_covariances": np.asarray(gmm.covariances_, dtype=np.float32),
        "raw_contact_points": points,
    }


def track_object_backward(
    *,
    detections,
    cache: E12FrameCache,
    contact_frame: int,
    seed_bbox_px: Sequence[float],
    min_frame: int,
    config: E12Config,
) -> Dict[str, Any]:
    contact_frame = int(contact_frame)
    min_frame = max(0, int(min_frame))
    seed_img = cache.load_bgr(contact_frame)
    h, w = seed_img.shape[:2]
    track: Dict[int, Dict[str, Any]] = {
        contact_frame: {
            "bbox": tuple(float(v) for v in seed_bbox_px),
            "source": "seed_detection",
            "tracked_point_count": 0,
            "forward_backward_error": None,
            "track_confidence": 1.0,
        }
    }
    pair_affines: Dict[Tuple[int, int], Optional[np.ndarray]] = {}
    break_reason = None
    confidence = 1.0
    current_bbox = tuple(float(v) for v in seed_bbox_px)
    for cur_idx in range(contact_frame, min_frame, -1):
        prev_idx = cur_idx - 1
        cur_gray = cache.load_gray(cur_idx)
        prev_gray = cache.load_gray(prev_idx)
        pts = corners_in_bbox(cur_gray, current_bbox, max_corners=90)
        pts_cur, pts_prev, fb_err = track_points_backward(
            cur_gray,
            prev_gray,
            pts,
            config=config,
            cache=cache,
            cur_frame_idx=cur_idx,
            prev_frame_idx=prev_idx,
            fb_max=2.8,
        )
        flow_bbox = None
        affine = None
        flow_count = int(len(pts_cur))
        if flow_count >= 6:
            affine, inliers = cv2.estimateAffinePartial2D(
                pts_cur.reshape(-1, 1, 2),
                pts_prev.reshape(-1, 1, 2),
                method=cv2.RANSAC,
                ransacReprojThreshold=4.0,
                maxIters=1000,
                confidence=0.98,
            )
            if affine is not None and inliers is not None and int(inliers.sum()) >= 4:
                flow_bbox = transform_bbox_by_affine(current_bbox, affine.astype(np.float32), w, h)
        pair_affines[(cur_idx, prev_idx)] = affine.astype(np.float32) if affine is not None else None
        predicted_bbox = flow_bbox if flow_bbox is not None else current_bbox
        prev_dets = detection_object_bboxes_px(detections[prev_idx], (h, w), config.object_score_threshold)
        matched, assoc_score = associate_detection(predicted_bbox, prev_dets, (h, w))
        if matched is not None and flow_bbox is not None:
            det_bbox = matched["bbox"]
            mixed = tuple(0.65 * float(det_bbox[i]) + 0.35 * float(flow_bbox[i]) for i in range(4))
            source = "mixed"
            current_bbox = clip_bbox(mixed, w, h)
            confidence = min(1.0, confidence * 0.995 + 0.03)
        elif matched is not None:
            current_bbox = matched["bbox"]
            source = "detection"
            confidence = min(1.0, confidence * 0.99 + 0.02)
        elif flow_bbox is not None:
            current_bbox = flow_bbox
            source = "flow"
            confidence *= 0.985
        else:
            break_reason = "no_detection_or_flow"
            confidence *= 0.75
            if confidence < config.min_track_confidence:
                break
            current_bbox = predicted_bbox
            source = "low_confidence_carry"

        if bbox_area(current_bbox) <= 16:
            break_reason = "tracked_bbox_degenerate"
            break
        track[prev_idx] = {
            "bbox": tuple(float(v) for v in current_bbox),
            "source": source,
            "tracked_point_count": flow_count,
            "forward_backward_error": None if not np.isfinite(fb_err) else float(fb_err),
            "track_confidence": float(confidence),
            "assoc_score": float(assoc_score),
        }
        if confidence < config.min_track_confidence:
            break_reason = "track_confidence_below_threshold"
            break

    sources = Counter(item["source"] for item in track.values())
    min_conf = min(float(item.get("track_confidence", 0.0)) for item in track.values()) if track else 0.0
    return {
        "status": "success" if len(track) >= 2 and min_conf >= config.min_track_confidence else "failed",
        "track": track,
        "pair_affines": pair_affines,
        "track_length": int(len(track)),
        "track_confidence": float(min_conf),
        "track_break_reason": break_reason,
        "source_counts": dict(sources),
    }


def track_contact_projection_local_flow(
    *,
    detections,
    cache: E12FrameCache,
    contact_frame: int,
    ref_idx: int,
    contact_means: np.ndarray,
    contact_covariances: Optional[np.ndarray],
    seed_bbox_px: Sequence[float],
    ref_bbox_px: Sequence[float],
    object_track: Dict[int, Dict[str, Any]],
    active_hand: str,
    config: E12Config,
) -> Dict[str, Any]:
    current_means = np.asarray(contact_means, dtype=np.float32).reshape(-1, 2)
    current_covariances = None if contact_covariances is None else np.asarray(contact_covariances, dtype=np.float32).reshape(-1, 2, 2)
    valid_steps = 0
    fb_errors = []
    inlier_ratios = []
    inlier_counts = []
    flow_point_counts = []
    residuals = []
    for cur_idx in range(int(contact_frame), int(ref_idx), -1):
        prev_idx = cur_idx - 1
        cur_gray = cache.load_gray(cur_idx)
        prev_gray = cache.load_gray(prev_idx)
        cur_object = object_track.get(cur_idx, {}).get("bbox")
        if cur_object is None:
            cur_object = seed_bbox_px
        hand_bbox = hand_bbox_px_for_frame(
            detections,
            cur_idx,
            active_hand,
            cur_gray.shape,
            score_threshold=config.hand_score_threshold,
        )
        pts = select_local_flow_points(
            gray=cur_gray,
            object_bbox=cur_object,
            contact_points=current_means,
            hand_bbox=hand_bbox,
            config=config,
        )
        pts_cur, pts_prev, fb_err = track_points_backward(
            cur_gray,
            prev_gray,
            pts,
            config=config,
            cache=cache,
            cur_frame_idx=cur_idx,
            prev_frame_idx=prev_idx,
            fb_max=2.2,
        )
        if len(pts_cur) < int(config.local_flow_min_points):
            return {"passed": False, "reason": "insufficient_contact_local_flow_points", "projected": None}
        affine_detail = estimate_weighted_affine_from_points(
            pts_cur,
            pts_prev,
            current_means.mean(axis=0),
            config=config,
        )
        if not affine_detail["passed"]:
            return {"passed": False, "reason": affine_detail["reason"], "projected": None}
        affine = np.asarray(affine_detail["affine"], dtype=np.float32)
        step_shift = cv2.transform(current_means.reshape(-1, 1, 2), affine).reshape(-1, 2) - current_means
        if float(np.max(np.linalg.norm(step_shift, axis=1))) > float(config.local_flow_max_step_px):
            return {"passed": False, "reason": "local_affine_step_too_large", "projected": None}
        current_means = cv2.transform(current_means.reshape(-1, 1, 2), affine.astype(np.float32)).reshape(-1, 2)
        current_covariances = transform_covariances_by_affine(current_covariances, affine)
        valid_steps += 1
        if np.isfinite(fb_err):
            fb_errors.append(float(fb_err))
        inlier_ratios.append(float(affine_detail["inlier_ratio"]))
        inlier_counts.append(int(affine_detail["inlier_count"]))
        flow_point_counts.append(int(affine_detail["point_count"]))
        residuals.append(float(affine_detail["weighted_residual"]))
    h_ref, w_ref = cache.load_gray(ref_idx).shape[:2]
    if not np.isfinite(current_means).all():
        return {"passed": False, "reason": "local_flow_projection_nonfinite", "projected": None}
    if current_covariances is not None and not np.isfinite(current_covariances).all():
        current_covariances = None
    if np.any(current_means[:, 0] < -20) or np.any(current_means[:, 0] > w_ref + 20) or np.any(current_means[:, 1] < -20) or np.any(current_means[:, 1] > h_ref + 20):
        return {"passed": False, "reason": "local_flow_projection_out_of_bounds", "projected": None}
    return {
        "passed": True,
        "reason": None,
        "projected": current_means.astype(np.float32),
        "projected_covariances": current_covariances.astype(np.float32) if current_covariances is not None else None,
        "projection_source": "object_aware_local_flow_affine",
        "local_flow_steps": int(valid_steps),
        "local_flow_fb_error": float(np.median(fb_errors)) if fb_errors else None,
        "local_flow_points": float(np.median(flow_point_counts)) if flow_point_counts else None,
        "local_flow_inliers": float(np.median(inlier_counts)) if inlier_counts else None,
        "local_flow_inlier_ratio": float(np.median(inlier_ratios)) if inlier_ratios else None,
        "local_flow_weighted_residual": float(np.median(residuals)) if residuals else None,
    }


def trajectory_pixels(
    *,
    detections,
    frame_idx: int,
    active_hand: str,
    image_shape: Tuple[int, int],
    horizon: int = 6,
) -> Tuple[List[np.ndarray], List[int]]:
    h, w = image_shape[:2]
    points = []
    missing = []
    for offset in range(horizon):
        idx = int(frame_idx) + offset
        if idx >= len(detections):
            break
        found = False
        for hand in getattr(detections[idx], "hands", []):
            if hand.score >= 0.5 and hand.side.name.lower() == active_hand:
                bbox = [hand.bbox.left, hand.bbox.top, hand.bbox.right, hand.bbox.bottom]
                px = norm_bbox_to_px(bbox, w, h)
                points.append(bbox_center(px))
                found = True
                break
        if not found:
            missing.append(offset)
    return points, missing


def project_trajectory_bbox_relative(
    trajectory_pts: Sequence[np.ndarray],
    seed_bbox_px: Sequence[float],
    ref_bbox_px: Sequence[float],
) -> List[np.ndarray]:
    if not trajectory_pts:
        return []
    arr = np.asarray(trajectory_pts, dtype=np.float32).reshape(-1, 2)
    return [pt for pt in bbox_relative_project(arr, seed_bbox_px, ref_bbox_px)]


def evaluate_reference_candidates(
    *,
    detections,
    cache: E12FrameCache,
    object_track: Dict[int, Dict[str, Any]],
    contact_frame: int,
    seed_bbox_px: Sequence[float],
    contact_means: np.ndarray,
    contact_covariances: Optional[np.ndarray],
    trajectory_pts: Sequence[np.ndarray],
    tracking_points: np.ndarray,
    whole_tracking_points: Optional[np.ndarray],
    contact_part_mask: np.ndarray,
    contact_part_info: Dict[str, Any],
    active_hand: str,
    hand_bbox_px: Optional[Sequence[float]],
    noun: Any,
    all_nouns: Any,
    narration: Any,
    search_start: int,
    search_end: int,
    config: E12Config,
) -> Dict[str, Any]:
    if not object_track:
        return {"passed": False, "reason": "empty_object_track"}
    available = sorted(idx for idx in object_track if idx < int(contact_frame))
    if not available:
        return {"passed": False, "reason": "no_reference_before_contact_in_track"}

    preferred = [idx for idx in available if int(search_start) <= idx <= int(search_end)]
    ref_order = sorted(preferred, reverse=True) if preferred else sorted(available, reverse=True)
    ref_reason = "pre_contact_gap_tracked_reference" if preferred else "earliest_reliable_tracked_reference_no_pre_gap"

    best_reject = Counter()
    best_candidate = None
    contact_mask = None
    contact_mask_source = None
    contact_mask_bbox = None
    contact_point_mask_info = get_object_mask(int(contact_frame), cache.load_bgr(int(contact_frame)), seed_bbox_px, noun, config)
    contact_mask = contact_point_mask_info.get("mask")
    contact_mask_source = contact_point_mask_info.get("mask_source")
    contact_mask_bbox = contact_point_mask_info.get("mask_bbox")
    if contact_mask is None:
        contact_mask = np.zeros(cache.load_bgr(int(contact_frame)).shape[:2], dtype=np.uint8)

    for ref_idx in ref_order:
        ref_img = cache.load_bgr(ref_idx)
        h_ref, w_ref = ref_img.shape[:2]
        ref_bbox = object_track[ref_idx]["bbox"]
        frame_indices = list(range(int(contact_frame), int(ref_idx) - 1, -1))
        track_detail = track_object_points_to_reference(frame_indices, tracking_points, config, cache)
        if not track_detail["passed"]:
            best_reject[str(track_detail.get("reason") or "track_failed")] += 1
            continue
        projected_detail = estimate_contact_projection_from_tracks(
            contact_means=contact_means,
            src_points=tracking_points,
            ref_points=track_detail["ref_points_xy"],
            visibility=track_detail["visibility"],
            config=config,
            projection_method="contact_part_local_affine",
        )
        local_projection_passed = bool(projected_detail.get("passed"))
        if (
            local_projection_passed
            and float(projected_detail.get("affine_inlier_ratio") or 0.0) >= float(config.local_projection_min_inlier_ratio)
            and float(projected_detail.get("affine_median_residual") or 0.0) <= float(config.local_projection_max_median_residual)
        ):
            projected = np.asarray(projected_detail["projected_means"], dtype=np.float32)
            projected_covariances = projected_detail.get("projected_covariances")
            projection_method = str(projected_detail.get("projection_method") or "contact_part_local_affine")
        else:
            fallback_detail: Dict[str, Any] = {"passed": False, "reason": "no_whole_tracking_points"}
            if whole_tracking_points is not None and len(whole_tracking_points) >= int(config.min_tracking_points):
                whole_track = track_object_points_to_reference(frame_indices, whole_tracking_points, config, cache)
                if whole_track["passed"]:
                    fallback_detail = estimate_contact_projection_from_tracks(
                        contact_means=contact_means,
                        src_points=whole_tracking_points,
                        ref_points=whole_track["ref_points_xy"],
                        visibility=whole_track["visibility"],
                        config=config,
                        projection_method="whole_object_affine_fallback",
                    )
                    if fallback_detail.get("passed"):
                        track_detail = whole_track
                        projected_detail = fallback_detail
            if fallback_detail.get("passed"):
                projected = np.asarray(fallback_detail["projected_means"], dtype=np.float32)
                projected_covariances = fallback_detail.get("projected_covariances")
                projection_method = "whole_object_affine_fallback"
            else:
                best_reject[str(projected_detail.get("reason") or fallback_detail.get("reason") or "local_affine_failed")] += 1
                continue

        if not np.isfinite(projected).all():
            best_reject["projection_nonfinite"] += 1
            continue
        ref_object_mask_info = get_reference_object_mask(
            ref_idx=ref_idx,
            ref_image=ref_img,
            tracked_points=track_detail["ref_points_xy"][track_detail["visibility"]],
            noun=noun,
            config=config,
            tracked_bbox=ref_bbox,
        )
        reference_object_mask = ref_object_mask_info["mask"]
        if reference_object_mask is None or int(np.sum(reference_object_mask > 0)) == 0:
            best_reject["reference_mask_empty"] += 1
            continue
        reference_part_info = build_contact_part_mask(
            object_mask=reference_object_mask,
            contact_means=projected,
            raw_contact_points=projected,
            hand_bbox=None,
            noun=noun,
            all_nouns=all_nouns,
            narration=narration,
            config=config,
        )
        reference_part_mask = reference_part_info["mask"]
        allow_whole_object_gate = str(contact_part_info.get("prior")) == "small_object_whole_mask_allowed"
        gate = part_aware_projection_gate(
            projected_contact_means=projected,
            reference_part_mask=reference_part_mask,
            reference_object_mask=reference_object_mask,
            visibility=track_detail["visibility"],
            affine_inlier_ratio=float(projected_detail.get("affine_inlier_ratio") or 0.0),
            config=config,
            allow_whole_object_gate=allow_whole_object_gate,
        )
        if not gate["passed"]:
            best_reject[str(gate["reason"] or "mask_gate_failed")] += 1
            continue

        centroid = projected.mean(axis=0)
        crop_bbox = compute_centered_crop_bbox(centroid, ref_img.shape, crop_size=config.crop_size)
        hand_overlap = compute_crop_hand_overlap_ratio(detections[ref_idx], ref_img.shape, crop_bbox, score_threshold=config.hand_score_threshold)
        clean_detail = evaluate_reference_cleanliness(
            frame_det=detections[ref_idx],
            image_shape=ref_img.shape,
            reference_object_mask=reference_object_mask,
            reference_part_mask=reference_part_mask,
            crop_bbox=crop_bbox,
            tracked_points=track_detail["ref_points_xy"],
            tracked_visibility=track_detail["visibility"],
            config=config,
        )
        tau_ref = project_trajectory_bbox_relative(trajectory_pts, seed_bbox_px, ref_bbox)
        if not tau_ref:
            best_reject["missing_trajectory_points"] += 1
            continue
        if any(pt[0] < 0 or pt[1] < 0 or pt[0] > w_ref or pt[1] > h_ref for pt in tau_ref):
            best_reject["trajectory_out_of_bounds"] += 1
            continue
        track_conf = float(object_track[ref_idx].get("track_confidence", 0.0))
        ref_gap = max(0, int(contact_frame) - int(ref_idx))
        projection_score = (
            2.1 * track_conf
            + 2.0 * float(gate["final_quality_score"])
            + 0.25 * int(gate["projected_contact_inside_mask_count"])
            - 0.01 * float(gate["projected_contact_to_mask_distance"])
        )
        if projection_method == "cotracker_weighted_affine":
            projection_score += 0.35
        if projection_method == "contact_part_local_affine":
            projection_score += 0.55
            if gate.get("part_gate_passed"):
                projection_score += 0.25
        elif projection_method == "whole_object_affine_fallback":
            projection_score -= 0.18
        if gate.get("gate_source") == "whole_object_mask_fallback":
            projection_score -= 0.15
        if projection_method in {"cotracker_weighted_affine", "contact_part_local_affine", "whole_object_affine_fallback"}:
            if projected_detail.get("affine_inlier_ratio") is not None:
                projection_score += 0.35 * float(projected_detail["affine_inlier_ratio"])
            if projected_detail.get("affine_median_residual") is not None:
                projection_score -= 0.03 * float(projected_detail["affine_median_residual"])
        clean_score = float(clean_detail["reference_clean_score"])
        score = (
            float(projection_score)
            + float(config.clean_score_weight) * clean_score
            - float(config.reference_gap_penalty) * ref_gap
        )
        current = {
            "passed": True,
            "ref_idx": int(ref_idx),
            "ref_selection_reason": f"{ref_reason};target_aware_clean_rank",
            "projection_source": projection_method,
            "projection_method": projection_method,
            "projection_status": "success",
            "mask_source_contact": contact_mask_source,
            "mask_source_reference": ref_object_mask_info.get("mask_source"),
            "tracker_backend": str(track_detail.get("backend") or "fallback_lk"),
            "cotracker_init_status": track_detail.get("cotracker_init_status"),
            "cotracker_init_error": track_detail.get("cotracker_init_error"),
            "contact_part_prior": contact_part_info.get("prior"),
            "contact_part_mask_area": int(contact_part_info.get("area") or 0),
            "contact_part_mask_area_ratio": float(contact_part_info.get("area_ratio") or 0.0),
            "reference_part_mask_area": int(reference_part_info.get("area") or 0),
            "reference_part_mask_area_ratio": float(reference_part_info.get("area_ratio") or 0.0),
            "part_gate_passed": bool(gate.get("part_gate_passed")),
            "part_inside_count": int(gate.get("part_inside_count") or 0),
            "part_distance": float(gate.get("part_distance") or 0.0),
            "part_gate_source": gate.get("gate_source"),
            "track_point_count": int(track_detail.get("track_point_count", len(tracking_points))),
            "visible_point_count": int(track_detail.get("visible_point_count", 0)),
            "affine_inlier_count": int(projected_detail.get("affine_inlier_count") or 0),
            "affine_inlier_ratio": float(projected_detail.get("affine_inlier_ratio") or 0.0),
            "affine_median_residual": float(projected_detail.get("affine_median_residual") or 0.0),
            "projected_contact_inside_mask_count": int(gate["projected_contact_inside_mask_count"]),
            "projected_contact_to_mask_distance": float(gate["projected_contact_to_mask_distance"]),
            "reference_mask_area": int(gate["reference_mask_area"]),
            "projected_contact_inside_count": int(gate["projected_contact_inside_mask_count"]),
            "projected_contact_to_object_distance": float(gate["projected_contact_to_mask_distance"]),
            "projected_contact_means": projected.astype(np.float32),
            "projected_contact_covariances": projected_covariances.astype(np.float32) if projected_covariances is not None else None,
            "reference_object_mask": reference_object_mask.astype(np.uint8),
            "reference_part_mask": reference_part_mask.astype(np.uint8),
            "reference_object_mask_source": ref_object_mask_info.get("mask_source"),
            "reference_object_mask_backend": ref_object_mask_info.get("backend"),
            "projection_quality": float(gate["final_quality_score"]),
            "final_quality_score": float(gate["final_quality_score"]),
            "reference_hand_overlap_ratio": float(hand_overlap),
            "reference_global_hand_count": int(clean_detail["reference_global_hand_count"]),
            "reference_hand_area_ratio": float(clean_detail["reference_hand_area_ratio"]),
            "reference_object_hand_overlap_ratio": float(clean_detail["reference_object_hand_overlap_ratio"]),
            "reference_contact_crop_hand_overlap_ratio": float(clean_detail["reference_contact_crop_hand_overlap_ratio"]),
            "reference_contact_part_hand_overlap_ratio": float(clean_detail["reference_contact_part_hand_overlap_ratio"]),
            "reference_hand_to_object_distance": float(clean_detail["reference_hand_to_object_distance"]),
            "reference_hand_to_contact_part_distance": float(clean_detail["reference_hand_to_contact_part_distance"]),
            "reference_target_visible_score": float(clean_detail["reference_target_visible_score"]),
            "reference_clean_class": clean_detail["reference_clean_class"],
            "reference_clean_score": float(clean_detail["reference_clean_score"]),
            "reference_selection_tier": clean_detail["reference_selection_tier"],
            "reference_gap": int(ref_gap),
            "trajectory_projected": tau_ref,
            "crop_bbox": tuple(int(v) for v in crop_bbox),
            "projection_score": float(projection_score),
            "sample_score": float(score),
            "tracked_reference_points": track_detail["ref_points_xy"].astype(np.float32),
            "tracked_reference_visibility": track_detail["visibility"].astype(bool),
            "tracked_reference_confidence": track_detail["confidence"].astype(np.float32),
            "contact_mask_bbox": contact_mask_bbox,
            "contact_mask": contact_mask,
        }
        if best_candidate is None or current["sample_score"] > best_candidate["sample_score"]:
            best_candidate = current

    if best_candidate is not None:
        return best_candidate
    reason = best_reject.most_common(1)[0][0] if best_reject else "no_reference_candidate_evaluated"
    return {"passed": False, "reason": reason, "reject_counts": dict(best_reject)}


def save_sample_outputs(
    *,
    sample_dir: Path,
    cache: E12FrameCache,
    record: Dict[str, Any],
    contact_weights: np.ndarray,
) -> Dict[str, Any]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    ref_idx = int(record["ref_idx"])
    ref_bgr = cache.load_bgr(ref_idx)
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    h, w = ref_rgb.shape[:2]
    centers = np.asarray(record["projected_contact_means"], dtype=np.float32)
    weights = np.asarray(contact_weights, dtype=np.float32)
    covariances = record.get("projected_contact_covariances")
    if isinstance(covariances, str):
        covariances = json.loads(covariances)
    covariances = None if covariances is None else np.asarray(covariances, dtype=np.float32)
    heatmap_mode = "covariance_e12_object_aware" if covariances is not None else "isotropic_e7"
    per_mode = build_label_heatmaps(
        image_shape=(h, w),
        centers_xy=centers,
        sigma_px=12.0,
        weights=weights,
        normalize_each=True,
        covariances_xy=covariances,
        covariance_scale=6.0,
        min_sigma_px=7.0,
        max_sigma_px=40.0,
    )
    merged = merge_label_heatmaps(per_mode_heatmaps=per_mode, merge_method="sum", normalize_output=True)
    heatmap_paths = silent_call(
        save_label_heatmap_outputs,
        merged_heatmap=merged,
        ref_image=ref_rgb,
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
    cv2.imwrite(str(reference_path), ref_bgr)
    tau = [np.asarray(pt, dtype=np.float32) for pt in record.get("trajectory_projected", [])]
    anchor = centers.mean(axis=0)
    vrb_img, arrow_info = draw_vrb_style_affordance_overlay(ref_rgb, merged, tau, anchor)
    mask = record.get("reference_object_mask")
    if isinstance(mask, str):
        try:
            mask = json.loads(mask)
        except Exception:
            mask = None
    if mask is not None:
        mask = np.asarray(mask, dtype=np.uint8)
        mask = _make_binary_mask(mask, (h, w))
        green = np.zeros_like(vrb_img)
        green[..., 1] = mask
        vrb_img = cv2.addWeighted(vrb_img, 1.0, green, 0.35, 0.0)
        bbox = _mask_bbox(mask)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vrb_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    tracked_points = record.get("tracked_reference_points")
    tracked_vis = record.get("tracked_reference_visibility")
    if isinstance(tracked_points, str):
        tracked_points = json.loads(tracked_points)
    if isinstance(tracked_vis, str):
        tracked_vis = json.loads(tracked_vis)
    if tracked_points is not None:
        tracked_points = np.asarray(tracked_points, dtype=np.float32).reshape(-1, 2)
        tracked_vis = np.asarray(tracked_vis, dtype=bool).reshape(-1) if tracked_vis is not None else np.ones(len(tracked_points), dtype=bool)
        for pt, vis in zip(tracked_points, tracked_vis):
            color = (0, 255, 255) if vis else (128, 128, 128)
            cv2.circle(vrb_img, (int(round(pt[0])), int(round(pt[1]))), 2, color, -1)
    for pt in centers:
        cv2.circle(vrb_img, (int(round(pt[0])), int(round(pt[1]))), 3, (255, 255, 255), -1)
    vrb_path = sample_dir / "vrb_style_affordance.png"
    cv2.imwrite(str(vrb_path), cv2.cvtColor(vrb_img, cv2.COLOR_RGB2BGR))
    mask_overlay_path = sample_dir / "mask_projection_overlay.png"
    cv2.imwrite(str(mask_overlay_path), cv2.cvtColor(vrb_img, cv2.COLOR_RGB2BGR))
    return {
        "reference_frame": str(reference_path),
        "label_heatmap_npy": heatmap_paths["npy_path"],
        "label_heatmap_png": heatmap_paths["png_path"],
        "label_heatmap_overlay": heatmap_paths["overlay_path"],
        "vrb_style_affordance": str(vrb_path),
        "mask_projection_overlay": str(mask_overlay_path),
        "heatmap_shape": (int(h), int(w)),
        "heatmap_mode": heatmap_mode,
        "arrow_generated": arrow_info is not None,
    }


def run_one_candidate(
    *,
    row: pd.Series,
    candidate: Dict[str, Any],
    candidate_index: int,
    detections,
    cache: E12FrameCache,
    contact_config: ContactExtractionConfig,
    config: E12Config,
    raw_candidate_count: int,
    tried_candidate_count: int,
) -> Dict[str, Any]:
    start_0, stop_0 = subaction_bounds(row, len(detections))
    sample_dir = (
        config.output_root
        / "experiments"
        / EXPERIMENT["experiment_id"]
        / "pipeline_outputs"
        / f"subaction_{int(row.name):02d}_{row['narration_id']}"
        / f"candidate_{int(candidate_index):02d}_frame_{int(candidate['frame']):06d}_{candidate['hand']}"
    )
    record: Dict[str, Any] = {
        "experiment_id": EXPERIMENT["experiment_id"],
        "subaction_index": int(row.name),
        "narration_id": row["narration_id"],
        "narration": row["narration"],
        "verb": row["verb"],
        "noun": row["noun"],
        "all_nouns": row.get("all_nouns"),
        "start_frame": int(row["start_frame"]),
        "stop_frame": int(row["stop_frame"]),
        "clipped_start_frame_0_based": int(start_0),
        "clipped_stop_frame_0_based": int(stop_0),
        "candidate_index": int(candidate_index),
        "frame_0_based": int(candidate["frame"]),
        "hand": candidate["hand"],
        "candidate_source": candidate["candidate_source"],
        "raw_candidate_count": int(raw_candidate_count),
        "tried_candidate_count": int(tried_candidate_count),
        "contact_points": 0,
        "active_object_bbox": None,
        "active_object_selection_score": None,
        "mask_source_contact": None,
        "mask_source_reference": None,
        "tracker_backend": None,
        "cotracker_init_status": None,
        "cotracker_init_error": None,
        "contact_part_prior": None,
        "contact_part_mask_area": 0,
        "contact_part_mask_area_ratio": None,
        "reference_part_mask_area": None,
        "reference_part_mask_area_ratio": None,
        "part_gate_passed": None,
        "part_inside_count": 0,
        "part_distance": None,
        "part_gate_source": None,
        "part_tracking_point_count": 0,
        "whole_boundary_fallback_point_count": 0,
        "projection_method": None,
        "articulated_object": bool(is_articulated_object_candidate(row["noun"], row.get("all_nouns"), row["narration"])),
        "final_quality_score": None,
        "track_point_count": 0,
        "visible_point_count": 0,
        "affine_inlier_count": 0,
        "affine_inlier_ratio": None,
        "affine_median_residual": None,
        "projected_contact_inside_mask_count": 0,
        "projected_contact_to_mask_distance": None,
        "reference_mask_area": None,
        "object_track_status": "not_run",
        "object_track_length": 0,
        "object_track_confidence": 0.0,
        "track_break_reason": None,
        "ref_idx": None,
        "ref_selection_reason": None,
        "reference_gap": None,
        "projection_source": None,
        "projection_status": "not_run",
        "projected_contact_inside_count": 0,
        "projected_contact_to_object_distance": None,
        "reference_hand_overlap_ratio": None,
        "reference_global_hand_count": None,
        "reference_hand_area_ratio": None,
        "reference_object_hand_overlap_ratio": None,
        "reference_contact_crop_hand_overlap_ratio": None,
        "reference_contact_part_hand_overlap_ratio": None,
        "reference_hand_to_object_distance": None,
        "reference_hand_to_contact_part_distance": None,
        "reference_target_visible_score": None,
        "reference_clean_class": None,
        "reference_clean_score": None,
        "reference_selection_tier": None,
        "local_flow_steps": None,
        "local_flow_fb_error": None,
        "local_flow_points": None,
        "local_flow_inliers": None,
        "local_flow_inlier_ratio": None,
        "local_flow_weighted_residual": None,
        "trajectory_status": "not_run",
        "status": "discard",
        "fail_reason": None,
        "sample_score": None,
        "sample_dir": str(sample_dir),
        "best_sample_dir": None,
        "reference_frame": None,
        "label_heatmap_overlay": None,
        "vrb_style_affordance": None,
    }
    try:
        img = cache.load_bgr(int(candidate["frame"]))
        h, w = img.shape[:2]
        frame_det = detections[int(candidate["frame"])]
        hand_norm = get_active_hand_bbox(frame_det, candidate["hand"], score_threshold=config.hand_score_threshold)
        object_norms = get_valid_object_bboxes(frame_det, score_threshold=config.object_score_threshold)
        if hand_norm is None or not object_norms:
            record.update(fail_reason="missing_hand_or_object_bbox")
            return record
        preliminary = extract_contact_points(img, hand_norm, object_norms, config=contact_config)
        pre_points = np.asarray(preliminary["contact_points"], dtype=np.float32).reshape(-1, 2)
        active = temporal_active_object_selection(
            detections=detections,
            cache=cache,
            frame_idx=int(candidate["frame"]),
            active_hand=candidate["hand"],
            contact_points=pre_points,
            config=config,
        )
        if not active["passed"]:
            record.update(fail_reason=active["reason"])
            return record
        record["active_object_bbox"] = json.dumps([round(float(v), 2) for v in active["bbox"]], ensure_ascii=False)
        record["active_object_selection_score"] = float(active["score"])
        cell2 = fit_contact_gmm_with_object(
            detections=detections,
            cache=cache,
            frame_idx=int(candidate["frame"]),
            active_hand=candidate["hand"],
            object_bbox_norm=active["bbox_norm"],
            contact_config=contact_config,
            config=config,
        )
        record["contact_points"] = int(cell2.get("contact_points", 0))
        if not cell2["passed"]:
            record.update(fail_reason=cell2["reason"])
            return record

        contact_mask_info = get_object_mask(
            frame_idx=int(candidate["frame"]),
            image_bgr=img,
            active_object_bbox=active["bbox"],
            noun=row["noun"],
            config=config,
        )
        record["mask_source_contact"] = contact_mask_info.get("mask_source")
        contact_object_mask = contact_mask_info.get("mask")
        if contact_object_mask is None or int(np.sum(contact_object_mask > 0)) == 0:
            record.update(fail_reason="contact_mask_empty")
            return record
        contact_part_info = build_contact_part_mask(
            object_mask=contact_object_mask,
            contact_means=cell2["contact_means"],
            raw_contact_points=cell2.get("raw_contact_points"),
            hand_bbox=hand_norm,
            noun=row["noun"],
            all_nouns=row.get("all_nouns"),
            narration=row["narration"],
            config=config,
        )
        record.update(
            contact_part_prior=contact_part_info.get("prior"),
            contact_part_mask_area=int(contact_part_info.get("area") or 0),
            contact_part_mask_area_ratio=float(contact_part_info.get("area_ratio") or 0.0),
        )
        part_tracking_pack = sample_contact_part_points_for_tracking(
            object_mask=contact_object_mask,
            contact_part_mask=contact_part_info["mask"],
            contact_means=cell2["contact_means"],
            hand_bbox=hand_norm if hand_norm is not None else None,
            config=config,
        )
        whole_tracking_pack = sample_object_points_for_tracking(
            object_mask=contact_object_mask,
            contact_means=cell2["contact_means"],
            hand_bbox=hand_norm if hand_norm is not None else None,
            config=config,
        )
        tracking_points = part_tracking_pack["points_xy"]
        record["part_tracking_point_count"] = int(part_tracking_pack.get("part_point_count", 0))
        record["whole_boundary_fallback_point_count"] = int(part_tracking_pack.get("whole_boundary_fallback_point_count", 0))
        record["track_point_count"] = int(part_tracking_pack["count"])
        if len(tracking_points) < int(config.part_min_tracking_points):
            record.update(fail_reason="insufficient_tracking_points")
            return record

        min_frame = max(0, int(candidate["frame"]) - int(config.max_ref_backtrack))
        track_result = track_object_backward(
            detections=detections,
            cache=cache,
            contact_frame=int(candidate["frame"]),
            seed_bbox_px=active["bbox"],
            min_frame=min_frame,
            config=config,
        )
        record.update(
            object_track_status=track_result["status"],
            object_track_length=int(track_result["track_length"]),
            object_track_confidence=float(track_result["track_confidence"]),
            track_break_reason=track_result.get("track_break_reason"),
        )
        if track_result["status"] != "success":
            record.update(fail_reason=track_result.get("track_break_reason") or "object_track_failed")
            return record
        traj_pts, missing_traj = trajectory_pixels(
            detections=detections,
            frame_idx=int(candidate["frame"]),
            active_hand=candidate["hand"],
            image_shape=img.shape,
        )
        record["trajectory_points"] = int(len(traj_pts))
        record["missing_trajectory_count"] = int(len(missing_traj))
        if not traj_pts:
            record.update(trajectory_status="failed", fail_reason="missing_trajectory_points")
            return record
        ref_eval = evaluate_reference_candidates(
            detections=detections,
            cache=cache,
            object_track=track_result["track"],
            contact_frame=int(candidate["frame"]),
            seed_bbox_px=active["bbox"],
            contact_means=cell2["contact_means"],
            contact_covariances=cell2["contact_covariances"],
            trajectory_pts=traj_pts,
            tracking_points=tracking_points,
            whole_tracking_points=whole_tracking_pack["points_xy"],
            contact_part_mask=contact_part_info["mask"],
            contact_part_info=contact_part_info,
            active_hand=candidate["hand"],
            hand_bbox_px=hand_norm,
            noun=row["noun"],
            all_nouns=row.get("all_nouns"),
            narration=row["narration"],
            search_start=int(candidate["reference_search_window_start_0_based"]),
            search_end=int(candidate["reference_search_window_end_0_based"]),
            config=config,
        )
        if not ref_eval["passed"]:
            record.update(
                projection_status="failed",
                trajectory_status="failed" if str(ref_eval["reason"]).startswith("trajectory") else "not_run",
                fail_reason=ref_eval["reason"],
            )
            return record
        ref_idx = int(ref_eval["ref_idx"])
        ref_bbox = track_result["track"][ref_idx]["bbox"]
        record.update(
            ref_idx=ref_idx,
            ref_selection_reason=ref_eval["ref_selection_reason"],
            projection_source=ref_eval["projection_source"],
            projection_method=ref_eval.get("projection_method", ref_eval["projection_source"]),
            mask_source_contact=ref_eval.get("mask_source_contact", record.get("mask_source_contact")),
            mask_source_reference=ref_eval.get("mask_source_reference"),
            tracker_backend=ref_eval.get("tracker_backend"),
            cotracker_init_status=ref_eval.get("cotracker_init_status"),
            cotracker_init_error=ref_eval.get("cotracker_init_error"),
            contact_part_prior=ref_eval.get("contact_part_prior"),
            contact_part_mask_area=ref_eval.get("contact_part_mask_area"),
            contact_part_mask_area_ratio=ref_eval.get("contact_part_mask_area_ratio"),
            reference_part_mask_area=ref_eval.get("reference_part_mask_area"),
            reference_part_mask_area_ratio=ref_eval.get("reference_part_mask_area_ratio"),
            part_gate_passed=ref_eval.get("part_gate_passed"),
            part_inside_count=ref_eval.get("part_inside_count"),
            part_distance=ref_eval.get("part_distance"),
            part_gate_source=ref_eval.get("part_gate_source"),
            final_quality_score=ref_eval.get("final_quality_score"),
            track_point_count=ref_eval.get("track_point_count", record["track_point_count"]),
            visible_point_count=ref_eval.get("visible_point_count", record["visible_point_count"]),
            affine_inlier_count=ref_eval.get("affine_inlier_count", 0),
            affine_inlier_ratio=ref_eval.get("affine_inlier_ratio"),
            affine_median_residual=ref_eval.get("affine_median_residual"),
            projected_contact_inside_mask_count=ref_eval.get("projected_contact_inside_mask_count", 0),
            projected_contact_to_mask_distance=ref_eval.get("projected_contact_to_mask_distance"),
            reference_mask_area=ref_eval.get("reference_mask_area"),
            reference_gap=ref_eval.get("reference_gap"),
            projection_status="success",
            projected_contact_inside_count=int(ref_eval["projected_contact_inside_count"]),
            projected_contact_to_object_distance=float(ref_eval["projected_contact_to_object_distance"]),
            reference_hand_overlap_ratio=float(ref_eval["reference_hand_overlap_ratio"]),
            reference_global_hand_count=ref_eval.get("reference_global_hand_count"),
            reference_hand_area_ratio=ref_eval.get("reference_hand_area_ratio"),
            reference_object_hand_overlap_ratio=ref_eval.get("reference_object_hand_overlap_ratio"),
            reference_contact_crop_hand_overlap_ratio=ref_eval.get("reference_contact_crop_hand_overlap_ratio"),
            reference_contact_part_hand_overlap_ratio=ref_eval.get("reference_contact_part_hand_overlap_ratio"),
            reference_hand_to_object_distance=ref_eval.get("reference_hand_to_object_distance"),
            reference_hand_to_contact_part_distance=ref_eval.get("reference_hand_to_contact_part_distance"),
            reference_target_visible_score=ref_eval.get("reference_target_visible_score"),
            reference_clean_class=ref_eval.get("reference_clean_class"),
            reference_clean_score=ref_eval.get("reference_clean_score"),
            reference_selection_tier=ref_eval.get("reference_selection_tier"),
            trajectory_status="success",
            sample_score=float(ref_eval["sample_score"]),
            tracked_reference_object_bbox=[round(float(v), 2) for v in ref_bbox],
            projected_contact_means=np.asarray(ref_eval["projected_contact_means"], dtype=np.float32),
            projected_contact_covariances=ref_eval.get("projected_contact_covariances"),
            trajectory_projected=ref_eval["trajectory_projected"],
            reference_object_mask=ref_eval.get("reference_object_mask"),
            reference_part_mask=ref_eval.get("reference_part_mask"),
            tracked_reference_points=ref_eval.get("tracked_reference_points"),
            tracked_reference_visibility=ref_eval.get("tracked_reference_visibility"),
            tracked_reference_confidence=ref_eval.get("tracked_reference_confidence"),
            contact_mask=ref_eval.get("contact_mask"),
            local_flow_steps=ref_eval.get("local_flow_steps"),
            local_flow_fb_error=ref_eval.get("local_flow_fb_error"),
            local_flow_points=ref_eval.get("local_flow_points"),
            local_flow_inliers=ref_eval.get("local_flow_inliers"),
            local_flow_inlier_ratio=ref_eval.get("local_flow_inlier_ratio"),
            local_flow_weighted_residual=ref_eval.get("local_flow_weighted_residual"),
            status="keep",
            fail_reason=None,
        )
        artifacts = save_sample_outputs(sample_dir=sample_dir, cache=cache, record=record, contact_weights=cell2["contact_weights"])
        record.update(artifacts)
        record["best_sample_dir"] = str(sample_dir)
        (sample_dir / "candidate_result.json").write_text(json.dumps(_json_safe(record), ensure_ascii=False, indent=2), encoding="utf-8")
        record.pop("projected_contact_means", None)
        record.pop("projected_contact_covariances", None)
        record.pop("trajectory_projected", None)
        record.pop("reference_object_mask", None)
        record.pop("reference_part_mask", None)
        record.pop("tracked_reference_points", None)
        record.pop("tracked_reference_visibility", None)
        record.pop("tracked_reference_confidence", None)
        record.pop("contact_mask", None)
        return record
    except Exception as exc:
        record.update(status="discard", fail_reason=f"exception:{type(exc).__name__}:{exc}")
        return record


def empty_subaction_candidate(row: pd.Series, reason: str, raw_candidate_count: int) -> Dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT["experiment_id"],
        "subaction_index": int(row.name),
        "narration_id": row["narration_id"],
        "narration": row["narration"],
        "verb": row["verb"],
        "noun": row["noun"],
        "start_frame": int(row["start_frame"]),
        "stop_frame": int(row["stop_frame"]),
        "raw_candidate_count": int(raw_candidate_count),
        "tried_candidate_count": 0,
        "frame_0_based": None,
        "hand": None,
        "candidate_source": None,
        "contact_points": 0,
        "active_object_bbox": None,
        "active_object_selection_score": None,
        "mask_source_contact": None,
        "mask_source_reference": None,
        "tracker_backend": None,
        "cotracker_init_status": None,
        "cotracker_init_error": None,
        "contact_part_prior": None,
        "contact_part_mask_area": 0,
        "contact_part_mask_area_ratio": None,
        "reference_part_mask_area": None,
        "reference_part_mask_area_ratio": None,
        "part_gate_passed": None,
        "part_inside_count": 0,
        "part_distance": None,
        "part_gate_source": None,
        "part_tracking_point_count": 0,
        "whole_boundary_fallback_point_count": 0,
        "projection_method": None,
        "articulated_object": bool(is_articulated_object_candidate(row["noun"], row.get("all_nouns"), row["narration"])),
        "final_quality_score": None,
        "track_point_count": 0,
        "visible_point_count": 0,
        "affine_inlier_count": 0,
        "affine_inlier_ratio": None,
        "affine_median_residual": None,
        "projected_contact_inside_mask_count": 0,
        "projected_contact_to_mask_distance": None,
        "reference_mask_area": None,
        "object_track_status": "not_run",
        "object_track_length": 0,
        "object_track_confidence": 0.0,
        "track_break_reason": None,
        "ref_idx": None,
        "ref_selection_reason": None,
        "reference_gap": None,
        "projection_source": None,
        "projection_status": "not_run",
        "projected_contact_inside_count": 0,
        "projected_contact_to_object_distance": None,
        "reference_hand_overlap_ratio": None,
        "reference_global_hand_count": None,
        "reference_hand_area_ratio": None,
        "reference_object_hand_overlap_ratio": None,
        "reference_contact_crop_hand_overlap_ratio": None,
        "reference_contact_part_hand_overlap_ratio": None,
        "reference_hand_to_object_distance": None,
        "reference_hand_to_contact_part_distance": None,
        "reference_target_visible_score": None,
        "reference_clean_class": None,
        "reference_clean_score": None,
        "reference_selection_tier": None,
        "local_flow_steps": None,
        "local_flow_fb_error": None,
        "local_flow_points": None,
        "local_flow_inliers": None,
        "local_flow_inlier_ratio": None,
        "local_flow_weighted_residual": None,
        "trajectory_status": "not_run",
        "status": "discard",
        "fail_reason": reason,
        "sample_score": None,
        "sample_dir": None,
    }


def finalize_one_sample_per_subaction(candidate_df: pd.DataFrame, config: Optional[E12Config] = None) -> pd.DataFrame:
    df = candidate_df.copy()
    config = config or E12Config()
    keep_idx = []
    for _, sub_df in df[df["status"] == "keep"].groupby("subaction_index"):
        chosen_frames: List[float] = []
        ordered = sub_df.sort_values("sample_score", ascending=False)
        for row in ordered.itertuples():
            frame_val = float(getattr(row, "frame_0_based"))
            if any(abs(frame_val - prev) < int(config.final_frame_dedup_gap) for prev in chosen_frames):
                continue
            keep_idx.append(row.Index)
            chosen_frames.append(frame_val)
            if len(chosen_frames) >= int(config.final_max_tuples_per_subaction):
                break
    df["is_final_tuple"] = False
    if keep_idx:
        df.loc[keep_idx, "is_final_tuple"] = True
    return df


def summarize_subactions(candidate_df: pd.DataFrame, subactions: pd.DataFrame, plan_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in subactions.iterrows():
        sub_df = candidate_df[candidate_df["subaction_index"] == idx].copy()
        final_df = sub_df[sub_df.get("is_final_tuple", False) == True].copy()
        best = final_df.iloc[0] if not final_df.empty else None
        if best is not None:
            final_status = "keep"
            fail_reason = None
        elif not sub_df.empty:
            final_status = "discard"
            fail_reason = sub_df["fail_reason"].fillna("unknown").value_counts().index[0]
        else:
            final_status = "discard"
            fail_reason = "no_candidate_record"
        plan = plan_df[plan_df["subaction_index"] == idx]
        plan_row = plan.iloc[0] if not plan.empty else None
        rows.append(
            {
                "experiment_id": EXPERIMENT["experiment_id"],
                "subaction_index": int(idx),
                "narration_id": row["narration_id"],
                "narration": row["narration"],
                "verb": row["verb"],
                "noun": row["noun"],
                "start_frame": int(row["start_frame"]),
                "stop_frame": int(row["stop_frame"]),
                "raw_candidate_count": int(plan_row["raw_candidate_count"]) if plan_row is not None else 0,
                "tried_candidate_count": int(len(sub_df[sub_df["frame_0_based"].notna()])) if not sub_df.empty else 0,
                "final_status": final_status,
                "final_fail_reason": fail_reason,
                "best_candidate_frame_0_based": int(best["frame_0_based"]) if best is not None and pd.notna(best["frame_0_based"]) else None,
                "best_ref_idx": int(best["ref_idx"]) if best is not None and pd.notna(best["ref_idx"]) else None,
                "best_reference_gap": int(best["reference_gap"]) if best is not None and "reference_gap" in best and pd.notna(best["reference_gap"]) else None,
                "best_projection_source": best["projection_source"] if best is not None else None,
                "best_projection_method": best.get("projection_method") if best is not None else None,
                "best_track_confidence": float(best["object_track_confidence"]) if best is not None else None,
                "best_sample_score": float(best["sample_score"]) if best is not None else None,
                "best_sample_dir": best["sample_dir"] if best is not None else None,
                "contact_points": int(best["contact_points"]) if best is not None else 0,
                "projected_contact_inside_count": int(best["projected_contact_inside_count"]) if best is not None else 0,
                "reference_hand_overlap_ratio": float(best["reference_hand_overlap_ratio"]) if best is not None and pd.notna(best["reference_hand_overlap_ratio"]) else None,
                "reference_object_hand_overlap_ratio": float(best["reference_object_hand_overlap_ratio"]) if best is not None and "reference_object_hand_overlap_ratio" in best and pd.notna(best["reference_object_hand_overlap_ratio"]) else None,
                "reference_contact_part_hand_overlap_ratio": float(best["reference_contact_part_hand_overlap_ratio"]) if best is not None and "reference_contact_part_hand_overlap_ratio" in best and pd.notna(best["reference_contact_part_hand_overlap_ratio"]) else None,
                "reference_clean_class": best.get("reference_clean_class") if best is not None else None,
                "reference_selection_tier": best.get("reference_selection_tier") if best is not None else None,
                "best_mask_source_contact": best.get("mask_source_contact") if best is not None else None,
                "best_mask_source_reference": best.get("mask_source_reference") if best is not None else None,
                "best_tracker_backend": best.get("tracker_backend") if best is not None else None,
                "best_final_quality_score": float(best["final_quality_score"]) if best is not None and pd.notna(best.get("final_quality_score")) else None,
                "best_affine_inlier_ratio": float(best["affine_inlier_ratio"]) if best is not None and pd.notna(best.get("affine_inlier_ratio")) else None,
            }
        )
    return pd.DataFrame(rows)


def failure_bucket(reason: Any) -> str:
    text = str(reason or "unknown")
    if "mask" in text or "sam2" in text or "visor" in text:
        return "mask"
    if "object" in text or "bbox" in text or "detection" in text:
        return "detection"
    if "track" in text or "flow" in text or "affine" in text:
        return "tracking"
    if "projection" in text or "projected" in text or "reference" in text:
        return "projection"
    if "trajectory" in text:
        return "trajectory"
    if "contact" in text or "gmm" in text:
        return "contact_gmm"
    return "other"


def build_overview(candidate_df: pd.DataFrame, subaction_summary_df: pd.DataFrame, plan_df: pd.DataFrame) -> pd.DataFrame:
    final = candidate_df[candidate_df.get("is_final_tuple", False) == True].copy()
    failure_counts = candidate_df[candidate_df["status"] != "keep"]["fail_reason"].fillna("unknown").value_counts()
    main_failure = failure_counts.index[0] if len(failure_counts) else None
    articulated_df = candidate_df[candidate_df["articulated_object"] == True].copy() if "articulated_object" in candidate_df else pd.DataFrame()
    articulated_success = int(((articulated_df["status"] == "keep").sum()) if not articulated_df.empty else 0)
    articulated_failure = int(((articulated_df["status"] != "keep").sum()) if not articulated_df.empty else 0)
    cotracker_success = int((candidate_df["tracker_backend"] == "cotracker").sum()) if "tracker_backend" in candidate_df else 0
    fallback_success = int((candidate_df["tracker_backend"] == "fallback_lk").sum()) if "tracker_backend" in candidate_df else 0
    clean_counts = final["reference_clean_class"].fillna("unknown").value_counts().to_dict() if "reference_clean_class" in final else {}
    tier_counts = final["reference_selection_tier"].fillna("unknown").value_counts().to_dict() if "reference_selection_tier" in final else {}
    return pd.DataFrame(
        [
            {
                "experiment_id": EXPERIMENT["experiment_id"],
                "subactions_total": int(len(subaction_summary_df)),
                "subactions_success": int((subaction_summary_df["final_status"] == "keep").sum()),
                "final_tuple_count": int(len(final)),
                "raw_candidate_total": int(plan_df["raw_candidate_count"].sum()) if not plan_df.empty else 0,
                "deep_run_candidate_total": int(plan_df["deep_run_candidate_count"].sum()) if not plan_df.empty else 0,
                "contact_gmm_pass": int((candidate_df["contact_points"] >= 5).sum()) if not candidate_df.empty else 0,
                "active_object_selected": int(candidate_df["active_object_bbox"].notna().sum()) if "active_object_bbox" in candidate_df else 0,
                "object_track_success": int((candidate_df["object_track_status"] == "success").sum()) if "object_track_status" in candidate_df else 0,
                "reference_found": int(candidate_df["ref_idx"].notna().sum()) if "ref_idx" in candidate_df else 0,
                "projection_success": int((candidate_df["projection_status"] == "success").sum()) if "projection_status" in candidate_df else 0,
                "mask_gate_pass": int(((candidate_df["status"] == "keep") & candidate_df["final_quality_score"].notna()).sum()) if "final_quality_score" in candidate_df else 0,
                "part_gate_pass": int((candidate_df["part_gate_passed"] == True).sum()) if "part_gate_passed" in candidate_df else 0,
                "whole_object_gate_fallback_pass": int((candidate_df["part_gate_source"] == "whole_object_mask_fallback").sum()) if "part_gate_source" in candidate_df else 0,
                "cotracker_success": cotracker_success,
                "fallback_tracker_success": fallback_success,
                "cotracker_init_status": _COTRACKER_INIT_STATUS,
                "cotracker_init_error": _COTRACKER_INIT_ERROR,
                "strict_target_handless_count": int(clean_counts.get("strict_target_handless", 0)),
                "local_target_clean_count": int(clean_counts.get("local_target_clean", 0)),
                "hand_visible_target_not_occluded_count": int(clean_counts.get("hand_visible_target_not_occluded", 0)),
                "target_partially_occluded_count": int(clean_counts.get("target_partially_occluded", 0)),
                "target_occluded_by_hand_count": int(clean_counts.get("target_occluded_by_hand", 0)),
                "reference_clean_class_counts": json.dumps(clean_counts, ensure_ascii=False),
                "reference_selection_tier_counts": json.dumps(tier_counts, ensure_ascii=False),
                "mean_reference_hand_overlap_ratio": float(final["reference_hand_overlap_ratio"].dropna().astype(float).mean()) if "reference_hand_overlap_ratio" in final and final["reference_hand_overlap_ratio"].notna().any() else None,
                "mean_reference_object_hand_overlap_ratio": float(final["reference_object_hand_overlap_ratio"].dropna().astype(float).mean()) if "reference_object_hand_overlap_ratio" in final and final["reference_object_hand_overlap_ratio"].notna().any() else None,
                "mean_reference_contact_part_hand_overlap_ratio": float(final["reference_contact_part_hand_overlap_ratio"].dropna().astype(float).mean()) if "reference_contact_part_hand_overlap_ratio" in final and final["reference_contact_part_hand_overlap_ratio"].notna().any() else None,
                "mean_reference_target_visible_score": float(final["reference_target_visible_score"].dropna().astype(float).mean()) if "reference_target_visible_score" in final and final["reference_target_visible_score"].notna().any() else None,
                "articulated_success_count": articulated_success,
                "articulated_failure_count": articulated_failure,
                "articulated_projection_quality": float(articulated_df["final_quality_score"].dropna().mean()) if not articulated_df.empty and articulated_df["final_quality_score"].notna().any() else None,
                "trajectory_success": int((candidate_df["trajectory_status"] == "success").sum()) if "trajectory_status" in candidate_df else 0,
                "heatmap_success": int(len(final)),
                "main_failure": main_failure,
            }
        ]
    )


def save_success_manifest(candidate_df: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    final = candidate_df[candidate_df.get("is_final_tuple", False) == True].copy()
    cols = [
        "subaction_index",
        "narration_id",
        "narration",
        "verb",
        "noun",
        "frame_0_based",
        "hand",
        "candidate_source",
        "ref_idx",
        "reference_gap",
        "projection_source",
        "projection_method",
        "mask_source_contact",
        "mask_source_reference",
        "tracker_backend",
        "cotracker_init_status",
        "cotracker_init_error",
        "contact_part_prior",
        "contact_part_mask_area",
        "contact_part_mask_area_ratio",
        "reference_part_mask_area",
        "reference_part_mask_area_ratio",
        "part_gate_passed",
        "part_inside_count",
        "part_distance",
        "part_gate_source",
        "part_tracking_point_count",
        "whole_boundary_fallback_point_count",
        "articulated_object",
        "final_quality_score",
        "track_point_count",
        "visible_point_count",
        "affine_inlier_count",
        "affine_inlier_ratio",
        "affine_median_residual",
        "projected_contact_inside_mask_count",
        "projected_contact_to_mask_distance",
        "object_track_confidence",
        "local_flow_steps",
        "local_flow_fb_error",
        "local_flow_points",
        "local_flow_inliers",
        "local_flow_inlier_ratio",
        "local_flow_weighted_residual",
        "sample_score",
        "contact_points",
        "projected_contact_inside_count",
        "reference_hand_overlap_ratio",
        "reference_global_hand_count",
        "reference_hand_area_ratio",
        "reference_object_hand_overlap_ratio",
        "reference_contact_crop_hand_overlap_ratio",
        "reference_contact_part_hand_overlap_ratio",
        "reference_hand_to_object_distance",
        "reference_hand_to_contact_part_distance",
        "reference_target_visible_score",
        "reference_clean_class",
        "reference_clean_score",
        "reference_selection_tier",
        "heatmap_mode",
        "sample_dir",
        "reference_frame",
        "label_heatmap_overlay",
        "vrb_style_affordance",
        "mask_projection_overlay",
    ]
    for col in cols:
        if col not in final.columns:
            final[col] = None
    final[cols].to_csv(output_root / "successful_sample_manifest.csv", index=False)
    return final


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
    ax.set_title("E12 pipeline funnel")
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
    ax.set_title("E12 failure reasons")
    ax.set_xlabel("candidate count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_track_confidence_hist(candidate_df: pd.DataFrame, path: Path) -> None:
    values = candidate_df["object_track_confidence"].dropna().astype(float) if "object_track_confidence" in candidate_df else pd.Series(dtype=float)
    if values.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=20, color="#188038", alpha=0.85)
    ax.set_title("E12 track confidence")
    ax.set_xlabel("confidence")
    ax.set_ylabel("candidate count")
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
    ax.set_title("E12 projection source counts")
    ax.set_xlabel("projection_source")
    ax.set_ylabel("success candidate count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_clean_class_counts(final_df: pd.DataFrame, path: Path) -> None:
    if final_df.empty or "reference_clean_class" not in final_df:
        return
    counts = final_df["reference_clean_class"].fillna("unknown").value_counts()
    fig, ax = plt.subplots(figsize=(9, 4))
    counts.plot(kind="bar", ax=ax, color="#0b8043")
    ax.set_title("E12 reference clean class counts")
    ax.set_xlabel("reference_clean_class")
    ax.set_ylabel("final tuple count")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_success_by_subaction(subaction_df: pd.DataFrame, path: Path) -> None:
    y = (subaction_df["final_status"] == "keep").astype(int)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(subaction_df["subaction_index"], y, color=["#188038" if v else "#dadce0" for v in y])
    ax.set_title("E12 success by subaction")
    ax.set_xlabel("subaction_index")
    ax.set_ylabel("success")
    ax.set_ylim(0, 1.2)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_contact_sheet(final_df: pd.DataFrame, path: Path, title: str, page_size: int = 36) -> List[Path]:
    if final_df.empty:
        return []
    pages = []
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
                f"{row.get('verb')} {row.get('noun')} | {row.get('reference_clean_class') or row.get('projection_source')}"
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
        if page_idx == 0 and len(rows) <= page_size:
            out = path
        else:
            out = path.with_name(f"{path.stem}_page_{page_idx + 1:03d}{path.suffix}")
        plt.savefig(out, dpi=160)
        plt.close()
        pages.append(out)
    return pages


def load_baseline_overview(root: Path, experiment_id: str) -> Optional[Dict[str, Any]]:
    path = root / "experiment_overview.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "experiment_id" in df.columns:
        df = df[df["experiment_id"] == experiment_id]
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def write_summary(
    *,
    config: E12Config,
    overview_df: pd.DataFrame,
    subaction_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    final_df: pd.DataFrame,
    contact_sheet_pages: Sequence[Path],
) -> Path:
    out = config.output_root
    failed = candidate_df[candidate_df["status"] != "keep"].copy()
    failed["failure_bucket"] = failed["fail_reason"].map(failure_bucket) if not failed.empty else []
    bucket_counts = failed["failure_bucket"].value_counts() if not failed.empty else pd.Series(dtype=int)
    top_failures = failed["fail_reason"].fillna("unknown").value_counts().head(12) if not failed.empty else pd.Series(dtype=int)
    e6b = load_baseline_overview(
        VRBREPRODUCTION_ROOT / "outputs" / "数据处理小批量测试前100个subaction_E6b_pre_contact_gap_crop_fallback",
        "E6b",
    )
    e6c = load_baseline_overview(
        VRBREPRODUCTION_ROOT / "outputs" / "数据处理小批量测试前100个subaction_E6c_object_consistency",
        "E6c",
    )
    e7 = load_baseline_overview(
        VRBREPRODUCTION_ROOT / "outputs" / "100subaction-e7",
        "E7",
    )
    e10 = load_baseline_overview(
        VRBREPRODUCTION_ROOT / "outputs" / "100subaction-e10",
        "E10",
    ) or load_baseline_overview(
        VRBREPRODUCTION_ROOT / "outputs" / "e10",
        "E10",
    )
    e11 = load_baseline_overview(
        VRBREPRODUCTION_ROOT / "outputs" / "e11",
        "E11",
    )
    row = overview_df.iloc[0]
    compare_lines = ["| 实验 | 覆盖 subaction（子动作片段） | tuple（训练样本） | 备注 |", "| --- | ---: | ---: | --- |"]
    if e7:
        compare_lines.append(f"| E7 | {int(e7.get('subactions_success', 0))} | {int(e7.get('final_tuple_count', 0))} | object tracklet + local flow + bbox gate baseline |")
    if e10:
        compare_lines.append(f"| E10 | {int(e10.get('subactions_success', 0))} | {int(e10.get('final_tuple_count', 0))} | CoTracker contact-part projection baseline |")
    if e11:
        compare_lines.append(f"| E11 | {int(e11.get('subactions_success', 0))} | {int(e11.get('final_tuple_count', 0))} | E10 selection + bbox inpainting postprocess |")
    if e6b:
        compare_lines.append(f"| E6b | {int(e6b.get('subactions_success', 0))} | {int(e6b.get('heatmap_pass', 0))} | crop-level humanless fallback（局部无手补救）但无最终 object-centric gate（物体中心门控） |")
    if e6c:
        compare_lines.append(f"| E6c conservative | {int(e6c.get('subactions_success', e6c.get('e6c_conservative_final_subactions', 0)))} | {int(e6c.get('heatmap_pass', e6c.get('e6c_conservative_final_success', 0)))} | reference-frame object consistency（参考帧物体一致性）保守口径 |")
    compare_lines.append(
        f"| E12 | {int(row['subactions_success'])} | {int(row['final_tuple_count'])} | "
        f"contact-part local projection + part-aware gate；每个 subaction 最多 {int(config.final_max_tuples_per_subaction)} 个、帧间隔至少 {int(config.final_frame_dedup_gap)} 帧 |"
    )
    failure_lines = ["| fail_reason | count |", "| --- | ---: |"]
    for reason, count in top_failures.items():
        failure_lines.append(f"| {reason} | {int(count)} |")
    bucket_lines = ["| 失败来源 | count |", "| --- | ---: |"]
    for bucket, count in bucket_counts.items():
        bucket_lines.append(f"| {bucket} | {int(count)} |")
    sheet_lines = [f"- `{p}`" for p in contact_sheet_pages] or ["- 未生成：没有最终留存样本。"]
    projection_counts = final_df["projection_source"].fillna("unknown").value_counts() if not final_df.empty else pd.Series(dtype=int)
    projection_lines = ["| projection_source（投影来源） | count |", "| --- | ---: |"]
    for source, count in projection_counts.items():
        projection_lines.append(f"| {source} | {int(count)} |")
    clean_counts = final_df["reference_clean_class"].fillna("unknown").value_counts() if not final_df.empty and "reference_clean_class" in final_df else pd.Series(dtype=int)
    tier_counts = final_df["reference_selection_tier"].fillna("unknown").value_counts() if not final_df.empty and "reference_selection_tier" in final_df else pd.Series(dtype=int)
    clean_lines = ["| reference_clean_class | final tuple count |", "| --- | ---: |"]
    for name, count in clean_counts.items():
        clean_lines.append(f"| {name} | {int(count)} |")
    tier_lines = ["| reference_selection_tier | final tuple count |", "| --- | ---: |"]
    for name, count in tier_counts.items():
        tier_lines.append(f"| {name} | {int(count)} |")
    lines = [
        "# E12 contact-part-aware CoTracker projection 实验报告",
        "",
        f"- 输出目录：`{out}`",
        f"- video_id：`{config.video_id}`",
        f"- subaction（子动作片段）总数：`{int(row['subactions_total'])}`",
        f"- 成功覆盖 subaction（子动作片段）：`{int(row['subactions_success'])}`",
        f"- final tuple（最终训练样本）：`{int(row['final_tuple_count'])}`",
        f"- object mask gate 通过数：`{int(row.get('mask_gate_pass', 0))}`",
        f"- part gate 通过数：`{int(row.get('part_gate_pass', 0))}`",
        f"- whole-object gate fallback 通过数：`{int(row.get('whole_object_gate_fallback_pass', 0))}`",
        f"- CoTracker 成功数：`{int(row.get('cotracker_success', 0))}`",
        f"- fallback tracker 成功数：`{int(row.get('fallback_tracker_success', 0))}`",
        f"- fallback tracker 比例：`{(float(row.get('fallback_tracker_success', 0)) / max(1, int(row.get('fallback_tracker_success', 0)) + int(row.get('cotracker_success', 0)))):.2%}`",
        f"- CoTracker 初始化状态：`{row.get('cotracker_init_status')}`",
        f"- CoTracker 初始化错误：`{row.get('cotracker_init_error')}`",
        f"- strict_target_handless：`{int(row.get('strict_target_handless_count', 0))}`",
        f"- local_target_clean：`{int(row.get('local_target_clean_count', 0))}`",
        f"- hand_visible_target_not_occluded：`{int(row.get('hand_visible_target_not_occluded_count', 0))}`",
        f"- target_partially_occluded：`{int(row.get('target_partially_occluded_count', 0))}`",
        f"- target_occluded_by_hand：`{int(row.get('target_occluded_by_hand_count', 0))}`",
        f"- mean reference hand overlap：`{row.get('mean_reference_hand_overlap_ratio')}`",
        f"- mean object-hand overlap：`{row.get('mean_reference_object_hand_overlap_ratio')}`",
        f"- mean contact-part hand overlap：`{row.get('mean_reference_contact_part_hand_overlap_ratio')}`",
        f"- articulated object 成功数：`{int(row.get('articulated_success_count', 0))}`",
        f"- articulated object 失败数：`{int(row.get('articulated_failure_count', 0))}`",
        "",
        "## 算法改动",
        "",
        "- contact frame 先得到 whole-object mask，再基于 contact GMM、raw contact points、hand bbox 和 noun soft prior 构造 `contact_part_mask`。",
        "- 跟踪点采样改成 `contact_part_mask` 优先，只在局部点不足时补 whole-object boundary，避免 interior 大面积主导局部运动。",
        "- 投影优先使用 `contact_part_local_affine`；若局部仿射不稳定，再退回 `whole_object_affine_fallback`，并在 diagnostics 中明确记录。",
        "- 最终 gate 先看 projected contact 是否落在 reference contact part 上；只有小物体等特例才允许 whole-object gate fallback。",
        "- 对每个通过投影 gate 的 reference frame 计算全局手数、手面积、object/contact crop/contact part 的手遮挡比例、手到目标/接触部件距离和 target visible score。",
        "- 选帧排序改为 `projection_score + clean_score - gap_penalty`：projection gate 不放松，clean score 主导同等投影质量下的 reference 选择。",
        "- CoTracker 初始化状态与错误会单独记录；`fallback_lk` 不会被伪装成 CoTracker。",
        f"- 最终导出允许每个 subaction（子动作片段）最多保留 `{int(config.final_max_tuples_per_subaction)}` 个高分 tuple（训练样本），并做 `{int(config.final_frame_dedup_gap)}` 帧的近邻去重。",
        "",
        "## 与 E6b / E6c 的关键区别",
        "",
        "\n".join(compare_lines),
        "",
        "E12 的主变化不是回到 background homography，也不是简单放宽阈值，而是把 whole-object projection/gate 升级成 contact-local / part-aware projection/gate。",
        "",
        "## Funnel",
        "",
        f"- raw candidate total（原始候选数）：`{int(row['raw_candidate_total'])}`",
        f"- deep run candidate total（实际深跑候选数）：`{int(row['deep_run_candidate_total'])}`",
        f"- contact_gmm_pass（接触点 GMM 通过）：`{int(row['contact_gmm_pass'])}`",
        f"- active_object_selected（当前接触物体选中）：`{int(row['active_object_selected'])}`",
        f"- object_track_success（物体跟踪成功）：`{int(row['object_track_success'])}`",
        f"- reference_found（参考帧找到）：`{int(row['reference_found'])}`",
        f"- projection_success（投影成功）：`{int(row['projection_success'])}`",
        f"- object_mask_gate_pass（object mask gate 通过）：`{int(row.get('mask_gate_pass', 0))}`",
        f"- part_gate_pass（part gate 通过）：`{int(row.get('part_gate_pass', 0))}`",
        f"- trajectory_success（轨迹成功）：`{int(row['trajectory_success'])}`",
        f"- heatmap_success（热图成功，即最终 tuple）：`{int(row['heatmap_success'])}`",
        f"- final export top-k（每个 subaction 最多导出）：`{int(config.final_max_tuples_per_subaction)}`",
        f"- final frame dedup gap（同一 subaction 内近邻去重帧距）：`{int(config.final_frame_dedup_gap)}`",
        f"- reference gap penalty（参考帧距离惩罚）：`{float(config.reference_gap_penalty):.4f}`",
        "",
        "## projection_source（投影来源）",
        "",
        "\n".join(projection_lines),
        "",
        "## reference_clean_class（参考帧干净程度）",
        "",
        "\n".join(clean_lines),
        "",
        "## reference_selection_tier（选帧层级）",
        "",
        "\n".join(tier_lines),
        "",
        "## 主要失败原因",
        "",
        "\n".join(failure_lines),
        "",
        "## 失败来源归类",
        "",
        "\n".join(bucket_lines),
        "",
        "## 是否接近 80 个 subaction 目标",
        "",
        f"E12 当前覆盖 `{int(row['subactions_success'])}/100` 个 subaction（子动作片段），距离 82 个 subaction 目标还差 `{max(0, 82 - int(row['subactions_success']))}` 个。`final_tuple_count` 现在反映 top-k 导出口径，不再等同于 subaction 覆盖数。",
        "",
        "## 质量风险",
        "",
        "- 如果 CoTracker 仍不可用，E12 的提升会主要来自 contact-part sampling 和 gate，而不是更强的时序跟踪。",
        "- 当前 reference part mask 仍是基于 projected contact 的轻量规则构造，不是显式 part segmentation；对于严重遮挡或视角切换，仍会退化。",
        "- articulated object 的部件区域在参考帧可能发生开合或遮挡，仍可能触发 `whole_object_affine_fallback` 或 `part_gate_failed`。",
        "",
        "## Contact Sheet",
        "",
        "\n".join(sheet_lines),
        "",
        "## 图表",
        "",
        "- `charts/e12_pipeline_funnel.png`",
        "- `charts/e12_failure_reasons.png`",
        "- `charts/e12_mask_projection_contact_sheet.png`",
        "- `charts/e12_success_contact_sheet_page_001.png`",
        "- `charts/e12_projection_source_counts.png`",
        "- `charts/e12_reference_clean_class_counts.png`",
    ]
    path = out / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_e12_experiment(config: Optional[E12Config] = None) -> Dict[str, Any]:
    global _GPU_FLOW_STATUS_PRINTED
    config = config or E12Config()
    config.output_root.mkdir(parents=True, exist_ok=True)
    charts_dir = config.output_root / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    (VRBREPRODUCTION_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)

    detections = load_detections(str(config.hoa_pkl))
    subactions = get_subactions(config)
    frame_coverage = compute_subaction_frame_coverage(subactions)
    left_binary, right_binary = build_contact_arrays(detections)
    all_runs = build_new_contact_runs(left_binary, right_binary)
    cache = E12FrameCache(config.image_dir, config=config)
    reserve_gpu_workspace(config)
    preload_gpu_gray_cache(cache, config)
    contact_config = ContactExtractionConfig(
        use_object_mask=True,
        use_object_boundary=True,
        boundary_distance_px=6.0,
        project_points_to_object_boundary=True,
    )

    all_records: List[Dict[str, Any]] = []
    plan_rows: List[Dict[str, Any]] = []
    print(f"Running {EXPERIMENT['experiment_id']}: {EXPERIMENT['description']}")
    print(f"Total frames: {len(detections)}")
    print(f"Selected subactions: {len(subactions)}")
    if not _GPU_FLOW_STATUS_PRINTED:
        print(f"E12 optical-flow backend: {describe_gpu_flow_backend(config)}", flush=True)
        _GPU_FLOW_STATUS_PRINTED = True

    total_subactions = len(subactions)
    for progress_idx, (idx, row) in enumerate(subactions.iterrows(), start=1):
        candidates, decision, reason, raw_count = build_candidates_for_subaction(
            row,
            all_runs,
            left_binary,
            right_binary,
            len(detections),
            config,
        )
        plan_rows.append(
            {
                "experiment_id": EXPERIMENT["experiment_id"],
                "subaction_index": int(idx),
                "narration_id": row["narration_id"],
                "narration": row["narration"],
                "verb": row["verb"],
                "noun": row["noun"],
                "raw_candidate_count": int(raw_count),
                "deep_run_candidate_total": int(len(candidates)),
                "deep_run_candidate_count": int(len(candidates)),
                "episode_decision": decision,
                "episode_reason": reason,
                "candidate_frames_0_based": ";".join(str(c["frame"]) for c in candidates),
                "candidate_sources": ";".join(str(c["candidate_source"]) for c in candidates),
            }
        )
        pct = progress_idx / max(1, total_subactions) * 100.0
        bar_width = 28
        filled = int(round(bar_width * progress_idx / max(1, total_subactions)))
        bar = "#" * filled + "-" * (bar_width - filled)
        print(
            f"[{bar}] {progress_idx:03d}/{total_subactions:03d} ({pct:5.1f}%) "
            f"E12 subaction {idx:02d} {row['narration_id']} | {row['narration']} | "
            f"raw={raw_count} | deep_run={len(candidates)} | {decision}",
            flush=True,
        )
        if not candidates:
            all_records.append(empty_subaction_candidate(row, reason, raw_count))
            continue
        for candidate_index, candidate in enumerate(candidates):
            cand_pct = (candidate_index + 1) / max(1, len(candidates)) * 100.0
            cand_bar_width = 18
            cand_filled = int(round(cand_bar_width * (candidate_index + 1) / max(1, len(candidates))))
            cand_bar = "#" * cand_filled + "-" * (cand_bar_width - cand_filled)
            print(
                f"    candidate [{cand_bar}] {candidate_index + 1:02d}/{len(candidates):02d} "
                f"({cand_pct:5.1f}%) subaction={idx:02d} frame={int(candidate['frame']):06d} "
                f"hand={candidate['hand']} source={candidate['candidate_source']} start",
                flush=True,
            )
            rec = run_one_candidate(
                row=row,
                candidate=candidate,
                candidate_index=candidate_index,
                detections=detections,
                cache=cache,
                contact_config=contact_config,
                config=config,
                raw_candidate_count=raw_count,
                tried_candidate_count=len(candidates),
            )
            all_records.append(rec)
            print(
                f"    candidate result subaction={idx:02d} candidate={candidate_index + 1:02d}/{len(candidates):02d} "
                f"status={rec.get('status')} fail_reason={rec.get('fail_reason')} "
                f"track_len={rec.get('object_track_length')} ref={rec.get('ref_idx')} "
                f"score={rec.get('sample_score')}",
                flush=True,
            )
        keep_so_far = sum(1 for rec in all_records if rec.get("status") == "keep")
        print(
            f"    progress summary: candidates_done={len(all_records)} "
            f"candidate_keep_so_far={keep_so_far}",
            flush=True,
        )

    candidate_df = pd.DataFrame(all_records)
    candidate_df = finalize_one_sample_per_subaction(candidate_df, config=config)
    plan_df = pd.DataFrame(plan_rows)
    subaction_df = summarize_subactions(candidate_df, subactions, plan_df)
    overview_df = build_overview(candidate_df, subaction_df, plan_df)
    final_df = save_success_manifest(candidate_df, config.output_root)

    overview_df.to_csv(config.output_root / "experiment_overview.csv", index=False)
    subaction_df.to_csv(config.output_root / "subaction_summary.csv", index=False)
    candidate_df.to_csv(config.output_root / "candidate_diagnostics.csv", index=False)
    plan_df.to_csv(config.output_root / "candidate_plan.csv", index=False)
    (config.output_root / "experiment_results.json").write_text(
        json.dumps(
            _json_safe(
                {
                    "experiment": EXPERIMENT,
                    "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()},
                    "frame_coverage": frame_coverage,
                    "overview": overview_df.to_dict(orient="records"),
                    "subaction_summary": subaction_df.to_dict(orient="records"),
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    save_funnel_chart(overview_df, charts_dir / "e12_pipeline_funnel.png")
    save_failure_chart(candidate_df, charts_dir / "e12_failure_reasons.png")
    save_track_confidence_hist(candidate_df, charts_dir / "e12_track_confidence_hist.png")
    save_projection_source_counts(candidate_df, charts_dir / "e12_projection_source_counts.png")
    save_clean_class_counts(final_df, charts_dir / "e12_reference_clean_class_counts.png")
    save_success_by_subaction(subaction_df, charts_dir / "e12_success_by_subaction.png")
    make_contact_sheet(final_df, charts_dir / "e12_mask_projection_contact_sheet.png", "E12 mask projection samples", page_size=1000)
    contact_sheet_pages = make_contact_sheet(final_df, charts_dir / "e12_success_contact_sheet_page_001.png", "E12 final success samples", page_size=16)
    summary_path = write_summary(
        config=config,
        overview_df=overview_df,
        subaction_df=subaction_df,
        candidate_df=candidate_df,
        final_df=final_df,
        contact_sheet_pages=contact_sheet_pages,
    )

    print("\nExperiment overview")
    print(overview_df)
    print(f"Summary: {summary_path}")
    print(f"Candidate diagnostics CSV: {config.output_root / 'candidate_diagnostics.csv'}")
    print(f"Contact sheet pages: {[str(p) for p in contact_sheet_pages]}")
    print(f"Optical-flow runtime stats: {dict(FLOW_RUNTIME_STATS)}")
    return {
        "config": config,
        "overview": overview_df,
        "subaction_summary": subaction_df,
        "candidate_diagnostics": candidate_df,
        "candidate_plan": plan_df,
        "output_root": config.output_root,
        "summary_path": summary_path,
    }


if __name__ == "__main__":
    run_e12_experiment()
