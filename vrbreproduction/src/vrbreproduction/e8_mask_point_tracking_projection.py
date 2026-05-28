"""E8 experiment: SAM2 mask + point tracking + mask-gated projection.

This script keeps E7's candidate / reference-frame search structure, but
replaces the projection core with object-mask-guided point sampling, point
tracking from contact frame back to reference frame, and a final mask gate on
the reference frame.
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
    "experiment_id": "E8",
    "name": "E8_sam2_cotracker_mask_point_projection",
    "description": (
        "使用 SAM2/可见物体 mask 获取 contact/reference object mask，"
        "使用 CoTracker 接口或当前环境 fallback LK 把物体表面点从接触帧跟踪回参考帧，"
        "再用 weighted affine 投影 contact GMM，并以 reference object mask 做最终门控。"
    ),
}


@dataclass(frozen=True)
class E8Config:
    video_id: str = "P01_109"
    num_subactions: int = 100
    max_ref_backtrack: int = 360
    early_candidate_offsets: Tuple[int, ...] = (0, 1, 2)
    max_candidates_per_subaction: int = 12
    object_score_threshold: float = 0.5
    hand_score_threshold: float = 0.5
    min_contact_points: int = 5
    gmm_components: int = 5
    contact_inside_tolerance_px: float = 16.0
    crop_size: int = 150
    reference_hand_overlap_max: float = 0.08
    min_track_confidence: float = 0.18
    sam2_enabled: bool = os.environ.get("E8_SAM2_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
    sam2_model_id: str = os.environ.get("E8_SAM2_MODEL_ID", "facebook/sam2-hiera-tiny")
    sam2_device: str = os.environ.get("E8_SAM2_DEVICE", "cuda")
    sam2_multimask_output: bool = os.environ.get("E8_SAM2_MULTIMASK", "0").strip().lower() in {"1", "true", "yes", "on"}
    use_visor_mask: bool = os.environ.get("E8_USE_VISOR_MASK", "1").strip().lower() not in {"0", "false", "no", "off"}
    cotracker_enabled: bool = os.environ.get("E8_COTRACKER_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
    cotracker_repo: str = os.environ.get("E8_COTRACKER_REPO", "facebookresearch/co-tracker")
    cotracker_model: str = os.environ.get("E8_COTRACKER_MODEL", "cotracker3_online")
    min_tracking_points: int = int(os.environ.get("E8_MIN_TRACKING_POINTS", "20"))
    max_tracking_points: int = int(os.environ.get("E8_MAX_TRACKING_POINTS", "36"))
    hand_exclusion_pad_px: int = int(os.environ.get("E8_HAND_EXCLUSION_PAD_PX", "6"))
    point_boundary_band_px: int = int(os.environ.get("E8_POINT_BOUNDARY_BAND_PX", "10"))
    point_contact_band_px: int = int(os.environ.get("E8_POINT_CONTACT_BAND_PX", "28"))
    track_fb_max: float = float(os.environ.get("E8_TRACK_FB_MAX", "3.0"))
    track_min_visible_points: int = int(os.environ.get("E8_TRACK_MIN_VISIBLE_POINTS", "8"))
    affine_ransac_reproj_threshold: float = float(os.environ.get("E8_AFFINE_RANSAC_REPROJ_THRESHOLD", "4.0"))
    affine_min_inliers: int = int(os.environ.get("E8_AFFINE_MIN_INLIERS", "6"))
    affine_min_inlier_ratio: float = float(os.environ.get("E8_AFFINE_MIN_INLIER_RATIO", "0.35"))
    affine_max_median_residual: float = float(os.environ.get("E8_AFFINE_MAX_MEDIAN_RESIDUAL", "12.0"))
    mask_gate_min_inside_count: int = int(os.environ.get("E8_MASK_GATE_MIN_INSIDE_COUNT", "3"))
    mask_gate_boundary_tolerance_px: float = float(os.environ.get("E8_MASK_GATE_BOUNDARY_TOLERANCE_PX", "14.0"))
    mask_gate_centroid_max_distance_px: float = float(os.environ.get("E8_MASK_GATE_CENTROID_MAX_DISTANCE_PX", "18.0"))
    mask_gate_min_visible_points: int = int(os.environ.get("E8_MASK_GATE_MIN_VISIBLE_POINTS", "8"))
    mask_gate_min_inlier_ratio: float = float(os.environ.get("E8_MASK_GATE_MIN_INLIER_RATIO", "0.35"))
    final_quality_keep_threshold: float = float(os.environ.get("E8_FINAL_QUALITY_KEEP_THRESHOLD", "0.34"))
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
    output_root: Path = VRBREPRODUCTION_ROOT / "outputs" / "100subaction-e8"
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


def get_subactions(config: E8Config) -> pd.DataFrame:
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
    config: E8Config,
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


class E8FrameCache:
    def __init__(self, image_dir: Path, config: Optional[E8Config] = None):
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

    def load_gray_torch(self, frame_idx: int, config: E8Config):
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


def _ensure_sam2_predictor(config: E8Config):
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


def _ensure_cotracker_model(config: E8Config):
    global _COTRACKER_MODEL, _COTRACKER_INIT_ERROR
    if not config.cotracker_enabled:
        return None
    if _COTRACKER_MODEL is not None:
        return _COTRACKER_MODEL
    if _COTRACKER_INIT_ERROR is not None:
        return None
    try:
        torch_mod = get_torch_for_gpu_flow(config)
        if torch_mod is None:
            torch_mod = _TORCH_MODULE
        if torch_mod is None:
            import torch as torch_mod  # type: ignore
        _COTRACKER_MODEL = torch_mod.hub.load(config.cotracker_repo, config.cotracker_model, trust_repo=True)
        if hasattr(_COTRACKER_MODEL, "eval"):
            _COTRACKER_MODEL.eval()
        return _COTRACKER_MODEL
    except Exception as exc:
        _COTRACKER_INIT_ERROR = f"{type(exc).__name__}:{exc}"
        return None


def _load_visor_mask(frame_idx: int, image_bgr: np.ndarray, active_object_bbox: Sequence[float], config: E8Config) -> Dict[str, Any]:
    h, w = image_bgr.shape[:2]
    candidates: List[Path] = []
    visor_root = os.environ.get("E8_VISOR_MASK_ROOT")
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


def get_object_mask(frame_idx: int, image_bgr: np.ndarray, active_object_bbox: Sequence[float], noun: Any, config: E8Config) -> Dict[str, Any]:
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
    config: E8Config,
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


def track_object_points_to_reference(
    frame_indices: Sequence[int],
    points_xy: np.ndarray,
    config: E8Config,
    cache: E8FrameCache,
) -> Dict[str, Any]:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    frame_indices = [int(i) for i in frame_indices]
    if len(pts) == 0 or len(frame_indices) < 2:
        return {
            "passed": False,
            "reason": "insufficient_tracking_setup",
            "backend": "fallback_lk",
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
            frames = [cache.load_bgr(idx) for idx in chronological_indices]
            video = torch_mod.from_numpy(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], axis=0)).permute(0, 3, 1, 2)[None].float()
            device = getattr(model, "device", None) or torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
            if hasattr(video, "to"):
                video = video.to(device)
            queries = torch_mod.as_tensor(pts[None, :, :], dtype=torch.float32, device=device)
            query_frame = len(frames) - 1
            output = None
            try:
                output = model(video, queries=queries, grid_query_frame=query_frame, backward_tracking=True)
            except Exception:
                try:
                    output = model(video_chunk=video, is_first_step=True, queries=queries, grid_query_frame=query_frame, backward_tracking=True)
                except Exception:
                    output = None
            if output is not None:
                if isinstance(output, tuple) and len(output) >= 2:
                    pred_tracks, pred_visibility = output[:2]
                    pred_tracks = np.asarray(pred_tracks)
                    pred_visibility = np.asarray(pred_visibility)
                    if pred_tracks.ndim == 4:
                        pred_tracks = pred_tracks[0]
                    if pred_visibility.ndim == 4:
                        pred_visibility = pred_visibility[0]
                    if pred_tracks.ndim == 3:
                        ref_points = pred_tracks[0]
                    elif pred_tracks.ndim == 2:
                        ref_points = pred_tracks
                    else:
                        ref_points = None
                    if ref_points is not None and len(ref_points) == len(pts):
                        vis = pred_visibility[0] if pred_visibility.ndim >= 3 else pred_visibility
                        vis = np.asarray(vis).reshape(-1).astype(bool)
                        if vis.shape[0] == len(pts):
                            backend = "cotracker"
                            confidence = np.clip(np.where(vis, 1.0, 0.0), 0.0, 1.0).astype(np.float32)
                            return {
                                "passed": int(vis.sum()) >= int(config.track_min_visible_points),
                                "reason": None if int(vis.sum()) >= int(config.track_min_visible_points) else "track_visibility_below_threshold",
                                "backend": backend,
                                "ref_points_xy": np.asarray(ref_points, dtype=np.float32).reshape(-1, 2),
                                "visibility": vis,
                                "confidence": confidence,
                                "track_point_count": int(len(pts)),
                                "visible_point_count": int(vis.sum()),
                                "median_fb_error": None,
                            }
        except Exception:
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
    config: E8Config,
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
        "projection_method": "cotracker_weighted_affine",
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
    config: E8Config,
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
    config: E8Config,
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
    best_score = -1e9
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
    config: E8Config,
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
    config: E8Config,
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
    config: E8Config,
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


def get_torch_for_gpu_flow(config: E8Config):
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


def describe_gpu_flow_backend(config: E8Config) -> str:
    torch = get_torch_for_gpu_flow(config)
    if torch is None:
        if not config.gpu_flow_enabled:
            return "disabled_by_E7_GPU_FLOW"
        return f"cpu_opencv_fallback({_TORCH_IMPORT_ERROR or 'cuda_unavailable'})"
    device_name = torch.cuda.get_device_name(0) if config.gpu_flow_device.startswith("cuda") else config.gpu_flow_device
    return f"torch_patch_cuda(device={config.gpu_flow_device}, name={device_name})"


def reserve_gpu_workspace(config: E8Config) -> None:
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


def preload_gpu_gray_cache(cache: E8FrameCache, config: E8Config) -> None:
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
    config: E8Config,
    cache: Optional[E8FrameCache] = None,
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
    config: Optional[E8Config] = None,
    cache: Optional[E8FrameCache] = None,
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
    cache: E8FrameCache,
    frame_idx: int,
    active_hand: str,
    contact_points: np.ndarray,
    config: E8Config,
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
    best_score = -1e9
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
    cache: E8FrameCache,
    frame_idx: int,
    active_hand: str,
    object_bbox_norm: Sequence[float],
    contact_config: ContactExtractionConfig,
    config: E8Config,
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
    cache: E8FrameCache,
    contact_frame: int,
    seed_bbox_px: Sequence[float],
    min_frame: int,
    config: E8Config,
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
    cache: E8FrameCache,
    contact_frame: int,
    ref_idx: int,
    contact_means: np.ndarray,
    contact_covariances: Optional[np.ndarray],
    seed_bbox_px: Sequence[float],
    ref_bbox_px: Sequence[float],
    object_track: Dict[int, Dict[str, Any]],
    active_hand: str,
    config: E8Config,
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
    cache: E8FrameCache,
    object_track: Dict[int, Dict[str, Any]],
    contact_frame: int,
    seed_bbox_px: Sequence[float],
    contact_means: np.ndarray,
    contact_covariances: Optional[np.ndarray],
    trajectory_pts: Sequence[np.ndarray],
    tracking_points: np.ndarray,
    active_hand: str,
    hand_bbox_px: Optional[Sequence[float]],
    noun: Any,
    all_nouns: Any,
    narration: Any,
    search_start: int,
    search_end: int,
    config: E8Config,
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
        )
        if projected_detail.get("passed"):
            projected = np.asarray(projected_detail["projected_means"], dtype=np.float32)
            projected_covariances = projected_detail.get("projected_covariances")
            projection_method = str(projected_detail.get("projection_method") or "cotracker_weighted_affine")
        else:
            projected = bbox_relative_project(contact_means, seed_bbox_px, ref_bbox)
            projected_covariances = bbox_relative_covariance_project(contact_covariances, seed_bbox_px, ref_bbox)
            projection_method = "bbox_relative_fallback"

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
        gate = mask_based_projection_gate(
            projected_contact_means=projected,
            reference_object_mask=reference_object_mask,
            visibility=track_detail["visibility"],
            affine_inlier_ratio=float(projected_detail.get("affine_inlier_ratio") or 0.0),
            config=config,
        )
        if not gate["passed"]:
            best_reject[str(gate["reason"] or "mask_gate_failed")] += 1
            continue

        centroid = projected.mean(axis=0)
        crop_bbox = compute_centered_crop_bbox(centroid, ref_img.shape, crop_size=config.crop_size)
        hand_overlap = compute_crop_hand_overlap_ratio(detections[ref_idx], ref_img.shape, crop_bbox, score_threshold=config.hand_score_threshold)
        tau_ref = project_trajectory_bbox_relative(trajectory_pts, seed_bbox_px, ref_bbox)
        if not tau_ref:
            best_reject["missing_trajectory_points"] += 1
            continue
        if any(pt[0] < 0 or pt[1] < 0 or pt[0] > w_ref or pt[1] > h_ref for pt in tau_ref):
            best_reject["trajectory_out_of_bounds"] += 1
            continue
        track_conf = float(object_track[ref_idx].get("track_confidence", 0.0))
        ref_gap = max(0, int(contact_frame) - int(ref_idx))
        score = (
            2.1 * track_conf
            + 2.0 * float(gate["final_quality_score"])
            + 0.25 * int(gate["projected_contact_inside_mask_count"])
            - 0.01 * float(gate["projected_contact_to_mask_distance"])
            - 1.5 * hand_overlap
            - float(config.reference_gap_penalty) * ref_gap
        )
        if projection_method == "cotracker_weighted_affine":
            score += 0.35
            if projected_detail.get("affine_inlier_ratio") is not None:
                score += 0.35 * float(projected_detail["affine_inlier_ratio"])
            if projected_detail.get("affine_median_residual") is not None:
                score -= 0.03 * float(projected_detail["affine_median_residual"])
        current = {
            "passed": True,
            "ref_idx": int(ref_idx),
            "ref_selection_reason": ref_reason,
            "projection_source": projection_method,
            "projection_method": projection_method,
            "projection_status": "success",
            "mask_source_contact": contact_mask_source,
            "mask_source_reference": ref_object_mask_info.get("mask_source"),
            "tracker_backend": str(track_detail.get("backend") or "fallback_lk"),
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
            "reference_object_mask_source": ref_object_mask_info.get("mask_source"),
            "reference_object_mask_backend": ref_object_mask_info.get("backend"),
            "projection_quality": float(gate["final_quality_score"]),
            "final_quality_score": float(gate["final_quality_score"]),
            "reference_hand_overlap_ratio": float(hand_overlap),
            "reference_gap": int(ref_gap),
            "trajectory_projected": tau_ref,
            "crop_bbox": tuple(int(v) for v in crop_bbox),
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
    cache: E8FrameCache,
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
    heatmap_mode = "covariance_e8_object_aware" if covariances is not None else "isotropic_e7"
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
    cache: E8FrameCache,
    contact_config: ContactExtractionConfig,
    config: E8Config,
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
        tracking_pack = sample_object_points_for_tracking(
            object_mask=contact_object_mask,
            contact_means=cell2["contact_means"],
            hand_bbox=hand_norm if hand_norm is not None else None,
            config=config,
        )
        tracking_points = tracking_pack["points_xy"]
        record["track_point_count"] = int(tracking_pack["count"])
        if len(tracking_points) < int(config.min_tracking_points):
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
            trajectory_status="success",
            sample_score=float(ref_eval["sample_score"]),
            tracked_reference_object_bbox=[round(float(v), 2) for v in ref_bbox],
            projected_contact_means=np.asarray(ref_eval["projected_contact_means"], dtype=np.float32),
            projected_contact_covariances=ref_eval.get("projected_contact_covariances"),
            trajectory_projected=ref_eval["trajectory_projected"],
            reference_object_mask=ref_eval.get("reference_object_mask"),
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


def finalize_one_sample_per_subaction(candidate_df: pd.DataFrame, config: Optional[E8Config] = None) -> pd.DataFrame:
    df = candidate_df.copy()
    config = config or E8Config()
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
                "cotracker_success": cotracker_success,
                "fallback_tracker_success": fallback_success,
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
    ax.set_title("E8 pipeline funnel")
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
    ax.set_title("E8 failure reasons")
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
    ax.set_title("E8 track confidence")
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
    ax.set_title("E8 projection source counts")
    ax.set_xlabel("projection_source")
    ax.set_ylabel("success candidate count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_success_by_subaction(subaction_df: pd.DataFrame, path: Path) -> None:
    y = (subaction_df["final_status"] == "keep").astype(int)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(subaction_df["subaction_index"], y, color=["#188038" if v else "#dadce0" for v in y])
    ax.set_title("E8 success by subaction")
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
                f"{row.get('verb')} {row.get('noun')} | {row.get('projection_source')}"
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
    config: E8Config,
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
    row = overview_df.iloc[0]
    compare_lines = ["| 实验 | 覆盖 subaction（子动作片段） | tuple（训练样本） | 备注 |", "| --- | ---: | ---: | --- |"]
    if e7:
        compare_lines.append(f"| E7 | {int(e7.get('subactions_success', 0))} | {int(e7.get('final_tuple_count', 0))} | object tracklet + local flow + bbox gate baseline |")
    if e6b:
        compare_lines.append(f"| E6b | {int(e6b.get('subactions_success', 0))} | {int(e6b.get('heatmap_pass', 0))} | crop-level humanless fallback（局部无手补救）但无最终 object-centric gate（物体中心门控） |")
    if e6c:
        compare_lines.append(f"| E6c conservative | {int(e6c.get('subactions_success', e6c.get('e6c_conservative_final_subactions', 0)))} | {int(e6c.get('heatmap_pass', e6c.get('e6c_conservative_final_success', 0)))} | reference-frame object consistency（参考帧物体一致性）保守口径 |")
    compare_lines.append(
        f"| E8 | {int(row['subactions_success'])} | {int(row['final_tuple_count'])} | "
        f"SAM2 mask + tracked points + mask gate；每个 subaction 最多 {int(config.final_max_tuples_per_subaction)} 个、帧间隔至少 {int(config.final_frame_dedup_gap)} 帧 |"
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
    lines = [
        "# E8 SAM2 + CoTracker mask-point tracking 实验报告",
        "",
        f"- 输出目录：`{out}`",
        f"- video_id：`{config.video_id}`",
        f"- subaction（子动作片段）总数：`{int(row['subactions_total'])}`",
        f"- 成功覆盖 subaction（子动作片段）：`{int(row['subactions_success'])}`",
        f"- final tuple（最终训练样本）：`{int(row['final_tuple_count'])}`",
        f"- mask gate 通过数：`{int(row.get('mask_gate_pass', 0))}`",
        f"- CoTracker 成功数：`{int(row.get('cotracker_success', 0))}`",
        f"- fallback tracker 成功数：`{int(row.get('fallback_tracker_success', 0))}`",
        f"- fallback tracker 比例：`{(float(row.get('fallback_tracker_success', 0)) / max(1, int(row.get('fallback_tracker_success', 0)) + int(row.get('cotracker_success', 0)))):.2%}`",
        f"- articulated object 成功数：`{int(row.get('articulated_success_count', 0))}`",
        f"- articulated object 失败数：`{int(row.get('articulated_failure_count', 0))}`",
        "",
        "## 算法改动",
        "",
        "- contact frame 先通过 VISOR / SAM2 / grabcut fallback 得到目标物 mask，再从 mask 内优先采样 contact-near / boundary 点，显式排除手部附近点。",
        "- 用 CoTracker 接口做点跟踪；当前环境未安装 CoTracker 时，tracker_backend 会明确记为 `fallback_lk`，不冒充 CoTracker。",
        "- 用 tracked points 在参考帧拟合 weighted affine，再把 contact GMM means 投到 reference frame。",
        "- 最终 gate 不再看 bbox 一致性，而是看 projected contact 是否真正落在 reference object mask 上，或者是否足够贴近 mask boundary。",
        f"- 最终导出允许每个 subaction（子动作片段）最多保留 `{int(config.final_max_tuples_per_subaction)}` 个高分 tuple（训练样本），并做 `{int(config.final_frame_dedup_gap)}` 帧的近邻去重。",
        "",
        "## 与 E6b / E6c 的关键区别",
        "",
        "\n".join(compare_lines),
        "",
        "E8 的主变化不是再调几何阈值，而是把接触点投影改成物体 mask 内点的点跟踪 + mask gate。E7 的主要失败点里，reference_crop_hand_overlap_high、missing_hand_or_object_bbox 和 projected_contact_not_on_tracked_object 会被这个版本直接对冲。",
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
        f"- mask_gate_pass（mask gate 通过）：`{int(row.get('mask_gate_pass', 0))}`",
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
        f"E8 当前覆盖 `{int(row['subactions_success'])}/100` 个 subaction（子动作片段），距离 80 个 subaction 目标还差 `{max(0, 80 - int(row['subactions_success']))}` 个。`final_tuple_count` 现在反映 top-k 导出口径，不再等同于 subaction 覆盖数。",
        "",
        "## 质量风险",
        "",
        "- 当前环境里 CoTracker 未安装，正在用 `fallback_lk` 维持可跑通性；这会影响最终覆盖率和几何稳定性。",
        "- SAM2 也可能因 checkpoint 下载失败而退回 grabcut fallback；这时 mask 质量会明显低于目标方案。",
        "- articulated object 的参考 mask 更容易变形或遮挡，仍可能出现误投。",
        "",
        "## Contact Sheet",
        "",
        "\n".join(sheet_lines),
        "",
        "## 图表",
        "",
        "- `charts/e8_pipeline_funnel.png`",
        "- `charts/e8_failure_reasons.png`",
        "- `charts/e8_mask_projection_contact_sheet.png`",
        "- `charts/e8_success_contact_sheet_page_001.png`",
        "- `charts/e8_projection_source_counts.png`",
    ]
    path = out / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_e8_experiment(config: Optional[E8Config] = None) -> Dict[str, Any]:
    global _GPU_FLOW_STATUS_PRINTED
    config = config or E8Config()
    config.output_root.mkdir(parents=True, exist_ok=True)
    charts_dir = config.output_root / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    (VRBREPRODUCTION_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)

    detections = load_detections(str(config.hoa_pkl))
    subactions = get_subactions(config)
    frame_coverage = compute_subaction_frame_coverage(subactions)
    left_binary, right_binary = build_contact_arrays(detections)
    all_runs = build_new_contact_runs(left_binary, right_binary)
    cache = E8FrameCache(config.image_dir, config=config)
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
        print(f"E8 optical-flow backend: {describe_gpu_flow_backend(config)}", flush=True)
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
            f"E8 subaction {idx:02d} {row['narration_id']} | {row['narration']} | "
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

    save_funnel_chart(overview_df, charts_dir / "e8_pipeline_funnel.png")
    save_failure_chart(candidate_df, charts_dir / "e8_failure_reasons.png")
    save_track_confidence_hist(candidate_df, charts_dir / "e8_track_confidence_hist.png")
    save_projection_source_counts(candidate_df, charts_dir / "e8_projection_source_counts.png")
    save_success_by_subaction(subaction_df, charts_dir / "e8_success_by_subaction.png")
    make_contact_sheet(final_df, charts_dir / "e8_mask_projection_contact_sheet.png", "E8 mask projection samples", page_size=1000)
    contact_sheet_pages = make_contact_sheet(final_df, charts_dir / "e8_success_contact_sheet_page_001.png", "E8 final success samples", page_size=16)
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
    run_e8_experiment()
