"""Shared helpers for E20 export, GPU worker, and finalization."""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


E18B_ANCHOR = "outputs/e18b/"


def fix_e18b_path(stale_path: str, repo_root: Path) -> Path:
    """Reconnect a stale absolute E18b path to the current repository root."""
    text = str(stale_path)
    if E18B_ANCHOR not in text:
        raise ValueError(f"path does not contain {E18B_ANCHOR!r}: {stale_path}")
    suffix = text.split(E18B_ANCHOR, 1)[1]
    return Path(repo_root) / "outputs" / "e18b" / suffix


def sample_key_from_rel_dir(rel_dir: str) -> str:
    parts = Path(str(rel_dir)).parts
    if len(parts) < 3:
        raise ValueError(f"sample rel_dir must have at least three parts: {rel_dir}")
    return "__".join(parts[-3:])


def rel_dir_from_sample_key(sample_key: str) -> str:
    parts = str(sample_key).split("__")
    if len(parts) != 3 or any(not p for p in parts):
        raise ValueError(f"invalid sample_key: {sample_key}")
    return str(Path(*parts))


def _bbox_norm_to_px(bbox: Any, width: int, height: int) -> list[float]:
    vals = [bbox.left, bbox.top, bbox.right, bbox.bottom]
    x1 = float(vals[0]) * width
    y1 = float(vals[1]) * height
    x2 = float(vals[2]) * width
    y2 = float(vals[3]) * height
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [
        float(np.clip(x1, 0, width)),
        float(np.clip(y1, 0, height)),
        float(np.clip(x2, 0, width)),
        float(np.clip(y2, 0, height)),
    ]


def collect_hand_bboxes_px(frame_det: Any, image_hw: Sequence[int], score_threshold: float = 0.5) -> list[list[float]]:
    """Collect all score-qualified left/right hand boxes in pixel coordinates."""
    height, width = int(image_hw[0]), int(image_hw[1])
    out: list[list[float]] = []
    for hand in getattr(frame_det, "hands", []) or []:
        if float(getattr(hand, "score", 0.0)) < float(score_threshold):
            continue
        bbox = _bbox_norm_to_px(hand.bbox, width, height)
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            out.append(bbox)
    return out


def collect_active_hand_bbox_px(
    frame_det: Any,
    active_hand: str,
    image_hw: Sequence[int],
    score_threshold: float = 0.5,
) -> list[float] | None:
    height, width = int(image_hw[0]), int(image_hw[1])
    active = str(active_hand).lower()
    for hand in getattr(frame_det, "hands", []) or []:
        if float(getattr(hand, "score", 0.0)) < float(score_threshold):
            continue
        side = getattr(getattr(hand, "side", None), "name", "")
        if str(side).lower() != active:
            continue
        bbox = _bbox_norm_to_px(hand.bbox, width, height)
        return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None
    return None


def extend_mask_to_border(
    mask_u8: np.ndarray,
    bbox_px: Sequence[float],
    image_hw: Sequence[int],
    max_extend_frac: float = 0.45,
    width_scale: float = 1.2,
) -> np.ndarray:
    """Extend a hand mask downward with a trapezoid when a first-person arm likely exits the frame."""
    height, width = int(image_hw[0]), int(image_hw[1])
    mask = ((mask_u8 > 0).astype(np.uint8) * 255).copy()
    if not bbox_px or len(bbox_px) != 4:
        return mask
    x1, y1, x2, y2 = [float(v) for v in bbox_px]
    if y2 >= height - 1:
        return mask
    gap_frac = float(height - y2) / max(1.0, float(height))
    if gap_frac > float(max_extend_frac):
        return mask

    cx = (x1 + x2) / 2.0
    top_w = max(1.0, (x2 - x1) * float(width_scale))
    bot_w = top_w * 1.3
    top_y = int(np.clip(round(y2), 0, height - 1))
    pts = np.array(
        [
            [np.clip(cx - top_w / 2, 0, width - 1), top_y],
            [np.clip(cx + top_w / 2, 0, width - 1), top_y],
            [np.clip(cx + bot_w / 2, 0, width - 1), height - 1],
            [np.clip(cx - bot_w / 2, 0, width - 1), height - 1],
        ],
        dtype=np.int32,
    )
    extension = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(extension, pts, 255)
    return cv2.bitwise_or(mask, extension)


def dilate_mask(mask_u8: np.ndarray, pad_px: int = 12) -> np.ndarray:
    mask = ((mask_u8 > 0).astype(np.uint8) * 255)
    pad = int(max(0, pad_px))
    if pad == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
    return cv2.dilate(mask, kernel, iterations=1)


def letterbox_pad(image_bgr: np.ndarray, mask_u8: np.ndarray, target: int = 512):
    h, w = image_bgr.shape[:2]
    target = int(target)
    if h > target or w > target:
        raise ValueError(f"letterbox target {target} must be >= image shape {(h, w)}")
    top = (target - h) // 2
    bottom = target - h - top
    left = (target - w) // 2
    right = target - w - left
    image_padded = cv2.copyMakeBorder(image_bgr, top, bottom, left, right, cv2.BORDER_REPLICATE)
    mask_padded = cv2.copyMakeBorder((mask_u8 > 0).astype(np.uint8) * 255, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    meta = {"top": top, "bottom": bottom, "left": left, "right": right, "height": h, "width": w, "target": target}
    return image_padded, mask_padded, meta


def unletterbox(image: np.ndarray, meta: dict[str, int]) -> np.ndarray:
    top, left = int(meta["top"]), int(meta["left"])
    height, width = int(meta["height"]), int(meta["width"])
    return image[top : top + height, left : left + width].copy()


def composite_inside_mask(original: np.ndarray, edited: np.ndarray, mask: np.ndarray, feather_px: int = 3) -> np.ndarray:
    if original.shape != edited.shape:
        raise ValueError("original and edited must have the same shape")
    mask01 = (mask > 0).astype(np.float32)
    feather = int(max(0, feather_px))
    if feather > 0 and float(mask01.max()) > 0:
        k = 2 * feather + 1
        alpha = cv2.GaussianBlur(mask01, (k, k), 0)
        alpha = np.clip(alpha, 0.0, 1.0)
    else:
        alpha = mask01
    alpha3 = alpha[:, :, None]
    out = original.astype(np.float32) * (1.0 - alpha3) + edited.astype(np.float32) * alpha3
    outside = mask01 == 0
    out = np.clip(np.round(out), 0, 255).astype(np.uint8)
    out[outside] = original[outside]
    return out


def skin_ratio_in_mask(image_bgr: np.ndarray, mask_u8: np.ndarray, erode_px: int = 3) -> float:
    mask = (mask_u8 > 0).astype(np.uint8)
    if int(mask.sum()) == 0:
        mask = np.ones(image_bgr.shape[:2], dtype=np.uint8)
    erode = int(max(0, erode_px))
    if erode > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1))
        eroded = cv2.erode(mask, kernel, iterations=1)
        if int(eroded.sum()) > 0:
            mask = eroded
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, np.array((0, 133, 77), dtype=np.uint8), np.array((255, 173, 127), dtype=np.uint8))
    region = mask > 0
    if not bool(region.any()):
        return 0.0
    return float(np.count_nonzero(skin[region] > 0) / np.count_nonzero(region))


def heatmap_top_bbox(heatmap: np.ndarray, thresh_frac: float = 0.5) -> list[int] | None:
    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.size == 0 or float(np.max(hm)) <= 0:
        return None
    ys, xs = np.where(hm > float(np.max(hm)) * float(thresh_frac))
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def mask_overlap_ratio(mask_u8: np.ndarray, bbox_px: Sequence[float] | None) -> float:
    if bbox_px is None or len(bbox_px) != 4:
        return 0.0
    h, w = mask_u8.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_px]
    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = mask_u8[y1:y2, x1:x2] > 0
    return float(np.count_nonzero(region) / region.size)


def choose_final_backend(sample_key: str, passed: dict[str, bool], seed: int = 20260707) -> str:
    choices = [name for name in ("lama", "sd") if bool(passed.get(name))]
    if not choices:
        return "none"
    if len(choices) == 1:
        return choices[0]
    rng = np.random.default_rng(int(seed) + zlib.crc32(str(sample_key).encode("utf-8")))
    return choices[int(rng.integers(0, len(choices)))]


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def bbox_mask(image_hw: Sequence[int], bboxes: Iterable[Sequence[float]], pad_px: int = 0) -> np.ndarray:
    h, w = int(image_hw[0]), int(image_hw[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    for bbox in bboxes or []:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return dilate_mask(mask, pad_px=pad_px) if pad_px else mask


# --- E20-B additions -------------------------------------------------------
# E20-A failure analysis showed the single YCrCb range also matches wooden
# counters/floors, and hand-only masks make LaMa/SD regrow the arm from the
# unmasked skin left at the mask boundary. The helpers below build arm-aware
# masks and a stricter residual-skin QC.


def dual_skin_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Skin detector requiring both YCrCb and HSV evidence (fewer wood false positives)."""
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    skin_ycrcb = cv2.inRange(ycrcb, np.array((0, 133, 77), dtype=np.uint8), np.array((255, 173, 127), dtype=np.uint8))
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    skin_hsv = cv2.bitwise_or(
        cv2.inRange(hsv, np.array((0, 25, 60), dtype=np.uint8), np.array((25, 200, 255), dtype=np.uint8)),
        cv2.inRange(hsv, np.array((170, 25, 60), dtype=np.uint8), np.array((180, 200, 255), dtype=np.uint8)),
    )
    return cv2.bitwise_and(skin_ycrcb, skin_hsv)


def dual_skin_ratio_in_mask(image_bgr: np.ndarray, mask_u8: np.ndarray, erode_px: int = 2) -> float:
    mask = (mask_u8 > 0).astype(np.uint8)
    if int(mask.sum()) == 0:
        mask = np.ones(image_bgr.shape[:2], dtype=np.uint8)
    erode = int(max(0, erode_px))
    if erode > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1))
        eroded = cv2.erode(mask, kernel, iterations=1)
        if int(eroded.sum()) > 0:
            mask = eroded
    skin = dual_skin_mask(image_bgr)
    region = mask > 0
    if not bool(region.any()):
        return 0.0
    return float(np.count_nonzero(skin[region] > 0) / np.count_nonzero(region))


def skin_ratio_in_bbox(image_bgr: np.ndarray, bbox_px: Sequence[float]) -> float:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_px]
    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = dual_skin_mask(image_bgr[y1:y2, x1:x2])
    return float(np.count_nonzero(region) / region.size)


def fill_mask_holes(mask_u8: np.ndarray) -> np.ndarray:
    """Fill enclosed holes (e.g. a watch strap splitting hand from forearm)."""
    mask = ((mask_u8 > 0).astype(np.uint8)) * 255
    h, w = mask.shape[:2]
    flood = mask.copy()
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    for seed in ((w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if flood[seed[1], seed[0]] == 0:
            cv2.floodFill(flood, ff_mask, seed, 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def grow_mask_into_skin(
    mask_u8: np.ndarray,
    image_bgr: np.ndarray,
    bridge_px: int = 5,
    max_area_ratio: float = 0.55,
    cap_geodesic_px: int = 120,
) -> np.ndarray:
    """Union the mask with connected skin regions so an arm attached to a masked hand is fully covered.

    Growth is component-based (skin | mask); if the grown area explodes (skin-like
    background leak) it falls back to geodesic growth capped at cap_geodesic_px.
    """
    h, w = mask_u8.shape[:2]
    base = ((mask_u8 > 0).astype(np.uint8)) * 255
    if int(np.count_nonzero(base)) == 0:
        return base
    skin = dual_skin_mask(image_bgr)
    union = cv2.bitwise_or(skin, base)
    bridge = int(max(0, bridge_px))
    if bridge > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bridge + 1, 2 * bridge + 1))
        union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, kernel)
    num, labels = cv2.connectedComponents((union > 0).astype(np.uint8), connectivity=8)
    touched = np.unique(labels[base > 0])
    touched = touched[touched != 0]
    grown = np.isin(labels, touched).astype(np.uint8) * 255
    grown = cv2.bitwise_or(grown, base)
    if float(np.count_nonzero(grown) / grown.size) <= float(max_area_ratio):
        return grown
    # leak fallback: geodesic dilation of the base mask restricted to the union map
    step = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    capped = base.copy()
    for _ in range(int(max(1, cap_geodesic_px // 5))):
        expanded = cv2.dilate(capped, step, iterations=1)
        expanded = cv2.bitwise_and(expanded, union)
        expanded = cv2.bitwise_or(expanded, base)
        if np.array_equal(expanded, capped):
            break
        capped = expanded
        if float(np.count_nonzero(capped) / capped.size) > float(max_area_ratio):
            break
    return capped


def bridge_components_to_border(
    mask_u8: np.ndarray,
    image_hw: Sequence[int],
    width_scale: float = 1.4,
    max_extend_frac: float = 0.6,
) -> np.ndarray:
    """Extend every mask component toward its nearest border (arms always exit the frame).

    Unlike extend_mask_to_border this is direction-aware (bottom/left/right/top).
    """
    height, width = int(image_hw[0]), int(image_hw[1])
    mask = ((mask_u8 > 0).astype(np.uint8)) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    out = mask.copy()
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < 30:
            continue
        x2, y2 = x + bw, y + bh
        touches = x <= 0 or y <= 0 or x2 >= width or y2 >= height
        if touches:
            continue
        gaps = {
            "bottom": (height - y2) / height,
            "top": y / height,
            "left": x / width,
            "right": (width - x2) / width,
        }
        side, gap_frac = min(gaps.items(), key=lambda kv: kv[1])
        if gap_frac > float(max_extend_frac):
            continue
        if side in ("bottom", "top"):
            cx = x + bw / 2.0
            top_w = max(1.0, bw * float(width_scale))
            bot_w = top_w * 1.5
            if side == "bottom":
                y_edge, y_border = y2 - 1, height - 1
            else:
                y_edge, y_border = y, 0
            pts = np.array(
                [
                    [np.clip(cx - top_w / 2, 0, width - 1), y_edge],
                    [np.clip(cx + top_w / 2, 0, width - 1), y_edge],
                    [np.clip(cx + bot_w / 2, 0, width - 1), y_border],
                    [np.clip(cx - bot_w / 2, 0, width - 1), y_border],
                ],
                dtype=np.int32,
            )
        else:
            cy = y + bh / 2.0
            near_h = max(1.0, bh * float(width_scale))
            far_h = near_h * 1.5
            if side == "left":
                x_edge, x_border = x, 0
            else:
                x_edge, x_border = x2 - 1, width - 1
            pts = np.array(
                [
                    [x_edge, np.clip(cy - near_h / 2, 0, height - 1)],
                    [x_edge, np.clip(cy + near_h / 2, 0, height - 1)],
                    [x_border, np.clip(cy + far_h / 2, 0, height - 1)],
                    [x_border, np.clip(cy - far_h / 2, 0, height - 1)],
                ],
                dtype=np.int32,
            )
        quad = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(quad, pts, 255)
        out = cv2.bitwise_or(out, quad)
    return out


def residual_skin_components(
    edited_bgr: np.ndarray,
    mask_u8: np.ndarray,
    attach_pad_px: int = 12,
    min_area_frac: float = 0.002,
) -> np.ndarray:
    """Leftover skin regions in an inpainted image that connect to the removed area.

    Returns a mask of full skin components (e.g. a forearm that survived pass 1);
    used to expand the mask for a re-inpaint round.
    """
    h, w = edited_bgr.shape[:2]
    skin = dual_skin_mask(edited_bgr)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    attach = dilate_mask(mask_u8, pad_px=int(attach_pad_px)) > 0
    num, labels, stats, _ = cv2.connectedComponentsWithStats((skin > 0).astype(np.uint8), connectivity=8)
    out = np.zeros((h, w), dtype=np.uint8)
    min_area = float(min_area_frac) * h * w
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        comp = labels == i
        if bool(np.any(comp & attach)):
            out[comp] = 255
    return out


def laplacian_variance(image_bgr: np.ndarray, mask_u8: np.ndarray | None = None) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    if mask_u8 is not None and int(np.count_nonzero(mask_u8)) > 0:
        return float(lap[mask_u8 > 0].var())
    return float(lap.var())


def boundary_gradient_ratio(original: np.ndarray, edited: np.ndarray, mask_u8: np.ndarray, band_px: int = 4) -> float:
    """Seam visibility: gradient energy across the mask boundary, edited vs original.

    <=1 means the inpainted seam is no harsher than the original hand edge.
    Replaces E20-A boundary_delta, which was 0 by construction.
    """
    mask01 = (mask_u8 > 0).astype(np.uint8)
    if int(mask01.sum()) == 0:
        return 0.0
    band = int(max(1, band_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    ring = (cv2.dilate(mask01, kernel) > 0) & (cv2.erode(mask01, kernel) == 0)
    if not bool(ring.any()):
        return 0.0

    def _grad_energy(img: np.ndarray) -> float:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        return float(mag[ring].mean())

    orig_energy = _grad_energy(original)
    edit_energy = _grad_energy(edited)
    return float(edit_energy / max(orig_energy, 1e-6))


def merge_overlapping_boxes(boxes: Sequence[Sequence[float]], iou_threshold: float = 0.55) -> list[list[float]]:
    """Greedy dedupe: earlier boxes win, later boxes with high IoU are dropped."""
    kept: list[list[float]] = []
    for box in boxes or []:
        bx = [float(v) for v in box]
        duplicate = False
        for ref in kept:
            ix1, iy1 = max(bx[0], ref[0]), max(bx[1], ref[1])
            ix2, iy2 = min(bx[2], ref[2]), min(bx[3], ref[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            a1 = (bx[2] - bx[0]) * (bx[3] - bx[1])
            a2 = (ref[2] - ref[0]) * (ref[3] - ref[1])
            union = a1 + a2 - inter
            if union > 0 and inter / union >= float(iou_threshold):
                duplicate = True
                break
        if not duplicate:
            kept.append(bx)
    return kept


def rank_backends(qc_metrics: dict[str, dict[str, float]], passed: dict[str, bool]) -> list[str]:
    """Deterministic quality ranking replacing the E20-A seeded 50/50 pick.

    Lower residual skin wins; seam ratio breaks ties; LaMa preferred on equal footing.
    """
    order = {"lama": 0, "sd": 1}
    names = [n for n in ("lama", "sd") if n in qc_metrics]
    return sorted(
        names,
        key=lambda n: (
            0 if passed.get(n) else 1,
            round(float(qc_metrics[n].get("residual_skin_after", 1.0)), 4),
            round(float(qc_metrics[n].get("boundary_gradient_ratio", 99.0)), 3),
            order.get(n, 9),
        ),
    )

