"""
Problem 3 Utilities for VRB data processing pipeline.

This module contains helper functions for:
1. Human-less frame detection with strict 3-frame streak rule
2. Dynamic mask building for homography computation
3. Pairwise homography computation with quality gating
4. Point and polygon transformations
5. Geometric consistency checks
6. Diagnostic visualization
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
from epic_kitchens.hoa.types import HandState


def count_valid_hands(frame_det, score_threshold: float = 0.5) -> int:
    """
    Count the number of valid hand detections in a frame.

    Args:
        frame_det: Frame detection object from epic_kitchens
        score_threshold: Minimum score for a valid detection

    Returns:
        Number of hands with score >= threshold
    """
    count = 0
    if hasattr(frame_det, 'hands'):
        for hand in frame_det.hands:
            if hand.score >= score_threshold:
                count += 1
    return count


def find_strict_humanless_frame(
    detections,
    t_contact: int,
    score_threshold: float = 0.5,
    min_no_hand_streak: int = 3,
    min_frame_idx: int = 0,
) -> Tuple[Optional[int], Dict[str, Any]]:
    """
    Find a strict human-less frame by requiring consecutive no-hand frames.

    New rule (NOT the old "first frame with no hands"):
    - Search backwards from t_contact - 1
    - Only accept a frame idx as reference if:
      - idx has no valid hands
      - idx - 1 has no valid hands
      - idx - 2 has no valid hands
    - This ensures we don't pick "isolated no-hand frames" like frame 69

    Args:
        detections: List of frame detections
        t_contact: Contact frame index
        score_threshold: Hand detection score threshold
        min_no_hand_streak: Minimum consecutive no-hand frames (default 3)

    Returns:
        Tuple of (ref_idx or None, debug_dict)
        debug_dict contains:
            - 'rejected_isolated_frames': list of frames that had no hands but failed streak
            - 'streak_lengths': dict mapping frame_idx to streak length found
            - 'final_streak_start': the start of the streak that passed
    """
    rejected_isolated = []
    streak_lengths = {}

    min_frame_idx = max(0, int(min_frame_idx))
    idx = t_contact - 1
    while idx >= min_frame_idx:
        # Count consecutive no-hand frames ending at idx
        streak_start = idx
        consecutive_count = 0

        for check_idx in range(idx, min_frame_idx - 1, -1):
            hand_count = count_valid_hands(detections[check_idx], score_threshold)
            if hand_count == 0:
                consecutive_count += 1
                streak_start = check_idx
            else:
                break

        streak_lengths[idx] = consecutive_count

        if consecutive_count >= min_no_hand_streak:
            # Found a valid reference frame
            # Return the LAST frame of the streak (closest to t_contact)
            ref_idx = idx
            return ref_idx, {
                'rejected_isolated_frames': rejected_isolated,
                'streak_lengths': streak_lengths,
                'final_streak_start': streak_start,
                'final_streak_length': consecutive_count
            }
        else:
            # This frame doesn't have enough streak
            # Check if it was an isolated no-hand frame
            hand_count_at_idx = count_valid_hands(detections[idx], score_threshold)
            if hand_count_at_idx == 0:
                rejected_isolated.append(idx)

        idx -= 1

    # No valid reference frame found
    return None, {
        'rejected_isolated_frames': rejected_isolated,
        'streak_lengths': streak_lengths,
        'final_streak_start': None,
        'final_streak_length': 0
    }


def build_dynamic_mask(
    frame_det,
    image_shape: Tuple[int, int],
    hand_score_threshold: float = 0.5,
    object_score_threshold: float = 0.5,
    expand_px: int = 8
) -> np.ndarray:
    """
    Build a mask that excludes dynamic regions (hands and objects) from feature detection.

    The mask is True for STATIC background regions where features should be detected.
    Dynamic regions (hands and detected objects) are set to False.

    Args:
        frame_det: Frame detection object
        image_shape: (height, width) of the image
        hand_score_threshold: Minimum score for hand detection
        object_score_threshold: Minimum score for object detection
        expand_px: Pixels to expand around bboxes

    Returns:
        Boolean mask of shape (h, w) where True = static background
    """
    h, w = image_shape[:2]
    mask = np.ones((h, w), dtype=bool)

    # Exclude hand regions
    if hasattr(frame_det, 'hands'):
        for hand in frame_det.hands:
            if hand.score >= hand_score_threshold:
                x1 = int(hand.bbox.left * w) - expand_px
                y1 = int(hand.bbox.top * h) - expand_px
                x2 = int(hand.bbox.right * w) + expand_px
                y2 = int(hand.bbox.bottom * h) + expand_px
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                mask[y1:y2, x1:x2] = False

    # Exclude object regions
    if hasattr(frame_det, 'objects'):
        for obj in frame_det.objects:
            if obj.score >= object_score_threshold:
                x1 = int(obj.bbox.left * w) - expand_px
                y1 = int(obj.bbox.top * h) - expand_px
                x2 = int(obj.bbox.right * w) + expand_px
                y2 = int(obj.bbox.bottom * h) + expand_px
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                mask[y1:y2, x1:x2] = False

    return mask


def compute_centered_crop_bbox(
    center_xy: np.ndarray,
    image_shape: Tuple[int, int],
    crop_size: int = 150,
) -> Tuple[int, int, int, int]:
    """
    Build a fixed-size crop bbox centered on a point, shifted inside image bounds.

    Args:
        center_xy: Center point as (x, y)
        image_shape: Image shape as (height, width)
        crop_size: Desired square crop size

    Returns:
        Crop bbox as (x1, y1, x2, y2), with x2/y2 exclusive.
    """
    h, w = image_shape[:2]
    crop_size = int(crop_size)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    half = crop_size / 2.0

    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        x1 -= x2 - w
        x2 = w
    if y2 > h:
        y1 -= y2 - h
        y2 = h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    return int(x1), int(y1), int(x2), int(y2)


def compute_crop_hand_overlap_ratio(
    frame_det,
    image_shape: Tuple[int, int],
    crop_bbox: Tuple[int, int, int, int],
    score_threshold: float = 0.5,
) -> float:
    """
    Compute how much of a crop is covered by valid hand detections in that frame.

    The ratio is measured as union(hand bbox intersections with crop) / crop area,
    using a small raster mask to avoid double-counting overlapping hand boxes.
    """
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in crop_bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop_area = max(0, x2 - x1) * max(0, y2 - y1)
    if crop_area <= 0:
        return 1.0

    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    if hasattr(frame_det, 'hands'):
        for hand in frame_det.hands:
            if hand.score < score_threshold:
                continue
            hx1 = int(round(hand.bbox.left * w))
            hy1 = int(round(hand.bbox.top * h))
            hx2 = int(round(hand.bbox.right * w))
            hy2 = int(round(hand.bbox.bottom * h))
            ix1 = max(x1, hx1)
            iy1 = max(y1, hy1)
            ix2 = min(x2, hx2)
            iy2 = min(y2, hy2)
            if ix2 > ix1 and iy2 > iy1:
                mask[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] = 1

    return float(mask.sum() / crop_area)


def compute_pairwise_homography(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    prev_mask: Optional[np.ndarray] = None,
    cur_mask: Optional[np.ndarray] = None,
    nfeatures: int = 2000,
    ratio_test: float = 0.75,
    ransac_reproj_threshold: float = 5.0
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Compute pairwise homography between two consecutive frames.

    Uses ORB features with ratio test and RANSAC filtering.
    Optionally uses masks to exclude dynamic regions.

    Args:
        prev_gray: Previous frame grayscale image
        cur_gray: Current frame grayscale image
        prev_mask: Boolean mask for previous frame (True = static region)
        cur_mask: Boolean mask for current frame (True = static region)
        nfeatures: Maximum number of ORB features
        ratio_test: Lowe's ratio test threshold
        ransac_reproj_threshold: RANSAC reprojection threshold in pixels

    Returns:
        Tuple of (H_cur_to_prev or None, stats_dict)
        stats_dict contains:
            - 'good_matches': number of good matches after ratio test
            - 'inliers': number of RANSAC inliers
            - 'inlier_ratio': inliers / good_matches
            - 'H': the homography matrix (or None if failed)
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)

    # Detect and compute features
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(cur_gray, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, {
            'good_matches': 0,
            'inliers': 0,
            'inlier_ratio': 0.0,
            'H': None,
            'fail_reason': 'insufficient_features',
            'passed_quality_gate': False
        }

    # Apply masks if provided
    if prev_mask is not None:
        valid_kp1 = []
        valid_des1 = []
        for kp, des in zip(kp1, des1):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= x < prev_mask.shape[1] and 0 <= y < prev_mask.shape[0]:
                if prev_mask[y, x]:
                    valid_kp1.append(kp)
                    valid_des1.append(des)
        if len(valid_kp1) < 10:
            return None, {
                'good_matches': 0,
                'inliers': 0,
                'inlier_ratio': 0.0,
                'H': None,
                'fail_reason': 'insufficient_masked_features_prev',
                'passed_quality_gate': False
            }
        kp1, des1 = valid_kp1, np.array(valid_des1)

    if cur_mask is not None:
        valid_kp2 = []
        valid_des2 = []
        for kp, des in zip(kp2, des2):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= x < cur_mask.shape[1] and 0 <= y < cur_mask.shape[0]:
                if cur_mask[y, x]:
                    valid_kp2.append(kp)
                    valid_des2.append(des)
        if len(valid_kp2) < 10:
            return None, {
                'good_matches': 0,
                'inliers': 0,
                'inlier_ratio': 0.0,
                'H': None,
                'fail_reason': 'insufficient_masked_features_cur',
                'passed_quality_gate': False
            }
        kp2, des2 = valid_kp2, np.array(valid_des2)

    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Ratio test
    good = []
    for m in matches:
        if len(m) == 2 and m[0].distance < ratio_test * m[1].distance:
            good.append(m[0])

    if len(good) < 10:
        return None, {
            'good_matches': len(good),
            'inliers': 0,
            'inlier_ratio': 0.0,
            'H': None,
            'fail_reason': 'insufficient_good_matches',
            'passed_quality_gate': False
        }

    # Compute homography with RANSAC
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_threshold)

    if H is None:
        return None, {
            'good_matches': len(good),
            'inliers': 0,
            'inlier_ratio': 0.0,
            'H': None,
            'fail_reason': 'ransac_failed',
            'passed_quality_gate': False
        }

    inliers = int(np.sum(inlier_mask > 0))
    inlier_ratio = inliers / len(good)

    return H, {
        'good_matches': len(good),
        'inliers': inliers,
        'inlier_ratio': inlier_ratio,
        'H': H,
        'fail_reason': None,
        'passed_quality_gate': True
    }


def accumulate_homography_to_ref(
    ref_idx: int,
    target_idx: int,
    image_loader,
    detections,
    hand_score_threshold: float = 0.5,
    object_score_threshold: float = 0.5
) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]], Optional[str]]:
    """
    Accumulate pairwise homographies from target frame back to reference frame.

    Computes: H(target -> target-1) @ H(target-1 -> target-2) @ ... @ H(ref+1 -> ref)

    Args:
        ref_idx: Reference frame index (human-less frame)
        target_idx: Target frame index (to transform from)
        image_loader: Function that loads grayscale image given frame index
        detections: List of frame detections
        hand_score_threshold: Hand detection threshold for dynamic mask
        object_score_threshold: Object detection threshold for dynamic mask

    Returns:
        Tuple of (H_target_to_ref or None, pair_stats_list, fail_reason or None)
        pair_stats_list contains stats for each pairwise step
    """
    if target_idx == ref_idx:
        return np.eye(3, dtype=np.float64), [], None

    if target_idx < ref_idx:
        return None, [], 'target_idx_less_than_ref_idx'

    H_total = np.eye(3, dtype=np.float64)
    pair_stats = []

    # Accumulate from target down to ref+1, then ref
    for cur_idx in range(target_idx, ref_idx, -1):
        prev_idx = cur_idx - 1

        cur_gray = image_loader(cur_idx)
        prev_gray = image_loader(prev_idx)

        cur_mask = build_dynamic_mask(
            detections[cur_idx],
            cur_gray.shape,
            hand_score_threshold,
            object_score_threshold
        )
        prev_mask = build_dynamic_mask(
            detections[prev_idx],
            prev_gray.shape,
            hand_score_threshold,
            object_score_threshold
        )

        H_cur_to_prev, stats = compute_pairwise_homography(
            prev_gray, cur_gray,
            prev_mask, cur_mask,
            nfeatures=2000,
            ratio_test=0.75,
            ransac_reproj_threshold=5.0
        )

        stats['cur'] = cur_idx
        stats['prev'] = prev_idx
        pair_stats.append(stats)

        # Quality gating
        if H_cur_to_prev is None:
            stats['passed_quality_gate'] = False
            return None, pair_stats, f'pairwise_homography_failed_f{cur_idx}_to_f{prev_idx}'

        if stats['good_matches'] < 60:
            stats['passed_quality_gate'] = False
            return None, pair_stats, f'pairwise_homography_low_quality_f{cur_idx}_to_f{prev_idx}'
        if stats['inliers'] < 40:
            stats['passed_quality_gate'] = False
            return None, pair_stats, f'pairwise_homography_low_quality_f{cur_idx}_to_f{prev_idx}'
        if stats['inlier_ratio'] < 0.45:
            stats['passed_quality_gate'] = False
            return None, pair_stats, f'pairwise_homography_low_quality_f{cur_idx}_to_f{prev_idx}'

        stats['passed_quality_gate'] = True

        # Accumulate: H_total = H_cur_to_prev @ H_total
        H_total = H_cur_to_prev @ H_total

    return H_total, pair_stats, None


def transform_points(points: np.ndarray, H: np.ndarray) -> Optional[np.ndarray]:
    """
    Transform points using homography matrix.

    Args:
        points: Nx2 array of points
        H: 3x3 homography matrix

    Returns:
        Transformed Nx2 array or None if H is None
    """
    if H is None or len(points) == 0:
        return None

    pts = points.reshape(-1, 1, 2).astype(np.float32)
    dst = cv2.perspectiveTransform(pts, H)
    return dst.reshape(-1, 2)


def transform_bbox_to_polygon(bbox_xyxy: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Transform a bounding box to a polygon using homography.

    Args:
        bbox_xyxy: [x1, y1, x2, y2] normalized or pixel coordinates
        H: 3x3 homography matrix

    Returns:
        4x2 array of transformed corner points
    """
    x1, y1, x2, y2 = bbox_xyxy

    # 4 corners: top-left, top-right, bottom-right, bottom-left
    corners = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], dtype=np.float32)

    transformed = transform_points(corners, H)
    return transformed


def polygon_area(polygon: np.ndarray) -> float:
    """
    Calculate the area of a polygon using the shoelace formula.

    Args:
        polygon: Nx2 array of vertices in order

    Returns:
        Absolute area value
    """
    n = len(polygon)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i, 0] * polygon[j, 1]
        area -= polygon[j, 0] * polygon[i, 1]

    return abs(area) / 2.0


def count_points_near_polygon(
    points: np.ndarray,
    polygon: np.ndarray,
    tolerance_px: float = 8.0
) -> int:
    """
    Count how many points are inside or near a polygon.

    Uses cv2.pointPolygonTest with distance tolerance.

    Args:
        points: Nx2 array of points
        polygon: Mx2 array of polygon vertices
        tolerance_px: Maximum distance outside polygon to still count as "near"

    Returns:
        Number of points inside or within tolerance of polygon
    """
    count = 0
    polygon_contour = polygon.astype(np.float32)

    for pt in points:
        dist = cv2.pointPolygonTest(polygon_contour, tuple(pt), True)
        if dist >= -tolerance_px:
            count += 1

    return count


def draw_problem3_full_overlay(
    ref_img: np.ndarray,
    object_polygon: np.ndarray,
    hand_polygon: Optional[np.ndarray],
    mu_transformed: np.ndarray,
    tau_transformed: List[Optional[np.ndarray]],
    t_contact: int,
    active_hand: str,
    ref_idx: int,
    discard: bool,
    discard_reason: Optional[str],
    contact_centroid: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draw full-frame diagnostic overlay for Problem 3.

    Args:
        ref_img: Reference frame image (BGR)
        object_polygon: Transformed object bbox as 4x2 polygon
        hand_polygon: Transformed hand bbox as 4x2 polygon (or None)
        mu_transformed: Transformed contact means (5x2)
        tau_transformed: List of transformed trajectory points (may contain None)
        t_contact: Contact frame index
        active_hand: 'left' or 'right'
        ref_idx: Reference frame index
        discard: Whether sample was discarded
        discard_reason: Reason for discard (or None)
        contact_centroid: Centroid of mu_transformed (or None)

    Returns:
        BGR image with overlays
    """
    # Upscale image 3x so tiny trajectory movements become visible
    scale = 3
    overlay = cv2.resize(ref_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    h, w = overlay.shape[:2]

    def s(points):
        """Scale point coordinates."""
        return (points * scale).astype(np.int32)

    # Draw object polygon (blue)
    obj_pts = s(object_polygon).reshape((-1, 1, 2))
    cv2.polylines(overlay, [obj_pts], True, (255, 0, 0), 2)

    # Draw hand polygon (green) if available
    if hand_polygon is not None:
        hand_pts = s(hand_polygon).reshape((-1, 1, 2))
        cv2.polylines(overlay, [hand_pts], True, (0, 255, 0), 2)

    # Draw tau trajectory: polyline + arrow at end + numbered waypoints
    valid_tau = [pt for pt in tau_transformed if pt is not None]
    if len(valid_tau) >= 2:
        pts_arr = np.array([s(pt.reshape(1, -1)).flatten() for pt in valid_tau])
        cv2.polylines(overlay, [pts_arr], False, (0, 255, 0), 2)
        p_prev = tuple(pts_arr[-2])
        p_last = tuple(pts_arr[-1])
        cv2.arrowedLine(overlay, p_prev, p_last, (0, 255, 0), 2, tipLength=0.4)
        for i, pt in enumerate(pts_arr):
            cv2.circle(overlay, tuple(pt), 4, (0, 255, 255), -1)
            cv2.putText(overlay, str(i), (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Draw mu points (red circles)
    for mu in mu_transformed:
        pt = s(mu.reshape(1, -1)).flatten()
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(overlay, tuple(pt), 3, (0, 0, 255), -1)

    # Draw contact centroid (cyan)
    if contact_centroid is not None:
        ct = s(contact_centroid.reshape(1, -1)).flatten()
        if 0 <= ct[0] < w and 0 <= ct[1] < h:
            cv2.circle(overlay, tuple(ct), 5, (255, 255, 0), 1)

    # Add text info
    y_text = 30
    line_height = 25

    cv2.putText(overlay, f't_contact: {t_contact}', (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_text += line_height

    cv2.putText(overlay, f'active_hand: {active_hand}', (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_text += line_height

    cv2.putText(overlay, f'ref_idx: {ref_idx}', (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_text += line_height

    if discard:
        cv2.putText(overlay, f'DISCARD: {discard_reason}', (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(overlay, 'KEEP', (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return overlay


def draw_problem3_crop_overlay(
    ref_img: np.ndarray,
    mu_transformed: np.ndarray,
    tau_transformed: List[Optional[np.ndarray]],
    object_polygon: np.ndarray,
    crop_size: int = 150
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Draw cropped diagnostic overlay centered on contact points.

    Args:
        ref_img: Reference frame image (BGR)
        mu_transformed: Transformed contact means (5x2)
        tau_transformed: List of transformed trajectory points
        object_polygon: Transformed object bbox polygon
        crop_size: Size of crop (default 150x150)

    Returns:
        Tuple of (cropped RGB image, crop_bbox as (x1, y1, x2, y2))
    """
    h, w = ref_img.shape[:2]
    display_size = 400  # final output size
    raw_crop = 60       # larger crop to show trajectory and context

    # Center crop on mu_transformed centroid
    mu_center = mu_transformed.mean(axis=0)
    cx, cy = int(mu_center[0]), int(mu_center[1])

    half = raw_crop // 2
    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, w)
    y2 = min(cy + half, h)
    if x2 - x1 < raw_crop:
        x1 = max(x2 - raw_crop, 0)
    if y2 - y1 < raw_crop:
        y1 = max(y2 - raw_crop, 0)

    crop_img = ref_img[y1:y2, x1:x2].copy()
    crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

    # Upscale with linear interpolation for better quality at this size
    sc = display_size / max(crop_rgb.shape[0], crop_rgb.shape[1])
    overlay = cv2.resize(crop_rgb, (display_size, display_size), interpolation=cv2.INTER_LINEAR)

    def to_disp(pt):
        return (int((pt[0] - x1) * sc), int((pt[1] - y1) * sc))

    # Draw object polygon
    obj_disp = np.array([to_disp(p) for p in object_polygon], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [obj_disp], True, (255, 0, 0), 2)

    # Draw tau trajectory
    valid_tau = [pt for pt in tau_transformed if pt is not None]
    if len(valid_tau) >= 2:
        pts_disp = np.array([to_disp(pt) for pt in valid_tau], dtype=np.int32)
        cv2.polylines(overlay, [pts_disp], False, (0, 255, 0), 2)
        cv2.arrowedLine(overlay, tuple(pts_disp[-2]), tuple(pts_disp[-1]), (0, 255, 0), 2, tipLength=0.3)
        for i, pt in enumerate(pts_disp):
            cv2.circle(overlay, tuple(pt), 5, (0, 255, 255), -1)
            cv2.putText(overlay, str(i), (pt[0] + 6, pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Draw mu points
    for mu in mu_transformed:
        pt = to_disp(mu)
        cv2.circle(overlay, pt, 4, (255, 0, 0), -1)

    return overlay, (x1, y1, x2, y2)
