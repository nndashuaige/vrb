from pathlib import Path

import cv2
import numpy as np

from .problem3_utils import (
    find_strict_humanless_frame,
    accumulate_homography_to_ref,
    transform_points,
    transform_bbox_to_polygon,
    polygon_area,
    count_points_near_polygon,
    draw_problem3_full_overlay,
    draw_problem3_crop_overlay,
)
from .contact_point_utils import (
    get_active_hand_bbox,
    get_valid_object_bboxes,
    select_active_object_bbox,
)


def run_problem3_cell3(
    detections,
    t_contact,
    active_hand,
    contact_means,
    discard=False,
    discard_reason=None,
    image_dir=Path("data/P01_109_frames"),
    output_dir=Path("outputs"),
    min_ref_frame_idx=0,
):
    """
    Notebook-friendly Problem 3 runner.

    This function never raises SystemExit for normal DISCARD branches.
    Instead, it returns a result dict with:
      - discard: bool
      - discard_reason: str | None
      - status: "KEEP" | "DISCARD"
      - optional transformed geometry and output paths
    """

    result = {
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
    }

    print("=" * 80)
    print("Cell 3: Problem 3 Fixed Implementation")
    print("=" * 80)
    print(f"当前样本：frame {t_contact}, active_hand = {active_hand}")

    if result["discard"]:
        print(f"Cell 2 已将该样本标记为丢弃：{result['discard_reason']}")
        result["status"] = "DISCARD"
        print(f"样本已丢弃：{result['discard_reason']}")
        return result

    if contact_means is None:
        result["discard"] = True
        result["discard_reason"] = "missing_contact_means"
        result["status"] = "DISCARD"
        print("Cell 2 没有提取到 contact_means，无法继续")
        print(f"样本已丢弃：{result['discard_reason']}")
        return result

    print("\n" + "=" * 60)
    print("Step 1: 寻找 Strict Human-less Frame")
    print("=" * 60)

    min_no_hand_streak = 3
    ref_idx, debug_info = find_strict_humanless_frame(
        detections,
        t_contact,
        score_threshold=0.5,
        min_no_hand_streak=min_no_hand_streak,
        min_frame_idx=min_ref_frame_idx,
    )
    result["ref_idx"] = ref_idx

    print(f"min_no_hand_streak = {min_no_hand_streak}")
    print(f"搜索过程中拒绝的孤立无手帧：{debug_info['rejected_isolated_frames']}")
    print(f"最终选择的 ref_idx = {ref_idx}")

    if ref_idx is not None:
        print(
            f"最终 streak 起始帧：{debug_info['final_streak_start']}, "
            f"长度：{debug_info['final_streak_length']}"
        )
    else:
        result["discard"] = True
        result["discard_reason"] = "no_strict_humanless_frame"
        result["status"] = "DISCARD"
        print(f"样本已丢弃：{result['discard_reason']}")
        return result

    print("\n" + "=" * 60)
    print("Step 2: 读取参考帧")
    print("=" * 60)

    ref_path = image_dir / f"frame_{ref_idx + 1:010d}.jpg"
    ref_img = cv2.imread(str(ref_path))
    if ref_img is None:
        raise FileNotFoundError(f"找不到参考帧图像：{ref_path}")

    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = ref_img.shape[:2]
    print(f"参考帧路径：{ref_path}")
    print(f"参考帧尺寸：{w_img}x{h_img}")

    print("\n" + "=" * 60)
    print("Step 3: 提取轨迹点")
    print("=" * 60)

    trajectory_pixels = []
    trajectory_missing_offsets = []
    for offset in range(6):
        idx = t_contact + offset
        if idx >= len(detections):
            print(f"警告：帧 {idx} 超出范围，轨迹截断")
            break

        frame_det = detections[idx]
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

    result["trajectory_pixels"] = trajectory_pixels
    result["trajectory_missing_offsets"] = trajectory_missing_offsets

    print(f"轨迹点数：{len(trajectory_pixels)} / 6")
    print(f"缺失的轨迹 offset: {trajectory_missing_offsets if trajectory_missing_offsets else 'None'}")
    for i, pt in enumerate(trajectory_pixels):
        print(f"  t+{i}: ({pt[0]:.1f}, {pt[1]:.1f})")

    if len(trajectory_pixels) == 0:
        result["discard"] = True
        result["discard_reason"] = "missing_trajectory_points"
        result["status"] = "DISCARD"
        print(f"样本已丢弃：{result['discard_reason']}")
        return result

    print("\n" + "=" * 60)
    print("Step 4: 逐帧累计 Homography")
    print("=" * 60)

    def load_gray(frame_idx):
        img_path = image_dir / f"frame_{frame_idx + 1:010d}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"找不到图像：{img_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cumulative_H_list = []
    all_pair_stats = []
    fail_reason = None
    homography_failed = False

    for offset in range(len(trajectory_pixels)):
        target_idx = t_contact + offset
        H_target_to_ref, pair_stats, fail_reason = accumulate_homography_to_ref(
            ref_idx,
            target_idx,
            load_gray,
            detections,
            hand_score_threshold=0.5,
            object_score_threshold=0.5,
        )
        cumulative_H_list.append(H_target_to_ref)
        all_pair_stats.extend(pair_stats)
        if H_target_to_ref is None:
            print(f"  t+{offset} (frame {target_idx}): FAILED - {fail_reason}")
            homography_failed = True
        else:
            print(f"  t+{offset} (frame {target_idx}): OK")

    result["all_pair_stats"] = all_pair_stats

    print("\nPairwise Homography 质量统计:")
    seen_pairs = set()
    for stats in all_pair_stats:
        if stats.get("cur") is None:
            continue
        pair = (stats["cur"], stats["prev"])
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        status = "OK" if stats.get("passed_quality_gate", False) else "FAIL"
        print(
            f"  f{stats['cur']} -> f{stats['prev']}: "
            f"good={stats['good_matches']}, "
            f"inliers={stats['inliers']}, "
            f"ratio={stats['inlier_ratio']:.2f} [{status}]"
        )

    if homography_failed:
        result["discard"] = True
        result["discard_reason"] = fail_reason
        result["status"] = "DISCARD"
        print(f"样本已丢弃：{result['discard_reason']}")
        return result

    print("\n" + "=" * 60)
    print("Step 5: 变换接触点和轨迹到参考帧")
    print("=" * 60)

    H_contact = cumulative_H_list[0]
    result["H_contact_to_ref"] = H_contact
    mu_transformed = transform_points(contact_means.astype(np.float32), H_contact)
    if mu_transformed is None:
        result["discard"] = True
        result["discard_reason"] = "missing_transformed_contact_points"
        result["status"] = "DISCARD"
        print(f"样本已丢弃：{result['discard_reason']}")
        return result

    result["mu_transformed"] = mu_transformed

    print("变换后接触点 μ_k:")
    for i, mu in enumerate(mu_transformed):
        print(f"  μ_{i+1}: ({mu[0]:.1f}, {mu[1]:.1f})")

    contact_centroid = mu_transformed.mean(axis=0)
    result["contact_centroid"] = contact_centroid
    print(f"接触点中心：({contact_centroid[0]:.1f}, {contact_centroid[1]:.1f})")

    tau_transformed = []
    for offset, H in enumerate(cumulative_H_list):
        if H is not None:
            pt = transform_points(
                trajectory_pixels[offset].reshape(1, -1).astype(np.float32),
                H,
            )
            tau_transformed.append(pt[0])
        else:
            tau_transformed.append(None)
    result["tau_transformed"] = tau_transformed

    print("\n变换后轨迹 τ:")
    for i, pt in enumerate(tau_transformed):
        if pt is not None:
            print(f"  t+{i}: ({pt[0]:.1f}, {pt[1]:.1f})")
        else:
            print(f"  t+{i}: None")

    print("\n" + "=" * 60)
    print("Step 6: 几何一致性门控")
    print("=" * 60)

    frame_det_contact = detections[t_contact]
    active_hand_bbox_norm = get_active_hand_bbox(frame_det_contact, active_hand, score_threshold=0.5)
    object_bboxes_norm = get_valid_object_bboxes(frame_det_contact, score_threshold=0.5)
    active_object_bbox_norm = (
        select_active_object_bbox(active_hand_bbox_norm, object_bboxes_norm)
        if active_hand_bbox_norm is not None else None
    )
    active_object_bbox = None
    if active_object_bbox_norm is not None:
        active_object_bbox = np.array(
            [
                active_object_bbox_norm[0] * w_img,
                active_object_bbox_norm[1] * h_img,
                active_object_bbox_norm[2] * w_img,
                active_object_bbox_norm[3] * h_img,
            ]
        )

    if active_object_bbox is None:
        result["discard"] = True
        result["discard_reason"] = "no_active_object_bbox"
        result["status"] = "DISCARD"
        print(f"样本已丢弃：{result['discard_reason']}")
        return result

    object_polygon_ref = transform_bbox_to_polygon(active_object_bbox, H_contact)
    result["object_polygon_ref"] = object_polygon_ref

    original_area = (active_object_bbox[2] - active_object_bbox[0]) * (
        active_object_bbox[3] - active_object_bbox[1]
    )
    projected_area = polygon_area(object_polygon_ref)
    area_ratio = projected_area / original_area if original_area > 0 else 0.0
    result["projected_area"] = projected_area
    result["area_ratio"] = area_ratio

    print(f"原始物体 bbox 面积：{original_area:.1f}")
    print(f"投影后物体多边形面积：{projected_area:.1f}")
    print(f"面积比例：{area_ratio:.3f}")

    if projected_area <= 0 or area_ratio < 0.15:
        result["discard"] = True
        result["discard_reason"] = "projected_object_polygon_invalid"
        result["status"] = "DISCARD"
        print(f"样本已丢弃：{result['discard_reason']}")
        return result

    inside_count = count_points_near_polygon(mu_transformed, object_polygon_ref, tolerance_px=8.0)
    result["inside_count"] = inside_count
    print(f"\n接触点在物体区域内/附近的数量：{inside_count} / {len(mu_transformed)}")
    if inside_count < 4:
        result["discard"] = True
        result["discard_reason"] = "transformed_contact_points_off_object"
        result["status"] = "DISCARD"
        print(f"样本已丢弃：{result['discard_reason']}")
        return result

    centroid_dist = cv2.pointPolygonTest(
        object_polygon_ref.astype(np.float32),
        tuple(contact_centroid),
        True,
    )
    result["centroid_dist"] = centroid_dist
    print(f"接触点中心到物体多边形的距离：{centroid_dist:.1f}")
    if centroid_dist < -8.0:
        result["discard"] = True
        result["discard_reason"] = "transformed_contact_centroid_off_object"
        result["status"] = "DISCARD"
        print(f"样本已丢弃：{result['discard_reason']}")
        return result

    print("\n" + "=" * 60)
    print("Step 7: 边界检查")
    print("=" * 60)

    for i, pt in enumerate(tau_transformed):
        if pt is None:
            continue
        if pt[0] < 0 or pt[1] < 0 or pt[0] > w_img or pt[1] > h_img:
            result["discard"] = True
            result["discard_reason"] = f"trajectory_out_of_bounds_t+{i}"
            result["status"] = "DISCARD"
            print(f"轨迹点 t+{i} 越界：({pt[0]:.1f}, {pt[1]:.1f})")
            print(f"样本已丢弃：{result['discard_reason']}")
            return result

    for i, mu in enumerate(mu_transformed):
        if mu[0] < 0 or mu[1] < 0 or mu[0] > w_img or mu[1] > h_img:
            result["discard"] = True
            result["discard_reason"] = f"contact_point_out_of_bounds_mu_{i+1}"
            result["status"] = "DISCARD"
            print(f"接触点 μ_{i+1} 越界：({mu[0]:.1f}, {mu[1]:.1f})")
            print(f"样本已丢弃：{result['discard_reason']}")
            return result

    print("所有点均在图像边界内")

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

    print("\n" + "=" * 60)
    print("Step 8: 生成诊断可视化")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    full_overlay = draw_problem3_full_overlay(
        ref_img,
        object_polygon_ref,
        hand_polygon_ref,
        mu_transformed,
        tau_transformed,
        t_contact,
        active_hand,
        ref_idx,
        result["discard"],
        result["discard_reason"],
        contact_centroid,
    )
    full_overlay_path = output_dir / "problem3_ref_full_overlay.png"
    cv2.imwrite(str(full_overlay_path), full_overlay)
    result["full_overlay_path"] = str(full_overlay_path)
    print(f"Full-frame 诊断图：{full_overlay_path}")

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
    print(f"Crop 图：{crop_path}")

    print("\n" + "=" * 80)
    print("Cell 3 最终摘要")
    print("=" * 80)
    print(f"  frame: {t_contact}")
    print(f"  active_hand: {active_hand}")
    print(f"  ref_idx: {ref_idx}")
    print(f"  trajectory_points: {len(trajectory_pixels)}")
    print(f"  transformed_contact_points: {len(mu_transformed)}")
    print(f"  projected_object_area: {projected_area:.1f}")
    print(f"  area_ratio: {area_ratio:.3f}")
    print(f"  points_inside_object: {inside_count} / {len(mu_transformed)}")
    print(f"  centroid_distance: {centroid_dist:.1f}")
    print("\n最终状态：KEEP")
    print("接触点通过几何一致性检查")
    print("\n输出文件:")
    print(f"  Full-frame: {full_overlay_path}")
    print(f"  Crop: {crop_path}")

    result["status"] = "KEEP"
    return result
