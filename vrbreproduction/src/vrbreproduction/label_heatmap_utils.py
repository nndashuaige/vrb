"""
Label Heatmap Utilities for VRB Data Processing Pipeline
========================================================

本模块提供 label heatmap 生成功能，用于将 GMM 标签中心点转换为可视化热力分布。

核心功能：
1. 基于高斯核生成单点热图
2. 批量生成多模式热图
3. 合并热图（支持 max/sum 方式）
4. 热图叠加到 RGB 图像
5. 保存输出文件

使用方式：
    from vrbreproduction.label_heatmap_utils import (
        make_gaussian_heatmap,
        build_label_heatmaps,
        merge_label_heatmaps,
        overlay_heatmap_on_rgb,
        save_label_heatmap_outputs
    )

作者：VRB 复现项目
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Union


def make_gaussian_heatmap(
    height: int,
    width: int,
    center_xy: Tuple[float, float],
    sigma_px: float,
    amplitude: float = 1.0
) -> np.ndarray:
    """
    生成二维高斯热图（单中心点）

    参数说明：
    ----------
    height : int
        热图高度（像素）
    width : int
        热图宽度（像素）
    center_xy : Tuple[float, float]
        热区中心点坐标 (x, y)，像素单位
    sigma_px : float
        高斯核标准差，控制热区扩散范围
        - 值越大 → 热点越平滑分散
        - 值越小 → 热点越尖锐集中
        - 推荐范围：8-20 像素
    amplitude : float, default=1.0
        热区振幅（峰值强度），可接入 GMM 权重进行缩放

    返回值：
    --------
    np.ndarray
        shape=[height, width]，归一化到 [0, 1] 的高斯热图

    示例：
    --------
    >>> heatmap = make_gaussian_heatmap(256, 456, (100, 50), sigma_px=12.0)
    >>> print(heatmap.shape)
    (256, 456)
    """
    # 创建网格坐标
    y_coords, x_coords = np.ogrid[:height, :width]

    # 中心点坐标
    cx, cy = center_xy

    # 计算高斯值：exp(-((x-cx)^2 + (y-cy)^2) / (2 * sigma^2))
    # 使用向量化操作避免循环
    gaussian_value = np.exp(
        -((x_coords - cx) ** 2 + (y_coords - cy) ** 2) / (2 * sigma_px ** 2)
    )

    # 乘以振幅并归一化
    heatmap = amplitude * gaussian_value

    return heatmap


def _sanitize_covariance(
    covariance_xy: np.ndarray,
    min_sigma_px: float = 3.0,
    max_sigma_px: Optional[float] = 40.0
) -> np.ndarray:
    """
    Clamp a 2x2 covariance matrix to a numerically stable pixel range.

    GMM covariance comes from raw contact pixels. Very thin hand-edge clusters can
    produce near-singular covariances, which make the visual heatmap disappear.
    """
    cov = np.asarray(covariance_xy, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)

    eigvals, eigvecs = np.linalg.eigh(cov)
    min_var = float(min_sigma_px) ** 2
    max_var = None if max_sigma_px is None else float(max_sigma_px) ** 2
    eigvals = np.maximum(eigvals, min_var)
    if max_var is not None:
        eigvals = np.minimum(eigvals, max_var)

    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return cov.astype(np.float32)


def make_covariance_gaussian_heatmap(
    height: int,
    width: int,
    center_xy: Tuple[float, float],
    covariance_xy: np.ndarray,
    amplitude: float = 1.0,
    covariance_scale: float = 1.0,
    min_sigma_px: float = 3.0,
    max_sigma_px: Optional[float] = 40.0
) -> np.ndarray:
    """
    生成二维椭圆高斯热图（单中心点）。

    covariance_xy 应为像素坐标系下的 2x2 协方差矩阵，坐标顺序为 (x, y)。
    """
    y_coords, x_coords = np.ogrid[:height, :width]
    cx, cy = center_xy

    covariance_xy = np.asarray(covariance_xy, dtype=np.float32) * float(covariance_scale)
    covariance_xy = _sanitize_covariance(
        covariance_xy,
        min_sigma_px=min_sigma_px,
        max_sigma_px=max_sigma_px,
    )
    inv_cov = np.linalg.inv(covariance_xy)

    dx = x_coords - cx
    dy = y_coords - cy
    mahalanobis = (
        inv_cov[0, 0] * dx ** 2
        + 2.0 * inv_cov[0, 1] * dx * dy
        + inv_cov[1, 1] * dy ** 2
    )
    heatmap = amplitude * np.exp(-0.5 * mahalanobis)
    return heatmap.astype(np.float32)


def transform_covariances_by_homography(
    covariances_xy: np.ndarray,
    centers_xy: np.ndarray,
    H: np.ndarray,
    min_variance: float = 1e-3
) -> np.ndarray:
    """
    将 contact frame 中的 GMM covariance 近似变换到 reference frame。

    Homography 是非线性映射，这里使用每个 GMM 均值处的局部 Jacobian：
    Sigma_ref = J * Sigma_contact * J.T
    """
    covariances_xy = np.asarray(covariances_xy, dtype=np.float64)
    centers_xy = np.asarray(centers_xy, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)

    if covariances_xy.shape != (len(centers_xy), 2, 2):
        raise ValueError(
            f"covariances shape {covariances_xy.shape} must be (K, 2, 2), "
            f"where K={len(centers_xy)}"
        )

    transformed = np.zeros_like(covariances_xy, dtype=np.float64)
    for k, (x, y) in enumerate(centers_xy):
        a = H[0, 0] * x + H[0, 1] * y + H[0, 2]
        b = H[1, 0] * x + H[1, 1] * y + H[1, 2]
        d = H[2, 0] * x + H[2, 1] * y + H[2, 2]
        if abs(d) < 1e-8:
            d = 1e-8 if d >= 0 else -1e-8

        jacobian = np.array([
            [
                (H[0, 0] * d - a * H[2, 0]) / (d ** 2),
                (H[0, 1] * d - a * H[2, 1]) / (d ** 2),
            ],
            [
                (H[1, 0] * d - b * H[2, 0]) / (d ** 2),
                (H[1, 1] * d - b * H[2, 1]) / (d ** 2),
            ],
        ])
        cov = jacobian @ covariances_xy[k] @ jacobian.T
        cov = 0.5 * (cov + cov.T)
        cov += np.eye(2) * float(min_variance)
        transformed[k] = cov

    return transformed.astype(np.float32)


def build_label_heatmaps(
    image_shape: Tuple[int, int],
    centers_xy: np.ndarray,
    sigma_px: float = 12.0,
    weights: Optional[np.ndarray] = None,
    normalize_each: bool = True,
    covariances_xy: Optional[np.ndarray] = None,
    covariance_scale: float = 1.0,
    min_sigma_px: float = 3.0,
    max_sigma_px: Optional[float] = 40.0
) -> np.ndarray:
    """
    根据多个中心点生成每组 mode 对应的 heatmaps

    参数说明：
    ----------
    image_shape : Tuple[int, int]
        图像尺寸 (height, width)
    centers_xy : np.ndarray
        中心点坐标，shape=[K, 2]，每行为 (x, y)
    sigma_px : float, default=12.0
        高斯核标准差（像素）
        - 值越大 → 热点越平滑分散
        - 值越小 → 热点越尖锐集中
        - 默认 12.0，经过测试适合 EPIC-Kitchens 图像尺度
    weights : np.ndarray, optional
        每个 mode 的权重，shape=[K]
        - 传入时按权重缩放每个 mode 的振幅
        - 不传时所有 mode 等权重
    normalize_each : bool, default=True
        是否对每个 mode 热图单独归一化

    返回值：
    --------
    np.ndarray
        shape=[K, H, W]，K 个模式对应的热图

    covariances_xy : np.ndarray, optional
        每个 mode 的 2x2 covariance，shape=[K, 2, 2]。
        传入时生成椭圆高斯；不传时保留原来的圆形高斯。
    """
    height, width = image_shape
    K = len(centers_xy)
    centers_xy = np.asarray(centers_xy, dtype=np.float32)

    # 处理权重：默认为等权重
    if weights is None:
        weights = np.ones(K)
    else:
        weights = np.asarray(weights).flatten()
        if len(weights) != K:
            raise ValueError(f"Weights length ({len(weights)}) must match centers ({K})")

    # 归一化权重（可选）
    if normalize_each:
        weights = weights / weights.sum()

    # 初始化输出数组
    heatmaps = np.zeros((K, height, width), dtype=np.float32)
    if covariances_xy is not None:
        covariances_xy = np.asarray(covariances_xy, dtype=np.float32)
        if covariances_xy.shape != (K, 2, 2):
            raise ValueError(
                f"covariances_xy shape {covariances_xy.shape} must be (K, 2, 2), "
                f"where K={K}"
            )

    # 为每个 mode 生成热图
    for k in range(K):
        cx, cy = centers_xy[k]
        amplitude = weights[k] if normalize_each else 1.0

        # 使用权重调整振幅（当 normalize_each=False 时）
        final_amplitude = amplitude if not normalize_each else weights[k]

        if covariances_xy is None:
            heatmaps[k] = make_gaussian_heatmap(
                height, width,
                center_xy=(cx, cy),
                sigma_px=sigma_px,
                amplitude=final_amplitude
            )
        else:
            heatmaps[k] = make_covariance_gaussian_heatmap(
                height, width,
                center_xy=(cx, cy),
                covariance_xy=covariances_xy[k],
                amplitude=final_amplitude,
                covariance_scale=covariance_scale,
                min_sigma_px=min_sigma_px,
                max_sigma_px=max_sigma_px,
            )

    return heatmaps


def merge_label_heatmaps(
    per_mode_heatmaps: np.ndarray,
    merge_method: str = "max",
    normalize_output: bool = True
) -> np.ndarray:
    """
    将多模式热图合并为单张热图

    参数说明：
    ----------
    per_mode_heatmaps : np.ndarray
        输入热图，shape=[K, H, W]
    merge_method : str, default="max"
        合并方式：
        - "max": 取各模式最大值，适合绘制「可能接触区域」联合热点图
        - "sum": 累加各模式强度，重叠区域会过亮
    normalize_output : bool, default=True
        是否将输出归一化到 [0, 1]

    返回值：
    --------
    np.ndarray
        shape=[H, W]，合并后的热图

    为什么默认使用 "max" 而非 "sum"：
    - max 方式能更好地展示「可能的接触区域」分布
    - sum 方式会让重叠区域过亮，降低可视化效果
    - 当多个 mode 指向同一区域时，max 会突出该区域而非叠加强度
    """
    if merge_method == "max":
        merged = np.max(per_mode_heatmaps, axis=0)
    elif merge_method == "sum":
        merged = np.sum(per_mode_heatmaps, axis=0)
    else:
        raise ValueError(f"Unknown merge_method: {merge_method}. Use 'max' or 'sum'.")

    if normalize_output:
        if merged.max() > 0:
            merged = merged / merged.max()

    return merged


def overlay_heatmap_on_rgb(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
    max_intensity: float = 1.0
) -> np.ndarray:
    """
    将热力图叠加到 RGB 图像上

    参数说明：
    ----------
    image_rgb : np.ndarray
        RGB 图像，shape=[H, W, 3]
    heatmap : np.ndarray
        归一化热图，shape=[H, W]，值范围 [0, 1]
    alpha : float, default=0.45
        热图透明度
        - 值越大 → 热图显示越明显
        - 值越小 → 原始图像显示越清晰
        - 推荐范围：0.3-0.6
    colormap : int, default=cv2.COLORMAP_JET
        OpenCV 色图类型
        - cv2.COLORMAP_JET: 蓝→青→黄→红渐变
        - cv2.COLORMAP_HOT: 黑→红→黄→白渐变
        - cv2.COLORMAP_VIRIDIS: 紫→蓝→绿渐变
    max_intensity : float, default=1.0
        热图强度上限，用于控制热图亮度

    返回值：
    --------
    np.ndarray
        shape=[H, W, 3]，叠加效果图（RGB）
    """
    # 确保图像格式正确
    if image_rgb.dtype != np.uint8:
        image_rgb = (image_rgb * 255).astype(np.uint8) if image_rgb.max() <= 1.0 else image_rgb.astype(np.uint8)

    # 准备热图伪彩色
    heatmap_display = (heatmap * 255 * max_intensity).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_display, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # 叠加
    overlay = cv2.addWeighted(image_rgb, 1.0, heatmap_colored, alpha, 0)

    return overlay


def make_contact_anchored_arrow(
    tau_points: List[Optional[np.ndarray]],
    contact_anchor_xy: np.ndarray,
    display_length_px: float = 70.0,
    source_step: Union[int, str] = "last",
    min_motion_px: float = 1.0
) -> Optional[dict]:
    """
    Convert absolute hand/wrist waypoints into a VRB-style display arrow.

    The paper trains the trajectory head on relative post-contact shifts, while
    contact points provide the spatial grounding. For visualization, keep the
    motion direction from the hand trajectory but anchor the arrow at the contact
    heatmap center.
    """
    valid_tau = [
        np.asarray(pt, dtype=np.float32).reshape(2)
        for pt in tau_points
        if pt is not None
    ]
    if len(valid_tau) < 2 or contact_anchor_xy is None:
        return None

    start_tau = valid_tau[0]
    if isinstance(source_step, str):
        if source_step != "last":
            raise ValueError("source_step must be 'last' or an integer index")
        end_tau = valid_tau[-1]
        end_index = len(valid_tau) - 1
    else:
        end_index = int(source_step)
        end_index = max(1, min(len(valid_tau) - 1, end_index))
        end_tau = valid_tau[end_index]

    raw_motion = end_tau - start_tau
    motion_norm = float(np.linalg.norm(raw_motion))
    if motion_norm < float(min_motion_px):
        return None

    axis_norm = max(abs(float(raw_motion[0])), abs(float(raw_motion[1])), 1e-6)
    scale = float(display_length_px) / axis_norm
    display_motion = raw_motion * scale
    start_xy = np.asarray(contact_anchor_xy, dtype=np.float32).reshape(2)
    end_xy = start_xy + display_motion

    anchored_points = [start_xy + (pt - start_tau) * scale for pt in valid_tau]
    return {
        "start_xy": start_xy,
        "end_xy": end_xy.astype(np.float32),
        "display_motion": display_motion.astype(np.float32),
        "raw_motion": raw_motion.astype(np.float32),
        "scale": scale,
        "source_step": end_index,
        "anchored_points": anchored_points,
    }


def estimate_local_heatmap_radius_px(
    heatmap: np.ndarray,
    anchor_xy: np.ndarray,
    threshold_ratio: float = 0.35,
    percentile: float = 85.0
) -> Optional[float]:
    """
    Estimate the visible size of the contact heatmap around the arrow anchor.

    This keeps the final arrow visually tied to the heatmap scale instead of a
    fixed pixel length.
    """
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.size == 0 or float(np.max(heatmap)) <= 0:
        return None

    anchor_xy = np.asarray(anchor_xy, dtype=np.float32).reshape(2)
    threshold = float(np.max(heatmap)) * float(threshold_ratio)
    mask = (heatmap >= threshold).astype(np.uint8)
    num_labels, labels, _, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    h, w = heatmap.shape[:2]
    ax = int(np.clip(round(float(anchor_xy[0])), 0, w - 1))
    ay = int(np.clip(round(float(anchor_xy[1])), 0, h - 1))
    chosen_label = int(labels[ay, ax])
    if chosen_label == 0:
        component_centers = centroids[1:].astype(np.float32)
        distances_to_anchor = np.linalg.norm(component_centers - anchor_xy[None, :], axis=1)
        chosen_label = int(np.argmin(distances_to_anchor)) + 1

    ys, xs = np.where(labels == chosen_label)
    if len(xs) == 0:
        return None
    distances = np.sqrt((xs.astype(np.float32) - anchor_xy[0]) ** 2 + (ys.astype(np.float32) - anchor_xy[1]) ** 2)
    if len(distances) == 0:
        return None
    return float(np.percentile(distances, percentile))


def draw_vrb_style_affordance_overlay(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    tau_points: List[Optional[np.ndarray]],
    contact_anchor_xy: np.ndarray,
    heatmap_alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
    arrow_length_px: Optional[float] = None,
    arrow_length_heatmap_ratio: float = 1.2,
    arrow_width_px: Optional[int] = None,
    arrow_tip_length: float = 0.28,
    source_step: Union[int, str] = "last",
    draw_shadow: bool = True
) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Draw the final VRB-style affordance visualization.

    Output semantics:
    - heatmap: likely contact region on the human-less reference frame
    - white arrow: post-contact hand/wrist motion direction, anchored at contact
    """
    overlay = overlay_heatmap_on_rgb(
        image_rgb,
        heatmap,
        alpha=heatmap_alpha,
        colormap=colormap,
    )

    estimated_radius = estimate_local_heatmap_radius_px(
        heatmap,
        contact_anchor_xy,
        threshold_ratio=0.35,
        percentile=85.0,
    )
    if arrow_length_px is None:
        if estimated_radius is None:
            short_side = min(image_rgb.shape[:2])
            arrow_length_px = short_side * 0.10
        else:
            arrow_length_px = estimated_radius * float(arrow_length_heatmap_ratio)

    if arrow_width_px is None:
        arrow_width_px = max(2, int(round(float(arrow_length_px) * 0.08)))

    arrow_info = make_contact_anchored_arrow(
        tau_points=tau_points,
        contact_anchor_xy=contact_anchor_xy,
        display_length_px=arrow_length_px,
        source_step=source_step,
    )
    if arrow_info is None:
        return overlay, None

    h, w = overlay.shape[:2]
    start = np.round(arrow_info["start_xy"]).astype(int)
    end = np.round(arrow_info["end_xy"]).astype(int)
    start_tuple = (int(np.clip(start[0], 0, w - 1)), int(np.clip(start[1], 0, h - 1)))
    end_tuple = (int(np.clip(end[0], 0, w - 1)), int(np.clip(end[1], 0, h - 1)))

    if draw_shadow:
        cv2.arrowedLine(
            overlay,
            start_tuple,
            end_tuple,
            (0, 0, 0),
            int(arrow_width_px + 2),
            line_type=cv2.LINE_AA,
            tipLength=arrow_tip_length,
        )
    cv2.arrowedLine(
        overlay,
        start_tuple,
        end_tuple,
        (255, 255, 255),
        int(arrow_width_px),
        line_type=cv2.LINE_AA,
        tipLength=arrow_tip_length,
    )
    arrow_info["estimated_heatmap_radius_px"] = estimated_radius
    arrow_info["display_length_px"] = float(arrow_length_px)
    arrow_info["arrow_width_px"] = int(arrow_width_px)
    return overlay, arrow_info


def save_label_heatmap_outputs(
    merged_heatmap: np.ndarray,
    ref_image: np.ndarray,
    output_dir: Union[str, Path],
    prefix: str = "step4_label_heatmap",
    sigma_px: float = 12.0,
    merge_method: str = "max",
    use_weights: bool = True,
    heatmap_mode: str = "isotropic",
    t_contact: Optional[int] = None,
    active_hand: Optional[str] = None,
    ref_idx: Optional[int] = None
) -> dict:
    """
    保存 label heatmap 相关输出文件

    参数说明：
    ----------
    merged_heatmap : np.ndarray
        合并后的热图，shape=[H, W]
    ref_image : np.ndarray
        参考帧图像（RGB），shape=[H, W, 3]
    output_dir : Union[str, Path]
        输出目录
    prefix : str, default="step4_label_heatmap"
        输出文件名前缀
    sigma_px : float, default=12.0
        高斯核标准差（用于标注）
    merge_method : str, default="max"
        合并方式（用于标注）
    use_weights : bool, default=True
        是否使用了 GMM 权重（用于标注）
    heatmap_mode : str, default="isotropic"
        热图生成方式（用于标注）
    t_contact : int, optional
        接触帧号（用于标注）
    active_hand : str, optional
        主动手（用于标注）
    ref_idx : int, optional
        参考帧号（用于标注）

    返回值：
    --------
    dict
        包含各输出文件路径的字典

    必存文件：
    ---------
    - {prefix}_merged.npy: 原始热图数据
    - {prefix}_merged.png: 热图伪彩色图
    - {prefix}_merged_overlay.png: 热图叠加效果图
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 保存原始热图 (.npy)
    npy_path = output_dir / f"{prefix}_merged.npy"
    np.save(npy_path, merged_heatmap)

    # 2. 生成并保存热图伪彩色图 (.png)
    heatmap_uint8 = (merged_heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    png_path = output_dir / f"{prefix}_merged.png"
    cv2.imwrite(str(png_path), cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR))

    # 3. 生成并保存叠加效果图 (.png)
    overlay = overlay_heatmap_on_rgb(ref_image, merged_heatmap, alpha=0.45)
    overlay_path = output_dir / f"{prefix}_merged_overlay.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # 打印摘要
    print("=" * 60)
    print("Label Heatmap 生成摘要")
    print("=" * 60)
    if t_contact is not None:
        print(f"  当前帧 (t_contact): {t_contact}")
    if active_hand is not None:
        print(f"  主动手 (active_hand): {active_hand}")
    if ref_idx is not None:
        print(f"  参考帧 (ref_idx): {ref_idx}")
    print(f"  heatmap_mode: {heatmap_mode}")
    if heatmap_mode == "isotropic":
        print(f"  sigma_px: {sigma_px} (热区扩散半径)")
    else:
        print(f"  sigma_px: {sigma_px} (isotropic fallback only)")
    print(f"  merge_method: {merge_method}")
    print(f"  使用 GMM 权重: {'是' if use_weights else '否'}")
    print(f"  热图尺寸: {merged_heatmap.shape}")
    print("\n输出文件:")
    print(f"  原始热图: {npy_path}")
    print(f"  伪彩色热图: {png_path}")
    print(f"  叠加效果图: {overlay_path}")

    return {
        "npy_path": str(npy_path),
        "png_path": str(png_path),
        "overlay_path": str(overlay_path)
    }
