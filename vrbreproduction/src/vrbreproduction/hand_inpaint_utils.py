"""Hand-mask helpers and inpainting backends for E19 reference cleanup."""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class InpaintConfig:
    backend: str = "opencv_telea"
    mask_pad_px: int = 12
    opencv_radius: float = 3.0
    lama_command: Optional[str] = None
    sd_command: Optional[str] = None
    max_mask_area_ratio: float = 0.35
    boundary_delta_max: float = 35.0


def _image_hw(image_shape) -> Tuple[int, int]:
    if len(image_shape) < 2:
        raise ValueError(f"image_shape must have at least two dimensions, got {image_shape!r}")
    return int(image_shape[0]), int(image_shape[1])


def bbox_to_mask(image_shape, bbox_px, pad_px: int = 12, ellipse: bool = False) -> np.ndarray:
    """Convert an ``(x1, y1, x2, y2)`` pixel bbox to a uint8 mask."""
    h, w = _image_hw(image_shape)
    mask = np.zeros((h, w), dtype=np.uint8)
    if bbox_px is None or len(bbox_px) != 4:
        return mask

    x1, y1, x2, y2 = [float(v) for v in bbox_px]
    x1_i = max(0, min(w, int(np.floor(x1))))
    y1_i = max(0, min(h, int(np.floor(y1))))
    x2_i = max(0, min(w, int(np.ceil(x2))))
    y2_i = max(0, min(h, int(np.ceil(y2))))
    if x2_i <= x1_i or y2_i <= y1_i:
        return mask

    if ellipse:
        center = ((x1_i + x2_i) // 2, (y1_i + y2_i) // 2)
        axes = (max(1, (x2_i - x1_i) // 2), max(1, (y2_i - y1_i) // 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, thickness=-1)
    else:
        mask[y1_i:y2_i, x1_i:x2_i] = 255

    pad = int(max(0, pad_px))
    if pad > 0:
        kernel_size = 2 * pad + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask.astype(np.uint8, copy=False)


def combine_hand_masks(image_shape, bboxes_px, pad_px: int = 12, ellipse: bool = False) -> np.ndarray:
    h, w = _image_hw(image_shape)
    combined = np.zeros((h, w), dtype=np.uint8)
    for bbox in bboxes_px or []:
        combined = cv2.bitwise_or(combined, bbox_to_mask(image_shape, bbox, pad_px=pad_px, ellipse=ellipse))
    return combined


def inpaint_opencv_telea(image_bgr: np.ndarray, mask_u8: np.ndarray, radius: float = 3.0) -> np.ndarray:
    if image_bgr is None or image_bgr.ndim != 3:
        raise ValueError("image_bgr must be an HxWx3 image")
    if mask_u8 is None or mask_u8.shape[:2] != image_bgr.shape[:2]:
        raise ValueError("mask_u8 must match image_bgr height and width")
    mask = (mask_u8 > 0).astype(np.uint8) * 255
    if int(mask.sum()) == 0:
        return image_bgr.copy()
    return cv2.inpaint(image_bgr, mask, float(radius), cv2.INPAINT_TELEA)


def compute_mask_area_ratio(mask_u8: np.ndarray) -> float:
    if mask_u8 is None or mask_u8.size == 0:
        return 0.0
    return float(np.count_nonzero(mask_u8 > 0) / mask_u8.size)


def compute_boundary_delta(original_bgr: np.ndarray, edited_bgr: np.ndarray, mask_u8: np.ndarray, band_px: int = 5) -> float:
    if original_bgr.shape != edited_bgr.shape:
        raise ValueError("original_bgr and edited_bgr must have the same shape")
    if mask_u8.shape[:2] != original_bgr.shape[:2]:
        raise ValueError("mask_u8 must match image height and width")
    mask = (mask_u8 > 0).astype(np.uint8)
    if int(mask.sum()) == 0:
        return 0.0
    band = int(max(1, band_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    ring = (dilated > 0) & (mask == 0)
    if not bool(ring.any()):
        return 0.0
    diff = np.abs(original_bgr.astype(np.float32) - edited_bgr.astype(np.float32))
    return float(diff[ring].mean())


def _write_backend_inputs(image_bgr: np.ndarray, mask_u8: np.ndarray, work_dir: Path) -> Dict[str, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / "input.png"
    mask_path = work_dir / "mask.png"
    output_path = work_dir / "output.png"
    cv2.imwrite(str(input_path), image_bgr)
    cv2.imwrite(str(mask_path), (mask_u8 > 0).astype(np.uint8) * 255)

    lama_in = work_dir / "lama_in"
    lama_out = work_dir / "lama_out"
    lama_in.mkdir(parents=True, exist_ok=True)
    lama_out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(lama_in / "input.png"), image_bgr)
    cv2.imwrite(str(lama_in / "input_mask.png"), (mask_u8 > 0).astype(np.uint8) * 255)
    return {
        "input": input_path,
        "mask": mask_path,
        "output": output_path,
        "work_dir": work_dir,
        "lama_in": lama_in,
        "lama_out": lama_out,
    }


def _format_command(command_template: str, paths: Dict[str, Path]) -> str:
    return command_template.format(
        input=str(paths["input"]),
        mask=str(paths["mask"]),
        output=str(paths["output"]),
        work_dir=str(paths["work_dir"]),
    )


def _run_cli_backend(command_template: str, paths: Dict[str, Path]) -> Dict[str, Any]:
    command = _format_command(command_template, paths)
    started = time.time()
    completed = subprocess.run(command, shell=True, cwd=str(paths["work_dir"]), text=True, capture_output=True)
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-2000:] if completed.stdout else "",
        "stderr_tail": completed.stderr[-2000:] if completed.stderr else "",
        "elapsed_sec": float(time.time() - started),
    }


def _latest_image(path: Path) -> Optional[Path]:
    candidates = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        candidates.extend(Path(path).glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _read_backend_output(output_path: Path, fallback_dir: Optional[Path] = None) -> Tuple[np.ndarray, Path]:
    chosen = output_path if output_path.exists() else None
    if chosen is None and fallback_dir is not None:
        chosen = _latest_image(fallback_dir)
    if chosen is None:
        raise FileNotFoundError(f"inpaint backend did not produce output: {output_path}")
    edited = cv2.imread(str(chosen))
    if edited is None:
        raise ValueError(f"failed to read inpaint backend output: {chosen}")
    return edited, chosen


def inpaint_image_with_backend(
    image_bgr: np.ndarray,
    mask_u8: np.ndarray,
    config: InpaintConfig,
    work_dir: Path,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    backend = str(config.backend or "none").lower()
    metadata: Dict[str, Any] = {
        "backend": backend,
        "inpaint_applied": False,
        "work_dir": str(work_dir),
        "output_path": None,
        "command": None,
        "returncode": None,
        "stdout_tail": None,
        "stderr_tail": None,
        "elapsed_sec": None,
    }

    if backend == "none":
        return image_bgr.copy(), metadata

    if backend == "opencv_telea":
        edited = inpaint_opencv_telea(image_bgr, mask_u8, radius=config.opencv_radius)
        metadata["inpaint_applied"] = True
        return edited, metadata

    paths = _write_backend_inputs(image_bgr, mask_u8, Path(work_dir))
    fallback_dir = None
    if backend == "lama_cli":
        if not config.lama_command:
            raise ValueError("lama_cli backend requires InpaintConfig.lama_command")
        run_meta = _run_cli_backend(config.lama_command, paths)
        fallback_dir = paths["lama_out"]
    elif backend == "sd_cli":
        if not config.sd_command:
            raise ValueError("sd_cli backend requires InpaintConfig.sd_command")
        run_meta = _run_cli_backend(config.sd_command, paths)
    else:
        raise ValueError(f"unknown inpaint backend: {backend}")

    metadata.update(run_meta)
    if int(run_meta["returncode"]) != 0:
        raise RuntimeError(f"{backend} command failed with code {run_meta['returncode']}: {run_meta['stderr_tail']}")
    edited, output_path = _read_backend_output(paths["output"], fallback_dir=fallback_dir)
    if edited.shape != image_bgr.shape:
        raise ValueError(f"inpaint output shape {edited.shape} does not match input shape {image_bgr.shape}")
    metadata["inpaint_applied"] = True
    metadata["output_path"] = str(output_path)

    if backend == "lama_cli" and output_path != paths["output"]:
        shutil.copyfile(output_path, paths["output"])
    return edited, metadata
