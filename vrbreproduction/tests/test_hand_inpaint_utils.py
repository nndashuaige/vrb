from pathlib import Path

import cv2
import numpy as np
import pytest

from vrbreproduction.hand_inpaint_utils import (
    InpaintConfig,
    bbox_to_mask,
    combine_hand_masks,
    compute_boundary_delta,
    compute_mask_area_ratio,
    inpaint_image_with_backend,
    inpaint_opencv_telea,
)


def test_bbox_to_mask_clips_and_dilates_without_leaving_image():
    image_shape = (100, 200, 3)
    bbox = [20, 10, 60, 50]

    mask = bbox_to_mask(image_shape, bbox, pad_px=0)

    assert mask.shape == (100, 200)
    assert mask.dtype == np.uint8
    assert mask[10:50, 20:60].min() == 255
    assert mask[:5, :5].max() == 0

    dilated = bbox_to_mask(image_shape, bbox, pad_px=12)
    assert dilated.shape == (100, 200)
    assert dilated.dtype == np.uint8
    assert int(dilated.sum()) > int(mask.sum())
    assert dilated[0, 0] == 0
    assert dilated[-1, -1] == 0


def test_bbox_to_mask_clips_bbox_at_edges():
    mask = bbox_to_mask((100, 200, 3), [-20, -10, 20, 20], pad_px=12)

    assert mask.shape == (100, 200)
    assert mask.dtype == np.uint8
    assert mask[:20, :20].max() == 255
    assert mask[-1, -1] == 0


def test_bbox_to_mask_can_use_ellipse_fill():
    mask = bbox_to_mask((100, 200, 3), [20, 10, 60, 50], pad_px=0, ellipse=True)

    assert mask[30, 40] == 255
    assert mask[10, 20] == 0


def test_combine_hand_masks_unions_boxes():
    mask = combine_hand_masks((100, 200, 3), [[20, 10, 60, 50], [100, 20, 140, 70]], pad_px=0)

    assert mask.shape == (100, 200)
    assert mask[20, 30] == 255
    assert mask[40, 120] == 255
    assert mask[90, 190] == 0


def test_inpaint_opencv_telea_preserves_shape_and_changes_masked_pixels():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :] = (20, 120, 200)
    image[10:50, 20:60] = (0, 0, 0)
    mask = bbox_to_mask(image.shape, [20, 10, 60, 50], pad_px=0)

    edited = inpaint_opencv_telea(image, mask, radius=3.0)

    assert edited.shape == image.shape
    assert edited.dtype == image.dtype
    assert np.mean(np.abs(edited[mask > 0].astype(np.float32) - image[mask > 0].astype(np.float32))) > 0


def test_compute_mask_area_ratio():
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[:2, :] = 255

    assert compute_mask_area_ratio(mask) == pytest.approx(0.2)


def test_compute_boundary_delta_uses_ring_not_mask_interior():
    original = np.zeros((30, 30, 3), dtype=np.uint8)
    edited = original.copy()
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    edited[10:20, 10:20] = 255

    assert compute_boundary_delta(original, edited, mask, band_px=3) == pytest.approx(0.0)

    ring = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    edited[(ring > 0) & (mask == 0)] = 60
    assert compute_boundary_delta(original, edited, mask, band_px=3) > 0


def test_inpaint_backend_none_returns_original(tmp_path: Path):
    image = np.full((100, 200, 3), 90, dtype=np.uint8)
    mask = bbox_to_mask(image.shape, [20, 10, 60, 50], pad_px=0)

    edited, meta = inpaint_image_with_backend(image, mask, InpaintConfig(backend="none"), tmp_path)

    assert np.array_equal(edited, image)
    assert meta["inpaint_applied"] is False
    assert meta["backend"] == "none"


def test_inpaint_backend_opencv_telea(tmp_path: Path):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :] = (20, 120, 200)
    image[10:50, 20:60] = (0, 0, 0)
    mask = bbox_to_mask(image.shape, [20, 10, 60, 50], pad_px=0)

    edited, meta = inpaint_image_with_backend(
        image,
        mask,
        InpaintConfig(backend="opencv_telea", opencv_radius=3.0),
        tmp_path,
    )

    assert edited.shape == image.shape
    assert meta["inpaint_applied"] is True
    assert meta["backend"] == "opencv_telea"
    assert np.mean(np.abs(edited[mask > 0].astype(np.float32) - image[mask > 0].astype(np.float32))) > 0


def test_inpaint_backend_unknown_raises(tmp_path: Path):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    mask = bbox_to_mask(image.shape, [20, 10, 60, 50], pad_px=0)

    with pytest.raises(ValueError, match="unknown inpaint backend"):
        inpaint_image_with_backend(image, mask, InpaintConfig(backend="bogus"), tmp_path)


def test_lama_cli_requires_command(tmp_path: Path):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    mask = bbox_to_mask(image.shape, [20, 10, 60, 50], pad_px=0)

    with pytest.raises(ValueError, match="lama_command"):
        inpaint_image_with_backend(image, mask, InpaintConfig(backend="lama_cli"), tmp_path)
