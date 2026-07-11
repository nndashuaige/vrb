from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vrbreproduction.e20.inpaint_utils import (
    choose_final_backend,
    collect_hand_bboxes_px,
    composite_inside_mask,
    extend_mask_to_border,
    fix_e18b_path,
    heatmap_top_bbox,
    letterbox_pad,
    mask_overlap_ratio,
    rel_dir_from_sample_key,
    sample_key_from_rel_dir,
    skin_ratio_in_mask,
    unletterbox,
)


def _hand(score, side, bbox):
    return SimpleNamespace(
        score=score,
        side=SimpleNamespace(name=side),
        bbox=SimpleNamespace(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3]),
    )


def test_fix_e18b_path_reanchors_stale_absolute_path(tmp_path):
    fixed = fix_e18b_path("/old/session/outputs/e18b/a/b.png", tmp_path)
    assert fixed == tmp_path / "outputs" / "e18b" / "a" / "b.png"
    with pytest.raises(ValueError):
        fix_e18b_path("/old/session/outputs/e18/a.png", tmp_path)


def test_sample_key_round_trip():
    rel = "subaction_00_P01_109_0/episode_01_left/frame_000088_left"
    key = sample_key_from_rel_dir(rel)
    assert key == "subaction_00_P01_109_0__episode_01_left__frame_000088_left"
    assert rel_dir_from_sample_key(key) == rel


def test_collect_hand_bboxes_filters_and_clips():
    det = SimpleNamespace(
        hands=[
            _hand(0.9, "left", (-0.1, 0.25, 0.5, 1.2)),
            _hand(0.4, "right", (0.1, 0.1, 0.2, 0.2)),
            _hand(0.8, "right", (0.5, 0.0, 1.2, 0.5)),
        ]
    )
    boxes = collect_hand_bboxes_px(det, (100, 200), score_threshold=0.5)
    assert boxes == [[0.0, 25.0, 100.0, 100.0], [100.0, 0.0, 200.0, 50.0]]


def test_extend_mask_to_border_branching():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:50, 45:55] = 255
    extended = extend_mask_to_border(mask, [40, 40, 60, 70], (100, 100))
    assert extended[-1].sum() > 0
    touching = extend_mask_to_border(mask, [40, 40, 60, 99], (100, 100))
    assert np.array_equal(touching, mask)
    too_far = extend_mask_to_border(mask, [40, 10, 60, 20], (100, 100), max_extend_frac=0.2)
    assert np.array_equal(too_far, mask)


def test_letterbox_round_trip_is_exact():
    image = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
    mask = np.zeros((20, 30), dtype=np.uint8)
    padded_img, padded_mask, meta = letterbox_pad(image, mask, target=40)
    assert padded_img.shape == (40, 40, 3)
    assert padded_mask.shape == (40, 40)
    assert np.array_equal(unletterbox(padded_img, meta), image)


def test_composite_keeps_mask_outside_exact():
    original = np.zeros((20, 20, 3), dtype=np.uint8)
    edited = np.full((20, 20, 3), 255, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:10, 5:10] = 255
    out = composite_inside_mask(original, edited, mask, feather_px=0)
    assert np.array_equal(out[mask == 0], original[mask == 0])
    assert out[7, 7, 0] == 255


def test_skin_ratio_synthetic_positive_and_negative():
    skin_bgr = cv2.cvtColor(np.array([[[120, 150, 100]]], dtype=np.uint8), cv2.COLOR_YCrCb2BGR)[0, 0]
    image = np.tile(skin_bgr, (10, 10, 1))
    mask = np.ones((10, 10), dtype=np.uint8) * 255
    assert skin_ratio_in_mask(image, mask, erode_px=0) == 1.0
    image[:] = (255, 0, 0)
    assert skin_ratio_in_mask(image, mask, erode_px=0) == 0.0


def test_heatmap_bbox_overlap_and_backend_choice():
    hm = np.zeros((10, 10), dtype=np.float32)
    hm[2:5, 3:7] = 1.0
    assert heatmap_top_bbox(hm, 0.5) == [3, 2, 7, 5]
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 255
    assert mask_overlap_ratio(mask, [3, 2, 7, 5]) == 1.0
    assert choose_final_backend("abc", {"lama": True, "sd": False}) == "lama"
    assert choose_final_backend("abc", {"lama": False, "sd": False}) == "none"
    assert choose_final_backend("abc", {"lama": True, "sd": True}) == choose_final_backend("abc", {"lama": True, "sd": True})

