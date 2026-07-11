#!/usr/bin/env python3
"""Self-contained E20-C GPU worker.

E20-C changes vs E20-B (see docs/learning_note/实验记录/实验记录e20b.md §九):
1. Label-region protection (the headline fix): E20-B masks swallowed the
   interaction target in 20/56 samples (overlap_heatmap_top >= 0.99, e.g. the
   rucksack in subaction_01 and the cutting board in subaction_66) because HOA
   and DINO hand boxes often contain the held object and SAM segments the whole
   salient thing in the box. E20-C builds a protect mask from the exported
   label geometry (SAM object segmentation prompted by heatmap_top_bbox_px,
   plus the bbox rect and contact-point disks) and subtracts it from the
   inpaint mask as a hard constraint, every round.
2. SD hallucination gate: in-mask sharpness ratio vs the ORIGINAL frame.
   Calibrated on all 56 E20-B samples: every visually hallucinated SD final
   scored > 2.0, every clean one <= 1.95. SD is also barred from final on
   large holes (mask area > 0.40) where it invents new scenes.
3. Less aggressive masks: dilate 14 -> 8, bridge width 1.4 -> 1.3. The
   protect subtraction itself removes the biggest over-expansions.
4. Skin-confirmed re-detection: DINO hand/arm boxes only count (for QC and for
   re-inpaint rounds) if the box also shows dual-rule skin evidence. Kills the
   trouser-leg/knee/hose false fires behind the 3 spurious E20-B QC fails.
   Skin is an AND condition only - wooden counters pass skin rules alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

try:
    from e20_inpaint_utils import (
        boundary_gradient_ratio,
        bridge_components_to_border,
        composite_inside_mask,
        dilate_mask,
        disk_mask,
        dual_skin_ratio_in_mask,
        extend_mask_to_border,
        fill_mask_holes,
        laplacian_variance,
        mask_intersection_frac,
        mask_overlap_ratio,
        merge_overlapping_boxes,
        sharpness_ratio_vs_original,
        skin_ratio_in_bbox,
        skin_ratio_in_mask,
    )
except ModuleNotFoundError:
    from vrbreproduction.e20.inpaint_utils import (
        boundary_gradient_ratio,
        bridge_components_to_border,
        composite_inside_mask,
        dilate_mask,
        disk_mask,
        dual_skin_ratio_in_mask,
        extend_mask_to_border,
        fill_mask_holes,
        laplacian_variance,
        mask_intersection_frac,
        mask_overlap_ratio,
        merge_overlapping_boxes,
        sharpness_ratio_vs_original,
        skin_ratio_in_bbox,
        skin_ratio_in_mask,
    )


BOX_SKIN_EVIDENCE_MIN = 0.05
LARGE_DINO_BOX_AREA = 0.30
LARGE_DINO_SKIN_MIN = 0.03
MASK_AREA_CAP = 0.55
MASK_AREA_PASS = 0.65
DINO_DETECT_THRESHOLD = 0.30
DINO_REDETECT_THRESHOLD = 0.25
REDETECT_PASS_SCORE = 0.35
REDETECT_SKIN_MIN = 0.10
MAX_INPAINT_ROUNDS = 3
UPSCALE = 2
DINO_PROMPT = "a human hand. a human arm."
MASK_DILATE_PX = 8
BRIDGE_WIDTH_SCALE = 1.3

PROTECT_HM_PAD = 2
CONTACT_RADIUS = 6
PROTECT_AREA_MAX = 0.30
PROTECT_MIN_HM_COVER = 0.50
PROTECT_HAND_OVERLAP_MAX = 0.60
LABEL_CONFLICT_FRAC = 0.50

SD_HALLU_SHARP_RATIO = 2.0
SD_FINAL_MAX_AREA = 0.40


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha8(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def _pil_from_bgr(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _bgr_from_pil(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _expand_box(box: list[float], pad: float, image_hw: tuple[int, int]) -> list[float]:
    h, w = image_hw
    return [
        max(0.0, float(box[0]) - pad),
        max(0.0, float(box[1]) - pad),
        min(float(w), float(box[2]) + pad),
        min(float(h), float(box[3]) + pad),
    ]


def _rect_mask(image_hw: tuple[int, int], box: list[float] | None, pad: float = 0.0) -> np.ndarray:
    h, w = image_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if not box or len(box) != 4:
        return mask
    x1, y1, x2, y2 = _expand_box([float(v) for v in box], pad, image_hw)
    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255
    return mask


class HandArmDetector:
    """GroundingDINO zero-shot hand/arm detector via transformers (same as E20-B)."""

    def __init__(self, device: str):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        model_id = os.environ.get("E20_DINO_MODEL", "IDEA-Research/grounding-dino-tiny")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.device = device
        self.torch = torch

    def detect(self, image_bgr: np.ndarray, threshold: float) -> list[tuple[list[float], float]]:
        pil = _pil_from_bgr(image_bgr)
        inputs = self.processor(images=pil, text=DINO_PROMPT, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        target_size = [(image_bgr.shape[0], image_bgr.shape[1])]
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=threshold, text_threshold=0.2, target_sizes=target_size
            )
        except TypeError:
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=threshold, text_threshold=0.2, target_sizes=target_size
            )
        out = []
        for box, score in zip(results[0]["boxes"], results[0]["scores"]):
            out.append(([float(v) for v in box.tolist()], float(score)))
        return out


def gather_boxes(
    sample: dict[str, Any],
    image_bgr: np.ndarray,
    detector: HandArmDetector | None,
) -> tuple[list[list[float]], dict[str, int]]:
    """Merge box sources in priority order: HOA ref-frame, DINO, HOA low/neighbor.

    E20-C change: very large DINO boxes (>0.30 of the frame) additionally need
    skin evidence - E20-B let a frame-wide "arm" fire cover the freezer door in
    subaction_17.
    """
    h, w = image_bgr.shape[:2]
    primary = list(sample.get("hand_bboxes_px") or [])
    low = list(sample.get("hand_bboxes_low_px") or [])
    neighbor = [_expand_box(b, 12.0, (h, w)) for b in (sample.get("neighbor_hand_bboxes_px") or [])]
    dino: list[list[float]] = []
    if detector is not None:
        dino = [box for box, _score in detector.detect(image_bgr, DINO_DETECT_THRESHOLD)]
    sources = {"primary": 0, "dino": 0, "low": 0, "neighbor": 0, "dino_skin_dropped": 0}

    merged: list[list[float]] = []
    for name, boxes, skin_gate in (
        ("primary", primary, False),
        ("dino", dino, False),
        ("low", low, True),
        ("neighbor", neighbor, True),
    ):
        for box in boxes:
            area_frac = max(0.0, (box[2] - box[0]) * (box[3] - box[1])) / float(h * w)
            if area_frac > 0.70 or area_frac <= 0.0:
                continue
            if skin_gate and skin_ratio_in_bbox(image_bgr, box) < BOX_SKIN_EVIDENCE_MIN:
                continue
            if name == "dino" and area_frac > LARGE_DINO_BOX_AREA and skin_ratio_in_bbox(image_bgr, box) < LARGE_DINO_SKIN_MIN:
                sources["dino_skin_dropped"] += 1
                continue
            before = len(merged)
            merged = merge_overlapping_boxes(merged + [box])
            if len(merged) > before:
                sources[name] += 1
    return merged, sources


class SamBoxMasker:
    def __init__(self, device: str):
        from segment_anything import SamPredictor, sam_model_registry

        ckpt = Path(os.environ.get("SAM_CHECKPOINT", "/root/workspace/vrb/models/sam_vit_h_4b8939.pth"))
        if not ckpt.exists():
            ckpt = Path("/root/workspace/vrb/sam_vit_h_4b8939.pth")
        if not ckpt.exists():
            raise FileNotFoundError("SAM checkpoint missing; set SAM_CHECKPOINT")
        sam = sam_model_registry["vit_h"](checkpoint=str(ckpt))
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.device = device
        self.checkpoint = ckpt

    def masks(self, image_bgr: np.ndarray, bboxes: list[list[float]]) -> tuple[np.ndarray, list[float]]:
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        combined = np.zeros((h, w), dtype=np.uint8)
        scores_out: list[float] = []
        for bbox in bboxes:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            box = np.array(
                [max(0, x1 - 8), max(0, y1 - 8), min(w - 1, x2 + 8), min(h - 1, y2 + 8)],
                dtype=np.float32,
            )
            masks, scores, _ = self.predictor.predict(box=box, multimask_output=True)
            best = int(np.argmax(scores))
            mask = masks[best].astype(np.uint8) * 255
            combined = cv2.bitwise_or(combined, mask)
            scores_out.append(float(scores[best]))
        return combined, scores_out

    def object_candidates(self, image_bgr: np.ndarray, bbox: list[float]) -> list[np.ndarray]:
        """All three SAM multimask outputs for a box prompt (part -> whole)."""
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        x1, y1, x2, y2 = [float(v) for v in bbox]
        box = np.array([max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)], dtype=np.float32)
        masks, scores, _ = self.predictor.predict(box=box, multimask_output=True)
        order = np.argsort(-scores)
        return [masks[i].astype(np.uint8) * 255 for i in order]


def build_protect_mask(
    image_bgr: np.ndarray,
    sample: dict[str, Any],
    masker: SamBoxMasker | None,
    hand_raw_mask: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Protect mask for the interaction target from exported label geometry.

    heatmap_top_bbox_px marks where the affordance labels will be redrawn; the
    redrawn labels must land on the object, so that region (plus the SAM object
    segmentation it prompts, plus contact-point disks) may never be inpainted.
    """
    h, w = image_bgr.shape[:2]
    info: dict[str, Any] = {"protect_source": "none", "protect_sam_valid": False}
    hm_box = sample.get("heatmap_top_bbox_px")
    contacts = sample.get("contact_points_px") or []
    protect = disk_mask((h, w), contacts, radius=CONTACT_RADIUS)
    anchor = sample.get("contact_anchor_xy")
    if anchor:
        protect = cv2.bitwise_or(protect, disk_mask((h, w), [anchor], radius=CONTACT_RADIUS))
    if hm_box:
        rect = _rect_mask((h, w), hm_box, pad=PROTECT_HM_PAD)
        protect = cv2.bitwise_or(protect, rect)
        info["protect_source"] = "bbox"
        if masker is not None:
            # SAM object segmentation prompted by the heatmap-core bbox: protect
            # as much of the target object as we can identify, not just the bbox.
            best: np.ndarray | None = None
            best_area = -1.0
            for cand in masker.object_candidates(image_bgr, _expand_box([float(v) for v in hm_box], 4.0, (h, w))):
                area_frac = float(np.count_nonzero(cand) / cand.size)
                if area_frac > PROTECT_AREA_MAX:
                    continue
                if mask_overlap_ratio(cand, hm_box) < PROTECT_MIN_HM_COVER:
                    continue
                if hand_raw_mask is not None and mask_intersection_frac(cand, hand_raw_mask) > PROTECT_HAND_OVERLAP_MAX:
                    # SAM latched onto the grasping hand instead of the object
                    continue
                if area_frac > best_area:
                    best, best_area = cand, area_frac
            if best is not None:
                protect = cv2.bitwise_or(protect, best)
                info["protect_source"] = "sam+bbox"
                info["protect_sam_valid"] = True
    info["protect_area_ratio"] = float(np.count_nonzero(protect) / protect.size)
    return protect, info


def build_mask(
    image_bgr: np.ndarray,
    boxes: list[list[float]],
    masker: SamBoxMasker | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Returns (expanded mask BEFORE protection, raw SAM mask, info)."""
    h, w = image_bgr.shape[:2]
    info: dict[str, Any] = {"sam_scores": [], "capped": False}
    if not boxes:
        empty = np.zeros((h, w), dtype=np.uint8)
        return empty, empty.copy(), info
    if masker is None:
        sam_mask = np.zeros((h, w), dtype=np.uint8)
        for box in boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            sam_mask[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = 255
    else:
        sam_mask, scores = masker.masks(image_bgr, boxes)
        info["sam_scores"] = scores
    mask = sam_mask
    for box in boxes:
        mask = extend_mask_to_border(mask, box, (h, w), max_extend_frac=0.6, width_scale=1.4)
    mask = bridge_components_to_border(mask, (h, w), width_scale=BRIDGE_WIDTH_SCALE, max_extend_frac=0.6)
    mask = fill_mask_holes(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    mask = dilate_mask(mask, pad_px=MASK_DILATE_PX)
    area = float(np.count_nonzero(mask) / mask.size)
    if area > MASK_AREA_CAP:
        info["capped"] = True
        tight = fill_mask_holes(sam_mask)
        tight = dilate_mask(tight, pad_px=6)
        if float(np.count_nonzero(tight) / tight.size) < area:
            mask = tight
    return mask, sam_mask, info


def subtract_protect(mask_u8: np.ndarray, protect_u8: np.ndarray) -> np.ndarray:
    out = ((mask_u8 > 0).astype(np.uint8) * 255).copy()
    out[protect_u8 > 0] = 0
    return out


def redetect_hands(
    detector: HandArmDetector | None,
    edited_bgr: np.ndarray,
    mask_u8: np.ndarray,
    threshold: float = DINO_REDETECT_THRESHOLD,
) -> tuple[float, list[list[float]], float]:
    """Skin-confirmed hand/arm score near the inpainted region.

    Returns (skin-confirmed max score, offending boxes, unconfirmed max score).
    E20-B counted bare DINO fires and failed 3 clean samples on trouser legs and
    knees; skin evidence is required IN ADDITION (never alone - wood passes skin).
    """
    if detector is None:
        return 0.0, [], 0.0
    h, w = edited_bgr.shape[:2]
    vicinity = dilate_mask(mask_u8, pad_px=24) > 0
    max_score = 0.0
    max_raw = 0.0
    boxes_out: list[list[float]] = []
    for box, score in detector.detect(edited_bgr, threshold):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) * (y2 - y1) / float(h * w) > 0.70:
            continue
        if not bool(vicinity[y1:y2, x1:x2].any()):
            continue
        max_raw = max(max_raw, float(score))
        if skin_ratio_in_bbox(edited_bgr, [float(x1), float(y1), float(x2), float(y2)]) < REDETECT_SKIN_MIN:
            continue
        max_score = max(max_score, float(score))
        boxes_out.append([float(x1), float(y1), float(x2), float(y2)])
    return max_score, boxes_out, max_raw


class Backends:
    def __init__(self, device: str):
        self.device = device
        self._lama = None
        self._sd = None
        self._img2img = None

    def lama(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self._lama is None:
            from simple_lama_inpainting import SimpleLama

            self._lama = SimpleLama()
        out = self._lama(_pil_from_bgr(image_bgr), Image.fromarray((mask > 0).astype(np.uint8) * 255))
        return composite_inside_mask(image_bgr, _bgr_from_pil(out), mask, feather_px=3)

    def sd(self, image_bgr: np.ndarray, mask: np.ndarray, seed: int) -> np.ndarray:
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        if self._sd is None:
            model_id = os.environ.get("E20_SD_INPAINT_MODEL", "runwayml/stable-diffusion-inpainting")
            load_kwargs: dict[str, Any] = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "safety_checker": None,
                "use_safetensors": True,
            }
            variant = os.environ.get("E20_DIFFUSERS_VARIANT")
            if variant:
                load_kwargs["variant"] = variant
            self._sd = StableDiffusionInpaintPipeline.from_pretrained(model_id, **load_kwargs).to(self.device)
            self._sd.enable_attention_slicing()
        h, w = image_bgr.shape[:2]
        up_w, up_h = w * UPSCALE, h * UPSCALE
        img_up = cv2.resize(image_bgr, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4)
        mask_up = cv2.resize((mask > 0).astype(np.uint8) * 255, (up_w, up_h), interpolation=cv2.INTER_NEAREST)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        out = self._sd(
            prompt="an empty kitchen counter, clean background, natural indoor scene, photorealistic",
            negative_prompt=(
                "hand, hands, fingers, arm, wrist, person, human, skin, glove, "
                "new object, extra object, text, watermark, blurry, deformed"
            ),
            image=_pil_from_bgr(img_up),
            mask_image=Image.fromarray(mask_up),
            width=up_w,
            height=up_h,
            num_inference_steps=40,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]
        edited = cv2.resize(_bgr_from_pil(out), (w, h), interpolation=cv2.INTER_AREA)
        return composite_inside_mask(image_bgr, edited, mask, feather_px=3)

    def sdedit_masked(self, image_bgr: np.ndarray, mask: np.ndarray, seed: int, protect: np.ndarray | None = None) -> np.ndarray:
        """SDEdit harmonization at 2x, composited only inside the (padded) mask.

        E20-C: the composite region additionally excludes the protect mask so
        the VAE roundtrip can never touch the target object's pixels.
        """
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline

        if self._img2img is None:
            model_id = os.environ.get("E20_SDEDIT_MODEL", "runwayml/stable-diffusion-v1-5")
            load_kwargs: dict[str, Any] = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "safety_checker": None,
                "use_safetensors": True,
            }
            variant = os.environ.get("E20_DIFFUSERS_VARIANT")
            if variant:
                load_kwargs["variant"] = variant
            self._img2img = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, **load_kwargs).to(self.device)
            self._img2img.enable_attention_slicing()
        h, w = image_bgr.shape[:2]
        up = cv2.resize(image_bgr, (w * UPSCALE, h * UPSCALE), interpolation=cv2.INTER_LANCZOS4)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        out = self._img2img(
            prompt="photo of a kitchen scene, sharp focus, high detail, photorealistic",
            negative_prompt="hand, arm, skin, blurry, lowres, deformed",
            image=_pil_from_bgr(up),
            strength=0.15,
            num_inference_steps=30,
            generator=generator,
        ).images[0]
        down = cv2.resize(_bgr_from_pil(out), (w, h), interpolation=cv2.INTER_AREA)
        region = dilate_mask(mask, pad_px=4)
        if protect is not None:
            region = subtract_protect(region, protect)
        return composite_inside_mask(image_bgr, down, region, feather_px=3)


def _qc_backend(original: np.ndarray, edited: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    return {
        "skin_ratio_after": skin_ratio_in_mask(edited, mask, erode_px=3),
        "residual_skin_after": dual_skin_ratio_in_mask(edited, mask, erode_px=2),
        "boundary_gradient_ratio": boundary_gradient_ratio(original, edited, mask, band_px=4),
        "sharpness_in_mask": laplacian_variance(edited, mask),
        "sharpness_ratio_vs_original": sharpness_ratio_vs_original(edited, original, mask),
        "mask_outside_exact": bool(np.array_equal(original[mask == 0], edited[mask == 0])),
    }


def process_sample(
    sample: dict[str, Any],
    job_dir: Path,
    out_dir: Path,
    args,
    masker: SamBoxMasker | None,
    detector: HandArmDetector | None,
    backends: Backends,
) -> dict[str, Any]:
    started = time.time()
    key = sample["sample_key"]
    image = cv2.imread(str(job_dir / sample["image"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(job_dir / sample["image"])
    h, w = image.shape[:2]
    seed = int(args.seed) + int(sample.get("subaction_index", 0))

    sample_out = out_dir / key
    sample_out.mkdir(parents=True, exist_ok=True)

    boxes, box_sources = gather_boxes(sample, image, detector)
    mask_raw, sam_raw, mask_info = build_mask(image, boxes, masker)
    protect, protect_info = build_protect_mask(image, sample, masker, sam_raw if boxes else None)
    # Hard constraint: the label region may never be inpainted. When the hand
    # is grasping the object the conflicting pixels stay untouched (a partial
    # hand next to the object beats labels pointing at hallucinated floor).
    protect_conflict = mask_intersection_frac(protect, mask_raw)
    mask = subtract_protect(mask_raw, protect)
    already_clean = False
    if not boxes:
        already_clean = dual_skin_ratio_in_mask(image, np.zeros((h, w), dtype=np.uint8), erode_px=0) < 0.02

    qc: dict[str, Any] = {
        "sample_key": key,
        "worker": "e20c",
        "already_clean": already_clean,
        "hand_box_count": len(boxes),
        "box_sources": box_sources,
        "sam_scores": mask_info.get("sam_scores", []),
        "mask_capped": bool(mask_info.get("capped")),
        "inpaint_rounds": 0,
        "passed": {},
        "skin_ratio_after": {},
        "redetect_score_after": {},
        "redetect_raw_after": {},
        "boundary_gradient_ratio": {},
        "sharpness": {},
        "sharpness_ratio_vs_original": {},
        "sd_final_eligible": None,
        "sdedit_sharpness_ratio": {},
        "backend_rank": [],
        "elapsed_sec": None,
        **protect_info,
        "protect_conflict_frac": float(protect_conflict),
        "label_region_conflict": bool(protect_conflict > LABEL_CONFLICT_FRAC),
        "mask_area_ratio_raw": float(np.count_nonzero(mask_raw > 0) / mask_raw.size),
        "overlap_heatmap_top_raw": mask_overlap_ratio(mask_raw, sample.get("heatmap_top_bbox_px")),
    }

    lama_img = None
    if boxes and int(np.count_nonzero(mask)) > 0:
        # LaMa-driven re-inpaint loop as in E20-B, but every expansion is
        # skin-confirmed and re-clipped against the protect mask.
        for round_idx in range(MAX_INPAINT_ROUNDS):
            try:
                lama_img = backends.lama(image, mask)
            except Exception as exc:
                qc["lama_error"] = repr(exc)
                break
            qc["inpaint_rounds"] = round_idx + 1
            score, offend_boxes, _raw = redetect_hands(detector, lama_img, mask)
            if score < REDETECT_PASS_SCORE or not offend_boxes or round_idx == MAX_INPAINT_ROUNDS - 1:
                break
            extra, _scores = (
                masker.masks(image, offend_boxes) if masker is not None else (np.zeros((h, w), dtype=np.uint8), [])
            )
            if masker is None:
                for box in offend_boxes:
                    x1, y1, x2, y2 = [int(round(v)) for v in box]
                    extra[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = 255
            new_mask = cv2.bitwise_or(mask, dilate_mask(extra, pad_px=10))
            new_mask = bridge_components_to_border(new_mask, (h, w), width_scale=BRIDGE_WIDTH_SCALE, max_extend_frac=0.6)
            new_mask = fill_mask_holes(new_mask)
            new_mask = subtract_protect(new_mask, protect)
            if float(np.count_nonzero(new_mask) / new_mask.size) > MASK_AREA_CAP or np.array_equal(new_mask, mask):
                break
            mask = new_mask

    cv2.imwrite(str(sample_out / "mask.png"), mask)
    cv2.imwrite(str(sample_out / "protect.png"), protect)
    mask_area_ratio = float(np.count_nonzero(mask > 0) / mask.size)
    qc["mask_area_ratio"] = mask_area_ratio
    qc["mask_touches_bottom"] = bool(np.any(mask[-1] > 0))
    qc["overlap_heatmap_top"] = mask_overlap_ratio(mask, sample.get("heatmap_top_bbox_px"))
    qc["protect_intact_final"] = bool(mask_intersection_frac(protect, mask) == 0.0)
    qc["overlap_crop150"] = mask_overlap_ratio(mask, sample.get("crop_bbox_150"))
    redetect_before, _boxes_b, redetect_before_raw = redetect_hands(detector, image, mask)
    qc["redetect_score_before"] = float(redetect_before)
    qc["redetect_raw_before"] = float(redetect_before_raw)

    candidates: dict[str, np.ndarray] = {}
    if lama_img is not None:
        candidates["lama"] = lama_img
    if boxes and int(np.count_nonzero(mask)) > 0 and "sd" in args.backends:
        try:
            candidates["sd"] = backends.sd(image, mask, seed=seed)
        except Exception as exc:
            qc["sd_error"] = repr(exc)

    metrics: dict[str, dict[str, float]] = {}
    for name, edited in candidates.items():
        cv2.imwrite(str(sample_out / f"{name}.png"), edited)
        q = _qc_backend(image, edited, mask)
        redetect_score, _boxes_a, redetect_raw = redetect_hands(detector, edited, mask)
        q["redetect_score_after"] = redetect_score
        metrics[name] = q
        passed = (
            bool(q["mask_outside_exact"])
            and (redetect_score < REDETECT_PASS_SCORE or redetect_score <= 0.55 * redetect_before)
            and mask_area_ratio <= MASK_AREA_PASS
        )
        qc["passed"][name] = bool(passed)
        qc["skin_ratio_after"][name] = float(q["skin_ratio_after"])
        qc["redetect_score_after"][name] = float(redetect_score)
        qc["redetect_raw_after"][name] = float(redetect_raw)
        qc["boundary_gradient_ratio"][name] = float(q["boundary_gradient_ratio"])
        qc["sharpness"][name] = float(q["sharpness_in_mask"])
        qc["sharpness_ratio_vs_original"][name] = float(q["sharpness_ratio_vs_original"])
        if args.sdedit:
            try:
                enhanced = backends.sdedit_masked(edited, mask, seed=seed + 1000, protect=protect)
                cv2.imwrite(str(sample_out / f"sdedit_{name}.png"), enhanced)
                base_sharp = laplacian_variance(edited, mask)
                enh_sharp = laplacian_variance(enhanced, mask)
                qc["sdedit_sharpness_ratio"][name] = float(enh_sharp / max(base_sharp, 1e-6))
            except Exception as exc:
                qc[f"sdedit_{name}_error"] = repr(exc)

    # SD hallucination gate: calibrated on all 56 E20-B samples, every visually
    # hallucinated SD final had in-mask sharpness > 2x the ORIGINAL frame.
    # Large holes additionally bar SD from final (it invents whole new scenes).
    sd_eligible = None
    if "sd" in metrics:
        sd_eligible = bool(
            metrics["sd"]["sharpness_ratio_vs_original"] <= SD_HALLU_SHARP_RATIO
            and mask_area_ratio <= SD_FINAL_MAX_AREA
        )
    qc["sd_final_eligible"] = sd_eligible

    # hallucination gate outranks the redetect pass: a hallucinated-but-hand-free
    # SD must never beat a clean-texture LaMa that still trips the re-detector
    # (subaction_08: SD invented a bowl of food, LaMa merely left a leg nearby).
    order = {"lama": 0, "sd": 1}
    qc["backend_rank"] = sorted(
        metrics.keys(),
        key=lambda n: (
            1 if (n == "sd" and sd_eligible is False) else 0,
            0 if qc["passed"].get(n) else 1,
            round(float(metrics[n]["redetect_score_after"]), 1),
            round(float(metrics[n]["boundary_gradient_ratio"]), 2),
            order.get(n, 9),
        ),
    )
    for name in ("lama", "sd"):
        if name not in qc["passed"] and not already_clean:
            cv2.imwrite(str(sample_out / f"{name}.png"), image)
            qc["passed"][name] = False
    qc["elapsed_sec"] = float(time.time() - started)
    _json_write(sample_out / "qc.json", qc)
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E20-C GPU inpainting job.")
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backends", default="lama,sd")
    parser.add_argument("--sdedit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--no-sam", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job_dir = Path(args.job_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = json.loads((job_dir / "tasks.json").read_text(encoding="utf-8"))
    samples = tasks["samples"][: None if int(args.limit) < 0 else int(args.limit)]
    use_sam = not args.no_sam and args.device != "cpu"
    masker = SamBoxMasker(args.device) if use_sam else None
    try:
        detector = HandArmDetector(args.device)
    except Exception as exc:
        print(f"E20C_PROGRESS stage=gpu warn=dino_unavailable error={exc!r}", flush=True)
        detector = None
    backends = Backends(args.device)
    summary = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "worker": "e20c",
        "seed": int(args.seed),
        "device": args.device,
        "backends": args.backends,
        "dino_available": detector is not None,
        "sam_checkpoint_sha8": _sha8(masker.checkpoint) if masker is not None else None,
        "samples": [],
    }
    for idx, sample in enumerate(samples, start=1):
        qc = process_sample(sample, job_dir, out_dir, args, masker, detector, backends)
        summary["samples"].append(qc)
        print(
            f"E20C_PROGRESS stage=gpu sample={idx}/{len(samples)} key={sample['sample_key']} "
            f"boxes={qc.get('hand_box_count')} rounds={qc.get('inpaint_rounds')} "
            f"area={qc.get('mask_area_ratio', 0.0):.4f} ov_hm={qc.get('overlap_heatmap_top', 0.0):.3f} "
            f"protect={qc.get('protect_source')} conflict={qc.get('protect_conflict_frac', 0.0):.2f} "
            f"sd_ok={qc.get('sd_final_eligible')} rank={qc.get('backend_rank')} "
            f"passed={qc.get('passed')} elapsed={qc.get('elapsed_sec'):.1f}s",
            flush=True,
        )
        _json_write(out_dir / "results_summary.json", summary)
    summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _json_write(out_dir / "results_summary.json", summary)


if __name__ == "__main__":
    main()
