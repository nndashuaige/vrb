#!/usr/bin/env python3
"""Self-contained E20-B GPU worker.

E20-B changes vs E20-A (see docs/learning_note/实验记录/实验记录e20-a.md §九-十一):
1. Hand boxes: ref-frame HOA @0.5 plus low-score @0.1 plus neighbor-frame (±6)
   fallback boxes exported by pack, plus GroundingDINO "hand/arm" text-prompted
   detection (arms have no HOA boxes at all; color-based skin growth was tested
   and rejected because EPIC wooden counters pass every skin-color rule).
2. Mask: SAM per box -> direction-aware border bridging -> hole fill -> dilation.
3. Inpaint: LaMa with DINO-redetect re-inpaint rounds (kills regrown arms),
   SD inpainting run at 2x resolution instead of 512 letterbox.
4. Enhancement: SDEdit at 2x resolution composited ONLY inside the mask, so the
   rest of the frame stays pixel-identical (no global VAE blur).
5. QC: hand/arm re-detection score on the inpainted image + seam gradient ratio
   + sharpness; deterministic backend ranking instead of a seeded 50/50 pick.
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
        dual_skin_ratio_in_mask,
        extend_mask_to_border,
        fill_mask_holes,
        laplacian_variance,
        mask_overlap_ratio,
        merge_overlapping_boxes,
        skin_ratio_in_bbox,
        skin_ratio_in_mask,
    )
except ModuleNotFoundError:
    from vrbreproduction.e20.inpaint_utils import (
        boundary_gradient_ratio,
        bridge_components_to_border,
        composite_inside_mask,
        dilate_mask,
        dual_skin_ratio_in_mask,
        extend_mask_to_border,
        fill_mask_holes,
        laplacian_variance,
        mask_overlap_ratio,
        merge_overlapping_boxes,
        skin_ratio_in_bbox,
        skin_ratio_in_mask,
    )


BOX_SKIN_EVIDENCE_MIN = 0.05
MASK_AREA_CAP = 0.55
MASK_AREA_PASS = 0.65
DINO_DETECT_THRESHOLD = 0.30
DINO_REDETECT_THRESHOLD = 0.25
REDETECT_PASS_SCORE = 0.35
MAX_INPAINT_ROUNDS = 3
UPSCALE = 2
DINO_PROMPT = "a human hand. a human arm."


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


def _mediapipe_bboxes(image_bgr: np.ndarray) -> list[list[float]]:
    try:
        import mediapipe as mp
    except Exception:
        return []
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
        return []
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_bgr.shape[:2]
    boxes = []
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.4) as hands:
        result = hands.process(rgb)
        for landmarks in result.multi_hand_landmarks or []:
            xs = [lm.x * w for lm in landmarks.landmark]
            ys = [lm.y * h for lm in landmarks.landmark]
            boxes.append([max(0.0, min(xs)), max(0.0, min(ys)), min(float(w), max(xs)), min(float(h), max(ys))])
    return boxes


def _expand_box(box: list[float], pad: float, image_hw: tuple[int, int]) -> list[float]:
    h, w = image_hw
    return [
        max(0.0, float(box[0]) - pad),
        max(0.0, float(box[1]) - pad),
        min(float(w), float(box[2]) + pad),
        min(float(h), float(box[3]) + pad),
    ]


class HandArmDetector:
    """GroundingDINO zero-shot hand/arm detector via transformers.

    HOA boxes only cover hands; arms crossing the frame (the worst E20-A failure
    mode) need text-prompted detection. Color-based skin proposals were rejected:
    EPIC wooden counters pass both YCrCb and HSV skin rules.
    """

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
    """Merge box sources in priority order: HOA ref-frame, DINO, HOA low/neighbor."""
    h, w = image_bgr.shape[:2]
    primary = list(sample.get("hand_bboxes_px") or [])
    low = list(sample.get("hand_bboxes_low_px") or [])
    neighbor = [_expand_box(b, 12.0, (h, w)) for b in (sample.get("neighbor_hand_bboxes_px") or [])]
    dino: list[list[float]] = []
    if detector is not None:
        dino = [box for box, _score in detector.detect(image_bgr, DINO_DETECT_THRESHOLD)]
    sources = {"primary": 0, "dino": 0, "low": 0, "neighbor": 0, "mediapipe": 0}

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
            before = len(merged)
            merged = merge_overlapping_boxes(merged + [box])
            if len(merged) > before:
                sources[name] += 1
    if not merged:
        for box in _mediapipe_bboxes(image_bgr):
            merged = merge_overlapping_boxes(merged + [_expand_box(box, 8.0, (h, w))])
        sources["mediapipe"] = len(merged)
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


def build_mask(
    image_bgr: np.ndarray,
    boxes: list[list[float]],
    masker: SamBoxMasker | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = image_bgr.shape[:2]
    info: dict[str, Any] = {"sam_scores": [], "capped": False}
    if not boxes:
        return np.zeros((h, w), dtype=np.uint8), info
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
        mask = extend_mask_to_border(mask, box, (h, w), max_extend_frac=0.6, width_scale=1.5)
    mask = bridge_components_to_border(mask, (h, w), width_scale=1.4, max_extend_frac=0.6)
    mask = fill_mask_holes(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    mask = dilate_mask(mask, pad_px=14)
    area = float(np.count_nonzero(mask) / mask.size)
    if area > MASK_AREA_CAP:
        # over-large masks make SD hallucinate whole new scenes; retry tighter
        info["capped"] = True
        tight = fill_mask_holes(sam_mask)
        tight = dilate_mask(tight, pad_px=10)
        if float(np.count_nonzero(tight) / tight.size) < area:
            mask = tight
    return mask, info


def redetect_hands(
    detector: HandArmDetector | None,
    edited_bgr: np.ndarray,
    mask_u8: np.ndarray,
    threshold: float = DINO_REDETECT_THRESHOLD,
) -> tuple[float, list[list[float]]]:
    """Max hand/arm score near the inpainted region + the offending boxes.

    Replaces the E20-A skin-ratio QC that confused wooden counters with skin.
    """
    if detector is None:
        return 0.0, []
    h, w = edited_bgr.shape[:2]
    vicinity = dilate_mask(mask_u8, pad_px=24) > 0
    max_score = 0.0
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
        max_score = max(max_score, float(score))
        boxes_out.append([float(x1), float(y1), float(x2), float(y2)])
    return max_score, boxes_out


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

    def sdedit_masked(self, image_bgr: np.ndarray, mask: np.ndarray, seed: int) -> np.ndarray:
        """SDEdit harmonization at 2x, composited only inside the (padded) mask."""
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
        return composite_inside_mask(image_bgr, down, dilate_mask(mask, pad_px=6), feather_px=5)


def _qc_backend(original: np.ndarray, edited: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    return {
        "skin_ratio_after": skin_ratio_in_mask(edited, mask, erode_px=3),
        "residual_skin_after": dual_skin_ratio_in_mask(edited, mask, erode_px=2),
        "boundary_gradient_ratio": boundary_gradient_ratio(original, edited, mask, band_px=4),
        "sharpness_in_mask": laplacian_variance(edited, mask),
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
    mask, mask_info = build_mask(image, boxes, masker)
    already_clean = False
    if not boxes:
        already_clean = dual_skin_ratio_in_mask(image, np.zeros((h, w), dtype=np.uint8), erode_px=0) < 0.02

    qc: dict[str, Any] = {
        "sample_key": key,
        "worker": "e20b",
        "already_clean": already_clean,
        "hand_box_count": len(boxes),
        "box_sources": box_sources,
        "sam_scores": mask_info.get("sam_scores", []),
        "mask_capped": bool(mask_info.get("capped")),
        "inpaint_rounds": 0,
        "passed": {},
        "skin_ratio_after": {},
        "redetect_score_after": {},
        "boundary_delta": {},
        "boundary_gradient_ratio": {},
        "sharpness": {},
        "sdedit_sharpness_ratio": {},
        "backend_rank": [],
        "elapsed_sec": None,
    }

    lama_img = None
    if boxes and int(np.count_nonzero(mask)) > 0:
        # LaMa-driven re-inpaint loop: if the hand/arm detector still fires on the
        # result (LaMa regrew the arm or the mask missed part of it), segment the
        # offending boxes, expand the mask and retry; SD then reuses the converged mask.
        for round_idx in range(MAX_INPAINT_ROUNDS):
            try:
                lama_img = backends.lama(image, mask)
            except Exception as exc:
                qc["lama_error"] = repr(exc)
                break
            qc["inpaint_rounds"] = round_idx + 1
            score, offend_boxes = redetect_hands(detector, lama_img, mask)
            if score < REDETECT_PASS_SCORE or not offend_boxes or round_idx == MAX_INPAINT_ROUNDS - 1:
                break
            extra, _scores = (
                masker.masks(image, offend_boxes) if masker is not None else (np.zeros((h, w), dtype=np.uint8), [])
            )
            if masker is None:
                for box in offend_boxes:
                    x1, y1, x2, y2 = [int(round(v)) for v in box]
                    extra[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = 255
            new_mask = cv2.bitwise_or(mask, dilate_mask(extra, pad_px=12))
            new_mask = bridge_components_to_border(new_mask, (h, w), width_scale=1.4, max_extend_frac=0.6)
            new_mask = fill_mask_holes(new_mask)
            if float(np.count_nonzero(new_mask) / new_mask.size) > MASK_AREA_CAP or np.array_equal(new_mask, mask):
                break
            mask = new_mask

    cv2.imwrite(str(sample_out / "mask.png"), mask)
    mask_area_ratio = float(np.count_nonzero(mask > 0) / mask.size)
    qc["mask_area_ratio"] = mask_area_ratio
    qc["mask_touches_bottom"] = bool(np.any(mask[-1] > 0))
    qc["overlap_heatmap_top"] = mask_overlap_ratio(mask, sample.get("heatmap_top_bbox_px"))
    qc["overlap_crop150"] = mask_overlap_ratio(mask, sample.get("crop_bbox_150"))
    # DINO baseline near the mask on the ORIGINAL frame; kitchens produce mid-score
    # false fires (bag straps, hoses), so the pass rule below is partly relative.
    redetect_before, _ = redetect_hands(detector, image, mask)
    qc["redetect_score_before"] = float(redetect_before)

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
        redetect_score, _ = redetect_hands(detector, edited, mask)
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
        qc["boundary_delta"][name] = float(q["boundary_gradient_ratio"])
        qc["boundary_gradient_ratio"][name] = float(q["boundary_gradient_ratio"])
        qc["sharpness"][name] = float(q["sharpness_in_mask"])
        if args.sdedit:
            try:
                enhanced = backends.sdedit_masked(edited, mask, seed=seed + 1000)
                cv2.imwrite(str(sample_out / f"sdedit_{name}.png"), enhanced)
                base_sharp = laplacian_variance(edited, mask)
                enh_sharp = laplacian_variance(enhanced, mask)
                qc["sdedit_sharpness_ratio"][name] = float(enh_sharp / max(base_sharp, 1e-6))
            except Exception as exc:
                qc[f"sdedit_{name}_error"] = repr(exc)

    # redetect bucketed to 0.1 so near-ties fall through to seam quality, where
    # LaMa (texture continuation) usually beats SD (hallucination-prone).
    order = {"lama": 0, "sd": 1}
    qc["backend_rank"] = sorted(
        metrics.keys(),
        key=lambda n: (
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
    parser = argparse.ArgumentParser(description="Run E20-B GPU inpainting job.")
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
        print(f"E20B_PROGRESS stage=gpu warn=dino_unavailable error={exc!r}", flush=True)
        detector = None
    backends = Backends(args.device)
    summary = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "worker": "e20b",
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
            f"E20B_PROGRESS stage=gpu sample={idx}/{len(samples)} key={sample['sample_key']} "
            f"boxes={qc.get('hand_box_count')} rounds={qc.get('inpaint_rounds')} "
            f"area={qc.get('mask_area_ratio', 0.0):.4f} rank={qc.get('backend_rank')} "
            f"passed={qc.get('passed')} elapsed={qc.get('elapsed_sec'):.1f}s",
            flush=True,
        )
        _json_write(out_dir / "results_summary.json", summary)
    summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _json_write(out_dir / "results_summary.json", summary)


if __name__ == "__main__":
    main()
