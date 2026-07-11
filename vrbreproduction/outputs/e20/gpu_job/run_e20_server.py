#!/usr/bin/env python3
"""Self-contained E20 GPU worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

try:
    from e20_inpaint_utils import (
        bbox_mask,
        composite_inside_mask,
        dilate_mask,
        extend_mask_to_border,
        letterbox_pad,
        mask_overlap_ratio,
        skin_ratio_in_mask,
        unletterbox,
    )
except ModuleNotFoundError:
    from vrbreproduction.e20.inpaint_utils import (
        bbox_mask,
        composite_inside_mask,
        dilate_mask,
        extend_mask_to_border,
        letterbox_pad,
        mask_overlap_ratio,
        skin_ratio_in_mask,
        unletterbox,
    )


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
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_bgr.shape[:2]
    boxes = []
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        result = hands.process(rgb)
        for landmarks in result.multi_hand_landmarks or []:
            xs = [lm.x * w for lm in landmarks.landmark]
            ys = [lm.y * h for lm in landmarks.landmark]
            boxes.append([max(0.0, min(xs)), max(0.0, min(ys)), min(float(w), max(xs)), min(float(h), max(ys))])
    return boxes


class SamBoxMasker:
    def __init__(self, device: str):
        from segment_anything import SamPredictor, sam_model_registry

        ckpt = Path(os.environ.get("SAM_CHECKPOINT", "/root/workspace/vrb/sam_vit_h_4b8939.pth"))
        if not ckpt.exists():
            ckpt = Path("/root/sam_vit_h_4b8939.pth")
        if not ckpt.exists():
            raise FileNotFoundError("SAM checkpoint missing; set SAM_CHECKPOINT")
        sam = sam_model_registry["vit_h"](checkpoint=str(ckpt))
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.device = device
        self.checkpoint = ckpt

    def mask(self, image_bgr: np.ndarray, bboxes: list[list[float]]) -> tuple[np.ndarray, list[float]]:
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
            mask = (masks[best].astype(np.uint8) * 255)
            mask = extend_mask_to_border(mask, bbox, (h, w))
            combined = cv2.bitwise_or(combined, mask)
            scores_out.append(float(scores[best]))
        return combined, scores_out


class Backends:
    def __init__(self, device: str, enable_lama: bool, enable_sd: bool, enable_sdedit: bool):
        self.device = device
        self._lama = None
        self._sd = None
        self._img2img = None
        self.enable_lama = enable_lama
        self.enable_sd = enable_sd
        self.enable_sdedit = enable_sdedit

    def telea(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if int(np.count_nonzero(mask)) == 0:
            return image_bgr.copy()
        edited = cv2.inpaint(image_bgr, (mask > 0).astype(np.uint8) * 255, 3.0, cv2.INPAINT_TELEA)
        return composite_inside_mask(image_bgr, edited, mask, feather_px=3)

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
            model_id = os.environ.get("E20_SD_INPAINT_MODEL", "stabilityai/stable-diffusion-2-inpainting")
            self._sd = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
            ).to(self.device)
        img_pad, mask_pad, meta = letterbox_pad(image_bgr, mask, target=512)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        out = self._sd(
            prompt="a photo of a kitchen, natural indoor scene, high quality",
            negative_prompt="hand, hands, fingers, arm, wrist, person, human, skin, glove, text, watermark",
            image=_pil_from_bgr(img_pad),
            mask_image=Image.fromarray(mask_pad),
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]
        edited = unletterbox(_bgr_from_pil(out), meta)
        return composite_inside_mask(image_bgr, edited, mask, feather_px=3)

    def sdedit(self, image_bgr: np.ndarray, seed: int) -> np.ndarray:
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline

        if self._img2img is None:
            model_id = os.environ.get("E20_SDEDIT_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
            self._img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
            ).to(self.device)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        out = self._img2img(
            prompt="high resolution, 4k",
            image=_pil_from_bgr(image_bgr),
            strength=0.1,
            num_inference_steps=50,
            generator=generator,
        ).images[0]
        return _bgr_from_pil(out)


def _qc_backend(original: np.ndarray, edited: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    return {
        "skin_ratio_after": skin_ratio_in_mask(edited, mask, erode_px=3),
        "boundary_delta": _compute_boundary_delta(original, edited, mask, band_px=5),
        "mask_outside_exact": bool(np.array_equal(original[mask == 0], edited[mask == 0])),
    }


def _compute_boundary_delta(original: np.ndarray, edited: np.ndarray, mask: np.ndarray, band_px: int = 5) -> float:
    if original.shape != edited.shape:
        return float("inf")
    mask01 = (mask > 0).astype(np.uint8)
    if int(mask01.sum()) == 0:
        return 0.0
    band = int(max(1, band_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    dilated = cv2.dilate(mask01, kernel, iterations=1)
    ring = (dilated > 0) & (mask01 == 0)
    if not bool(ring.any()):
        return 0.0
    diff = np.abs(original.astype(np.float32) - edited.astype(np.float32))
    return float(diff[ring].mean())


def process_sample(sample: dict[str, Any], job_dir: Path, out_dir: Path, args, masker: SamBoxMasker | None, backends: Backends) -> dict[str, Any]:
    started = time.time()
    key = sample["sample_key"]
    image = cv2.imread(str(job_dir / sample["image"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(job_dir / sample["image"])
    h, w = image.shape[:2]
    boxes = list(sample.get("hand_bboxes_px") or [])
    mediapipe_used = False
    if not boxes:
        boxes = _mediapipe_bboxes(image)
        mediapipe_used = bool(boxes)

    sample_out = out_dir / key
    sample_out.mkdir(parents=True, exist_ok=True)
    already_clean = False
    sam_scores: list[float] = []
    if boxes:
        if masker is None:
            mask = bbox_mask((h, w), boxes, pad_px=0)
        else:
            mask, sam_scores = masker.mask(image, boxes)
        mask = dilate_mask(mask, pad_px=12)
        if float(np.count_nonzero(mask) / mask.size) > 0.35:
            mask = bbox_mask((h, w), boxes, pad_px=12)
            mask_fallback_bbox = True
        else:
            mask_fallback_bbox = False
    else:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask_fallback_bbox = False
        already_clean = skin_ratio_in_mask(image, mask, erode_px=0) < 0.02

    cv2.imwrite(str(sample_out / "mask.png"), mask)
    mask_area_ratio = float(np.count_nonzero(mask > 0) / mask.size)
    overlap_heatmap = mask_overlap_ratio(mask, sample.get("heatmap_top_bbox_px"))
    overlap_crop = mask_overlap_ratio(mask, sample.get("crop_bbox_150"))
    qc: dict[str, Any] = {
        "sample_key": key,
        "already_clean": already_clean,
        "mediapipe_fallback_used": mediapipe_used,
        "hand_box_count": len(boxes),
        "sam_scores": sam_scores,
        "mask_area_ratio": mask_area_ratio,
        "mask_fallback_bbox": mask_fallback_bbox,
        "mask_touches_bottom": bool(np.any(mask[-1] > 0)),
        "overlap_heatmap_top": overlap_heatmap,
        "overlap_crop150": overlap_crop,
        "passed": {},
        "skin_ratio_after": {},
        "boundary_delta": {},
        "elapsed_sec": None,
    }

    backend_names = [b.strip() for b in args.backends.split(",") if b.strip()]
    if already_clean or int(np.count_nonzero(mask)) == 0:
        backend_names = []
    for backend_name in backend_names:
        try:
            if backend_name == "telea":
                edited = backends.telea(image, mask)
                out_name = "lama"
            elif backend_name == "lama":
                edited = backends.lama(image, mask)
                out_name = "lama"
            elif backend_name == "sd":
                edited = backends.sd(image, mask, seed=int(args.seed) + int(sample.get("subaction_index", 0)))
                out_name = "sd"
            else:
                continue
            cv2.imwrite(str(sample_out / f"{out_name}.png"), edited)
            q = _qc_backend(image, edited, mask)
            passed = (
                q["skin_ratio_after"] <= 0.10
                and q["boundary_delta"] <= 35.0
                and mask_area_ratio <= 0.35
                and bool(q["mask_outside_exact"])
            )
            qc["passed"][out_name] = bool(passed)
            qc["skin_ratio_after"][out_name] = float(q["skin_ratio_after"])
            qc["boundary_delta"][out_name] = float(q["boundary_delta"])
            if args.sdedit and out_name in ("lama", "sd"):
                try:
                    cv2.imwrite(str(sample_out / f"sdedit_{out_name}.png"), backends.sdedit(edited, seed=int(args.seed) + 1000 + int(sample.get("subaction_index", 0))))
                except Exception as exc:
                    qc[f"sdedit_{out_name}_error"] = repr(exc)
        except Exception as exc:
            qc["passed"][backend_name] = False
            qc[f"{backend_name}_error"] = repr(exc)
    if "lama" not in qc["passed"] and not already_clean:
        cv2.imwrite(str(sample_out / "lama.png"), image)
        qc["passed"]["lama"] = False
    if "sd" not in qc["passed"] and "sd" in args.backends and not already_clean:
        cv2.imwrite(str(sample_out / "sd.png"), image)
        qc["passed"]["sd"] = False
    qc["elapsed_sec"] = float(time.time() - started)
    _json_write(sample_out / "qc.json", qc)
    return qc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E20 GPU inpainting job.")
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
    use_sam = not args.no_sam and args.device != "cpu" and args.backends != "telea"
    masker = SamBoxMasker(args.device) if use_sam else None
    backend_set = {b.strip() for b in args.backends.split(",")}
    backends = Backends(args.device, "lama" in backend_set, "sd" in backend_set, bool(args.sdedit))
    summary = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(args.seed),
        "device": args.device,
        "backends": args.backends,
        "sam_checkpoint_sha8": _sha8(masker.checkpoint) if masker is not None else None,
        "samples": [],
    }
    for idx, sample in enumerate(samples, start=1):
        qc = process_sample(sample, job_dir, out_dir, args, masker, backends)
        summary["samples"].append(qc)
        print(
            f"E20_PROGRESS stage=gpu sample={idx}/{len(samples)} key={sample['sample_key']} "
            f"passed={qc.get('passed')} area={qc.get('mask_area_ratio'):.4f} elapsed={qc.get('elapsed_sec'):.1f}s",
            flush=True,
        )
        _json_write(out_dir / "results_summary.json", summary)
    summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _json_write(out_dir / "results_summary.json", summary)


if __name__ == "__main__":
    main()
