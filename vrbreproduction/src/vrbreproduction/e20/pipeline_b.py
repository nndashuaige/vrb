"""E20-B pipeline: E20-A two-pass design with arm-aware masking and quality-ranked finals.

E20-B keeps the E20/E20-A boundary (export label geometry -> GPU inpaint ->
local redraw) and reuses the E20-A export verbatim (same reference frames, same
label geometry). Only the GPU job content and the final selection change:

- pack_b exports three tiers of hand boxes (ref @0.5, ref @0.1, neighbor ±6 @0.5)
  because 12/56 E20-A samples had zero boxes at the 0.5-only threshold;
- the GPU worker is server_worker_b.py (arm-aware masks, residual re-inpaint,
  2x SD, mask-composited SDEdit);
- ingest_b picks the final backend from the worker's deterministic quality rank
  instead of the seeded 50/50;
- report_b adds the before/after/enhanced contact sheet.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd

from vrbreproduction.e20 import pipeline as e20_pipeline
from vrbreproduction.e20.inpaint_utils import (
    _bbox_norm_to_px,
    dilate_mask,
    heatmap_top_bbox,
    merge_overlapping_boxes,
)
from vrbreproduction.e20.pipeline import (
    SEED,
    VRBREPRODUCTION_ROOT,
    _copy_sample_static,
    _json_write,
    _load_detections_cached,
    _read_bgr,
)

DEFAULT_A_ROOT = VRBREPRODUCTION_ROOT / "outputs" / "e20-a"
DEFAULT_B_ROOT = VRBREPRODUCTION_ROOT / "outputs" / "e20-b"
HOA_PKL = VRBREPRODUCTION_ROOT / "data" / "P01_109.pkl"

PRIMARY_SCORE = 0.5
LOW_SCORE = 0.1
NEIGHBOR_SCORE = 0.5
NEIGHBOR_OFFSETS = (1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6)


def _collect_boxes_at(frame_det: Any, image_hw: tuple[int, int], threshold: float) -> list[list[float]]:
    height, width = image_hw
    out = []
    for hand in getattr(frame_det, "hands", []) or []:
        if float(getattr(hand, "score", 0.0)) < threshold:
            continue
        bbox = _bbox_norm_to_px(hand.bbox, width, height)
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            out.append(bbox)
    return out


def collect_neighbor_boxes(detections, ref_idx: int, image_hw: tuple[int, int]) -> list[list[float]]:
    """Nearest-frame fallback boxes, one per hand side (hands barely move in ±6 frames)."""
    found: dict[str, list[float]] = {}
    for offset in NEIGHBOR_OFFSETS:
        idx = ref_idx + offset
        if idx < 0 or idx >= len(detections):
            continue
        frame_det = detections[idx]
        for hand in getattr(frame_det, "hands", []) or []:
            side = str(getattr(getattr(hand, "side", None), "name", "")).lower()
            if side in found or float(getattr(hand, "score", 0.0)) < NEIGHBOR_SCORE:
                continue
            bbox = _bbox_norm_to_px(hand.bbox, image_hw[1], image_hw[0])
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                found[side] = bbox
        if len(found) >= 2:
            break
    return list(found.values())


def run_prepare(output_root: Path, source_root: Path = DEFAULT_A_ROOT) -> Path:
    """Copy the E20-A export (identical label geometry) into the E20-B root."""
    output_root = Path(output_root)
    source_export = Path(source_root) / "export"
    if not source_export.exists():
        raise FileNotFoundError(source_export)
    dest_export = output_root / "export"
    if dest_export.exists():
        shutil.rmtree(dest_export)
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_export, dest_export)
    manifest_path = dest_export / "export_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    for col in ("sample_dir", "reference_frame", "label_heatmap_npy", "labels_json"):
        manifest[col] = manifest[col].str.replace(str(Path(source_root)), str(output_root), regex=False)
    manifest.to_csv(manifest_path, index=False)
    print(f"E20B_PROGRESS stage=prepare status=done export={dest_export} rows={len(manifest)}", flush=True)
    return dest_export


def run_pack(output_root: Path) -> Path:
    output_root = Path(output_root)
    export_root = output_root / "export"
    manifest = pd.read_csv(export_root / "export_manifest.csv")
    job_dir = output_root / "gpu_job"
    if job_dir.exists():
        shutil.rmtree(job_dir)
    (job_dir / "images").mkdir(parents=True, exist_ok=True)

    detections = _load_detections_cached(HOA_PKL)
    samples = []
    counts = {"primary_empty": 0, "low_rescued": 0, "neighbor_rescued": 0, "still_empty": 0}
    for _, row in manifest.iterrows():
        labels = json.loads(Path(row["labels_json"]).read_text(encoding="utf-8"))
        sample_key = labels["sample_key"]
        image_name = f"{sample_key}.png"
        shutil.copyfile(row["reference_frame"], job_dir / "images" / image_name)
        heatmap = np.load(row["label_heatmap_npy"])
        h_img, w_img = [int(v) for v in labels["image_hw"]]
        ref_idx = int(labels["ref_idx"])
        frame_det = detections[ref_idx]
        primary = _collect_boxes_at(frame_det, (h_img, w_img), PRIMARY_SCORE)
        low = merge_overlapping_boxes(_collect_boxes_at(frame_det, (h_img, w_img), LOW_SCORE))
        neighbor = collect_neighbor_boxes(detections, ref_idx, (h_img, w_img))
        if not primary:
            counts["primary_empty"] += 1
            if low:
                counts["low_rescued"] += 1
            elif neighbor:
                counts["neighbor_rescued"] += 1
            else:
                counts["still_empty"] += 1
        samples.append(
            {
                "sample_key": sample_key,
                "rel_sample_dir": labels["rel_sample_dir"],
                "image": f"images/{image_name}",
                "subaction_index": int(row["subaction_index"]),
                "narration_id": row["narration_id"],
                "hand": labels["hand"],
                "reference_mode": labels["reference_mode"],
                "ref_idx": ref_idx,
                "noun": labels.get("noun"),
                "hand_bboxes_px": primary,
                "hand_bboxes_low_px": low,
                "neighbor_hand_bboxes_px": neighbor,
                "active_hand_bbox_px": labels.get("active_hand_bbox_px"),
                "crop_bbox_150": labels.get("crop_bbox_150"),
                "heatmap_top_bbox_px": heatmap_top_bbox(heatmap, thresh_frac=0.5),
            }
        )
    tasks = {
        "experiment_id": "E20B",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "video_id": "P01_109",
        "image_hw": [256, 456],
        "seed": SEED,
        "box_score_thresholds": {"primary": PRIMARY_SCORE, "low": LOW_SCORE, "neighbor": NEIGHBOR_SCORE},
        "samples": samples,
    }
    _json_write(job_dir / "tasks.json", tasks)
    worker_src = Path(__file__).with_name("server_worker_b.py")
    shutil.copyfile(worker_src, job_dir / "run_e20b_server.py")
    shutil.copyfile(Path(__file__).with_name("inpaint_utils.py"), job_dir / "e20_inpaint_utils.py")
    (job_dir / "requirements_server.txt").write_text(
        "\n".join(
            [
                "segment-anything",
                "simple-lama-inpainting",
                "diffusers>=0.27",
                "transformers",
                "accelerate",
                "safetensors",
                "opencv-python-headless",
                "pillow",
                "numpy",
                "torch",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (job_dir / "README_server.md").write_text(
        """# E20-B GPU job

```bash
cd /root/workspace/vrb/e20b_gpu_job
export SAM_CHECKPOINT=/root/workspace/vrb/models/sam_vit_h_4b8939.pth
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HOME=/root/workspace/vrb/cache/huggingface
export E20_SD_INPAINT_MODEL=runwayml/stable-diffusion-inpainting
export E20_SDEDIT_MODEL=runwayml/stable-diffusion-v1-5
export E20_DIFFUSERS_VARIANT=fp16
python run_e20b_server.py --job-dir . --out-dir /root/workspace/vrb/e20b_gpu_results --device cuda --backends lama,sd --sdedit 1
tar czf /root/workspace/vrb/e20b_gpu_results.tar.gz -C /root/workspace/vrb e20b_gpu_results
```
""",
        encoding="utf-8",
    )
    print(
        f"E20B_PROGRESS stage=pack status=done samples={len(samples)} box_counts={counts} job_dir={job_dir}",
        flush=True,
    )
    return job_dir


def run_ingest(output_root: Path, results_dir: Optional[Path] = None, results_tar: Optional[Path] = None) -> Path:
    output_root = Path(output_root)
    if results_tar:
        results_tar = Path(results_tar)
        with tarfile.open(results_tar, "r:gz") as tf:
            tf.extractall(output_root / "_ingest")
        ingest_root = output_root / "_ingest"
        candidates = sorted(path for path in ingest_root.glob("*_gpu_results") if path.is_dir())
        results_dir = candidates[0] if candidates else ingest_root / "e20b_gpu_results"
    if results_dir is None:
        results_dir = output_root / "gpu_results_local"
    results_dir = Path(results_dir)
    manifest = pd.read_csv(output_root / "export" / "export_manifest.csv")
    final_root = output_root / "experiments" / "E20" / "pipeline_outputs"
    final_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, row in manifest.iterrows():
        sample_key = row["sample_key"]
        rel_dir = row["rel_sample_dir"]
        export_sample_dir = Path(row["sample_dir"])
        src_sample = results_dir / sample_key
        final_sample = final_root / rel_dir
        final_sample.mkdir(parents=True, exist_ok=True)
        original = _read_bgr(Path(row["reference_frame"]))
        if not src_sample.exists():
            raise FileNotFoundError(src_sample)
        qc = json.loads((src_sample / "qc.json").read_text(encoding="utf-8"))
        mask = cv2.imread(str(src_sample / "mask.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(src_sample / "mask.png")
        cv2.imwrite(str(final_sample / "reference_frame_original.png"), original)
        cv2.imwrite(str(final_sample / "reference_hand_mask.png"), mask)
        _copy_sample_static(export_sample_dir, final_sample)

        rank = [name for name in (qc.get("backend_rank") or []) if (src_sample / f"{name}.png").exists()]
        passed = qc.get("passed", {})
        if qc.get("already_clean"):
            backend, status, qc_passed_final = "original", "already_clean", True
            final_img = original.copy()
        elif rank:
            backend = rank[0]
            qc_passed_final = bool(passed.get(backend))
            status = "final_inpainted" if qc_passed_final else "final_inpainted_qc_failed"
            final_img = _read_bgr(src_sample / f"{backend}.png")
            allowed = dilate_mask(mask, pad_px=3) > 0
            outside = ~allowed
            if not np.array_equal(final_img[outside], original[outside]):
                raise RuntimeError(f"mask-outside pixel check failed for {sample_key} {backend}")
        else:
            backend, status, qc_passed_final = "original", "inpaint_failed_kept_original", False
            final_img = original.copy()
        cv2.imwrite(str(final_sample / "reference_frame_inpainted.png"), final_img)
        for name in ("lama.png", "sd.png", "sdedit_lama.png", "sdedit_sd.png"):
            src = src_sample / name
            if src.exists():
                suffix = name.replace(".png", "")
                shutil.copyfile(src, final_sample / f"reference_frame_inpainted_{suffix}.png")
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(final_sample / "reference_inpaint_before_after.png"), np.concatenate([original, mask_bgr, final_img], axis=1))
        cand_path = final_sample / "candidate_result.json"
        cand = json.loads(cand_path.read_text(encoding="utf-8")) if cand_path.exists() else {}
        cand["e20"] = {
            "status": status,
            "backend_final": backend,
            "seed": SEED,
            "sample_key": sample_key,
            "final_policy": "qc_rank",
            "qc_passed_final": qc_passed_final,
            "qc": qc,
        }
        _json_write(cand_path, cand)
        rows.append(
            {
                "sample_key": sample_key,
                "rel_sample_dir": rel_dir,
                "status": status,
                "backend_final": backend,
                "final_policy": "qc_rank",
                "qc_passed_final": qc_passed_final,
                "path_final": str(final_sample / "reference_frame_inpainted.png"),
                "path_mask": str(final_sample / "reference_hand_mask.png"),
                "hand_box_count": qc.get("hand_box_count"),
                "box_sources": json.dumps(qc.get("box_sources") or {}),
                "inpaint_rounds": qc.get("inpaint_rounds"),
                "mask_area_ratio": qc.get("mask_area_ratio"),
                "skin_ratio_after_final": (qc.get("skin_ratio_after") or {}).get(backend),
                "residual_skin_after_final": (qc.get("residual_skin_after") or {}).get(backend),
                "boundary_delta_final": (qc.get("boundary_delta") or {}).get(backend),
                "sdedit_sharpness_ratio_final": (qc.get("sdedit_sharpness_ratio") or {}).get(backend),
                "overlap_heatmap_top": qc.get("overlap_heatmap_top"),
                "overlap_crop150": qc.get("overlap_crop150"),
                "inpaint_on_label_region": bool((qc.get("overlap_heatmap_top") or 0) > 0),
            }
        )
        print(f"E20B_PROGRESS stage=ingest sample={i+1}/{len(manifest)} key={sample_key} status={status} backend={backend}", flush=True)
    pd.DataFrame(rows).to_csv(output_root / "e20_ingest_manifest.csv", index=False)
    return final_root


def _put_header(canvas: np.ndarray, cols: list[str], tile_w: int, header_h: int) -> None:
    for c, name in enumerate(cols):
        cv2.putText(
            canvas,
            name,
            (c * tile_w + 8, header_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )


def make_before_after_enhanced_sheets(
    output_root: Path,
    manifest: pd.DataFrame,
    e20a_root: Path = DEFAULT_A_ROOT,
) -> list[Path]:
    """Per-sample comparison: original | E20-B final | enhanced | mask | lama | sd | E20-A final."""
    charts = Path(output_root) / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    final_root = Path(output_root) / "experiments" / "E20" / "pipeline_outputs"
    e20a_final_root = Path(e20a_root) / "experiments" / "E20" / "pipeline_outputs"
    cols = ["original", "e20b_final", "enhanced", "mask", "lama", "sd", "e20a_final"]
    rows_per_page = 6
    tile_w, tile_h, header_h = 228, 128, 26
    written: list[Path] = []
    for page_start in range(0, len(manifest), rows_per_page):
        subset = manifest.iloc[page_start : page_start + rows_per_page]
        canvas = np.full((header_h + tile_h * len(subset), tile_w * len(cols), 3), 245, dtype=np.uint8)
        _put_header(canvas, cols, tile_w, header_h)
        for r, (_, row) in enumerate(subset.iterrows()):
            sample = final_root / row["rel_sample_dir"]
            backend = str(row.get("backend_final") or "")
            enhanced = sample / f"reference_frame_inpainted_sdedit_{backend}.png"
            if not enhanced.exists():
                enhanced = sample / "reference_frame_inpainted_sdedit_lama.png"
            paths = [
                sample / "reference_frame_original.png",
                sample / "reference_frame_inpainted.png",
                enhanced,
                sample / "reference_hand_mask.png",
                sample / "reference_frame_inpainted_lama.png",
                sample / "reference_frame_inpainted_sd.png",
                e20a_final_root / row["rel_sample_dir"] / "reference_frame_inpainted.png",
            ]
            for c, path in enumerate(paths):
                if not path.exists():
                    continue
                mode = cv2.IMREAD_GRAYSCALE if path.name == "reference_hand_mask.png" else cv2.IMREAD_COLOR
                img = cv2.imread(str(path), mode)
                if img is None:
                    continue
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
                canvas[header_h + r * tile_h : header_h + (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = img
            cv2.putText(
                canvas,
                str(row["sample_key"])[:34],
                (4, header_h + r * tile_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )
        page = page_start // rows_per_page + 1
        out = charts / f"e20b_before_after_enhanced_contact_sheet_page_{page:03d}.png"
        cv2.imwrite(str(out), canvas)
        written.append(out)
    return written


def run_report(output_root: Path, e20a_root: Path = DEFAULT_A_ROOT) -> Path:
    output_root = Path(output_root)
    export_manifest = pd.read_csv(output_root / "export" / "export_manifest.csv")
    ingest = pd.read_csv(output_root / "e20_ingest_manifest.csv")
    quality_path = output_root / "e20_quality.csv"
    quality = pd.read_csv(quality_path) if quality_path.exists() else pd.DataFrame()
    manifest = export_manifest.merge(ingest, on=["sample_key", "rel_sample_dir"], how="left")
    if not quality.empty:
        manifest = manifest.merge(quality, on=["sample_key", "rel_sample_dir"], how="left")
    manifest["tier"] = np.where(
        manifest["status"].isin(["final_inpainted", "already_clean"])
        & (manifest["inpaint_on_label_region"].fillna(False) == False)
        & (manifest.get("redraw_selfcheck_max_diff", pd.Series(999, index=manifest.index)).fillna(999) == 0),
        "A",
        "B",
    )
    manifest.to_csv(output_root / "e20_manifest.csv", index=False)
    e20_pipeline._make_contact_sheets(output_root, manifest)
    sheets = make_before_after_enhanced_sheets(output_root, manifest, e20a_root=e20a_root)

    status_counts = manifest["status"].fillna("missing").value_counts().to_dict()
    backend_counts = manifest["backend_final"].fillna("missing").value_counts().to_dict()
    tier_counts = manifest["tier"].fillna("missing").value_counts().to_dict()
    qc_pass = int(manifest["qc_passed_final"].fillna(False).sum())
    mask_nonzero = int((manifest["mask_area_ratio"].fillna(0) > 0).sum())
    summary = [
        "# E20-B Summary",
        "",
        "E20-B keeps the E20/E20-A two-pass design (export label geometry, GPU inpaint, local redraw)",
        "and fixes the E20-A failure modes: missed hand boxes (HOA low-score @0.1 + neighbor-frame ±6",
        "fallback + GroundingDINO hand/arm text detection), palm-only masks (DINO arm boxes + SAM +",
        "direction-aware border bridging), arm regrowth (DINO-redetect re-inpaint rounds), SD letterbox",
        "artifacts (2x native-aspect inpainting), SDEdit global blur (2x img2img composited only inside",
        "the mask), and the broken QC (boundary_delta was 0 by construction; YCrCb skin confused wood",
        "counters with skin — replaced by DINO re-detection + seam gradient ratio; deterministic",
        "quality ranking replaces the seeded 50/50 backend pick).",
        "",
        f"- samples: {len(manifest)}",
        f"- status: {status_counts}",
        f"- backend_final: {backend_counts}",
        f"- qc_passed_final=true: {qc_pass}",
        f"- mask_nonzero: {mask_nonzero}",
        f"- tier: {tier_counts}",
        f"- redraw_selfcheck_max_diff_max: {manifest['redraw_selfcheck_max_diff'].max() if 'redraw_selfcheck_max_diff' in manifest else 'NA'}",
        "",
        "Outputs:",
        "- `e20_manifest.csv`",
        "- `e20_quality.csv`",
        "- `charts/e20_inpaint_contact_sheet_page_*.png`",
        f"- `charts/{sheets[0].name}` … ({len(sheets)} pages, original | e20b_final | enhanced | mask | lama | sd | e20a_final)",
    ]
    path = output_root / "summary.md"
    path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"E20B_PROGRESS stage=report status=done manifest={output_root / 'e20_manifest.csv'} summary={path}", flush=True)
    return path
