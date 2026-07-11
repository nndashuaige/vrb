"""E20-C pipeline: E20-B two-pass design plus label-region protection and hallucination gating.

E20-C keeps the E20/E20-A/E20-B boundary (export label geometry -> GPU inpaint
-> local redraw) and reuses the E20-B export verbatim (which is itself the
E20-A export: same reference frames, same label geometry). Changes:

- pack_c additionally ships contact_points_px / contact_anchor_xy so the GPU
  worker can build the label-region protect mask (E20-B swallowed the
  interaction target in 20/56 samples, overlap_heatmap_top >= 0.99);
- the GPU worker is server_worker_c.py (protect-mask hard constraint, SD
  hallucination gate, tamer dilation, skin-confirmed re-detection);
- ingest_c records protection metrics and copies protect.png;
- report_c renders annotated contact sheets: per-row label bands with
  subaction / narration_id / narration (from EPIC_100_train.csv) and a column
  header, text never painted over image content.
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
from vrbreproduction.e20.pipeline_b import (
    HOA_PKL,
    LOW_SCORE,
    NEIGHBOR_SCORE,
    PRIMARY_SCORE,
    _collect_boxes_at,
    collect_neighbor_boxes,
)

DEFAULT_B_ROOT = VRBREPRODUCTION_ROOT / "outputs" / "e20-b"
DEFAULT_C_ROOT = VRBREPRODUCTION_ROOT / "outputs" / "e20-c"
EPIC_TRAIN_CSV = (
    VRBREPRODUCTION_ROOT / "data" / "annotations" / "epic-kitchens-100-annotations" / "EPIC_100_train.csv"
)


def run_prepare(output_root: Path, source_root: Path = DEFAULT_B_ROOT) -> Path:
    """Copy the E20-B export (identical label geometry, E18B-derived) into the E20-C root."""
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
    print(f"E20C_PROGRESS stage=prepare status=done export={dest_export} rows={len(manifest)}", flush=True)
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
                # E20-C: label geometry for the protect mask
                "contact_points_px": labels.get("contact_points_px"),
                "contact_anchor_xy": labels.get("contact_anchor_xy"),
            }
        )
    tasks = {
        "experiment_id": "E20C",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "video_id": "P01_109",
        "image_hw": [256, 456],
        "seed": SEED,
        "box_score_thresholds": {"primary": PRIMARY_SCORE, "low": LOW_SCORE, "neighbor": NEIGHBOR_SCORE},
        "samples": samples,
    }
    _json_write(job_dir / "tasks.json", tasks)
    shutil.copyfile(Path(__file__).with_name("server_worker_c.py"), job_dir / "run_e20c_server.py")
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
        """# E20-C GPU job

```bash
cd /root/workspace/vrb/e20c_gpu_job
export LD_LIBRARY_PATH=/opt/anaconda/lib
export SAM_CHECKPOINT=/root/workspace/vrb/models/sam_vit_h_4b8939.pth
export LAMA_MODEL=/root/workspace/vrb/models/big-lama.pt
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HOME=/root/workspace/vrb/cache/huggingface
export E20_SD_INPAINT_MODEL=runwayml/stable-diffusion-inpainting
export E20_SDEDIT_MODEL=runwayml/stable-diffusion-v1-5
export E20_DIFFUSERS_VARIANT=fp16
export E20_DINO_MODEL=/root/workspace/vrb/models/grounding-dino-tiny
/opt/anaconda/bin/python run_e20c_server.py --job-dir . --out-dir /root/workspace/vrb/e20c_gpu_results --device cuda --backends lama,sd --sdedit 1
tar czf /root/workspace/vrb/e20c_gpu_results.tar.gz -C /root/workspace/vrb e20c_gpu_results
```
""",
        encoding="utf-8",
    )
    print(f"E20C_PROGRESS stage=pack status=done samples={len(samples)} job_dir={job_dir}", flush=True)
    return job_dir


def run_ingest(output_root: Path, results_dir: Optional[Path] = None, results_tar: Optional[Path] = None) -> Path:
    output_root = Path(output_root)
    if results_tar:
        results_tar = Path(results_tar)
        with tarfile.open(results_tar, "r:gz") as tf:
            tf.extractall(output_root / "_ingest")
        ingest_root = output_root / "_ingest"
        candidates = sorted(path for path in ingest_root.glob("*_gpu_results") if path.is_dir())
        results_dir = candidates[0] if candidates else ingest_root / "e20c_gpu_results"
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
        protect_src = src_sample / "protect.png"
        if protect_src.exists():
            shutil.copyfile(protect_src, final_sample / "reference_protect_mask.png")
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
        cv2.imwrite(
            str(final_sample / "reference_inpaint_before_after.png"),
            np.concatenate([original, mask_bgr, final_img], axis=1),
        )
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
                "mask_area_ratio_raw": qc.get("mask_area_ratio_raw"),
                "skin_ratio_after_final": (qc.get("skin_ratio_after") or {}).get(backend),
                "boundary_delta_final": (qc.get("boundary_gradient_ratio") or {}).get(backend),
                "sdedit_sharpness_ratio_final": (qc.get("sdedit_sharpness_ratio") or {}).get(backend),
                "overlap_heatmap_top": qc.get("overlap_heatmap_top"),
                "overlap_heatmap_top_raw": qc.get("overlap_heatmap_top_raw"),
                "overlap_crop150": qc.get("overlap_crop150"),
                "protect_source": qc.get("protect_source"),
                "protect_area_ratio": qc.get("protect_area_ratio"),
                "protect_conflict_frac": qc.get("protect_conflict_frac"),
                "protect_intact_final": qc.get("protect_intact_final"),
                "label_region_conflict": qc.get("label_region_conflict"),
                "sd_final_eligible": qc.get("sd_final_eligible"),
                "sharpness_ratio_vs_original_final": (qc.get("sharpness_ratio_vs_original") or {}).get(backend),
                "inpaint_on_label_region": bool((qc.get("overlap_heatmap_top") or 0) > 0),
            }
        )
        print(
            f"E20C_PROGRESS stage=ingest sample={i+1}/{len(manifest)} key={sample_key} status={status} backend={backend}",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(output_root / "e20_ingest_manifest.csv", index=False)
    return final_root


def _load_narrations() -> dict[str, str]:
    if not EPIC_TRAIN_CSV.exists():
        return {}
    df = pd.read_csv(EPIC_TRAIN_CSV, usecols=["narration_id", "narration"])
    return dict(zip(df["narration_id"].astype(str), df["narration"].astype(str)))


def make_annotated_sheets(
    output_root: Path,
    manifest: pd.DataFrame,
    e20b_root: Path = DEFAULT_B_ROOT,
) -> list[Path]:
    """Annotated contact sheets: label band per row, header row, no text over images.

    Row = one sample. Columns: original | e20c_final | affordance | enhanced |
    mask+protect | lama | sd | e20b_final. The mask+protect tile shows the
    inpaint mask in white and the protected label region in green.
    """
    charts = Path(output_root) / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    final_root = Path(output_root) / "experiments" / "E20" / "pipeline_outputs"
    e20b_final_root = Path(e20b_root) / "experiments" / "E20" / "pipeline_outputs"
    narrations = _load_narrations()
    cols = ["original", "e20c_final", "affordance_redrawn", "enhanced", "mask(+protect)", "lama", "sd", "e20b_final"]
    rows_per_page = 6
    tile_w, tile_h = 228, 128
    header_h, band_h = 30, 22
    row_h = band_h + tile_h
    written: list[Path] = []
    for page_start in range(0, len(manifest), rows_per_page):
        subset = manifest.iloc[page_start : page_start + rows_per_page]
        canvas = np.full((header_h + row_h * len(subset), tile_w * len(cols), 3), 245, dtype=np.uint8)
        for c, name in enumerate(cols):
            cv2.putText(
                canvas, name, (c * tile_w + 8, header_h - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA,
            )
        for r, (_, row) in enumerate(subset.iterrows()):
            y0 = header_h + r * row_h
            # label band (never overlaps the tiles below it)
            canvas[y0 : y0 + band_h, :] = (225, 225, 225)
            sub_id = str(row["sample_key"]).split("__")[0]
            nid = str(row["narration_id"])
            narration = narrations.get(nid, "?")
            band_text = f"{sub_id}  |  {nid}  |  {narration}  |  final={row.get('backend_final')}  qc={'PASS' if row.get('qc_passed_final') else 'FAIL'}"
            cv2.putText(
                canvas, band_text[:180], (8, y0 + band_h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA,
            )
            sample = final_root / row["rel_sample_dir"]
            backend = str(row.get("backend_final") or "")
            enhanced = sample / f"reference_frame_inpainted_sdedit_{backend}.png"
            if not enhanced.exists():
                enhanced = sample / "reference_frame_inpainted_sdedit_lama.png"
            paths = [
                sample / "reference_frame_original.png",
                sample / "reference_frame_inpainted.png",
                sample / "vrb_style_affordance_inpainted.png",
                enhanced,
                None,  # mask+protect composite
                sample / "reference_frame_inpainted_lama.png",
                sample / "reference_frame_inpainted_sd.png",
                e20b_final_root / row["rel_sample_dir"] / "reference_frame_inpainted.png",
            ]
            ty = y0 + band_h
            for c, path in enumerate(paths):
                if c == 4:
                    mask = cv2.imread(str(sample / "reference_hand_mask.png"), cv2.IMREAD_GRAYSCALE)
                    protect = cv2.imread(str(sample / "reference_protect_mask.png"), cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        continue
                    tile = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    if protect is not None:
                        tile[protect > 0] = (0, 200, 0)
                    img = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
                    canvas[ty : ty + tile_h, c * tile_w : (c + 1) * tile_w] = img
                    continue
                if path is None or not path.exists():
                    continue
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
                canvas[ty : ty + tile_h, c * tile_w : (c + 1) * tile_w] = img
        page = page_start // rows_per_page + 1
        out = charts / f"e20c_before_after_enhanced_contact_sheet_page_{page:03d}.png"
        cv2.imwrite(str(out), canvas)
        written.append(out)
    return written


def run_report(output_root: Path, e20b_root: Path = DEFAULT_B_ROOT) -> Path:
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
    sheets = make_annotated_sheets(output_root, manifest, e20b_root=e20b_root)

    status_counts = manifest["status"].fillna("missing").value_counts().to_dict()
    backend_counts = manifest["backend_final"].fillna("missing").value_counts().to_dict()
    tier_counts = manifest["tier"].fillna("missing").value_counts().to_dict()
    qc_pass = int(manifest["qc_passed_final"].fillna(False).sum())
    mask_nonzero = int((manifest["mask_area_ratio"].fillna(0) > 0).sum())
    protect_intact = int(manifest["protect_intact_final"].fillna(False).sum())
    protect_sam = int((manifest["protect_source"] == "sam+bbox").sum())
    conflict = int(manifest["label_region_conflict"].fillna(False).sum())
    ov_hm_zero = int((manifest["overlap_heatmap_top"].fillna(1) == 0).sum())
    summary = [
        "# E20-C Summary",
        "",
        "E20-C keeps the E20/E20-A/E20-B two-pass design (export label geometry, GPU inpaint, local",
        "redraw) and fixes the four residual E20-B failure modes:",
        "",
        "1. **Target-object protection**: E20-B masks swallowed the interaction target in 20/56",
        "   samples (overlap_heatmap_top >= 0.99, e.g. the rucksack in subaction_01 and the cutting",
        "   board in subaction_66). E20-C builds a protect mask from the exported label geometry",
        "   (SAM object segmentation prompted by heatmap_top_bbox_px + the bbox rect + contact-point",
        "   disks) and subtracts it from the inpaint mask as a hard constraint, every round.",
        "2. **SD hallucination gate**: SD may not be the final if its in-mask sharpness exceeds 2x the",
        "   original frame's (calibrated on all 56 E20-B samples) or the mask covers >0.40 of the frame.",
        "3. **Tamer masks**: dilate 14->8, bridge width 1.4->1.3 reduce unnecessary hole growth.",
        "4. **Skin-confirmed re-detection**: DINO hand/arm fires only count when the box also has",
        "   dual-rule skin evidence (AND condition; skin alone matches wooden counters), removing the",
        "   trouser-leg/knee false fails of E20-B.",
        "",
        "## Chart layout (charts/e20c_before_after_enhanced_contact_sheet_page_*.png)",
        "",
        "One row = one sample. The grey band above each row states: subaction id | narration_id |",
        "narration (action text from EPIC_100_train.csv) | chosen final backend | strict-QC verdict.",
        "Columns, labelled in the header: original (reference frame), e20c_final (chosen inpainted",
        "result), affordance_redrawn (vrb_style_affordance_inpainted.png - labels redrawn on the",
        "inpainted canvas; heatmap/arrow should still sit on the target object), enhanced (SDEdit",
        "candidate), mask(+protect) (inpaint mask in white, protected label region in green), lama /",
        "sd (backend candidates), e20b_final (previous iteration, for regression checking).",
        "",
        "## Numbers",
        "",
        f"- samples: {len(manifest)}",
        f"- status: {status_counts}",
        f"- backend_final: {backend_counts}",
        f"- qc_passed_final=true: {qc_pass}",
        f"- mask_nonzero: {mask_nonzero}",
        f"- protect_intact_final=true: {protect_intact} (protect region untouched by the final mask)",
        f"- protect_source=sam+bbox: {protect_sam} (SAM object segmentation accepted)",
        f"- label_region_conflict=true: {conflict} (hand overlaps label region; mask clipped there)",
        f"- overlap_heatmap_top==0: {ov_hm_zero} (E20-B had 18/56 at 1.0)",
        f"- tier: {tier_counts}",
        f"- redraw_selfcheck_max_diff_max: {manifest['redraw_selfcheck_max_diff'].max() if 'redraw_selfcheck_max_diff' in manifest else 'NA'}",
        "",
        "Outputs:",
        "- `e20_manifest.csv` / `e20_quality.csv`",
        "- `charts/e20_inpaint_contact_sheet_page_*.png` (legacy layout)",
        f"- `charts/{sheets[0].name}` … ({len(sheets)} pages, annotated layout described above)",
    ]
    path = output_root / "summary.md"
    path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"E20C_PROGRESS stage=report status=done manifest={output_root / 'e20_manifest.csv'} summary={path}", flush=True)
    return path
