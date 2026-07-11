"""E20 two-pass hand-inpaint pipeline.

E20 reuses the E18b candidate/reference logic and only changes the rendering
boundary: export serializes label geometry and hand boxes, then finalize redraws
the same labels on an inpainted reference canvas.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd

from vrbreproduction import e18b_paper_faithful_cell2_retry as e18b
from vrbreproduction.e20.inpaint_utils import (
    choose_final_backend,
    collect_active_hand_bbox_px,
    collect_hand_bboxes_px,
    dilate_mask,
    heatmap_top_bbox,
    jsonable,
    mask_overlap_ratio,
    rel_dir_from_sample_key,
    sample_key_from_rel_dir,
)
from vrbreproduction.label_heatmap_utils import (
    build_label_heatmaps,
    draw_vrb_style_affordance_overlay,
    merge_label_heatmaps,
    overlay_heatmap_on_rgb,
    save_label_heatmap_outputs,
    transform_covariances_by_homography,
)


VRBREPRODUCTION_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = VRBREPRODUCTION_ROOT / "outputs" / "e20"
SEED = 20260707

E20_EXPERIMENT = {
    **e18b.EXPERIMENT,
    "experiment_id": "E20",
    "name": "E20_hand_inpaint_label_redraw",
    "description": (
        "E20: E18b-faithful export of label geometry, GPU hand inpainting, "
        "then same-function same-parameter label redraw on the inpainted canvas."
    ),
}


@dataclass(frozen=True)
class E20RunConfig:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    video_id: str = "P01_109"
    num_subactions: int = 100
    seed: int = SEED

    @property
    def export_root(self) -> Path:
        return self.output_root / "export"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _load_detections_cached(hoa_pkl: Path):
    from epic_kitchens.hoa import load_detections

    key = str(Path(hoa_pkl).resolve())
    cache = getattr(_load_detections_cached, "_cache", {})
    if key not in cache:
        cache[key] = load_detections(key)
        setattr(_load_detections_cached, "_cache", cache)
    return cache[key]


def _rel_sample_dir(sample_dir: Path) -> str:
    parts = Path(sample_dir).parts
    try:
        idx = parts.index("pipeline_outputs")
        return str(Path(*parts[idx + 1 :]))
    except ValueError:
        return str(Path(*parts[-3:]))


def _read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _e20_write_cell4_pipeline_outputs(
    *,
    config: e18b.E18BConfig,
    record: dict[str, Any],
    cell2_result: dict[str, Any],
    cell3_result: dict[str, Any],
    sample_dir: Path,
) -> dict[str, Any]:
    """E20-CHANGE-2/3: original E18b rendering plus labels.json and hand boxes."""
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    ref_idx = int(cell3_result["ref_idx"])
    ref_path = config.image_dir / f"frame_{ref_idx + 1:010d}.jpg"
    ref_img_bgr = _read_bgr(ref_path)
    ref_img_rgb = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img = ref_img_rgb.shape[:2]

    contact_means = cell2_result["contact_means"]
    contact_weights = cell2_result["contact_weights"]
    contact_covariances = cell2_result["contact_covariances"]
    mu_transformed = np.asarray(cell3_result["mu_transformed"], dtype=np.float32)
    h_contact_to_ref = cell3_result["H_contact_to_ref"]

    heatmap_mode = "covariance"
    covariances_ref = transform_covariances_by_homography(contact_covariances, contact_means, h_contact_to_ref)
    heatmap_kwargs = {
        "image_shape": (h_img, w_img),
        "centers_xy": mu_transformed,
        "sigma_px": 12.0,
        "weights": contact_weights,
        "normalize_each": True,
        "covariances_xy": covariances_ref,
        "covariance_scale": 6.0,
        "min_sigma_px": 7.0,
        "max_sigma_px": 40.0,
    }
    per_mode_heatmaps = build_label_heatmaps(**heatmap_kwargs)
    merged_heatmap = merge_label_heatmaps(per_mode_heatmaps=per_mode_heatmaps, merge_method="sum", normalize_output=True)

    heatmap_paths = e18b.silent_call(
        save_label_heatmap_outputs,
        merged_heatmap=merged_heatmap,
        ref_image=ref_img_rgb,
        output_dir=sample_dir,
        prefix="label_heatmap",
        sigma_px=12.0,
        merge_method="sum",
        use_weights=True,
        heatmap_mode=heatmap_mode,
        t_contact=record["frame_0_based"],
        active_hand=record["hand"],
        ref_idx=ref_idx,
    )

    reference_path = sample_dir / "reference_frame.png"
    cv2.imwrite(str(reference_path), ref_img_bgr)

    contact_anchor = cell3_result.get("contact_centroid")
    if contact_anchor is None:
        contact_anchor = mu_transformed.mean(axis=0)
    affordance_kwargs = {
        "heatmap_alpha": 0.45,
        "colormap": cv2.COLORMAP_JET,
        "arrow_length_px": None,
        "arrow_length_heatmap_ratio": 1.2,
        "arrow_width_px": None,
        "source_step": "last",
    }
    tau_transformed = cell3_result.get("tau_transformed", [])
    vrb_style_img, arrow_info = draw_vrb_style_affordance_overlay(
        ref_img_rgb,
        merged_heatmap,
        tau_transformed,
        contact_anchor,
        **affordance_kwargs,
    )
    vrb_style_path = sample_dir / "vrb_style_affordance.png"
    cv2.imwrite(str(vrb_style_path), cv2.cvtColor(vrb_style_img, cv2.COLOR_RGB2BGR))

    detections = _load_detections_cached(config.hoa_pkl)
    frame_det = detections[ref_idx]
    all_hand_bboxes = collect_hand_bboxes_px(frame_det, (h_img, w_img), score_threshold=0.5)
    active_hand_bbox = collect_active_hand_bbox_px(frame_det, record["hand"], (h_img, w_img), score_threshold=0.5)
    rel_dir = _rel_sample_dir(sample_dir)
    sample_key = sample_key_from_rel_dir(rel_dir)
    labels = {
        "schema_version": 1,
        "sample_key": sample_key,
        "rel_sample_dir": rel_dir,
        "image_hw": [h_img, w_img],
        "hand": record["hand"],
        "ref_idx": ref_idx,
        "reference_mode": cell3_result.get("reference_mode"),
        "contact_points_px": mu_transformed,
        "contact_anchor_xy": contact_anchor,
        "gmm": {
            "means_contact_px": contact_means,
            "means_ref_px": mu_transformed,
            "covariances_contact": contact_covariances,
            "covariances_ref": covariances_ref,
            "weights": contact_weights,
            "n_components": int(len(mu_transformed)),
        },
        "heatmap_mode": heatmap_mode,
        "trajectory_px": tau_transformed,
        "arrow": arrow_info,
        "crop_bbox_150": cell3_result.get("crop_bbox_150"),
        "hand_bboxes_px": all_hand_bboxes,
        "active_hand_bbox_px": active_hand_bbox,
        "render": {
            "heatmap_overlay_fn": "label_heatmap_utils.overlay_heatmap_on_rgb",
            "heatmap_overlay_kwargs": {"alpha": 0.45, "colormap": int(cv2.COLORMAP_JET), "max_intensity": 1.0},
            "affordance_fn": "label_heatmap_utils.draw_vrb_style_affordance_overlay",
            "affordance_kwargs": affordance_kwargs,
            "build_label_heatmaps_kwargs": heatmap_kwargs,
            "merge_label_heatmaps_kwargs": {"merge_method": "sum", "normalize_output": True},
        },
        "paths": {
            "reference_frame": str(reference_path),
            "label_heatmap_npy": heatmap_paths["npy_path"],
            "label_heatmap_overlay": heatmap_paths["overlay_path"],
            "vrb_style_affordance": str(vrb_style_path),
        },
    }
    labels_path = sample_dir / "labels.json"
    _json_write(labels_path, labels)

    return {
        "sample_dir": str(sample_dir),
        "sample_key_e20": sample_key,
        "rel_sample_dir": rel_dir,
        "reference_frame": str(reference_path),
        "label_heatmap_npy": heatmap_paths["npy_path"],
        "label_heatmap_png": heatmap_paths["png_path"],
        "label_heatmap_overlay": heatmap_paths["overlay_path"],
        "vrb_style_affordance": str(vrb_style_path),
        "labels_json": str(labels_path),
        "heatmap_mode": heatmap_mode,
        "heatmap_shape": tuple(int(v) for v in merged_heatmap.shape),
        "arrow_generated": arrow_info is not None,
        "hand_bboxes_px": all_hand_bboxes,
        "active_hand_bbox_px": active_hand_bbox,
        "problem3_full_overlay": str(sample_dir / "diagnostics" / "problem3_ref_full_overlay.png"),
        "problem3_crop": str(sample_dir / "diagnostics" / "problem3_ref_crop.png"),
    }


class _PatchedE18B:
    def __enter__(self):
        self.old_experiment = dict(e18b.EXPERIMENT)
        self.old_writer = e18b.write_cell4_pipeline_outputs
        e18b.EXPERIMENT.clear()
        e18b.EXPERIMENT.update(E20_EXPERIMENT)
        e18b.write_cell4_pipeline_outputs = _e20_write_cell4_pipeline_outputs
        return self

    def __exit__(self, exc_type, exc, tb):
        e18b.EXPERIMENT.clear()
        e18b.EXPERIMENT.update(self.old_experiment)
        e18b.write_cell4_pipeline_outputs = self.old_writer
        return False


def run_export(output_root: Path, limit: int = 100) -> dict[str, Any]:
    cfg = E20RunConfig(output_root=Path(output_root), num_subactions=int(limit))
    export_root = cfg.export_root
    print(f"E20_PROGRESS stage=export status=start output_root={export_root} limit={limit}", flush=True)
    with _PatchedE18B():
        result = e18b.run_e18b_experiment(
            e18b.E18BConfig(
                video_id=cfg.video_id,
                num_subactions=int(limit),
                output_root=export_root,
            )
        )
    report = build_export_manifest_and_consistency(Path(output_root), limit=int(limit))
    print(
        "E20_PROGRESS stage=export status=done "
        f"keep={report['e20_keep_count']} consistency={report['keep_set_match']}",
        flush=True,
    )
    result["consistency_report"] = report
    return result


def build_export_manifest_and_consistency(output_root: Path, limit: int = 100) -> dict[str, Any]:
    export_root = Path(output_root) / "export"
    diag_path = export_root / "candidate_diagnostics.csv"
    if not diag_path.exists():
        raise FileNotFoundError(diag_path)
    df = pd.read_csv(diag_path)
    keep = df[df["status"] == "keep"].copy()
    rows = []
    for _, row in keep.iterrows():
        sample_dir = Path(row["sample_dir"])
        rel_dir = row.get("rel_sample_dir") if isinstance(row.get("rel_sample_dir"), str) else _rel_sample_dir(sample_dir)
        sample_key = row.get("sample_key_e20") if isinstance(row.get("sample_key_e20"), str) else sample_key_from_rel_dir(rel_dir)
        npy_path = Path(row["label_heatmap_npy"])
        ref_path = Path(row["reference_frame"])
        rows.append(
            {
                "sample_key": sample_key,
                "rel_sample_dir": rel_dir,
                "subaction_index": int(row["subaction_index"]),
                "narration_id": row["narration_id"],
                "hand": row["hand"],
                "reference_mode": row["reference_mode"],
                "ref_idx": int(row["ref_idx"]),
                "sample_dir": str(sample_dir),
                "reference_frame": str(ref_path),
                "label_heatmap_npy": str(npy_path),
                "labels_json": str(sample_dir / "labels.json"),
                "npy_sha256": _sha256_file(npy_path),
                "reference_sha256": _sha256_file(ref_path),
            }
        )
    manifest = pd.DataFrame(rows).sort_values(["subaction_index", "sample_key"])
    manifest.to_csv(export_root / "export_manifest.csv", index=False)

    e18b_manifest_path = VRBREPRODUCTION_ROOT / "outputs" / "e18b" / "successful_sample_manifest.csv"
    keep_set_match = None
    mismatches: list[dict[str, Any]] = []
    if e18b_manifest_path.exists():
        base = pd.read_csv(e18b_manifest_path)
        if int(limit) < 100:
            base = base[base["subaction_index"] < int(limit)].copy()
        cols = ["subaction_index", "ref_idx", "reference_mode"]
        left = manifest[cols].copy()
        right = base[cols].copy()
        for frame in (left, right):
            frame["subaction_index"] = frame["subaction_index"].astype(int)
            frame["ref_idx"] = frame["ref_idx"].astype(int)
            frame["reference_mode"] = frame["reference_mode"].astype(str)
        left = left.sort_values(cols).reset_index(drop=True)
        right = right.sort_values(cols).reset_index(drop=True)
        keep_set_match = bool(left.equals(right))
        if not keep_set_match:
            mismatches.append(
                {
                    "e20_rows": left.to_dict(orient="records"),
                    "e18b_rows": right.to_dict(orient="records"),
                }
            )
    report = {
        "e20_keep_count": int(len(manifest)),
        "limit": int(limit),
        "e18b_manifest": str(e18b_manifest_path),
        "keep_set_match": keep_set_match,
        "mismatches": mismatches,
        "manifest": str(export_root / "export_manifest.csv"),
    }
    _json_write(export_root / "consistency_report.json", report)
    if keep_set_match is False:
        raise RuntimeError("E20 export keep set does not match E18b; see consistency_report.json")
    return report


def run_pack(output_root: Path) -> Path:
    output_root = Path(output_root)
    export_root = output_root / "export"
    manifest = pd.read_csv(export_root / "export_manifest.csv")
    job_dir = output_root / "gpu_job"
    if job_dir.exists():
        shutil.rmtree(job_dir)
    (job_dir / "images").mkdir(parents=True, exist_ok=True)

    samples = []
    empty_boxes = 0
    for _, row in manifest.iterrows():
        labels = json.loads(Path(row["labels_json"]).read_text(encoding="utf-8"))
        sample_key = labels["sample_key"]
        image_name = f"{sample_key}.png"
        shutil.copyfile(row["reference_frame"], job_dir / "images" / image_name)
        heatmap = np.load(row["label_heatmap_npy"])
        boxes = labels.get("hand_bboxes_px") or []
        empty_boxes += int(len(boxes) == 0)
        samples.append(
            {
                "sample_key": sample_key,
                "rel_sample_dir": labels["rel_sample_dir"],
                "image": f"images/{image_name}",
                "subaction_index": int(row["subaction_index"]),
                "narration_id": row["narration_id"],
                "hand": labels["hand"],
                "reference_mode": labels["reference_mode"],
                "ref_idx": int(labels["ref_idx"]),
                "noun": labels.get("noun"),
                "hand_bboxes_px": boxes,
                "active_hand_bbox_px": labels.get("active_hand_bbox_px"),
                "crop_bbox_150": labels.get("crop_bbox_150"),
                "heatmap_top_bbox_px": heatmap_top_bbox(heatmap, thresh_frac=0.5),
            }
        )
    tasks = {
        "experiment_id": "E20",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "video_id": "P01_109",
        "image_hw": [256, 456],
        "seed": SEED,
        "samples": samples,
    }
    _json_write(job_dir / "tasks.json", tasks)
    shutil.copyfile(Path(__file__).with_name("server_worker.py"), job_dir / "run_e20_server.py")
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
                "mediapipe",
                "pillow",
                "numpy",
                "torch",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (job_dir / "README_server.md").write_text(
        """# E20 GPU job

```bash
cd /root/workspace/vrb/e20_gpu_job
python -m pip install -r requirements_server.txt
export HF_ENDPOINT=https://hf-mirror.com
wget -nc https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O /root/workspace/vrb/sam_vit_h_4b8939.pth
export SAM_CHECKPOINT=/root/workspace/vrb/sam_vit_h_4b8939.pth
python run_e20_server.py --job-dir . --out-dir /root/workspace/vrb/e20_gpu_results --device cuda --backends lama,sd --sdedit 1
tar czf /root/workspace/vrb/e20_gpu_results.tar.gz -C /root/workspace/vrb e20_gpu_results
```
""",
        encoding="utf-8",
    )
    print(
        f"E20_PROGRESS stage=pack status=done samples={len(samples)} empty_hand_bbox={empty_boxes} job_dir={job_dir}",
        flush=True,
    )
    return job_dir


def _copy_sample_static(export_sample_dir: Path, final_sample_dir: Path) -> None:
    for name in ("labels.json", "candidate_result.json", "label_heatmap_merged.npy"):
        src = export_sample_dir / name
        if src.exists():
            shutil.copyfile(src, final_sample_dir / name)


def run_ingest(output_root: Path, results_dir: Optional[Path] = None, results_tar: Optional[Path] = None) -> Path:
    output_root = Path(output_root)
    if results_tar:
        results_tar = Path(results_tar)
        with tarfile.open(results_tar, "r:gz") as tf:
            tf.extractall(output_root / "_ingest")
        ingest_root = output_root / "_ingest"
        default_results = ingest_root / "e20_gpu_results"
        if default_results.exists():
            results_dir = default_results
        else:
            candidates = sorted(path for path in ingest_root.glob("*_gpu_results") if path.is_dir())
            if len(candidates) == 1:
                results_dir = candidates[0]
            else:
                results_dir = default_results
    if results_dir is None:
        results_dir = output_root / "gpu_results_local"
    results_dir = Path(results_dir)
    manifest = pd.read_csv(output_root / "export" / "export_manifest.csv")
    final_root = output_root / "experiments" / "E20" / "pipeline_outputs"
    final_root.mkdir(parents=True, exist_ok=True)
    final_policy = os.environ.get("E20_FINAL_POLICY", "strict_qc").strip().lower()

    rows = []
    for i, row in manifest.iterrows():
        sample_key = row["sample_key"]
        rel_dir = row["rel_sample_dir"]
        export_sample_dir = Path(row["sample_dir"])
        src_sample = results_dir / sample_key
        final_sample = final_root / rel_dir
        final_sample.mkdir(parents=True, exist_ok=True)
        labels = json.loads((export_sample_dir / "labels.json").read_text(encoding="utf-8"))
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
        passed = qc.get("passed", {})
        backend = choose_final_backend(sample_key, passed, seed=SEED)
        qc_passed_final = bool(passed.get(backend)) if backend != "none" else False
        status = "final_inpainted"
        if qc.get("already_clean"):
            backend = "original"
            status = "already_clean"
            qc_passed_final = True
            final_img = original.copy()
        elif backend == "none":
            generated = {name: True for name in ("lama", "sd") if (src_sample / f"{name}.png").exists()}
            if final_policy == "prefer_generated" and generated:
                backend = choose_final_backend(sample_key, generated, seed=SEED)
                status = "final_inpainted_qc_failed"
                qc_passed_final = False
                final_img = _read_bgr(src_sample / f"{backend}.png")
                allowed = dilate_mask(mask, pad_px=3) > 0
                outside = ~allowed
                if not np.array_equal(final_img[outside], original[outside]):
                    raise RuntimeError(f"mask-outside pixel check failed for {sample_key} {backend}")
            else:
                backend = "original"
                status = "inpaint_failed_kept_original"
                final_img = original.copy()
        else:
            final_img = _read_bgr(src_sample / f"{backend}.png")
            allowed = dilate_mask(mask, pad_px=3) > 0
            outside = ~allowed
            if not np.array_equal(final_img[outside], original[outside]):
                raise RuntimeError(f"mask-outside pixel check failed for {sample_key} {backend}")
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
            "final_policy": final_policy,
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
                "final_policy": final_policy,
                "qc_passed_final": qc_passed_final,
                "path_final": str(final_sample / "reference_frame_inpainted.png"),
                "path_mask": str(final_sample / "reference_hand_mask.png"),
                "mask_area_ratio": qc.get("mask_area_ratio"),
                "skin_ratio_after_final": qc.get("skin_ratio_after", {}).get(backend) if isinstance(qc.get("skin_ratio_after"), dict) else None,
                "boundary_delta_final": qc.get("boundary_delta", {}).get(backend) if isinstance(qc.get("boundary_delta"), dict) else None,
                "overlap_heatmap_top": qc.get("overlap_heatmap_top"),
                "overlap_crop150": qc.get("overlap_crop150"),
                "inpaint_on_label_region": bool((qc.get("overlap_heatmap_top") or 0) > 0),
            }
        )
        print(f"E20_PROGRESS stage=ingest sample={i+1}/{len(manifest)} key={sample_key} status={status} backend={backend}", flush=True)
    pd.DataFrame(rows).to_csv(output_root / "e20_ingest_manifest.csv", index=False)
    return final_root


def _render_from_labels(base_bgr: np.ndarray, heatmap: np.ndarray, labels: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    base_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
    overlay = overlay_heatmap_on_rgb(base_rgb, heatmap, **labels["render"]["heatmap_overlay_kwargs"])
    tau_points = [np.asarray(pt, dtype=np.float32) if pt is not None else None for pt in labels.get("trajectory_px", [])]
    contact_anchor = np.asarray(labels.get("contact_anchor_xy") or np.asarray(labels["contact_points_px"]).mean(axis=0), dtype=np.float32)
    affordance, _ = draw_vrb_style_affordance_overlay(
        base_rgb,
        heatmap,
        tau_points,
        contact_anchor,
        **labels["render"]["affordance_kwargs"],
    )
    return cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), cv2.cvtColor(affordance, cv2.COLOR_RGB2BGR)


def run_finalize(output_root: Path) -> Path:
    output_root = Path(output_root)
    manifest = pd.read_csv(output_root / "export" / "export_manifest.csv")
    final_root = output_root / "experiments" / "E20" / "pipeline_outputs"
    quality_rows = []
    for i, row in manifest.iterrows():
        rel_dir = row["rel_sample_dir"]
        export_sample = Path(row["sample_dir"])
        final_sample = final_root / rel_dir
        labels = json.loads((final_sample / "labels.json").read_text(encoding="utf-8"))
        heatmap = np.load(final_sample / "label_heatmap_merged.npy")
        final_bgr = _read_bgr(final_sample / "reference_frame_inpainted.png")
        overlay_bgr, affordance_bgr = _render_from_labels(final_bgr, heatmap, labels)
        overlay_path = final_sample / "label_heatmap_merged_overlay_inpainted.png"
        affordance_path = final_sample / "vrb_style_affordance_inpainted.png"
        cv2.imwrite(str(overlay_path), overlay_bgr)
        cv2.imwrite(str(affordance_path), affordance_bgr)

        original_bgr = _read_bgr(export_sample / "reference_frame.png")
        check_overlay, check_affordance = _render_from_labels(original_bgr, heatmap, labels)
        base_overlay = _read_bgr(export_sample / "label_heatmap_merged_overlay.png")
        base_affordance = _read_bgr(export_sample / "vrb_style_affordance.png")
        overlay_diff = int(np.max(np.abs(check_overlay.astype(np.int16) - base_overlay.astype(np.int16))))
        affordance_diff = int(np.max(np.abs(check_affordance.astype(np.int16) - base_affordance.astype(np.int16))))
        max_diff = max(overlay_diff, affordance_diff)
        if max_diff != 0:
            raise RuntimeError(f"redraw selfcheck failed for {row['sample_key']}: max_diff={max_diff}")
        quality_rows.append(
            {
                "sample_key": row["sample_key"],
                "rel_sample_dir": rel_dir,
                "redraw_selfcheck_max_diff": max_diff,
                "path_overlay_inpainted": str(overlay_path),
                "path_affordance_inpainted": str(affordance_path),
            }
        )
        print(f"E20_PROGRESS stage=finalize sample={i+1}/{len(manifest)} key={row['sample_key']} selfcheck={max_diff}", flush=True)
    pd.DataFrame(quality_rows).to_csv(output_root / "e20_quality.csv", index=False)
    return output_root / "e20_quality.csv"


def _read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _make_contact_sheets(output_root: Path, manifest: pd.DataFrame) -> None:
    charts = output_root / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    final_root = output_root / "experiments" / "E20" / "pipeline_outputs"
    rows_per_page = 6
    tile_w, tile_h = 228, 128
    cols = ["orig", "mask", "lama", "sd", "afford", "overlay"]
    for page_start in range(0, len(manifest), rows_per_page):
        subset = manifest.iloc[page_start : page_start + rows_per_page]
        canvas = np.full((tile_h * len(subset), tile_w * len(cols), 3), 245, dtype=np.uint8)
        for r, (_, row) in enumerate(subset.iterrows()):
            sample = final_root / row["rel_sample_dir"]
            paths = [
                sample / "reference_frame_original.png",
                sample / "reference_hand_mask.png",
                sample / "reference_frame_inpainted_lama.png",
                sample / "reference_frame_inpainted_sd.png",
                sample / "vrb_style_affordance_inpainted.png",
                sample / "label_heatmap_merged_overlay_inpainted.png",
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
                canvas[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = img
        page = page_start // rows_per_page + 1
        cv2.imwrite(str(charts / f"e20_inpaint_contact_sheet_page_{page:03d}.png"), canvas)


def run_report(output_root: Path) -> Path:
    output_root = Path(output_root)
    experiment_label = output_root.name.upper() if output_root.name != "e20" else "E20"
    export_manifest = pd.read_csv(output_root / "export" / "export_manifest.csv")
    ingest = _read_optional_csv(output_root / "e20_ingest_manifest.csv")
    quality = _read_optional_csv(output_root / "e20_quality.csv")
    manifest = export_manifest.merge(ingest, on=["sample_key", "rel_sample_dir"], how="left").merge(
        quality, on=["sample_key", "rel_sample_dir"], how="left"
    )
    manifest["tier"] = np.where(
        manifest["status"].isin(["final_inpainted", "already_clean"])
        & (manifest["inpaint_on_label_region"].fillna(False) == False)
        & (manifest["redraw_selfcheck_max_diff"].fillna(999) == 0),
        "A",
        "B",
    )
    manifest.to_csv(output_root / "e20_manifest.csv", index=False)
    _make_contact_sheets(output_root, manifest)
    status_counts = manifest["status"].fillna("missing").value_counts().to_dict()
    backend_counts = manifest["backend_final"].fillna("missing").value_counts().to_dict()
    tier_counts = manifest["tier"].fillna("missing").value_counts().to_dict()
    summary = [
        f"# {experiment_label} Summary",
        "",
        f"{experiment_label} uses the two-pass design: export E18b-faithful label geometry, inpaint hands on GPU, then redraw heatmap and trajectory with the same functions and cached parameters on the inpainted canvas.",
        "",
        f"- samples: {len(manifest)}",
        f"- status: {status_counts}",
        f"- backend_final: {backend_counts}",
        f"- tier: {tier_counts}",
        f"- redraw_selfcheck_max_diff_max: {manifest['redraw_selfcheck_max_diff'].max() if 'redraw_selfcheck_max_diff' in manifest else 'NA'}",
        "",
        "Outputs:",
        "- `e20_manifest.csv`",
        "- `e20_quality.csv`",
        "- `charts/e20_inpaint_contact_sheet_page_*.png`",
    ]
    path = output_root / "summary.md"
    path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"E20_PROGRESS stage=report status=done manifest={output_root / 'e20_manifest.csv'} summary={path}", flush=True)
    return path
