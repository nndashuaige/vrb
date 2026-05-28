from collections import Counter
from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings


@dataclass(frozen=True)
class StageResult:
    stage: str
    passed: bool
    details: Dict[str, Any]


def build_event_records(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sorted_samples = sorted(samples, key=lambda sample: (int(sample["frame"]), str(sample["hand"]).lower()))
    events: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for sample in sorted_samples:
        frame = int(sample["frame"])
        hand = str(sample["hand"]).lower()
        if current is None or current["hand"] != hand or frame != current["end_frame"] + 1:
            current = {
                "event_id": len(events),
                "hand": hand,
                "start_frame": frame,
                "end_frame": frame,
                "frames": [frame],
                "representative_frame": frame,
                "representative_sample": dict(sample),
                "status": "pending",
                "failed_stage": None,
                "discard_reason": None,
                "stages": {},
            }
            events.append(current)
        else:
            current["end_frame"] = frame
            current["frames"].append(frame)

    return events


def update_event_stage(event: Dict[str, Any], result: StageResult) -> None:
    event["stages"][result.stage] = {
        "passed": bool(result.passed),
        **dict(result.details),
    }
    if result.passed:
        if event["status"] == "pending":
            event["status"] = "keep"
        return

    if event["failed_stage"] is None:
        event["status"] = "discard"
        event["failed_stage"] = result.stage
        event["discard_reason"] = result.details.get("reason", result.stage)


def summarize_stage_counts(events: Sequence[Dict[str, Any]], stage_order: Sequence[str]) -> List[Dict[str, int]]:
    summary: List[Dict[str, int]] = []
    input_events = list(events)

    for stage in stage_order:
        kept = []
        discarded_count = 0
        for event in input_events:
            stage_info = event.get("stages", {}).get(stage)
            if stage_info is not None and stage_info.get("passed") is True:
                kept.append(event)
            else:
                discarded_count += 1
        summary.append({
            "stage": stage,
            "input_events": len(input_events),
            "kept_events": len(kept),
            "discarded_events": discarded_count,
        })
        input_events = kept

    return summary


def _json_safe(value: Any) -> Any:
    """Convert common numpy/path values into JSON-serializable values."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - only used outside the project env
        np = None

    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _silent_call(func, *args, quiet: bool = True, **kwargs):
    if not quiet:
        return func(*args, **kwargs)

    with redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _default_contact_config():
    from .contact_point_utils import ContactExtractionConfig

    return ContactExtractionConfig(
        use_object_mask=True,
        use_object_boundary=True,
        boundary_distance_px=6.0,
        project_points_to_object_boundary=True,
    )


def scan_contact_point_frames(
    detections,
    scan_limit: int = 500,
    image_dir: Path = Path("data/P01_109_frames"),
    contact_extraction_config=None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """
    Frame-level version of the notebook's Cell 1-test precheck.

    Returns valid frame records, rejected frame records, and reason counts.
    A frame is represented at most once, matching the notebook's "first valid
    contact hand then break" behavior.
    """
    import cv2
    from epic_kitchens.hoa.types import HandState
    from .contact_point_utils import extract_contact_points, get_valid_object_bboxes

    config = contact_extraction_config or _default_contact_config()
    image_dir = Path(image_dir)
    scanned_frames = min(int(scan_limit), len(detections))
    valid_records: List[Dict[str, Any]] = []
    rejected_records: List[Dict[str, Any]] = []
    reason_counts: Counter = Counter()

    for frame_idx in range(scanned_frames):
        frame_det = detections[frame_idx]
        object_bboxes = get_valid_object_bboxes(
            frame_det,
            score_threshold=config.object_score_threshold,
        )
        if len(object_bboxes) == 0:
            reason_counts["no_object_bbox"] += 1
            rejected_records.append({
                "frame": int(frame_idx),
                "hand": None,
                "status": "no_object_bbox",
                "failed_cell": "Cell 1-test",
                "discard_reason": "no_object_bbox",
            })
            continue

        selected_hand = None
        if hasattr(frame_det, "hands"):
            for hand in frame_det.hands:
                if (
                    hand.score > config.hand_score_threshold
                    and hand.state in [HandState.PORTABLE_OBJECT, HandState.STATIONARY_OBJECT]
                ):
                    selected_hand = hand
                    break

        if selected_hand is None:
            reason_counts["no_contact_hand"] += 1
            rejected_records.append({
                "frame": int(frame_idx),
                "hand": None,
                "status": "no_contact_hand",
                "failed_cell": "Cell 1-test",
                "discard_reason": "no_contact_hand",
            })
            continue

        side = selected_hand.side.name.lower()
        hand_bbox = [
            selected_hand.bbox.left,
            selected_hand.bbox.top,
            selected_hand.bbox.right,
            selected_hand.bbox.bottom,
        ]
        frame_path = image_dir / f"frame_{frame_idx + 1:010d}.jpg"
        img = cv2.imread(str(frame_path))
        if img is None:
            reason_counts["missing_frame"] += 1
            rejected_records.append({
                "frame": int(frame_idx),
                "hand": side,
                "status": "missing_frame",
                "failed_cell": "Cell 1-test",
                "discard_reason": "missing_frame",
            })
            continue

        extraction = extract_contact_points(
            img,
            hand_bbox,
            object_bboxes,
            config=config,
        )
        contact_count = int(len(extraction["contact_points"]))
        record = {
            "frame": int(frame_idx),
            "hand": side,
            "source": "filtered_contact_points",
            "note": "Contact frame that passed contact point extraction precheck",
            "num_objects": len(object_bboxes),
            "hand_bbox": hand_bbox,
            "active_object_bbox": extraction["active_object_bbox"],
            "skin_pixels": int(extraction["skin_pixels"]),
            "edge_pixels": int(extraction["edge_pixels"]),
            "raw_bbox_contact_points": int(len(extraction["raw_bbox_contact_points"])),
            "candidate_hand_edge_points": int(len(extraction["candidate_hand_edge_points"])),
            "contact_points": contact_count,
        }
        if contact_count >= 5:
            reason_counts["valid"] += 1
            record["status"] = "valid"
            valid_records.append(record)
        else:
            reason_counts["discard_contact_points"] += 1
            record["status"] = "discard_contact_points"
            record["failed_cell"] = "Cell 1-test"
            record["discard_reason"] = "discard_contact_points"
            rejected_records.append(record)

    reason_counts["scanned_frames"] = scanned_frames
    return valid_records, rejected_records, dict(reason_counts)


def fit_cell2_contact_gmm(
    detections,
    sample: Dict[str, Any],
    image_dir: Path = Path("data/P01_109_frames"),
    contact_extraction_config=None,
    n_components: int = 5,
) -> Dict[str, Any]:
    """Run the notebook's Cell 2 contact extraction and GMM gate for one frame."""
    import cv2
    import numpy as np
    from sklearn.mixture import GaussianMixture
    from .contact_point_utils import extract_contact_points, get_active_hand_bbox, get_valid_object_bboxes

    config = contact_extraction_config or _default_contact_config()
    image_dir = Path(image_dir)
    frame_idx = int(sample["frame"])
    active_hand = str(sample["hand"]).lower()
    frame_path = image_dir / f"frame_{frame_idx + 1:010d}.jpg"
    img = cv2.imread(str(frame_path))
    if img is None:
        return {
            "passed": False,
            "reason": "missing_frame",
            "contact_points": 0,
        }

    frame_det = detections[frame_idx]
    active_hand_bbox = get_active_hand_bbox(
        frame_det,
        active_hand,
        score_threshold=config.hand_score_threshold,
    )
    object_bboxes = get_valid_object_bboxes(
        frame_det,
        score_threshold=config.object_score_threshold,
    )
    if active_hand_bbox is None or len(object_bboxes) == 0:
        return {
            "passed": False,
            "reason": "missing_hand_or_object_bbox",
            "contact_points": 0,
        }

    extraction = extract_contact_points(
        img,
        active_hand_bbox,
        object_bboxes,
        config=config,
    )
    contact_points = extraction["contact_points"]
    contact_count = int(len(contact_points))
    if contact_count < int(n_components):
        return {
            "passed": False,
            "reason": "contact_points_lt5_after_object_boundary_filter",
            "contact_points": contact_count,
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gmm = GaussianMixture(n_components=int(n_components), random_state=42)
            gmm.fit(contact_points)
    except Exception as exc:  # pragma: no cover - defensive for numerical edge cases
        return {
            "passed": False,
            "reason": f"gmm_failed:{type(exc).__name__}",
            "contact_points": contact_count,
        }

    return {
        "passed": True,
        "reason": None,
        "contact_points": contact_count,
        "contact_means": np.asarray(gmm.means_, dtype=np.float32),
        "contact_weights": np.asarray(gmm.weights_, dtype=np.float32),
        "contact_covariances": np.asarray(gmm.covariances_, dtype=np.float32),
    }


def run_cell4_heatmap_gate(
    cell3_result: Dict[str, Any],
    contact_means,
    contact_weights,
    contact_covariances,
    image_dir: Path = Path("data/P01_109_frames"),
) -> Dict[str, Any]:
    """Run the notebook's Cell 4 heatmap-generation gate without saving images."""
    import cv2
    import numpy as np
    from .label_heatmap_utils import (
        build_label_heatmaps,
        merge_label_heatmaps,
        transform_covariances_by_homography,
    )

    if cell3_result.get("discard"):
        return {
            "passed": False,
            "reason": cell3_result.get("discard_reason") or "discarded_before_heatmap",
        }

    ref_idx = cell3_result.get("ref_idx")
    mu_transformed = cell3_result.get("mu_transformed")
    H_contact_to_ref = cell3_result.get("H_contact_to_ref")
    if ref_idx is None or mu_transformed is None:
        return {
            "passed": False,
            "reason": "missing_ref_or_transformed_contact_points",
        }

    image_dir = Path(image_dir)
    ref_path = image_dir / f"frame_{int(ref_idx) + 1:010d}.jpg"
    ref_img = cv2.imread(str(ref_path))
    if ref_img is None:
        return {
            "passed": False,
            "reason": "missing_reference_frame",
        }

    h_img, w_img = ref_img.shape[:2]
    heatmap_mode = "covariance"
    sigma_px = 12.0
    weights = contact_weights
    covariances_ref = None
    if contact_covariances is not None and contact_means is not None and H_contact_to_ref is not None:
        covariances_ref = transform_covariances_by_homography(
            contact_covariances,
            contact_means,
            H_contact_to_ref,
        )
    else:
        heatmap_mode = "isotropic"

    try:
        per_mode_heatmaps = build_label_heatmaps(
            image_shape=(h_img, w_img),
            centers_xy=np.asarray(mu_transformed, dtype=np.float32),
            sigma_px=sigma_px,
            weights=weights,
            normalize_each=True,
            covariances_xy=covariances_ref,
            covariance_scale=6.0,
            min_sigma_px=7.0,
            max_sigma_px=40.0,
        )
        merged_heatmap = merge_label_heatmaps(
            per_mode_heatmaps=per_mode_heatmaps,
            merge_method="sum",
            normalize_output=True,
        )
    except Exception as exc:  # pragma: no cover - defensive for numerical edge cases
        return {
            "passed": False,
            "reason": f"heatmap_failed:{type(exc).__name__}",
        }

    if not np.all(np.isfinite(merged_heatmap)) or float(np.max(merged_heatmap)) <= 0.0:
        return {
            "passed": False,
            "reason": "empty_or_invalid_heatmap",
        }

    return {
        "passed": True,
        "reason": None,
        "heatmap_mode": heatmap_mode,
        "heatmap_shape": tuple(int(v) for v in merged_heatmap.shape),
    }


def run_pipeline_retention_scan(
    detections,
    scan_limit: int = 500,
    image_dir: Path = Path("data/P01_109_frames"),
    output_dir: Path = Path("outputs/pipeline_retention_debug"),
    output_json_path: Optional[Path] = None,
    contact_extraction_config=None,
    precomputed_cell1_valid: Optional[Sequence[Dict[str, Any]]] = None,
    precomputed_cell1_rejected: Optional[Sequence[Dict[str, Any]]] = None,
    precomputed_cell1_summary: Optional[Dict[str, int]] = None,
    quiet: bool = True,
) -> Dict[str, Any]:
    """
    Run the current notebook data-processing pipeline on the first scan_limit frames.

    The returned summary is frame-level:
    Cell 1-test: raw scanned frames -> contact-point-valid frames
    Cell 2:      contact-point-valid frames -> GMM-valid frames
    Cell 3:      GMM-valid frames -> geometry/homography-valid frames
    Cell 4:      geometry-valid frames -> heatmap-generated frames
    """
    from .problem3_runner import run_problem3_cell3

    config = contact_extraction_config or _default_contact_config()
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    if precomputed_cell1_valid is not None:
        valid_cell1 = [dict(sample) for sample in precomputed_cell1_valid]
        rejected_cell1 = [dict(sample) for sample in (precomputed_cell1_rejected or [])]
        if precomputed_cell1_summary is None:
            scanned_frames = min(int(scan_limit), len(detections))
            cell1_reasons = {
                "scanned_frames": scanned_frames,
                "valid": len(valid_cell1),
                "discarded": scanned_frames - len(valid_cell1),
            }
        else:
            cell1_reasons = dict(precomputed_cell1_summary)
            cell1_reasons.setdefault("scanned_frames", min(int(scan_limit), len(detections)))
            cell1_reasons.setdefault("valid", len(valid_cell1))
    else:
        valid_cell1, rejected_cell1, cell1_reasons = scan_contact_point_frames(
            detections=detections,
            scan_limit=scan_limit,
            image_dir=image_dir,
            contact_extraction_config=config,
        )

    records: List[Dict[str, Any]] = []
    for item in rejected_cell1:
        records.append({
            "frame": item["frame"],
            "hand": item.get("hand"),
            "status": "discard",
            "failed_cell": "Cell 1-test",
            "discard_reason": item.get("discard_reason", item.get("status", "cell1_rejected")),
        })

    cell2_passed = []
    cell2_reasons: Counter = Counter()
    for sample in valid_cell1:
        cell2_result = fit_cell2_contact_gmm(
            detections=detections,
            sample=sample,
            image_dir=image_dir,
            contact_extraction_config=config,
        )
        if cell2_result["passed"]:
            cell2_passed.append((sample, cell2_result))
        else:
            cell2_reasons[cell2_result["reason"]] += 1
            records.append({
                "frame": sample["frame"],
                "hand": sample["hand"],
                "status": "discard",
                "failed_cell": "Cell 2",
                "discard_reason": cell2_result["reason"],
                "contact_points": cell2_result.get("contact_points", 0),
            })

    cell3_passed = []
    cell3_reasons: Counter = Counter()
    for sample, cell2_result in cell2_passed:
        cell3_result = _silent_call(
            run_problem3_cell3,
            detections=detections,
            t_contact=sample["frame"],
            active_hand=sample["hand"],
            contact_means=cell2_result["contact_means"],
            discard=False,
            discard_reason=None,
            image_dir=image_dir,
            output_dir=output_dir,
            quiet=quiet,
        )
        if not cell3_result.get("discard") and cell3_result.get("status") == "KEEP":
            cell3_passed.append((sample, cell2_result, cell3_result))
        else:
            reason = cell3_result.get("discard_reason") or "cell3_discard"
            cell3_reasons[reason] += 1
            records.append({
                "frame": sample["frame"],
                "hand": sample["hand"],
                "status": "discard",
                "failed_cell": "Cell 3",
                "discard_reason": reason,
                "contact_points": cell2_result.get("contact_points", 0),
                "ref_idx": cell3_result.get("ref_idx"),
            })

    cell4_passed = []
    cell4_reasons: Counter = Counter()
    for sample, cell2_result, cell3_result in cell3_passed:
        cell4_result = run_cell4_heatmap_gate(
            cell3_result=cell3_result,
            contact_means=cell2_result["contact_means"],
            contact_weights=cell2_result["contact_weights"],
            contact_covariances=cell2_result["contact_covariances"],
            image_dir=image_dir,
        )
        if cell4_result["passed"]:
            cell4_passed.append((sample, cell2_result, cell3_result, cell4_result))
            records.append({
                "frame": sample["frame"],
                "hand": sample["hand"],
                "status": "keep",
                "failed_cell": None,
                "discard_reason": None,
                "contact_points": cell2_result.get("contact_points", 0),
                "ref_idx": cell3_result.get("ref_idx"),
                "heatmap_mode": cell4_result.get("heatmap_mode"),
            })
        else:
            cell4_reasons[cell4_result["reason"]] += 1
            records.append({
                "frame": sample["frame"],
                "hand": sample["hand"],
                "status": "discard",
                "failed_cell": "Cell 4",
                "discard_reason": cell4_result["reason"],
                "contact_points": cell2_result.get("contact_points", 0),
                "ref_idx": cell3_result.get("ref_idx"),
            })

    scanned_frames = int(cell1_reasons.get("scanned_frames", min(int(scan_limit), len(detections))))
    summary = [
        {
            "cell": "Cell 1-test",
            "stage": "contact_point_precheck",
            "entered_frames": scanned_frames,
            "kept_frames": len(valid_cell1),
            "discarded_frames": scanned_frames - len(valid_cell1),
        },
        {
            "cell": "Cell 2",
            "stage": "contact_gmm",
            "entered_frames": len(valid_cell1),
            "kept_frames": len(cell2_passed),
            "discarded_frames": len(valid_cell1) - len(cell2_passed),
        },
        {
            "cell": "Cell 3",
            "stage": "homography_geometry",
            "entered_frames": len(cell2_passed),
            "kept_frames": len(cell3_passed),
            "discarded_frames": len(cell2_passed) - len(cell3_passed),
        },
        {
            "cell": "Cell 4",
            "stage": "label_heatmap",
            "entered_frames": len(cell3_passed),
            "kept_frames": len(cell4_passed),
            "discarded_frames": len(cell3_passed) - len(cell4_passed),
        },
    ]

    result = {
        "scan_limit": int(scan_limit),
        "scanned_frames": scanned_frames,
        "full_pipeline_frames": len(cell4_passed),
        "summary": summary,
        "reason_counts": {
            "Cell 1-test": {
                k: int(v)
                for k, v in cell1_reasons.items()
                if k != "scanned_frames"
            },
            "Cell 2": {k: int(v) for k, v in cell2_reasons.items()},
            "Cell 3": {k: int(v) for k, v in cell3_reasons.items()},
            "Cell 4": {k: int(v) for k, v in cell4_reasons.items()},
        },
        "records": sorted(records, key=lambda r: (int(r["frame"]), str(r.get("hand")))),
    }

    if output_json_path is not None:
        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(
            json.dumps(_json_safe(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return result
