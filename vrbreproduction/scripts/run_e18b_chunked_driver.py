"""Chunked driver for E18B: process subactions until deadline, persist per-subaction pickles."""
import pickle, sys, time
from pathlib import Path

sys.path.insert(0, str(Path("~/mnt/vrbreproduction/src").expanduser()))

from vrbreproduction.e18b_paper_faithful_cell2_retry import (
    E18BConfig, EXPERIMENT, ContactExtractionConfig, Problem3CachedRunnerE18B,
    build_contact_arrays, build_new_contact_runs, get_subactions,
    contact_frame_candidates_for_subaction, runs_for_subaction, subaction_bounds,
    run_one_candidate, empty_subaction_record, VRBREPRODUCTION_ROOT,
)
from epic_kitchens.hoa import load_detections

STATE_DIR = Path("/tmp/e18b_state")
STATE_DIR.mkdir(exist_ok=True)
t_deadline = time.time() + (float(sys.argv[1]) if len(sys.argv) > 1 else 30.0)

config = E18BConfig()
config.output_root.mkdir(parents=True, exist_ok=True)
(VRBREPRODUCTION_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)

detections = load_detections(str(config.hoa_pkl))
subactions = get_subactions(config)
left_binary, right_binary = build_contact_arrays(detections)
all_runs = build_new_contact_runs(left_binary, right_binary)
contact_config = ContactExtractionConfig(
    use_object_mask=True, use_object_boundary=True,
    boundary_distance_px=6.0, project_points_to_object_boundary=True,
)
runner = Problem3CachedRunnerE18B(
    detections, config.image_dir, left_binary, right_binary,
    hand_object_iou_max=config.hand_object_iou_max,
    min_hand_object_center_distance_px=config.min_hand_object_center_distance_px,
)

done, todo = [], []
for idx, row in subactions.iterrows():
    f = STATE_DIR / f"sub_{int(idx):03d}.pkl"
    (done if f.exists() else todo).append(int(idx))

processed = 0
for idx in todo:
    if time.time() > t_deadline:
        break
    row = subactions.loc[idx]
    start_0, stop_0 = subaction_bounds(row, len(detections))
    raw_candidates = contact_frame_candidates_for_subaction(row, left_binary, right_binary, len(detections))
    sub_runs, episode_decision, episode_reason = runs_for_subaction(
        row, all_runs, left_binary, right_binary, len(detections),
        config.early_candidate_offsets, config.max_ref_backtrack,
    )
    flattened = [c for run in sub_runs for c in run["candidates"]]
    plan_row = {
        "experiment_id": EXPERIMENT["experiment_id"],
        "subaction_index": int(idx),
        "narration_id": row["narration_id"], "narration": row["narration"],
        "verb": row["verb"], "noun": row["noun"], "all_nouns": row.get("all_nouns"),
        "start_frame_0_based": int(start_0), "stop_frame_0_based": int(stop_0),
        "raw_candidate_count": int(len(raw_candidates)),
        "episode_filtered_candidate_count": int(len(sub_runs)),
        "deep_run_candidate_count": int(len(flattened)),
        "episode_decision": episode_decision, "episode_reason": episode_reason,
        "subaction_new_contact_count": int(len(sub_runs)),
        "episode_start_frames_0_based": ";".join(str(int(r["start_frame_0_based"])) for r in sub_runs),
        "episode_hands": ";".join(str(r["hand"]) for r in sub_runs),
    }
    records = []
    if not flattened:
        records.append(empty_subaction_record(
            row=row, start_0=start_0, stop_0=stop_0,
            raw_candidate_count=len(raw_candidates),
            subaction_new_contact_count=len(sub_runs),
            episode_decision=episode_decision, episode_reason=episode_reason,
        ))
    else:
        for ci, cand in enumerate(flattened):
            records.append(run_one_candidate(
                row=row, candidate=cand, candidate_index=ci, detections=detections,
                contact_config=contact_config, problem3_runner=runner, config=config,
                subaction_new_contact_count=len(sub_runs), raw_candidate_count=len(raw_candidates),
            ))
    with open(STATE_DIR / f"sub_{int(idx):03d}.pkl", "wb") as f:
        pickle.dump({"plan": plan_row, "records": records}, f)
    processed += 1
    ks = sum(1 for r in records if r.get("status") == "keep")
    print(f"sub {idx:03d} {row['narration_id']} keep={ks}", flush=True)

print(f"PROGRESS processed={processed} done_total={len(done)+processed} remaining={len(todo)-processed}")
