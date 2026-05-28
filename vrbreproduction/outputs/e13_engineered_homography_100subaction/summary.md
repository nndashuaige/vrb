# E13 engineered homography projection

## Result

- subactions evaluated: 100
- automatic gate successes: 31/100
- successful candidates / heatmap pass: 158
- reference policy: nearest pre-contact frame within max_ref_backtrack=60
- homography gate: good_matches>=12, inliers>=8, inlier_ratio>=0.25
- object gate: raw projected GMM means snapped to reference object mask, max snap distance <= 20px

Automatic success is not the same as human-usable success. E13 reached 31/100 automatically, but the manual audit below is the decision surface that matters.

## Baseline comparison

| experiment | reported_success | note |
| --- | --- | --- |
| E6b homography | 18/100 | task baseline |
| E6c conservative homography | 11/100 | task baseline |
| E9 part-aware LK | 74/100 | non-CoTracker upper reference |
| E10 CoTracker | 79/100 | not used by E13 |
| E12 clean reference + CoTracker | 79/100 | not used by E13 |
| E13 engineered homography | 31/100 | automatic gate before audit |

Judgment rule from the task: 50+/100 supports a cheap homography first pass; 20-30/100 points toward a cascade.

## Pipeline overview

| metric | value |
| --- | --- |
| raw_candidate_total | 24687 |
| deep_run_candidate_total | 765 |
| cell2_pass | 563 |
| ref_found | 280 |
| homography_available | 278 |
| geometry_pass | 158 |
| heatmap_pass | 158 |
| main_failure | no_pre_episode_reference_window |

## Manual audit

30 automatic successes were manually reviewed from `charts/manual_audit_sheets/success_page_*.png`.
20 failures were sampled from `charts/manual_audit_sheets/failure_page_*.png`; most fail before final projection output, so failure-side visual judgment is limited and was recorded as `unclear`.

| success audit label | count |
| --- | --- |
| accurate_same_part | 19 |
| same_object_but_shifted | 6 |
| wrong_object | 5 |
| background_or_empty | 0 |
| unclear | 0 |

- strict same-part usable rate on audited successes: 19/30 = 63.3%
- relaxed same-object usable rate on audited successes: 25/30 = 83.3%
- rough strict usable estimate over the 31 automatic successes: 19.6/100
- rough relaxed usable estimate over the 31 automatic successes: 25.8/100

On this audit, E13 looks closer to the task's 20-30/100 cascade regime than to a standalone 50+/100 cheap-pass win.

## Failure audit note

- failure audit counts: {'unclear': 20}
- These 20 failures should be read as gate failures, not projection-quality failures, because the pipeline usually stopped before a final snapped output existed.

## Homography diagnostics

- feature methods among candidates: {'none': 506, 'SIFT': 278}

## Top failure reasons

| fail_reason | count |
| --- | --- |
| no_pre_episode_reference_window | 283 |
| contact_points_lt5_after_object_boundary_filter | 97 |
| missing_hand_or_object_bbox | 97 |
| no_reference_object_bbox | 79 |
| projected_contact_too_far_from_reference_object_mask | 41 |
| continuation_contact_run | 18 |
| gmm_failed:ValueError | 8 |
| no_smoothed_contact_in_subaction | 1 |
| pairwise_homography_failed_f3856_to_f3855 | 1 |
| pairwise_homography_failed_f3857_to_f3856 | 1 |

## Artifacts

- `experiment_overview.csv`
- `subaction_summary.csv`
- `candidate_diagnostics.csv`
- `candidate_plan.csv`
- `successful_sample_manifest.csv`
- `manual_audit_sample_plan.csv`
- `manual_audit_completed.csv`
- `charts/e13_pipeline_funnel.png`
- `charts/e13_failure_reasons.png`
- `charts/e13_success_contact_sheet_page_001.png`
- `charts/e13_homography_match_sheet.png`
- `charts/manual_audit_sheets/`
