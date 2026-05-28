# E15 paper-faithful reference baseline - 诊断实验报告

- video_id: `P01_109`
- subactions: first `100` annotations of this video
- output root: `outputs/e15`
- max ref backtrack: `360` frames
- candidate offsets from episode start: `[0]`
- reference gate: `strict_global_no_hand` only; E14 clean hand-object/crop gates are diagnostics only and not used for keep.
- annotated frame intervals total: `15073` frames
- unique covered frames after overlap removal: `14330` frames

## E15 定义

- 一个 subaction 视作一个 training sequence，最多输出一个 final tuple。
- first contact timestep 是该 subaction 内最早 new contact run 的 start frame，active hand 使用该 run 的 hand。
- reference 必须在 episode start 之前，并只接受最近的 `strict_global_no_hand` frame。
- 找不到 strict humanless reference 时主结果直接失败，不用 E14 clean reference 救回。
- E15 是 paper-faithful baseline，不是 clean reference 优化版。

## Funnel

| experiment_id | raw_candidate_total | episode_filtered_candidate_total | deep_run_candidate_total | cell2_pass | strict_ref_found | clean_ref_found | homography_available | geometry_pass | heatmap_pass | subactions_success | main_failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E15 | 24687 | 81 | 81 | 59 | 27 | 0 | 16 | 14 | 14 | 14 | no_strict_humanless_reference |

## E15 vs E14 vs E6b vs E6c 对比

E6b/E6c 是 multi-candidate per subaction，candidate-level heatmap_pass 不能直接和 E15 的 candidate count 对齐；`subactions_success` 是更公平口径。E14 和 E15 都是 one candidate per subaction，更可比。

| experiment_id | subactions_success | heatmap_pass | candidates_total | episode_filtered_candidate_total | ref_found | strict_ref_found | clean_ref_found | homography_available | geometry_pass | reference_anchor_gap_median_keep | main_failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E14 | 20 | 20 | 100 | 81 | 23 | 3 | 23.0 | 21 | 20 | 10 | no_clean_pre_contact_reference |
| E6b | 18 | 74 | 784 | 293 | 99 | 46 |  | 85 | 74 | 2 | no_pre_episode_reference_window |
| E6c | 11 | 30 | 784 | 293 | 99 | 46 |  | 85 | 74 | 5 | no_pre_episode_reference_window |
| E15 | 14 | 14 | 100 | 81 | 27 | 27 | 0.0 | 16 | 14 | 42 | no_strict_humanless_reference |

## 核心回答

- total success count: `14`
- median reference_anchor_gap: `42.0`
- median ref_gap: `42.0`
- homography_available / geometry_pass / heatmap_pass: `16` / `14` / `14`
- no_pre_episode_reference_window_not_rescued: `0`

## success 分布

- strict-global-no-hand ref_gap stats: `{'count': 14, 'mean': 71.57142857142857, 'q1': 13.5, 'median': 42.0, 'q3': 107.5, 'min': 2.0, 'max': 274.0}`
- success reference_anchor_gap stats: `{'count': 14, 'mean': 71.57142857142857, 'q1': 13.5, 'median': 42.0, 'q3': 107.5, 'min': 2.0, 'max': 274.0}`
- strict reference anchor gap stats: `{'count': 27, 'mean': 62.074074074074076, 'q1': 15.0, 'median': 34.0, 'q3': 83.0, 'min': 2.0, 'max': 274.0}`
- gap_length_before_episode_start stats: `{'count': 81, 'mean': 33.5679012345679, 'q1': 4.0, 'median': 14.0, 'q3': 34.0, 'min': 1.0, 'max': 291.0}`

| candidate_frame_offset_from_episode_start | count |
| --- | --- |
| 0.0 | 14.0 |

## 主要失败原因

| fail_reason | count |
| --- | --- |
| no_strict_humanless_reference | 32 |
| continuation_contact_run | 18 |
| contact_points_lt5_after_object_boundary_filter | 11 |
| missing_hand_or_object_bbox | 11 |
| pairwise_homography_low_quality_f630_to_f629 | 2 |
| pairwise_homography_failed_f2305_to_f2304 | 2 |
| pairwise_homography_failed_f4385_to_f4384 | 2 |
| pairwise_homography_low_quality_f93_to_f92 | 1 |
| pairwise_homography_low_quality_f161_to_f160 | 1 |
| trajectory_out_of_bounds_t+3 | 1 |
| pairwise_homography_low_quality_f779_to_f778 | 1 |
| trajectory_out_of_bounds_t+0 | 1 |
| no_smoothed_contact_in_subaction | 1 |
| pairwise_homography_failed_f3541_to_f3540 | 1 |
| pairwise_homography_failed_f3646_to_f3645 | 1 |

## 重点动作类别状态

| subaction_index | narration_id | narration | candidate_index | episode_id_within_subaction | frame_0_based | candidate_frame_offset_from_episode_start | hand | status | fail_reason | reference_mode | ref_idx | reference_anchor_gap | gap_length_before_episode_start |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | P01_109_11 | open freezer | 0.0 | 1.0 | 980.0 | 0.0 | right | discard | missing_hand_or_object_bbox |  |  |  | 3.0 |
| 12 | P01_109_12 | open drawer | 0.0 | 1.0 | 1070.0 | 0.0 | left | discard | no_strict_humanless_reference | no_strict_humanless_reference |  |  | 255.0 |
| 17 | P01_109_17 | close freezer | 0.0 | 1.0 | 1422.0 | 0.0 | left | discard | no_strict_humanless_reference | no_strict_humanless_reference |  |  | 37.0 |
| 20 | P01_109_20 | close door | 0.0 | 1.0 | 1827.0 | 0.0 | left | discard | trajectory_out_of_bounds_t+0 | strict_global_no_hand | 1798.0 | 29.0 | 291.0 |
| 24 | P01_109_24 | open drawer | 0.0 | 1.0 | 2325.0 | 0.0 | left | discard | pairwise_homography_failed_f2305_to_f2304 | strict_global_no_hand | 2257.0 | 68.0 | 234.0 |
| 47 | P01_109_47 | close cupboard | 0.0 | 1.0 | 4648.0 | 0.0 | left | keep |  | strict_global_no_hand | 4636.0 | 12.0 | 19.0 |
| 52 | P01_109_52 | open drawer | 0.0 | 1.0 | 4907.0 | 0.0 | right | discard | contact_points_lt5_after_object_boundary_filter |  |  |  | 1.0 |

## 图表

- pipeline funnel: `charts/e15_pipeline_funnel.png`
- failure reasons: `charts/e15_failure_reasons.png`
- success ref_gap histogram: `charts/e15_ref_gap_success_hist.png`
- success reference_anchor_gap histogram: `charts/e15_reference_anchor_gap_hist.png`
- success contact sheet: `charts/e15_success_contact_sheet_page_001.png`
- success heatmap+trajectory contact sheet: `charts/e15_success_heatmap_trajectory_contact_sheet_page_001.png`

## 明细文件

- experiment JSON: `experiment_results.json`
- overview CSV: `experiment_overview.csv`
- subaction summary CSV: `subaction_summary.csv`
- candidate diagnostics CSV: `candidate_diagnostics.csv`
- candidate plan CSV: `candidate_plan.csv`
- successful sample manifest: `successful_sample_manifest.csv`
- E15 vs E6b success table: `e15_success_vs_e6b.csv`
- E15 vs E6c success table: `e15_success_vs_e6c.csv`
- pipeline outputs: `experiments/E15/pipeline_outputs/`
