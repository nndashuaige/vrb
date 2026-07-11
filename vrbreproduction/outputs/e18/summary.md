# E18 hybrid active-hand-invisible plus min-IoU fallback reference baseline - 诊断实验报告

- video_id: `P01_109`
- subactions: first `100` annotations of this video
- output root: `/Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/outputs/e18`
- max ref backtrack: `360` frames
- candidate offsets from episode start: `[0]`
- reference gate: E16-style pre-contact backtrack window; first choose `active_hand_invisible_pre_contact`, otherwise fallback to `min_hand_object_iou_pre_contact_fallback`.
- annotated frame intervals total: `15073` frames
- unique covered frames after overlap removal: `14330` frames

## E18 定义

- 一个 subaction 视作一个 training sequence，最多输出一个 final tuple。
- first contact timestep 是该 subaction 内最早 new contact run 的 start frame，active hand 使用该 run 的 hand。
- reference 在 episode start 之前搜索，最多向前回看 `max_ref_backtrack` 帧，和 E16 的窗口逻辑一致。
- 先找最近的 active hand 不可见帧；如果找不到，再在窗口内选 hand-object IoU 最小的帧。
- 找不到 active-hand-invisible frame 时，不再使用手工 `5-10` 帧 offset，而是使用自动 fallback。
- strict global no-hand 仍然只作为 shadow diagnostics 记录。

## Funnel

| experiment_id | raw_candidate_total | episode_filtered_candidate_total | deep_run_candidate_total | cell2_pass | strict_ref_found | active_hand_invisible_ref_found | fallback_ref_found | fallback_ref_keep | clean_ref_found | homography_available | geometry_pass | heatmap_pass | subactions_success | main_failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E18 | 24687 | 81 | 81 | 58 | 27 | 38 | 20 | 19 | 0 | 49 | 45 | 45 | 45 | continuation_contact_run |

## E18 vs E15 vs E14 vs E6b vs E6c 对比

E6b/E6c 是 multi-candidate per subaction，candidate-level heatmap_pass 不能直接和 E18 的 candidate count 对齐；`subactions_success` 是更公平口径。E14/E15/E18 都是 one candidate per subaction，更可比。

| experiment_id | subactions_success | heatmap_pass | candidates_total | episode_filtered_candidate_total | ref_found | strict_ref_found | active_hand_invisible_ref_found | fallback_ref_found | fallback_ref_keep | clean_ref_found | homography_available | geometry_pass | reference_anchor_gap_median_keep | main_failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E15 | 14 | 14 | 100 | 81 | 27 | 27 |  | 0 |  | 0 | 16 | 14 | 42 | no_strict_humanless_reference |
| E14 | 20 | 20 | 100 | 81 | 23 | 3 |  | 0 |  | 23 | 21 | 20 | 10 | no_clean_pre_contact_reference |
| E18 | 45 | 45 | 100 | 81 | 58 | 27 | 38.0 | 20 | 19.0 | 0 | 49 | 45 | 14 | continuation_contact_run |

## 核心回答

- total success count: `45`
- median reference_anchor_gap: `14.0`
- median ref_gap: `14.0`
- active_hand_invisible_ref_found / fallback_ref_found / fallback_ref_keep: `38` / `20` / `19`
- homography_available / geometry_pass / heatmap_pass: `49` / `45` / `45`
- no_pre_episode_reference_window_not_rescued: `0`

## success 分布

- active-hand-invisible ref_gap stats: `{'count': 45, 'mean': 48.733333333333334, 'q1': 2.0, 'median': 14.0, 'q3': 68.0, 'min': 1.0, 'max': 326.0}`
- success reference_anchor_gap stats: `{'count': 45, 'mean': 48.733333333333334, 'q1': 2.0, 'median': 14.0, 'q3': 68.0, 'min': 1.0, 'max': 326.0}`
- fallback selected hand-object IoU stats: `{'count': 20, 'mean': 0.0, 'q1': 0.0, 'median': 0.0, 'q3': 0.0, 'min': 0.0, 'max': 0.0}`
- strict shadow reference anchor gap stats: `{'count': 27, 'mean': 62.074074074074076, 'q1': 15.0, 'median': 34.0, 'q3': 83.0, 'min': 2.0, 'max': 274.0}`
- gap_length_before_episode_start stats: `{'count': 81, 'mean': 33.5679012345679, 'q1': 4.0, 'median': 14.0, 'q3': 34.0, 'min': 1.0, 'max': 291.0}`

| candidate_frame_offset_from_episode_start | count |
| --- | --- |
| 0.0 | 45.0 |

## 主要失败原因

| fail_reason | count |
| --- | --- |
| continuation_contact_run | 18 |
| contact_points_lt5_after_object_boundary_filter | 11 |
| missing_hand_or_object_bbox | 11 |
| pairwise_homography_low_quality_f630_to_f629 | 2 |
| trajectory_out_of_bounds_t+0 | 2 |
| pairwise_homography_failed_f4385_to_f4384 | 2 |
| pairwise_homography_low_quality_f93_to_f92 | 1 |
| pairwise_homography_low_quality_f161_to_f160 | 1 |
| trajectory_out_of_bounds_t+3 | 1 |
| pairwise_homography_low_quality_f779_to_f778 | 1 |
| no_smoothed_contact_in_subaction | 1 |
| pairwise_homography_failed_f3541_to_f3540 | 1 |
| pairwise_homography_failed_f3646_to_f3645 | 1 |
| gmm_failed:ValueError | 1 |
| contact_point_out_of_bounds_mu_5 | 1 |

## 重点动作类别状态

| subaction_index | narration_id | narration | candidate_index | episode_id_within_subaction | frame_0_based | candidate_frame_offset_from_episode_start | hand | status | fail_reason | reference_mode | ref_idx | reference_anchor_gap | gap_length_before_episode_start |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | P01_109_11 | open freezer | 0.0 | 1.0 | 980.0 | 0.0 | right | discard | missing_hand_or_object_bbox |  |  |  | 3.0 |
| 12 | P01_109_12 | open drawer | 0.0 | 1.0 | 1070.0 | 0.0 | left | keep |  | active_hand_invisible_pre_contact | 1068.0 | 2.0 | 255.0 |
| 17 | P01_109_17 | close freezer | 0.0 | 1.0 | 1422.0 | 0.0 | left | keep |  | active_hand_invisible_pre_contact | 1420.0 | 2.0 | 37.0 |
| 20 | P01_109_20 | close door | 0.0 | 1.0 | 1827.0 | 0.0 | left | keep |  | active_hand_invisible_pre_contact | 1826.0 | 1.0 | 291.0 |
| 24 | P01_109_24 | open drawer | 0.0 | 1.0 | 2325.0 | 0.0 | left | keep |  | active_hand_invisible_pre_contact | 2311.0 | 14.0 | 234.0 |
| 47 | P01_109_47 | close cupboard | 0.0 | 1.0 | 4648.0 | 0.0 | left | keep |  | active_hand_invisible_pre_contact | 4636.0 | 12.0 | 19.0 |
| 52 | P01_109_52 | open drawer | 0.0 | 1.0 | 4907.0 | 0.0 | right | discard | contact_points_lt5_after_object_boundary_filter |  |  |  | 1.0 |

## 图表

- pipeline funnel: `charts/e18_pipeline_funnel.png`
- failure reasons: `charts/e18_failure_reasons.png`
- success ref_gap histogram: `charts/e18_ref_gap_success_hist.png`
- success reference_anchor_gap histogram: `charts/e18_reference_anchor_gap_hist.png`
- success contact sheet: `charts/e18_success_contact_sheet_page_001.png`
- success heatmap+trajectory contact sheet: `charts/e18_success_heatmap_trajectory_contact_sheet_page_001.png`

## 明细文件

- experiment JSON: `experiment_results.json`
- overview CSV: `experiment_overview.csv`
- subaction summary CSV: `subaction_summary.csv`
- candidate diagnostics CSV: `candidate_diagnostics.csv`
- candidate plan CSV: `candidate_plan.csv`
- successful sample manifest: `successful_sample_manifest.csv`
- E18 vs E6b success table: `e18_success_vs_e6b.csv`
- E18 vs E6c success table: `e18_success_vs_e6c.csv`
- pipeline outputs: `experiments/E18/pipeline_outputs/`
