# E17 near-contact reference baseline - 诊断实验报告

- video_id: `P01_109`
- subactions: first `100` annotations of this video
- output root: `outputs/e17`
- reference offsets: `[5, 6, 7, 8, 9, 10]`
- candidate offsets from episode start: `[0]`
- reference strategy: closest available pre-contact frame at offsets 5-10; humanless/active-hand-invisible/E14 clean gates are shadow-only.
- annotated frame intervals total: `15073` frames
- unique covered frames after overlap removal: `14330` frames

## E17 定义

- 一个 subaction 视作一个 training sequence，最多输出一个 final tuple。
- first contact timestep 是该 subaction 内最早 new contact run 的 start frame，active hand 使用该 run 的 hand。
- reference 从 episode start 前 5-10 帧里选最近可用帧，不再要求 humanless 或 active-hand-invisible。
- E14 clean gate、strict humanless、active-hand-invisible 只记录 shadow diagnostics，不进入主筛选。
- E17 是 near-contact diagnostic baseline，不是 paper-faithful humanless baseline。

## Funnel

| experiment_id | raw_candidate_total | episode_filtered_candidate_total | deep_run_candidate_total | cell2_pass | strict_ref_found | active_hand_invisible_ref_found | near_contact_ref_found | clean_ref_found | homography_available | geometry_pass | heatmap_pass | subactions_success | main_failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E17 | 24687 | 81 | 81 | 59 | 5 | 20 | 81 | 12 | 52 | 47 | 47 | 47 | continuation_contact_run |

## E17 vs E16 vs E15 vs E14 vs E6b vs E6c 对比

E6b/E6c 是 multi-candidate per subaction，candidate-level heatmap_pass 不能直接和 E17 的 candidate count 对齐；`subactions_success` 是更公平口径。E14/E15/E16/E17 中，E16 和 E17 都是 one candidate per subaction，最可比。

| experiment_id | subactions_success | heatmap_pass | candidates_total | episode_filtered_candidate_total | ref_found | strict_ref_found | active_hand_invisible_ref_found | near_contact_ref_found | clean_ref_found | homography_available | geometry_pass | homography_available_without_geometry_gate | reference_anchor_gap_median_keep | main_failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E16 | 26 | 26 | 100 | 81 | 38 | 27 | 38.0 |  | 0.0 | 29 | 26 |  |  | no_active_hand_invisible_reference |
| E15 | 14 | 14 | 100 | 81 | 27 | 27 |  |  | 0.0 | 16 | 14 |  |  | no_strict_humanless_reference |
| E14 | 20 | 20 | 100 | 81 | 23 | 3 |  |  | 23.0 | 21 | 20 |  |  | no_clean_pre_contact_reference |
| E6b | 18 | 74 | 784 | 293 | 99 | 46 |  |  |  | 85 | 74 |  |  | no_pre_episode_reference_window |
| E6c | 11 | 30 | 784 | 293 | 99 | 46 |  |  |  | 85 | 74 |  |  | no_pre_episode_reference_window |
| E17 | 47 | 47 | 100 | 81 | 81 | 5 | 20.0 | 81.0 | 12.0 | 52 | 47 | 52.0 | 5.0 | continuation_contact_run |

## 核心回答

- total success count: `47`
- candidates_total: `100`
- episode_filtered_candidate_total: `81`
- median reference_anchor_gap: `5.0`
- median ref_gap: `5.0`
- homography_available / geometry_pass / heatmap_pass: `52` / `47` / `47`
- homography_available_without_geometry_gate: `52`

## success 分布

- ref_gap stats: `{'count': 47, 'mean': 5.0, 'q1': 5.0, 'median': 5.0, 'q3': 5.0, 'min': 5.0, 'max': 5.0}`
- success reference_anchor_gap stats: `{'count': 47, 'mean': 5.0, 'q1': 5.0, 'median': 5.0, 'q3': 5.0, 'min': 5.0, 'max': 5.0}`
- strict shadow reference anchor gap stats: `{'count': 5, 'mean': 5.6, 'q1': 5.0, 'median': 5.0, 'q3': 5.0, 'min': 5.0, 'max': 8.0}`
- active-hand-invisible shadow anchor gap stats: `{'count': 20, 'mean': 5.7, 'q1': 5.0, 'median': 5.0, 'q3': 5.25, 'min': 5.0, 'max': 10.0}`
- clean shadow anchor gap stats: `{'count': 12, 'mean': 6.166666666666667, 'q1': 5.0, 'median': 5.0, 'q3': 7.25, 'min': 5.0, 'max': 9.0}`
- gap_length_before_episode_start stats: `{'count': 81, 'mean': 33.5679012345679, 'q1': 4.0, 'median': 14.0, 'q3': 34.0, 'min': 1.0, 'max': 291.0}`

| candidate_frame_offset_from_episode_start | count |
| --- | --- |
| 0.0 | 47.0 |

## 主要失败原因

| fail_reason | count |
| --- | --- |
| continuation_contact_run | 18 |
| contact_points_lt5_after_object_boundary_filter | 11 |
| missing_hand_or_object_bbox | 11 |
| trajectory_out_of_bounds_t+0 | 3 |
| pairwise_homography_low_quality_f630_to_f629 | 2 |
| pairwise_homography_failed_f4385_to_f4384 | 2 |
| pairwise_homography_low_quality_f93_to_f92 | 1 |
| pairwise_homography_low_quality_f161_to_f160 | 1 |
| trajectory_out_of_bounds_t+1 | 1 |
| pairwise_homography_low_quality_f779_to_f778 | 1 |
| no_contact_candidate | 1 |
| contact_point_out_of_bounds_mu_5 | 1 |

## 重点动作类别状态

| subaction_index | narration_id | narration | candidate_index | episode_id_within_subaction | frame_0_based | candidate_frame_offset_from_episode_start | hand | status | fail_reason | reference_mode | ref_idx | reference_anchor_gap | gap_length_before_episode_start |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | P01_109_11 | open freezer | 0.0 | 1.0 | 980.0 | 0.0 | right | discard | missing_hand_or_object_bbox | near_contact_pre_frame | 975.0 | 5.0 | 3.0 |
| 12 | P01_109_12 | open drawer | 0.0 | 1.0 | 1070.0 | 0.0 | left | keep |  | near_contact_pre_frame | 1065.0 | 5.0 | 255.0 |
| 17 | P01_109_17 | close freezer | 0.0 | 1.0 | 1422.0 | 0.0 | left | discard | trajectory_out_of_bounds_t+0 | near_contact_pre_frame | 1417.0 | 5.0 | 37.0 |
| 20 | P01_109_20 | close door | 0.0 | 1.0 | 1827.0 | 0.0 | left | keep |  | near_contact_pre_frame | 1822.0 | 5.0 | 291.0 |
| 24 | P01_109_24 | open drawer | 0.0 | 1.0 | 2325.0 | 0.0 | left | keep |  | near_contact_pre_frame | 2320.0 | 5.0 | 234.0 |
| 47 | P01_109_47 | close cupboard | 0.0 | 1.0 | 4648.0 | 0.0 | left | keep |  | near_contact_pre_frame | 4643.0 | 5.0 | 19.0 |
| 52 | P01_109_52 | open drawer | 0.0 | 1.0 | 4907.0 | 0.0 | right | discard | contact_points_lt5_after_object_boundary_filter | near_contact_pre_frame | 4902.0 | 5.0 | 1.0 |

## 图表

- pipeline funnel: `charts/e17_pipeline_funnel.png`
- failure reasons: `charts/e17_failure_reasons.png`
- success ref_gap histogram: `charts/e17_ref_gap_success_hist.png`
- success reference_anchor_gap histogram: `charts/e17_reference_anchor_gap_hist.png`
- success contact sheet: `charts/e17_success_contact_sheet_page_001.png`
- success heatmap+trajectory contact sheet: `charts/e17_success_heatmap_trajectory_contact_sheet_page_001.png`
- 排查汇总: `charts/排查汇总/`

## 明细文件

- experiment JSON: `experiment_results.json`
- overview CSV: `experiment_overview.csv`
- subaction summary CSV: `subaction_summary.csv`
- candidate diagnostics CSV: `candidate_diagnostics.csv`
- candidate plan CSV: `candidate_plan.csv`
- successful sample manifest: `successful_sample_manifest.csv`
- E17 vs E6b success table: `e17_success_vs_e6b.csv`
- E17 vs E6c success table: `e17_success_vs_e6c.csv`
- pipeline outputs: `experiments/E17/pipeline_outputs/`

## Interpretation

- E17 是 near-contact diagnostic baseline，不是 paper-faithful humanless baseline。
- 如果 E17 仍沿用当前 Cell2，那么 100 个 subaction 的成功上限大概率仍被 Cell2 限制在约 59，而不是 80。
