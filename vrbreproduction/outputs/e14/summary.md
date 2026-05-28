# E14 clean pre-contact reference - 诊断实验报告

- video_id: `P01_109`
- subactions: first `100` annotations of this video
- output root: `/Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/outputs/e14`
- max ref backtrack: `360` frames
- candidate offsets from episode start: `[0]`
- clean crop gate: `150x150`, hand overlap <= `0.05`
- hand-object IoU max: `0.05`
- hand-object center distance min: `None`
- annotated frame intervals total: `15073` frames
- unique covered frames after overlap removal: `14330` frames

## E14 定义

- 每个 subaction 只保留第一个 `new contact episode` 的 `t0` 作为主样本。
- reference 仅在当前 episode 前的 contiguous no-contact gap 内向后扫描。
- 优先寻找 active hand 可见但尚未接触、且对目标物 / contact crop 遮挡低的最近 clean frame。
- 如果 active hand 不可见，也允许 `closest_clean_pre_contact_humanless` 作为次级模式。
- `strict_global_no_hand` 只做 shadow diagnostics，不参与最终选样。

## Funnel

| experiment_id | raw_candidate_total | episode_filtered_candidate_total | deep_run_candidate_total | cell2_pass | strict_ref_found | clean_ref_found | homography_available | geometry_pass | heatmap_pass | subactions_success | main_failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E14 | 24687 | 81 | 81 | 59 | 3 | 23 | 21 | 20 | 20 | 20 | no_clean_pre_contact_reference |

## E6b / E6c / E14 对比

| experiment_id | subactions_success | strict_ref_found | clean_ref_found | heatmap_pass | homography_available | geometry_pass | main_failure |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E6b | 18 | 46 |  | 74 | 85 | 74 | no_pre_episode_reference_window |
| E6c | 11 | 46 |  | 30 | 85 | 74 | no_pre_episode_reference_window |
| E14 | 20 | 3 | 23.0 | 20 | 21 | 20 | no_clean_pre_contact_reference |

## 核心回答

- active-hand-visible success count: `8`
- humanless success count: `12`
- total success count: `20`
- median reference_anchor_gap: `10.0`
- median strict shadow anchor gap: `21.0`
- median anchor gap delta vs strict: `-6.0`
- no_pre_episode_reference_window_not_rescued: `0`

## success 分布

- visible-mode ref_gap stats: `{'count': 8, 'mean': 13.625, 'q1': 6.75, 'median': 10.0, 'q3': 20.5, 'min': 1.0, 'max': 33.0}`
- humanless-mode ref_gap stats: `{'count': 12, 'mean': 19.0, 'q1': 2.0, 'median': 10.5, 'q3': 17.0, 'min': 1.0, 'max': 105.0}`
- success reference_anchor_gap stats: `{'count': 20, 'mean': 16.85, 'q1': 2.75, 'median': 10.0, 'q3': 20.0, 'min': 1.0, 'max': 105.0}`
- strict shadow anchor gap stats: `{'count': 3, 'mean': 30.0, 'q1': 21.0, 'median': 21.0, 'q3': 34.5, 'min': 21.0, 'max': 48.0}`
- gap_length_before_episode_start stats: `{'count': 81, 'mean': 33.5679012345679, 'q1': 4.0, 'median': 14.0, 'q3': 34.0, 'min': 1.0, 'max': 291.0}`

| candidate_frame_offset_from_episode_start | count |
| --- | --- |
| 0.0 | 20.0 |

## 主要失败原因

| fail_reason | count |
| --- | --- |
| no_clean_pre_contact_reference | 36 |
| continuation_contact_run | 18 |
| contact_points_lt5_after_object_boundary_filter | 11 |
| missing_hand_or_object_bbox | 11 |
| pairwise_homography_failed_f4385_to_f4384 | 2 |
| trajectory_out_of_bounds_t+3 | 1 |
| no_smoothed_contact_in_subaction | 1 |

## 重点动作类别状态

| subaction_index | narration_id | narration | candidate_index | episode_id_within_subaction | frame_0_based | candidate_frame_offset_from_episode_start | hand | status | fail_reason | reference_mode | ref_idx | reference_anchor_gap | gap_length_before_episode_start |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | P01_109_11 | open freezer | 0.0 | 1.0 | 980.0 | 0.0 | right | discard | missing_hand_or_object_bbox |  |  |  | 3.0 |
| 12 | P01_109_12 | open drawer | 0.0 | 1.0 | 1070.0 | 0.0 | left | keep |  | closest_clean_pre_contact_active_hand_visible | 1069.0 | 1.0 | 255.0 |
| 17 | P01_109_17 | close freezer | 0.0 | 1.0 | 1422.0 | 0.0 | left | keep |  | closest_clean_pre_contact_humanless | 1420.0 | 2.0 | 37.0 |
| 20 | P01_109_20 | close door | 0.0 | 1.0 | 1827.0 | 0.0 | left | keep |  | closest_clean_pre_contact_humanless | 1818.0 | 9.0 | 291.0 |
| 24 | P01_109_24 | open drawer | 0.0 | 1.0 | 2325.0 | 0.0 | left | keep |  | closest_clean_pre_contact_humanless | 2311.0 | 14.0 | 234.0 |
| 47 | P01_109_47 | close cupboard | 0.0 | 1.0 | 4648.0 | 0.0 | left | keep |  | closest_clean_pre_contact_humanless | 4636.0 | 12.0 | 19.0 |
| 52 | P01_109_52 | open drawer | 0.0 | 1.0 | 4907.0 | 0.0 | right | discard | contact_points_lt5_after_object_boundary_filter |  |  |  | 1.0 |

## 图表

- pipeline funnel: `charts/e14_pipeline_funnel.png`
- failure reasons: `charts/e14_failure_reasons.png`
- success ref_gap histogram: `charts/e14_ref_gap_success_hist.png`
- success reference_anchor_gap histogram: `charts/e14_reference_anchor_gap_hist.png`
- success contact sheet: `charts/e14_success_contact_sheet_page_001.png`
- success heatmap+trajectory contact sheet: `charts/e14_success_heatmap_trajectory_contact_sheet_page_001.png`

## 明细文件

- experiment JSON: `experiment_results.json`
- overview CSV: `experiment_overview.csv`
- subaction summary CSV: `subaction_summary.csv`
- candidate diagnostics CSV: `candidate_diagnostics.csv`
- candidate plan CSV: `candidate_plan.csv`
- successful sample manifest: `successful_sample_manifest.csv`
- E14 vs E6b success table: `e14_success_vs_e6b.csv`
- E14 vs E6c success table: `e14_success_vs_e6c.csv`
- pipeline outputs: `experiments/E14/pipeline_outputs/`
