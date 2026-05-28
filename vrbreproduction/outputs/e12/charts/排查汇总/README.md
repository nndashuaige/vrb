# E12 排查汇总

内容：

- `e12_detected_objects_page_*.png`: contact frame 上的 object detection bbox，绿色为 HOA active object，青色虚线为 E12 记录的 active object。
- `e12_detected_hands_page_*.png`: contact frame 上的 hand detection bbox，蓝色为 active hand。
- `e12_contact_frame_contact_points_page_*.png`: contact frame 上的 active hand、active object、raw/contact points 和重新拟合的 5 个 GMM means。
- `e12_reference_frames_page_*.png`: E12 最终选择的 reference frame，青色框为 tracked reference object bbox。
- `e12_debug_funnel_table_subaction.*`: subaction 级主漏斗表，E12 全部候选口径。
- `e12_debug_funnel_table_subaction_new_contact_only.*`: subaction 级，仅 new-contact candidate 口径。
- `e12_debug_funnel_table_candidate.*`: candidate/tuple 级主漏斗表，E12 全部候选口径。
- `e12_debug_funnel_table_candidate_new_contact_only.*`: candidate/tuple 级，仅 new-contact candidate 口径。
- `e12_final_tuples_by_candidate_source.csv`: final tuple 来源分布。

## Subaction-level funnel: E12 all candidates

一个 subaction 只要至少有一个候选过该阶段，就计入该阶段。E12 包含 `new_contact_episode` 和 `continuation_entry_rescue`。

| 阶段 | 剩余 |
| --- | ---: |
| 100 subactions | 100 |
| 有 new-contact candidate | 81 |
| Cell2 contact/GMM pass | 91 |
| reference found | 79 |
| homography available | 79 |
| final keep | 79 |

## Subaction-level funnel: new-contact only

完全不用 `continuation_entry_rescue`，只统计 `candidate_source == new_contact_episode`。

| 阶段 | 剩余 |
| --- | ---: |
| 100 subactions | 100 |
| 有 new-contact candidate | 81 |
| Cell2 contact/GMM pass | 73 |
| reference found | 63 |
| homography available | 63 |
| final keep | 63 |

## Candidate-level funnel: E12 all candidates

| 阶段 | 数量 |
| --- | ---: |
| candidate rows | 624 |
| Cell2 contact/GMM pass | 460 |
| reference found | 365 |
| homography available | 365 |
| final tuples | 156 |

## Candidate-level funnel: new-contact only

| 阶段 | 数量 |
| --- | ---: |
| candidate rows | 556 |
| Cell2 contact/GMM pass | 398 |
| reference found | 321 |
| homography available | 321 |
| final tuples | 140 |

## Final tuple source

| candidate_source | final tuples |
| --- | ---: |
| new_contact_episode | 140 |
| continuation_entry_rescue | 16 |

说明：

- 本脚本只生成排查图，不重跑 E12 CoTracker 或投影。
- contact point 图中的 GMM means 是按 E12 同一 Cell2 逻辑在 contact frame 上重新拟合，用于排查节点可视化。
