# E10 contact-part-aware CoTracker projection 实验报告

- 输出目录：`/root/workspace/vrb/vrbreproduction/outputs/100subaction-e10`
- video_id：`P01_109`
- subaction（子动作片段）总数：`100`
- 成功覆盖 subaction（子动作片段）：`79`
- final tuple（最终训练样本）：`157`
- object mask gate 通过数：`365`
- part gate 通过数：`365`
- whole-object gate fallback 通过数：`0`
- CoTracker 成功数：`365`
- fallback tracker 成功数：`0`
- fallback tracker 比例：`0.00%`
- CoTracker 初始化状态：`ready`
- CoTracker 初始化错误：`None`
- articulated object 成功数：`56`
- articulated object 失败数：`48`

## 算法改动

- contact frame 先得到 whole-object mask，再基于 contact GMM、raw contact points、hand bbox 和 noun soft prior 构造 `contact_part_mask`。
- 跟踪点采样改成 `contact_part_mask` 优先，只在局部点不足时补 whole-object boundary，避免 interior 大面积主导局部运动。
- 投影优先使用 `contact_part_local_affine`；若局部仿射不稳定，再退回 `whole_object_affine_fallback`，并在 diagnostics 中明确记录。
- 最终 gate 先看 projected contact 是否落在 reference contact part 上；只有小物体等特例才允许 whole-object gate fallback。
- CoTracker 初始化状态与错误会单独记录；`fallback_lk` 不会被伪装成 CoTracker。
- 最终导出允许每个 subaction（子动作片段）最多保留 `3` 个高分 tuple（训练样本），并做 `2` 帧的近邻去重。

## 与 E6b / E6c 的关键区别

| 实验 | 覆盖 subaction（子动作片段） | tuple（训练样本） | 备注 |
| --- | ---: | ---: | --- |
| E10 | 79 | 157 | contact-part local projection + part-aware gate；每个 subaction 最多 3 个、帧间隔至少 2 帧 |

E10 的主变化不是回到 background homography，也不是简单放宽阈值，而是把 whole-object projection/gate 升级成 contact-local / part-aware projection/gate。

## Funnel

- raw candidate total（原始候选数）：`24687`
- deep run candidate total（实际深跑候选数）：`624`
- contact_gmm_pass（接触点 GMM 通过）：`474`
- active_object_selected（当前接触物体选中）：`545`
- object_track_success（物体跟踪成功）：`409`
- reference_found（参考帧找到）：`365`
- projection_success（投影成功）：`365`
- object_mask_gate_pass（object mask gate 通过）：`365`
- part_gate_pass（part gate 通过）：`365`
- trajectory_success（轨迹成功）：`365`
- heatmap_success（热图成功，即最终 tuple）：`157`
- final export top-k（每个 subaction 最多导出）：`3`
- final frame dedup gap（同一 subaction 内近邻去重帧距）：`2`
- reference gap penalty（参考帧距离惩罚）：`0.0060`

## projection_source（投影来源）

| projection_source（投影来源） | count |
| --- | ---: |
| contact_part_local_affine | 157 |

## 主要失败原因

| fail_reason | count |
| --- | ---: |
| missing_hand_or_object_bbox | 79 |
| contact_points_lt5_after_object_boundary_filter | 71 |
| track_confidence_below_threshold | 51 |
| part_gate_failed | 23 |
| track_visibility_below_threshold | 17 |
| gmm_failed:ValueError | 14 |
| trajectory_out_of_bounds | 4 |
| no_smoothed_contact_in_subaction | 1 |

## 失败来源归类

| 失败来源 | count |
| --- | ---: |
| detection | 150 |
| tracking | 68 |
| other | 23 |
| contact_gmm | 15 |
| trajectory | 4 |

## 是否接近 80 个 subaction 目标

E10 当前覆盖 `79/100` 个 subaction（子动作片段），距离 82 个 subaction 目标还差 `3` 个。`final_tuple_count` 现在反映 top-k 导出口径，不再等同于 subaction 覆盖数。

## 质量风险

- 如果 CoTracker 仍不可用，E10 的提升会主要来自 contact-part sampling 和 gate，而不是更强的时序跟踪。
- 当前 reference part mask 仍是基于 projected contact 的轻量规则构造，不是显式 part segmentation；对于严重遮挡或视角切换，仍会退化。
- articulated object 的部件区域在参考帧可能发生开合或遮挡，仍可能触发 `whole_object_affine_fallback` 或 `part_gate_failed`。

## Contact Sheet

- `/root/workspace/vrb/vrbreproduction/outputs/100subaction-e10/charts/e10_success_contact_sheet_page_001_page_001.png`
- `/root/workspace/vrb/vrbreproduction/outputs/100subaction-e10/charts/e10_success_contact_sheet_page_001_page_002.png`
- `/root/workspace/vrb/vrbreproduction/outputs/100subaction-e10/charts/e10_success_contact_sheet_page_001_page_003.png`
- `/root/workspace/vrb/vrbreproduction/outputs/100subaction-e10/charts/e10_success_contact_sheet_page_001_page_004.png`
- `/root/workspace/vrb/vrbreproduction/outputs/100subaction-e10/charts/e10_success_contact_sheet_page_001_page_005.png`
- `/root/workspace/vrb/vrbreproduction/outputs/100subaction-e10/charts/e10_success_contact_sheet_page_001_page_006.png`
- `/root/workspace/vrb/vrbreproduction/outputs/100subaction-e10/charts/e10_success_contact_sheet_page_001_page_007.png`
- `/root/workspace/vrb/vrbreproduction/outputs/100subaction-e10/charts/e10_success_contact_sheet_page_001_page_008.png`
- `/root/workspace/vrb/vrbreproduction/outputs/100subaction-e10/charts/e10_success_contact_sheet_page_001_page_009.png`
- `/root/workspace/vrb/vrbreproduction/outputs/100subaction-e10/charts/e10_success_contact_sheet_page_001_page_010.png`

## 图表

- `charts/e10_pipeline_funnel.png`
- `charts/e10_failure_reasons.png`
- `charts/e10_mask_projection_contact_sheet.png`
- `charts/e10_success_contact_sheet_page_001.png`
- `charts/e10_projection_source_counts.png`
