# E8 SAM2 + CoTracker mask-point tracking 实验报告

- 输出目录：`vrbreproduction/outputs/100subaction-e8-smoke`
- video_id：`P01_109`
- subaction（子动作片段）总数：`3`
- 成功覆盖 subaction（子动作片段）：`2`
- final tuple（最终训练样本）：`6`
- mask gate 通过数：`17`
- CoTracker 成功数：`0`
- fallback tracker 成功数：`17`
- fallback tracker 比例：`100.00%`
- articulated object 成功数：`0`
- articulated object 失败数：`0`

## 算法改动

- contact frame 先通过 VISOR / SAM2 / grabcut fallback 得到目标物 mask，再从 mask 内优先采样 contact-near / boundary 点，显式排除手部附近点。
- 用 CoTracker 接口做点跟踪；当前环境未安装 CoTracker 时，tracker_backend 会明确记为 `fallback_lk`，不冒充 CoTracker。
- 用 tracked points 在参考帧拟合 weighted affine，再把 contact GMM means 投到 reference frame。
- 最终 gate 不再看 bbox 一致性，而是看 projected contact 是否真正落在 reference object mask 上，或者是否足够贴近 mask boundary。
- 最终导出允许每个 subaction（子动作片段）最多保留 `3` 个高分 tuple（训练样本），并做 `2` 帧的近邻去重。

## 与 E6b / E6c 的关键区别

| 实验 | 覆盖 subaction（子动作片段） | tuple（训练样本） | 备注 |
| --- | ---: | ---: | --- |
| E7 | 43 | 43 | object tracklet + local flow + bbox gate baseline |
| E6b | 18 | 74 | crop-level humanless fallback（局部无手补救）但无最终 object-centric gate（物体中心门控） |
| E6c conservative | 11 | 30 | reference-frame object consistency（参考帧物体一致性）保守口径 |
| E8 | 2 | 6 | SAM2 mask + tracked points + mask gate；每个 subaction 最多 3 个、帧间隔至少 2 帧 |

E8 的主变化不是再调几何阈值，而是把接触点投影改成物体 mask 内点的点跟踪 + mask gate。E7 的主要失败点里，reference_crop_hand_overlap_high、missing_hand_or_object_bbox 和 projected_contact_not_on_tracked_object 会被这个版本直接对冲。

## Funnel

- raw candidate total（原始候选数）：`278`
- deep run candidate total（实际深跑候选数）：`22`
- contact_gmm_pass（接触点 GMM 通过）：`18`
- active_object_selected（当前接触物体选中）：`22`
- object_track_success（物体跟踪成功）：`18`
- reference_found（参考帧找到）：`17`
- projection_success（投影成功）：`17`
- mask_gate_pass（mask gate 通过）：`17`
- trajectory_success（轨迹成功）：`17`
- heatmap_success（热图成功，即最终 tuple）：`6`
- final export top-k（每个 subaction 最多导出）：`3`
- final frame dedup gap（同一 subaction 内近邻去重帧距）：`2`
- reference gap penalty（参考帧距离惩罚）：`0.0060`

## projection_source（投影来源）

| projection_source（投影来源） | count |
| --- | ---: |
| cotracker_weighted_affine | 6 |

## 主要失败原因

| fail_reason | count |
| --- | ---: |
| contact_points_lt5_after_object_boundary_filter | 4 |
| track_visibility_below_threshold | 1 |

## 失败来源归类

| 失败来源 | count |
| --- | ---: |
| detection | 4 |
| tracking | 1 |

## 是否接近 80 个 subaction 目标

E8 当前覆盖 `2/100` 个 subaction（子动作片段），距离 80 个 subaction 目标还差 `78` 个。`final_tuple_count` 现在反映 top-k 导出口径，不再等同于 subaction 覆盖数。

## 质量风险

- 当前环境里 CoTracker 未安装，正在用 `fallback_lk` 维持可跑通性；这会影响最终覆盖率和几何稳定性。
- SAM2 也可能因 checkpoint 下载失败而退回 grabcut fallback；这时 mask 质量会明显低于目标方案。
- articulated object 的参考 mask 更容易变形或遮挡，仍可能出现误投。

## Contact Sheet

- `vrbreproduction/outputs/100subaction-e8-smoke/charts/e8_success_contact_sheet_page_001.png`

## 图表

- `charts/e8_pipeline_funnel.png`
- `charts/e8_failure_reasons.png`
- `charts/e8_mask_projection_contact_sheet.png`
- `charts/e8_success_contact_sheet_page_001.png`
- `charts/e8_projection_source_counts.png`
