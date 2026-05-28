# VRB Reproduction Workspace

本目录整理了 VRB 复现实验代码、实验记录和输出结果。E12、E16、E17 是本轮重点实验版本，代码入口在 `src/vrbreproduction/`，实验记录和结果在 `outputs/`。

## 目录结构

- `src/vrbreproduction/`: 实验入口和共享函数文件。
- `scripts/`: 输出结果的后处理、排查图和 montage 生成脚本。
- `tests/`: 单元测试。
- `notebooks/`: 数据处理和探索 notebook。
- `docs/`: 部署记录、会议记录和学习笔记。
- `data/`: 本地 EPIC-KITCHENS 标注、HOA detections 和帧图像。该目录较大，默认不提交。
- `outputs/e12/`: E12 实验记录、CSV/JSON 结果、图表和样本级 pipeline outputs。
- `outputs/e16/`: E16 实验记录、CSV/JSON 结果、图表和样本级 pipeline outputs。
- `outputs/e17/`: E17 实验记录、CSV/JSON 结果、图表和样本级 pipeline outputs。

## E12 / E16 / E17 阅读顺序

建议先读共享数据流，再读每个版本的实验入口。

1. `src/vrbreproduction/contact_point_utils.py`
   - 手/物体 bbox 选择、contact points 提取、object mask/boundary mask 构建。
2. `src/vrbreproduction/pipeline_retention.py`
   - Cell2 contact GMM、Cell4 heatmap gate、JSON 安全序列化等保留逻辑。
3. `src/vrbreproduction/problem3_utils.py`
   - reference/contact frame 几何变换、homography、crop、dynamic mask、可视化 overlay。
4. `src/vrbreproduction/label_heatmap_utils.py`
   - label heatmap 构建、合并、保存和 VRB 风格 affordance overlay。
5. `src/vrbreproduction/e5b_reference_object_consistency.py`
   - bbox/polygon 几何辅助函数、articulated object token 判断相关工具。
6. 按实验版本读入口文件：
   - E12: `src/vrbreproduction/e12_target_aware_clean_reference_projection.py`
   - E16: `src/vrbreproduction/e16_active_hand_invisible_reference.py`
   - E17: `src/vrbreproduction/e17_near_contact_reference.py`
7. 最后读结果记录：
   - E12: `outputs/e12/summary.md`
   - E16: `outputs/e16/summary.md`
   - E17: `outputs/e17/summary.md`

## E12: Target-Aware Clean Reference Projection

入口文件：`src/vrbreproduction/e12_target_aware_clean_reference_projection.py`

推荐阅读顺序：

1. `EXPERIMENT` 和 `E12Config`: 实验定义、阈值、路径和 CoTracker/SAM2 开关。
2. `build_contact_arrays()`, `segment_contact_runs()`, `build_new_contact_runs()`: 从 HOA detection 得到平滑 contact episode。
3. contact/object helper functions: 构造 whole-object mask、contact-part mask、tracking points。
4. projection/tracking helper functions: CoTracker / local affine / fallback tracking 的投影逻辑。
5. `run_one_candidate()`: 单个候选样本的完整 Cell2 -> reference -> projection -> heatmap 流程。
6. `write_summary()`: 输出 `summary.md`、图表和结果摘要。
7. `run_e12_experiment()`: 总入口，负责读数据、跑候选、写 CSV/JSON/图表。

E12 直接调用的本地函数文件：

- `src/vrbreproduction/contact_point_utils.py`
- `src/vrbreproduction/e5b_reference_object_consistency.py`
- `src/vrbreproduction/label_heatmap_utils.py`
- `src/vrbreproduction/pipeline_retention.py`
- `src/vrbreproduction/problem3_utils.py`

E12 输出：

- 实验记录：`outputs/e12/summary.md`
- 汇总结果：`outputs/e12/experiment_results.json`, `outputs/e12/experiment_overview.csv`
- 明细表：`outputs/e12/candidate_diagnostics.csv`, `outputs/e12/candidate_plan.csv`, `outputs/e12/subaction_summary.csv`, `outputs/e12/successful_sample_manifest.csv`
- 图表：`outputs/e12/charts/`
- 样本级 pipeline outputs：`outputs/e12/experiments/E12/pipeline_outputs/`

当前结果摘要：E12 覆盖 `79/100` 个 subaction，导出 `156` 个 final tuples；核心改动是 contact-part local projection、part-aware gate 和 target-aware clean reference ranking。

## E16: Active-Hand-Invisible Reference Baseline

入口文件：`src/vrbreproduction/e16_active_hand_invisible_reference.py`

推荐阅读顺序：

1. `EXPERIMENT` 和 `E16Config`: one-candidate-per-subaction 设置、active-hand-invisible reference 策略和输出路径。
2. `build_contact_arrays()`, `build_new_contact_runs()`, `runs_for_subaction()`: 找每个 subaction 的 first new contact episode。
3. `Problem3CachedRunnerE16`: reference 搜索、homography 缓存和几何投影。
4. `run_one_candidate()`: 单个 subaction 的 candidate 处理、reference gate、geometry gate、heatmap gate。
5. `write_comparison_files()`: 生成与 E6b/E6c 的 success 对比表。
6. `write_markdown_report()`: 写 `outputs/e16/summary.md`。
7. `run_e16_experiment()`: 总入口。

E16 直接调用的本地函数文件：

- `src/vrbreproduction/contact_point_utils.py`
- `src/vrbreproduction/e5b_reference_object_consistency.py`
- `src/vrbreproduction/label_heatmap_utils.py`
- `src/vrbreproduction/pipeline_retention.py`
- `src/vrbreproduction/problem3_utils.py`

E16 输出：

- 实验记录：`outputs/e16/summary.md`
- 汇总结果：`outputs/e16/experiment_results.json`, `outputs/e16/experiment_overview.csv`
- 明细表：`outputs/e16/candidate_diagnostics.csv`, `outputs/e16/candidate_plan.csv`, `outputs/e16/subaction_summary.csv`, `outputs/e16/successful_sample_manifest.csv`
- 对比表：`outputs/e16/e16_success_vs_e6b.csv`, `outputs/e16/e16_success_vs_e6c.csv`
- 图表：`outputs/e16/charts/`
- 样本级 pipeline outputs：`outputs/e16/experiments/E16/pipeline_outputs/`

当前结果摘要：E16 成功 `26/100` 个 subaction；主要失败来自 `no_active_hand_invisible_reference`，说明严格 active-hand-invisible reference 会显著限制覆盖率。

## E17: Near-Contact Reference Diagnostic Baseline

入口文件：`src/vrbreproduction/e17_near_contact_reference.py`

推荐阅读顺序：

1. `EXPERIMENT` 和 `E17Config`: near-contact pre-frame offset `5-10` 的 reference 策略。
2. `build_contact_arrays()`, `build_new_contact_runs()`, `runs_for_subaction()`: 与 E16 相同，先确定 first new contact episode。
3. `Problem3CachedRunnerE17`: 近接触 reference、shadow diagnostics 和 homography 缓存。
4. `run_one_candidate()`: 单个 candidate 的主流程。
5. `write_comparison_files()`: 生成与 E6b/E6c 的 success 对比表。
6. `write_markdown_report()`: 写 `outputs/e17/summary.md`。
7. `run_e17_experiment()`: 总入口。

E17 直接调用的本地函数文件：

- `src/vrbreproduction/contact_point_utils.py`
- `src/vrbreproduction/e14_clean_pre_contact_reference.py`
- `src/vrbreproduction/e5b_reference_object_consistency.py`
- `src/vrbreproduction/label_heatmap_utils.py`
- `src/vrbreproduction/pipeline_retention.py`
- `src/vrbreproduction/problem3_utils.py`

E17 输出：

- 实验记录：`outputs/e17/summary.md`
- 汇总结果：`outputs/e17/experiment_results.json`, `outputs/e17/experiment_overview.csv`
- 明细表：`outputs/e17/candidate_diagnostics.csv`, `outputs/e17/candidate_plan.csv`, `outputs/e17/subaction_summary.csv`, `outputs/e17/successful_sample_manifest.csv`
- 对比表：`outputs/e17/e17_success_vs_e6b.csv`, `outputs/e17/e17_success_vs_e6c.csv`
- 图表：`outputs/e17/charts/`
- 样本级 pipeline outputs：`outputs/e17/experiments/E17/pipeline_outputs/`

当前结果摘要：E17 成功 `47/100` 个 subaction；near-contact reference 放宽了 humanless/active-hand-invisible 限制，但仍受 Cell2 contact/GMM 和几何质量限制。

## 排查和后处理脚本

- `src/vrbreproduction/e12_debug_summary_charts.py`: 从 E12 已有 CSV/JSON 和帧图像生成 `outputs/e12/charts/排查汇总/`。
- `scripts/make_e16_debug_montages.py`: 从 E16 输出生成 object、hand、contact、reference 排查 montage。
- `scripts/make_e17_debug_montages.py`: 从 E17 输出生成 object、hand、contact、reference 排查 montage。

运行示例：

```bash
PYTHONPATH=src python3 -m vrbreproduction.e12_target_aware_clean_reference_projection
PYTHONPATH=src python3 -m vrbreproduction.e16_active_hand_invisible_reference
PYTHONPATH=src python3 -m vrbreproduction.e17_near_contact_reference
PYTHONPATH=src python3 -m vrbreproduction.e12_debug_summary_charts
PYTHONPATH=src python3 scripts/make_e16_debug_montages.py
PYTHONPATH=src python3 scripts/make_e17_debug_montages.py
```

## 验证

语法检查：

```bash
python3 -m compileall src/vrbreproduction
```

单元测试：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```
