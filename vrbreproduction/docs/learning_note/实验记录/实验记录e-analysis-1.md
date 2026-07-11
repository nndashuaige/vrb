# e-analysis-1 2026-7-6

## 一、实验目的

在 EPIC-KITCHENS 上界定**可用数据**，并在可用数据上统计：

1. 被操作的对象都有哪些？各自出现的频次是多少？（一次 subaction 计一次）
2. 执行的动作都有哪些？各自的频次是多少？

### 可用数据的定义

一条 narration（= 一次 subaction）同时满足以下条件才算可用：

1. 公开 annotation 中包含"对 xx 物体做 xx 动作"（`verb` 与 `noun` 均非空）；
2. annotation 给出该动作的帧范围（`start_frame` / `stop_frame` 有效，且 stop > start ≥ 0）；
3. 该视频有公开的**手框 + 物体框**标注，且动作帧范围落在标注覆盖的帧数以内。

### 数据来源（均公开）

| 内容 | 来源 | 说明 |
| --- | --- | --- |
| 动作标注（verb/noun/帧范围） | [epic-kitchens-100-annotations](https://github.com/epic-kitchens/epic-kitchens-100-annotations) 的 `EPIC_100_train.csv` + `EPIC_100_validation.csv` | test 集无公开标签，不计入 |
| 手框 + 物体框 | [epic-kitchens-100-hand-object-bboxes](https://github.com/epic-kitchens/epic-kitchens-100-hand-object-bboxes)（数据托管在 data.bris.ac.uk，按 `{video_id}.pkl` 逐帧提供 hand bbox + object bbox） | 官方用 Shan et al. (100DOH) 检测器在全部 EPIC-100 视频上抽取并发布；覆盖清单为仓库中的 `EPIC_100_frame_counts.csv` |

注意：手/物体框是官方发布的**自动检测结果**（发布时用低阈值保留、由使用者按阈值过滤），不是人工逐帧标注；这也是 VRB 原文使用的同一份数据。人工标注版本是 VISOR（分割 mask），本实验以 bbox 发布为准。

## 二、代码位置

```text
vrbreproduction/src/vrbreproduction/e_analysis_1_usable_data_stats.py   # 主逻辑
vrbreproduction/scripts/run_e_analysis_1.py                             # 入口
```

运行方式（仓库根目录）：

```bash
venv/bin/python vrbreproduction/scripts/run_e_analysis_1.py
```

脚本优先使用本地 `data/annotations/epic-kitchens-100-annotations/` 下已有的标注 CSV；
缺失的文件以及手-物框覆盖清单会自动下载到 `outputs/e-analysis-1/downloads/`。

## 三、筛选漏斗结果

| 门控 | 剩余 subactions |
| --- | --- |
| train + validation 全部公开标注 | 76,885 |
| verb 与 noun 均非空 | 76,885 |
| 帧范围有效 | 76,885 |
| 视频有手框+物体框标注 | 76,885 |
| 帧范围在手-物框覆盖内 | 76,885 |

**结论：EPIC-100 train+validation 的全部 76,885 条 subaction 都可用。**
原因：EPIC-100 的 narration 标注本身保证每条都有 verb/noun 和帧范围；
官方 hand-object bboxes 发布覆盖全部 700 个视频的全部帧。
真正被排除的是 test 集（约 13k 条，无公开 verb/noun 标签）。

可用数据覆盖：633 个视频、34 位参与者（EPIC-100 共 700 个视频/45 位参与者，
其余为 test 集视频，无公开 verb/noun 标签故不计入）。

## 四、统计结果

统计给出两个粒度：

- **raw 粒度**：narration 归一化后的原始词（如 `pick-up`、`take` 分开计）；
- **class 粒度**：官方 verb/noun class 归并（如 `pick-up`/`grab`/`take` 同为 `take` 类）——建议以这个粒度为主。

总量：293 个 noun class（raw 2,214 个）、97 个 verb class（raw 902 个）、
verb-noun class 组合 5,594 种。

### 被操作对象 Top 10（class 粒度）

| noun class | 频次    | 占比    |
| ---------- | ----- | ----- |
| tap        | 3,997 | 5.20% |
| plate      | 2,887 | 3.76% |
| spoon      | 2,691 | 3.50% |
| cupboard   | 2,676 | 3.48% |
| knife      | 2,515 | 3.27% |
| pan        | 2,472 | 3.22% |
| lid        | 1,978 | 2.57% |
| bowl       | 1,976 | 2.57% |
| drawer     | 1,883 | 2.45% |
| glass      | 1,692 | 2.20% |

### 执行动作 Top 10（class 粒度）

| verb class | 频次 | 占比 |
| --- | --- | --- |
| take | 16,785 | 21.83% |
| put | 13,934 | 18.12% |
| wash | 8,068 | 10.49% |
| open | 5,680 | 7.39% |
| close | 3,997 | 5.20% |
| insert | 3,624 | 4.71% |
| turn-on | 2,596 | 3.38% |
| turn-off | 2,072 | 2.70% |
| cut | 2,034 | 2.65% |
| mix | 1,861 | 2.42% |

### 动作-对象组合 Top 5（class 粒度）

turn-on tap (1,985)、turn-off tap (1,742)、open cupboard (1,569)、
open drawer (1,084)、close cupboard (1,050)。

## 五、输出文件

```text
vrbreproduction/outputs/e-analysis-1/
├── experiment_results.json              # 实验总览（定义、漏斗、Top10）
├── usability_funnel.csv                 # 筛选漏斗
├── usable_subactions.csv                # 全部可用 subaction 明细（含 class key 列）
├── noun_frequency.csv                   # 被操作对象频次（raw 粒度）
├── noun_class_frequency.csv             # 被操作对象频次（class 粒度）
├── verb_frequency.csv                   # 动作频次（raw 粒度）
├── verb_class_frequency.csv             # 动作频次（class 粒度）
├── verb_noun_pair_frequency.csv         # 动作-对象组合频次（raw 粒度）
├── verb_noun_class_pair_frequency.csv   # 动作-对象组合频次（class 粒度）
├── charts/                              # Top-30 条形图 x5
└── downloads/                           # 中间下载产物（手-物框覆盖清单等）
```
