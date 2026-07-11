# E18-a new-contact episode 划分实验

## 目的

E18-a 用 new-contact episode 替代 annotation subaction 作为训练样本划分单位。除划分单位外，Cell2 contact point、reference 搜索、homography 投影、geometry gate、heatmap gate 继续沿用 E18。

## 划分口径

- source annotations：`P01_109` 按 E18 相同规则取前 100 个 subaction。
- source window：这 100 个 subaction 的连续帧范围 `79-16103`。
- contact 信号：HOA/100DOH 中 `hand.state in {PORTABLE_OBJECT, STATIONARY_OBJECT}` 且 `score > 0.5`，再用 Savitzky-Golay `window=7, polyorder=2` 平滑，阈值 `>0.75`。
- episode：左右手分开统计，每个平滑后 contact run 的真实 `0->1` 起点落在 source window 内，即生成一个 new-contact episode。
- candidate：每个 episode 只取 first contact timestep，即 `candidate_frame_offset_from_episode_start = 0`。
- metadata：每个 episode 记录 `source_subaction_index/source_narration_id/source_assignment`，用于回查它借用的 annotation 语义和 noun。

## 产物位置

- code：`vrbreproduction/src/vrbreproduction/e18a_new_contact_episode.py`
- output：`vrbreproduction/outputs/e18-a`
- smoke output：`vrbreproduction/outputs/e18-a-smoke1`

## 本次运行结果

- source subactions：`100`
- new-contact episodes：`322`
- left episodes：`160`
- right episodes：`162`
- final success / heatmap pass：`162`
- Cell2 pass：`216`
- reference found：`216`
- homography available：`175`
- geometry pass：`162`
- main failure：`missing_hand_or_object_bbox`

## 备注

当前 `summary.md` 仍复用 E18 的报告模板，所以部分标题和说明文字仍写着 E18/subaction；实际 CSV 字段里已增加 episode/source-subaction 字段，并额外输出 `episode_summary.csv`。
