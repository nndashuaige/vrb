# E7B 实验设计思路

## 1. E7B 相较于 E7 主要改了什么

E7 的核心思路是：

- 用 detection bbox association 维持 object identity
- 用 sparse optical flow 在检测缺失时传播物体框
- 用 contact-local optical flow 或 bbox-relative projection 把 contact point 投回 reference frame
- 最后再用 tracked reference object 做门控

E7B 不是推翻这个框架，而是把重点从“能不能通过筛选”转到“投影是否更准、样本是否更可用”。

相较于 E7，E7B 主要做了三类修改：

1. **局部光流特征选择更对象中心化**
   - 不再只在 contact 点附近固定半径取点。
   - 改成优先在“contact 区域”和“tracked object bbox”交集内选点。
   - 显式排除 active hand bbox 附近的点，减少手部和背景对局部仿射估计的污染。

2. **投影不再只返回单点，增加不确定性建模**
   - 由固定圆形 Gaussian heatmap，改为尽量使用投影协方差来生成 heatmap。
   - 这样标签形状能更贴近局部流估计的不确定性，而不是把所有样本都压成同样的圆形监督。

3. **final export 口径更灵活**
   - 不再把 “final_tuple_count” 当成核心目标。
   - 允许每个 subaction 导出多个高分 tuple，并做帧去重。
   - 但真正要看的仍然是 subaction 覆盖率，而不是 tuple 数量。

## 2. E7B 想解决什么问题

E7 的问题不是单纯“通过率不够”，而是：

- 通过的样本里，contact point 投影到 reference frame 后仍不够贴近真实接触区域
- heatmap 形状过于固定，不能反映局部投影误差
- 局部光流经常把背景、手边缘、物体边缘混在一起，导致 affine 漂移

所以 E7B 的目标是同时解决两件事：

1. **提高 subaction 覆盖率**
   - 让更多 subaction 至少能产出一个可用训练样本。

2. **提高几何精度**
   - 让 projected contact point 更稳定地落在 reference object 的合理接触区域。
   - 不只是“过筛”，而是让标签质量真的可训练。

换句话说，E7B 不是继续单纯加严过滤，而是尽量修正“为什么投影不准”。

## 3. E7B 是怎么改算法的

### 3.1 局部光流改成 object-aware 采样

E7 里局部光流的风险是：特征点来源太宽，容易混入无关区域。

E7B 的处理方式是：

- 先得到 tracked object bbox
- 再把 contact neighborhood 和 object bbox 约束起来
- 只在有效交集内选局部特征点
- 把 active hand 附近点排除掉

这样做的目的，是让仿射估计更像“物体表面点的局部运动”，而不是“手、背景、边缘一起平均后的运动”。

### 3.2 用加权仿射估计替代更粗的局部投影

E7B 不只是找一组点做 affine，而是对点引入更明确的权重和内点筛选：

- 先做 forward-backward / residual 约束
- 再基于内点数量、内点比例、残差等指标判断这次局部投影是否可信
- 如果局部流不够稳，再回退到 bbox-relative projection

也就是说，E7B 把投影路径分成了：

1. object-aware local flow affine
2. bbox-relative fallback

并且让前者只在“局部几何确实可信”时才生效。

### 3.3 heatmap 从固定形状改成协方差驱动

E7 的 heatmap 更像固定模板。

E7B 里增加了投影协方差的传播：

- 如果局部仿射估计可信，就把 contact 的不确定性一起投过去
- 生成的 heatmap 不再总是同样大小的圆
- 这样标签更接近真实接触区域，也更符合投影误差的空间分布

### 3.4 最终保留逻辑从“单个最优样本”放宽到“每个 subaction 的 top-k”

E7 里常见问题是：一个 subaction 可能有多个候选，但最后只保留一个，容易把可用样本数量压得太低。

E7B 改成：

- 每个 subaction 允许保留多个高分 tuple
- 再做近邻帧去重
- 这样既能保留更多训练样本，也不会让同一段连续帧重复刷屏

但这里的重点不是制造更多 tuple，而是给后续人工检查和质量筛选留更多“可用候选”。

## 4. E7B 的判断标准

E7B 成败不看单一指标，而看两条线是否同时变好：

- **数量线**：subaction pass rate 是否上升
- **质量线**：projected contact 是否更贴近 reference object 的真实接触区域

如果只提高数量，但投影还漂，那不能算成功。
如果只追求几何精度但通过率仍太低，也不能满足训练数据需求。

所以 E7B 的定位是：

**在不放弃 object-centric projection 的前提下，减少局部光流漂移，并让 heatmap / 监督标签更贴近真实接触几何。**

