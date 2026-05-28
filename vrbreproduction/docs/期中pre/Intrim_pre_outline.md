# Interim Pre 讲稿大纲

## 1. 总体思路

这次汇报的核心是：**我们如何根据论文中描述的数据处理方案，复现 contact point / trajectory 的生成流程，并在复现过程中逐步发现问题、分析原因、改进算法。**

整体工作可以概括为一句话：

> 论文只给出了方法描述，没有开源代码，也没有处理好的中间数据，因此我们需要从原始视频和 annotation 出发，逐步复现整套数据处理与可视化流程。

围绕这个目标，汇报可以按“总-分”的逻辑展开：

1. 先说明复现任务的整体目标和主要难点；
2. 再按阶段介绍我们已经完成的工作；
3. 接着用 E16 实验说明原论文路线的问题；
4. 然后解释问题主要出在哪里；
5. 最后介绍我们尝试过的改进方案，以及后续计划。

---

## 2. 总体难点

论文描述的是一个完整的数据处理 pipeline，但复现时遇到的主要困难是：

- 论文没有数据处理部分的开源代码；
- 没有提供处理好的中间数据；
- 很多关键细节只在论文中粗略描述；
- contact point、reference frame、homography projection 等步骤都需要自己根据论文叙述实现；
- 单个步骤失败会导致整个 subaction 被丢弃，因此数据留存率对最终结果影响很大。

因此，我们的工作不是简单跑一个已有代码，而是要从头搭建并验证整套流程。

---

## 目标介绍

数据处理的最终目标是，在一张无手，只有物体的帧中，绘制出接触点、热力图，以及表示物体稍后运动轨迹方向的箭头。

groud truth是一个形状为(5,2,2)的张量。含义是：五个点，每个点二维坐标，以及同样维度大小的置信度。

附图：

---

## 3. 阶段性工作

### 第一阶段：学习模型代码，绘制模型流程图。搞清楚数据处理流程。目的是搞懂ground truth是啥，明确目标。

附模型流程图：

简单介绍两句模型。

### 3.1 第二阶段：跑通物体检测

第一阶段的目标是先跑通基础检测模块，为后续 contact point 生成提供物体框信息。

这一阶段主要完成：

- 读取 EPIC annotation；
- 定位 subaction 对应的视频片段；
- 对视频帧进行 object detection；
- 检查检测结果是否能覆盖正在操作的物体。

### 3.2 第三阶段：跑通 hand detection 和 contact point 算法

第二阶段的目标是把手部检测、接触点生成和轨迹可视化串起来。

这一阶段主要完成：

- 跑通 hand detection；
- 找到 hand-object contact frame；
- 根据手框和物体框生成 contact point；
- 对 contact point 和 trajectory 进行可视化；
- 初步检查 contact point 是否落在合理区域。

### 3.3 第三阶段：单帧投影验证

第三阶段的目标是先选一个比较理想的单帧样本，验证投影逻辑是否可行。

这一阶段主要完成：

- 选择理想 contact frame；
- 找到对应 reference frame；
- 计算从 contact frame 到 reference frame 的投影关系；
- 将 contact point 投影到 reference frame；
- 打通完整的数据处理和可视化流程。

### 3.4 第四阶段：小 batch 复现完整流程

第四阶段的目标是用小 batch 验证整个 pipeline 的稳定性。

这一阶段开始暴露出比较明显的问题：

- 论文对 homography projection 的描述比较粗略；
- 很多实现细节没有写清楚；
- 严格按照论文方法投影到 handless frame 时，成功率偏低；
- 数据被大量筛掉，最终可用样本数量不足。

---

## 4. 最符合原论文语义路线复现结果：E16 实验

在 E16 实验中，我们严格按照论文思路，尝试将 contact point 通过单应性矩阵投影到 handless reference frame。

实验结果：

- 测试范围：`100` 个 subactions；
- 最终保留：`26` 个 subactions；
- 成功率：`26%`。

论文中提到的数据量约为 `54k`，数据集共有约 `67k` 个 subactions。也就是说，论文对应的通过率大约是：

```text
54k / 67k ≈ 81%
```

而我们严格复现 E16 路线时，小 batch 上只有 `26%` 左右的通过率，说明原方法中有一些关键实现细节可能没有在论文中充分展开。

---

## 5. E16 通过率漏斗

实验编号：`E16`

| 阶段 | 剩余数量 | 这一关在筛什么 | 被筛掉的直观含义 |
| --- | ---: | --- | --- |
| 原始输入 | 100 | P01_109 前 100 条 EPIC annotation | 原始 subaction 输入 |
| 有 new-contact candidate | 81 | 这个 subaction 里有没有“刚开始接触”的那一帧 | 19 个 subaction 没找到新的接触起点 |
| Cell2 contact / GMM pass | 59 | contact frame 上能不能提取出足够稳定的接触点 | 22 个虽然有接触帧，但手/物体检测或接触点太差，GMM 拟合不了 |
| reference found | 38 | contact 之前能不能找到合格 reference frame | 21 个找不到 E16 要求的 active-hand-invisible pre-contact reference |
| homography available | 29 | reference frame 到 contact frame 之间能不能算出可靠单应性矩阵链 | 9 个图像匹配或单应性质量不够，没法可靠回投 |
| final keep | 26 | 投影后的几何和 heatmap 是否可用 | 3 个虽然能投影，但投影结果几何不合理或 heatmap gate 没过 |

从这个漏斗可以看出，主要损失发生在三个位置：

1. contact point 生成失败；
2. 找不到合适的 reference frame；
3. homography chain 不稳定。

---

## 6. 失败原因拆解

### 6.1 contact point 生成失败

这一类问题对应的是：没有明确的新接触起点，或者 contact point 质量不够好。

常见原因包括：

- 手框没检测好；
- 物体框没检测好；
- 手和物体框没有足够重叠或足够接近；
- 接触边界点太少；
- contact points 数量少于 GMM 需要的组件数；
- active hand 找到了，但对应 object detection 不稳定。

### 6.2 reference found 条件过严

E16 找 reference 的条件可以直观理解为：

> 在 first contact frame 之前，往前找一帧：这一帧里“正在接触的那只手”看不见。

具体流程是：

1. 先找到 subaction 里的 first contact frame；
2. 判断 first contact 是左手还是右手；
3. 只在 first contact 之前寻找 reference；
4. 从 contact frame 往前搜索，最多回看 360 帧；
5. 找最近的一帧，要求 active hand 不可见。

这里的 active hand 不可见指的是：

- 如果 contact 是右手，那么 reference frame 里右手不能被 HOA 检测到；
- 如果 contact 是左手，那么 reference frame 里左手不能被 HOA 检测到。

需要注意的是，E16 的 reference 并不是 strict humanless frame。它不要求：

- 画面里完全没人；
- 另一只手消失；
- 没有物体；
- 没有 contact；
- 是干净背景；
- 满足 E14 那种 clean hand-object gate。

所以 E16 的 reference 更准确地说是：

```text
接触前最近的一帧：active hand 还没出现 / 检测不到
```

它筛掉的 21 个样本，直观上就是：

> 从 first contact 往前看，主动手一直在画面里，所以找不到“主动手消失”的 pre-contact frame。

可能情况包括：

- 手从 subaction 一开始就在画面里；
- contact 很快发生，前面没有足够早的帧；
- 手虽然没真正接触，但一直伸在画面里；
- HOA 检测一直检测到了 active hand；
- 动作本身是持续拿着、移动或放下，主动手没有消失阶段。

### 6.3 homography available 不稳定

`homography available` 可以直观理解为：

> reference frame 和 contact frame 之间，每相邻两帧都要能“对齐画面”，而且对齐质量不能太差。

它不是直接从 reference frame 到 contact frame 算一个大变换，而是逐帧计算：

```text
contact frame
-> contact - 1
-> contact - 2
-> ...
-> reference frame
```

每一步都要计算 pairwise homography。每一对相邻帧的流程大致是：

1. 在两帧图像里找视觉特征点，比如桌角、柜门边缘、纹理、物体边界等；
2. 匹配两帧里的特征点；
3. 用匹配点估计单应性矩阵；
4. 检查这个变换是否可靠。

可靠性主要看：

- 匹配点够不够多；
- 好匹配够不够多；
- RANSAC 内点够不够多；
- 内点比例够不够高；
- 变换后的结果是否合理；
- 有没有明显畸变、翻转或面积异常。

只要中间某一对相邻帧失败，整条 homography chain 就失败。

因此，E16 中 `homography available = 29` 意味着：

> 38 个已有 reference 的样本里，只有 29 个能从 contact frame 一帧帧稳定地对齐回 reference frame。

被筛掉的 9 个样本，直观上就是：

> 有 reference，也有 contact point，但中间图像对齐不可靠，投回去的位置不可信。

常见原因包括：

- reference 离 contact 太远，画面变化大；
- 相机移动太快；
- 手或物体遮挡了关键背景纹理；
- 画面模糊；
- 厨房台面或墙面纹理太少，找不到足够特征点；
- 两帧里匹配点很多是错的；
- 中间某一帧突然变化，例如手大面积遮挡、曝光变化、运动模糊；
- 算出来的 homography 会把点投到离谱位置，所以被质量门槛拒绝。

可以总结为：

```text
reference found 问的是：有没有一张合格的接触前参考图。
homography available 问的是：能不能把 contact frame 的点可靠地一路对齐投回那张参考图。
```

---

## 7. 准确率定义与当前主要矛盾

我们对准确率的定义是：

> 投影后的 heatmap / contact point 是否落在正在操作物体的对应操作位置上。

例如：

- 如果动作是拿锅，理想结果应该投到锅柄位置；
- bad case 是投影偏移，例如投到锅面，甚至没有投到锅上。

当前的主要矛盾是：

1. 如果严格使用 handless reference frame，数据留存率太低；
2. 如果放宽 reference 条件，数据量会增加，但 reference frame 里可能有手，需要处理遮挡；
3. homography projection 的准确率仍然不稳定；
4. contact 的物体有时和 annotation 语义不匹配。

因此，后续改进的核心目标是：

> 在不明显降低质量的前提下，提高数据留存率，并提升投影到目标物体正确位置的准确率。

---

## 8. 已尝试的改进方案

### 8.1 放宽 reference frame 条件

后续考虑放宽 handless 的条件限制：

- 不一定要求投到严格的 handless frame；
- 可以投到 contact frame 前几帧；
- 再把手 mask 掉。

这样做的预期收益是：

- reference 离 contact 更近；
- homography chain 更短；
- 画面变化更小；
- 成功率可能提高。

其中，E17 使用 contact 前 5-10 帧作为 reference，主要就是为了降低 homography chain 的难度：

> 两帧距离近，画面变化小，homography chain 短，所以更容易稳定。

### 8.2 光流方案：E7

我们尝试过用光流解决投影问题。

实验结论：

- 精准度略有提高；
- 但整体质量仍然不达标；
- 投影仍然容易出现偏移。

因此，单纯使用光流还不能稳定解决 contact point 投影问题。而且计算时间太长。pass。

### 8.3 SAM2 + Tracking + Weighted Affine 方案

之后我们尝试了基于 SAM2 的方案，整体流程如下：

1. **SAM**  
   找出目标物体 mask，判断哪些像素属于物体。

2. **CoTracker / LK**  
   跟踪物体 mask 上的点，判断这些物体点从 contact frame 移动到 reference frame 后的位置。

3. **Weighted affine**  
   根据这些被跟踪的点，估计一个局部变换，再把 5 个 contact GMM center 投影到 reference frame。

4. **Mask gate**  
   使用 SAM / reference mask 检查投影后的点是否真的落在物体上。

目前效果：

- 精准度和质量都有明显提升；
- E12 的结果比较好；
- 投影问题已经有了较大改善。

但是这个方案仍然存在一些问题：

- 算力成本大幅增加；
- 语义匹配仍然可能出错；
- reference frame 中大量样本仍然有手；
- 部分样本 contact point 精度仍然不够；
- 数据量还需要继续增加。

---

## 9. 待优化点

### 9.1 语义匹配问题

现在识别出来的 contact 物体有时和 annotation 中标注的物体对应不上。

例如：

- annotation 标注的是锅；
- 当前方法可能识别出来的是刀；
- heatmap 可能出现在正在运动的手上；
- 或者出现在运动物体附近，但不是 annotation 标注的目标物体上。

后续可能尝试：

1. 使用有语义的 open-vocabulary detection；
2. 先检测 annotation 中想要的目标物体；
3. 再做手和目标物体相关的检测；
4. 最后继续 contact point / trajectory pipeline。

### 9.2 reference frame 中有手

现在大量 reference frame 里仍然有手。

后续需要解决：

- 如何检测 reference frame 中的手；
- 如何把手 mask 掉；
- 如何避免手区域影响 heatmap 或物体 mask；
- 如何保证去手之后的结果仍然自然、可用。

### 9.3 部分样本精度仍然不够

有些样本仍然存在 contact point 位置不准确的问题。

例如：

> Subaction45 中的动作是接触锅，理想 contact point 应该在锅柄上；但当前结果投到了锅中央。虽然 contact point 的形状像一条锅柄，但没有投到物体的正确位置。

这一类问题说明，当前方法虽然比原来的 homography route 更稳定，但仍然需要进一步约束 contact point 和目标物体语义位置的一致性。

### 9.4 数据留存率仍有提高空间

按照论文中写的数据量是54k，总共有约67k数据，我们现在约能达到论文所述的的1/3-1/2，处于同一数量级

---

## 10. 后续计划

数据处理部分后续工作可以分为三条线：

### 10.1 提高数据留存率

- 放宽 handless reference frame 限制；
- 优先使用 contact 前较近的 reference frame；
- 缩短投影链路；
- 减少 homography / tracking 失败导致的样本丢弃。

### 10.2 提高投影准确率

- 以 SAM + tracking + weighted affine 方案为对照，寻找准确率和计算资源平衡的处理方法；

### 10.3 解决语义目标不匹配

- 引入 open-vocabulary detection；
- 根据 annotation 先锁定目标物体；
- 再围绕目标物体做 contact point 检测；
- 减少把 heatmap 投到错误物体或手上的情况。

### 10.4 寻找把手mask掉的办法。

整体后续工作：

- 全量的数据处理
- 调模型结构
- 投入训练

---

## 11. 汇报总结

这部分可以作为讲稿结尾：

> 总体来说，我们已经从物体检测、手部检测、contact point 生成、reference frame 寻找、homography projection 到可视化，复现了论文中描述的数据处理流程。  
> 但在严格复现论文路线时，小 batch 上只有 26% 的 subaction 能最终保留，主要瓶颈在 reference frame 寻找和 homography projection 的稳定性。  
> 因此，我们后续尝试了光流、SAM、tracking 和 weighted affine 等方案。目前 SAM + tracking 的路线在投影质量上有明显提升，但由于需要解决算力成本过高，目前仅作为参考方案。  
> 下一步的重点是在保证质量的前提下提高数据留存率，让 contact point 更稳定地落在 annotation 对应的目标物体和正确操作区域上，引入动作和名词的语义对应，以及寻找把手mask掉的办法。
