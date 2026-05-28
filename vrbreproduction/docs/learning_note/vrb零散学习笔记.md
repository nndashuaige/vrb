# VRB 零散学习笔记

## GMM 与滤波器在 VRB 中的作用

### 一、GMM（高斯混合模型）是干嘛的？

通俗版：GMM 就是一种"用一堆高斯分布（钟形曲线）去拟合一堆点"的方法。

想象你在一张图上撒了一把沙子，沙子分布得比较集中但有几个明显的"堆"。GMM 能自动告诉你：
- 有几堆？（`n_components`）
- 每堆的中心在哪？（`mean`）
- 每堆有多大、多散？（`covariance`）
- 哪堆沙子更多？（`weight`，权重）

#### 在 VRB 里的两个用途

**用途 1：把稀疏预测变成稠密候选点（inference 阶段）**

看 `inference.py` 第 86 行：

```python
gm = GaussianMixture(n_components=3, covariance_type='diag')
gm.fit(np.vstack(centers))
cp, indx = gm.sample(50)
```

这里干了什么？
- 网络预测了 contact point，但这些点很"稀疏"（可能只有几个 mode）
- GMM 把这稀疏的几个点拟合成 3 个高斯分布
- 然后从中 **采样 50 个点**（`gm.sample(50)`）

为什么要这么做？因为机器人最后执行的时候，你不能只说"接触点大概在这儿"，你需要给出一堆候选点让它去试。GMM 采样能生成"围绕在预测中心附近、服从一定分布"的一堆候选点。

**用途 2：给 heatmap 的不同热区分配权重（label 生成阶段）**

看 `src/vrbreproduction/label_heatmap_utils.py` 第 107–116 行：

```python
weights = weights / weights.sum()  # 归一化权重
...
amplitude = weights[k]
heatmaps[k] = make_gaussian_heatmap(..., amplitude=amplitude)
```

这里 GMM 的 `weight` 表示：这个 contact mode 有多"靠谱"。
- 权重高 → 生成的高斯热区更亮（amplitude 大）
- 权重低 → 热区更暗

也就是说，GMM 不仅告诉你"接触点在哪"，还告诉你"哪个接触点更可信"，可视化的时候重点突出更可信的区域。

---

### 二、滤波器（filter）是干嘛的？

通俗版：滤波器就是一个"抹匀"工具。你把一把盐撒在桌上是几颗离散的点，拿块布一擦，就变成了一小片模糊的盐渍。

#### 在 VRB 里的用途：把离散点变成连续 heatmap

看 `inference.py` 的 `compute_heatmap` 函数（第 30–52 行）：

```python
# 1. 先打 delta 点
heatmap[col, row] += 1.0

# 2. 用高斯滤波器"抹开"
heatmap = cv2.GaussianBlur(heatmap, (k_size, k_size), 0)
```

具体过程：
1. **先在图上打点**：网络预测了 contact point 的坐标，你就在图上对应像素位置标个 1，其他地方是 0。这时 heatmap 就是一张"只有几个亮点、其他地方全黑"的图。
2. **高斯模糊（滤波）**：用 `cv2.GaussianBlur` 把这个点"抹开"，从一个亮点变成一团逐渐变暗的云雾。中心最亮，往外越来越暗，形状就是一个钟形曲线。

#### 为什么要滤波？直接打点不行吗？

**不行。** 原因有两个：

**原因 1：训练更稳定**

如果 ground truth 只是一个精确的坐标点，网络要么预测得完全对（loss = 0），要么稍微偏一点（loss 很大），梯度很"硬"，训练容易崩。

换成 heatmap 之后，网络预测的是一个"概率分布"——中心位置 loss 最小，稍微偏一点 loss 也小一点，再远才变大。这样网络学起来更舒服，容错率更高。

**原因 2：对抗标注误差**

人手标的 contact point 不可能 100% 精确，总会偏个几像素。如果是精确坐标，这几像素的偏差会给网络传递错误信号；如果是 heatmap，这几像素还在高斯热区里，影响很小。

---

### 三、GMM + 滤波器在 VRB 里的配合关系

```
┌─────────────────────────────────────────────────────────────┐
│                    你的数据生成流程                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: 从视频/标注中提取 5 个 contact points               │
│          ↓                                                  │
│  Step 2: 拟合 GMM → 得到 5 个高斯分布                       │
│          （每个分布有：中心 μ、方差 σ²、权重 w）              │
│          ↓                                                  │
│  Step 3: 对每个高斯分布，用"高斯核"生成一张 heatmap           │
│          （这里的高斯核本质上就是一个小型滤波器）              │
│          ↓                                                  │
│  Step 4: 把 5 张 heatmap 合并（max 或 sum）                 │
│          ↓                                                  │
│  Step 5: 得到最终的 ground truth heatmap                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

- **GMM** 负责建模"有 5 个可能的接触位置，每个位置的可信度不一样"
- **滤波器（高斯模糊）** 负责把每个精确位置变成一个平滑的"热区"，让网络能学

---

### 四、总结成一句话

> **GMM 用来建模"哪几个地方可能接触、各自有多大概率"；滤波器用来把这些概率点"抹开"成平滑的热图，让神经网络能学得稳定、学得舒服。**

---

## Trajectory 相关问题

### 一、逐字稿里老师是怎么问的？

原话三段：

> trajectory 的话，像我刚问你的问题，**它五个点，它的起始点有没有？**
> 还是说它只是平为它的五个点？
> 然后如果 5 个点就是那个，就是**只输出 5 个点的话，你到时候你自己生成的数据也只要生成这 5 个点就好了**，对吧？

核心：确认 trajectory 到底有 6 个点（含起始点）还是 5 个点（纯未来路径），这样生成 ground truth 才知道 tensor 维度该写几。

### 二、代码告诉我们答案：5 个点，不含起始点

看 `demo.py` 和 `traj.py`：

```python
# demo.py
hand_head = TrajAffCVAE(in_dim=2*args.traj_len, ...)  # traj_len=5
# in_dim = 2 * 5 = 10，即 5 个 (x, y) 坐标
```

```python
# model.py inference
future_hand = self.hand_head.inference(memory[:, 0, :])
future_hand = future_hand.reshape(B, -1, 2)  # reshape 成 (B, 5, 2)
```

**结论：trajectory 就是 5 个未来的 waypoint，不含起始点。** 起始点（contact point）是另外独立预测的。

### 三、trajectory 的方向是怎么确定的？

这是最关键的问题。看 `inference.py` 的可视化代码：

```python
ic, pc = net.inference(...)
trajs.append(ic[0, 2:])   # 取 predicted trajectory 的第 2 个点往后
...
x2, y2 = np.vstack(trajs)[np.random.choice(len(trajs))]
dx, dy = np.array([x2, y2])*np.array([h, w]) + np.random.randn(2)*traj_scale
...
plt.arrow(int(np.mean(x)), int(np.mean(y)), scale*dx, -scale*dy, ...)
```

注意这里 `dx, dy` 被直接当成 **arrow 的方向向量** 用了。`plt.arrow(x, y, dx, dy)` 的第三、四个参数是相对位移，不是绝对坐标。

这说明网络输出的 trajectory **不是绝对坐标，而是相对于 contact point 的位移（delta/偏移量）**。

#### 通俗解释

想象 contact point 是"你现在站的位置"，trajectory 不是告诉你"5 秒后你在天安门"，而是告诉你：
- 第 1 步：往前 10cm、往右 5cm
- 第 2 步：再往前 8cm、往右 3cm
- 第 3 步：……

**5 个 (dx, dy) 向量依次排开，方向自然就定了**——从 contact point 指向第一个向量就是初始运动方向，后续每个向量指向下一个运动方向。

### 四、综合答案：ground truth 该长什么样？

| 问题 | 答案 |
|---|---|
| **几个点？** | 5 个 future waypoint |
| **含起始点？** | **不含**。起始点就是 contact point，trajectory 只描述"从 contact point 出发之后往哪走" |
| **方向怎么定？** | 网络直接输出 5 个 **相对位移向量** (dx, dy)，不是绝对坐标。第一个向量 (dx₀, dy₀) 就是初始运动方向 |
| **数据格式** | trajectory 的 ground truth 应该是一个形状为 **(5, 2)** 的 tensor，每行是 `[dx, dy]`（相对于 contact point 的偏移） |
| **loss 怎么算** | `torch.mean((pred_hand - target_hand)**2)`，直接对 5 个 (dx, dy) 做 MSE |

### 五、一个可能的误区

如果这 5 个点是**绝对坐标**，那网络需要同时"知道 contact point 在哪"才能输出合理的位置。但代码里 `hand_head` 的 `condition_contact=False`，说明 trajectory 分支**没有显式接收 contact point 作为条件输入**——它只接收图像编码后的 context。

这进一步印证了：**trajectory 输出的是相对位移**，因为相对位移和 contact point 的绝对位置解耦了。网络只学"看到这张图，手应该往哪个方向动"，具体从哪开始是 contact point 分支的事。

---

## Inpainting（Navier-Stokes Inpainting）

笔记里提到的 inpainting 方法是指：

**Navier-Stokes, Fluid Dynamics, and Image and Video Inpainting**
作者：Marcelo Bertalmío, Andrea L. Bertozzi, Guillermo Sapiro
会议：CVPR 2001

这是一篇非常经典的图像修复（image inpainting）论文。核心思想是把图像中缺失的区域当成"流体空洞"，用 Navier-Stokes 方程（流体力学里的基本方程）来把周围的信息"流"进去，把缺失区域自然地补全。

### 为什么带教会让你看这个？

VRB 项目里有个很类似的"填空"问题。VRB 只预测了 **5 个离散的接触点**，但从这 5 个点到生成一条连续、平滑、有物理合理性的 **trajectory**，中间有很多"空白"需要填补。Navier-Stokes inpainting 本质上就是一个从"局部已知信息"向"未知区域"做平滑传播和补全的方法。带教可能是想让你：

1. **借鉴这个思路**来把 sparse contact points 插值/补全成连续的 trajectory
2. 或者是在处理 video frame 时，有些帧因为遮挡需要把被遮住的手/物体"修复"出来
