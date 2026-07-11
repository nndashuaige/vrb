# E20 实施计划：Seeing-the-Unseen 同款手部 Inpaint + 标签重绘（基于 E18b 口径）

- 日期：2026-07-06（v2，按用户反馈修订：① 新增"heatmap+轨迹重绘到无手图上"的两遍式流程设计；② 代码文件分类存放，不再平铺在 `src/vrbreproduction/` 下）
- 状态：待执行（本文件只是计划，尚未执行任何步骤）
- 计划作者：Claude（与用户讨论后定稿）；执行者：下一个 AI（见 §17 执行提示词）
- 范围：前 100 个 subaction（与 E18b 同口径），预期复现出同样的 **56 个 keep 样本**
- 文件布局约定（用户要求分类存放）：

```text
src/vrbreproduction/e20/__init__.py
src/vrbreproduction/e20/inpaint_utils.py     # 任务 1、2：mask/复检/几何纯函数
src/vrbreproduction/e20/pipeline.py          # 任务 3：e18b 复制改造 + E19 重绘机制移植（export/finalize 两遍）
src/vrbreproduction/e20/server_worker.py     # 任务 5：GPU 服务器自包含脚本
scripts/run_e20.py                           # 唯一 CLI 入口：--stage export|pack|ingest|finalize|report
scripts/run_e20_chunked_driver.py            # 仿 scripts/run_e18b_chunked_driver.py 的分块驱动（VM 45s 超时对策）
tests/test_e20_inpaint_utils.py
outputs/e20/  outputs/e20_smoke10/           # 输出（与其它实验平行）
docs/learning_note/实验记录/实验记录e20.md    # 实验记录（与其它实验平行）
```

---

## 〇、编号、核心设计变更与范围声明（执行 AI 必读）

1. **编号冲突说明**：`docs/plans/2026-07-02-e20-e23-sam3-mask-pipeline.zh.md` 曾把 E20–E23 预留给 SAM3 管线。经用户 2026-07-06 指示，**E20 现定义为本实验**。SAM3 管线编号待用户重排（建议 E24–E27）。不要修改旧 plan 文件，只在实验记录 e20 里记一句此决定。
2. **为什么必须是"两遍式"而不是纯后处理（v2 的核心变更）**：E18b 的样本目录里，`label_heatmap_merged_overlay.png` 和 `vrb_style_affordance.png` 是**已经画在原图（含手）上的成品图**；`reference_frame.png` 是干净的无标注底图；heatmap 本体存了 `label_heatmap_merged.npy`，**但轨迹和接触点的像素坐标没有被持久化**（`candidate_result.json` 只有 `trajectory_points=N` 这样的计数）。用户需求是把 heatmap **和轨迹**画到 P 掉手的图上 → 必须先拿到坐标。E19 已经验证了正确机制（`outputs/e19_.../summary.md` 原话）："E19 先在原始帧上计算 homography、projected contact points、heatmap 参数和 trajectory 坐标；手部 inpainting 只用于最终 plausible hand-free reference canvas 上重绘标签"。因此 E20 采用：
   - **第一遍（export，本地 VM）**：以 E18b 口径原样重放管线，重现 56 个 keep，并把**传给渲染函数的全部几何参数**（接触点、GMM 参数、轨迹像素坐标、箭头端点）序列化成 `labels.json`；同时导出干净底图 + 手 bbox → 打包给 GPU；
   - **GPU 阶段**：只做 inpaint（SAM+LaMa/SD+复检），不碰任何标签；
   - **第二遍（finalize，本地 VM）**：**调用与 E18b 完全相同的渲染函数、传完全相同的参数、只把底图换成 inpaint 后的图**，重新生成 overlay 和 affordance 图。
   - 重绘的黄金法则：**同函数、同参数、换底图**。禁止重新拟合 GMM、重新算 homography、重新投影——第二遍不做任何计算，只做渲染。
3. **管线语义零改动**：export 遍是 `e18b_paper_faithful_cell2_retry.py` 的忠实复制 + 仅两类无侵入改动：(a) 序列化 labels.json；(b) 移植 E19 的手 bbox 收集。所有阈值、gate、重试逻辑一律不动。E18b 的输出目录**只读**。
4. 硬验收：export 遍复现的 keep 集合（56 个 sample_key + ref_idx + reference_mode）必须与 `outputs/e18b/successful_sample_manifest.csv` **完全一致**，否则停下报告（见 §15）。

---

## 一、Seeing the Unseen 原始管线事实核查（方案依据）

以下事实 2026-07-06 已逐条核对原文与官方源码（arXiv 2401.07770，CVPR 2024；GitHub `ram81/seeing-unseen`，`seeing_unseen/dataset/generator.py`）。执行 AI 无需重查。

### 1.1 SU 管线五步（原文 §3.1）

| 步骤               | SU 做法                                                                                                                               |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| (A) 收集           | 文本查询从 LAION-400M 收集 100 万张室内图                                                                                                       |
| (B) 检测           | Detic 开放词表检测目标物体                                                                                                                    |
| (C) 分割           | SAM **ViT-H**（`sam_vit_h_4b8939.pth`），bbox 中心点 point prompt，`multimask_output=3` 取最高分 mask                                          |
| (D) Inpaint + 过滤 | **LaMa（big-lama）与 SD2-inpainting（`stabilityai/stable-diffusion-2-inpainting`）各 50% 概率**；inpaint 后 Detic 复检，与原实例 IoU>90% → 判失败**丢弃** |
| (E) 质量增强         | SDEdit（加 5% 高斯噪声再去噪）+ SD img2img（sd-v1-5，prompt "high resolution, 4k"）                                                              |

训练侧另有两个设计（E21 范畴，本计划仅预留素材）：**distractor 共 inpaint**（每图额外 inpaint 1–4 个无关物体，原文："This also helps prevent the model from overfitting to inpainting artifacts"）；常规增广（blur/noise/hflip/jitter，原文目的 "to mitigate inpainting artifacts"）。

### 1.2 规模与耗时

1M 输入 → **48,728 张 unique 通过** → 生成 1,329,186 张训练图。数据生成耗时/GPU **论文与 repo 均未报告**。本实验 56 张，4090 上纯计算 <15 分钟（§14）。

### 1.3 为什么必须带 (D)(E)

SU 原文："models tend to latch onto inpainting artifacts... high performance on inpainted images, but lower performance on real images"；不做 Step E 时 zero-shot **TP≈0**。我们的 heatmap 标签与 inpaint 区域空间相关，同构且更凶险，(D)(E) 不是可选项。

### 1.4 SU → E20 迁移映射表

| SU 环节 | E20 迁移 | 差异原因 |
| --- | --- | --- |
| (A) LAION 100 万图 | export 遍复现的 56 张 keep reference 帧（456×256，干净底图） | 数据源不同 |
| (B) Detic 检测物体 | **HOA 标注手 bbox**（`data/P01_109.pkl`，score≥0.5，左右手全要）+ MediaPipe Hands 兜底召回 | 有现成手标注；HOA 可能漏检 → 兜底 |
| (C) SAM 中心点 prompt | SAM ViT-H **box prompt** + **手臂延伸**（§8.3）+ 12px 膨胀 | HOA bbox 紧，box 更稳；手连着臂，不延伸会留"断臂涂抹" |
| (D) LaMa/SD 50/50 + Detic 复检 | 每张 **LaMa 和 SD 都跑**，final 按 seeded 50/50 选（双变体保留供 E21 消融）；复检 = YCrCb 肤色比 + MediaPipe + boundary_delta + mask_area_ratio（§8.6） | 56 张全跑双后端成本可忽略；复检目标从"物体没了"换成"手没了" |
| (E) SDEdit + img2img | SDEdit 变体**另存**，不覆盖主结果 | 主结果保证 mask 外像素零改动（标签保真）；SDEdit 作 E21 增广候选 |
| Distractor 1–4 | 可选任务 9 | E21 反捷径素材 |
| （无对应） | **标签重绘**：heatmap+轨迹用缓存坐标重绘到 inpaint 图上（§0.2、§9） | SU 没有空间标签要保；我们有，这是 E20 的核心交付物 |
| （无对应） | **overlap 诊断**：hand mask ∩ heatmap 高值区 / ∩ crop_bbox_150 | 手压住接触区时 inpaint 是幻觉，必须标记（SU 只要"看起来空"，我们要"还原被手挡的物体"） |

---

## 二、当前仓库事实（执行前逐条核对，不符先停下报告）

### 2.1 E18b 输出（作为对照基准与部分素材，目录只读）

- 清单：`outputs/e18b/successful_sample_manifest.csv`，**56 行 keep**；reference_mode 分布：`active_hand_invisible_pre_contact` 32、`min_hand_object_iou_pre_contact_fallback` 24。invisible 只代表 active hand 的 HOA 标注缺失，画面可能仍有另一只手/漏标手 → **56 张全部过手检测**。
- **⚠️ manifest 与 candidate_result.json 里的绝对路径已失效**（含旧 VM 会话挂载名 `/sessions/gracious-amazing-sagan/...`）。凡引用一律按 `outputs/e18b/` 锚点重接到当前仓库根。
- 每个样本目录（如 `outputs/e18b/experiments/E18B/pipeline_outputs/subaction_00_P01_109_0/episode_01_left/frame_000088_left/`）：
  - `reference_frame.png`：**456×256×3 BGR 干净底图（无任何标注绘制）**——inpaint 的输入语义即它（但 E20 用 export 遍自产的同名图，见 §5，避免依赖失效路径）；
  - `label_heatmap_merged_overlay.png`、`vrb_style_affordance.png`：**已画在含手原图上的成品**——E20 要在无手图上重出这两类图；
  - `label_heatmap_merged.npy`（256×456 float）：heatmap 本体；
  - `candidate_result.json`（156 键）：有 `hand, ref_idx, reference_mode, noun, crop_bbox_150, trajectory_points(计数), arrow_generated` 等，**无轨迹/接触点像素坐标**——这就是必须重放 export 遍的原因（§0.2）。
- 交叉校验：`outputs/e18b/experiments/E18B/pipeline_outputs/**/candidate_result.json` 共 81 个，`status=="keep"` 恰 56 个。

### 2.2 可复用代码资产（不要重造轮子）

| 资产 | 位置 | 用途 |
| --- | --- | --- |
| **E18b 管线本体** | `src/vrbreproduction/e18b_paper_faithful_cell2_retry.py`（2804 行） | 任务 3 复制为 `e20/pipeline.py` 的基底（E18b 相对 E18 的 5 处改动已在其中，无需重做；改动清单见 `docs/learning_note/实验记录/实验记录e18b.md`） |
| **E19 的"缓存坐标→inpaint→重绘"机制** | `src/vrbreproduction/e19_e18_hand_inpaint_reference.py`：`build_reference_hand_mask`(:500-535，收集 ref 帧全部 score≥0.5 手 bbox)、inpaint call site 与产物命名(:1160-1200，`reference_frame_original/ reference_hand_mask/ reference_frame_inpainted/ reference_inpaint_before_after.png`)、`_trajectory_pixels_for_ref_shape`(:811)、`write_cell4_pipeline_outputs`(:1279 起，重绘发生地) | 任务 3 移植的参考实现。E19 是"算完坐标→当场 inpaint→当场重绘"单遍结构；E20 拆成两遍（中间隔着 GPU），移植时把"当场 inpaint"换成"序列化 labels.json / 读回 inpaint 结果" |
| 渲染函数 | `src/vrbreproduction/label_heatmap_utils.py`（heatmap overlay 与 VRB 风格 affordance 的绘制入口；在 e18b 的 `write_cell4_pipeline_outputs` 附近可定位确切调用与实参） | §9 重绘必须调**同一函数**；labels.json 的字段=这些函数的实参 |
| mask/质量基础件 | `src/vrbreproduction/hand_inpaint_utils.py`：`bbox_to_mask / combine_hand_masks / inpaint_opencv_telea / compute_mask_area_ratio / compute_boundary_delta / InpaintConfig(mask_pad_px=12, max_mask_area_ratio=0.35, boundary_delta_max=35)` | `e20/inpaint_utils.py` import 它（**不要修改此文件**，E19 依赖） |
| HOA detections | `from epic_kitchens.hoa import load_detections`（e19:32）；pkl=`data/P01_109.pkl`（e19:119）；`detections[int(ref_idx)]` 即 ref 帧 | export 遍收 bbox |
| YCrCb 肤色阈值 | `contact_point_utils.py:23-24`：lower=(0,133,77)，upper=(255,173,127) | §8.6 复检必须同阈值（与 Cell2 口径一致） |
| 分块驱动 | `scripts/run_e18b_chunked_driver.py` | export 遍在 VM 的 45s bash 超时对策，任务 3 仿写 |
| E19 基线数字 | `outputs/e19_.../summary.md`：Telea 19 张 inpaint，median mask_area_ratio=0.059，boundary_delta=0.0 | summary 对比表 |
| 质量汇总脚本 | `scripts/summarize_hand_inpaint_quality.py` | 任务 7 可扩展 |

### 2.3 执行环境约束（三个环境）

| 环境 | 事实 | 承担阶段 |
| --- | --- | --- |
| 本地 VM（Cowork 会话） | Linux aarch64，4 核，**3GB RAM**，磁盘余 ~4.4GB，Python 3.10.12，已有 cv2/numpy/pandas/matplotlib，**无 torch、无 GPU**。SAM ViT-H / SD **不可行** | export / pack / ingest / finalize / report + Telea 假后端 plumbing |
| GPU 服务器（AutoDL RTX 4090，用户租用） | 需 CUDA torch；无特殊 CUDA 版本要求（不需要 SAM3 那套 CUDA≥12.6） | GPU inpaint 阶段 |
| 传输 | gpu_job ≈15MB 上行；结果 ≈60–100MB 下行；用户手动 scp/网盘 | — |
| 网络 | 服务器端 `export HF_ENDPOINT=https://hf-mirror.com`；SAM ckpt 直链 `dl.fbaipublicfiles.com` | — |

VM 需补装（export 遍用）：
```bash
pip install epic-kitchens-100-hoa || pip install git+https://github.com/epic-kitchens/epic-kitchens-100-hand-object-annotations.git
python3 -c "from epic_kitchens.hoa import load_detections; print('ok')"   # 验收；两条都失败则停下报告
```
另外 export 遍复用 e18b 的依赖（该管线此前已在 VM 会话内跑通全量，依赖均为 CPU 库；缺什么按 import 报错补装即可）。

---

## 三、目标行为与产出物

### 3.1 目标

1. export 遍以 E18b 口径重现 56 keep，并导出每样本 `labels.json`（渲染所需全部几何参数）+ 干净底图 + 手 bbox；
2. GPU 遍：SAM 像素级 mask（含手臂延伸）→ LaMa 与 SD2 各出一版 → SDEdit 变体 → 复检过滤；
3. finalize 遍：seeded 50/50 选 final，**用缓存参数把 heatmap overlay 和 vrb_style_affordance（含轨迹箭头）重绘到无手图上**；
4. mask 外像素与原图逐位一致（lama/sd 变体）；无手且复检干净的样本走 `already_clean` 原样通过；
5. 全量质量报告 + contact sheet 人工审查。

### 3.2 产出目录树

```text
outputs/e20/
  export/                                   # 第一遍产物（管线重放）
    experiments/E20/pipeline_outputs/<镜像 e18b 三级样本树>/
      reference_frame.png                   # 干净底图（自产）
      labels.json                           # ★ 渲染参数快照（schema 见 §5.2）
      label_heatmap_merged.npy              # 自产 heatmap（与 e18b 比对用）
      candidate_result.json                 # e18b 同构 + e20 新增字段
    export_manifest.csv                     # 56 行；与 e18b manifest 硬比对
    consistency_report.json                 # keep 集合/npy 比对结果（§5.3）
  gpu_job/                                  # 上传服务器的自包含包
    images/<sample_key>.png
    tasks.json                              # schema 见 §7.2
    run_e20_server.py + e20_inpaint_utils.py 副本
    requirements_server.txt  README_server.md
  gpu_results/                              # 服务器回传解包
    <sample_key>/ mask.png lama.png sd.png sdedit_lama.png sdedit_sd.png qc.json
    results_summary.json
  experiments/E20/pipeline_outputs/<镜像三级样本树>/   # 最终成品（finalize 产出）
    reference_frame_original.png            # 含手底图
    reference_hand_mask.png
    reference_frame_inpainted.png           # final（50/50 选定，训练用底图）
    reference_frame_inpainted_lama.png / _sd.png / _sdedit.png
    reference_inpaint_before_after.png      # 原图|mask|final 三联
    label_heatmap_merged.npy                # = export 遍产物（sha256 一致）
    label_heatmap_merged_overlay_inpainted.png   # ★ heatmap 重绘在 final 上
    vrb_style_affordance_inpainted.png           # ★ 接触点+轨迹箭头重绘在 final 上
    labels.json  candidate_result.json      # candidate_result 顶层加 "e20" 块
  e20_manifest.csv   e20_quality.csv   charts/   summary.md
outputs/e20_smoke10/                        # 同构，仅前 10 个 subaction
```

### 3.3 明确不做什么

- 不修改 E18b/E19/E18 任何代码与输出（复制不算修改）；不改管线任何阈值/gate/逻辑；
- finalize 遍**零计算**：不重拟合 GMM、不重算 homography、不重投影，只换底图渲染；
- 不在本地 VM 装 torch / 跑 SAM/SD；
- 不做训练（伪影捷径的训练侧对策 = E21；任务 9 仅预生成 distractor 素材）。

---

## 四、总体架构：五个子阶段

```text
[A0 export]   VM   scripts/run_e20.py --stage export    重放 E18b 管线 → 56 keep + labels.json
                   （keep 集合与 e18b 硬比对，不一致即中止）
[A1 pack]     VM   --stage pack                          组装 gpu_job/（图+bbox+tasks.json+worker）
[B  gpu]      4090 python run_e20_server.py ...          SAM→mask→LaMa/SD→SDEdit→复检 → tar 回传
[C1 ingest]   VM   --stage ingest                        解包、像素零改动复核、50/50 选 final
[C2 finalize] VM   --stage finalize                      ★ 同函数同参数换底图，重绘 overlay+affordance
[C3 report]   VM   --stage report                        manifest/quality CSV、charts、summary
```

`scripts/run_e20.py` 是唯一入口（argparse `--stage` 分发到 `e20/pipeline.py`）；`e20/server_worker.py` 自包含（打包时连同 `e20/inpaint_utils.py` 复制进 gpu_job，两端逻辑同源）。worker 支持 `--backends telea --device cpu`，使 A0→C3 全链路可先在 VM 用 Telea 假后端跑通。

---

## 五、任务 3 详细规范（先读；任务 1、2 是它的地基）：export 遍与 labels.json

> 任务顺序仍按 §13 执行（1→2→3→…）；把任务 3 放最前详述是因为它是本 plan v2 的核心。

**文件：** `Create: src/vrbreproduction/e20/pipeline.py`（以 `e18b_paper_faithful_cell2_retry.py` 整体复制为基底改造）

### 5.1 相对 e18b 的改动清单（严格限于此四处，每处以 `# E20-CHANGE-n` 注释标记）

1. **实验标识与输出根**：`experiment_id="E20"`，输出根 `outputs/e20/export/`（config 化，smoke 可覆盖）。
2. **labels.json 序列化**（核心）：定位 e18b 中生成 `label_heatmap_merged_overlay.png` 与 `vrb_style_affordance.png` 的渲染调用（在 `write_cell4_pipeline_outputs` 内，渲染函数来自 `label_heatmap_utils.py`）。在**调用处**把传入的全部几何实参原样序列化：接触点像素坐标、GMM 参数（means/covariances/weights/n_components）、heatmap_mode、轨迹像素坐标序列、箭头端点、crop_bbox_150、heatmap_shape、以及渲染函数名和非几何 kwargs（颜色/alpha/线宽等，一并存下以便重绘时零歧义）。**先读代码确认实参的真实结构再定 schema 细节**——§5.2 是目标形状，字段名以实际渲染签名为准，宁可多存不可少存。
3. **手 bbox 收集**：移植 E19 `build_reference_hand_mask` 的 bbox 遍历部分（只收集，不建 mask——mask 在 GPU 端由 SAM 生成），把 `all_hand_bboxes_px / active_hand_bbox_px` 写入 candidate_result.json 与 labels.json。
4. **一致性钩子**：每个 keep 落盘后记录 sample_key、ref_idx、reference_mode、npy sha256，汇入 `export_manifest.csv`。

### 5.2 labels.json schema（目标形状）

```json
{
  "schema_version": 1,
  "sample_key": "subaction_00_P01_109_0__episode_01_left__frame_000088_left",
  "image_hw": [256, 456],
  "hand": "left", "ref_idx": 69, "reference_mode": "active_hand_invisible_pre_contact",
  "contact_points_px": [[x, y], ...],
  "gmm": {"means": [[x, y], ...], "covariances": [...], "weights": [...], "n_components": 5},
  "heatmap_mode": "covariance",
  "trajectory_px": [[x, y], ...],
  "arrow": {"start": [x, y], "end": [x, y]},
  "crop_bbox_150": [131, 77, 281, 227],
  "hand_bboxes_px": [[x1, y1, x2, y2], ...], "active_hand_bbox_px": null,
  "render": {
    "heatmap_overlay_fn": "label_heatmap_utils.<实际函数名>", "heatmap_overlay_kwargs": {...},
    "affordance_fn": "label_heatmap_utils.<实际函数名>", "affordance_kwargs": {...}
  }
}
```

### 5.3 一致性校验（export 结束时自动跑，写 `consistency_report.json`）

- **硬指标**：keep 集合（56 个 sample_key + ref_idx + reference_mode）与 `outputs/e18b/successful_sample_manifest.csv` 逐行一致 → 不一致**中止并报告**（先检查管线随机源，如 GMM 是否带 `random_state`；不许静默放行）。
- **软指标**：逐样本 `label_heatmap_merged.npy` 与 e18b 同名文件比对，报告 `max_abs_diff`。若非 0（GMM 未固定 seed 之类），记录并继续——以 export 自产 npy 为 E20 标签本体（同一管线的合法产物），但必须写进 summary 与实验记录。
- 顺带校验：自产 `reference_frame.png` 与 e18b 同名文件逐位一致（同一 `ref_idx` 同一原始帧，理应一致；不一致要查）。

### 5.4 运行方式（VM 45s 超时对策）

仿 `scripts/run_e18b_chunked_driver.py` 写 `scripts/run_e20_chunked_driver.py`：按 subaction 区间分块断点续跑，产物幂等（已完成样本跳过）。预计总时长与 E18b 全量相当（此前已在 VM 内完成过，量级可行）。

---

## 六、任务 1：基础工具（`e20/inpaint_utils.py` 上半部分）

**文件：**
```text
Create: src/vrbreproduction/e20/__init__.py
Create: src/vrbreproduction/e20/inpaint_utils.py
Create: tests/test_e20_inpaint_utils.py
```

先写测试再实现（仓库惯例）。纯函数：

```python
def fix_e18b_path(stale_path: str, repo_root: Path) -> Path
    # 按 'outputs/e18b/' 锚点重接失效绝对路径；无锚点抛 ValueError

def sample_key_from_rel_dir(rel_dir: str) -> str   # 末三段 '__' 连接，可逆
def rel_dir_from_sample_key(sample_key: str) -> str

def collect_hand_bboxes_px(frame_det, image_hw, score_threshold=0.5) -> list[list[float]]
    # 该帧全部 score>=0.5 手的像素 bbox（左右手都要）；参考 e19:500-535；越界 clip
```

**测试要点**（不依赖 torch/epic_kitchens；detections 用最小 stub 模拟 `hands[i].score/.side.name/.bbox.left...`）：路径修复正反例、sample_key 可逆、bbox 过滤/换算/clip、空列表合法（`already_clean` 候选）。

```bash
python -m pytest tests/test_e20_inpaint_utils.py -v
```

---

## 七、任务 2：几何与质量工具（`e20/inpaint_utils.py` 下半部分）

这些函数 GPU worker 也用（pack 时把本文件复制进 gpu_job，worker 同目录 import，两端同源）：

```python
def extend_mask_to_border(mask_u8, bbox_px, image_hw, max_extend_frac=0.45, width_scale=1.2)
    # 手臂延伸：mask 未触下边界且 (H-bbox_y2)/H<=max_extend_frac 时，bbox 底边到下边界补梯形
    # （顶宽=bbox宽*width_scale，底宽=顶宽*1.3，水平居中）取并；EPIC 第一视角手臂几乎总从下方进入
def dilate_mask(mask_u8, pad_px=12)                      # 椭圆核，口径与 hand_inpaint_utils 一致
def letterbox_pad(image_bgr, mask_u8, target=512)        # REPLICATE 填充，零重采样；配套 unletterbox
def composite_inside_mask(original, edited, mask, feather_px=3)  # 羽化带以外逐位等于原图
def skin_ratio_in_mask(image_bgr, mask_u8, erode_px=3)   # YCrCb (0,133,77)-(255,173,127)，同 Cell2
def heatmap_top_bbox(heatmap, thresh_frac=0.5)           # heatmap>0.5max 外接框（overlap 诊断/distractor 避让）
def mask_overlap_ratio(mask_u8, bbox_px) -> float
def choose_final_backend(sample_key, passed: dict, seed=20260707) -> str
    # rng=np.random.default_rng(seed + zlib.crc32(sample_key.encode()))，在通过的 {'lama','sd'} 里选；
    # 只一个通过选它；都不过返回 'none'。禁止用内置 hash()（跨进程不稳定）
```

**测试要点**：梯形几何三种分支；letterbox 往返逐位一致；composite 在 mask 外 `np.array_equal`；skin_ratio 合成图 1/0；choose_final_backend 确定性 + 大致均匀。

---

## 八、任务 4 与任务 5：pack 与 GPU worker

### 8.1 任务 4：pack（`--stage pack`）

从 `outputs/e20/export/` 读 56 个样本 → `gpu_job/images/<sample_key>.png`（干净底图副本）+ `tasks.json` + worker/utils 副本 + `requirements_server.txt` + `README_server.md`（写给用户的逐条命令，内容即 §8.2）。打印统计：样本数、`hand_bboxes_px` 为空的样本数（预期集中在 32 张 invisible-ref）。

**tasks.json schema**：
```json
{ "experiment_id": "E20", "created_utc": "...", "video_id": "P01_109",
  "image_hw": [256, 456], "seed": 20260707,
  "samples": [ { "sample_key": "...", "rel_sample_dir": "...", "image": "images/<key>.png",
      "subaction_index": 0, "narration_id": "P01_109_0", "hand": "left",
      "reference_mode": "...", "ref_idx": 69, "noun": "rucksack",
      "hand_bboxes_px": [[...]], "active_hand_bbox_px": null,
      "crop_bbox_150": [131, 77, 281, 227], "heatmap_top_bbox_px": [...] } ] }
```
`heatmap_top_bbox_px` 由 pack 从 export 遍的 npy 算出（服务器不需要 npy）。

### 8.2 任务 5：GPU worker（`e20/server_worker.py`，自包含）

CLI：`python run_e20_server.py --job-dir . --out-dir /root/e20_gpu_results --device cuda --backends lama,sd --sdedit 1 --seed 20260707 [--limit -1] [--distractor 0]`。torch/diffusers/segment_anything/mediapipe 全部函数内延迟 import（`--backends telea` 在无 torch 的 VM 可跑）。

**服务器环境（README_server.md 正文，用户在 AutoDL 4090 执行）**：
```bash
pip install segment-anything simple-lama-inpainting "diffusers>=0.27" transformers accelerate safetensors opencv-python-headless mediapipe pillow numpy
export HF_ENDPOINT=https://hf-mirror.com
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O /root/sam_vit_h_4b8939.pth
# big-lama 首次运行自动下载；github 直连失败则手动下
#   https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt 并 export LAMA_MODEL=/root/big-lama.pt
cd /root/e20_gpu_job && python run_e20_server.py --job-dir . --out-dir /root/e20_gpu_results --device cuda --backends lama,sd --sdedit 1
tar czf /root/e20_gpu_results.tar.gz -C /root e20_gpu_results
```
模型（对齐 SU）：SAM `vit_h`；LaMa big-lama；SD2-inpaint `stabilityai/stable-diffusion-2-inpainting`（fp16）；SDEdit 用 `stable-diffusion-v1-5/stable-diffusion-v1-5`（**runwayml 原 repo 已下架**）。simple-lama 装不上的备胎：`pip install iopaint`（`iopaint run --model=lama`）。

**逐环节规范**：
1. **SAM**：`SamPredictor.set_image(RGB)`；每个 `hand_bboxes_px` → box prompt（外扩 8px、clip）→ `multimask_output=True` 取最高分（对齐 SU "3 选 1"）；全手取并 M1，记 sam_score。`hand_bboxes_px` 为空 → MediaPipe Hands（`static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5`）兜底；检出用 landmark 外接框作 prompt；仍无 → `already_clean_candidate`，mask 全零，只测全图肤色比，不 inpaint。
2. **mask 后处理**：逐手 `extend_mask_to_border` → 可选 SAM 二次精化（扩展 box 再跑一次，取新 mask 与梯形区交集并入，让"补臂"贴真实手臂像素，默认开）→ 并集 → `dilate_mask(12)` → gate：`mask_area_ratio>0.35` 回退纯 bbox mask+膨胀并记 `mask_fallback_bbox=true`。诊断：`mask_touches_bottom / arm_extension_applied / mask_area_ratio / overlap_heatmap_top / overlap_crop150`；**`overlap_heatmap_top>0` 打 `inpaint_on_label_region` 标记**（手压接触区 → 幻觉风险 → Tier-B）。
3. **LaMa**：`SimpleLama()` 原生 456×256；结果过 `composite_inside_mask`。
4. **SD2-inpaint**：`letterbox_pad` 512×512 → `steps=30, guidance=7.5, generator=manual_seed(seed+subaction_index)` → unletterbox → composite。`DEFAULT_SD_PROMPT="a photo of a kitchen, natural indoor scene, high quality"`；`DEFAULT_SD_NEGATIVE="hand, hands, fingers, arm, wrist, person, human, skin, glove, text, watermark"`（本方案自定，SU 未公开其 SD prompt；CLI 可覆盖）。
5. **硬保证**：composite 后 mask（含羽化带）之外像素与原图 `np.array_equal`，worker 自检写入 qc。
6. **SDEdit 变体**：对 lama.png / sd.png 各跑 img2img：`strength=0.1, steps=50, prompt="high resolution, 4k"`（≈SU 的 5% 噪声口径），456×256 原生跑。全图像素都变，仅作 E21 增广候选。
7. **复检**（对 lama/sd 各判）：`skin_ratio_after>0.10`→fail（Cell2 同款阈值）；MediaPipe 复检检出手且外接框中心落在膨胀 mask 内→fail；`boundary_delta>35`→fail；`mask_area_ratio>0.35`→fail；`lap_var_ratio`（mask 内/外 Laplacian 方差比）只记录。两后端都 fail→`inpaint_failed_kept_original` 进 discard-flag 清单。`already_clean_candidate` 且全图 skin_ratio<0.02 且 MediaPipe 无检出→`already_clean`。
8. **输出**：逐样本 `qc.json`；全局 `results_summary.json`（库版本、权重 sha256 前 8 位、seed、逐样本耗时）。

---

## 九、任务 6：ingest 与 finalize（★ 重绘发生地）

### 9.1 ingest（`--stage ingest --results-tar <tar>`）

1. 解包校验样本数与 tasks.json 一致；
2. **像素零改动本地复核**（不信任服务器自检）：lama/sd 变体在膨胀 mask+羽化带之外 `np.array_equal(original, variant)` 100% 通过，失败即中止；
3. `choose_final_backend` 选 final；按 §3.2 落盘（三级样本树、三联图、candidate_result.json 顶层加 `"e20"` 块：backend_final/status/qc 摘要/seed/sample_key）。

### 9.2 finalize（`--stage finalize`）——同函数、同参数、换底图

对每个 final 产出（含 `already_clean` 用原图）：
1. 读该样本 `labels.json`；
2. 按 `render.heatmap_overlay_fn` + 缓存 kwargs + **底图=reference_frame_inpainted.png** 调 `label_heatmap_utils` 同一函数 → `label_heatmap_merged_overlay_inpainted.png`；
3. 同法用 `render.affordance_fn` + 缓存的接触点/GMM means/轨迹/箭头参数 → `vrb_style_affordance_inpainted.png`；
4. **自校验**：用同一参数在**原图**上重绘一份（临时文件），与 e18b 的 `label_heatmap_merged_overlay.png` / `vrb_style_affordance.png` 求逐像素 diff——若渲染管道正确，diff 应为 0（或仅 npy 软差异传导的微小值）。该校验证明"换底图前渲染是保真的"，逐样本结果写入 `e20_quality.csv` 的 `redraw_selfcheck_max_diff` 列。任何样本 selfcheck 大于阈值（默认 0）→ 停下检查渲染参数序列化是否漏了字段；
5. `label_heatmap_merged.npy` 从 export 遍复制并 sha256 比对。

### 9.3 为什么不从 e18b 成品图上"抠掉手再补画"

已绘制的 overlay 是 alpha 混合结果，heatmap/箭头与底图像素不可逆地融合，任何"图上修补"都会破坏标签保真性。唯一正确路径就是参数级重绘（§0.2 黄金法则）。此段写进实验记录，作为 E20 与 E19 一脉相承的方法论表述。

---

## 十、任务 7：report（质量统计与图表）

### 10.1 charts（命名仿 e19）

- `e20_inpaint_contact_sheet_page_NNN.png`：每行一样本，六列 = 原图 | mask 叠加 | LaMa | SD | **affordance_inpainted** | **overlay_inpainted**（每页 ≤6 行）——人工审查主界面（同时看"手没了"和"标签画对了"）；
- `e20_sdedit_contact_sheet_page_NNN.png`：final | sdedit(final)；
- `e20_funnel.png`：56 → 需 inpaint → LaMa 过检/SD 过检 → final / already_clean / failed；
- `e20_skin_ratio_before_after_hist.png`、`e20_boundary_delta_by_backend_hist.png`、`e20_mask_area_ratio_hist.png`。

### 10.2 `e20_manifest.csv`（56 行）字段

`sample_key, rel_sample_dir, subaction_index, narration_id, verb, noun, hand, reference_mode, ref_idx, status(final_inpainted|already_clean|inpaint_failed_kept_original), backend_final, hand_count_masked, mask_area_ratio, arm_extension_applied, mask_fallback_bbox, skin_ratio_before, skin_ratio_after_final, boundary_delta_final, mediapipe_after_final, overlap_heatmap_top, overlap_crop150, inpaint_on_label_region, redraw_selfcheck_max_diff, tier(A|B), path_final, path_lama, path_sd, path_sdedit, path_mask, path_overlay_inpainted, path_affordance_inpainted`

Tier：`(final_inpainted|already_clean) 且 inpaint_on_label_region==False 且 selfcheck 通过` → **Tier-A**；否则 **Tier-B**。`e20_quality.csv`：逐样本×逐变体全指标。

---

## 十一、任务 8：summary.md 与 实验记录e20.md

summary 骨架：实验定义（含 §0.2 两遍式设计一段话 + SU 引用）→ 环境与模型版本 → export 一致性结果（keep 集合比对 + npy max_abs_diff）→ Funnel → 复检通过率（分后端×分 reference_mode）→ 质量指标中位数（对比 E19 Telea：19 张、area 0.059、boundary_delta 0.0）→ redraw selfcheck 结果 → Tier 分布 → 失败清单 → 实测耗时/成本。

实验记录（Motivation/Method/Results/Limitations 格式）：Motivation 引 §1.3 伪影捷径证据；Method 必须写"同函数同参数换底图"与 §9.3 的方法论；Limitations 两条必写：(1) inpaint 是 plausible 非 true，`inpaint_on_label_region` 样本尤甚；(2) 伪影捷径的训练侧对策（distractor/增广/三臂消融）= E21。另记编号决定（§0.1）与 npy 软差异（若有）。

---

## 十二、任务 9（可选，默认执行）：distractor 变体

对全部 56 张（含天然无手图）随机采 1–3 个矩形/椭圆区域（各占 3–8% 面积；与膨胀手 mask、`heatmap_top_bbox` 膨胀 20px、`crop_bbox_150` 三者零相交；rng seed=seed+1000+subaction_index），用与 final 同后端 inpaint → `reference_frame_inpainted_distractor.png` + mask。worker `--distractor 1` 实现，零相交自动断言。E20 final 不使用；为 E21 反捷径训练备料（SU 同款设计）。

---

## 十三、执行顺序与命令速查（严格按序，每步验收后再进下一步）

```bash
# ① 环境 + 单元测试（VM）
pip install epic-kitchens-100-hoa || pip install git+https://github.com/epic-kitchens/epic-kitchens-100-hand-object-annotations.git
python -m pytest tests/test_e20_inpaint_utils.py -v                  # 全绿

# ② export smoke（前 10 个 subaction）+ Telea 假后端全链路 plumbing（VM，无 torch）
python scripts/run_e20.py --stage export --output-root outputs/e20_smoke10 --limit 10
#    ↳ 检查 consistency_report.json：keep 子集与 e18b 前 10 subaction 的 keep 一致
python scripts/run_e20.py --stage pack   --output-root outputs/e20_smoke10
python outputs/e20_smoke10/gpu_job/run_e20_server.py --job-dir outputs/e20_smoke10/gpu_job \
  --out-dir outputs/e20_smoke10/gpu_results_local --device cpu --backends telea --sdedit 0
python scripts/run_e20.py --stage ingest   --output-root outputs/e20_smoke10 --results-dir outputs/e20_smoke10/gpu_results_local
python scripts/run_e20.py --stage finalize --output-root outputs/e20_smoke10   # ★ 重绘 selfcheck 必须全 0
python scripts/run_e20.py --stage report   --output-root outputs/e20_smoke10

# ③ export 全量（VM，分块断点续跑；时长与 E18b 全量同量级）
python scripts/run_e20_chunked_driver.py --output-root outputs/e20
#    ↳ consistency_report.json：56 keep 与 e18b 完全一致（硬验收）
python scripts/run_e20.py --stage pack --output-root outputs/e20

# ④ 【用户操作】上传 outputs/e20/gpu_job 至 AutoDL 4090，按 README_server.md 执行；
#    建议先 --limit 10 GPU smoke → 回传 → 本地出 contact sheet 人工审查 → 通过再全量
# ⑤ 回收 + 重绘 + 报告（VM）
python scripts/run_e20.py --stage ingest   --output-root outputs/e20 --results-tar <回传tar>
python scripts/run_e20.py --stage finalize --output-root outputs/e20
python scripts/run_e20.py --stage report   --output-root outputs/e20
# ⑥ 写实验记录；人工审查 contact sheet；裁决 discard-flag 清单
```

---

## 十四、算力与耗时预估

| 项 | 预估 |
| --- | --- |
| export smoke10 / 全量（VM CPU） | smoke 分钟级；全量与 E18b 相当（小时级，chunked 断点续跑，此前 VM 已跑通过同量级） |
| pack / ingest / finalize / report（VM） | 各 <5 分钟 |
| 服务器环境准备（模型 ~12GB） | 30–60 分钟（一次性） |
| GPU 计算：SAM 56×0.2-0.4s + LaMa 56×0.1s + SD 56×1-2s + SDEdit 112×0.5-1s + 复检 | **<15 分钟** |
| AutoDL 4090 成本 | ≈ ¥2–5 |
| 传输 | 上行 ~15MB，下行 ~60–100MB |

---

## 十五、验收标准（总）

1. `pytest tests/test_e20_inpaint_utils.py` 全绿（VM，无 torch）；
2. **export 一致性（硬）**：全量 keep 集合（56 个 sample_key+ref_idx+reference_mode）与 e18b manifest 完全一致；`reference_frame.png` 与 e18b 逐位一致；npy `max_abs_diff` 已报告（非 0 需在 summary/记录中解释成因）；
3. Telea plumbing smoke（②）A0→C3 全链路无错，目录结构符合 §3.2；
4. **重绘保真（硬）**：finalize selfcheck（同参数原图重绘 vs e18b 成品图）56/56 通过（diff=0，或仅 npy 软差异传导且已解释）；56/56 样本产出 `label_heatmap_merged_overlay_inpainted.png` 和 `vrb_style_affordance_inpainted.png`（含轨迹箭头）且底图为无手图；
5. GPU smoke10 contact sheet 人工审查通过后才允许全量；
6. 复检通过率（`final_inpainted`+`already_clean`）≥ 90%（≥51/56），未过者入 discard-flag 清单且原因明确；
7. **像素保真（硬）**：lama/sd 变体 mask 外零改动本地复核 100% 通过；final 目录 npy sha256 与 export 遍一致；
8. e18b 目录零写入；e18/e18b/e19 源码零修改；
9. 可复现：seed/版本/权重 hash 记录齐全；`choose_final_backend` 与重绘幂等（重跑 ingest/finalize 结果不变）；
10. 无手样本正确走 `already_clean`（不 inpaint、不产伪影）；`overlap_heatmap_top>0` 样本全部标 `inpaint_on_label_region` 且归 Tier-B；
11. `e20_manifest.csv`(56 行)、`e20_quality.csv`、charts、summary.md、实验记录e20.md 齐全且数字互相一致；summary 含 E19(Telea) 对比表；
12. （若执行任务 9）56 张 distractor 变体齐全且三避让区零相交（自动断言）。

---

## 十六、风险与已知局限

| 风险 | 对策 |
| --- | --- |
| export 遍不能精确复现 e18b keep 集合（随机源/环境差异） | 硬验收 §15-2 兜底；先查 GMM `random_state` 等随机源；不一致中止报告，不许静默放行 |
| labels.json 漏存渲染参数导致重绘失真 | §9.2-4 selfcheck（原图重绘 vs e18b 成品逐像素比）设计上就是抓这个的；宁可多存参数 |
| HOA 漏检手（invisible 判定本就来自标注缺失） | MediaPipe 兜底 + 复检肤色比 + contact sheet 人工审查（三道关） |
| 手臂延伸梯形误伤背景 | SAM 二次精化 + area gate + 人工审查；`width_scale/max_extend_frac` CLI 可调，GPU smoke 阶段标定 |
| SD 幻觉出错误物体/几何 | negative prompt + composite 限制改动范围 + `inpaint_on_label_region` 标记 + LaMa 对照 |
| **伪影捷径** | E20 数据侧无法根除（SU 结论）；素材已备齐（双后端+SDEdit+distractor），训练侧消融=E21，不在 E20 扩大战线 |
| runwayml/sd-v1-5 已下架 | 用 `stable-diffusion-v1-5/stable-diffusion-v1-5`（§8.2 写死） |
| 国内 HF 网络 / 权重下载 | `HF_ENDPOINT=https://hf-mirror.com`；SAM/LaMa 直链或手动上传 |
| `epic_kitchens` 装不上 | §2.3 两条路径；都失败停下报告（不要自造 pkl 解析器） |
| VM 3GB 内存 / 45s bash 超时 | 严禁 VM 装 torch；export 用 chunked driver；Telea 假后端专为 plumbing 验证 |

---

## 十七、给下一个 AI 的执行提示词

> 你在 `/Users/huhu/Documents/Code/bishe/vrb/vrbreproduction` 仓库中执行毕设实验 E20。**唯一权威规范是 `docs/plans/e20plan.md`（v2），先完整读完再动手，严格照做，不要自行改设计。**
>
> 背景一句话：E18b 有 56 个 keep 样本，但其 heatmap/轨迹成品图画在含手原图上，且轨迹坐标未持久化；E20 要 (a) 以 E18b 口径重放管线并把渲染参数序列化（export 遍），(b) 用 Seeing the Unseen（arXiv 2401.07770）同款流程在 GPU 上把手 P 掉（SAM ViT-H → LaMa/SD2 50/50 → 复检 → SDEdit），(c) **用"同函数、同参数、换底图"的方式把 heatmap 和轨迹箭头重绘到无手图上**（finalize 遍）。管线语义零改动，E18b 目录只读。
>
> 文件布局（plan 开头）：管线与工具代码放 `src/vrbreproduction/e20/` 包内，CLI 入口与分块驱动放 `scripts/`，测试放 `tests/`——**不要把新 py 文件平铺在 `src/vrbreproduction/` 根下**。
>
> 执行顺序（plan §13）：
> 1. 核对 plan §2 仓库事实，不符先停下报告；装 `epic_kitchens` HOA 库（§2.3）；
> 2. 任务 1、2：写 `e20/inpaint_utils.py` + 测试，先测试后实现，pytest 全绿；
> 3. 任务 3：复制 `e18b_paper_faithful_cell2_retry.py` 为 `e20/pipeline.py`，只做 §5.1 的四处标记改动（核心是在渲染调用处序列化 labels.json——先读 `label_heatmap_utils.py` 和 e18b 的 `write_cell4_pipeline_outputs` 确认渲染函数真实签名，再定 labels.json 字段，宁多存不少存）；写 `scripts/run_e20.py` 与 `scripts/run_e20_chunked_driver.py`；
> 4. 任务 4、5：pack 与自包含 GPU worker（`e20/server_worker.py`）；
> 5. 跑 §13-② smoke：export 前 10 subaction（keep 子集须与 e18b 一致）+ Telea 假后端全链路 + **finalize selfcheck 必须全 0**（同参数在原图重绘应与 e18b 成品逐像素相同——这是重绘保真的关键证明）；
> 6. 跑 §13-③ export 全量（chunked，56 keep 与 e18b 完全一致为硬验收）并 pack，**停下来把 `outputs/e20/gpu_job/` 和 README_server.md 交给用户**——GPU 步骤由用户在 AutoDL 4090 执行，你不要假装跑过；建议用户先 `--limit 10` GPU smoke，回传后你出 contact sheet 请用户人工审查，通过再全量；
> 7. 拿到回传 tar：ingest（mask 外像素零改动本地复核）→ finalize（重绘两类标签图）→ report，逐条核对 plan §15 的 12 条验收标准；
> 8. 写 `docs/learning_note/实验记录/实验记录e20.md`（含 E19 Telea 对比、Tier 分布、discard-flag 清单、编号冲突决定、npy 软差异说明），向用户汇报。
>
> 硬约束：finalize 零计算（不重拟合 GMM/不重算 homography，只换底图渲染）；mask 外像素零改动；npy/坐标零改动（sha256）；e18b 只读、e18/e18b/e19 源码零修改；所有随机性带 seed；VM 里不装 torch；每完成一个任务节点向用户简报。plan 未覆盖的情况，停下来问，不要即兴发挥。

---

## 附：与 E 系列的关系速查

- **输入口径**：E18b（论文忠实版 + Cell2 邻帧重试，56/100 keep）——E20 的 export 遍是它的忠实重放 + 参数序列化
- **机制前身**：E19（"原图算坐标→inpaint→重绘"单遍结构，Telea/bbox mask；E20 = 该机制拆两遍 + SAM 像素 mask + 真·LaMa/SD 双后端 + SU 式复检/增强 + 轨迹重绘完整化）
- **下游**：E21（建议）= 训练侧三臂消融 {不P手, LaMa-only, 全套(50/50+distractor+增广)}，eval 只用 Tier-A——检验伪影捷径
- **并行**：SAM3 管线（原 E20–E23 plan，编号待重排）——将来可把 HOA bbox 源换成 SAM3 hand masklet，其余复用 E20 代码
