# 去手 Inpaint Reference E19 实施计划

> **给下一个 AI/执行者：** 请按任务逐步执行。不要覆盖现有 `E18_hybrid_min_iou_reference` 实验。

**目标：** 基于现有 E18 的 hybrid reference 逻辑，增加“手部 mask 引导的 inpainting 去手”步骤，过滤明显失败样本，并评估更干净的 reference 是否能改善最终可视化和训练 tuple 的画面质量，同时尽量不改变 E18 的样本选择、reference 选择和几何投影链路。

**总体架构：** 保持 E18 的 contact 选择、hybrid reference 选择、homography 和投影逻辑不变。先在原始 contact/reference frame 上完成 Cell2、Cell3，得到并缓存投影后的 contact points、heatmap 参数和 trajectory 坐标；然后只对原始 reference image 做 hand inpainting；最后在去手后的 clean reference 上重新绘制 heatmap、contact points、trajectory 和 VRB-style affordance。第一版不要让 inpaint 后的图参与 homography 特征匹配；几何仍在原图上算，inpaint 图只作为最终 clean reference canvas。

**技术栈：** Python、OpenCV、NumPy、pandas、现有 VRB reproduction pipeline、可选 LaMa 命令行包装、可选 Stable Diffusion inpainting 命令行包装、现有 HOA hand detections。

---

## 一、论文依据

本地文献目录：

```text
/Users/huhu/Documents/Code/bishe/去除手的算法文献
```

最相关的是 `Seeing the Unseen.pdf`。它不是自己提出新的 inpainting 模型，而是使用现成模型构造训练数据。它的流程是：

```text
原始图片
-> 检测目标物体
-> 用 SAM 得到目标物体 segmentation mask
-> 用 LaMa 或 Stable Diffusion inpainting 把物体补没
-> 再检测一次，过滤 inpaint 失败的样本
-> 可选：用 SDEdit 或 Stable Diffusion img2img 做质量增强
```

对本项目有用的点：

- 它证明“检测/分割目标 -> mask-guided inpainting -> 过滤失败样本”是合理的数据构造路线。
- 它大规模造数据时，对每张图随机用 LaMa 或 Stable Diffusion inpainting，概率各 50%。
- 它不是一张图同时用两个模型。
- 它强调 inpainting artifact 会污染训练，所以必须做过滤。
- 它还做 SDEdit / SD img2img 质量增强，但这一步可能改动整张图，对本项目的 contact/geometry 有风险。

对 VRB 项目，不建议第一版照搬 50/50 随机策略。我们的目标不是造百万级多样化数据，而是让 E18 最终训练 reference 尽量干净且保持几何可信。推荐顺序：

```text
第一步：OpenCV Telea，作为无依赖 smoke baseline
第二步：LaMa，作为主要可尝试后端
第三步：Stable Diffusion inpainting，作为困难样本的定性对比
暂不默认启用：SDEdit / SD img2img，因为它可能改动物体边缘和接触区域
```

`High-Resolution Image.pdf` 支持 latent diffusion / Stable Diffusion inpainting 作为强生成式后端。但它可能 hallucinate 出看似合理、实际不存在的结构。

`SDEDIT（减少 inpainting 痕迹）.pdf` 支持“加噪再去噪以增强真实感”的思路。对本项目，它适合作为后续视觉增强实验，不适合作为第一版定量 pipeline 的默认步骤。

当前文献目录缺少 LaMa 原论文。如果最终论文正文要正式引用 LaMa，请补充：

```text
Resolution-Robust Large Mask Inpainting with Fourier Convolutions
GitHub: https://github.com/advimman/lama
```

## 二、当前仓库事实

相关文件：

```text
/Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/src/vrbreproduction/e18_hybrid_min_iou_reference.py
/Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/src/vrbreproduction/contact_point_utils.py
/Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/src/vrbreproduction/problem3_utils.py
/Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/src/vrbreproduction/pipeline_retention.py
/Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/outputs/e18/summary.md
```

注意：当前 `E18` 已经被占用：

```text
src/vrbreproduction/e18_hybrid_min_iou_reference.py
outputs/e18/
experiment name: E18_hybrid_min_iou_reference
```

所以去手 inpainting 实验不要覆盖现有 E18。建议实现成：

```text
src/vrbreproduction/e19_e18_hand_inpaint_reference.py
outputs/e19_e18_hand_inpaint_reference/
experiment_id: E19
experiment name: E19_e18_hand_inpaint_reference
```

如果论文叙事里必须叫 E18，可以写成 `E18-inpaint variant`，但代码和输出目录用 E19，避免和已有结果冲突。

## 三、目标行为

新实验应尽量保持 E18 的样本、reference 和几何路径不变：

```text
E18 candidate selection
-> E18 hybrid reference selection:
   active_hand_invisible_pre_contact 优先；
   找不到时 fallback 到 min_hand_object_iou_pre_contact_fallback
-> Cell2 contact/GMM 在原始 contact frame 上运行
-> Cell3 homography 在原始帧上运行
-> 如果 reference 里 active hand 可见，则生成 hand mask
-> 在原始 reference 坐标系中缓存 projected contact points / heatmap 参数 / trajectory points
-> 对 reference image 做 inpainting
-> 跑 inpaint 质量检查
-> 在 inpainted reference 上重新绘制 heatmap / contact points / trajectory / overlay
```

不要 inpaint contact frame。contact frame 里的手和物体交互对 contact point extraction 和 trajectory 是必要信息。

第一版不要把 inpainted image 用于 feature matching。homography 仍然使用原始 frame，这样 E19 能和 E18 做公平对比。

明确禁止这个顺序：

```text
原始 reference -> 先画 heatmap/trajectory -> 再 inpaint 手
```

这样会让 inpainting 把已经画上去的热力图、轨迹线、箭头当成图像内容一起修掉或污染掉。

正确顺序是：

```text
原始 reference -> 算并保存所有投影坐标/heatmap 参数 -> inpaint 手 -> 在 clean reference 上重绘
```

这个顺序成立的前提是：inpainting 不 resize、不 crop、不 warp 图像，只替换 mask 内像素，因此原始 reference 和 clean reference 的宽高、坐标系完全一致。

核心对比是：

```text
E18：hybrid reference 原图，active-hand-invisible 优先，必要时 fallback 到 min-IoU reference
E19：同一张 E18-selected reference、同一组已缓存的投影坐标/heatmap 参数，但最终在去手后的 reference canvas 上重绘标签
```

这样可以隔离“去手后 reference 是否更干净”的效果。

因为 E18 有两种 reference mode，E19 的所有 inpainting 统计都要按 `reference_mode` 分桶报告：

```text
active_hand_invisible_pre_contact
min_hand_object_iou_pre_contact_fallback
```

直观预期：`active_hand_invisible_pre_contact` 本来就更干净，通常不需要或很少需要 inpaint；`min_hand_object_iou_pre_contact_fallback` 更可能残留手，因此更需要重点检查 inpainting 质量。

## 四、任务 1：新增手部 mask 和 inpainting 工具模块

**文件：**

```text
Create: /Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/src/vrbreproduction/hand_inpaint_utils.py
Create: /Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/tests/test_hand_inpaint_utils.py
```

先写测试，使用合成图像 shape `(100, 200, 3)` 和 bbox `[20, 10, 60, 50]`。

期望：

- mask shape 为 `(100, 200)`
- dtype 为 `uint8`
- 手部区域为 255
- 外部区域为 0
- dilation 会扩大 mask，但不越界

运行：

```bash
cd /Users/huhu/Documents/Code/bishe/vrb/vrbreproduction
pytest tests/test_hand_inpaint_utils.py -v
```

初始预期：失败，因为模块还不存在。

实现函数：

```python
def bbox_to_mask(image_shape, bbox_px, pad_px=12, ellipse=False) -> np.ndarray:
    ...

def combine_hand_masks(image_shape, bboxes_px, pad_px=12, ellipse=False) -> np.ndarray:
    ...

def inpaint_opencv_telea(image_bgr: np.ndarray, mask_u8: np.ndarray, radius: float = 3.0) -> np.ndarray:
    ...

def compute_mask_area_ratio(mask_u8: np.ndarray) -> float:
    ...

def compute_boundary_delta(original_bgr: np.ndarray, edited_bgr: np.ndarray, mask_u8: np.ndarray, band_px: int = 5) -> float:
    ...
```

实现要求：

- `bbox_px` 格式为 `(x1, y1, x2, y2)`。
- bbox 必须 clip 到图像边界内。
- `pad_px` 默认 12。
- 先用矩形 mask，之后用 elliptical kernel 做 dilation。
- 第一版用 bbox mask，不要阻塞在 SAM hand segmentation 上。
- `inpaint_opencv_telea` 调用 `cv2.inpaint(image_bgr, mask_u8, radius, cv2.INPAINT_TELEA)`。
- `compute_boundary_delta` 只比较 mask 边缘的一圈 ring，不比较 mask 内部，返回 `[0, 255]` 范围内的平均绝对差异。

完成后运行：

```bash
cd /Users/huhu/Documents/Code/bishe/vrb/vrbreproduction
pytest tests/test_hand_inpaint_utils.py -v
```

预期：通过。

## 五、任务 2：新增 LaMa / Stable Diffusion 后端包装

**文件：**

```text
Modify: /Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/src/vrbreproduction/hand_inpaint_utils.py
Test: /Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/tests/test_hand_inpaint_utils.py
```

新增配置和统一入口：

```python
@dataclass(frozen=True)
class InpaintConfig:
    backend: str = "opencv_telea"
    mask_pad_px: int = 12
    opencv_radius: float = 3.0
    lama_command: Optional[str] = None
    sd_command: Optional[str] = None
    max_mask_area_ratio: float = 0.35
    boundary_delta_max: float = 35.0

def inpaint_image_with_backend(image_bgr, mask_u8, config: InpaintConfig, work_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    ...
```

支持 backend：

```text
opencv_telea
lama_cli
sd_cli
none
```

要求：

- `none` 返回原图，metadata 里 `inpaint_applied=False`。
- `opencv_telea` 必须可用，不依赖额外模型。
- `lama_cli` 和 `sd_cli` 只做通用命令行包装，不在项目里硬编码安装路径。
- 命令模板支持占位符：

```text
{input}
{mask}
{output}
{work_dir}
```

LaMa 命令示例：

```bash
python /path/to/lama/bin/predict.py model.path=/path/to/big-lama indir={work_dir}/lama_in outdir={work_dir}/lama_out
```

SD 命令示例：

```bash
python scripts/run_sd_inpaint.py --input {input} --mask {mask} --output {output} --prompt "clean kitchen surface, high resolution, natural lighting"
```

测试要求：

- `backend="none"` 返回原图。
- `backend="opencv_telea"` 保持 shape 不变，masked pixels 发生变化。
- unknown backend 抛 `ValueError`。
- `lama_cli` 但没传 command 时抛 `ValueError`。

运行：

```bash
cd /Users/huhu/Documents/Code/bishe/vrb/vrbreproduction
pytest tests/test_hand_inpaint_utils.py -v
```

## 六、任务 3：复制 E18 创建 E19

**文件：**

```text
Create: /Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/src/vrbreproduction/e19_e18_hand_inpaint_reference.py
Do not modify: /Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/src/vrbreproduction/e18_hybrid_min_iou_reference.py
```

从这个文件复制：

```text
src/vrbreproduction/e18_hybrid_min_iou_reference.py
```

替换标识：

```text
E18Config -> E19Config
Problem3CachedRunnerE18 -> Problem3CachedRunnerE19
run_e18_experiment -> run_e19_experiment
E18 -> E19
e18 -> e19_e18_hand_inpaint_reference
```

设置输出目录：

```python
output_root: Path = VRBREPRODUCTION_ROOT / "outputs" / "e19_e18_hand_inpaint_reference"
```

设置 `EXPERIMENT`：

```python
EXPERIMENT = {
    "experiment_id": "E19",
    "name": "E19_e18_hand_inpaint_reference",
    "contact_strategy": "first_new_contact_episode_first_timestep",
    "reference_strategy": "e18_hybrid_active_hand_invisible_then_min_iou_then_hand_inpaint",
    "reference_fallback": "min_hand_object_iou_pre_contact",
    "description": (
        "E18 hybrid reference baseline plus hand-mask-guided inpainting on visible-hand reference frames; "
        "geometry and homography are kept on original frames for comparability."
    ),
}
```

新增 config 字段：

```python
inpaint_backend: str = os.environ.get("E19_INPAINT_BACKEND", "opencv_telea")
inpaint_mask_pad_px: int = int(os.environ.get("E19_INPAINT_MASK_PAD_PX", "12"))
inpaint_opencv_radius: float = float(os.environ.get("E19_INPAINT_OPENCV_RADIUS", "3.0"))
inpaint_only_if_hand_visible: bool = os.environ.get("E19_INPAINT_ONLY_IF_HAND_VISIBLE", "1") != "0"
inpaint_fail_policy: str = os.environ.get("E19_INPAINT_FAIL_POLICY", "keep_original")
inpaint_lama_command: Optional[str] = os.environ.get("E19_LAMA_COMMAND")
inpaint_sd_command: Optional[str] = os.environ.get("E19_SD_COMMAND")
inpaint_max_mask_area_ratio: float = float(os.environ.get("E19_INPAINT_MAX_MASK_AREA_RATIO", "0.35"))
inpaint_boundary_delta_max: float = float(os.environ.get("E19_INPAINT_BOUNDARY_DELTA_MAX", "35.0"))
```

合法 `inpaint_fail_policy`：

```text
keep_original
discard
```

默认使用 `keep_original`，这样 smoke test 不会因为 inpaint 失败导致 pipeline 大量掉样本。

新增 import：

```python
from .hand_inpaint_utils import (
    InpaintConfig,
    bbox_to_mask,
    combine_hand_masks,
    compute_boundary_delta,
    compute_mask_area_ratio,
    inpaint_image_with_backend,
)
```

## 七、任务 4：在 Cell3 里生成 hand mask

**文件：**

```text
Modify: /Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/src/vrbreproduction/e19_e18_hand_inpaint_reference.py
```

在 `Problem3CachedRunnerE19` 中新增：

```python
def build_reference_hand_mask(self, ref_idx: int, active_hand: str, ref_img_shape, pad_px: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    ...
```

行为：

- 使用已有 `hand_bbox_and_meta(self.detections[ref_idx], active_hand, ref_img_shape, score_threshold=0.5)`。
- 如果 active hand 可见，用它的 bbox 生成 mask。
- 如果另一只手也可见且 score >= 0.5，也纳入 mask，但 metadata 里单独记录。
- 返回 mask 和 metadata：

```python
{
    "active_hand_masked": bool,
    "any_hand_masked": bool,
    "masked_hand_count": int,
    "active_hand_bbox_px_for_inpaint": list | None,
    "all_hand_bboxes_px_for_inpaint": list,
    "inpaint_mask_area_ratio": float,
}
```

第一版使用 bbox mask。不要为了 SAM 阻塞。

在 `run_cell3` 中，`ref_img = self.load_bgr(ref_idx)` 之后调用：

```python
mask_u8, mask_meta = self.build_reference_hand_mask(
    ref_idx=ref_idx,
    active_hand=active_hand,
    ref_img_shape=ref_img.shape,
    pad_px=int(self.inpaint_config.mask_pad_px),
)
```

这要求 `Problem3CachedRunnerE19.__init__` 接收：

```python
inpaint_config: InpaintConfig
```

不要立刻替换 `ref_img`。`ref_img` 始终代表原始 reference，用于几何、投影和 heatmap 参数计算。另设一个最终绘图底图：

```python
ref_img_visual = ref_img
```

只有在所有投影坐标和 heatmap 参数已经由原始 reference 坐标系算完并缓存后，且 inpainting 通过质量检查，才设置：

```python
ref_img_visual = inpainted_bgr
```

后续绘制 heatmap、contact points、trajectory、VRB-style affordance 时使用 `ref_img_visual` 作为底图。不要先把这些标识画到原图上再去 inpaint。

## 八、任务 5：执行 reference inpainting 并记录质量

**文件：**

```text
Modify: /Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/src/vrbreproduction/e19_e18_hand_inpaint_reference.py
```

在 `run_cell3` 的 `result` 默认字段里新增：

```python
"inpaint_backend": None,
"inpaint_attempted": False,
"inpaint_applied": False,
"inpaint_required": False,
"inpaint_status": "not_attempted",
"inpaint_fail_reason": None,
"inpaint_fail_policy": None,
"inpaint_mask_area_ratio": None,
"inpaint_boundary_delta": None,
"inpaint_active_hand_masked": False,
"inpaint_any_hand_masked": False,
"inpaint_masked_hand_count": 0,
"inpaint_reference_frame_original": None,
"inpaint_reference_frame": None,
"inpaint_mask_path": None,
"inpaint_before_after_path": None,
```

定义是否需要 inpaint：

```python
inpaint_required = bool(result.get("active_hand_visible_at_ref")) or not config.inpaint_only_if_hand_visible
```

第一版只在 active hand 可见时 inpaint，避免修改已经干净的 reference。

执行后端。注意：执行 inpaint 前，Cell3 已经应该完成并缓存以下内容：

```text
mu_transformed
tau_transformed
contact_centroid
H_contact_to_ref
contact_covariances/contact_weights
object_polygon_ref
```

这些内容都在原始 reference 坐标系里。inpainting 后图像尺寸不变，所以这些坐标可以直接用于重绘。

```python
inpainted_bgr, inpaint_meta = inpaint_image_with_backend(
    ref_img,
    mask_u8,
    self.inpaint_config,
    output_dir / "inpaint_work",
)
```

计算质量指标：

```python
mask_area_ratio = compute_mask_area_ratio(mask_u8)
boundary_delta = compute_boundary_delta(ref_img, inpainted_bgr, mask_u8, band_px=5)
```

质量 gate：

- `mask_area_ratio > inpaint_max_mask_area_ratio`：说明造假区域太大。
- `boundary_delta > inpaint_boundary_delta_max`：说明编辑影响了 mask 边界外的内容。
- 需要 inpaint 但 mask 为空：失败。

如果失败且 `inpaint_fail_policy == "discard"`：

```text
discard_reason = "inpaint_quality_gate_failed:<reason>"
```

如果失败且 `keep_original`：

```python
inpaint_status = "failed_kept_original"
inpaint_applied = False
```

如果通过：

```python
ref_img_visual = inpainted_bgr
inpaint_status = "applied"
inpaint_applied = True
```

关键规则：

- 原始 `ref_img` 用于所有几何计算、投影计算和 heatmap 参数计算。
- `ref_img_visual` 只用于最后重新绘制并保存图像。
- 不能对已经画好 heatmap/trajectory 的图做 inpainting。
- 必须先保存坐标/参数，再 inpaint，再重绘。

推荐保存：

```text
reference_frame_original.png
reference_frame_inpainted.png
reference_hand_mask.png
reference_inpaint_before_after.png
```

保存位置：当前 sample 的 diagnostics 或 sample_dir 下。

可以临时把 `ref_img_visual_bgr` 存进 `cell3_result` 供 Cell4 写图使用，但不要写进 JSON/CSV。Cell4 中生成 heatmap 时，仍使用 `mu_transformed`、`contact_weights`、`contact_covariances` 和 `H_contact_to_ref`；只是把最终 overlay 的底图从原始 `ref_img_bgr` 替换为 `ref_img_visual_bgr`。

## 九、任务 6：新增 candidate CSV 字段

**文件：**

```text
Modify: /Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/src/vrbreproduction/e19_e18_hand_inpaint_reference.py
```

在 `base_candidate_record` 默认字段中加入所有 inpaint 字段。

在 `run_one_candidate` 中，把这些字段从 `cell3` 拷贝到 `record`：

```python
"inpaint_backend",
"inpaint_attempted",
"inpaint_applied",
"inpaint_required",
"inpaint_status",
"inpaint_fail_reason",
"inpaint_fail_policy",
"inpaint_mask_area_ratio",
"inpaint_boundary_delta",
"inpaint_active_hand_masked",
"inpaint_any_hand_masked",
"inpaint_masked_hand_count",
"inpaint_reference_frame_original",
"inpaint_reference_frame",
"inpaint_mask_path",
"inpaint_before_after_path",
```

在 `build_overview` 中新增统计：

```python
"inpaint_required"
"inpaint_attempted"
"inpaint_applied"
"inpaint_failed_kept_original"
"inpaint_discarded"
"active_hand_visible_ref_keep"
"active_hand_visible_ref_inpaint_applied_keep"
"inpaint_required_active_hand_invisible_ref"
"inpaint_applied_active_hand_invisible_ref"
"inpaint_required_fallback_ref"
"inpaint_applied_fallback_ref"
```

所有统计都要处理空 dataframe，避免 smoke test 报错。

同时保留 E18 已有的 `reference_mode`、`fallback_reference_used`、`active_hand_invisible_ref_found`、`fallback_ref_found`、`fallback_ref_keep` 等字段。E19 的 inpaint 统计必须能区分：

```text
reference_mode == "active_hand_invisible_pre_contact"
reference_mode == "min_hand_object_iou_pre_contact_fallback"
```

## 十、任务 7：新增 inpainting contact sheet

**文件：**

```text
Modify: /Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/src/vrbreproduction/e19_e18_hand_inpaint_reference.py
```

新增函数：

```python
def save_inpaint_contact_sheet(candidate_df: pd.DataFrame, charts_dir: Path, per_page: int = 24) -> None:
    ...
```

每个 tile 显示：

```text
original reference | mask | inpainted reference | heatmap overlay
subaction index, narration_id, hand, ref_idx, inpaint_status, boundary_delta
```

使用字段：

```text
inpaint_reference_frame_original
inpaint_mask_path
inpaint_reference_frame
label_heatmap_overlay
```

输出：

```text
outputs/e19_e18_hand_inpaint_reference/charts/e19_inpaint_contact_sheet_page_001.png
```

在已有 success contact sheet 之后调用，过滤：

```python
candidate_df[candidate_df["inpaint_attempted"] == True]
```

## 十一、任务 8：更新 Markdown summary

**文件：**

```text
Modify: /Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/src/vrbreproduction/e19_e18_hand_inpaint_reference.py
```

在 `write_markdown_report` 中新增：

```markdown
## Hand Inpainting

- backend: `...`
- mask pad px: `...`
- fail policy: `...`
- inpaint required / attempted / applied: `...`
- failed kept original: `...`
- discarded by inpaint gate: `...`
- active-hand-visible keep before/after inpaint: `...`
- median mask area ratio: `...`
- median boundary delta: `...`
```

解释文字：

```text
E19 keeps E18's geometry path comparable by computing homography, projected contact points, heatmap parameters, and trajectory coordinates on the original frames. Hand inpainting is applied only after these coordinates are cached, and the final heatmap/trajectory overlays are redrawn on the hand-free reference canvas. Inpainting statistics are reported separately for `active_hand_invisible_pre_contact` and `min_hand_object_iou_pre_contact_fallback` reference modes.
```

不要说 inpainting 恢复了真实隐藏表面。只能说：

```text
inpainting produces a plausible hand-free reference
```

## 十二、任务 9：新增 CLI 参数

**文件：**

```text
Modify: /Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/src/vrbreproduction/e19_e18_hand_inpaint_reference.py
```

新增：

```python
parser.add_argument("--inpaint-backend", default=E19Config.inpaint_backend)
parser.add_argument("--inpaint-mask-pad-px", type=int, default=E19Config.inpaint_mask_pad_px)
parser.add_argument("--inpaint-opencv-radius", type=float, default=E19Config.inpaint_opencv_radius)
parser.add_argument("--inpaint-fail-policy", choices=["keep_original", "discard"], default=E19Config.inpaint_fail_policy)
parser.add_argument("--inpaint-lama-command", default=E19Config.inpaint_lama_command)
parser.add_argument("--inpaint-sd-command", default=E19Config.inpaint_sd_command)
parser.add_argument("--inpaint-max-mask-area-ratio", type=float, default=E19Config.inpaint_max_mask_area_ratio)
parser.add_argument("--inpaint-boundary-delta-max", type=float, default=E19Config.inpaint_boundary_delta_max)
```

传入 `E19Config`。

## 十三、任务 10：无额外依赖 smoke test

先用 OpenCV Telea：

```bash
cd /Users/huhu/Documents/Code/bishe/vrb/vrbreproduction
python -m vrbreproduction.e19_e18_hand_inpaint_reference \
  --num-subactions 10 \
  --output-root outputs/e19_e18_hand_inpaint_reference_smoke10 \
  --inpaint-backend opencv_telea \
  --inpaint-mask-pad-px 12 \
  --inpaint-fail-policy keep_original
```

预期：

- 命令完成。
- 生成 `outputs/e19_e18_hand_inpaint_reference_smoke10/summary.md`。
- `candidate_diagnostics.csv` 包含 inpaint 字段。
- 如果有样本需要 inpaint，生成 `charts/e19_inpaint_contact_sheet_page_001.png`。
- 因为默认 `keep_original`，smoke 结果不应明显差于 E18。

检查：

```bash
python - <<'PY'
import pandas as pd
root = "outputs/e19_e18_hand_inpaint_reference_smoke10"
df = pd.read_csv(f"{root}/candidate_diagnostics.csv")
cols = ["status", "active_hand_visible_at_ref", "inpaint_required", "inpaint_attempted", "inpaint_applied", "inpaint_status", "inpaint_mask_area_ratio", "inpaint_boundary_delta"]
print(df[cols].head(20).to_string())
print(df["inpaint_status"].value_counts(dropna=False))
PY
```

预期：

- 字段存在。
- active hand 可见的 reference 有 `inpaint_required=True`。
- 如果 smoke 范围内有 visible-hand reference，则应有 `inpaint_attempted=True`。

## 十四、任务 11：人工检查 smoke 输出

打开：

```text
/Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/outputs/e19_e18_hand_inpaint_reference_smoke10/charts/e19_inpaint_contact_sheet_page_001.png
```

检查标准：

- mask 是否覆盖手和手边缘/运动模糊。
- mask 是否过度覆盖接触物体。
- inpaint 区域是否是明显糊块、黑块、白块。
- heatmap overlay 是否仍然落在目标物体合理区域。

如果漏掉手指，增大 `--inpaint-mask-pad-px`。如果物体被破坏，减小 pad，或后续引入 SAM hand mask。

## 十五、任务 12：跑完整 OpenCV baseline

```bash
cd /Users/huhu/Documents/Code/bishe/vrb/vrbreproduction
python -m vrbreproduction.e19_e18_hand_inpaint_reference \
  --num-subactions 100 \
  --output-root outputs/e19_e18_hand_inpaint_reference \
  --inpaint-backend opencv_telea \
  --inpaint-mask-pad-px 12 \
  --inpaint-fail-policy keep_original
```

和 E18 对比：

```bash
python - <<'PY'
import pandas as pd
for root in ["outputs/e18", "outputs/e19_e18_hand_inpaint_reference"]:
    o = pd.read_csv(f"{root}/experiment_overview.csv")
    print(root)
    print(o.T.to_string())
PY
```

预期：

- `subactions_success` 应接近 E18。
- 如果明显下降，说明实现误改了几何路径或 discard policy。

## 十六、任务 13：可选 LaMa 后端

前置条件：在 repo 外安装 LaMa，或指向已有本地安装。不要把 LaMa vendor 到当前代码库。

命令形式：

```bash
export E19_LAMA_COMMAND='python /ABS/PATH/TO/lama/bin/predict.py model.path=/ABS/PATH/TO/big-lama indir={work_dir}/lama_in outdir={work_dir}/lama_out'
```

运行：

```bash
cd /Users/huhu/Documents/Code/bishe/vrb/vrbreproduction
python -m vrbreproduction.e19_e18_hand_inpaint_reference \
  --num-subactions 30 \
  --output-root outputs/e19_e18_hand_inpaint_reference_lama30 \
  --inpaint-backend lama_cli \
  --inpaint-mask-pad-px 12 \
  --inpaint-fail-policy keep_original \
  --inpaint-lama-command "$E19_LAMA_COMMAND"
```

预期：

- 输出结构和 OpenCV baseline 一致。
- 大 mask 场景下，LaMa contact sheet 应比 OpenCV 更自然。

如果 LaMa 输出文件名和 wrapper 假设不一致，修改 wrapper：读取 `{work_dir}/lama_out` 中最新的 PNG/JPG。

## 十七、任务 14：可选 Stable Diffusion inpainting

只在 OpenCV/LaMa 跑通后尝试。

创建本地 wrapper：

```text
scripts/run_sd_inpaint.py
```

支持参数：

```text
--input
--mask
--output
--prompt
--negative-prompt
--strength
--guidance-scale
```

推荐 prompt：

```text
clean kitchen surface, realistic object surface, natural lighting, no hand, no fingers, high resolution
```

推荐 negative prompt：

```text
hand, fingers, arm, glove, blur, artifact, distorted object, extra object
```

运行示例：

```bash
cd /Users/huhu/Documents/Code/bishe/vrb/vrbreproduction
python -m vrbreproduction.e19_e18_hand_inpaint_reference \
  --num-subactions 30 \
  --output-root outputs/e19_e18_hand_inpaint_reference_sd30 \
  --inpaint-backend sd_cli \
  --inpaint-mask-pad-px 12 \
  --inpaint-fail-policy keep_original \
  --inpaint-sd-command 'python scripts/run_sd_inpaint.py --input {input} --mask {mask} --output {output} --prompt "clean kitchen surface, realistic object surface, natural lighting, no hand, no fingers, high resolution" --negative-prompt "hand, fingers, arm, glove, blur, artifact, distorted object, extra object" --strength 0.75 --guidance-scale 7.5'
```

只做定性评估。SD 可能更好看，但更容易生成假的物体/contact 细节。

## 十八、任务 15：统计失败率/可用率

**文件：**

```text
Create: /Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/scripts/summarize_hand_inpaint_quality.py
```

脚本读取 `candidate_diagnostics.csv`，输出：

```text
total candidates
keep candidates
active hand visible at reference
inpaint required
inpaint attempted
inpaint applied
failed kept original
discarded by inpaint gate
median mask area ratio
median boundary delta
manual review needed count
```

需要人工复查的条件：

```python
inpaint_attempted and (
    inpaint_boundary_delta > 25
    or inpaint_mask_area_ratio > 0.20
    or inpaint_status != "applied"
)
```

运行：

```bash
cd /Users/huhu/Documents/Code/bishe/vrb/vrbreproduction
python scripts/summarize_hand_inpaint_quality.py outputs/e19_e18_hand_inpaint_reference/candidate_diagnostics.csv
```

这组统计应作为本项目自己的去手可用率。不要从 Seeing the Unseen 里推断，因为那篇论文没有报告 inpainting 丢弃比例。

## 十九、任务 16：写实验记录

**文件：**

```text
Create: /Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/docs/learning_note/实验记录/实验记录e19_e18_hand_inpaint_reference.md
```

内容结构：

```markdown
# E19 hand-inpaint reference

## Motivation

E18 is the current hybrid reference baseline: it first searches for `active_hand_invisible_pre_contact`, and falls back to `min_hand_object_iou_pre_contact_fallback` when needed. Some fallback references can still contain visible hands. Seeing the Unseen suggests a practical data-construction pattern: detect/segment an entity, inpaint it, filter failures, and use the cleaned image downstream.

## Method

E19 keeps E18's contact/reference selection and geometry unchanged. It first computes and caches projected contact points, heatmap parameters, and trajectory coordinates on the original reference frame. If the active hand is visible, it then inpaints the reference frame and redraws the already-computed labels on the hand-free reference canvas. Results are reported separately for `active_hand_invisible_pre_contact` and `min_hand_object_iou_pre_contact_fallback`.

## Backend

OpenCV Telea smoke/full baseline; LaMa optional; Stable Diffusion optional qualitative comparison.

## Results

Paste overview table, inpaint quality table, and contact sheet paths.

## Limitations

The inpainted region is plausible, not true. Do not treat generated object/contact details as ground truth. If the hand covers the true contact surface, this may improve appearance but damage semantic/geometric fidelity.
```

## 二十、验收标准

实现完成标准：

- `pytest tests/test_hand_inpaint_utils.py -v` 通过。
- smoke 命令完成：

```bash
python -m vrbreproduction.e19_e18_hand_inpaint_reference --num-subactions 10 --output-root outputs/e19_e18_hand_inpaint_reference_smoke10 --inpaint-backend opencv_telea
```

- smoke 输出包含 `summary.md`、`candidate_diagnostics.csv` 和 inpaint 字段。
- 100 subaction OpenCV baseline 完成。
- 默认 `keep_original` 下，`subactions_success` 应接近 E18。
- 至少生成一张 inpaint contact sheet 用于人工检查。
- 实验记录明确说明：inpainting 生成的是 plausible 内容，不是真实隐藏内容。

## 二十一、推荐论文/报告表述

可以这样写：

```text
E18 is the current hybrid reference baseline, using active-hand-invisible pre-contact references when available and min hand-object IoU pre-contact fallback otherwise. Inspired by Seeing the Unseen's object-removal data curation pipeline, E19 adds hand-mask-guided inpainting on the selected E18 reference frame. To keep geometry comparable, homography and label projection are computed on the original frames, while inpainting is used only for the final reference canvas and visual label export.
```

不要写：

```text
inpainting recovers the true hidden surface
```

应该写：

```text
inpainting produces a plausible hand-free reference
```

## 二十二、还需要补充阅读什么

代码执行不需要额外论文，可以直接按计划开始。

如果要正式写论文/报告并引用 LaMa，需要补读并加入文献目录：

```text
Resolution-Robust Large Mask Inpainting with Fourier Convolutions
https://github.com/advimman/lama
```

如果 bbox mask 经常破坏目标物体，再考虑补读/引入 hand segmentation 或 SAM-based hand mask 方法。
