# 数据处理小批量测试10000帧 - 前 500 帧 frame-level baseline 输出

- notebook: `数据处理小批量测试10000帧.ipynb`
- scan limit: `500`
- full pipeline frames: `11`

## Stage Summary

| cell | input | kept | discarded |
|---|---:|---:|---:|
| Cell 1-test | 500 | 375 | 125 |
| Cell 2 | 375 | 367 | 8 |
| Cell 3 | 367 | 11 | 356 |
| Cell 4 | 11 | 11 | 0 |

## Kept Frames

| frame_0_based | frame_1_based | hand | ref_frame_0_based | contact_points | output_dir |
|---:|---:|---|---:|---:|---|
| 72 | 73 | left | 67 | 71 | `outputs/数据处理小批量测试10000帧/frame_level_500/frame_0072_left` |
| 73 | 74 | left | 67 | 76 | `outputs/数据处理小批量测试10000帧/frame_level_500/frame_0073_left` |
| 74 | 75 | left | 67 | 119 | `outputs/数据处理小批量测试10000帧/frame_level_500/frame_0074_left` |
| 75 | 76 | left | 67 | 127 | `outputs/数据处理小批量测试10000帧/frame_level_500/frame_0075_left` |
| 76 | 77 | left | 67 | 86 | `outputs/数据处理小批量测试10000帧/frame_level_500/frame_0076_left` |
| 77 | 78 | left | 67 | 107 | `outputs/数据处理小批量测试10000帧/frame_level_500/frame_0077_left` |
| 78 | 79 | left | 67 | 109 | `outputs/数据处理小批量测试10000帧/frame_level_500/frame_0078_left` |
| 79 | 80 | left | 67 | 42 | `outputs/数据处理小批量测试10000帧/frame_level_500/frame_0079_left` |
| 80 | 81 | left | 67 | 90 | `outputs/数据处理小批量测试10000帧/frame_level_500/frame_0080_left` |
| 81 | 82 | left | 67 | 53 | `outputs/数据处理小批量测试10000帧/frame_level_500/frame_0081_left` |
| 82 | 83 | left | 67 | 126 | `outputs/数据处理小批量测试10000帧/frame_level_500/frame_0082_left` |

说明：这是不绑定 subaction 的 frame-level 对照结果；Cell 5 的 subaction 入口结果仍保存在上一层目录。
