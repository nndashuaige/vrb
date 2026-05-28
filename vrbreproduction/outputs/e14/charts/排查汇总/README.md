# E14 排查汇总

本目录只重跑 contact frame 上的检测和 contact point 抽取可视化，不重跑 reference 投影 / homography。

## 节点统计

| metric | count | definition |
| --- | --- | --- |
| 100 subactions | 100 | E14 configured first 100 P01_109 subactions |
| 有 new-contact candidate | 81 | rows with a selected first new-contact t0 candidate |
| Cell2 contact/GMM pass | 59 | candidate reached/passed Cell2 contact point extraction + GMM |
| reference found | 23 | candidate has non-null ref_idx |
| homography available | 21 | candidate has H_contact_to_ref available |
| final keep | 20 | status=keep / Cell4 heatmap pass |

## 输出图

- `e14_object_detection_contact_sheet_page_*.png`: contact frame 物体检测框；黄色星号为 active object。
- `e14_hand_detection_contact_sheet_page_*.png`: contact frame 手检测框；绿色星号为 active hand。
- `e14_contact_points_contact_sheet_page_*.png`: contact frame 上的 active hand/object 和 contact points；红点为最终 contact points。
- `e14_node_counts_table.png`: 用户指定节点统计表。
- `e14_contact_frame_node_manifest.csv`: 每个 contact-frame candidate 的节点可视化元数据。

- `e14_reference_frame_contact_sheet_page_*.png`: 已找到的 reference frame 拼图。
- `e14_reference_frame_manifest.csv`: 这些 reference frame 的索引和路径。
