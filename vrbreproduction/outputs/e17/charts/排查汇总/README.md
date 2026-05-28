# E17 排查汇总

内容：
- `e17_detected_objects_page_*.png`: contact frame 上的 object detection bbox，绿色为 active object。
- `e17_detected_hands_page_*.png`: contact frame 上的 hand detection bbox，蓝色为 active hand。
- `e17_contact_frame_contact_points_page_*.png`: contact frame 上的 active hand、active object、raw/contact points 和 5 个 GMM means。
- `e17_reference_frames_page_*.png`: E17 找到的 reference frame；没有 ref 的样本显示 no reference。
- `e17_debug_funnel_table.png/csv/md`: E17 漏斗表。

图例：
- object 图：绿色 active object，黄色其它 object。
- hand 图：蓝色 active hand，紫色其它 hand。
- contact 图：黄色小点为 filtered contact points，红色大点为 Cell2 GMM means，粉色小点为 raw bbox-intersection hand-edge points。
