# 数据处理小批量测试10000帧 - subaction 入口统计

- video_id: `P01_109`
- frame limit: `1..500`
- subactions: `7`
- strict reference inside subaction: `True`
- full pipeline usable subactions: `0`

## Stage Summary

| stage | input | kept | discarded |
|---|---:|---:|---:|
| subaction_contact | 7 | 7 | 0 |
| cell2_contact_gmm | 7 | 7 | 0 |
| cell3_homography_geometry | 7 | 0 | 7 |
| cell4_label_heatmap | 0 | 0 | 0 |

## Reason Counts

### subaction_contact
- None

### cell2_contact_gmm
- None

### cell3_homography_geometry
- `no_strict_humanless_frame`: 7

### cell4_label_heatmap
- None

## Usable Subactions

| narration_id | narration | frames | contact_frame | hand | contact_points | ref_idx |
|---|---|---:|---:|---|---:|---:|
| None |  |  |  |  |  |  |

## Discarded Subactions

| narration_id | narration | frames | contact_frame | hand | failed_stage | reason |
|---|---|---:|---:|---|---|---|
| P01_109_0 | grab rucksack | 79-126 | 79 | left | cell3_homography_geometry | no_strict_humanless_frame |
| P01_109_1 | open rucksack | 153-226 | 153 | right | cell3_homography_geometry | no_strict_humanless_frame |
| P01_109_2 | pick up eggs | 252-319 | 252 | left | cell3_homography_geometry | no_strict_humanless_frame |
| P01_109_3 | put down eggs | 319-374 | 319 | left | cell3_homography_geometry | no_strict_humanless_frame |
| P01_109_4 | pick up onion and potato | 424-475 | 424 | left | cell3_homography_geometry | no_strict_humanless_frame |
| P01_109_5 | put down onion and potato | 480-523 | 480 | left | cell3_homography_geometry | no_strict_humanless_frame |
| P01_109_8 | pick up potato | 500-622 | 500 | left | cell3_homography_geometry | no_strict_humanless_frame |