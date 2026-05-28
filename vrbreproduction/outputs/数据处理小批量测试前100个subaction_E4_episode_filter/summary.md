# 数据处理小批量测试前100个subaction E4 episode filter - 诊断实验报告

- video_id: `P01_109`
- subactions: first `100` annotations of this video
- output root: `Outputs/数据处理小批量测试前100个subaction_E4_episode_filter`
- reference window before subaction start: `120` frames
- all-contact deep-run cap per subaction: `5` time-uniform candidates
- E4 release/no-hand gap threshold: `8` frames
- E4 max continuation gap: `180` frames
- annotated frame intervals total: `15073` frames
- unique covered frames after overlap removal: `14330` frames

## 怎么读这个报告

- `subactions_success` 表示 100 个 subaction 里有多少个至少生成了 1 个完整 heatmap/trajectory 样本。
- `heatmap_pass` 表示候选帧级别最终成功样本数。
- E2/E3 的 all-contact 候选采用时间均匀抽样深跑，避免大量相邻帧重复做昂贵的 homography。
- `ref_found` 低，说明主要卡在 human-less reference frame。
- `homography_available / geometry_pass` 低，说明找到 reference 后投影或几何一致性不过。
- E4 的 `episode_decision` 是进入原 pipeline 前的 noun/release gate；`episode_final_label` 会把 new episode 但 pipeline 没跑通的 subaction 标成 `discarded_by_existing_pipeline`。

## 结论速览

- E1 比 E0 有提升：reference 被 subaction 起点限制是主要问题之一。
- E2 比 E1 有提升：只取 first contact 会漏掉 subaction 内更可用的帧。
- E3 比 E2 还有提升：120 帧 reference 窗口可能偏窄。
- E4 在 E2 候选前加入 episode gate：raw candidates 24687 -> filtered candidates 3459，最终成功 37 个样本、16 个 subaction。

## 实验总览

| experiment_id | subactions_success | raw_candidate_total | episode_filtered_candidate_total | deep_run_candidate_total | cell2_pass | ref_found | homography_available | geometry_pass | heatmap_pass | main_failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | 1 | 99 | 99 | 99 | 85 | 1 | 1 | 1 | 1 | no_strict_humanless_frame |
| E1 | 12 | 99 | 99 | 99 | 85 | 23 | 14 | 12 | 12 | no_strict_humanless_frame |
| E2 | 18 | 24687 | 24687 | 493 | 396 | 124 | 61 | 44 | 44 | no_strict_humanless_frame |
| E3 | 48 | 24687 | 24687 | 493 | 396 | 396 | 254 | 154 | 154 | projected_object_polygon_invalid |
| E4 | 16 | 24687 | 3459 | 243 | 182 | 84 | 52 | 37 | 37 | no_strict_humanless_frame |

## 主方案 E2 的 subaction 级结果

| subaction_index | narration_id | narration | frames_1_based | candidates | cell2_pass | ref_found | homography_available | geometry_pass | heatmap_pass | best_frame_0_based | best_ref_idx | main_failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | P01_109_0 | grab rucksack | 79-126 | 5 | 4 | 4 | 1 | 1 | 1 | 78.0 | 67.0 | pairwise_homography_low_quality_f98_to_f97 |
| 1 | P01_109_1 | open rucksack | 153-226 | 5 | 5 | 5 | 0 | 0 | 0 |  |  | pairwise_homography_low_quality_f161_to_f160 |
| 2 | P01_109_2 | pick up eggs | 252-319 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 3 | P01_109_3 | put down eggs | 319-374 | 5 | 5 | 2 | 1 | 0 | 0 |  |  | no_strict_humanless_frame |
| 4 | P01_109_4 | pick up onion and potato | 424-475 | 5 | 5 | 5 | 0 | 0 | 0 |  |  | pairwise_homography_low_quality_f430_to_f429 |
| 5 | P01_109_5 | put down onion and potato | 480-523 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 6 | P01_109_8 | pick up potato | 500-622 | 5 | 3 | 2 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 7 | P01_109_6 | pick up potatoes | 543-579 | 5 | 4 | 4 | 0 | 0 | 0 |  |  | pairwise_homography_low_quality_f545_to_f544 |
| 8 | P01_109_7 | put down potatoes | 550-734 | 5 | 4 | 4 | 1 | 0 | 0 |  |  | contact_points_lt5_after_object_boundary_filter |
| 9 | P01_109_9 | put down potato | 625-752 | 5 | 5 | 5 | 0 | 0 | 0 |  |  | pairwise_homography_low_quality_f630_to_f629 |
| 10 | P01_109_10 | pick up beers | 738-845 | 5 | 4 | 4 | 0 | 0 | 0 |  |  | pairwise_homography_low_quality_f811_to_f810 |
| 11 | P01_109_11 | open freezer | 876-1022 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 12 | P01_109_12 | open drawer | 1045-1123 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 13 | P01_109_13 | put beer into drawer | 1139-1353 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 14 | P01_109_14 | put beer into drawer | 1230-1283 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 15 | P01_109_15 | put beer into drawer | 1298-1335 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 16 | P01_109_16 | close drawer | 1345-1401 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 17 | P01_109_17 | close freezer | 1419-1464 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 18 | P01_109_18 | pick up rucksack | 1485-1527 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 19 | P01_109_19 | put down rucksack | 1657-1698 | 5 | 3 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 20 | P01_109_20 | close door | 1812-1946 | 5 | 3 | 3 | 3 | 2 | 2 | 1863.0 | 1798.0 | missing_hand_or_object_bbox |
| 21 | P01_109_21 | pick up cup | 2084-2120 | 5 | 1 | 1 | 1 | 1 | 1 | 2088.0 | 2076.0 | missing_hand_or_object_bbox |
| 22 | P01_109_22 | put down cup | 2132-2193 | 5 | 3 | 3 | 3 | 3 | 3 | 2142.0 | 2140.0 | missing_hand_or_object_bbox |
| 23 | P01_109_23 | pick up spoon | 2206-2236 | 5 | 4 | 4 | 4 | 0 | 0 |  |  | trajectory_out_of_bounds_t+0 |
| 24 | P01_109_24 | open drawer | 2269-2332 | 5 | 5 | 5 | 1 | 1 | 1 | 2268.0 | 2257.0 | pairwise_homography_failed_f2305_to_f2304 |
| 25 | P01_109_25 | put spoon into drawer | 2336-2378 | 5 | 5 | 5 | 0 | 0 | 0 |  |  | pairwise_homography_failed_f2305_to_f2304 |
| 26 | P01_109_26 | close drawer | 2377-2410 | 1 | 0 | 0 | 0 | 0 | 0 |  |  | no_smoothed_contact_in_subaction |
| 27 | P01_109_27 | pour hand wash | 2466-2505 | 5 | 4 | 4 | 4 | 4 | 4 | 2484.0 | 2434.0 | missing_hand_or_object_bbox |
| 28 | P01_109_28 | wash hands | 2478-2540 | 5 | 2 | 2 | 2 | 2 | 2 | 2479.0 | 2434.0 | missing_hand_or_object_bbox |
| 29 | P01_109_29 | turn on tap | 2545-2575 | 5 | 4 | 4 | 4 | 3 | 3 | 2544.0 | 2434.0 | contact_points_lt5_after_object_boundary_filter |
| 30 | P01_109_30 | rinse hands | 2579-2963 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 31 | P01_109_31 | turn off tap | 2967-2989 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 32 | P01_109_32 | shake hands | 3001-3072 | 5 | 1 | 0 | 0 | 0 | 0 |  |  | missing_hand_or_object_bbox |
| 33 | P01_109_33 | pick up cloth | 3084-3140 | 5 | 3 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 34 | P01_109_34 | dry hands | 3094-3349 | 5 | 4 | 3 | 3 | 2 | 2 | 3190.0 | 3165.0 | no_strict_humanless_frame |
| 35 | P01_109_35 | put down cloth | 3349-3372 | 5 | 5 | 5 | 3 | 3 | 3 | 3356.0 | 3330.0 | pairwise_homography_low_quality_f3371_to_f3370 |
| 36 | P01_109_36 | pick up potatoes | 3374-3496 | 5 | 5 | 5 | 0 | 0 | 0 |  |  | pairwise_homography_low_quality_f3371_to_f3370 |
| 37 | P01_109_37 | put down potatoes | 3517-3552 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 38 | P01_109_38 | move potatoes | 3640-3706 | 5 | 3 | 3 | 0 | 0 | 0 |  |  | pairwise_homography_failed_f3646_to_f3645 |
| 39 | P01_109_39 | open cupboard | 3719-3799 | 5 | 2 | 2 | 0 | 0 | 0 |  |  | missing_hand_or_object_bbox |
| 40 | P01_109_40 | move pans | 3826-4048 | 5 | 3 | 3 | 0 | 0 | 0 |  |  | pairwise_homography_failed_f3961_to_f3960 |
| 41 | P01_109_41 | pick up pan | 4038-4098 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 42 | P01_109_42 | put down pan | 4209-4280 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 43 | P01_109_44 | put down pan | 4269-4353 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 44 | P01_109_43 | pick up pan | 4330-4386 | 5 | 3 | 1 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 45 | P01_109_45 | put down pan | 4379-4432 | 5 | 5 | 5 | 0 | 0 | 0 |  |  | pairwise_homography_failed_f4385_to_f4384 |
| 46 | P01_109_46 | pick up cutting board | 4510-4604 | 5 | 1 | 1 | 0 | 0 | 0 |  |  | missing_hand_or_object_bbox |
| 47 | P01_109_47 | close cupboard | 4621-4664 | 5 | 5 | 3 | 3 | 1 | 1 | 4651.0 | 4636.0 | no_strict_humanless_frame |
| 48 | P01_109_48 | put down cutting board | 4677-4724 | 5 | 5 | 5 | 5 | 5 | 5 | 4689.0 | 4636.0 |  |
| 49 | P01_109_49 | pick up eggs | 4764-4797 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 50 | P01_109_50 | put down eggs | 4794-4827 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 51 | P01_109_51 | move kettle | 4832-4867 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 52 | P01_109_52 | open drawer | 4904-4942 | 5 | 1 | 0 | 0 | 0 | 0 |  |  | contact_points_lt5_after_object_boundary_filter |
| 53 | P01_109_53 | pick up peeler | 4938-4981 | 5 | 2 | 2 | 2 | 2 | 2 | 4980.0 | 4937.0 | contact_points_lt5_after_object_boundary_filter |
| 54 | P01_109_54 | close drawer | 4977-5004 | 5 | 2 | 2 | 2 | 2 | 2 | 4981.0 | 4937.0 | contact_points_lt5_after_object_boundary_filter |
| 55 | P01_109_55 | move cutting board | 5014-5049 | 5 | 3 | 3 | 3 | 1 | 1 | 5022.0 | 4937.0 | missing_hand_or_object_bbox |
| 56 | P01_109_56 | pick up potato | 5053-5122 | 5 | 5 | 5 | 5 | 2 | 2 | 5052.0 | 4937.0 | trajectory_out_of_bounds_t+0 |
| 57 | P01_109_57 | move potato | 5082-5094 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 58 | P01_109_58 | peel potato | 5182-9529 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 59 | P01_109_59 | put down potato | 9494-9582 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 60 | P01_109_60 | put down peeler | 9586-9607 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 61 | P01_109_61 | pick up knife | 9668-9762 | 3 | 0 | 0 | 0 | 0 | 0 |  |  | missing_hand_or_object_bbox |
| 62 | P01_109_62 | grab potato | 9805-9835 | 5 | 5 | 5 | 5 | 4 | 4 | 9834.0 | 9764.0 | transformed_contact_points_off_object |
| 63 | P01_109_63 | cut potato | 9831-10125 | 5 | 5 | 5 | 5 | 5 | 5 | 9903.0 | 9764.0 |  |
| 64 | P01_109_64 | pick up potato bit | 10119-10183 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 65 | P01_109_65 | put down potato bit | 10148-10191 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 66 | P01_109_66 | move cutting board | 10196-10232 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 67 | P01_109_67 | put down knife | 10237-10267 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 68 | P01_109_68 | pick up knife | 10264-10300 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 69 | P01_109_69 | slice potato | 10297-11900 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 70 | P01_109_70 | put down knife | 11914-11935 | 5 | 3 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 71 | P01_109_71 | put down potato slices | 11963-11994 | 5 | 3 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 72 | P01_109_72 | pick up potato slices | 12007-12071 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 73 | P01_109_73 | put down potato slices | 12070-12186 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 74 | P01_109_74 | pick up potato slices | 12181-12263 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 75 | P01_109_75 | put down potato slices | 12267-12332 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 76 | P01_109_76 | pick up potato slice | 12325-12391 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 77 | P01_109_78 | pick up potato slice | 12336-12390 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 78 | P01_109_79 | put down potato slice | 12389-12411 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 79 | P01_109_77 | put down potato slice | 12390-12417 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 80 | P01_109_80 | pick up potato slices | 12416-12455 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 81 | P01_109_81 | put down potato slices | 12458-12480 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 82 | P01_109_82 | pick up potato slice | 12484-12562 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 83 | P01_109_83 | put down potato slice | 12562-12585 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 84 | P01_109_84 | pick up potato slice | 12634-12667 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 85 | P01_109_85 | put down potato slice | 12670-12692 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 86 | P01_109_86 | pick up potato slice | 12723-12753 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 87 | P01_109_87 | put down potato slice | 12758-12786 | 5 | 1 | 0 | 0 | 0 | 0 |  |  | missing_hand_or_object_bbox |
| 88 | P01_109_88 | pick up knife | 12876-12926 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 89 | P01_109_89 | cut potato slice | 12915-13036 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 90 | P01_109_90 | put down knife | 13025-13055 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 91 | P01_109_91 | put down potato slice | 13065-13090 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 92 | P01_109_92 | pick up potato | 13096-13121 | 5 | 1 | 0 | 0 | 0 | 0 |  |  | missing_hand_or_object_bbox |
| 93 | P01_109_93 | pick up peeler | 13126-13155 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 94 | P01_109_94 | peel potato | 13152-15461 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 95 | P01_109_95 | put down potato | 15463-15503 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 96 | P01_109_96 | pick up peeler blade | 15508-15575 | 5 | 3 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 97 | P01_109_97 | fix peeler blade | 15574-15772 | 5 | 4 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 98 | P01_109_98 | pick up potato | 15772-15799 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |
| 99 | P01_109_99 | peel potato | 15883-16103 | 5 | 5 | 0 | 0 | 0 | 0 |  |  | no_strict_humanless_frame |

## E4 episode 筛选结果

| subaction_index | narration_id | narration | noun | episode_decision | episode_final_label | episode_id | episode_reason | candidates | heatmap_pass | main_failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | P01_109_0 | grab rucksack | rucksack | new_contact_episode | new_contact_episode | P01_109_ep0001 | first_seen_object_noun | 5 | 1 | pairwise_homography_low_quality_f98_to_f97 |
| 1 | P01_109_1 | open rucksack | rucksack | continuation_same_object | continuation_same_object | P01_109_ep0001 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 2 | P01_109_2 | pick up eggs | egg | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0002 | first_seen_object_noun | 5 | 0 | no_strict_humanless_frame |
| 3 | P01_109_3 | put down eggs | egg | continuation_same_object | continuation_same_object | P01_109_ep0002 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 4 | P01_109_4 | pick up onion and potato | onion | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0003 | first_seen_object_noun | 5 | 0 | pairwise_homography_low_quality_f430_to_f429 |
| 5 | P01_109_5 | put down onion and potato | onion | continuation_same_object | continuation_same_object | P01_109_ep0003 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 6 | P01_109_8 | pick up potato | potato | continuation_same_object | continuation_same_object | P01_109_ep0003 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 7 | P01_109_6 | pick up potatoes | potato | continuation_same_object | continuation_same_object | P01_109_ep0003 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 8 | P01_109_7 | put down potatoes | potato | continuation_same_object | continuation_same_object | P01_109_ep0003 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 9 | P01_109_9 | put down potato | potato | continuation_same_object | continuation_same_object | P01_109_ep0003 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 10 | P01_109_10 | pick up beers | beer | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0004 | first_seen_object_noun | 5 | 0 | pairwise_homography_low_quality_f811_to_f810 |
| 11 | P01_109_11 | open freezer | freezer | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0005 | first_seen_object_noun | 5 | 0 | no_strict_humanless_frame |
| 12 | P01_109_12 | open drawer | drawer | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0006 | first_seen_object_noun | 5 | 0 | no_strict_humanless_frame |
| 13 | P01_109_13 | put beer into drawer | beer | continuation_same_object | continuation_same_object | P01_109_ep0006 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 14 | P01_109_14 | put beer into drawer | beer | continuation_same_object | continuation_same_object | P01_109_ep0006 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 15 | P01_109_15 | put beer into drawer | beer | continuation_same_object | continuation_same_object | P01_109_ep0006 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 16 | P01_109_16 | close drawer | drawer | continuation_same_object | continuation_same_object | P01_109_ep0006 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 17 | P01_109_17 | close freezer | freezer | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0007 | release_or_no_hand_gap | 5 | 0 | no_strict_humanless_frame |
| 18 | P01_109_18 | pick up rucksack | rucksack | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0008 | release_or_no_hand_gap | 5 | 0 | no_strict_humanless_frame |
| 19 | P01_109_19 | put down rucksack | rucksack | continuation_same_object | continuation_same_object | P01_109_ep0008 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 20 | P01_109_20 | close door | door | new_contact_episode | new_contact_episode | P01_109_ep0009 | first_seen_object_noun | 5 | 2 | missing_hand_or_object_bbox |
| 21 | P01_109_21 | pick up cup | cup | new_contact_episode | new_contact_episode | P01_109_ep0010 | first_seen_object_noun | 5 | 1 | missing_hand_or_object_bbox |
| 22 | P01_109_22 | put down cup | cup | new_contact_episode | new_contact_episode | P01_109_ep0011 | release_or_no_hand_gap | 5 | 3 | missing_hand_or_object_bbox |
| 23 | P01_109_23 | pick up spoon | spoon | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0012 | first_seen_object_noun | 5 | 0 | trajectory_out_of_bounds_t+0 |
| 24 | P01_109_24 | open drawer | drawer | new_contact_episode | new_contact_episode | P01_109_ep0013 | release_or_no_hand_gap | 5 | 1 | pairwise_homography_failed_f2305_to_f2304 |
| 25 | P01_109_25 | put spoon into drawer | spoon | continuation_same_object | continuation_same_object | P01_109_ep0013 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 26 | P01_109_26 | close drawer | drawer | no_contact_candidate | no_contact_candidate |  | no_smoothed_contact_in_subaction | 1 | 0 | no_contact_candidate |
| 27 | P01_109_27 | pour hand wash | wash:hand | new_contact_episode | new_contact_episode | P01_109_ep0014 | first_seen_object_noun | 5 | 4 | missing_hand_or_object_bbox |
| 28 | P01_109_28 | wash hands | hand | continuation_same_object | continuation_same_object | P01_109_ep0014 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 29 | P01_109_29 | turn on tap | tap | new_contact_episode | new_contact_episode | P01_109_ep0015 | first_seen_object_noun | 5 | 3 | contact_points_lt5_after_object_boundary_filter |
| 30 | P01_109_30 | rinse hands | hand | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0016 | release_or_no_hand_gap | 5 | 0 | no_strict_humanless_frame |
| 31 | P01_109_31 | turn off tap | tap | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0017 | release_or_no_hand_gap | 5 | 0 | no_strict_humanless_frame |
| 32 | P01_109_32 | shake hands | hand | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0018 | release_or_no_hand_gap | 5 | 0 | missing_hand_or_object_bbox |
| 33 | P01_109_33 | pick up cloth | cloth | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0019 | first_seen_object_noun | 5 | 0 | no_strict_humanless_frame |
| 34 | P01_109_34 | dry hands | hand | new_contact_episode | new_contact_episode | P01_109_ep0020 | release_or_no_hand_gap | 5 | 2 | no_strict_humanless_frame |
| 35 | P01_109_35 | put down cloth | cloth | new_contact_episode | new_contact_episode | P01_109_ep0021 | release_or_no_hand_gap | 5 | 3 | pairwise_homography_low_quality_f3371_to_f3370 |
| 36 | P01_109_36 | pick up potatoes | potato | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0022 | release_or_no_hand_gap | 5 | 0 | pairwise_homography_low_quality_f3371_to_f3370 |
| 37 | P01_109_37 | put down potatoes | potato | continuation_same_object | continuation_same_object | P01_109_ep0022 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 38 | P01_109_38 | move potatoes | potato | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0023 | release_or_no_hand_gap | 5 | 0 | pairwise_homography_failed_f3646_to_f3645 |
| 39 | P01_109_39 | open cupboard | cupboard | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0024 | first_seen_object_noun | 5 | 0 | missing_hand_or_object_bbox |
| 40 | P01_109_40 | move pans | pan | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0025 | first_seen_object_noun | 5 | 0 | pairwise_homography_failed_f3961_to_f3960 |
| 41 | P01_109_41 | pick up pan | pan | continuation_same_object | continuation_same_object | P01_109_ep0025 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 42 | P01_109_42 | put down pan | pan | continuation_same_object | continuation_same_object | P01_109_ep0025 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 43 | P01_109_44 | put down pan | pan | continuation_same_object | continuation_same_object | P01_109_ep0025 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 44 | P01_109_43 | pick up pan | pan | continuation_same_object | continuation_same_object | P01_109_ep0025 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 45 | P01_109_45 | put down pan | pan | continuation_same_object | continuation_same_object | P01_109_ep0025 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 46 | P01_109_46 | pick up cutting board | board:cutting | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0026 | first_seen_object_noun | 5 | 0 | missing_hand_or_object_bbox |
| 47 | P01_109_47 | close cupboard | cupboard | new_contact_episode | new_contact_episode | P01_109_ep0027 | release_or_no_hand_gap | 5 | 1 | no_strict_humanless_frame |
| 48 | P01_109_48 | put down cutting board | board:cutting | new_contact_episode | new_contact_episode | P01_109_ep0028 | release_or_no_hand_gap | 5 | 5 |  |
| 49 | P01_109_49 | pick up eggs | egg | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0029 | release_or_no_hand_gap | 5 | 0 | no_strict_humanless_frame |
| 50 | P01_109_50 | put down eggs | egg | continuation_same_object | continuation_same_object | P01_109_ep0029 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 51 | P01_109_51 | move kettle | kettle | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0030 | first_seen_object_noun | 5 | 0 | no_strict_humanless_frame |
| 52 | P01_109_52 | open drawer | drawer | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0031 | release_or_no_hand_gap | 5 | 0 | contact_points_lt5_after_object_boundary_filter |
| 53 | P01_109_53 | pick up peeler | peeler | new_contact_episode | new_contact_episode | P01_109_ep0032 | first_seen_object_noun | 5 | 2 | contact_points_lt5_after_object_boundary_filter |
| 54 | P01_109_54 | close drawer | drawer | new_contact_episode | new_contact_episode | P01_109_ep0033 | release_or_no_hand_gap | 5 | 2 | contact_points_lt5_after_object_boundary_filter |
| 55 | P01_109_55 | move cutting board | board:cutting | new_contact_episode | new_contact_episode | P01_109_ep0034 | release_or_no_hand_gap | 5 | 1 | missing_hand_or_object_bbox |
| 56 | P01_109_56 | pick up potato | potato | new_contact_episode | new_contact_episode | P01_109_ep0035 | release_or_no_hand_gap | 5 | 2 | trajectory_out_of_bounds_t+0 |
| 57 | P01_109_57 | move potato | potato | continuation_same_object | continuation_same_object | P01_109_ep0035 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 58 | P01_109_58 | peel potato | potato | continuation_same_object | continuation_same_object | P01_109_ep0035 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 59 | P01_109_59 | put down potato | potato | continuation_same_object | continuation_same_object | P01_109_ep0035 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 60 | P01_109_60 | put down peeler | peeler | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0036 | large_temporal_gap | 5 | 0 | no_strict_humanless_frame |
| 61 | P01_109_61 | pick up knife | knife | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0037 | first_seen_object_noun | 3 | 0 | missing_hand_or_object_bbox |
| 62 | P01_109_62 | grab potato | potato | new_contact_episode | new_contact_episode | P01_109_ep0038 | release_or_no_hand_gap | 5 | 4 | transformed_contact_points_off_object |
| 63 | P01_109_63 | cut potato | potato | continuation_same_object | continuation_same_object | P01_109_ep0038 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 64 | P01_109_64 | pick up potato bit | bit:potato | continuation_same_object | continuation_same_object | P01_109_ep0038 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 65 | P01_109_65 | put down potato bit | bit:potato | continuation_same_object | continuation_same_object | P01_109_ep0038 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 66 | P01_109_66 | move cutting board | board:cutting | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0039 | release_or_no_hand_gap | 5 | 0 | no_strict_humanless_frame |
| 67 | P01_109_67 | put down knife | knife | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0040 | release_or_no_hand_gap | 5 | 0 | no_strict_humanless_frame |
| 68 | P01_109_68 | pick up knife | knife | continuation_same_object | continuation_same_object | P01_109_ep0040 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 69 | P01_109_69 | slice potato | potato | continuation_same_object | continuation_same_object | P01_109_ep0038 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 70 | P01_109_70 | put down knife | knife | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0041 | large_temporal_gap | 5 | 0 | no_strict_humanless_frame |
| 71 | P01_109_71 | put down potato slices | slice:potato | continuation_same_object | continuation_same_object | P01_109_ep0038 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 72 | P01_109_72 | pick up potato slices | slice:potato | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0042 | release_or_no_hand_gap | 5 | 0 | no_strict_humanless_frame |
| 73 | P01_109_73 | put down potato slices | slice:potato | continuation_same_object | continuation_same_object | P01_109_ep0042 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 74 | P01_109_74 | pick up potato slices | slice:potato | continuation_same_object | continuation_same_object | P01_109_ep0042 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 75 | P01_109_75 | put down potato slices | slice:potato | continuation_same_object | continuation_same_object | P01_109_ep0042 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 76 | P01_109_76 | pick up potato slice | slice:potato | continuation_same_object | continuation_same_object | P01_109_ep0042 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 77 | P01_109_78 | pick up potato slice | slice:potato | continuation_same_object | continuation_same_object | P01_109_ep0042 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 78 | P01_109_79 | put down potato slice | slice:potato | continuation_same_object | continuation_same_object | P01_109_ep0042 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 79 | P01_109_77 | put down potato slice | slice:potato | continuation_same_object | continuation_same_object | P01_109_ep0042 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 80 | P01_109_80 | pick up potato slices | slice:potato | continuation_same_object | continuation_same_object | P01_109_ep0042 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 81 | P01_109_81 | put down potato slices | slice:potato | continuation_same_object | continuation_same_object | P01_109_ep0042 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 82 | P01_109_82 | pick up potato slice | slice:potato | continuation_same_object | continuation_same_object | P01_109_ep0042 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 83 | P01_109_83 | put down potato slice | slice:potato | continuation_same_object | continuation_same_object | P01_109_ep0042 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 84 | P01_109_84 | pick up potato slice | slice:potato | continuation_same_object | continuation_same_object | P01_109_ep0042 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 85 | P01_109_85 | put down potato slice | slice:potato | continuation_same_object | continuation_same_object | P01_109_ep0042 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 86 | P01_109_86 | pick up potato slice | slice:potato | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0043 | release_or_no_hand_gap | 5 | 0 | no_strict_humanless_frame |
| 87 | P01_109_87 | put down potato slice | slice:potato | continuation_same_object | continuation_same_object | P01_109_ep0043 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 88 | P01_109_88 | pick up knife | knife | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0044 | release_or_no_hand_gap | 5 | 0 | no_strict_humanless_frame |
| 89 | P01_109_89 | cut potato slice | slice:potato | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0045 | release_or_no_hand_gap | 5 | 0 | no_strict_humanless_frame |
| 90 | P01_109_90 | put down knife | knife | continuation_same_object | continuation_same_object | P01_109_ep0044 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 91 | P01_109_91 | put down potato slice | slice:potato | continuation_same_object | continuation_same_object | P01_109_ep0045 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 92 | P01_109_92 | pick up potato | potato | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0046 | release_or_no_hand_gap | 5 | 0 | missing_hand_or_object_bbox |
| 93 | P01_109_93 | pick up peeler | peeler | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0047 | release_or_no_hand_gap | 5 | 0 | no_strict_humanless_frame |
| 94 | P01_109_94 | peel potato | potato | continuation_same_object | continuation_same_object | P01_109_ep0046 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 95 | P01_109_95 | put down potato | potato | continuation_same_object | continuation_same_object | P01_109_ep0046 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 96 | P01_109_96 | pick up peeler blade | blade:peeler | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0048 | large_temporal_gap | 5 | 0 | no_strict_humanless_frame |
| 97 | P01_109_97 | fix peeler blade | blade:peeler | continuation_same_object | continuation_same_object | P01_109_ep0048 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |
| 98 | P01_109_98 | pick up potato | potato | new_contact_episode | discarded_by_existing_pipeline | P01_109_ep0049 | large_temporal_gap | 5 | 0 | no_strict_humanless_frame |
| 99 | P01_109_99 | peel potato | potato | continuation_same_object | continuation_same_object | P01_109_ep0049 | same_noun_without_release_gap | 1 | 0 | continuation_same_object |

## 主要失败原因

| experiment_id | top_failures |
| --- | --- |
| E0 | no_strict_humanless_frame: 84, contact_points_lt5_after_object_boundary_filter: 7, missing_hand_or_object_bbox: 7, no_smoothed_contact_in_subaction: 1 |
| E1 | no_strict_humanless_frame: 62, contact_points_lt5_after_object_boundary_filter: 7, missing_hand_or_object_bbox: 7, pairwise_homography_low_quality_f157_to_f156: 1, pairwise_homography_low_quality_f428_to_f427: 1 |
| E2 | no_strict_humanless_frame: 272, missing_hand_or_object_bbox: 51, contact_points_lt5_after_object_boundary_filter: 39, trajectory_out_of_bounds_t+0: 10, pairwise_homography_failed_f2305_to_f2304: 7 |
| E3 | projected_object_polygon_invalid: 58, missing_hand_or_object_bbox: 51, contact_points_lt5_after_object_boundary_filter: 39, pairwise_homography_low_quality_f1002_to_f1001: 21, trajectory_out_of_bounds_t+0: 19 |
| E4 | no_strict_humanless_frame: 98, continuation_same_object: 50, missing_hand_or_object_bbox: 37, contact_points_lt5_after_object_boundary_filter: 20, trajectory_out_of_bounds_t+0: 9 |

## 图表

- pipeline funnel: `charts/pipeline_funnel.png`
- failure reasons: `charts/failure_reasons.png`
- E2 timeline: `charts/timeline_E2.png`
- E4 timeline: `charts/timeline_E4.png`

## 明细文件

- all results JSON: `experiment_results.json`
- overview CSV: `experiment_overview.csv`
- subaction summary CSV: `subaction_summary.csv`
- candidate diagnostics CSV: `candidate_diagnostics.csv`
- successful pipeline outputs: `experiments/<E*>/pipeline_outputs/`
