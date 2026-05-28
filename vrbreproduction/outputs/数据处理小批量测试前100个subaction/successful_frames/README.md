# Successful Frames Summary

- source: `/Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/outputs/数据处理小批量测试前100个subaction/candidate_diagnostics.csv`
- total successful samples: `211`
- output directory: `/Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/outputs/数据处理小批量测试前100个subaction/successful_frames`

## Counts By Experiment

| experiment_id | subactions_total | candidates_total | cell2_pass | ref_found | homography_available | geometry_pass | heatmap_pass | successful_samples | successful_subactions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | 100 | 100 | 85 | 1 | 1 | 1 | 1 | 1 | 1 |
| E1 | 100 | 100 | 85 | 23 | 14 | 12 | 12 | 12 | 12 |
| E2 | 100 | 494 | 396 | 124 | 61 | 44 | 44 | 44 | 18 |
| E3 | 100 | 494 | 396 | 396 | 254 | 154 | 154 | 154 | 48 |


## Files

- `successful_counts_by_experiment.csv`: each experiment's pass counts.
- `successful_subactions.csv`: one row per successful subaction per experiment.
- `successful_frames_all.csv`: one row per successful frame/sample.
- `successful_frames_E0.csv` ... `successful_frames_E3.csv`: per-experiment successful frames.

## How To Read A Successful Frame Row

- `frame_0_based`: frame index used by the pipeline.
- `frame_1_based`: image filename index; frame 2143 means `frame_0000002143.jpg`.
- `ref_idx` / `ref_frame_1_based`: selected human-less reference frame.
- `sample_dir`: directory containing the full visual output for that successful sample.

## E2 Preview

| subaction_index | narration_id | narration | frame_0_based | frame_1_based | hand | ref_idx | sample_dir |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | P01_109_0 | grab rucksack | 78.0 | 79.0 | left | 67.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_00_P01_109_0/frame_000078_left |
| 20 | P01_109_20 | close door | 1863.0 | 1864.0 | right | 1798.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_20_P01_109_20/frame_001863_right |
| 20 | P01_109_20 | close door | 1908.0 | 1909.0 | right | 1895.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_20_P01_109_20/frame_001908_right |
| 21 | P01_109_21 | pick up cup | 2088.0 | 2089.0 | left | 2076.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_21_P01_109_21/frame_002088_left |
| 22 | P01_109_22 | put down cup | 2142.0 | 2143.0 | right | 2140.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_22_P01_109_22/frame_002142_right |
| 22 | P01_109_22 | put down cup | 2148.0 | 2149.0 | right | 2140.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_22_P01_109_22/frame_002148_right |
| 22 | P01_109_22 | put down cup | 2154.0 | 2155.0 | right | 2140.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_22_P01_109_22/frame_002154_right |
| 24 | P01_109_24 | open drawer | 2268.0 | 2269.0 | right | 2257.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_24_P01_109_24/frame_002268_right |
| 27 | P01_109_27 | pour hand wash | 2468.0 | 2469.0 | right | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_27_P01_109_27/frame_002468_right |
| 27 | P01_109_27 | pour hand wash | 2470.0 | 2471.0 | right | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_27_P01_109_27/frame_002470_right |
| 27 | P01_109_27 | pour hand wash | 2477.0 | 2478.0 | right | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_27_P01_109_27/frame_002477_right |
| 27 | P01_109_27 | pour hand wash | 2484.0 | 2485.0 | right | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_27_P01_109_27/frame_002484_right |
| 28 | P01_109_28 | wash hands | 2477.0 | 2478.0 | right | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_28_P01_109_28/frame_002477_right |
| 28 | P01_109_28 | wash hands | 2479.0 | 2480.0 | right | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_28_P01_109_28/frame_002479_right |
| 29 | P01_109_29 | turn on tap | 2544.0 | 2545.0 | left | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_29_P01_109_29/frame_002544_left |
| 29 | P01_109_29 | turn on tap | 2561.0 | 2562.0 | left | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_29_P01_109_29/frame_002561_left |
| 29 | P01_109_29 | turn on tap | 2573.0 | 2574.0 | right | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_29_P01_109_29/frame_002573_right |
| 34 | P01_109_34 | dry hands | 3190.0 | 3191.0 | left | 3165.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_34_P01_109_34/frame_003190_left |
| 34 | P01_109_34 | dry hands | 3348.0 | 3349.0 | right | 3330.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_34_P01_109_34/frame_003348_right |
| 35 | P01_109_35 | put down cloth | 3348.0 | 3349.0 | right | 3330.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_35_P01_109_35/frame_003348_right |
| 35 | P01_109_35 | put down cloth | 3356.0 | 3357.0 | left | 3330.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_35_P01_109_35/frame_003356_left |
| 35 | P01_109_35 | put down cloth | 3360.0 | 3361.0 | right | 3330.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_35_P01_109_35/frame_003360_right |
| 47 | P01_109_47 | close cupboard | 4651.0 | 4652.0 | left | 4636.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_47_P01_109_47/frame_004651_left |
| 48 | P01_109_48 | put down cutting board | 4676.0 | 4677.0 | right | 4636.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_48_P01_109_48/frame_004676_right |
| 48 | P01_109_48 | put down cutting board | 4689.0 | 4690.0 | right | 4636.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_48_P01_109_48/frame_004689_right |
| 48 | P01_109_48 | put down cutting board | 4697.0 | 4698.0 | left | 4636.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_48_P01_109_48/frame_004697_left |
| 48 | P01_109_48 | put down cutting board | 4703.0 | 4704.0 | right | 4636.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_48_P01_109_48/frame_004703_right |
| 48 | P01_109_48 | put down cutting board | 4714.0 | 4715.0 | right | 4636.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_48_P01_109_48/frame_004714_right |
| 53 | P01_109_53 | pick up peeler | 4947.0 | 4948.0 | right | 4937.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_53_P01_109_53/frame_004947_right |
| 53 | P01_109_53 | pick up peeler | 4980.0 | 4981.0 | right | 4937.0 | outputs/数据处理小批量测试前100个subaction/experiments/E2/pipeline_outputs/subaction_53_P01_109_53/frame_004980_right |


## E3 Preview

| subaction_index | narration_id | narration | frame_0_based | frame_1_based | hand | ref_idx | sample_dir |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | P01_109_0 | grab rucksack | 78.0 | 79.0 | left | 67.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_00_P01_109_0/frame_000078_left |
| 20 | P01_109_20 | close door | 1863.0 | 1864.0 | right | 1798.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_20_P01_109_20/frame_001863_right |
| 20 | P01_109_20 | close door | 1908.0 | 1909.0 | right | 1895.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_20_P01_109_20/frame_001908_right |
| 21 | P01_109_21 | pick up cup | 2088.0 | 2089.0 | left | 2076.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_21_P01_109_21/frame_002088_left |
| 22 | P01_109_22 | put down cup | 2142.0 | 2143.0 | right | 2140.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_22_P01_109_22/frame_002142_right |
| 22 | P01_109_22 | put down cup | 2148.0 | 2149.0 | right | 2140.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_22_P01_109_22/frame_002148_right |
| 22 | P01_109_22 | put down cup | 2154.0 | 2155.0 | right | 2140.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_22_P01_109_22/frame_002154_right |
| 24 | P01_109_24 | open drawer | 2268.0 | 2269.0 | right | 2257.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_24_P01_109_24/frame_002268_right |
| 27 | P01_109_27 | pour hand wash | 2468.0 | 2469.0 | right | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_27_P01_109_27/frame_002468_right |
| 27 | P01_109_27 | pour hand wash | 2470.0 | 2471.0 | right | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_27_P01_109_27/frame_002470_right |
| 27 | P01_109_27 | pour hand wash | 2477.0 | 2478.0 | right | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_27_P01_109_27/frame_002477_right |
| 27 | P01_109_27 | pour hand wash | 2484.0 | 2485.0 | right | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_27_P01_109_27/frame_002484_right |
| 28 | P01_109_28 | wash hands | 2477.0 | 2478.0 | right | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_28_P01_109_28/frame_002477_right |
| 28 | P01_109_28 | wash hands | 2479.0 | 2480.0 | right | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_28_P01_109_28/frame_002479_right |
| 29 | P01_109_29 | turn on tap | 2544.0 | 2545.0 | left | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_29_P01_109_29/frame_002544_left |
| 29 | P01_109_29 | turn on tap | 2561.0 | 2562.0 | left | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_29_P01_109_29/frame_002561_left |
| 29 | P01_109_29 | turn on tap | 2573.0 | 2574.0 | right | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_29_P01_109_29/frame_002573_right |
| 30 | P01_109_30 | rinse hands | 2708.0 | 2709.0 | left | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_30_P01_109_30/frame_002708_left |
| 30 | P01_109_30 | rinse hands | 2762.0 | 2763.0 | left | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_30_P01_109_30/frame_002762_left |
| 30 | P01_109_30 | rinse hands | 2838.0 | 2839.0 | right | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_30_P01_109_30/frame_002838_right |
| 30 | P01_109_30 | rinse hands | 2878.0 | 2879.0 | right | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_30_P01_109_30/frame_002878_right |
| 31 | P01_109_31 | turn off tap | 2968.0 | 2969.0 | left | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_31_P01_109_31/frame_002968_left |
| 31 | P01_109_31 | turn off tap | 2970.0 | 2971.0 | left | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_31_P01_109_31/frame_002970_left |
| 31 | P01_109_31 | turn off tap | 2979.0 | 2980.0 | left | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_31_P01_109_31/frame_002979_left |
| 31 | P01_109_31 | turn off tap | 2982.0 | 2983.0 | left | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_31_P01_109_31/frame_002982_left |
| 31 | P01_109_31 | turn off tap | 2984.0 | 2985.0 | left | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_31_P01_109_31/frame_002984_left |
| 32 | P01_109_32 | shake hands | 3009.0 | 3010.0 | left | 2434.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_32_P01_109_32/frame_003009_left |
| 34 | P01_109_34 | dry hands | 3190.0 | 3191.0 | left | 3165.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_34_P01_109_34/frame_003190_left |
| 34 | P01_109_34 | dry hands | 3348.0 | 3349.0 | right | 3330.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_34_P01_109_34/frame_003348_right |
| 35 | P01_109_35 | put down cloth | 3348.0 | 3349.0 | right | 3330.0 | outputs/数据处理小批量测试前100个subaction/experiments/E3/pipeline_outputs/subaction_35_P01_109_35/frame_003348_right |
