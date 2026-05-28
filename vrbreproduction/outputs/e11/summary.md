# E11 handless-reference inpainting postprocess

- Source E10 output: `/Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/outputs/e10`
- E11 output: `/Users/huhu/Documents/Code/bishe/vrb/vrbreproduction/outputs/e11`
- Projection/candidate/reference selection: unchanged from E10; this run only dehands the exported reference image.
- Subactions covered: `79`
- Final tuples: `157`
- E10 baseline subactions/final tuples: `79` / `157`
- CoTracker success count: `365`
- fallback tracker success count: `0`

## reference_handless_mode

| mode | count |
| --- | ---: |
| inpainted_bbox | 143 |
| strict_no_hand | 9 |
| local_clean | 5 |

## inpaint_backend

| backend | count |
| --- | ---: |
| opencv_telea | 148 |
| none | 9 |

## inpaint_status

| status | count |
| --- | ---: |
| success | 148 |
| not_needed | 9 |

## Output files

- `experiment_overview.csv`
- `subaction_summary.csv`
- `candidate_diagnostics.csv`
- `successful_sample_manifest.csv`
- Per-sample: `reference_frame_original.png`, `reference_frame_dehanded.png`, `hand_mask.png`

## Rebuilt Charts

- `charts/e11_pipeline_funnel.png`
- `charts/e11_failure_reasons.png`
- `charts/e11_track_confidence_hist.png`
- `charts/e11_projection_source_counts.png`
- `charts/e11_success_by_subaction.png`
- `charts/e11_mask_projection_contact_sheet.png`
- `charts/e11_success_contact_sheet_page_001_page_001.png`
- `charts/e11_success_contact_sheet_page_001_page_002.png`
- `charts/e11_success_contact_sheet_page_001_page_003.png`
- `charts/e11_success_contact_sheet_page_001_page_004.png`
- `charts/e11_success_contact_sheet_page_001_page_005.png`
- `charts/e11_success_contact_sheet_page_001_page_006.png`
- `charts/e11_success_contact_sheet_page_001_page_007.png`
- `charts/e11_success_contact_sheet_page_001_page_008.png`
- `charts/e11_success_contact_sheet_page_001_page_009.png`
- `charts/e11_success_contact_sheet_page_001_page_010.png`
