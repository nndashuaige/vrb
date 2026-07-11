# E20-C Summary

E20-C keeps the E20/E20-A/E20-B two-pass design (export label geometry, GPU inpaint, local
redraw) and fixes the four residual E20-B failure modes:

1. **Target-object protection**: E20-B masks swallowed the interaction target in 20/56
   samples (overlap_heatmap_top >= 0.99, e.g. the rucksack in subaction_01 and the cutting
   board in subaction_66). E20-C builds a protect mask from the exported label geometry
   (SAM object segmentation prompted by heatmap_top_bbox_px + the bbox rect + contact-point
   disks) and subtracts it from the inpaint mask as a hard constraint, every round.
2. **SD hallucination gate**: SD may not be the final if its in-mask sharpness exceeds 2x the
   original frame's (calibrated on all 56 E20-B samples) or the mask covers >0.40 of the frame.
3. **Tamer masks**: dilate 14->8, bridge width 1.4->1.3 reduce unnecessary hole growth.
4. **Skin-confirmed re-detection**: DINO hand/arm fires only count when the box also has
   dual-rule skin evidence (AND condition; skin alone matches wooden counters), removing the
   trouser-leg/knee false fails of E20-B.

## Chart layout (charts/e20c_before_after_enhanced_contact_sheet_page_*.png)

One row = one sample. The grey band above each row states: subaction id | narration_id |
narration (action text from EPIC_100_train.csv) | chosen final backend | strict-QC verdict.
Columns, labelled in the header: original (reference frame), e20c_final (chosen inpainted
result), affordance_redrawn (vrb_style_affordance_inpainted.png - labels redrawn on the
inpainted canvas; heatmap/arrow should still sit on the target object), enhanced (SDEdit
candidate), mask(+protect) (inpaint mask in white, protected label region in green), lama /
sd (backend candidates), e20b_final (previous iteration, for regression checking).

## Numbers

- samples: 56
- status: {'final_inpainted': 38, 'final_inpainted_qc_failed': 18}
- backend_final: {'lama': 46, 'sd': 10}
- qc_passed_final=true: 38
- mask_nonzero: 56
- protect_intact_final=true: 56 (protect region untouched by the final mask)
- protect_source=sam+bbox: 25 (SAM object segmentation accepted)
- label_region_conflict=true: 28 (hand overlaps label region; mask clipped there)
- overlap_heatmap_top==0: 56 (E20-B had 18/56 at 1.0)
- tier: {'A': 38, 'B': 18}
- redraw_selfcheck_max_diff_max: 0

Outputs:
- `e20_manifest.csv` / `e20_quality.csv`
- `charts/e20_inpaint_contact_sheet_page_*.png` (legacy layout)
- `charts/e20c_before_after_enhanced_contact_sheet_page_001.png` … (10 pages, annotated layout described above)
