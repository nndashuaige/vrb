# E19 hand-inpaint reference

## Motivation

E18 is the current hybrid reference baseline: it first searches for `active_hand_invisible_pre_contact`, and falls back to `min_hand_object_iou_pre_contact_fallback` when needed. Some fallback references can still contain visible hands. Seeing the Unseen suggests a practical data-construction pattern: detect/segment an entity, inpaint it, filter failures, and use the cleaned image downstream.

## Method

E19 keeps E18's contact/reference selection and geometry unchanged. It first computes and caches projected contact points, heatmap parameters, and trajectory coordinates on the original reference frame. If the active hand is visible, it then inpaints the reference frame and redraws the already-computed labels on the hand-free reference canvas. Results are reported separately for `active_hand_invisible_pre_contact` and `min_hand_object_iou_pre_contact_fallback`.

## Backend

OpenCV Telea smoke/full baseline; LaMa optional; Stable Diffusion optional qualitative comparison.

## Results

- smoke summary: `outputs/e19_e18_hand_inpaint_reference_smoke10/summary.md`
- full summary: `outputs/e19_e18_hand_inpaint_reference/summary.md`
- inpaint contact sheet: `outputs/e19_e18_hand_inpaint_reference/charts/e19_inpaint_contact_sheet_page_001.png`
- quality summary command: `python scripts/summarize_hand_inpaint_quality.py outputs/e19_e18_hand_inpaint_reference/candidate_diagnostics.csv`

## Limitations

The inpainted region is plausible, not true. Do not treat generated object/contact details as ground truth. If the hand covers the true contact surface, this may improve appearance but damage semantic/geometric fidelity.
