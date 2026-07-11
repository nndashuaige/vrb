# E20-B Summary

E20-B keeps the E20/E20-A two-pass design (export label geometry, GPU inpaint, local redraw)
and fixes the E20-A failure modes: missed hand boxes (HOA low-score @0.1 + neighbor-frame ±6
fallback + GroundingDINO hand/arm text detection), palm-only masks (DINO arm boxes + SAM +
direction-aware border bridging), arm regrowth (DINO-redetect re-inpaint rounds), SD letterbox
artifacts (2x native-aspect inpainting), SDEdit global blur (2x img2img composited only inside
the mask), and the broken QC (boundary_delta was 0 by construction; YCrCb skin confused wood
counters with skin — replaced by DINO re-detection + seam gradient ratio; deterministic
quality ranking replaces the seeded 50/50 backend pick).

- samples: 56
- status: {'final_inpainted': 53, 'final_inpainted_qc_failed': 3}
- backend_final: {'lama': 46, 'sd': 10}
- qc_passed_final=true: 53
- mask_nonzero: 56
- tier: {'B': 40, 'A': 16}
- redraw_selfcheck_max_diff_max: 0

Outputs:
- `e20_manifest.csv`
- `e20_quality.csv`
- `charts/e20_inpaint_contact_sheet_page_*.png`
- `charts/e20b_before_after_enhanced_contact_sheet_page_001.png` … (10 pages, original | e20b_final | enhanced | mask | lama | sd | e20a_final)
