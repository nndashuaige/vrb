# E20-A Summary

E20-A uses the two-pass design: export E18b-faithful label geometry, inpaint hands on GPU, then redraw heatmap and trajectory with the same functions and cached parameters on the inpainted canvas.

- samples: 56
- status: {'final_inpainted_qc_failed': 55, 'final_inpainted': 1}
- backend_final: {'lama': 32, 'sd': 24}
- tier: {'B': 56}
- redraw_selfcheck_max_diff_max: 0

Outputs:
- `e20_manifest.csv`
- `e20_quality.csv`
- `charts/e20_inpaint_contact_sheet_page_*.png`
