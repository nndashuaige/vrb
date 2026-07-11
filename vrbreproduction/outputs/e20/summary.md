# E20 Summary

E20 uses the two-pass design: export E18b-faithful label geometry, inpaint hands on GPU, then redraw heatmap and trajectory with the same functions and cached parameters on the inpainted canvas.

- samples: 56
- status: {'inpaint_failed_kept_original': 55, 'final_inpainted': 1}
- backend_final: {'original': 55, 'lama': 1}
- tier: {'B': 55, 'A': 1}
- redraw_selfcheck_max_diff_max: 0

Outputs:
- `e20_manifest.csv`
- `e20_quality.csv`
- `charts/e20_inpaint_contact_sheet_page_*.png`
