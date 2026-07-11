from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

OUTPUT_ROOT = ROOT / "outputs" / "e18"
OUT_DIR = OUTPUT_ROOT / "charts" / "reference_mode_final_results"


def load_overlay(row: pd.Series) -> Optional[str]:
    for col in ["vrb_style_affordance", "label_heatmap_overlay", "problem3_full_overlay"]:
        path = row.get(col)
        if isinstance(path, str) and path and Path(path).exists():
            return path
    return None


def draw_label(img: np.ndarray, text: str, height: int = 52) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], min(height, img.shape[0])), (0, 0, 0), -1)
    for idx, line in enumerate(text.split("\n")[:3]):
        cv2.putText(
            img,
            line[:72],
            (6, 16 + idx * 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def render_tile(row: pd.Series, tile_size: Tuple[int, int] = (460, 320)) -> np.ndarray:
    tile_w, tile_h = tile_size
    canvas = np.full((tile_h, tile_w, 3), 245, dtype=np.uint8)
    path = load_overlay(row)
    if path is None:
        draw_label(canvas, f"s{int(row['subaction_index'])} {row.get('narration_id', '')}\nmissing result overlay")
        return canvas

    img = cv2.imread(path)
    if img is None:
        draw_label(canvas, f"s{int(row['subaction_index'])} {row.get('narration_id', '')}\nfailed to load overlay")
        return canvas

    img = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    mode = str(row.get("reference_mode", ""))
    iou = row.get("reference_hand_object_iou")
    iou_text = "NA" if pd.isna(iou) else f"{float(iou):.3f}"
    gap = row.get("reference_anchor_gap")
    cand = row.get("fallback_candidate_count")
    cand_text = "NA" if pd.isna(cand) else str(int(cand))
    title = (
        f"s{int(row['subaction_index'])} {row.get('narration_id', '')}\n"
        f"{row.get('narration', '')[:42]}\n"
        f"{mode[:30]} gap {gap} iou {iou_text} cand {cand_text}"
    )
    draw_label(img, title)
    return img


def save_sheet(rows: pd.DataFrame, out_prefix: Path, per_page: int = 12, cols: int = 3) -> List[Path]:
    paths: List[Path] = []
    rows = rows.sort_values(["subaction_index", "frame_0_based"]).reset_index(drop=True)
    tile_w, tile_h = 460, 320
    for page_idx, start in enumerate(range(0, len(rows), per_page), start=1):
        page = rows.iloc[start : start + per_page]
        page_rows = int(math.ceil(len(page) / cols))
        sheet = np.full((page_rows * tile_h, cols * tile_w, 3), 245, dtype=np.uint8)
        for local_idx, (_, row) in enumerate(page.iterrows()):
            r = local_idx // cols
            c = local_idx % cols
            tile = render_tile(row, tile_size=(tile_w, tile_h))
            sheet[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tile
        out_path = out_prefix.parent / f"{out_prefix.name}_page_{page_idx:03d}.png"
        cv2.imwrite(str(out_path), sheet)
        paths.append(out_path)
    return paths


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_df = pd.read_csv(OUTPUT_ROOT / "candidate_diagnostics.csv")
    keep = candidate_df[candidate_df["status"] == "keep"].copy()

    active = keep[keep["reference_mode"] == "active_hand_invisible_pre_contact"].copy()
    fallback = keep[keep["reference_mode"] == "min_hand_object_iou_pre_contact_fallback"].copy()

    active_paths = save_sheet(active, OUT_DIR / "e18_active_hand_invisible_final_results")
    fallback_paths = save_sheet(fallback, OUT_DIR / "e18_fallback_min_iou_final_results")

    readme = OUT_DIR / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# E18 final result comparison",
                "",
                f"- active_hand_invisible keep samples: `{len(active)}`",
                f"- fallback min-IoU keep samples: `{len(fallback)}`",
                "",
                "These sheets show only the final output overlay per sample.",
                "",
                "Generated files:",
                *[f"- `{p.name}`" for p in active_paths + fallback_paths],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(active_paths)} active pages and {len(fallback_paths)} fallback pages to {OUT_DIR}")


if __name__ == "__main__":
    main()
