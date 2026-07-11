"""e-analysis-1: EPIC-KITCHENS 可用数据筛选与统计。

可用数据定义（一条 narration/subaction 为一个统计单位）：
  1. 公开 annotation 中包含 "对 xx 物体做 xx 动作"（verb + noun 均非空）；
  2. annotation 给出动作的帧范围（start_frame / stop_frame 有效且 stop > start）；
  3. 该视频存在公开的手框 + 物体框标注（EPIC-KITCHENS-100 官方
     hand-object bboxes 发布，按 video_id 提供逐帧手/物体检测框），
     且动作帧范围落在该发布覆盖的帧数之内。

数据来源：
  - 动作标注: epic-kitchens-100-annotations 的 EPIC_100_train.csv /
    EPIC_100_validation.csv（test 集无公开标签，不计入可用数据）。
  - 手/物体框覆盖清单: epic-kitchens-100-hand-object-bboxes 仓库的
    EPIC_100_frame_counts.csv（该发布覆盖的全部视频及其帧数）。

统计输出（均只在可用数据上统计）：
  - noun_frequency.csv        被操作对象及出现频次
  - verb_frequency.csv        执行动作及出现频次
  - verb_noun_pair_frequency.csv  动作-对象组合频次
  - usability_funnel.csv      各筛选门控的保留数量
  - usable_subactions.csv     全部可用 subaction 明细
  - experiment_results.json   实验总览
  - charts/                   Top-N 频次条形图
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANNOTATION_FILE_URLS = {
    name: (
        "https://raw.githubusercontent.com/epic-kitchens/"
        f"epic-kitchens-100-annotations/master/{name}"
    )
    for name in (
        "EPIC_100_train.csv",
        "EPIC_100_validation.csv",
        "EPIC_100_noun_classes.csv",
        "EPIC_100_verb_classes.csv",
    )
}

HAND_OBJECT_COVERAGE_URL = (
    "https://raw.githubusercontent.com/epic-kitchens/"
    "epic-kitchens-100-hand-object-bboxes/master/EPIC_100_frame_counts.csv"
)


@dataclass
class EAnalysis1Config:
    annotations_dir: Path = (
        PROJECT_ROOT / "data" / "annotations" / "epic-kitchens-100-annotations"
    )
    output_root: Path = PROJECT_ROOT / "outputs" / "e-analysis-1"
    top_n_chart: int = 30
    splits: tuple = ("train", "validation")
    downloads_dir: Path = field(init=False)
    charts_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.downloads_dir = self.output_root / "downloads"
        self.charts_dir = self.output_root / "charts"


def _ensure_annotation_csv(cfg: EAnalysis1Config, filename: str) -> Path:
    """优先使用本地已有标注；缺失时下载到本实验 downloads 目录。"""
    local = cfg.annotations_dir / filename
    if local.exists():
        return local
    downloaded = cfg.downloads_dir / filename
    if not downloaded.exists():
        print(f"[e-analysis-1] downloading {filename} ...")
        urlretrieve(ANNOTATION_FILE_URLS[filename], downloaded)
    return downloaded


def _ensure_hand_object_coverage(cfg: EAnalysis1Config) -> Path:
    path = cfg.downloads_dir / "EPIC_100_frame_counts.csv"
    if not path.exists():
        print("[e-analysis-1] downloading hand-object bbox coverage list ...")
        urlretrieve(HAND_OBJECT_COVERAGE_URL, path)
    return path


def load_action_annotations(cfg: EAnalysis1Config) -> pd.DataFrame:
    frames = []
    for split in cfg.splits:
        csv_path = _ensure_annotation_csv(cfg, f"EPIC_100_{split}.csv")
        df = pd.read_csv(csv_path)
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def apply_usability_gates(
    annotations: pd.DataFrame, coverage: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict]]:
    """逐条应用可用性门控，返回可用子集与漏斗记录。"""
    funnel: list[dict] = []

    def record(gate: str, df: pd.DataFrame) -> None:
        funnel.append({"gate": gate, "remaining_subactions": len(df)})

    record("all_public_labelled_subactions (train+validation)", annotations)

    df = annotations[
        annotations["verb"].notna()
        & annotations["noun"].notna()
        & (annotations["verb"].astype(str).str.strip() != "")
        & (annotations["noun"].astype(str).str.strip() != "")
    ]
    record("has_verb_and_noun", df)

    start = pd.to_numeric(df["start_frame"], errors="coerce")
    stop = pd.to_numeric(df["stop_frame"], errors="coerce")
    df = df[start.notna() & stop.notna() & (stop > start) & (start >= 0)]
    record("has_valid_frame_range", df)

    covered_videos = set(coverage["video_id"])
    df = df[df["video_id"].isin(covered_videos)]
    record("video_has_hand_and_object_bboxes", df)

    n_frames = coverage.set_index("video_id")["rgb_n_frames"]
    df = df[df["stop_frame"] <= df["video_id"].map(n_frames)]
    record("frame_range_within_bbox_coverage", df)

    return df.copy(), funnel


def frequency_table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    freq = df.groupby(by, as_index=False).size().rename(columns={"size": "count"})
    freq = freq.sort_values("count", ascending=False, ignore_index=True)
    freq["percentage"] = (freq["count"] / freq["count"].sum() * 100).round(3)
    return freq


def _plot_top_n(freq: pd.DataFrame, label_col: str, title: str, path: Path, top_n: int) -> None:
    top = freq.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(top))))
    ax.barh(top[label_col].astype(str), top["count"], color="#4c72b0")
    ax.set_title(title)
    ax.set_xlabel("subaction count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_e_analysis_1(cfg: EAnalysis1Config) -> dict:
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    cfg.downloads_dir.mkdir(parents=True, exist_ok=True)
    cfg.charts_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_action_annotations(cfg)
    coverage = pd.read_csv(_ensure_hand_object_coverage(cfg))

    usable, funnel = apply_usability_gates(annotations, coverage)

    noun_classes = pd.read_csv(_ensure_annotation_csv(cfg, "EPIC_100_noun_classes.csv"))
    verb_classes = pd.read_csv(_ensure_annotation_csv(cfg, "EPIC_100_verb_classes.csv"))
    usable["noun_class_key"] = usable["noun_class"].map(
        noun_classes.set_index("id")["key"]
    )
    usable["verb_class_key"] = usable["verb_class"].map(
        verb_classes.set_index("id")["key"]
    )

    # raw 粒度：narration 归一化后的原始词（同一 class 内的变体分开计）
    noun_freq = frequency_table(usable, ["noun", "noun_class"])
    verb_freq = frequency_table(usable, ["verb", "verb_class"])
    pair_freq = frequency_table(usable, ["verb", "noun"])

    # class 粒度：官方 noun/verb class 归并（如 pick-up/take 同为 take 类）
    noun_class_freq = frequency_table(usable, ["noun_class_key", "noun_class"])
    verb_class_freq = frequency_table(usable, ["verb_class_key", "verb_class"])
    pair_class_freq = frequency_table(usable, ["verb_class_key", "noun_class_key"])

    usable.to_csv(cfg.output_root / "usable_subactions.csv", index=False)
    pd.DataFrame(funnel).to_csv(cfg.output_root / "usability_funnel.csv", index=False)
    noun_freq.to_csv(cfg.output_root / "noun_frequency.csv", index=False)
    verb_freq.to_csv(cfg.output_root / "verb_frequency.csv", index=False)
    pair_freq.to_csv(cfg.output_root / "verb_noun_pair_frequency.csv", index=False)
    noun_class_freq.to_csv(cfg.output_root / "noun_class_frequency.csv", index=False)
    verb_class_freq.to_csv(cfg.output_root / "verb_class_frequency.csv", index=False)
    pair_class_freq.to_csv(
        cfg.output_root / "verb_noun_class_pair_frequency.csv", index=False
    )

    _plot_top_n(
        noun_freq, "noun",
        f"e-analysis-1: top {cfg.top_n_chart} manipulated objects (usable subactions)",
        cfg.charts_dir / "top_nouns.png", cfg.top_n_chart,
    )
    _plot_top_n(
        verb_freq, "verb",
        f"e-analysis-1: top {cfg.top_n_chart} verbs (usable subactions)",
        cfg.charts_dir / "top_verbs.png", cfg.top_n_chart,
    )
    pair_labels = pair_freq.assign(pair=pair_freq["verb"] + " " + pair_freq["noun"])
    _plot_top_n(
        pair_labels, "pair",
        f"e-analysis-1: top {cfg.top_n_chart} verb-noun pairs (usable subactions)",
        cfg.charts_dir / "top_verb_noun_pairs.png", cfg.top_n_chart,
    )
    _plot_top_n(
        noun_class_freq, "noun_class_key",
        f"e-analysis-1: top {cfg.top_n_chart} manipulated object classes",
        cfg.charts_dir / "top_noun_classes.png", cfg.top_n_chart,
    )
    _plot_top_n(
        verb_class_freq, "verb_class_key",
        f"e-analysis-1: top {cfg.top_n_chart} verb classes",
        cfg.charts_dir / "top_verb_classes.png", cfg.top_n_chart,
    )

    summary = {
        "experiment": "e-analysis-1",
        "dataset": "EPIC-KITCHENS-100",
        "usable_data_definition": [
            "annotation contains verb + noun (对xx物体做xx动作)",
            "annotation contains valid start_frame/stop_frame",
            "video covered by official hand-object bboxes release "
            "(hand boxes + object boxes, per frame)",
            "subaction frame range within bbox release frame count",
        ],
        "splits_used": list(cfg.splits),
        "usability_funnel": funnel,
        "num_usable_subactions": int(len(usable)),
        "num_usable_videos": int(usable["video_id"].nunique()),
        "num_usable_participants": int(usable["participant_id"].nunique()),
        "num_distinct_nouns_raw": int(len(noun_freq)),
        "num_distinct_verbs_raw": int(len(verb_freq)),
        "num_distinct_verb_noun_pairs_raw": int(len(pair_freq)),
        "num_distinct_noun_classes": int(len(noun_class_freq)),
        "num_distinct_verb_classes": int(len(verb_class_freq)),
        "num_distinct_verb_noun_class_pairs": int(len(pair_class_freq)),
        "top10_noun_classes": noun_class_freq.head(10).to_dict(orient="records"),
        "top10_verb_classes": verb_class_freq.head(10).to_dict(orient="records"),
        "top10_verb_noun_class_pairs": pair_class_freq.head(10).to_dict(
            orient="records"
        ),
    }
    with open(cfg.output_root / "experiment_results.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[e-analysis-1] usable subactions: {summary['num_usable_subactions']}")
    print(
        "[e-analysis-1] distinct nouns: "
        f"{summary['num_distinct_nouns_raw']} raw / "
        f"{summary['num_distinct_noun_classes']} classes"
    )
    print(
        "[e-analysis-1] distinct verbs: "
        f"{summary['num_distinct_verbs_raw']} raw / "
        f"{summary['num_distinct_verb_classes']} classes"
    )
    print(f"[e-analysis-1] outputs written to {cfg.output_root}")
    return summary
