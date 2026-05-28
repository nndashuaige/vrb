#!/usr/bin/env python3
"""
Extract a small sample of frames from P01_109 annotations
"""
import pickle
from pathlib import Path

# Load annotations
PROJECT_ROOT = Path(__file__).resolve().parents[1]
anno_file = PROJECT_ROOT / "data/P01_109.pkl"
with open(anno_file, 'rb') as f:
    data = pickle.load(f)

print(f"Total frames: {len(data)}")
print(f"\n建议：")
print(f"1. 完整视频太大（18万帧），建议只下载一小段")
print(f"2. 可以从 frame_0010000 到 frame_0010100（100帧，约3-4秒）")
print(f"\n手动下载步骤：")
print(f"访问: https://data.bris.ac.uk/data/dataset/2g1n6qdydwa9u22shpxqzp0t8m")
print(f"搜索: P01_109")
print(f"下载 RGB 帧压缩包，解压后提取 frame_0010000 到 frame_0010100")
