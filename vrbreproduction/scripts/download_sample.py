#!/usr/bin/env python3
"""
Download a small sample from EPIC-KITCHENS 100 DOH dataset
"""
import os
import requests
import pandas as pd
from pathlib import Path

# Create data directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
data_dir = PROJECT_ROOT / "data"
data_dir.mkdir(exist_ok=True)

# Download hand-object bounding boxes annotation
print("Downloading 100 DOH annotations...")
anno_url = "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-hand-object-bboxes/master/annotations/EPIC_100_train.csv"
anno_path = data_dir / "EPIC_100_train.csv"

response = requests.get(anno_url)
with open(anno_path, 'wb') as f:
    f.write(response.content)

# Read annotations and pick a video_id
df = pd.read_csv(anno_path)
print(f"\nTotal annotations: {len(df)}")
print(f"Unique videos: {df['video_id'].nunique()}")

# Pick first video_id with annotations
sample_video = df['video_id'].iloc[0]
print(f"\nSelected video: {sample_video}")

# Filter annotations for this video
video_df = df[df['video_id'] == sample_video]
print(f"Frames in this video: {len(video_df)}")

# Save filtered annotations
filtered_path = data_dir / f"{sample_video}_annotations.csv"
video_df.to_csv(filtered_path, index=False)
print(f"\nSaved filtered annotations to: {filtered_path}")

print(f"\n✓ Sample video_id: {sample_video}")
print(f"✓ Annotation file: {filtered_path}")
print(f"\nNote: Frame images need to be downloaded from EPIC-KITCHENS website")
print(f"Visit: https://epic-kitchens.github.io/2024")
