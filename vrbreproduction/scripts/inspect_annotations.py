#!/usr/bin/env python3
import pickle
from pathlib import Path

# Load annotation file
PROJECT_ROOT = Path(__file__).resolve().parents[1]
anno_file = PROJECT_ROOT / "data/P01_109.pkl"
with open(anno_file, 'rb') as f:
    data = pickle.load(f)

print(f"Video: P01_109")
print(f"Data type: {type(data)}")
print(f"Total frames: {len(data)}")

if len(data) > 0:
    print(f"\nFirst frame data:")
    print(f"Type: {type(data[0])}")
    print(f"Content: {data[0]}")

    # Show a few frame numbers
    print(f"\nSample frame indices: 0, {len(data)//2}, {len(data)-1}")
    print(f"\nTo download frames, visit:")
    print(f"https://data.bris.ac.uk/data/dataset/2g1n6qdydwa9u22shpxqzp0t8m")
    print(f"Video ID: P01_109")
