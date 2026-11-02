"""Build the synthetic sharp/blurry dataset from the Sneakers dataset (Hugging Face: ipogorelov/sneakers).

Every source photo yields one sharp and one blurred version, and the train/val/test split is made
by source photo, so the same shoe never appears in two splits.

Usage: python src/prepare_data.py --parquet data/raw/*.parquet --out data/synthetic
"""
import argparse
import csv
import io
import os

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from degrade import blur, nuisance
from sharpness import resize_short

SIZE = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--min-side", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    sources = []
    for path in args.parquet:
        for raw in pq.read_table(path, columns=["image"]).column("image").to_pylist():
            img = cv2.cvtColor(np.array(Image.open(io.BytesIO(raw)).convert("RGB")), cv2.COLOR_RGB2BGR)
            if min(img.shape[:2]) >= args.min_side:
                sources.append(resize_short(img, SIZE))

    # The scraped photos aren't guaranteed sharp; drop the softest 10% so the "sharp" class is cleaner
    lap = np.array([cv2.Laplacian(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() for s in sources])
    keep = lap >= np.percentile(lap, 10)
    sources = [s for s, k in zip(sources, keep) if k]

    order = rng.permutation(len(sources))
    n_train, n_val = int(0.7 * len(order)), int(0.15 * len(order))
    split_of = {int(i): ("train" if r < n_train else "val" if r < n_train + n_val else "test") for r, i in enumerate(order)}

    rows = []
    for i, src in enumerate(sources):
        sharp = nuisance(src, rng)
        blurred, how = blur(src, rng)
        blurred = nuisance(blurred, rng)
        for label, img, note in [(0, sharp, "sharp"), (1, blurred, how)]:
            name = f"{i:05d}_{'blurry' if label else 'sharp'}.jpg"
            cv2.imwrite(os.path.join(args.out, "images", name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({"file": name, "label": label, "split": split_of[i], "source": i, "degradation": note})

    with open(os.path.join(args.out, "labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    print(f"{len(sources)} source photos (dropped {int((~keep).sum())} softest), images per split: {counts}")


if __name__ == "__main__":
    main()
