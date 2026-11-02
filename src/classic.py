"""Classical baselines: each sharpness score with a threshold tuned on validation,
plus a logistic regression on all three scores.

Usage: python src/classic.py --data data/synthetic --out results
"""
import argparse
import csv
import json
import os
import pickle
import time

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sharpness import features

NAMES = ["laplacian_var", "tenengrad", "fft_highfreq"]


def load_split(data, split):
    rows = [r for r in csv.DictReader(open(os.path.join(data, "labels.csv"))) if r["split"] == split]
    X = np.stack([features(cv2.imread(os.path.join(data, "images", r["file"]))) for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    return X, y


def best_threshold(score, y):
    """Blurry = score below the threshold. Pick the cut-off with the best balanced accuracy."""
    cands = np.quantile(score, np.linspace(0.01, 0.99, 197))
    bal = [((score < t) & (y == 1)).sum() / (y == 1).sum() / 2 + ((score >= t) & (y == 0)).sum() / (y == 0).sum() / 2 for t in cands]
    return float(cands[int(np.argmax(bal))])


def report(y, pred, blur_prob):
    return {"accuracy": accuracy_score(y, pred), "f1_blurry": f1_score(y, pred), "roc_auc": roc_auc_score(y, blur_prob)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    Xtr, ytr = load_split(args.data, "train")
    Xva, yva = load_split(args.data, "val")
    Xte, yte = load_split(args.data, "test")

    results, thresholds = {}, {}
    for j, name in enumerate(NAMES):
        t = best_threshold(Xva[:, j], yva)
        thresholds[name] = t
        results[name] = report(yte, (Xte[:, j] < t).astype(int), -Xte[:, j])

    clf = make_pipeline(StandardScaler(), LogisticRegression())
    clf.fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    results["logistic_regression"] = report(yte, clf.predict(Xte), clf.predict_proba(Xte)[:, 1])

    img = cv2.imread(os.path.join(args.data, "images", os.listdir(os.path.join(args.data, "images"))[0]))
    t0 = time.perf_counter()
    for _ in range(200):
        features(img)
    results["features_ms_per_image_cpu"] = (time.perf_counter() - t0) / 200 * 1000

    with open(os.path.join(args.out, "classic_metrics.json"), "w") as f:
        json.dump({"test": results, "thresholds": thresholds}, f, indent=2)
    with open(os.path.join(args.out, "classic_models.pkl"), "wb") as f:
        pickle.dump({"logreg": clf, "thresholds": thresholds}, f)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
