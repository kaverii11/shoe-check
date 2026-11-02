"""Evaluate every method on real phone photos: <dir>/sharp/* and <dir>/blurry/*.

Each classical score is tried on the full frame, the centre crop, and the shoe region (GrabCut),
because portrait-mode photos have a sharp shoe on a deliberately blurred background.

Usage: python src/evaluate_real.py --photos data/real --out results
"""
import argparse
import json
import os
import pickle

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sharpness import features, resize_short
from train_cnn import EVAL_TF

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()  # iPhone photos
except ImportError:
    pass

SIZE = 256  # same working resolution as the training data


def load(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")  # respect the phone's rotation tag
    return resize_short(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), SIZE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", default="data/real")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    items = [(os.path.join(args.photos, c, f), int(c == "blurry"), f)
             for c in ("sharp", "blurry") for f in sorted(os.listdir(os.path.join(args.photos, c))) if not f.startswith(".")]
    imgs = [load(p) for p, _, _ in items]
    y = np.array([l for _, l, _ in items])
    names = [f for _, _, f in items]
    print(f"{len(y)} photos: {int((y == 0).sum())} sharp, {int(y.sum())} blurry")

    models = pickle.load(open(os.path.join(args.out, "classic_models.pkl"), "rb"))
    results, per_photo = {}, {n: {"label": int(l)} for n, l in zip(names, y)}

    for region in ("full", "center", "shoe"):
        X = np.stack([features(im, region) for im in imgs])
        t = models["thresholds"]["laplacian_var"]
        pred = (X[:, 0] < t).astype(int)
        results[f"laplacian_var / {region}"] = {"accuracy": accuracy_score(y, pred), "f1_blurry": f1_score(y, pred), "roc_auc": roc_auc_score(y, -X[:, 0])}
        prob = models["logreg"].predict_proba(X)[:, 1]
        results[f"logistic_regression / {region}"] = {"accuracy": accuracy_score(y, prob >= 0.5), "f1_blurry": f1_score(y, prob >= 0.5), "roc_auc": roc_auc_score(y, prob)}
        for n, p in zip(names, prob):
            per_photo[n][f"logreg_{region}"] = round(float(p), 3)

    for variant in ("fp32", "fp16"):
        sess = ort.InferenceSession(os.path.join(args.out, f"shoe_check_{variant}.onnx"), providers=["CPUExecutionProvider"])
        prob = []
        for im in imgs:
            x = EVAL_TF(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))).unsqueeze(0).numpy()
            prob.append(1 / (1 + np.exp(-sess.run(None, {"input": x})[0].item())))
        prob = np.array(prob)
        results[f"mobilenetv3_small / {variant}"] = {"accuracy": accuracy_score(y, prob >= 0.5), "f1_blurry": f1_score(y, prob >= 0.5), "roc_auc": roc_auc_score(y, prob)}
        for n, p in zip(names, prob):
            per_photo[n][f"cnn_{variant}"] = round(float(p), 3)

    with open(os.path.join(args.out, "real_metrics.json"), "w") as f:
        json.dump({"n_sharp": int((y == 0).sum()), "n_blurry": int(y.sum()), "results": results, "per_photo": per_photo}, f, indent=2)
    for k, v in results.items():
        print(f"{k:38s} acc {v['accuracy']:.3f}  F1 {v['f1_blurry']:.3f}  AUC {v['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
