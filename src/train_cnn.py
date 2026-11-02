"""Fine-tune MobileNetV3-Small (ImageNet) to classify sharp vs blurry sneaker photos.

Usage: python src/train_cnn.py --data data/synthetic --out results
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# No resizing/blurring augmentations: those would change the very thing we're classifying
TRAIN_TF = transforms.Compose([transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
EVAL_TF = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class Shoes(Dataset):
    def __init__(self, data, split, tf):
        self.rows = [r for r in csv.DictReader(open(os.path.join(data, "labels.csv"))) if r["split"] == split]
        self.data, self.tf = data, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(os.path.join(self.data, "images", r["file"])).convert("RGB")
        return self.tf(img), torch.tensor(float(r["label"]))


def build_model():
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    return model


def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(torch.sigmoid(model(x.to(device)).squeeze(1)).cpu())
            ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    train = DataLoader(Shoes(args.data, "train", TRAIN_TF), batch_size=64, shuffle=True, num_workers=4)
    val = DataLoader(Shoes(args.data, "val", EVAL_TF), batch_size=128, num_workers=4)
    test = DataLoader(Shoes(args.data, "test", EVAL_TF), batch_size=128, num_workers=4)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.epochs * len(train))
    loss_fn = nn.BCEWithLogitsLoss()

    best = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)).squeeze(1), y.to(device))
            loss.backward()
            opt.step()
            sched.step()
        p, y = predict(model, val, device)
        acc = accuracy_score(y, p >= 0.5)
        print(f"epoch {epoch} val accuracy {acc:.4f} AUC {roc_auc_score(y, p):.4f}", flush=True)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), os.path.join(args.out, "mobilenetv3_small.pt"))

    model.load_state_dict(torch.load(os.path.join(args.out, "mobilenetv3_small.pt")))
    p, y = predict(model, test, device)
    res = {"accuracy": accuracy_score(y, p >= 0.5), "f1_blurry": f1_score(y, p >= 0.5), "roc_auc": roc_auc_score(y, p)}
    with open(os.path.join(args.out, "cnn_metrics.json"), "w") as f:
        json.dump({"test": res}, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
