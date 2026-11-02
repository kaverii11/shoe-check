"""Export the CNN to ONNX (float32 and float16), try int8 quantization, and compare size, CPU speed and accuracy.

int8 post-training quantization is included as an experiment: on MobileNetV3 it costs a lot of accuracy
(hard-swish activations and squeeze-excitation layers are sensitive to it), so the app uses float16.

Usage: python src/export.py --data data/synthetic --out results
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import onnxruntime as ort
import torch
import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image

from train_cnn import EVAL_TF, build_model


class Calibration(CalibrationDataReader):
    """Feeds ~100 training images through the model so int8 ranges match real activations."""

    def __init__(self, data, n=100):
        rows = [r for r in csv.DictReader(open(os.path.join(data, "labels.csv"))) if r["split"] == "train"][:n]
        self.batches = iter(
            [{"input": EVAL_TF(Image.open(os.path.join(data, "images", r["file"])).convert("RGB")).unsqueeze(0).numpy()} for r in rows]
        )

    def get_next(self):
        return next(self.batches, None)


def latency_ms(path, runs=200):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # one CPU thread, closer to a phone than a laptop's full CPU
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, 224, 224).astype(np.float32)
    for _ in range(20):
        sess.run(None, {"input": x})
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {"input": x})
    return (time.perf_counter() - t0) / runs * 1000


def accuracy(path, data):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    rows = [r for r in csv.DictReader(open(os.path.join(data, "labels.csv"))) if r["split"] == "test"]
    correct = 0
    for r in rows:
        x = EVAL_TF(Image.open(os.path.join(data, "images", r["file"])).convert("RGB")).unsqueeze(0).numpy()
        logit = sess.run(None, {"input": x})[0].item()
        correct += int((logit > 0) == bool(int(r["label"])))
    return correct / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    model = build_model()
    model.load_state_dict(torch.load(os.path.join(args.out, "mobilenetv3_small.pt"), map_location="cpu"))
    model.eval()

    fp32 = os.path.join(args.out, "shoe_check_fp32.onnx")
    fp16 = os.path.join(args.out, "shoe_check_fp16.onnx")
    pre = os.path.join(args.out, "shoe_check_pre.onnx")
    int8 = os.path.join(args.out, "shoe_check_int8.onnx")
    torch.onnx.export(model, torch.randn(1, 3, 224, 224), fp32, input_names=["input"], output_names=["logit"], opset_version=17, dynamo=False)
    onnx.save(float16.convert_float_to_float16(onnx.load(fp32), keep_io_types=True), fp16)
    quant_pre_process(fp32, pre)
    # Best int8 setting found: quantize only Conv/Gemm, entropy calibration on 300 images
    quantize_static(pre, int8, Calibration(args.data, n=300), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["Conv", "Gemm"],
                    calibrate_method=CalibrationMethod.Entropy)
    os.remove(pre)

    res = {}
    for name, path in [("fp32", fp32), ("fp16", fp16), ("int8", int8)]:
        res[name] = {"size_mb": os.path.getsize(path) / 1e6, "ms_per_image_1_cpu_thread": latency_ms(path), "test_accuracy": accuracy(path, args.data)}
    with open(os.path.join(args.out, "export_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
