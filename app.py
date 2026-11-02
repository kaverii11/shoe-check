"""Gradio demo: upload a sneaker photo, get sharp / blurry from the classical scorer and the CNN.

Usage: python app.py   (needs results/classic_models.pkl and results/shoe_check_fp16.onnx)
"""
import os
import pickle
import sys

import cv2
import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from sharpness import features, resize_short, shoe_mask  # noqa: E402
from train_cnn import EVAL_TF  # noqa: E402

HERE = os.path.dirname(__file__)
CLASSIC = pickle.load(open(os.path.join(HERE, "results", "classic_models.pkl"), "rb"))
CNN = ort.InferenceSession(os.path.join(HERE, "results", "shoe_check_fp16.onnx"), providers=["CPUExecutionProvider"])


def check(photo):
    img = resize_short(cv2.cvtColor(np.array(ImageOps.exif_transpose(photo).convert("RGB")), cv2.COLOR_RGB2BGR), 256)
    p_classic = float(CLASSIC["logreg"].predict_proba(features(img, "shoe")[None])[0, 1])
    x = EVAL_TF(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).unsqueeze(0).numpy()
    p_cnn = float(1 / (1 + np.exp(-CNN.run(None, {"input": x})[0].item())))

    mask = shoe_mask(img)
    overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    overlay[~mask] = (overlay[~mask] * 0.35).astype(np.uint8)  # dim everything except the region that was scored

    verdict = lambda p: f"{'BLURRY' if p >= 0.5 else 'SHARP'} (blur probability {p:.2f})"
    return verdict(p_classic), verdict(p_cnn), overlay


demo = gr.Interface(
    fn=check,
    inputs=gr.Image(type="pil", label="Sneaker photo"),
    outputs=[gr.Textbox(label="Classical sharpness scores (on the shoe)"), gr.Textbox(label="MobileNetV3-Small, float16"), gr.Image(label="Region scored")],
    title="shoe-check: is this sneaker photo sharp enough?",
)

if __name__ == "__main__":
    demo.launch()
