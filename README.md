# shoe-check: is this sneaker photo sharp enough?

Phone photos of sneakers are often blurry: the camera focused on the floor, or the hand moved. Before a photo is used for anything detailed (like checking stitching or texture), it should pass a quick quality gate. **shoe-check** flags blurry and out-of-focus sneaker photos, and compares classic computer vision against a small CNN that's light enough to run on a device.

## Results so far (synthetic blur, held-out test set of 1,180 images)

| Method | Accuracy | Blurry F1 | ROC-AUC | Speed (CPU) |
|---|---|---|---|---|
| Laplacian variance (one sharpness score) | 98.9% | 0.989 | 0.999 | ~2.5 ms for all 3 scores |
| Tenengrad (gradient energy) | 85.8% | 0.857 | 0.934 | |
| FFT high-frequency energy | 84.4% | 0.836 | 0.926 | |
| Logistic regression on all 3 scores | 99.2% | 0.992 | 1.000 | ~2.5 ms |
| **MobileNetV3-Small (fine-tuned)** | **99.9%** | **0.999** | **1.000** | 5.2 ms (1 thread) |

**Model size and speed (ONNX Runtime, 1 CPU thread):**

| Version | Size | Speed | Test accuracy |
|---|---|---|---|
| float32 | 6.1 MB | 5.2 ms | 99.9% |
| **float16** (used by the demo) | **3.1 MB** | 5.2 ms | **99.9%** |
| int8 (post-training quantization) | 1.8 MB | 2.5 ms | 89.6% |

int8 halves the latency but costs ~10 points of accuracy. MobileNetV3's hard-swish activations and squeeze-and-excitation layers are known to be sensitive to post-training quantization. The next step would be quantization-aware training; until then, float16 gives half the size with no loss.

> **Synthetic blur is the easy case**, so every method scores ~99% here. On **real** camera blur the CNN holds up (95.6%) and the classic scores drop to ~88%; see *Real-world evaluation* below.

## Real-world evaluation
Every model above was trained **only on synthetic blur**. These tests use real camera blur it never saw.

### 1. Blur Dataset: 1,050 real photos from 66 phones and cameras
[Blur Dataset](https://www.kaggle.com/datasets/kwentar/blur-dataset) (CC0): 350 sharp, 350 out-of-focus, and 350 motion-blurred photos of everyday scenes (not shoes).

| Method | Accuracy | Out-of-focus caught | Motion blur caught | Sharp kept | ROC-AUC |
|---|---|---|---|---|---|
| Laplacian variance, whole photo | 87.1% | | | | 0.954 |
| Logistic regression (3 scores), whole photo | 88.7% | 96.6% | 72.0% | 97.4% | 0.968 |
| Logistic regression, centre crop | 86.9% | 96.0% | 67.1% | 97.4% | 0.966 |
| Logistic regression, main object only (GrabCut) | 83.8% | 93.7% | 59.4% | 98.3% | 0.964 |
| **MobileNetV3-Small, float16** | **95.6%** | **98.6%** | **92.3%** | **96.0%** | **0.989** |

### 2. Phone photos of real sneakers: 30 photos, 3 pairs of shoes
17 sharp and 13 blurry (out of focus, strong and slight motion blur), on different floors and lighting. Labels were set by eye from the full-resolution photos. One photo first labelled sharp was re-checked by the photographer after all methods disagreed, and was confirmed blurry.

| Method | Accuracy | Blurry caught | Sharp kept | ROC-AUC |
|---|---|---|---|---|
| Logistic regression, whole photo | 93.3% | 11 / 13 | 17 / 17 | 0.941 |
| Logistic regression, centre crop | 90.0% | 10 / 13 | 17 / 17 | 0.950 |
| Logistic regression, shoe only (GrabCut) | 86.7% | 9 / 13 | 17 / 17 | 0.991 |
| **MobileNetV3-Small, float16** | **96.7%** | **12 / 13** | **17 / 17** | **1.000** |

### What this shows
- **The CNN transfers from synthetic to real blur far better than the classic scores:** 95.6% vs 88.7% on real camera photos.
- **Motion blur is where classic sharpness scores fail.** They catch 96% of out-of-focus photos but only 72% of motion-blurred ones. A blur streak along one direction still leaves sharp edges in the other direction, which keeps the Laplacian variance high.
- **Scoring only the shoe ranks photos best** among the classic variants on the sneaker photos (ROC-AUC 0.991), but with the threshold tuned on synthetic data it lets more blurry photos through. The threshold needs re-tuning on real photos.
- **The CNN's hardest case is *slight* motion blur:** its only miss on the sneaker photos was a pair that's only a little smeared.

## How it works
1. **Data** (`src/prepare_data.py`): 3,930 sneaker photos from the [Sneakers dataset](https://huggingface.co/datasets/ipogorelov/sneakers) (MIT). Each photo becomes one sharp and one blurred version, and the 70/15/15 split is done **by source photo**, so the same shoe never appears in two splits. The softest 10% of source photos were dropped, since scraped photos aren't guaranteed sharp. Working resolution is 256 px.
2. **Realistic blur** (`src/degrade.py`): out-of-focus blur as a disk (lens aperture) kernel, camera-shake blur as a line kernel at a random angle, and some Gaussian blur. **Both** classes also get random exposure, sensor noise and JPEG compression, so a model can't separate them by compression or brightness alone.
3. **Classic sharpness scores** (`src/sharpness.py`): variance of the Laplacian, Tenengrad (Sobel gradient energy), and FFT high-frequency energy. Each has a threshold tuned on validation, plus a logistic regression on all three (`src/classic.py`).
4. **CNN** (`src/train_cnn.py`): ImageNet MobileNetV3-Small fine-tuned with crops and flips only. Resizing or blurring augmentations would change the very thing being classified.
5. **Export** (`src/export.py`): ONNX float32, float16 and int8, each timed on a single CPU thread.
6. **Demo** (`app.py`): Gradio app. Upload a photo and get a verdict from both methods, plus the shoe region that was scored.

## Run it
```bash
pip install -r requirements.txt
# download a few batches of https://huggingface.co/datasets/ipogorelov/sneakers into data/raw/
python src/prepare_data.py --parquet data/raw/*.parquet --out data/synthetic
python src/classic.py --data data/synthetic --out results
python src/train_cnn.py --data data/synthetic --out results
python src/export.py --data data/synthetic --out results
python src/evaluate_real.py --sharp data/real/sharp --blurry data/real/blurry --name phone_shoes
python src/evaluate_real.py --sharp data/kaggle/sharp --blurry data/kaggle/defocused_blurred data/kaggle/motion_blurred --name kaggle_blur
python app.py
```
The trained float16 model (`results/shoe_check_fp16.onnx`) and classic models (`results/classic_models.pkl`) are included, so `python app.py` works without retraining.
