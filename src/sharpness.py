"""Classical sharpness scores. Higher = sharper. All work on a grayscale image.

Phone photos often have a deliberately blurred background (portrait mode), so each score can be
computed on the whole frame, on a centre crop, or on the shoe itself (found with GrabCut)."""
import cv2
import numpy as np


def resize_short(img, size):
    h, w = img.shape[:2]
    s = size / min(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def laplacian_var(g, mask=None):
    """Variance of the Laplacian: sharp edges give a strong, varied second-derivative response."""
    lap = cv2.Laplacian(g, cv2.CV_64F)
    return float(lap[mask].var() if mask is not None else lap.var())


def tenengrad(g, mask=None):
    """Mean squared Sobel gradient magnitude (Tenengrad focus measure)."""
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    mag = gx ** 2 + gy ** 2
    return float(mag[mask].mean() if mask is not None else mag.mean())


def fft_highfreq(g, mask=None, cutoff=0.15):
    """Share of spectral energy above a cutoff frequency: blur removes high frequencies.
    The mask is ignored here: the spectrum is taken over the bounding box of the region."""
    if mask is not None:
        ys, xs = np.where(mask)
        g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    f = np.fft.fftshift(np.fft.fft2(g.astype(np.float64) - g.mean()))
    power = np.abs(f) ** 2
    h, w = g.shape
    yy, xx = np.ogrid[-(h // 2):h - h // 2, -(w // 2):w - w // 2]
    r = np.sqrt((yy / (h / 2)) ** 2 + (xx / (w / 2)) ** 2)
    return float(power[r > cutoff].sum() / (power.sum() + 1e-9))


def center_mask(g, frac=0.6):
    h, w = g.shape
    m = np.zeros_like(g, dtype=bool)
    dh, dw = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
    m[dh:h - dh, dw:w - dw] = True
    return m


def shoe_mask(img, iters=3):
    """Foreground (the shoe) via GrabCut, started from a box covering the central 80% of the frame.
    Falls back to the centre crop if GrabCut finds almost nothing."""
    h, w = img.shape[:2]
    rect = (int(0.1 * w), int(0.1 * h), int(0.8 * w), int(0.8 * h))
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
        fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    except cv2.error:
        fg = np.zeros((h, w), bool)
    if fg.mean() < 0.05:
        return center_mask(gray(img))
    # Score the shoe's edges too, not just its interior
    return cv2.dilate(fg.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)


def features(img, region="full"):
    """[log Laplacian variance, log Tenengrad, high-frequency energy] for one image."""
    g = gray(img)
    mask = None if region == "full" else center_mask(g) if region == "center" else shoe_mask(img)
    return np.array([np.log1p(laplacian_var(g, mask)), np.log1p(tenengrad(g, mask)), fft_highfreq(g, mask)])
