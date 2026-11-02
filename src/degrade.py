"""Realistic phone-camera degradations: defocus and motion blur for the 'blurry' class,
and blur-free nuisances (JPEG, noise, lighting) applied to both classes so a model can't
tell them apart by compression or brightness alone."""
import cv2
import numpy as np


def defocus_kernel(radius):
    """Out-of-focus blur is close to a uniform disk (the lens aperture), not a Gaussian."""
    size = int(2 * np.ceil(radius) + 1)
    y, x = np.mgrid[:size, :size] - size // 2
    k = (x ** 2 + y ** 2 <= radius ** 2).astype(np.float32)
    return k / k.sum()


def motion_kernel(length, angle_deg):
    """Camera shake: a straight line of the given length and angle."""
    size = int(length) | 1
    k = np.zeros((size, size), np.float32)
    k[size // 2, :] = 1
    rot = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, rot, (size, size))
    return k / max(k.sum(), 1e-6)


def blur(img, rng):
    """Returns the blurred image and a short description of the blur applied."""
    kind = rng.choice(["defocus", "motion", "gaussian"], p=[0.45, 0.4, 0.15])
    if kind == "defocus":
        r = rng.uniform(1.5, 5.0)
        return cv2.filter2D(img, -1, defocus_kernel(r)), f"defocus r={r:.1f}"
    if kind == "motion":
        length, angle = rng.uniform(5, 17), rng.uniform(0, 180)
        return cv2.filter2D(img, -1, motion_kernel(length, angle)), f"motion len={length:.0f}"
    sigma = rng.uniform(1.2, 3.0)
    return cv2.GaussianBlur(img, (0, 0), sigma), f"gaussian s={sigma:.1f}"


def nuisance(img, rng):
    """Changes that don't affect sharpness: exposure, sensor noise, JPEG compression."""
    img = img.astype(np.float32)
    img = img * rng.uniform(0.7, 1.25) + rng.uniform(-20, 20)
    if rng.random() < 0.5:
        img = img + rng.normal(0, rng.uniform(1, 4), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 96))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
