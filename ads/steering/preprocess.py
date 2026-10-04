"""PilotNet preprocessing and OpenCV augmentation.

Pure NumPy/OpenCV, so it imports without the simulator. Augmentations operate on the YUV network
input: brightness and shadow scale the Y (luma) channel, which is what a lighting change does.
"""

from __future__ import annotations

import cv2
import numpy as np

from ads import config


def preprocess(frame: np.ndarray, channel_order: str = config.CAMERA_CHANNEL_ORDER) -> np.ndarray:
    """Camera frame (160x320x3 uint8) -> network input (66x200x3 uint8, YUV)."""
    if frame.shape != (config.CAM_H, config.CAM_W, 3) or frame.dtype != np.uint8:
        raise ValueError(f"expected uint8 frame of shape {(config.CAM_H, config.CAM_W, 3)}, got {frame.dtype} {frame.shape}")
    crop = frame[config.CROP_TOP : config.CROP_BOTTOM]
    resized = cv2.resize(crop, (config.NET_W, config.NET_H), interpolation=cv2.INTER_AREA)
    if channel_order == "BGR":
        return cv2.cvtColor(resized, cv2.COLOR_BGR2YUV)
    if channel_order == "RGB":
        return cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)
    raise ValueError(f"unknown channel order {channel_order!r}")


def adjust_brightness(yuv: np.ndarray, factor: float) -> np.ndarray:
    out = yuv.copy()
    out[..., 0] = np.clip(yuv[..., 0].astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return out


def random_shadow(yuv: np.ndarray, rng: np.random.Generator, strength: tuple[float, float] = (0.4, 0.8)) -> np.ndarray:
    """Darken the region on one side of a random line running from the top edge to the bottom edge."""
    h, w = yuv.shape[:2]
    x_top, x_bottom = rng.uniform(0, w, size=2)
    side_x = 0 if rng.random() < 0.5 else w
    poly = np.array([[x_top, 0], [x_bottom, h], [side_x, h], [side_x, 0]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    out = yuv.copy()
    y = out[..., 0].astype(np.float32)
    y[mask == 1] *= rng.uniform(*strength)
    out[..., 0] = np.clip(y, 0, 255).astype(np.uint8)
    return out


def hflip(yuv: np.ndarray, steering: float) -> tuple[np.ndarray, float]:
    """Mirror the image; the correct steering for the mirrored scene is the negated steering."""
    return np.ascontiguousarray(yuv[:, ::-1]), -steering


def blur(yuv: np.ndarray, ksize: int) -> np.ndarray:
    return cv2.GaussianBlur(yuv, (ksize, ksize), 0)


def augment(yuv: np.ndarray, steering: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Training-time augmentation: brightness, random shadow, horizontal flip, blur."""
    if rng.random() < 0.5:
        yuv = adjust_brightness(yuv, rng.uniform(0.6, 1.4))
    if rng.random() < 0.3:
        yuv = random_shadow(yuv, rng)
    if rng.random() < 0.5:
        yuv, steering = hflip(yuv, steering)
    if rng.random() < 0.2:
        yuv = blur(yuv, int(rng.choice([3, 5])))
    return yuv, steering
