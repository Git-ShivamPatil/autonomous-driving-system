"""Classic lane detection baseline: thresholding, perspective warp, sliding-window polynomial fit.

The output is a lane-line mask in the input image's frame, so it can be scored with the same IoU and
accuracy as the learned lane head. The warp is a fixed trapezoid given as fractions of the image size;
dashcams are mounted differently from car to car, which is one of the reasons this baseline is brittle.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LaneCVConfig:
    # Source trapezoid (fractions of width/height): bottom-left, top-left, top-right, bottom-right. The
    # bottom edge stays above the bonnet, which BDD100K's dashcams see in the lowest ~10% of the frame.
    src: tuple[tuple[float, float], ...] = ((0.12, 0.92), (0.43, 0.62), (0.57, 0.62), (0.88, 0.92))
    # Destination rectangle x positions (fractions of width) in the bird's-eye view.
    dst_x: tuple[float, float] = (0.25, 0.75)
    sobel_thresh: tuple[int, int] = (25, 255)
    sat_thresh: tuple[int, int] = (120, 255)
    white_l_min: int = 200
    n_windows: int = 9
    window_margin: float = 0.08  # fraction of width
    min_pixels: int = 40
    max_lines: int = 2  # histogram peaks to follow; 2 is the classic ego-lane pipeline
    min_peak_separation: float = 0.12  # fraction of width between followed peaks
    min_peak_fraction: float = 0.2  # a peak must reach this fraction of the highest one
    line_thickness: int = 8  # pixels, in the output frame


def _warp_matrices(w: int, h: int, cfg: LaneCVConfig) -> tuple[np.ndarray, np.ndarray]:
    src = np.float32([[x * w, y * h] for x, y in cfg.src])
    x0, x1 = cfg.dst_x[0] * w, cfg.dst_x[1] * w
    dst = np.float32([[x0, h], [x0, 0], [x1, 0], [x1, h]])
    return cv2.getPerspectiveTransform(src, dst), cv2.getPerspectiveTransform(dst, src)


def threshold(bgr: np.ndarray, cfg: LaneCVConfig) -> np.ndarray:
    """Binary map of likely lane-marking pixels: horizontal gradient OR saturated (yellow) OR bright (white)."""
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    lum, sat = hls[..., 1], hls[..., 2]
    sobel = np.abs(cv2.Sobel(lum, cv2.CV_64F, 1, 0, ksize=3))
    sobel = np.uint8(255 * sobel / max(sobel.max(), 1e-6))
    grad = (sobel >= cfg.sobel_thresh[0]) & (sobel <= cfg.sobel_thresh[1])
    yellow = (sat >= cfg.sat_thresh[0]) & (sat <= cfg.sat_thresh[1]) & (lum > 80)
    white = lum >= cfg.white_l_min
    return (grad | yellow | white).astype(np.uint8)


def histogram_peaks(histogram: np.ndarray, cfg: LaneCVConfig) -> list[int]:
    """Up to cfg.max_lines peak columns, strongest first, at least min_peak_separation apart."""
    w = len(histogram)
    if cfg.max_lines == 2:  # classic: the strongest column on each side of the centre
        mid = w // 2
        return [int(np.argmax(histogram[:mid])), int(np.argmax(histogram[mid:]) + mid)]
    histogram = histogram.astype(np.int64)
    sep = int(cfg.min_peak_separation * w)
    floor = cfg.min_peak_fraction * histogram.max()
    peaks: list[int] = []
    for x in np.argsort(-histogram, kind="stable"):
        if histogram[x] < floor or len(peaks) == cfg.max_lines:
            break
        if all(abs(int(x) - p) >= sep for p in peaks):
            peaks.append(int(x))
    return peaks


def sliding_window_fits(binary_warped: np.ndarray, cfg: LaneCVConfig) -> list[np.ndarray]:
    """Second-order fits x = a*y^2 + b*y + c, one per followed histogram peak (any may be missing)."""
    h, w = binary_warped.shape
    histogram = binary_warped[h // 2 :].sum(axis=0, dtype=np.int64)  # signed: histogram_peaks negates it
    bases = histogram_peaks(histogram, cfg)
    ys, xs = binary_warped.nonzero()
    window_h = h // cfg.n_windows
    margin = int(cfg.window_margin * w)
    fits = []
    for base in bases:
        if histogram[base] == 0:
            continue
        x_current, idx = base, []
        for k in range(cfg.n_windows):
            y_lo, y_hi = h - (k + 1) * window_h, h - k * window_h
            good = ((ys >= y_lo) & (ys < y_hi) & (xs >= x_current - margin) & (xs < x_current + margin)).nonzero()[0]
            idx.append(good)
            if len(good) > cfg.min_pixels:
                x_current = int(xs[good].mean())
        idx = np.concatenate(idx)
        if len(idx) >= 3 * cfg.min_pixels:
            fits.append(np.polyfit(ys[idx], xs[idx], 2))
    return fits


DEFAULT_CONFIG = LaneCVConfig()


def detect_lanes(bgr: np.ndarray, cfg: LaneCVConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Lane-line mask (uint8 0/1) in the frame of the input image."""
    h, w = bgr.shape[:2]
    m, m_inv = _warp_matrices(w, h, cfg)
    binary = threshold(bgr, cfg)
    warped = cv2.warpPerspective(binary, m, (w, h), flags=cv2.INTER_NEAREST)
    mask = np.zeros((h, w), dtype=np.uint8)
    plot_y = np.linspace(0, h - 1, h)
    for fit in sliding_window_fits(warped, cfg):
        x = np.polyval(fit, plot_y)
        pts = np.stack([x, plot_y], axis=1)
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w)]
        if len(pts) < 2:
            continue
        # Map the fitted curve back point by point and draw it in the image frame, the same way the
        # ground-truth lane masks are drawn; warping a raster line back would break it into dots.
        src = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), m_inv)
        cv2.polylines(mask, [np.round(src).astype(np.int32)], False, 1, thickness=cfg.line_thickness)
    return mask
