import cv2
import numpy as np

from ads.perception.lane_cv import LaneCVConfig, _warp_matrices, detect_lanes


def _synthetic_road(w=640, h=360):
    """A dark road whose two lane lines are straight and parallel in the bird's-eye view."""
    cfg = LaneCVConfig()
    _, m_inv = _warp_matrices(w, h, cfg)
    birdseye = np.zeros((h, w), dtype=np.uint8)
    for x in (int(0.32 * w), int(0.68 * w)):
        cv2.line(birdseye, (x, 0), (x, h - 1), 255, 6)
    lines = cv2.warpPerspective(birdseye, m_inv, (w, h), flags=cv2.INTER_NEAREST)
    img = np.full((h, w, 3), 60, dtype=np.uint8)
    img[lines > 0] = 235
    return img, (lines > 0).astype(np.uint8)


def test_detects_synthetic_lane_lines():
    img, truth = _synthetic_road()
    pred = detect_lanes(img)
    assert pred.shape == truth.shape and pred.dtype == np.uint8
    truth_thick = cv2.dilate(truth, np.ones((8, 8), np.uint8))
    inter = np.logical_and(pred, truth_thick).sum()
    union = np.logical_or(pred, truth_thick).sum()
    assert inter / union > 0.4


def test_multi_peak_search_handles_unsigned_histograms():
    from dataclasses import replace

    from ads.perception.lane_cv import histogram_peaks

    hist = np.zeros(640, dtype=np.uint64)
    hist[[100, 300, 500]] = [50, 80, 60]
    cfg = replace(LaneCVConfig(), max_lines=4)
    assert sorted(histogram_peaks(hist, cfg)) == [100, 300, 500]


def test_four_line_search_finds_the_synthetic_lines():
    from dataclasses import replace

    img, truth = _synthetic_road()
    pred = detect_lanes(img, replace(LaneCVConfig(), max_lines=4))
    assert np.logical_and(pred, cv2.dilate(truth, np.ones((8, 8), np.uint8))).sum() > 0


def test_blank_road_has_no_lanes():
    img = np.full((360, 640, 3), 60, dtype=np.uint8)
    assert detect_lanes(img).sum() == 0
