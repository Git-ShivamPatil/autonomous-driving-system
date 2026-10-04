import numpy as np
import pytest

from ads import config
from ads.steering import preprocess as pp


def _frame():
    """White outside the crop band, black inside it."""
    f = np.full((config.CAM_H, config.CAM_W, 3), 255, dtype=np.uint8)
    f[config.CROP_TOP : config.CROP_BOTTOM] = 0
    return f


def test_preprocess_shape_dtype_and_crop():
    out = pp.preprocess(_frame())
    assert out.shape == (config.NET_H, config.NET_W, 3)
    assert out.dtype == np.uint8
    # Only the black band survives the crop, so luma is 0 everywhere.
    assert out[..., 0].max() == 0


def test_preprocess_rejects_wrong_shape():
    with pytest.raises(ValueError):
        pp.preprocess(np.zeros((10, 10, 3), dtype=np.uint8))


def test_channel_order_matters():
    f = np.zeros((config.CAM_H, config.CAM_W, 3), dtype=np.uint8)
    f[..., 0] = 255  # pure blue if BGR, pure red if RGB
    y_bgr = pp.preprocess(f, "BGR")[..., 0].mean()
    y_rgb = pp.preprocess(f, "RGB")[..., 0].mean()
    assert y_bgr < y_rgb  # blue has much lower luma than red


def test_hflip_negates_steering_and_is_an_involution():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (config.NET_H, config.NET_W, 3), dtype=np.uint8)
    flipped, s = pp.hflip(img, 0.3)
    assert s == -0.3
    assert np.array_equal(flipped[:, 0], img[:, -1])
    back, s2 = pp.hflip(flipped, s)
    assert np.array_equal(back, img) and s2 == 0.3


def test_brightness_scales_only_luma():
    img = np.full((4, 4, 3), 100, dtype=np.uint8)
    out = pp.adjust_brightness(img, 1.5)
    assert (out[..., 0] == 150).all()
    assert (out[..., 1:] == 100).all()
    assert np.array_equal(pp.adjust_brightness(img, 1.0), img)


def test_shadow_only_darkens():
    rng = np.random.default_rng(1)
    img = np.full((config.NET_H, config.NET_W, 3), 200, dtype=np.uint8)
    out = pp.random_shadow(img, rng)
    assert (out[..., 0] <= 200).all()
    assert (out[..., 1:] == 200).all()


def test_augment_preserves_shape_and_flip_sign():
    rng = np.random.default_rng(2)
    img = np.random.default_rng(3).integers(0, 256, (config.NET_H, config.NET_W, 3), dtype=np.uint8)
    for _ in range(50):
        out, s = pp.augment(img, 0.25, rng)
        assert out.shape == img.shape and out.dtype == np.uint8
        assert s in (0.25, -0.25)
