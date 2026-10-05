import numpy as np
import pytest
import torch

from ads import config
from ads.model import PilotNet


def test_keras_rebuild_matches_pytorch():
    pytest.importorskip("tensorflow")
    from ads.export.tflite import build_keras, copy_weights, torch_predict

    torch.manual_seed(0)
    model = PilotNet().eval()
    keras_model = build_keras()
    copy_weights(model, keras_model)
    rng = np.random.default_rng(1)
    images = rng.integers(0, 256, (8, config.NET_H, config.NET_W, 3)).astype(np.uint8)
    speeds = rng.uniform(0, 60, 8).astype(np.float32)
    k = keras_model.predict([images.astype(np.float32), speeds.reshape(-1, 1)], verbose=0).reshape(-1)
    assert np.abs(k - torch_predict(model, images, speeds)).max() < 1e-4
