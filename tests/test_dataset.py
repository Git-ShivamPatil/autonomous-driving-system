import numpy as np
import pytest

from ads.dataset import InverseFrequencyWeights


def test_weights_have_mean_one_on_training_labels_when_uncapped():
    labels = np.concatenate([np.zeros(900), np.full(100, 0.5)])
    w = InverseFrequencyWeights(labels)
    assert w.bin_weights.max() < 10.0
    assert w(labels).mean() == pytest.approx(1.0)


def test_rare_steering_gets_more_weight_than_straight():
    labels = np.concatenate([np.zeros(900), np.full(100, 0.5)])
    w = InverseFrequencyWeights(labels)
    assert w(0.5) > w(0.0)


def test_weights_are_flip_symmetric():
    labels = np.concatenate([np.full(50, -0.3), np.full(500, 0.3), np.zeros(1000)])
    w = InverseFrequencyWeights(labels)
    assert w(0.3) == pytest.approx(w(-0.3))


def test_weights_are_capped_and_mean_does_not_exceed_one():
    labels = np.concatenate([np.zeros(100_000), [0.95]])
    w = InverseFrequencyWeights(labels, max_weight=10.0)
    assert w(0.95) == pytest.approx(10.0)
    assert w.bin_weights.max() <= 10.0
    assert w(labels).mean() <= 1.0 + 1e-12


def test_out_of_range_steering_maps_to_edge_bins():
    w = InverseFrequencyWeights(np.linspace(-1, 1, 1000))
    assert w(1.5) == pytest.approx(w(1.0))
    assert w(-7.0) == pytest.approx(w(-1.0))
