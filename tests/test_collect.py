import numpy as np
import pytest

from ads import config
from ads.sim import collect


def test_episode_sigma_is_deterministic_and_from_the_allowed_set():
    sigmas = [collect.episode_sigma(s) for s in config.TRAIN_SEEDS[:400]]
    assert sigmas == [collect.episode_sigma(s) for s in config.TRAIN_SEEDS[:400]]
    assert set(sigmas) == set(config.DART_SIGMAS)


def test_ar1_noise_has_the_requested_stationary_std():
    sigma, rho = 0.1, config.DART_RHO
    innov = collect.ar1_innovation_std(sigma, rho)
    rng = np.random.default_rng(0)
    e, xs = 0.0, []
    for _ in range(200_000):
        e = rho * e + innov * rng.standard_normal()
        xs.append(e)
    xs = np.array(xs[1000:])
    assert xs.std() == pytest.approx(sigma, rel=0.03)
    assert np.corrcoef(xs[:-1], xs[1:])[0, 1] == pytest.approx(rho, abs=0.01)


def test_collect_module_does_not_import_the_simulator():
    import sys

    assert "metadrive" not in sys.modules
