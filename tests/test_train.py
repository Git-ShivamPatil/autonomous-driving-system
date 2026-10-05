import numpy as np

from ads.train import select_training_indices

SIGMA = np.array([0.0, 0.05, 0.0, 0.15, 0.1, 0.0, 0.05, 0.0])


def test_no_filters_keeps_everything():
    assert select_training_indices(SIGMA, None, None, None, 0).tolist() == list(range(8))


def test_sigma_cap_selects_noise_free_episodes():
    assert select_training_indices(SIGMA, 0.0, None, None, 0).tolist() == [0, 2, 5, 7]


def test_subsample_is_seeded_sorted_and_without_replacement():
    a = select_training_indices(SIGMA, None, 4, None, seed=3)
    b = select_training_indices(SIGMA, None, 4, None, seed=3)
    assert a.tolist() == b.tolist() == sorted(set(a.tolist()))
    assert len(a) == 4


def test_subsample_larger_than_pool_keeps_the_pool():
    assert select_training_indices(SIGMA, 0.0, 100, None, 0).tolist() == [0, 2, 5, 7]
