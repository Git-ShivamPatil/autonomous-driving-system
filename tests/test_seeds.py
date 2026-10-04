from itertools import combinations

from ads import config


def test_split_sizes():
    assert len(config.TRAIN_SEEDS) == 8000
    assert len(config.VAL_SEEDS) == 200
    assert len(config.TEST_SEEDS) == 1000


def test_test_split_is_exactly_seeds_0_to_999():
    assert list(config.TEST_SEEDS) == list(range(1000))


def test_splits_are_pairwise_disjoint():
    for (a, ra), (b, rb) in combinations(config.SPLITS.items(), 2):
        assert not set(ra) & set(rb), f"{a} and {b} overlap"


def test_block_distribution_is_a_distribution_without_junctions():
    assert abs(sum(config.BLOCK_DISTRIBUTION.values()) - 1.0) < 1e-9
    for junction in ("StdInterSection", "StdTInterSection", "Roundabout"):
        assert junction not in config.BLOCK_DISTRIBUTION
