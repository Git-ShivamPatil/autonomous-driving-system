import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("variants_table", Path(__file__).parent.parent / "tools" / "variants_table.py")
vt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vt)


def test_ties_get_average_ranks():
    assert vt.ranks([3.0, 1.0, 3.0, 2.0]).tolist() == [2.5, 0.0, 2.5, 1.0]


def test_spearman_perfect_and_reversed():
    assert vt.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert vt.spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
