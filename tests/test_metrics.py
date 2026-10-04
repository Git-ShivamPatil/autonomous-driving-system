import math

import numpy as np
import pytest

from ads.eval import metrics as m


def test_wilson_matches_reference_values():
    lo, hi = m.wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)
    lo, hi = m.wilson_interval(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = m.wilson_interval(10, 10)
    assert hi == pytest.approx(1.0)
    assert lo == pytest.approx(0.7225, abs=1e-4)


def test_wilson_rejects_bad_input():
    with pytest.raises(ValueError):
        m.wilson_interval(1, 0)
    with pytest.raises(ValueError):
        m.wilson_interval(11, 10)


def test_autonomy_percent():
    assert m.autonomy_percent(0, 600) == 100.0
    assert m.autonomy_percent(1, 600) == pytest.approx(99.0)
    assert m.autonomy_percent(1000, 600) == 0.0  # floored


def test_per_km():
    assert m.per_km(3, 1500) == pytest.approx(2.0)
    assert math.isnan(m.per_km(3, 0))


def test_rising_edges():
    assert m.rising_edges([]) == 0
    assert m.rising_edges([True, True, False, True, False, False, True]) == 3
    assert m.rising_edges([False, False]) == 0


def test_steering_jerk_is_zero_for_linear_and_constant_for_quadratic():
    dt = 0.1
    t = np.arange(50) * dt
    assert m.steering_jerk([0.5 * t], dt) == pytest.approx(0.0, abs=1e-9)
    assert m.steering_jerk([t**2], dt) == pytest.approx(2.0)
    # segments shorter than 3 contribute nothing; segments are not joined across a takeover
    assert m.steering_jerk([[0.0, 1.0], t**2], dt) == pytest.approx(2.0)
    assert math.isnan(m.steering_jerk([[0.0]], dt))


def test_abs_offset_stats():
    mean, p95 = m.abs_offset_stats([-1.0, 1.0, 0.0, 2.0])
    assert mean == pytest.approx(1.0)
    assert p95 == pytest.approx(np.percentile([1, 1, 0, 2], 95))


def test_split_mae():
    out = m.split_mae([0.1, 0.0, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.01], threshold=1e-4)
    assert out["mae"] == pytest.approx(0.2)
    assert out["mae_straight"] == pytest.approx(0.05)
    assert out["mae_curve"] == pytest.approx(0.5)
    assert (out["n_curve"], out["n_straight"]) == (1, 2)


def _episode(**kw):
    base = dict(
        seed=0,
        elapsed_s=100.0,
        distance_m=1000.0,
        interventions=0,
        collisions=0,
        out_of_road=False,
        arrive_dest=True,
        route_completion=1.0,
        lateral_abs_m=[0.1, 0.2],
        steering_segments=[[0.0, 0.0, 0.0]],
    )
    base.update(kw)
    return m.EpisodeResult(**base)


def test_success_requires_clean_arrival():
    assert _episode().success
    assert not _episode(interventions=1).success
    assert not _episode(collisions=1).success
    assert not _episode(out_of_road=True).success
    assert not _episode(arrive_dest=False).success


def test_line_touches_are_reported_per_km_and_do_not_fail_an_episode():
    eps = [_episode(line_touches=3), _episode(seed=1)]
    s = m.summarize(eps, dt=0.1)
    assert s["line_touches_per_km"] == pytest.approx(1.5)
    assert s["successes"] == 2


def test_summarize_pools_episodes():
    eps = [_episode(seed=0), _episode(seed=1, interventions=2, arrive_dest=False, route_completion=0.5)]
    s = m.summarize(eps, dt=0.1)
    assert s["episodes"] == 2 and s["successes"] == 1
    assert s["success_rate"] == 0.5
    assert s["interventions_per_km"] == pytest.approx(1.0)
    assert s["autonomy_pct"] == pytest.approx(100 * (1 - 2 * 6 / 200))
    assert s["route_completion_mean"] == pytest.approx(0.75)
    assert s["steering_jerk"] == pytest.approx(0.0)
