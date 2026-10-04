"""Closed-loop and open-loop driving metrics. Pure NumPy: no simulator or torch import.

Definitions used throughout:

* Intervention: the safety driver takes over for TAKEOVER_SECONDS whenever |lateral offset| exceeds
  TAKEOVER_LATERAL_M while the model is steering. Each takeover counts once.
* Autonomy (Bojarski et al. 2016): 1 - interventions x 6 s / elapsed time, floored at 0, pooled over
  all episodes.
* Success: the episode reaches its destination with no intervention, no collision and no
  out-of-road termination.
* Steering jerk: mean |d^2 steering / dt^2| over the steps the model steered, in normalised
  steering units per s^2, computed within each uninterrupted model-controlled segment.
* Open-loop / on-policy gap: MAE against the expert on the model's own (closed-loop) states divided
  by MAE on expert-driven states. A ratio above 1 measures the covariate shift that open-loop
  scores do not see.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def autonomy_percent(interventions: int, elapsed_s: float, penalty_s: float = 6.0) -> float:
    if elapsed_s <= 0:
        raise ValueError("elapsed_s must be positive")
    return max(0.0, 100.0 * (1.0 - interventions * penalty_s / elapsed_s))


def per_km(count: float, distance_m: float) -> float:
    if distance_m <= 0:
        return float("nan")
    return count / (distance_m / 1000.0)


def rising_edges(flags: Iterable[bool]) -> int:
    """Number of False->True transitions; a flag that starts True counts once."""
    count, prev = 0, False
    for f in flags:
        f = bool(f)
        if f and not prev:
            count += 1
        prev = f
    return count


def steering_jerk(segments: Iterable[Sequence[float]], dt: float) -> float:
    """Mean |second difference| / dt^2 pooled over segments; segments shorter than 3 steps are skipped."""
    total, n = 0.0, 0
    for seg in segments:
        s = np.asarray(seg, dtype=np.float64)
        if s.size < 3:
            continue
        d2 = np.abs(np.diff(s, n=2)) / (dt * dt)
        total += float(d2.sum())
        n += d2.size
    return total / n if n else float("nan")


def abs_offset_stats(lateral_m: Iterable[float]) -> tuple[float, float]:
    """Mean and 95th percentile of |lateral offset|."""
    a = np.abs(np.asarray(list(lateral_m), dtype=np.float64))
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(np.percentile(a, 95))


def split_mae(pred: Sequence[float], target: Sequence[float], curvature: Sequence[float], threshold: float) -> dict:
    """MAE overall and split into curve (|curvature| > threshold) and straight frames."""
    p, t, c = (np.asarray(x, dtype=np.float64) for x in (pred, target, curvature))
    if not (p.shape == t.shape == c.shape):
        raise ValueError("pred, target and curvature must have the same shape")
    err = np.abs(p - t)
    curve = np.abs(c) > threshold

    def mae(mask: np.ndarray) -> float:
        return float(err[mask].mean()) if mask.any() else float("nan")

    return {
        "mae": mae(np.ones_like(curve)),
        "mae_curve": mae(curve),
        "mae_straight": mae(~curve),
        "n": int(err.size),
        "n_curve": int(curve.sum()),
        "n_straight": int((~curve).sum()),
    }


@dataclass
class EpisodeResult:
    seed: int
    elapsed_s: float
    distance_m: float
    interventions: int
    collisions: int
    out_of_road: bool
    arrive_dest: bool
    route_completion: float
    line_touches: int = 0  # chassis touching a solid lane line, counted as rising edges
    lateral_abs_m: list[float] = field(default_factory=list)  # every step the model was steering
    steering_segments: list[list[float]] = field(default_factory=list)  # model steering, split at takeovers

    @property
    def success(self) -> bool:
        return self.arrive_dest and self.interventions == 0 and self.collisions == 0 and not self.out_of_road


def summarize(episodes: Sequence[EpisodeResult], dt: float, penalty_s: float = 6.0) -> dict:
    """Aggregate closed-loop metrics over a set of episodes."""
    n = len(episodes)
    if n == 0:
        raise ValueError("no episodes")
    k = sum(e.success for e in episodes)
    lo, hi = wilson_interval(k, n)
    elapsed = sum(e.elapsed_s for e in episodes)
    distance = sum(e.distance_m for e in episodes)
    interventions = sum(e.interventions for e in episodes)
    collisions = sum(e.collisions for e in episodes)
    lat_mean, lat_p95 = abs_offset_stats(x for e in episodes for x in e.lateral_abs_m)
    return {
        "episodes": n,
        "successes": k,
        "success_rate": k / n,
        "success_ci95": [lo, hi],
        "route_completion_mean": float(np.mean([e.route_completion for e in episodes])),
        "arrive_rate": sum(e.arrive_dest for e in episodes) / n,
        "distance_km": distance / 1000.0,
        "elapsed_h": elapsed / 3600.0,
        "interventions": interventions,
        "interventions_per_km": per_km(interventions, distance),
        "autonomy_pct": autonomy_percent(interventions, elapsed, penalty_s),
        "collisions": collisions,
        "collisions_per_km": per_km(collisions, distance),
        "out_of_road_rate": sum(e.out_of_road for e in episodes) / n,
        "line_touches_per_km": per_km(sum(e.line_touches for e in episodes), distance),
        "lateral_abs_mean_m": lat_mean,
        "lateral_abs_p95_m": lat_p95,
        "steering_jerk": steering_jerk((s for e in episodes for s in e.steering_segments), dt),
    }
