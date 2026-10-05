"""The expert driver: MetaDrive's IDM policy with lane changes disabled.

The env keeps its default input policy, so actions are passed to env.step() explicitly. The IDM
policy is constructed after every env.reset(), because reset replaces the agent.
"""

from __future__ import annotations

import numpy as np


def make_expert(env, seed: int):
    from metadrive.policy.idm_policy import IDMPolicy

    policy = IDMPolicy(env.agent, seed)
    policy.enable_lane_change = False
    return policy


def reset_integral(policy) -> None:
    """Zero the integral terms of the expert's steering PIDs.

    MetaDrive's PID integral accumulates without bound. When the expert is queried but not obeyed (the
    model is steering), the accumulated error is the model's history, not the expert's, and it grew to
    53-77% of the expert's output on validation roads (tools/expert_terms.py). Zeroing it before each
    query while the model steers makes the expert a PD reference for the current state; when the
    expert takes over, its integral then starts from zero and accumulates normally.
    """
    policy.heading_pid.i_error = 0.0
    policy.lateral_pid.i_error = 0.0


def expert_action(policy) -> tuple[float, float]:
    """(steering, acceleration), both clipped to [-1, 1]; IDMPolicy.act() returns them unclipped."""
    steering, accel = policy.act()
    return float(np.clip(steering, -1.0, 1.0)), float(np.clip(accel, -1.0, 1.0))
