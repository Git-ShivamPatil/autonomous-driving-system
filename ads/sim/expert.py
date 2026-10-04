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


def expert_action(policy) -> tuple[float, float]:
    """(steering, acceleration), both clipped to [-1, 1]; IDMPolicy.act() returns them unclipped."""
    steering, accel = policy.act()
    return float(np.clip(steering, -1.0, 1.0)), float(np.clip(accel, -1.0, 1.0))
