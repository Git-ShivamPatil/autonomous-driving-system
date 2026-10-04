"""MetaDrive environment factory for the lane-keeping road suite.

MetaDrive is imported inside functions so that the rest of the package, and the unit tests, import
without the simulator. Only one MetaDrive engine can exist per process: use one env per process and
parallelise with processes.
"""

from __future__ import annotations

import numpy as np

from ads import config


def lane_keeping_block_dist():
    """A PGBlockDistConfig subclass that draws only curves, straights and on/off ramps."""
    from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig

    def remap(parent: dict) -> dict:
        missing = set(config.BLOCK_DISTRIBUTION) - set(parent)
        if missing:
            raise KeyError(f"MetaDrive has no block types {sorted(missing)}")
        return {name: config.BLOCK_DISTRIBUTION.get(name, 0.0) for name in parent}

    class LaneKeepingBlockDist(PGBlockDistConfig):
        BLOCK_TYPE_DISTRIBUTION_V1 = remap(PGBlockDistConfig.BLOCK_TYPE_DISTRIBUTION_V1)
        BLOCK_TYPE_DISTRIBUTION_V2 = remap(PGBlockDistConfig.BLOCK_TYPE_DISTRIBUTION_V2)

    return LaneKeepingBlockDist


def env_config(split: str, camera: bool = True, eval_mode: bool = False, **overrides) -> dict:
    seeds = config.SPLITS[split]
    cfg = dict(
        start_seed=seeds.start,
        num_scenarios=len(seeds),
        map=config.MAP_BLOCKS,
        block_dist_config=lane_keeping_block_dist(),
        random_lane_width=True,
        random_lane_num=True,
        traffic_density=config.TRAFFIC_DENSITY,
        physics_world_step_size=config.PHYSICS_DT,
        decision_repeat=config.DECISION_REPEAT,
        horizon=config.EPISODE_HORIZON,
        use_render=False,
        log_level=50,
    )
    if camera:
        from metadrive.component.sensors.rgb_camera import RGBCamera

        cfg.update(
            image_observation=True,
            norm_pixel=False,
            stack_size=1,
            window_size=(config.CAM_W, config.CAM_H),
            sensors={"rgb_camera": (RGBCamera, config.CAM_W, config.CAM_H)},
            vehicle_config=dict(image_source="rgb_camera"),
        )
    if eval_mode:
        # Keep driving after a collision so collisions can be counted per km.
        cfg.update(crash_vehicle_done=False, crash_object_done=False)
    cfg.update(overrides)
    return cfg


def make_env(split: str, camera: bool = True, eval_mode: bool = False, **overrides):
    from metadrive.envs.metadrive_env import MetaDriveEnv

    return MetaDriveEnv(env_config(split, camera=camera, eval_mode=eval_mode, **overrides))


def camera_frame(obs) -> np.ndarray:
    """Latest camera frame (H, W, 3) uint8 from an image observation."""
    return obs["image"][..., -1]


def lane_state(agent) -> tuple[float, float]:
    """(lateral offset from the current lane centre in m, signed curvature in 1/m; 0 on straights)."""
    lane = agent.lane
    _, lateral = lane.local_coordinates(agent.position)
    curvature = 0.0
    radius = getattr(lane, "radius", None)
    if radius:
        curvature = float(getattr(lane, "direction", 1)) / float(radius)
    return float(lateral), curvature


def speed_kmh(agent) -> float:
    return float(agent.speed_km_h)
