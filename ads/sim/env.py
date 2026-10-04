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
        # MetaDrive's default ends the episode as soon as the chassis touches a solid line, which on most
        # random lane widths happens before |lateral| reaches the 1.2 m takeover threshold. With it off,
        # out_of_road means the vehicle centre has left the lane surface; line touches are counted instead.
        on_continuous_line_done=False,
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


def set_camera_rendering(env, active: bool) -> None:
    """Render the camera on the next engine step, or skip it.

    A skipped step's observation holds the previous frame. Data collection saves every SAVE_EVERY-th
    step, so it renders only those; tools/frame_lag_check.py --every verifies the saved frames are
    identical to fully rendered ones.
    """
    if env.engine is None:  # MetaDrive starts its engine on the first reset, with the camera active
        return
    env.engine.get_sensor("rgb_camera").buffer.setActive(active)


def on_solid_line(agent) -> bool:
    """Whether the chassis is touching a solid (white or yellow) lane line."""
    return bool(agent.on_white_continuous_line or agent.on_yellow_continuous_line)
