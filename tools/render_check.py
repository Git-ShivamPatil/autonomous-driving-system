"""Render one camera frame from the lane-keeping road suite, save it, and time stepping with the camera on.

    python tools/render_check.py --out docs/media/first_frame.jpg --steps 200

Reports the OpenGL renderer in use, so it shows which GPU MetaDrive is drawing with.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from ads import config, provenance
from ads.sim.env import camera_frame, lane_state, make_env, speed_kmh


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--split", default="train")
    p.add_argument("--camera", type=int, default=1, help="0 to time physics without rendering")
    p.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="MetaDrive config override, e.g. show_terrain=false"
    )
    args = p.parse_args()
    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value

    t0 = time.perf_counter()
    env = make_env(args.split, camera=bool(args.camera), **overrides)
    seed = config.SPLITS[args.split].start
    obs, _ = env.reset(seed=seed)
    report = {
        "overrides": overrides,
        "env_create_reset_s": round(time.perf_counter() - t0, 1),
        "seed": seed,
    }
    if args.camera:
        frame = camera_frame(obs)
        report["frame_shape"] = list(frame.shape)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out), np.ascontiguousarray(frame), [cv2.IMWRITE_JPEG_QUALITY, 95])
        gsg = env.engine.win.getGsg()
        report["gl_renderer"] = f"{gsg.getDriverVendor()} | {gsg.getDriverRenderer()} | {gsg.getDriverVersion()}"
    # Resets rebuild the map and terrain and cost seconds; they are timed separately from stepping.
    step_time, reset_time, resets = 0.0, 0.0, 0
    for _ in range(args.steps):
        t = time.perf_counter()
        obs, _r, term, trunc, _info = env.step([0.0, 0.3])
        step_time += time.perf_counter() - t
        if term or trunc:
            t = time.perf_counter()
            env.reset(seed=seed)
            reset_time += time.perf_counter() - t
            resets += 1
    report["steps_per_s"] = round(args.steps / step_time, 2)
    report["resets"] = resets
    report["seconds_per_reset"] = round(reset_time / resets, 2) if resets else None
    report["lane_state"] = lane_state(env.agent)
    report["speed_kmh"] = round(speed_kmh(env.agent), 2)
    env.close()
    report["provenance"] = provenance.stamp()
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
