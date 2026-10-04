"""Check whether the camera frame lags the physics state under multi-threaded rendering.

Drives the same seed with the same actions twice, once per rendering mode (one engine per process),
and saves the frames. `--compare` then reports, for each mode pair, whether frame k of one run equals
frame k or frame k-1 of the other. Single-threaded rendering is the reference: its frame is drawn
synchronously after the physics step.

    python tools/frame_lag_check.py --mt 1 --out runs/lag_mt1.npz
    python tools/frame_lag_check.py --mt 0 --out runs/lag_mt0.npz
    python tools/frame_lag_check.py --compare runs/lag_mt1.npz runs/lag_mt0.npz

With --every 2 the run renders only every second frame, as data collection does; comparing it with a
fully rendered run at multiples of 2 checks that skipping renders does not change the saved frames:

    python tools/frame_lag_check.py --mt 1 --every 2 --out runs/lag_every2.npz
    python tools/frame_lag_check.py --compare runs/lag_every2.npz runs/lag_mt1.npz --every 2
"""

from __future__ import annotations

import argparse

import numpy as np


def record(mt: bool, out: str, steps: int, every: int) -> None:
    from ads.sim.env import camera_frame, make_env, set_camera_rendering

    env = make_env("train", multi_thread_render=mt)
    obs, _ = env.reset(seed=1000)
    frames = [camera_frame(obs).copy()]
    for i in range(steps):
        steer = 0.3 * np.sin(i / 5.0)  # varying steering so consecutive frames differ
        set_camera_rendering(env, (i + 1) % every == 0)
        obs, *_ = env.step([steer, 0.6])
        frames.append(camera_frame(obs).copy())
    env.close()
    np.savez_compressed(out, frames=np.stack(frames))


def compare(a_path: str, b_path: str, every: int) -> None:
    a, b = np.load(a_path)["frames"].astype(np.int16), np.load(b_path)["frames"].astype(np.int16)
    n = min(len(a), len(b))
    stride = max(every, (n // 8) // every * every)
    for k in range(every * 2, n, stride):
        same = np.abs(a[k] - b[k]).mean()
        prev = np.abs(a[k] - b[k - 1]).mean()
        nxt = np.abs(a[k - 1] - b[k]).mean()
        print(f"step {k:3d}: |A[k]-B[k]| {same:6.2f}   |A[k]-B[k-1]| {prev:6.2f}   |A[k-1]-B[k]| {nxt:6.2f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mt", type=int)
    p.add_argument("--out")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--every", type=int, default=1, help="render only every N-th frame (frames in between are stale)")
    p.add_argument("--compare", nargs=2)
    args = p.parse_args()
    if args.compare:
        compare(*args.compare, args.every)
    else:
        record(bool(args.mt), args.out, args.steps, args.every)


if __name__ == "__main__":
    main()
