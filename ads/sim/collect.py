"""DART data collection (Laskey et al. 2017) on the lane-keeping road suite.

    python -m ads.sim.collect --split train --target 75000 --workers 2

Executed steering = expert steering + AR(1) noise (rho 0.9; per-episode sigma drawn from DART_SIGMAS
by the seed). The recorded label is the expert's clean steering in the state actually visited, so the
car drifts and the labels teach the recovery. Every SAVE_EVERY-th step is saved; frames below
MIN_SPEED_KMH are skipped.

Each seed's rows are written together, atomically, to <out>/<split>/rows/<seed>.csv, so an interrupted
run resumes by skipping finished seeds. Seeds are consumed in order; the dataset is the shortest prefix
of seeds whose frame count reaches the target, written to <out>/<split>.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

from ads import config, provenance

_ENV = None


def episode_sigma(seed: int) -> float:
    return float(np.random.default_rng([seed, 0xDA27]).choice(config.DART_SIGMAS))


def ar1_innovation_std(sigma: float, rho: float = config.DART_RHO) -> float:
    """Innovation std that makes sigma the stationary std of e_t = rho * e_{t-1} + innovation."""
    return sigma * math.sqrt(1.0 - rho * rho)


def rows_path(out: Path, split: str, seed: int) -> Path:
    return out / split / "rows" / f"{seed:05d}.csv"


def count_rows(path: Path) -> int:
    with open(path, newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _init_worker(split: str) -> None:
    global _ENV
    from ads.sim.env import make_env

    _ENV = make_env(split, camera=True)


def collect_seed(seed: int, split: str, out: Path) -> dict:
    import cv2

    from ads.sim.env import camera_frame, lane_state, speed_kmh
    from ads.sim.expert import expert_action, make_expert

    final = rows_path(out, split, seed)
    if final.exists():
        return {"seed": seed, "frames": count_rows(final), "resumed": True}

    env = _ENV
    t0 = time.perf_counter()
    obs, _info = env.reset(seed=seed)
    expert = make_expert(env, seed)
    sigma = episode_sigma(seed)
    innov = ar1_innovation_std(sigma)
    rng = np.random.default_rng([seed, 1])
    frame_dir = out / split / "frames" / f"{seed:05d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    rows, noise, end, step = [], 0.0, "horizon", 0
    for step in range(config.EPISODE_HORIZON):
        steering, accel = expert_action(expert)
        noise = config.DART_RHO * noise + innov * rng.standard_normal()
        executed = float(np.clip(steering + noise, -1.0, 1.0))
        v = speed_kmh(env.agent)
        if step % config.SAVE_EVERY == 0 and v >= config.MIN_SPEED_KMH:
            lateral, curvature = lane_state(env.agent)
            frame = camera_frame(obs)
            bgr = frame if config.CAMERA_CHANNEL_ORDER == "BGR" else frame[..., ::-1]
            rel = f"{split}/frames/{seed:05d}/{step:05d}.jpg"
            cv2.imwrite(str(out / rel), np.ascontiguousarray(bgr), [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append(
                {
                    "path": rel,
                    "seed": seed,
                    "step": step,
                    "steering": f"{steering:.6f}",
                    "executed_steering": f"{executed:.6f}",
                    "throttle": f"{accel:.6f}",
                    "speed_kmh": f"{v:.3f}",
                    "lateral_m": f"{lateral:.4f}",
                    "curvature": f"{curvature:.6f}",
                    "noise_sigma": sigma,
                }
            )
        obs, _r, terminated, truncated, info = env.step([executed, accel])
        if terminated or truncated:
            end = next(
                (k for k in ("arrive_dest", "out_of_road", "crash_vehicle", "crash_object", "max_step") if info.get(k)), "other"
            )
            break

    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=config.CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, final)
    return {"seed": seed, "frames": len(rows), "steps": step + 1, "end": end, "sigma": sigma, "seconds": time.perf_counter() - t0}


def _worker_collect(seed: int, split: str, out: str) -> dict:
    return collect_seed(seed, split, Path(out))


def merge(out: Path, split: str, seeds: list[int]) -> int:
    n = 0
    with open(out / f"{split}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=config.CSV_COLUMNS)
        w.writeheader()
        for seed in seeds:
            with open(rows_path(out, split, seed), newline="") as g:
                for row in csv.DictReader(g):
                    w.writerow(row)
                    n += 1
    return n


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="DART data collection.")
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--target", type=int, required=True, help="frames to collect")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", type=Path, default=config.DATA_DIR / "metadrive")
    p.add_argument("--max-seeds", type=int, default=None, help="stop after this many seeds (timing runs)")
    args = p.parse_args(argv)

    seed_range = config.SPLITS[args.split]
    seeds = list(seed_range)[: args.max_seeds] if args.max_seeds else list(seed_range)
    done: dict[int, dict] = {}
    next_idx, t_start = 0, time.perf_counter()

    def prefix() -> tuple[list[int], int]:
        chosen, total = [], 0
        for s in seeds:
            if s not in done or total >= args.target:
                break
            chosen.append(s)
            total += done[s]["frames"]
        return chosen, total

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(args.split,)) as pool:
        pending = set()
        while True:
            chosen, total = prefix()
            if total >= args.target or (next_idx >= len(seeds) and not pending):
                break
            while next_idx < len(seeds) and len(pending) < 2 * args.workers:
                pending.add(pool.submit(_worker_collect, seeds[next_idx], args.split, str(args.out)))
                next_idx += 1
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                r = fut.result()
                done[r["seed"]] = r
            _, total = prefix()
            frames = sum(r["frames"] for r in done.values() if not r.get("resumed"))
            elapsed = time.perf_counter() - t_start
            rate = frames / elapsed if elapsed > 0 else 0.0
            eta_h = max(args.target - total, 0) / rate / 3600 if rate > 0 else float("nan")
            print(
                f"[{args.split}] seeds {len(done)} | prefix frames {total}/{args.target}"
                f" | {rate:.2f} new frames/s | eta {eta_h:.2f} h",
                flush=True,
            )
        for fut in pending:
            fut.cancel()

    chosen, total = prefix()
    n = merge(args.out, args.split, chosen)
    ends: dict[str, int] = {}
    for s in chosen:
        e = done[s].get("end", "resumed")
        ends[e] = ends.get(e, 0) + 1
    manifest = {
        "provenance": provenance.stamp("ads.sim.collect"),
        "split": args.split,
        "seeds": [chosen[0], chosen[-1]] if chosen else [],
        "num_seeds": len(chosen),
        "frames": n,
        "episode_ends": ends,
        "sigma_counts": {str(s): sum(1 for c in chosen if episode_sigma(c) == s) for s in config.DART_SIGMAS},
        "wall_seconds": time.perf_counter() - t_start,
        "workers": args.workers,
    }
    (args.out / f"{args.split}_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    main()
