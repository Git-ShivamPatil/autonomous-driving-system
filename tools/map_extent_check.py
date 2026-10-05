"""Measure how far each road map extends from the origin.

MetaDrive textures the road only inside a square terrain region of side map_region_size centred on the
origin (default 1024 m). The region's semantic texture costs (side x 22 px/m)^2 floats, about 2 GB at
1024 m, rebuilt on every reset. A smaller region is safe only if every map used fits inside it.

    python tools/map_extent_check.py --out runs/map_extents.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ads import config
from ads.sim.env import make_env


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-seeds", type=int, default=400, help="first N training seeds")
    args = p.parse_args()

    splits = {
        "test": list(config.TEST_SEEDS),
        "val": list(config.VAL_SEEDS),
        "train": list(config.TRAIN_SEEDS)[: args.train_seeds],
    }
    result, t0 = {}, time.perf_counter()
    for split, seeds in splits.items():
        env = make_env(split, camera=False)
        extents = {}
        for s in seeds:
            env.reset(seed=s)
            x_min, x_max, y_min, y_max = env.engine.current_map.road_network.get_bounding_box()
            extents[s] = round(float(max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))), 1)
        env.close()
        values = sorted(extents.values())
        result[split] = {
            "seeds": [seeds[0], seeds[-1]],
            "max_extent_m": values[-1],
            "p99_extent_m": values[int(0.99 * (len(values) - 1))],
            "over_240m": int(sum(v > 240 for v in values)),
            # The default 1024 m terrain region textures roads only within +-512 m of the origin.
            "over_512m": [s for s, v in extents.items() if v > 512],
            "per_seed": extents,
        }
        print(split, {k: v for k, v in result[split].items() if k != "per_seed"}, f"{time.perf_counter() - t0:.0f}s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
