"""Latency of a TFLite steering model: one frame at a time, as on a vehicle.

    python -m ads.export.bench --model runs/export/pilotnet/pilotnet_int8.tflite --threads 1 --runs 2000

Reports the median and p90 of per-invoke time after a warm-up, with the host recorded. Only numbers
from a stable host are published (see CLAIMS.md for where each figure was measured).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--runs", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    import tensorflow as tf

    from ads import config, provenance

    interp = tf.lite.Interpreter(model_path=str(args.model), num_threads=args.threads)
    interp.allocate_tensors()
    rng = np.random.default_rng(0)
    for d in interp.get_input_details():
        shape = d["shape"]
        value = rng.uniform(0, 255 if "image" in d["name"] else 60, shape).astype(d["dtype"])
        interp.set_tensor(d["index"], value)
    times = []
    for k in range(args.warmup + args.runs):
        t = time.perf_counter()
        interp.invoke()
        if k >= args.warmup:
            times.append(time.perf_counter() - t)
    times_ms = np.array(times) * 1000
    result = {
        "provenance": provenance.stamp("ads.export.bench"),
        "model": str(args.model),
        "bytes": args.model.stat().st_size,
        "input": [1, config.NET_H, config.NET_W, 3],
        "threads": args.threads,
        "runs": args.runs,
        "median_ms": float(np.median(times_ms)),
        "p90_ms": float(np.percentile(times_ms, 90)),
        "p99_ms": float(np.percentile(times_ms, 99)),
        "fps_at_median": 1000 / float(np.median(times_ms)),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "provenance"}, indent=1))
    return result


if __name__ == "__main__":
    main()
