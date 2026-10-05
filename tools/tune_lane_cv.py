"""Grid-search the OpenCV lane baseline on training images, so the baseline is tuned, not a straw man.

    python tools/tune_lane_cv.py --cache <cache> --limit 2000 --out results/lane_cv_config.json

Scores lane IoU against the 640x360 training masks (8 px at source scale), predicting with the same
line thickness. Validation images are never used here; the chosen config is then scored on val by
ads.perception.evaluate --lane-baseline --lane-cv-config <json>.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np

from ads.perception.data import read_records
from ads.perception.lane_cv import LaneCVConfig, detect_lanes

SRC_OPTIONS = {
    "narrow": ((0.12, 0.92), (0.43, 0.62), (0.57, 0.62), (0.88, 0.92)),
    "above_bonnet": ((0.05, 0.88), (0.42, 0.60), (0.58, 0.60), (0.95, 0.88)),
    "wide": ((0.0, 0.86), (0.40, 0.60), (0.60, 0.60), (1.0, 0.86)),
}


def score(cfg: LaneCVConfig, items: list[tuple[np.ndarray, np.ndarray]]) -> float:
    tp = fp = fn = 0
    for img, truth in items:
        pred = detect_lanes(img, cfg).astype(bool)
        t = truth.astype(bool)
        tp += int(np.count_nonzero(pred & t))
        fp += int(np.count_nonzero(pred & ~t))
        fn += int(np.count_nonzero(~pred & t))
    return tp / max(tp + fp + fn, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    names = [r["name"] for r in read_records(args.cache, "train")[: args.limit]]
    items = [
        (
            cv2.imread(str(args.cache / "images" / "train" / f"{n}.jpg"), cv2.IMREAD_COLOR),
            cv2.imread(str(args.cache / "lane" / "train" / f"{n}.png"), cv2.IMREAD_GRAYSCALE),
        )
        for n in names
    ]
    thickness = 4  # 8 px at 1280x720, as in the training masks
    results = []
    for (src_name, src), max_lines, white, sobel in itertools.product(SRC_OPTIONS.items(), (2, 4), (180, 210), (25, 50)):
        cfg = replace(
            LaneCVConfig(), src=src, max_lines=max_lines, white_l_min=white, sobel_thresh=(sobel, 255), line_thickness=thickness
        )
        iou = score(cfg, items)
        results.append({"src": src_name, "max_lines": max_lines, "white_l_min": white, "sobel_low": sobel, "iou": iou})
        print(json.dumps(results[-1]), flush=True)
    best = max(results, key=lambda r: r["iou"])
    best_cfg = replace(
        LaneCVConfig(),
        src=SRC_OPTIONS[best["src"]],
        max_lines=best["max_lines"],
        white_l_min=best["white_l_min"],
        sobel_thresh=(best["sobel_low"], 255),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"train_images": len(items), "best": best, "config": asdict(best_cfg), "grid": results}, indent=1)
    )
    print("best:", json.dumps(best))


if __name__ == "__main__":
    main()
