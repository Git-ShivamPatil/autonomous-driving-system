"""Render dataset samples with boxes, drivable area and lane masks overlaid, to check label alignment.

python tools/show_bdd_sample.py --cache data/bdd_dev_cache --split train --n 4 --out docs/media/bdd_samples.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ads.perception.bdd import CLASSES
from ads.perception.data import NET_H, NET_W, BDDMultiTask


def render(sample: dict) -> np.ndarray:
    img = sample["img"].numpy().transpose(1, 2, 0)[..., ::-1].copy()  # RGB CHW -> BGR HWC
    da = sample["drivable"].numpy() > 0
    ll = sample["lane"].numpy() > 0
    img[da] = (0.6 * img[da] + 0.4 * np.array([255, 120, 0])).astype(np.uint8)
    img[ll] = (0, 255, 255)
    for c, (x, y, w, h) in zip(sample["cls"].tolist(), sample["bboxes"].tolist(), strict=True):
        x1, y1 = int((x - w / 2) * NET_W), int((y - h / 2) * NET_H)
        x2, y2 = int((x + w / 2) * NET_W), int((y + h / 2) * NET_H)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(img, CLASSES[int(c)], (x1, max(y1 - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    return img


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--augment", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    ds = BDDMultiTask(args.cache, args.split, train=bool(args.augment))
    ds.rng = np.random.default_rng(1)
    tiles = [render(ds[i]) for i in range(args.n)]
    rows = [np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles) - 1, 2)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
