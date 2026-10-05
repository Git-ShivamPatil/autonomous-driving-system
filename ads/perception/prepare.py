"""Build the multi-task training cache from the extracted BDD100K archives.

    python -m ads.perception.prepare --images <dir> --labels <dir> --drivable <dir> --out <cache>

Inputs are the archives' own layouts: <images>/100k/<split>/<name>.jpg,
<labels>/100k/<split>/<name>.json and <drivable>/labels/<split>/<name>_drivable_id.png.

Output, per split:
  images/<split>/<name>.jpg      640x360 training image
  drivable/<split>/<name>.png    640x360 binary drivable area (direct + alternative)
  lane/<split>/<name>.png        640x360 lane lines, 8 px at source scale
  labels/<split>.jsonl           12-class boxes in 1280x720 source pixels, scene attributes
  eval/<split>/{drivable,lane}/  1280x720 evaluation masks (val only; lanes 2 px, as YOLOP)
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from ads.perception import bdd

TRAIN_W, TRAIN_H = 640, 360
PNG_FAST = [cv2.IMWRITE_PNG_COMPRESSION, 1]


def _write(path: Path, image: np.ndarray, params: list[int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, params or []):
        raise OSError(f"failed to write {path}")


def process_one(job: tuple[str, str, str, str, str, str, bool]) -> dict:
    name, split, images, labels, drivable, out, with_eval = job
    images, labels, drivable, out = Path(images), Path(labels), Path(drivable), Path(out)
    img = cv2.imread(str(images / "100k" / split / f"{name}.jpg"), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(name)
    label = json.loads((labels / "100k" / split / f"{name}.json").read_text())
    targets, dropped = bdd.detection_targets(label)
    lines = bdd.lane_polylines(label)

    small = cv2.resize(img, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_AREA)
    _write(out / "images" / split / f"{name}.jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])

    id_path = drivable / "labels" / split / f"{name}_drivable_id.png"
    id_map = cv2.imread(str(id_path), cv2.IMREAD_GRAYSCALE)
    has_drivable = id_map is not None
    da_full = bdd.drivable_binary(id_map) if has_drivable else np.zeros((bdd.IMAGE_H, bdd.IMAGE_W), np.uint8)
    da_small = cv2.resize(da_full, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_NEAREST)
    _write(out / "drivable" / split / f"{name}.png", da_small, PNG_FAST)
    _write(out / "lane" / split / f"{name}.png", bdd.lane_mask(lines, TRAIN_W, TRAIN_H, bdd.LANE_TRAIN_PX), PNG_FAST)
    if with_eval:
        _write(out / "eval" / split / "drivable" / f"{name}.png", da_full, PNG_FAST)
        lane_full = bdd.lane_mask(lines, bdd.IMAGE_W, bdd.IMAGE_H, bdd.LANE_EVAL_PX)
        _write(out / "eval" / split / "lane" / f"{name}.png", lane_full, PNG_FAST)
    return {
        "name": name,
        "boxes": [[int(r[0]), *(round(float(v), 2) for v in r[1:])] for r in targets],
        "dropped_lights": dropped,
        "has_drivable": has_drivable,
        "lanes": len(lines),
        "attributes": label.get("attributes", {}),
    }


def build_split(split: str, images: Path, labels: Path, drivable: Path, out: Path, workers: int, limit: int | None) -> dict:
    names = sorted(p.stem for p in (images / "100k" / split).glob("*.jpg"))
    if limit:
        names = names[:limit]
    with_eval = split == "val"
    jobs = [(n, split, str(images), str(labels), str(drivable), str(out), with_eval) for n in names]
    (out / "labels").mkdir(parents=True, exist_ok=True)
    counts, dropped, no_drivable, no_lanes = Counter(), 0, 0, 0
    with Pool(workers) as pool, open(out / "labels" / f"{split}.jsonl", "w") as f:
        for i, rec in enumerate(pool.imap(process_one, jobs, chunksize=32)):
            f.write(json.dumps(rec) + "\n")
            counts.update(bdd.CLASSES[b[0]] for b in rec["boxes"])
            dropped += rec["dropped_lights"]
            no_drivable += not rec["has_drivable"]
            no_lanes += rec["lanes"] == 0
            if (i + 1) % 5000 == 0:
                print(f"{split}: {i + 1}/{len(jobs)}", flush=True)
    return {
        "images": len(names),
        "boxes_per_class": {c: counts.get(c, 0) for c in bdd.CLASSES},
        "dropped_colourless_lights": dropped,
        "images_without_drivable_map": no_drivable,
        "images_without_lanes": no_lanes,
    }


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--drivable", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    p.add_argument("--limit", type=int, default=None, help="first N images per split (smoke test)")
    args = p.parse_args(argv)
    manifest = {"classes": list(bdd.CLASSES), "train_size": [TRAIN_W, TRAIN_H], "splits": {}}
    for split in args.splits:
        manifest["splits"][split] = build_split(
            split, args.images, args.labels, args.drivable, args.out, args.workers, args.limit
        )
        print(split, json.dumps(manifest["splits"][split]), flush=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    main()
