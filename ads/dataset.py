"""Steering dataset: a preprocessed uint8 cache on disk, OpenCV augmentation, inverse-frequency weights.

The cache is built once from the collected JPEGs and memory-mapped, so 75k frames (~3 GB as
66x200x3 uint8) never have to fit in RAM.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from ads import config
from ads.steering.preprocess import augment, preprocess

N_BINS = 21
MAX_WEIGHT = 10.0


def read_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def build_cache(csv_path: Path, frames_root: Path, out_dir: Path) -> int:
    """Preprocess every frame listed in csv_path into out_dir/images.u8 plus out_dir/meta.npz."""
    rows = read_rows(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    shape = (n, config.NET_H, config.NET_W, 3)
    images = np.lib.format.open_memmap(out_dir / "images.npy", mode="w+", dtype=np.uint8, shape=shape)
    for i, row in enumerate(rows):
        frame = cv2.imread(str(frames_root / row["path"]), cv2.IMREAD_COLOR)  # collected frames are stored as BGR
        if frame is None:
            raise FileNotFoundError(frames_root / row["path"])
        images[i] = preprocess(frame, "BGR")
    images.flush()
    del images
    np.savez(
        out_dir / "meta.npz",
        steering=np.array([float(r["steering"]) for r in rows], dtype=np.float32),
        speed_kmh=np.array([float(r["speed_kmh"]) for r in rows], dtype=np.float32),
        curvature=np.array([float(r["curvature"]) for r in rows], dtype=np.float32),
        seed=np.array([int(r["seed"]) for r in rows], dtype=np.int32),
        step=np.array([int(r["step"]) for r in rows], dtype=np.int32),
        noise_sigma=np.array([float(r["noise_sigma"]) for r in rows], dtype=np.float32),
    )
    (out_dir / "source.json").write_text(json.dumps({"csv": str(csv_path), "frames": n}))
    return n


class InverseFrequencyWeights:
    """Per-sample loss weight = 1 / frequency of the sample's steering bin.

    The histogram is built from the labels and their negations, so a horizontally flipped sample keeps
    its weight. Weights are normalised to mean 1 over the training labels and then capped at
    max_weight, so a handful of extreme samples cannot dominate a batch (capping lowers the mean
    slightly below 1).
    """

    def __init__(self, labels: np.ndarray, n_bins: int = N_BINS, max_weight: float = MAX_WEIGHT) -> None:
        labels = np.clip(np.asarray(labels, dtype=np.float64), -1, 1)
        self.edges = np.linspace(-1.0, 1.0, n_bins + 1)
        counts, _ = np.histogram(np.concatenate([labels, -labels]), bins=self.edges)
        inverse = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
        inverse /= inverse[self.bin_of(labels)].mean()
        self.bin_weights = np.minimum(inverse, max_weight)

    def bin_of(self, steering: np.ndarray | float) -> np.ndarray:
        s = np.clip(np.asarray(steering, dtype=np.float64), -1, 1)
        return np.clip(np.digitize(s, self.edges) - 1, 0, len(self.edges) - 2)

    def __call__(self, steering: np.ndarray | float) -> np.ndarray:
        return self.bin_weights[self.bin_of(steering)]


class SteeringDataset:
    """torch-compatible map-style dataset over a cache directory.

    Returns (image CHW uint8, speed_kmh, steering, weight, curvature). The memmap is opened lazily in
    each DataLoader worker; pickling an open memmap would copy the whole array.
    """

    def __init__(self, cache_dir: Path, train: bool, weights: InverseFrequencyWeights | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.train = train
        meta = np.load(self.cache_dir / "meta.npz")
        self.steering = meta["steering"]
        self.speed = meta["speed_kmh"]
        self.curvature = meta["curvature"]
        self.weights = weights
        self._images: np.ndarray | None = None
        self.rng = np.random.default_rng(0)

    def __len__(self) -> int:
        return len(self.steering)

    def _image(self, i: int) -> np.ndarray:
        if self._images is None:
            self._images = np.load(self.cache_dir / "images.npy", mmap_mode="r")
        return np.asarray(self._images[i])

    def __getitem__(self, i: int):
        import torch

        img = self._image(i)
        steer = float(self.steering[i])
        if self.train:
            img, steer = augment(img, steer, self.rng)
        weight = float(self.weights(steer)) if self.weights is not None else 1.0
        chw = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        return chw, float(self.speed[i]), steer, weight, float(self.curvature[i])


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker its own reproducible augmentation stream."""
    import torch

    info = torch.utils.data.get_worker_info()
    dataset = info.dataset
    while not isinstance(dataset, SteeringDataset):  # unwrap torch Subset
        dataset = dataset.dataset
    dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)


def main(argv: list[str] | None = None) -> None:
    """Build the train and val caches: python -m ads.dataset"""
    import argparse

    p = argparse.ArgumentParser(description="Preprocess collected frames into memory-mapped caches.")
    p.add_argument("--frames", type=Path, default=config.DATA_DIR / "metadrive")
    p.add_argument("--out", type=Path, default=config.DATA_DIR / "cache")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    args = p.parse_args(argv)
    for split in args.splits:
        n = build_cache(args.frames / f"{split}.csv", args.frames, args.out / split)
        print(f"{split}: {n} frames -> {args.out / split}")


if __name__ == "__main__":
    main()
