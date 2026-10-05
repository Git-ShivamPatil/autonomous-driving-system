"""Dataset over the prepared BDD100K cache, with augmentation applied jointly to image, masks and boxes.

Images are 640x360 in the cache and are letterboxed to the network input of 640x384 (12 px of padding
top and bottom). Boxes in the label file are in 1280x720 source pixels and are halved here.
Augmentation (training only): random scale and translation, horizontal flip, HSV jitter. Mosaic is
not used, because it would break the drivable-area and lane masks' spatial context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

IMG_W, IMG_H = 640, 360
NET_W, NET_H = 640, 384
PAD_TOP = (NET_H - IMG_H) // 2
SOURCE_SCALE = IMG_W / 1280


@dataclass(frozen=True)
class AugmentConfig:
    scale: float = 0.25  # random zoom in [1 - scale, 1 + scale]
    translate: float = 0.1  # fraction of width/height
    flip: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4


def read_records(cache: Path, split: str) -> list[dict]:
    with open(cache / "labels" / f"{split}.jsonl") as f:
        return [json.loads(line) for line in f]


def hsv_jitter(img: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    gains = rng.uniform(-1, 1, 3) * [cfg.hsv_h, cfg.hsv_s, cfg.hsv_v] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(256, dtype=np.float32)
    lut_h = ((x * gains[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * gains[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * gains[2], 0, 255).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def random_affine(img, masks, boxes, rng: np.random.Generator, cfg: AugmentConfig):
    """Zoom about the centre and translate; boxes (N, 4) xyxy are transformed, clipped and filtered."""
    h, w = img.shape[:2]
    s = rng.uniform(1 - cfg.scale, 1 + cfg.scale)
    tx, ty = rng.uniform(-cfg.translate, cfg.translate) * w, rng.uniform(-cfg.translate, cfg.translate) * h
    m = np.array([[s, 0, (1 - s) * w / 2 + tx], [0, s, (1 - s) * h / 2 + ty]], dtype=np.float32)
    img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
    masks = [cv2.warpAffine(mk, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0) for mk in masks]
    if len(boxes):
        new = boxes * s
        new[:, [0, 2]] += m[0, 2]
        new[:, [1, 3]] += m[1, 2]
        clipped = new.copy()
        clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0, w)
        clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0, h)
        area = (new[:, 2] - new[:, 0]) * (new[:, 3] - new[:, 1])
        area_c = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
        keep = ((clipped[:, 2] - clipped[:, 0]) > 2) & ((clipped[:, 3] - clipped[:, 1]) > 2) & (area_c > 0.2 * area)
        return img, masks, clipped, keep
    return img, masks, boxes, np.zeros(0, dtype=bool)


def flip_lr(img, masks, boxes):
    w = img.shape[1]
    img = np.ascontiguousarray(img[:, ::-1])
    masks = [np.ascontiguousarray(mk[:, ::-1]) for mk in masks]
    if len(boxes):
        boxes = boxes.copy()
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    return img, masks, boxes


class BDDMultiTask(torch.utils.data.Dataset):
    def __init__(self, cache: Path, split: str, train: bool, augment: AugmentConfig | None = None, limit: int | None = None):
        self.cache, self.split, self.train = Path(cache), split, train
        self.records = read_records(self.cache, split)[:limit] if limit else read_records(self.cache, split)
        self.augment = augment or AugmentConfig()
        self.rng = np.random.default_rng(0)

    def __len__(self) -> int:
        return len(self.records)

    def load(self, i: int):
        rec = self.records[i]
        name = rec["name"]
        img = cv2.imread(str(self.cache / "images" / self.split / f"{name}.jpg"), cv2.IMREAD_COLOR)
        da = cv2.imread(str(self.cache / "drivable" / self.split / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        ll = cv2.imread(str(self.cache / "lane" / self.split / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or da is None or ll is None:
            raise FileNotFoundError(name)
        b = np.array(rec["boxes"], dtype=np.float32).reshape(-1, 5)
        return name, img, da, ll, b[:, 0].astype(np.int64), b[:, 1:] * SOURCE_SCALE

    def __getitem__(self, i: int):
        name, img, da, ll, cls, boxes = self.load(i)
        if self.train:
            img, (da, ll), boxes, keep = random_affine(img, [da, ll], boxes, self.rng, self.augment)
            if len(cls):
                cls, boxes = cls[keep], boxes[keep]
            if self.rng.random() < self.augment.flip:
                img, (da, ll), boxes = flip_lr(img, [da, ll], boxes)
            img = hsv_jitter(img, self.rng, self.augment)
        pad = NET_H - IMG_H - PAD_TOP
        img = cv2.copyMakeBorder(img, PAD_TOP, pad, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        da = cv2.copyMakeBorder(da, PAD_TOP, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
        ll = cv2.copyMakeBorder(ll, PAD_TOP, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
        xywh = np.zeros((len(boxes), 4), dtype=np.float32)
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, [1, 3]] += PAD_TOP
            xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2 / NET_W
            xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2 / NET_H
            xywh[:, 2] = (boxes[:, 2] - boxes[:, 0]) / NET_W
            xywh[:, 3] = (boxes[:, 3] - boxes[:, 1]) / NET_H
        rgb = np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))  # Ultralytics' weights expect RGB
        return {
            "img": torch.from_numpy(rgb),
            "drivable": torch.from_numpy(da.astype(np.float32)),
            "lane": torch.from_numpy(ll.astype(np.float32)),
            "cls": torch.from_numpy(cls.astype(np.float32)),
            "bboxes": torch.from_numpy(xywh),
            "name": name,
        }


def collate(samples: list[dict]) -> dict:
    """Stack images and masks; concatenate boxes with a batch index, as Ultralytics' detection loss expects."""
    return {
        "img": torch.stack([s["img"] for s in samples]),
        "drivable": torch.stack([s["drivable"] for s in samples]),
        "lane": torch.stack([s["lane"] for s in samples]),
        "cls": torch.cat([s["cls"] for s in samples]).view(-1, 1),
        "bboxes": torch.cat([s["bboxes"] for s in samples]),
        "batch_idx": torch.cat([torch.full((len(s["cls"]),), i, dtype=torch.float32) for i, s in enumerate(samples)]),
        "name": [s["name"] for s in samples],
    }


def worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)
