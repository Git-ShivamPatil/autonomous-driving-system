"""BDD100K label conversion for the multi-task model.

Source: the per-image 2018 label JSON (bdd100k_labels.zip) and the legacy drivable-area id maps
(bdd100k_drivable_maps.zip), both from the Berkeley server. The 2020 det_20 labels are not available
from an official source any more, so detection uses the 2018 labels mapped to det_20's class names.

Classes: det_20's 10 classes with "traffic light" split by the trafficLightColor attribute into red,
yellow and green, giving 12. Lights whose colour is "none" (seen from the side or behind) have no
colour class and are left out of training and evaluation targets; the count is reported.

Lane masks: BDD100K publishes no lane masks for these labels, so they are drawn from the lane
polylines: every lane category except crosswalk, direction "parallel", Bezier segments evaluated.
Following YOLOP's protocol, training lines are 8 px wide at 1280x720 and evaluation lines 2 px.
"""

from __future__ import annotations

import cv2
import numpy as np

CLASSES = (
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light red",
    "traffic light yellow",
    "traffic light green",
    "traffic sign",
)
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}

# 2018 category -> det_20 name (det.toml maps person/bike/motor the same way)
CATEGORY_2018 = {
    "person": "pedestrian",
    "rider": "rider",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "motor": "motorcycle",
    "bike": "bicycle",
    "traffic sign": "traffic sign",
}
LIGHT_COLOUR = {"red": "traffic light red", "yellow": "traffic light yellow", "green": "traffic light green"}

IMAGE_W, IMAGE_H = 1280, 720
LANE_TRAIN_PX = 8
LANE_EVAL_PX = 2


def objects(label: dict) -> list[dict]:
    frames = label.get("frames") or [{}]
    return frames[0].get("objects", []) or []


def detection_targets(label: dict) -> tuple[np.ndarray, int]:
    """(N, 5) float32 array of [class, x1, y1, x2, y2] in source pixels, and the number of colourless lights dropped."""
    rows, dropped = [], 0
    for obj in objects(label):
        box = obj.get("box2d")
        if box is None:
            continue
        cat = obj.get("category", "")
        if cat == "traffic light":
            name = LIGHT_COLOUR.get((obj.get("attributes") or {}).get("trafficLightColor", "none"))
            if name is None:
                dropped += 1
                continue
        else:
            name = CATEGORY_2018.get(cat)
            if name is None:
                continue
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        rows.append([CLASS_ID[name], x1, y1, x2, y2])
    arr = np.array(rows, dtype=np.float32).reshape(-1, 5)
    if len(arr):
        arr[:, [1, 3]] = arr[:, [1, 3]].clip(0, IMAGE_W)
        arr[:, [2, 4]] = arr[:, [2, 4]].clip(0, IMAGE_H)
    return arr, dropped


def _cubic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, steps: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, steps + 1)[1:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def poly2d_points(poly2d: list, steps: int = 12) -> np.ndarray:
    """Vertices [x, y, type] with type 'L' (vertex) or 'C' (cubic Bezier control) -> dense (M, 2) polyline."""
    pts = [np.array(v[:2], dtype=np.float64) for v in poly2d]
    types = [v[2] if len(v) > 2 else "L" for v in poly2d]
    if not pts:
        return np.zeros((0, 2))
    out, i = [pts[0]], 1
    while i < len(pts):
        if types[i] == "C" and i + 1 < len(pts):  # control, control, end (end wraps to start if closed)
            end = pts[i + 2] if i + 2 < len(pts) else pts[0]
            out.extend(_cubic(out[-1], pts[i], pts[i + 1], end, steps))
            i += 3
        else:
            out.append(pts[i])
            i += 1
    return np.array(out)


def lane_polylines(label: dict) -> list[np.ndarray]:
    lines = []
    for obj in objects(label):
        cat = obj.get("category", "")
        if not cat.startswith("lane/") or cat == "lane/crosswalk":
            continue
        if (obj.get("attributes") or {}).get("direction", "parallel") != "parallel":
            continue
        pts = poly2d_points(obj.get("poly2d") or [])
        if len(pts) >= 2:
            lines.append(pts)
    return lines


def lane_mask(lines: list[np.ndarray], width: int, height: int, thickness_px_at_1280: float) -> np.ndarray:
    """Binary lane-line mask at (height, width); thickness is given at the 1280x720 source scale."""
    sx, sy = width / IMAGE_W, height / IMAGE_H
    thickness = max(1, int(round(thickness_px_at_1280 * sx)))
    mask = np.zeros((height, width), dtype=np.uint8)
    for pts in lines:
        scaled = np.round(pts * [sx, sy]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [scaled], False, 1, thickness=thickness, lineType=cv2.LINE_8)
    return mask


def drivable_binary(id_map: np.ndarray) -> np.ndarray:
    """Legacy *_drivable_id.png (0 background, 1 direct, 2 alternative) -> 1 where drivable (YOLOP merges both)."""
    return ((id_map == 1) | (id_map == 2)).astype(np.uint8)
