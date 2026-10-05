"""Evaluate the multi-task model on BDD100K val.

    python -m ads.perception.evaluate --weights runs/yolo/best.pt --cache <cache> --out results/perception.json

Detection: mAP@0.5 and mAP@0.5:0.95 over the 12 classes, with Ultralytics' ap_per_class. Boxes are
mapped back to 1280x720 source pixels. Colourless traffic lights are not targets (see bdd.py), so a
detection on one counts as a false positive.

Drivable area: IoU of each class and their mean (mIoU), at 1280x720.

Lane lines: lane-class IoU against 2 px ground-truth lines at 1280x720, plus two accuracy
definitions that published papers use under the same name: lane recall TP/(TP+FN) (YOLOP's code)
and balanced accuracy (TPR+TNR)/2 (A-YOLOM). The OpenCV baseline is scored on the same masks.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from ads import provenance
from ads.perception import bdd
from ads.perception.data import IMG_H, IMG_W, NET_H, NET_W, PAD_TOP, BDDMultiTask, collate

IOUV = np.linspace(0.5, 0.95, 10)
SRC_SCALE = bdd.IMAGE_W / IMG_W


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, 4) x (M, 4) xyxy -> (N, M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_predictions(pred_cls: np.ndarray, true_cls: np.ndarray, iou: np.ndarray) -> np.ndarray:
    """(n_pred, 10) bool: is each prediction a true positive at each IoU threshold?

    Same procedure as Ultralytics' DetectionValidator.match_predictions: candidate pairs are ordered by
    IoU, each prediction keeps its best ground truth, then each ground truth goes to the earliest
    prediction, which is the most confident one because NMS output is sorted by confidence.
    """
    correct = np.zeros((len(pred_cls), len(IOUV)), dtype=bool)
    if len(pred_cls) == 0 or len(true_cls) == 0:
        return correct
    iou = iou * (true_cls[:, None] == pred_cls[None, :])
    for k, t in enumerate(IOUV):
        matches = np.array(np.nonzero(iou >= t)).T  # rows of (gt index, prediction index)
        if len(matches) == 0:
            continue
        if len(matches) > 1:
            matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1], k] = True
    return correct


class BinaryConfusion:
    def __init__(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0

    def add(self, pred: np.ndarray, truth: np.ndarray) -> None:
        pred, truth = pred.astype(bool), truth.astype(bool)
        tp = int(np.count_nonzero(pred & truth))
        fp = int(np.count_nonzero(pred & ~truth))
        fn = int(np.count_nonzero(~pred & truth))
        self.tp, self.fp, self.fn = self.tp + tp, self.fp + fp, self.fn + fn
        self.tn += truth.size - tp - fp - fn

    def summary(self) -> dict:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        iou_fg = tp / max(tp + fp + fn, 1)
        iou_bg = tn / max(tn + fp + fn, 1)
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        return {
            "iou": iou_fg,
            "iou_background": iou_bg,
            "miou": (iou_fg + iou_bg) / 2,
            "recall": recall,
            "balanced_accuracy": (recall + specificity) / 2,
            "pixels": tp + fp + fn + tn,
        }


def logits_to_source_mask(logits: torch.Tensor) -> np.ndarray:
    """(1, 384, 640) logits -> (720, 1280) bool mask: crop the letterbox padding, upsample, threshold at 0."""
    x = logits[:, PAD_TOP : PAD_TOP + IMG_H, :].unsqueeze(0).float()
    x = F.interpolate(x, size=(bdd.IMAGE_H, bdd.IMAGE_W), mode="bilinear", align_corners=False)
    return (x[0, 0] > 0).cpu().numpy()


def detections_to_source(det: torch.Tensor) -> np.ndarray:
    """NMS output rows [x1, y1, x2, y2, conf, cls] in network pixels -> source pixels."""
    d = det.float().cpu().numpy().copy()
    d[:, [1, 3]] -= PAD_TOP
    d[:, :4] *= SRC_SCALE
    d[:, [0, 2]] = d[:, [0, 2]].clip(0, bdd.IMAGE_W)
    d[:, [1, 3]] = d[:, [1, 3]].clip(0, bdd.IMAGE_H)
    return d


def load_model(weights: Path, device: torch.device):
    from ads.perception.model import build_model

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model = build_model(ckpt.get("scale", "s"), pretrained=False)
    model.load_state_dict(ckpt["ema"] if ckpt.get("ema") is not None else ckpt["model"])
    return model.to(device).eval()


@torch.no_grad()
def evaluate(
    model,
    cache: Path,
    split: str,
    device: torch.device,
    batch: int = 16,
    limit: int | None = None,
    workers: int = 2,
    half: bool = False,
    multi_label: bool = True,
    lane_baseline: bool = False,
    lane_cv_config=None,
) -> dict:
    from ultralytics.utils.metrics import ap_per_class
    from ultralytics.utils.nms import non_max_suppression

    from ads.perception.lane_cv import LaneCVConfig, detect_lanes

    cv_cfg = replace(lane_cv_config or LaneCVConfig(), line_thickness=1)  # 1 px at 640x360 = the 2 px eval lines
    if half:
        model = model.half()
    ds = BDDMultiTask(cache, split, train=False, limit=limit)
    gt_boxes = {r["name"]: np.array(r["boxes"], dtype=np.float32).reshape(-1, 5) for r in ds.records}
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=collate)
    stats, target_cls = [], []
    drivable, lane, base = BinaryConfusion(), BinaryConfusion(), BinaryConfusion()
    model.eval()
    for b in loader:
        img = b["img"].to(device).float() / 255
        if half:
            img = img.half()
        det_out, da, ll = model(img)
        dets = non_max_suppression(det_out, conf_thres=0.001, iou_thres=0.7, multi_label=multi_label, max_det=300)
        for i, name in enumerate(b["name"]):
            gt = gt_boxes[name]
            d = detections_to_source(dets[i])
            iou = box_iou(gt[:, 1:], d[:, :4])
            stats.append((match_predictions(d[:, 5], gt[:, 0], iou), d[:, 4], d[:, 5]))
            target_cls.append(gt[:, 0])
            truth_da = cv2.imread(str(cache / "eval" / split / "drivable" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
            truth_ll = cv2.imread(str(cache / "eval" / split / "lane" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
            drivable.add(logits_to_source_mask(da[i]), truth_da)
            lane.add(logits_to_source_mask(ll[i]), truth_ll)
            if lane_baseline:
                small = cv2.imread(str(cache / "images" / split / f"{name}.jpg"), cv2.IMREAD_COLOR)
                cv_mask = detect_lanes(small, cv_cfg)
                base.add(cv2.resize(cv_mask, (bdd.IMAGE_W, bdd.IMAGE_H), interpolation=cv2.INTER_NEAREST), truth_ll)

    tp, conf, pred_cls = (np.concatenate([s[k] for s in stats]) for k in range(3))
    tcls = np.concatenate(target_cls)
    names = dict(enumerate(bdd.CLASSES))
    _, _, p, r, _, ap, classes, *_ = ap_per_class(tp, conf, pred_cls, tcls, names=names)
    per_class = {bdd.CLASSES[int(c)]: {"ap50": float(ap[k, 0]), "ap50_95": float(ap[k].mean())} for k, c in enumerate(classes)}
    out = {
        "images": len(ds),
        "detection": {
            "map50": float(ap[:, 0].mean()) if len(ap) else 0.0,
            "map50_95": float(ap.mean()) if len(ap) else 0.0,
            "precision": float(p.mean()) if len(p) else 0.0,
            "recall": float(r.mean()) if len(r) else 0.0,
            "classes_evaluated": len(classes),
            "per_class": per_class,
            "targets": int(len(tcls)),
        },
        "drivable": drivable.summary(),
        "lane": lane.summary(),
    }
    if lane_baseline:
        out["lane_opencv_baseline"] = {**base.summary(), "config": asdict(cv_cfg)}
    return out


def load_lane_cv_config(path: Path):
    """LaneCVConfig from tools/tune_lane_cv.py output (JSON lists back to the dataclass's tuples)."""
    from ads.perception.lane_cv import LaneCVConfig

    cfg = json.loads(Path(path).read_text())["config"]
    cfg["src"] = tuple(tuple(p) for p in cfg["src"])
    for key in ("dst_x", "sobel_thresh", "sat_thresh"):
        cfg[key] = tuple(cfg[key])
    return LaneCVConfig(**cfg)


@torch.no_grad()
def benchmark(model, device: torch.device, half: bool, batch: int = 1, warmup: int = 20, iters: int = 200) -> dict:
    """Median end-to-end forward latency (all three heads, plus NMS) on random input of the network size."""
    from ultralytics.utils.nms import non_max_suppression

    x = torch.rand(batch, 3, NET_H, NET_W, device=device)
    if half:
        x, model = x.half(), model.half()
    times = []
    for k in range(warmup + iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        det_out, _da, _ll = model(x)
        non_max_suppression(det_out, conf_thres=0.25, iou_thres=0.7)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if k >= warmup:
            times.append(time.perf_counter() - t)
    med = float(np.median(times))
    return {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else provenance.cpu_name(),
        "precision": "fp16" if half else "fp32",
        "batch": batch,
        "input": [NET_H, NET_W],
        "median_ms": med * 1000,
        "p90_ms": float(np.percentile(times, 90)) * 1000,
        "fps": batch / med,
    }


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Evaluate the multi-task model on BDD100K.")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--split", default="val")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--half", action="store_true")
    p.add_argument("--lane-baseline", action="store_true", help="also score the OpenCV lane baseline")
    p.add_argument("--lane-cv-config", type=Path, default=None, help="tuned baseline config from tools/tune_lane_cv.py")
    p.add_argument("--fps-only", action="store_true")
    args = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half = args.half and device.type == "cuda"  # FP16 only where it is supported; decided once for both steps
    model = load_model(args.weights, device)
    result = {"provenance": provenance.stamp("ads.perception.evaluate"), "weights": str(args.weights)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.fps_only:
        result.update(
            evaluate(
                model,
                args.cache,
                args.split,
                device,
                args.batch,
                args.limit,
                args.workers,
                half,
                lane_baseline=args.lane_baseline,
                lane_cv_config=load_lane_cv_config(args.lane_cv_config) if args.lane_cv_config else None,
            )
        )
        args.out.write_text(json.dumps(result, indent=1))  # saved before benchmarking, so a benchmark failure loses nothing
    result["speed"] = benchmark(model, device, half=half)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "provenance"}, indent=1))
    return result


if __name__ == "__main__":
    main()
