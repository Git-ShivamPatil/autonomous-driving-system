"""Train the multi-task YOLO11 on the prepared BDD100K cache.

    python -m ads.perception.train --cache <cache> --out runs/yolo --scale s --epochs 30 --budget-hours 11

Follows Ultralytics' detection recipe: SGD (momentum 0.937, nesterov) with separate parameter groups so
biases and BatchNorm weights get no weight decay; gradient accumulation to a nominal batch of 64; linear
warm-up; cosine learning-rate decay; gradient clipping; an exponential moving average of the weights,
which is what gets evaluated and saved. Mixed precision on GPUs with fast FP16.

Cloud sessions are time-limited, so training stops at an epoch boundary once the next epoch would not
fit in --budget-hours, writes last.pt, and exits; --resume last.pt continues from there.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ads import provenance
from ads.perception.data import BDDMultiTask, collate, worker_init_fn
from ads.perception.losses import MultiTaskLoss
from ads.perception.model import build_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the multi-task YOLO11 on BDD100K.")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scale", default="s", choices=["n", "s", "m"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--nbs", type=int, default=64, help="nominal batch size for gradient accumulation")
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--lrf", type=float, default=0.01, help="final learning rate as a fraction of lr0")
    p.add_argument("--momentum", type=float, default=0.937)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=float, default=3.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--budget-hours", type=float, default=None, help="stop cleanly before this much wall time")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--val-limit", type=int, default=2000, help="val images scored after each epoch")
    p.add_argument("--limit", type=int, default=None, help="smoke test: first N training images")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay, bn = [], [], []
    norm_types = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if name == "bias":
                no_decay.append(param)
            elif isinstance(module, norm_types):
                bn.append(param)
            else:
                decay.append(param)
    return [
        {"params": bn, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def fitness(val: dict) -> float:
    """One number for picking the best checkpoint: equal weight on detection, drivable area and lanes."""
    return (val["detection"]["map50"] + val["drivable"]["miou"] + val["lane"]["iou"]) / 3


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    from ultralytics.utils.torch_utils import ModelEMA

    from ads.perception.evaluate import evaluate

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 7
    args.out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    model = build_model(args.scale, pretrained=not (args.no_pretrained or args.resume)).to(device)
    criterion = MultiTaskLoss(model)
    ds = BDDMultiTask(args.cache, "train", train=True, limit=args.limit)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_init_fn,
        persistent_workers=args.workers > 0,
    )
    nb = len(loader)
    if nb == 0:
        raise SystemExit(f"{len(ds)} images < --batch {args.batch}")
    accumulate = max(round(args.nbs / args.batch), 1)
    weight_decay = args.weight_decay * args.batch * accumulate / args.nbs
    optimizer = torch.optim.SGD(param_groups(model, weight_decay), lr=args.lr0, momentum=args.momentum, nesterov=True)
    lf = lambda x: ((1 - math.cos(x * math.pi / args.epochs)) / 2) * (args.lrf - 1) + 1  # noqa: E731
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    ema = ModelEMA(model)

    start_epoch, best, history, best_ckpt = 0, -1.0, [], None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.ema.load_state_dict(ckpt["ema"])
        ema.updates = ckpt["ema_updates"]
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best, history = ckpt["epoch"] + 1, ckpt["best_fitness"], ckpt["history"]
        # The best weights travel inside last.pt, so a session resuming into a fresh --out (as on Kaggle)
        # still has a best.pt even if none of its own epochs beats the restored best fitness.
        best_ckpt = ckpt.get("best")
        if best_ckpt is not None:
            torch.save(best_ckpt, args.out / "best.pt")

    meta = {
        "provenance": provenance.stamp("ads.perception.train"),
        "args": {k: str(v) for k, v in vars(args).items()},
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "amp": use_amp,
        "train_images": len(ds),
        "transferred": getattr(model, "transferred", None),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    print(json.dumps(meta, indent=1), flush=True)
    nw = max(round(args.warmup_epochs * nb), 100)
    last_opt_step, epoch_seconds, stopped_early = -1, None, False

    for epoch in range(start_epoch, args.epochs):
        if args.budget_hours and epoch_seconds and (time.time() - t_start + 1.15 * epoch_seconds) > args.budget_hours * 3600:
            stopped_early = True
            break
        t_epoch = time.time()
        model.train()
        sums: dict[str, float] = {}
        optimizer.zero_grad()
        for i, batch in enumerate(loader):
            ni = i + nb * epoch
            if ni <= nw:  # warm-up: bias lr from 0.1 down, others from 0 up; momentum 0.8 up
                xi = [0, nw]
                accumulate = max(1, int(np.interp(ni, xi, [1, args.nbs / args.batch]).round()))
                for j, g in enumerate(optimizer.param_groups):
                    g["lr"] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, args.lr0 * lf(epoch)])
                    g["momentum"] = np.interp(ni, xi, [0.8, args.momentum])
            img = batch["img"].to(device, non_blocking=True).float() / 255
            targets = {k: batch[k].to(device, non_blocking=True) for k in ("cls", "bboxes", "batch_idx", "drivable", "lane")}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(img)
            loss, items = criterion(outputs, targets)
            scaler.scale(loss).backward()
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)
                last_opt_step = ni
            for k, v in items.items():
                sums[k] = sums.get(k, 0.0) + v
        if nb * (epoch + 1) > nw:
            for g in optimizer.param_groups:
                g["lr"] = args.lr0 * lf(epoch + 1)
        epoch_seconds = time.time() - t_epoch

        val = evaluate(
            ema.ema, args.cache, "val", device, batch=16, limit=args.val_limit, workers=args.workers, multi_label=False
        )
        fit = fitness(val)
        row = {
            "epoch": epoch,
            "seconds": round(epoch_seconds, 1),
            "lr": optimizer.param_groups[1]["lr"],
            **{f"train_{k}": v / nb for k, v in sums.items()},
            "val_map50": val["detection"]["map50"],
            "val_map50_95": val["detection"]["map50_95"],
            "val_drivable_miou": val["drivable"]["miou"],
            "val_lane_iou": val["lane"]["iou"],
            "fitness": fit,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if fit > best:
            best = fit
            best_ckpt = {
                "ema": copy.deepcopy(ema.ema.state_dict()),
                "scale": args.scale,
                "epoch": epoch,
                "val": val,
                "meta": meta,
            }
            torch.save(best_ckpt, args.out / "best.pt")
        ckpt = {
            "model": model.state_dict(),
            "ema": ema.ema.state_dict(),
            "ema_updates": ema.updates,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_fitness": best,
            "best": best_ckpt,
            "history": history,
            "scale": args.scale,
            "meta": meta,
        }
        torch.save(ckpt, args.out / "last.pt")

    summary = {
        **meta,
        "epochs_done": len(history),
        "stopped_for_budget": stopped_early,
        "complete": not stopped_early and len(history) >= args.epochs,
        "best_fitness": best,
        "history": history,
        "wall_hours": (time.time() - t_start) / 3600,
    }
    (args.out / "train_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
