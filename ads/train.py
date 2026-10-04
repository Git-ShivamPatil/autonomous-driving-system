"""Train the PilotNet steering model on the collected MetaDrive frames.

    python -m ads.train --out runs/steering/pilotnet

Loss is SmoothL1 weighted by the inverse frequency of the steering bin. AdamW with a per-step cosine
schedule; early stopping on validation MAE. Mixed precision is used only on GPUs with fast FP16
(compute capability 7.0 or newer); on older GPUs it gives no speed-up.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from ads import config, provenance
from ads.dataset import InverseFrequencyWeights, SteeringDataset, worker_init_fn
from ads.eval.metrics import split_mae
from ads.model import PilotNet, count_parameters

SMOOTH_L1_BETA = 0.1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the PilotNet steering model.")
    p.add_argument("--train-cache", type=Path, default=config.DATA_DIR / "cache" / "train")
    p.add_argument("--val-cache", type=Path, default=config.DATA_DIR / "cache" / "val")
    p.add_argument("--out", type=Path, default=config.RUNS_DIR / "steering" / "pilotnet")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-weights", action="store_true", help="ablation: unweighted SmoothL1")
    p.add_argument("--limit", type=int, default=None, help="smoke test: first N frames of each split")
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    return p.parse_args(argv)


def amp_supported(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 7


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    preds, targets, curvatures = [], [], []
    for img, speed, steer, _w, curv in loader:
        pred = model(img.to(device, non_blocking=True), speed.float().to(device)).clamp(-1, 1)
        preds.append(pred.float().cpu().numpy())
        targets.append(steer.float().numpy())
        curvatures.append(curv.float().numpy())
    p, t, c = (np.concatenate(x) for x in (preds, targets, curvatures))
    out = split_mae(p, t, c, config.CURVE_CURVATURE)
    out["rmse"] = float(np.sqrt(np.mean((p - t) ** 2)))
    return out


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_amp = amp_supported(device)
    args.out.mkdir(parents=True, exist_ok=True)

    train_ds = SteeringDataset(args.train_cache, train=True)
    if not args.no_weights:
        train_ds.weights = InverseFrequencyWeights(train_ds.steering)
    val_ds = SteeringDataset(args.val_cache, train=False)
    train_set = Subset(train_ds, range(min(args.limit, len(train_ds)))) if args.limit else train_ds
    val_set = Subset(val_ds, range(min(args.limit, len(val_ds)))) if args.limit else val_ds

    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda", worker_init_fn=worker_init_fn)
    if args.workers > 0:
        loader_kw["persistent_workers"] = True
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False, **loader_kw)
    if len(train_loader) == 0:
        raise SystemExit(f"{len(train_set)} training frames < --batch {args.batch}: no training batches")

    model = PilotNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    meta = {
        "provenance": provenance.stamp("ads.train"),
        "args": {k: str(v) for k, v in vars(args).items()},
        "device": str(device),
        "amp": use_amp,
        "parameters": count_parameters(model),
        "train_frames": len(train_set),
        "val_frames": len(val_set),
    }
    print(json.dumps(meta, indent=1))
    log_path = args.out / "epochs.jsonl"
    log_path.write_text("")

    best_mae, best_epoch, bad = float("inf"), -1, 0
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total, n = 0.0, 0
        for img, speed, steer, w, _curv in train_loader:
            img = img.to(device, non_blocking=True)
            speed, steer, w = (x.float().to(device, non_blocking=True) for x in (speed, steer, w))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(img, speed)
            loss = (w * F.smooth_l1_loss(pred.float(), steer, beta=SMOOTH_L1_BETA, reduction="none")).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * img.shape[0]
            n += img.shape[0]
        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": total / max(n, 1), "lr": sched.get_last_lr()[0], "seconds": time.perf_counter() - t0}
        row.update({f"val_{k}": v for k, v in val.items()})
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row))
        if val["mae"] < best_mae:
            best_mae, best_epoch, bad = val["mae"], epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val}, args.out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop: no val MAE improvement for {args.patience} epochs")
                break

    ckpt = torch.load(args.out / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    summary = dict(meta)
    summary.update(
        {
            "best_epoch": best_epoch,
            "epochs_run": epoch,
            "train_seconds": time.perf_counter() - t_start,
            "val": evaluate(model, val_loader, device),
        }
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["val"], indent=1))
    return summary


if __name__ == "__main__":
    main()
