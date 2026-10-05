"""Cross-domain evaluation of the steering model on Sully Chen's real driving dataset (2017, ~45k frames).

    python -m ads.eval.sully --root data/sully/driving_dataset --model runs/steering/<run>/best.pt --out results/sully.json

The dataset is one continuous drive: 455x256 dashcam frames labelled with the steering-wheel angle in
degrees (data.txt: "<n>.jpg <angle>"). It has no speed, so the speed input is fixed at SPEED_KMH.

The simulator model outputs MetaDrive's normalised front-wheel angle; the real labels are
steering-wheel degrees. Their scale (the steering ratio) and sign convention are unknown, so nothing is
fitted on the test frames. Frames are split in time: the first 70% for fitting, then a 5% gap (so
near-identical neighbouring frames cannot leak), then the last 25% for testing.

  zero-shot     the simulator model, with only a linear scale and sign fitted on the fitting part
  fine-tuned    the simulator weights trained further on the fitting part
  from scratch  the same architecture and recipe, from random weights

MAE is reported in steering-wheel degrees on the test part, with Pearson correlation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ads import config

SPEED_KMH = 30.0
FIT, GAP = 0.70, 0.05
LABEL_SCALE_DEG = 100.0  # training target = angle / 100, so typical values sit in [-1, 1]


def read_index(root: Path) -> tuple[list[str], np.ndarray]:
    names, angles = [], []
    for line in (root / "data.txt").read_text().splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            names.append(parts[0])
            angles.append(float(parts[1]))
    return names, np.array(angles, dtype=np.float32)


def preprocess_real(bgr: np.ndarray) -> np.ndarray:
    """Real 16:9 frame -> the simulator's 320x160 framing -> PilotNet input (66x200 YUV).

    The frame is scaled to 320 wide (320x180) and its top 20 rows dropped to reach 320x160, then the
    same crop, resize and colour conversion as for simulator frames. Fixed, not tuned on this data.
    """
    from ads.steering.preprocess import preprocess

    small = cv2.resize(bgr, (config.CAM_W, 180), interpolation=cv2.INTER_AREA)[20:]
    return preprocess(np.ascontiguousarray(small), "BGR")


def build_cache(root: Path, out: Path) -> tuple[np.ndarray, np.ndarray]:
    names, angles = read_index(root)
    cache = out / "sully_images.npy"
    if not cache.exists():
        out.mkdir(parents=True, exist_ok=True)
        arr = np.lib.format.open_memmap(cache, mode="w+", dtype=np.uint8, shape=(len(names), config.NET_H, config.NET_W, 3))
        for i, n in enumerate(names):
            img = cv2.imread(str(root / n), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(root / n)
            arr[i] = preprocess_real(img)
        arr.flush()
        del arr
    return np.load(cache, mmap_mode="r"), angles


def split_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    fit_end, test_start = int(FIT * n), int((FIT + GAP) * n)
    return np.arange(fit_end), np.arange(test_start, n)


def predict(model, images, device, batch: int = 512) -> np.ndarray:
    import torch

    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = torch.from_numpy(np.ascontiguousarray(np.asarray(images[i : i + batch]).transpose(0, 3, 1, 2))).to(device)
            s = torch.full((x.shape[0],), SPEED_KMH, device=device)
            out.append(model(x, s).float().cpu().numpy())
    return np.concatenate(out)


def scores(pred_deg: np.ndarray, true_deg: np.ndarray) -> dict:
    return {
        "mae_deg": float(np.abs(pred_deg - true_deg).mean()),
        "rmse_deg": float(np.sqrt(np.mean((pred_deg - true_deg) ** 2))),
        "pearson_r": float(np.corrcoef(pred_deg, true_deg)[0, 1]) if np.std(pred_deg) > 0 else 0.0,
        "frames": int(len(true_deg)),
    }


def finetune(model, images, targets, device, epochs: int, lr: float, seed: int = 0):
    """Train on (images, angle / LABEL_SCALE_DEG) with the steering recipe's loss and optimiser."""
    import torch
    from torch.nn import functional as F

    from ads.steering.preprocess import augment

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(images) // 256)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(images))
        for b in range(0, len(order) - 255, 256):
            idx = np.sort(order[b : b + 256])
            imgs, ys = [], []
            for i in idx:
                img, y = augment(np.asarray(images[i]), float(targets[i]), rng)
                imgs.append(img)
                ys.append(y)
            x = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2)).to(device)
            y = torch.tensor(ys, device=device)
            s = torch.full((len(idx),), SPEED_KMH, device=device)
            loss = F.smooth_l1_loss(model(x, s), y, beta=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return model.eval()


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Cross-domain steering evaluation on Sully Chen's dataset.")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=config.DATA_DIR / "cache" / "sully")
    p.add_argument("--epochs", type=int, default=10)
    args = p.parse_args(argv)

    import torch

    from ads import provenance
    from ads.model import PilotNet

    def load_model(path: Path) -> PilotNet:
        model = PilotNet()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
        return model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, angles = build_cache(args.root, args.cache)
    fit, test = split_indices(len(angles))
    report = {
        "provenance": provenance.stamp("ads.eval.sully"),
        "model": str(args.model),
        "frames": int(len(angles)),
        "fit_frames": int(len(fit)),
        "test_frames": int(len(test)),
        "speed_kmh_assumed": SPEED_KMH,
        "label_std_deg_test": float(angles[test].std()),
    }
    # Trivial reference: always predict the fitting part's mean angle.
    report["constant_mean"] = scores(np.full(len(test), angles[fit].mean()), angles[test])

    sim = load_model(args.model).to(device)
    raw = predict(sim, images, device)
    a, b = np.polyfit(raw[fit], angles[fit], 1)  # angle ~ a * output + b, fitted on the fitting part only
    report["zero_shot"] = {**scores(a * raw[test] + b, angles[test]), "fitted_scale_deg": float(a), "fitted_offset_deg": float(b)}

    targets = angles / LABEL_SCALE_DEG
    for name, init in (("fine_tuned", "sim"), ("from_scratch", "random")):
        torch.manual_seed(0)
        model = load_model(args.model).to(device) if init == "sim" else PilotNet().to(device)
        model.train()
        model = finetune(model, images[fit], targets[fit], device, args.epochs, lr=3e-4 if init == "sim" else 1e-3)
        report[name] = {**scores(predict(model, images[test], device) * LABEL_SCALE_DEG, angles[test]), "epochs": args.epochs}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=1))
    return report


if __name__ == "__main__":
    main()
