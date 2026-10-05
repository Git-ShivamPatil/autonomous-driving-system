"""Open-loop vs closed-loop comparison of steering-model variants, as a Markdown table.

    python tools/variants_table.py --out results/variants.json \
        A=runs/steering/pilotnet/best.pt:runs/closed_loop/sel2_A_es7 \
        C=runs/steering/pilotnet_cos30/last.pt:runs/closed_loop/sel2_C_cos30last ...

Each variant's validation MAE is computed here from its checkpoint (best- and last-epoch weights are
scored the same way), next to its closed-loop results on the same validation roads. The Spearman rank
correlation between validation MAE and each closed-loop metric answers "do better open-loop scores mean
better driving?" (Codevilla et al. 2018). With a handful of variants it is descriptive, not a test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def val_mae(ckpt: Path, val_cache: Path) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from ads.dataset import SteeringDataset
    from ads.eval.closed_loop import load_model
    from ads.train import evaluate

    torch.set_num_threads(2)
    model = load_model(ckpt)
    loader = DataLoader(SteeringDataset(val_cache, train=False), batch_size=512, shuffle=False, num_workers=0)
    return evaluate(model, loader, torch.device("cpu"))


def ranks(values: list[float]) -> np.ndarray:
    """Ranks with ties given their average rank, as in the standard Spearman coefficient."""
    v = np.asarray(values, dtype=float)
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v))
    r[order] = np.arange(len(v), dtype=float)
    for value in np.unique(v):
        tied = v == value
        r[tied] = r[tied].mean()
    return r


def spearman(x: list[float], y: list[float]) -> float:
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("variants", nargs="+", help="label=checkpoint:closed_loop_dir")
    p.add_argument("--val-cache", type=Path, default=Path("data/cache/val"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rows = []
    for spec in args.variants:
        label, rest = spec.split("=", 1)
        ckpt, run = rest.split(":", 1) if rest.count(":") == 1 else rest.rsplit(":", 1)
        s = json.loads((Path(run) / "summary.json").read_text())
        v = val_mae(Path(ckpt), args.val_cache)
        rows.append(
            {
                "label": label,
                "checkpoint": ckpt,
                "closed_loop": run,
                "val_mae": v["mae"],
                "val_mae_curve": v["mae_curve"],
                "val_mae_straight": v["mae_straight"],
                "roads": s["episodes"],
                "autonomy_pct": s["autonomy_pct"],
                "success_rate": s["success_rate"],
                "success_ci95": s["success_ci95"],
                "interventions_per_km": s["interventions_per_km"],
                "interventions_per_km_curve": s.get("interventions_per_km_curve"),
                "interventions_per_km_straight": s.get("interventions_per_km_straight"),
                "collisions_per_km": s["collisions_per_km"],
                "lateral_abs_mean_m": s["lateral_abs_mean_m"],
                "on_policy_mae": s.get("on_policy", {}).get("mae"),
            }
        )
    maes = [r["val_mae"] for r in rows]
    corr = {
        "val_mae_vs_autonomy": spearman(maes, [r["autonomy_pct"] for r in rows]),
        "val_mae_vs_success": spearman(maes, [r["success_rate"] for r in rows]),
        "val_mae_vs_interventions_per_km": spearman(maes, [r["interventions_per_km"] for r in rows]),
    }
    if all(r["on_policy_mae"] is not None for r in rows):
        on = [r["on_policy_mae"] for r in rows]
        corr["on_policy_mae_vs_autonomy"] = spearman(on, [r["autonomy_pct"] for r in rows])
        corr["on_policy_mae_vs_interventions_per_km"] = spearman(on, [r["interventions_per_km"] for r in rows])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"variants": rows, "spearman": corr}, indent=1))

    header = ["Variant", "Val MAE", "Autonomy", "Success (95% CI)", "Interventions/km (curve / straight)"]
    print("| " + " | ".join([*header, "Collisions/km", "On-policy MAE"]) + " |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        lo, hi = r["success_ci95"]
        print(
            f"| {r['label']} | {r['val_mae']:.4f} | {r['autonomy_pct']:.1f}% | {100 * r['success_rate']:.0f}% "
            f"({100 * lo:.0f}-{100 * hi:.0f}) | {r['interventions_per_km']:.2f} ({r['interventions_per_km_curve']:.2f} / "
            f"{r['interventions_per_km_straight']:.2f}) | {r['collisions_per_km']:.2f} | "
            f"{r['on_policy_mae']:.4f} |"
        )
    print("\nSpearman rank correlations across variants:", json.dumps(corr, indent=1))


if __name__ == "__main__":
    main()
