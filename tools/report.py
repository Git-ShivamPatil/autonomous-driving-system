"""Print the measured numbers behind README.md and CLAIMS.md, straight from the result files.

    python tools/report.py --runs runs --data data/metadrive

Every row carries the commit and hardware recorded by the run that produced it, so the claims ledger
is filled from evidence rather than copied by hand. Missing results print as "pending".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def hw(prov: dict | None) -> str:
    if not prov:
        return ""
    h = prov.get("hardware", {})
    parts = [h.get("gpu"), h.get("cpu")]
    return " / ".join(p for p in parts if p)


def commit(prov: dict | None) -> str:
    return (prov or {}).get("commit", "")[:7]


def row(claim: str, value: str, command: str, prov: dict | None) -> str:
    return f"| {claim} | {value or 'pending'} | `{command}` | {commit(prov)} | {hw(prov)} |"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--data", type=Path, default=Path("data/metadrive"))
    p.add_argument("--steering", default="pilotnet", help="run name under runs/steering")
    p.add_argument("--closed-loop", default="final_test", help="run name under runs/closed_loop")
    p.add_argument("--expert", default="expert_test")
    args = p.parse_args()

    print("| Claim | Value | Command | Commit | Hardware |\n|---|---|---|---|---|")
    for split in ("train", "val"):
        m = load(args.data / f"{split}_manifest.json")
        value = f"{m['frames']:,} frames from {m['num_seeds']} roads (seeds {m['seeds'][0]}-{m['seeds'][1]})" if m else ""
        print(row(f"{split} data (DART)", value, (m or {}).get("provenance", {}).get("command", ""), (m or {}).get("provenance")))

    s = load(args.runs / "steering" / args.steering / "summary.json")
    if s:
        v = s["val"]
        value = (
            f"val MAE {v['mae']:.4f} (curve {v['mae_curve']:.4f}, straight {v['mae_straight']:.4f}); "
            f"{s['parameters']:,} params; epoch {s['best_epoch']}"
        )
        print(row("Steering CNN, open loop", value, s["provenance"]["command"], s["provenance"]))

    for name, label in ((args.closed_loop, "Steering CNN, closed loop"), (args.expert, "Expert baseline, closed loop")):
        c = load(args.runs / "closed_loop" / name / "summary.json")
        if c:
            lo, hi = c["success_ci95"]
            value = (
                f"{c['episodes']} roads: autonomy {c['autonomy_pct']:.1f}%, success {100 * c['success_rate']:.1f}% "
                f"[{100 * lo:.1f}, {100 * hi:.1f}], {c['interventions_per_km']:.2f} interventions/km, "
                f"{c['collisions_per_km']:.2f} collisions/km, route completion {100 * c['route_completion_mean']:.1f}%, "
                f"lateral offset mean {c['lateral_abs_mean_m']:.2f} m / p95 {c['lateral_abs_p95_m']:.2f} m"
            )
            if "on_policy" in c:
                value += f", on-policy MAE {c['on_policy']['mae']:.4f}"
            print(row(label, value, c["provenance"]["command"], c["provenance"]))
        else:
            print(row(label, "", "", None))

    e = load(args.runs / "export" / args.steering / "export_report.json")
    if e:
        value = (
            f"FP32 {e['fp32']['bytes'] / 1e6:.2f} MB, MAE {e['fp32']['mae']:.4f}; INT8 {e['int8']['bytes'] / 1e3:.0f} KB, "
            f"MAE {e['int8']['mae']:.4f} (PyTorch {e['pytorch']['mae']:.4f}) on {e['val_frames']} val frames"
        )
        print(row("TFLite export", value, e["provenance"]["command"], e["provenance"]))

    su = load(args.runs / "sully" / f"{args.steering}.json")
    if su:
        value = "; ".join(
            f"{k.replace('_', ' ')} MAE {su[k]['mae_deg']:.1f} deg (r {su[k]['pearson_r']:.2f})"
            for k in ("constant_mean", "zero_shot", "fine_tuned", "from_scratch")
            if k in su
        )
        print(row("Sully Chen (real), test 25%", value, su["provenance"]["command"], su["provenance"]))

    pe = load(args.runs / "kaggle" / "train" / "yolo" / "eval.json")
    if pe:
        d, da, ll = pe["detection"], pe["drivable"], pe["lane"]
        value = (
            f"{d['classes_evaluated']} classes: mAP50 {d['map50']:.3f}, mAP50-95 {d['map50_95']:.3f}; "
            f"drivable mIoU {da['miou']:.3f}; lane IoU {ll['iou']:.3f} (recall {ll['recall']:.3f}, "
            f"balanced acc {ll['balanced_accuracy']:.3f})"
        )
        if "lane_opencv_baseline" in pe:
            value += f"; OpenCV lane IoU {pe['lane_opencv_baseline']['iou']:.3f}"
        print(row("Multi-task YOLO11, BDD100K val", value, pe["provenance"]["command"], pe["provenance"]))
    else:
        print(row("Multi-task YOLO11, BDD100K val", "", "", None))


if __name__ == "__main__":
    main()
