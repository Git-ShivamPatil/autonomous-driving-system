"""Push and follow the Kaggle jobs in kaggle/ (prepare on CPU, train on a T4 GPU).

Needs the Kaggle CLI (pip install kaggle) and an API token: KAGGLE_API_TOKEN, or
%USERPROFILE%\\.kaggle\\access_token, or a legacy kaggle.json. GPU sessions need a phone-verified account.

    python tools/kaggle_job.py push prepare --user <kaggle-username>
    python tools/kaggle_job.py push train --user <kaggle-username> --epochs 30
    python tools/kaggle_job.py push train --user <kaggle-username> --resume   # continue an incomplete run
    python tools/kaggle_job.py status train --user <kaggle-username>
    python tools/kaggle_job.py output train --user <kaggle-username> --dest runs/kaggle/train

Jobs clone this repository at the pushed commit, so local changes must be pushed to GitHub first; the
tool refuses to push a job from a dirty or unpushed checkout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "data" / "kaggle_jobs"


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


REQUIRED_AT_HEAD = (
    "ads/perception/prepare.py",
    "ads/perception/train.py",
    "ads/perception/evaluate.py",
    "tools/tune_lane_cv.py",
)


def pushed_commit() -> str:
    if git("status", "--porcelain"):  # untracked files count: a job runs only what is committed
        raise SystemExit("working tree has uncommitted or untracked changes; commit and push first")
    for path in REQUIRED_AT_HEAD:
        if subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"HEAD:{path}"], capture_output=True).returncode:
            raise SystemExit(f"{path} is not in HEAD")
    head = git("rev-parse", "HEAD")
    git("fetch", "--quiet", "origin")
    if not git("branch", "-r", "--contains", head):
        raise SystemExit(f"{head[:7]} is not on GitHub yet; push it first")
    return head


def slug(job: str) -> str:
    return f"ads-{job}"


def render(job: str, user: str, args: argparse.Namespace) -> Path:
    commit = pushed_commit()
    code = (ROOT / "kaggle" / f"{job}.py").read_text()
    values = {
        "COMMIT": commit,
        "SCALE": args.scale,
        "EPOCHS": str(args.epochs),
        "BATCH": str(args.batch),
        "BUDGET_HOURS": str(args.budget_hours),
    }
    for key, value in values.items():
        code = code.replace("{{" + key + "}}", value)
    if "{{" in code:
        raise SystemExit(f"unfilled placeholder in kaggle/{job}.py")
    meta = {
        "id": f"{user}/{slug(job)}",
        "title": slug(job),
        "code_file": f"{job}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_internet": True,
        "enable_gpu": job == "train",
        "dataset_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "kernel_sources": [],
    }
    if job == "train":
        meta["machine_shape"] = "NvidiaTeslaT4"  # the P100 is retired and its id silently maps to a T4
        meta["kernel_sources"].append(f"{user}/{slug('prepare')}")
        if args.resume:
            meta["kernel_sources"].append(f"{user}/{slug('train')}")
    out = JOBS_DIR / job
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{job}.py").write_text(code)
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    return out


def kaggle(*args: str) -> None:
    print("+ kaggle", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", "kaggle", *args], check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["push", "status", "logs", "output"])
    p.add_argument("job", choices=["prepare", "train"])
    p.add_argument("--user", required=True, help="Kaggle username")
    p.add_argument("--scale", default="s")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--budget-hours", type=float, default=11.0)
    p.add_argument("--resume", action="store_true", help="attach the previous train run's output and resume it")
    p.add_argument("--dest", type=Path, default=None)
    args = p.parse_args()
    ref = f"{args.user}/{slug(args.job)}"
    if args.action == "push":
        path = render(args.job, args.user, args)
        extra = ["--accelerator", "NvidiaTeslaT4"] if args.job == "train" else []
        kaggle("kernels", "push", "-p", str(path), *extra)
    elif args.action == "status":
        kaggle("kernels", "status", ref)
    elif args.action == "logs":
        kaggle("kernels", "logs", ref)
    else:
        dest = args.dest or ROOT / "runs" / "kaggle" / args.job
        dest.mkdir(parents=True, exist_ok=True)
        kaggle("kernels", "output", ref, "-p", str(dest), "-o")


if __name__ == "__main__":
    main()
