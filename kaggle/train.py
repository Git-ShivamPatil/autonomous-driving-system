"""Kaggle job: train the multi-task YOLO11 on the prepared cache, then evaluate and benchmark it.

Runs on a T4 GPU session. The cache comes from the prepare job's output (kernel_sources). If a previous
training job is also attached, its last.pt is resumed. Training stops on its own before Kaggle's 12-hour
session limit; a job whose summary says "complete": false is continued by pushing again with
--resume-from. Outputs, in /kaggle/working/yolo: last.pt, best.pt, train_summary.json, and, once
training is complete, eval.json (full val set, OpenCV lane baseline included) and speed_*.json.

The values below the marker are filled in by tools/kaggle_job.py when the job is pushed.
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

REPO = "https://github.com/Git-ShivamPatil/autonomous-driving-system"
COMMIT = "{{COMMIT}}"
SCALE = "{{SCALE}}"
EPOCHS = "{{EPOCHS}}"
BATCH = "{{BATCH}}"
BUDGET_HOURS = "{{BUDGET_HOURS}}"
ULTRALYTICS = "ultralytics==8.4.173"

WORK = Path("/kaggle/working")
SCRATCH = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else Path("/tmp")
INPUT = Path("/kaggle/input")


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, **kw)


def find(pattern: str) -> list[Path]:
    return sorted(INPUT.rglob(pattern))


import torch  # noqa: E402

print(
    json.dumps(
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "arch": torch.cuda.get_arch_list(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    ),
    flush=True,
)

src = SCRATCH / "ads"
run(["git", "clone", "--quiet", REPO, str(src)])
run(["git", "-C", str(src), "checkout", "--quiet", COMMIT])
run([sys.executable, "-m", "pip", "install", "--quiet", ULTRALYTICS])
env = {**os.environ, "PYTHONPATH": str(src)}

# The prepare job ships the cache as one tar (Kaggle zips outputs of 100,000+ files into _output_.zip).
# Extracting to local scratch also avoids reading 260,000 small files from the input mount every epoch.
tars = find("bdd_cache.tar")
zips = find("_output_.zip")
if tars:
    with tarfile.open(tars[0]) as tar:
        tar.extractall(SCRATCH)
elif zips:
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(SCRATCH)
else:
    raise SystemExit("no bdd_cache.tar under /kaggle/input: attach the prepare job's output")
manifests = sorted(SCRATCH.rglob("bdd_cache/manifest.json"))
if not manifests:
    raise SystemExit("the prepare output has no bdd_cache/manifest.json")
cache = manifests[0].parent
lane_cfg = find("lane_cv_config.json") or sorted(SCRATCH.rglob("lane_cv_config.json"))
out = WORK / "yolo"
out.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable,
    "-m",
    "ads.perception.train",
    "--cache",
    str(cache),
    "--out",
    str(out),
    "--scale",
    SCALE,
    "--epochs",
    EPOCHS,
    "--batch",
    BATCH,
    "--workers",
    "4",
    "--budget-hours",
    BUDGET_HOURS,
]
previous = find("yolo/last.pt")
if previous:
    cmd += ["--resume", str(previous[0])]
    prev_best = previous[0].parent / "best.pt"  # last.pt also carries it; this keeps it even for older runs
    if prev_best.exists():
        shutil.copy(prev_best, out / "best.pt")
run(cmd, env=env)
if not (out / "best.pt").exists():
    raise SystemExit("training finished without a best.pt")

summary = json.loads((out / "train_summary.json").read_text())
if summary["complete"]:
    ev = [
        sys.executable,
        "-m",
        "ads.perception.evaluate",
        "--weights",
        str(out / "best.pt"),
        "--cache",
        str(cache),
        "--out",
        str(out / "eval.json"),
        "--lane-baseline",
        "--half",
        "--workers",
        "4",
    ]
    if lane_cfg:
        ev += ["--lane-cv-config", str(lane_cfg[0])]
    run(ev, env=env)
    for half in (True, False):
        run(
            [
                sys.executable,
                "-m",
                "ads.perception.evaluate",
                "--weights",
                str(out / "best.pt"),
                "--fps-only",
                "--out",
                str(out / f"speed_{'fp16' if half else 'fp32'}.json"),
                *(["--half"] if half else []),
            ],
            env=env,
        )
