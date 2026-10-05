"""Kaggle job: download BDD100K from Berkeley, build the multi-task cache, tune the lane baseline.

Runs on a CPU session (no GPU quota). Outputs, in /kaggle/working:
  bdd_cache.tar         the cache built by ads.perception.prepare (with BDD100K's LICENSE.txt), as one
                        uncompressed tar: Kaggle saves an output of 100,000+ files as a single zip, and
                        the cache is about 260,000 files
  manifest.json         the cache manifest (also inside the tar)
  LICENSE.txt           BDD100K's licence, which must accompany every copy of the data
  lane_cv_config.json   OpenCV lane baseline tuned on 2,000 training images
  prepare_log.json      timings

The values in {{...}} are filled in by tools/kaggle_job.py when the job is pushed.
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = "https://github.com/Git-ShivamPatil/autonomous-driving-system"
COMMIT = "{{COMMIT}}"
BASE = "http://dl.yf.io/bdd100k/"  # Berkeley's server; plain HTTP only, no login, resumable
ARCHIVES = {"images": "bdd100k_images_100k.zip", "labels": "bdd100k_labels.zip", "drivable": "bdd100k_drivable_maps.zip"}

WORK = Path("/kaggle/working")
SCRATCH = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else Path("/tmp")
log, t0 = {}, time.time()


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, **kw)


log["scratch_free_gb"] = round(shutil.disk_usage(SCRATCH).free / 2**30, 1)
src = SCRATCH / "ads"
run(["git", "clone", "--quiet", REPO, str(src)])
run(["git", "-C", str(src), "checkout", "--quiet", COMMIT])
env = {**os.environ, "PYTHONPATH": str(src)}

for key, name in ARCHIVES.items():
    zpath = SCRATCH / name
    t = time.time()
    run(["wget", "-c", "-q", "--tries=10", "--timeout=60", "-O", str(zpath), BASE + name])
    log[f"download_{key}_s"] = round(time.time() - t, 1)
    log[f"{key}_bytes"] = zpath.stat().st_size
    t = time.time()
    run(["unzip", "-q", "-o", str(zpath), "-d", str(SCRATCH / key)])
    zpath.unlink()
    log[f"unzip_{key}_s"] = round(time.time() - t, 1)
run(["wget", "-q", "-O", str(SCRATCH / "LICENSE.txt"), BASE + "LICENSE.txt"])

cache = SCRATCH / "bdd_cache"
t = time.time()
run(
    [sys.executable, "-m", "ads.perception.prepare", "--images", str(SCRATCH / "images"), "--labels",
     str(SCRATCH / "labels"), "--drivable", str(SCRATCH / "drivable"), "--out", str(cache)],
    env=env,
)  # fmt: skip
log["prepare_s"] = round(time.time() - t, 1)
shutil.copy(SCRATCH / "LICENSE.txt", cache / "LICENSE.txt")

t = time.time()
run(
    [sys.executable, str(src / "tools" / "tune_lane_cv.py"), "--cache", str(cache), "--limit", "2000",
     "--out", str(WORK / "lane_cv_config.json")],
    env=env,
)  # fmt: skip
log["tune_lane_cv_s"] = round(time.time() - t, 1)

t = time.time()
with tarfile.open(WORK / "bdd_cache.tar", "w") as tar:  # uncompressed: JPEGs and PNGs are compressed already
    tar.add(cache, arcname="bdd_cache")
log["tar_s"] = round(time.time() - t, 1)
log["tar_bytes"] = (WORK / "bdd_cache.tar").stat().st_size
shutil.copy(cache / "manifest.json", WORK / "manifest.json")
shutil.copy(cache / "LICENSE.txt", WORK / "LICENSE.txt")

log["commit"] = COMMIT
log["total_s"] = round(time.time() - t0, 1)
(WORK / "prepare_log.json").write_text(json.dumps(log, indent=1))
print(json.dumps(log, indent=1))
