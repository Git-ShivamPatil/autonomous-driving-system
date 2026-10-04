"""Record the commit, command and hardware behind every result file, so CLAIMS.md can cite them."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import UTC, datetime

from ads import config


def git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=config.ROOT, capture_output=True, text=True
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def hardware() -> dict:
    info = {"os": platform.platform(), "cpu": cpu_name(), "logical_cpus": os.cpu_count(), "python": platform.python_version()}
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except ImportError:
        pass
    return info


def stamp(module: str | None = None) -> dict:
    """Provenance block; pass the module name when the entry point was run with ``python -m``."""
    argv = [f"python -m {module}", *sys.argv[1:]] if module else ["python", *sys.argv]
    return {
        "commit": git_commit(),
        "command": " ".join(argv),
        "utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "hardware": hardware(),
    }
