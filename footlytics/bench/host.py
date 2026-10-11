"""Host checks for a shared machine: disk floor, free VRAM, and who else is on the GPU.

Other people run jobs on the same 4090, so seconds-per-minute is noisy and VRAM is not ours
alone. Every run records the contention it ran under, and refuses to start rather than die
mid-clip when the room is not there.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional

FLOOR_BYTES = 1_500_000_000


class ResourceError(RuntimeError):
    """Not enough disk or VRAM to start; the message says how much there is and is needed."""


def free_disk_bytes(path: str | Path) -> int:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def require_disk(path: str | Path, need_bytes: int, floor: int = FLOOR_BYTES) -> None:
    free = free_disk_bytes(path)
    if free - need_bytes < floor:
        raise ResourceError(
            f"{free / 1e9:.1f} GB free at {path}, need {need_bytes / 1e9:.1f} GB and the disk "
            f"must keep {floor / 1e9:.1f} GB free (shared machine)")


def parse_gpu(gpu_csv: str, apps_csv: str, own_pids: Iterable[int] = ()) -> dict:
    used, total = (int(float(v)) for v in gpu_csv.strip().splitlines()[0].split(","))
    own = set(own_pids)
    procs = [tuple(int(float(v)) for v in line.split(",")) for line in apps_csv.strip().splitlines()
             if line.strip()]
    other = sum(mib for pid, mib in procs if pid not in own)
    return {"gpu_used_mib": used, "gpu_total_mib": total, "gpu_other_mib": other,
            "gpu_procs": len(procs)}


def _smi(query: str) -> str:
    return subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True, timeout=20).stdout


def contention(own_pids: Optional[Iterable[int]] = None) -> dict:
    load1, load5, _ = os.getloadavg()
    out = {"load1": round(load1, 2), "load5": round(load5, 2)}
    try:
        gpu = parse_gpu(_smi("--query-gpu=memory.used,memory.total"),
                        _smi("--query-compute-apps=pid,used_memory"),
                        own_pids if own_pids is not None else {os.getpid()})
    except (OSError, subprocess.SubprocessError, ValueError):
        gpu = {"gpu_used_mib": None, "gpu_total_mib": None, "gpu_other_mib": None,
               "gpu_procs": None}
    return {**out, **gpu}


def require_vram(need_mib: int, readings: Optional[dict] = None) -> None:
    r = readings or contention()
    if r["gpu_total_mib"] is None:
        raise ResourceError("no GPU readings (nvidia-smi failed)")
    free = r["gpu_total_mib"] - r["gpu_used_mib"]
    if free < need_mib:
        raise ResourceError(f"{free} MiB VRAM free, need {need_mib} MiB "
                            f"({r['gpu_other_mib']} MiB held by other processes)")


def torch_info() -> dict:
    import torch

    return {"torch": torch.__version__, "cuda": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
