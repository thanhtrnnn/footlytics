"""Locate or download the benchmark data on the host, pinned and checksummed.

GSR zips are kept whole and only their label files are extracted (images are read straight
out of the zip later), so the data is never held twice on a shared disk. Every download is
checked against the disk floor before it starts and against its sha256 after it ends.
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Callable

import requests

from . import host
from .adapters.gsr import check_version
from .paths import BenchPaths

GSR_REPO = "SoccerNet/SN-GSR-2025"
GSR_REVISION = "ce84ee7d9acb2f9fe999ceed2b38e303ffa5efb9"
CALIB_REPO = "SoccerNet/SN-Calibration-2023"
CALIB_REVISION = "bb74e1a888e33f839bcf8b7c6b87604a770f1da4"
#: The sn-gamestate baseline's own test predictions, as JSON (the Zenodo copy is a pickle).
BASELINE_URL = ("https://github.com/SoccerNet/sn-gamestate/raw/"
                "1c958345067218297d221e45e1a6405f975f83e0/examples_predictions/"
                "SoccerNetGS-test.zip")
BASELINE_SHA256 = "b51bea37ebd036ed95bd7cd89eea2d46134b6e3a704c02a7a30eaf6d91a734a5"

SOCCERMASTER_REPO = "xleprime/SoccerMaster"
SOCCERMASTER_REVISION = "0d94573662dbd678df22aa5c61ae77f474b939c9"
SOCCERMASTER_FILES = ("backbone.pt", "LinesDetection.pt", "KeypointsDetection.pt")
#: roboflow football-ball-detection (AGPL-3.0): internal benchmarking only.
BALL_REPO = "martinjolif/yolo-football-ball-detection"
BALL_REVISION = "63666b30e2a4973e089c982769f146af54530750"

#: Test seam: `(api, download)` used by fetch_hf when it is not given its own.
_HF = None

_LABELS = re.compile(r"^(SNGS-\d+)/Labels-GameState\.json$")


def sha256_file(path: str | Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _verified(dest: Path, sha: str) -> bool:
    marker = dest.with_name(dest.name + ".sha256")
    return dest.exists() and marker.exists() and marker.read_text().strip() == sha


def _verify(dest: Path, sha: str) -> Path:
    got = sha256_file(dest)
    if got != sha:
        dest.unlink()
        raise ValueError(f"sha256 mismatch for {dest.name}: got {got}, expected {sha}; deleted")
    dest.with_name(dest.name + ".sha256").write_text(sha)
    return dest


def fetch_hf(repo: str, filename: str, revision: str, dest: Path, api=None,
             download=None, repo_type: str = "dataset") -> Path:
    from huggingface_hub import HfApi, hf_hub_download

    api, download = api or (_HF[0] if _HF else None), download or (_HF[1] if _HF else None)
    api = api or HfApi()
    info = api.get_paths_info(repo, [filename], repo_type=repo_type, revision=revision)[0]
    if _verified(dest, info.lfs.sha256):
        return dest
    host.require_disk(dest.parent, info.size)
    dest.parent.mkdir(parents=True, exist_ok=True)
    got = Path((download or hf_hub_download)(repo_id=repo, filename=filename,
                                             repo_type=repo_type, revision=revision,
                                             local_dir=str(dest.parent)))
    if got != dest:
        got.replace(dest)
    return _verify(dest, info.lfs.sha256)


def fetch_url(url: str, dest: Path, sha256: str, get=requests.get) -> Path:
    if _verified(dest, sha256):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        host.require_disk(dest.parent, int(r.headers.get("Content-Length", 0)))
        with dest.open("wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
    return _verify(dest, sha256)


def extract_gsr_labels(zip_path: str | Path, out_dir: str | Path) -> list[str]:
    """Write `<out_dir>/<SEQ>/Labels-GameState.json` for every clip; images stay zipped."""
    out = Path(out_dir)
    seqs = []
    with zipfile.ZipFile(zip_path) as z:
        members = [(i, _LABELS.match(i.filename)) for i in z.infolist()]
        members = [(i, m) for i, m in members if m]
        host.require_disk(out, sum(i.file_size for i, _ in members))
        for info, m in members:
            raw = z.read(info)
            check_version(json.loads(raw))
            (out / m.group(1)).mkdir(parents=True, exist_ok=True)
            (out / m.group(1) / "Labels-GameState.json").write_bytes(raw)
            seqs.append(m.group(1))
    return sorted(seqs)


def clips(paths: BenchPaths, split: str) -> list[str]:
    d = paths.gsr_labels / split
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if (p / "Labels-GameState.json").exists())


def _gsr(split: str) -> Callable[[BenchPaths], Path]:
    def go(paths: BenchPaths) -> Path:
        z = fetch_hf(GSR_REPO, f"{split}.zip", GSR_REVISION, paths.gsr_zip(split))
        extract_gsr_labels(z, paths.gsr_labels / split)
        return z
    return go


def _calib(split: str) -> Callable[[BenchPaths], Path]:
    return lambda paths: fetch_hf(CALIB_REPO, f"{split}.zip", CALIB_REVISION,
                                  paths.calib_zip(split))


def _weights(paths: BenchPaths) -> Path:
    for f in SOCCERMASTER_FILES:
        fetch_hf(SOCCERMASTER_REPO, f, SOCCERMASTER_REVISION, paths.weights / f,
                 repo_type="model")
    return paths.weights


def _detector(paths: BenchPaths) -> Path:
    fetch_hf(SOCCERMASTER_REPO, "yolo_v8x6_finetuned.pt", SOCCERMASTER_REVISION,
             paths.detector_weights, repo_type="model")
    fetch_hf(BALL_REPO, "yolo-football-ball-detection.pt", BALL_REVISION, paths.ball_weights,
             repo_type="model")
    return paths.detector_weights.parent


FETCHABLE: dict[str, Callable[[BenchPaths], Path]] = {
    "gsr-valid": _gsr("valid"),
    "gsr-test": _gsr("test"),
    "calib-valid": _calib("valid"),
    "calib-test": _calib("test"),
    "soccermaster-weights": _weights,
    "detector-weights": _detector,
    "baseline": lambda paths: fetch_url(BASELINE_URL, paths.baseline_zip, BASELINE_SHA256),
}


def fetch(paths: BenchPaths, what: str) -> Path:
    if what not in FETCHABLE:
        raise KeyError(f"unknown dataset {what!r}; one of {sorted(FETCHABLE)}")
    return FETCHABLE[what](paths)
