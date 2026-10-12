"""V.League mini-GT: sampling (frames, jitter windows, manifest), labels and scoring."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from pathlib import Path
from typing import Callable

import numpy as np

WINDOW = 12
SIZE = (1920, 1080)
PER_MATCH = 15
PITCH_MIN_SHARE = 0.35


def _slug(text: str) -> str:
    text = text.replace("Đ", "D").replace("đ", "d")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def match_id_from_name(name: str) -> str:
    """'SHB ĐÀ NẴNG - HẢI PHÒNG - LPBANK V.LEAGUE 1 - 2025-26.mp4' -> 'shb-da-nang-hai-phong'."""
    teams = Path(name).stem.split(" - LPBANK")[0]
    return "-".join(_slug(t) for t in teams.split(" - "))


def pitch_share(frame_bgr: np.ndarray) -> float:
    import cv2

    from ..geometry.calib_assist import _pitch_mask

    return float((_pitch_mask(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)) > 0).mean())


def choose_frames(n_frames: int, fps: float, is_pitch: Callable[[int], bool],
                  k: int = PER_MATCH, step_s: float = 30.0) -> list[int]:
    step = max(1, round(step_s * fps))
    cands = [i for i in range(WINDOW, n_frames - WINDOW, step) if is_pitch(i)]
    if len(cands) <= k:
        return cands
    pick = np.linspace(0, len(cands) - 1, k).round().astype(int)
    return [cands[i] for i in sorted(set(pick.tolist()))]


def assign_splits(keys: list[str]) -> dict[str, dict]:
    """Rank within each match: even ranks dev, odd frozen; rank % 4 == 1 is labelled twice."""
    by_match: dict[str, list[str]] = {}
    for key in sorted(keys):
        by_match.setdefault(key.rsplit("/", 1)[0], []).append(key)
    out: dict[str, dict] = {}
    for ks in by_match.values():
        for rank, key in enumerate(ks):
            out[key] = {"split": "dev" if rank % 2 == 0 else "frozen", "relabel": rank % 4 == 1}
    return out


def make_is_pitch(cap) -> Callable[[int], bool]:
    import cv2

    def is_pitch(idx: int) -> bool:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        return bool(ok) and pitch_share(frame) >= PITCH_MIN_SHARE
    return is_pitch


def extract_windows(video, frames: list[int], out_dir, size=SIZE) -> None:
    """Write out_dir/<idx:07d>/<offset:+03d>.jpg for offsets -WINDOW..+WINDOW, skipping existing."""
    import cv2

    out_dir = Path(out_dir)
    cap = cv2.VideoCapture(str(video))
    try:
        for idx in frames:
            d = out_dir / f"{idx:07d}"
            targets = {off: d / f"{off:+03d}.jpg" for off in range(-WINDOW, WINDOW + 1)}
            if all(p.exists() for p in targets.values()):
                continue
            d.mkdir(parents=True, exist_ok=True)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx - WINDOW)
            for off, path in targets.items():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"{video}: cannot read frame {idx + off}")
                if path.exists():
                    continue
                img = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                tmp = path.with_name(f".tmp-{path.name}")
                if not cv2.imwrite(str(tmp), img, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(f"cannot write {path}")
                os.replace(tmp, path)
    finally:
        cap.release()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sample_match(video, match_id: str, data_dir, k: int | None = None, step_s: float = 30.0,
                 existing: dict | None = None) -> dict:
    """Pick the frames, write their jitter windows under data_dir/<match_id>, return the entry.

    With `existing` (the match's manifest entry) the frame set is reused, never re-chosen, so a
    different OpenCV/ffmpeg build cannot reshuffle dev/frozen. A changed video raises.
    """
    import cv2

    k = PER_MATCH if k is None else k
    video = Path(video)
    if existing is not None:
        sha = _sha256(video)
        if sha != existing["sha256"]:
            raise ValueError(f"{video.name}: video changed (sha256 {sha[:12]} != manifest "
                             f"{existing['sha256'][:12]}); refusing to resample")
        extract_windows(video, [int(i) for i in existing["frames"]], Path(data_dir) / match_id)
        return existing
    cap = cv2.VideoCapture(str(video))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not cap.isOpened() or n <= 2 * WINDOW or fps <= 0:
            raise ValueError(f"{video.name}: cannot open as a video (frames={n}, fps={fps})")
        frames = choose_frames(n, fps, make_is_pitch(cap), k=k, step_s=step_s)
    finally:
        cap.release()
    if len(frames) < k:
        raise ValueError(f"{video.name}: only {len(frames)} of {k} frames passed the pitch test")
    sha = _sha256(video)
    extract_windows(video, frames, Path(data_dir) / match_id)
    splits = assign_splits([f"{match_id}/{i:07d}" for i in frames])
    return {"video": video.name, "sha256": sha, "fps": fps,
            "frames": {f"{i:07d}": splits[f"{match_id}/{i:07d}"] for i in frames}}


def read_manifest(path) -> dict[str, dict]:
    """The matches in an existing manifest file ({} when there is none)."""
    path = Path(path)
    return json.loads(path.read_text())["matches"] if path.exists() else {}


def write_manifest(entries: dict[str, dict], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"version": 1, "size": list(SIZE), "window": WINDOW, "matches": entries}
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


# --- labels ----------------------------------------------------------------------------

LABELS_VERSION = 1


def frame_key(match_id: str, idx: int) -> str:
    return f"{match_id}/{idx:07d}"


def manifest_frames(manifest: dict) -> dict[str, dict]:
    """{frame_key: {"split", "relabel"}} over every match of a manifest document."""
    return {frame_key(mid, int(i)): meta
            for mid, m in manifest["matches"].items() for i, meta in m["frames"].items()}


def _landmark_names() -> set[str]:
    from ..geometry.pitch import DEFAULT_PITCH

    return set(DEFAULT_PITCH.landmarks())


def _check_point(name: str, uv, valid_names: set[str]) -> None:
    if name not in valid_names:
        raise ValueError(f"unknown landmark {name!r}")
    ok = (isinstance(uv, (list, tuple)) and len(uv) == 2
          and all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in uv)
          and all(math.isfinite(c) for c in uv))
    if not ok:
        raise ValueError(f"{name}: a point is [u, v] numbers, got {uv!r}")
    if not (0 <= uv[0] <= SIZE[0] and 0 <= uv[1] <= SIZE[1]):
        raise ValueError(f"{name}: {list(uv)} is outside the {SIZE[0]}x{SIZE[1]} image")


def new_labels(annotator: str, session: int) -> dict:
    return {"version": LABELS_VERSION, "annotator": annotator, "session": session,
            "image_size": list(SIZE), "frames": {}}


def _validate_labels(doc: dict, manifest: dict | None) -> None:
    if doc.get("version") != LABELS_VERSION:
        raise ValueError(f"labels version {doc.get('version')!r}, expected {LABELS_VERSION}")
    if doc.get("image_size") != list(SIZE):
        raise ValueError(f"labels image_size {doc.get('image_size')!r}, expected {list(SIZE)}")
    names = _landmark_names()
    known = manifest_frames(manifest) if manifest is not None else None
    for key, points in doc["frames"].items():
        if known is not None and key not in known:
            raise ValueError(f"frame {key!r} is not in the manifest")
        for name, uv in points.items():
            _check_point(name, uv, names)


def load_labels(path, manifest: dict) -> dict:
    """Read and validate a label file against a manifest document."""
    doc = json.loads(Path(path).read_text())
    _validate_labels(doc, manifest)
    return doc


def _write_atomic(path: Path, doc: dict) -> None:
    tmp = path.with_name(f".tmp-{path.name}")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def save_label(path, key: str, landmark: str, uv) -> dict:
    """Set (uv a point) or remove (uv None) one landmark of one frame; rewrite the file atomically."""
    path = Path(path)
    if uv is not None:
        _check_point(landmark, uv, _landmark_names())
    elif landmark not in _landmark_names():
        raise ValueError(f"unknown landmark {landmark!r}")
    doc = json.loads(path.read_text())
    points = doc["frames"].setdefault(key, {})
    if uv is None:
        points.pop(landmark, None)
    else:
        points[landmark] = list(uv)
    _write_atomic(path, doc)
    return doc
