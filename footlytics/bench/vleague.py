"""V.League mini-GT: sampling (frames, jitter windows, manifest), labels and scoring."""
from __future__ import annotations

import hashlib
import itertools
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


COORD_LIMIT = 10_000


def _check_point(name: str, uv, valid_names: set[str]) -> None:
    if name not in valid_names:
        raise ValueError(f"unknown landmark {name!r}")
    if not (isinstance(uv, (list, tuple)) and len(uv) == 2
            and all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in uv)):
        raise ValueError(f"{name}: a point is [u, v] numbers, got {str(uv)[:60]!r}")
    try:
        u, v = float(uv[0]), float(uv[1])
    except OverflowError:
        raise ValueError(f"{name}: coordinate too large for a pixel position") from None
    if not (math.isfinite(u) and math.isfinite(v)) or max(abs(u), abs(v)) > COORD_LIMIT:
        raise ValueError(f"{name}: coordinates must be finite and within +-{COORD_LIMIT} px")
    if not (0 <= u <= SIZE[0] and 0 <= v <= SIZE[1]):
        raise ValueError(f"{name}: {[u, v]} is outside the {SIZE[0]}x{SIZE[1]} image")


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
    try:
        with open(tmp, "w") as f:
            f.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def save_label(path, key: str, landmark: str, uv) -> dict:
    """Set (uv a point) or remove (uv None) one landmark of one frame; rewrite the file atomically.

    A frame with no points is never stored: deleting its last point drops the entry.
    """
    path = Path(path)
    if uv is not None:
        _check_point(landmark, uv, _landmark_names())
    elif landmark not in _landmark_names():
        raise ValueError(f"unknown landmark {landmark!r}")
    doc = json.loads(path.read_text())
    points = doc["frames"].get(key, {})
    if uv is None:
        points.pop(landmark, None)
    else:
        points[landmark] = list(uv)
    if points:
        doc["frames"][key] = points
    else:
        doc["frames"].pop(key, None)
    _write_atomic(path, doc)
    return doc


# ------------------------------------------------------------------- metrics
# Every H below maps image pixels (1920x1080) to pitch metres, the calibrator's
# `H_img_to_pitch`. Errors are in model metres (the 105 x 68 pitch model).

MOTION_SCALE = 0.5
MOTION_MIN_INLIERS = 30
MOTION_MIN_INLIER_RATIO = 0.5


def landmark_errors(H_img_to_pitch, points: dict, landmarks: dict) -> dict[str, float]:
    """Distance in metres, per labelled landmark, from its pixel mapped through H to its true spot.

    Landmarks whose error is not finite (a NaN pixel, a point on the horizon) are left out.
    """
    from ..geometry.homography import apply_homography

    unknown = set(points) - set(landmarks)
    if unknown:
        raise KeyError(f"unknown landmark name(s): {sorted(unknown)}")
    names = sorted(points)
    if not names:
        return {}
    got = apply_homography(H_img_to_pitch, np.array([points[n] for n in names], float))
    true = np.array([landmarks[n] for n in names], float)
    with np.errstate(invalid="ignore", over="ignore"):
        d = np.linalg.norm(got - true, axis=1)
    return {n: float(e) for n, e in zip(names, d) if np.isfinite(e)}


def frame_error(H_img_to_pitch, points: dict, landmarks: dict) -> float | None:
    """Median landmark error of a frame; None when uncalibrated or no finite error remains."""
    if H_img_to_pitch is None or not points:
        return None
    errs = landmark_errors(H_img_to_pitch, points, landmarks)
    return float(np.median(list(errs.values()))) if errs else None


def image_motion(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray | None:
    """Homography taking pixels of `prev_gray` to pixels of `gray`, by KLT and RANSAC.

    Tracked at half resolution and rescaled to the full-resolution pixel grid. None when fewer
    than 30 inliers survive or fewer than half of the tracks are inliers (a cut, or too little
    texture).
    """
    import cv2

    s = MOTION_SCALE
    a = cv2.resize(prev_gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    b = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    p0 = cv2.goodFeaturesToTrack(a, maxCorners=600, qualityLevel=0.01, minDistance=8)
    if p0 is None or len(p0) < MOTION_MIN_INLIERS:
        return None
    p1, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None)
    if p1 is None:
        return None
    ok = ok.ravel() == 1
    if ok.sum() < MOTION_MIN_INLIERS:
        return None
    M, inl = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 2.0)
    if M is None or inl is None:
        return None
    n_in = int(inl.sum())
    if n_in < MOTION_MIN_INLIERS or n_in < MOTION_MIN_INLIER_RATIO * int(ok.sum()):
        return None
    S = np.diag([s, s, 1.0])
    M = np.linalg.inv(S) @ M @ S
    return M / M[2, 2]


def _oriented(H, size) -> np.ndarray | None:
    """H with its projective sign fixed so that the image centre has a positive w; None when H is
    non-finite, singular, or has the centre on its horizon. (H and -H are the same mapping, but
    only one sign makes "w > 0" mean "in front of the camera".)"""
    H = np.asarray(H, float)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        return None
    try:
        np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    w = float(H[2] @ np.array([size[0] / 2, size[1] / 2, 1.0]))
    if not math.isfinite(w) or abs(w) < 1e-12:
        return None
    return H if w > 0 else -H


W_MIN_SHARE = 1e-3


def _project(H, pts) -> np.ndarray | None:
    """Points through H, or None when any lands behind the camera (w <= 0) or within 0.1 % of the
    median depth of the horizon (w near 0), or the result is not finite."""
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    w = hom[:, 2]
    if not np.isfinite(hom).all() or w.min() <= 0 or w.min() < W_MIN_SHARE * np.median(w):
        return None
    return hom[:, :2] / w[:, None]


def jitter_m(Hs: list, grays: list, pitch_pts: np.ndarray, sources: list[str]) -> dict:
    """Frame-to-frame jitter in metres of consecutive calibrations, against the image motion.

    For each pair with both H and a motion estimate M: x_t = inv(H_t) P, x_{t+1} = M x_t,
    P' = H_{t+1} x_{t+1}; the pair's jitter is mean |P' - P| over the points. A pair is a
    *switch* when either frame's source is "keyframe" (a fresh solve), otherwise *between*
    keyframes (both calibrations come from the camera tracker, so near zero by construction).

    A pair is skipped (counted in `pairs_skipped`) only when it is numerically degenerate: an H
    is missing, singular or non-finite, there is no motion estimate (a cut, low texture), a pitch
    point falls behind the camera or on the horizon, or a value is not finite. Large distances are
    never capped: a badly solved but finite keyframe must show up as large jitter. An empty
    bucket's mean is None.
    """
    if not (len(Hs) == len(grays) == len(sources)):
        raise ValueError("Hs, grays and sources must be the same length")
    P = np.asarray(pitch_pts, float).reshape(-1, 2)
    buckets: dict[str, list[float]] = {"switch": [], "between": []}
    skipped = 0
    for t in range(len(Hs) - 1):
        if Hs[t] is None or Hs[t + 1] is None:
            skipped += 1
            continue
        h, w = grays[t].shape[:2]
        H0, H1 = _oriented(Hs[t], (w, h)), _oriented(Hs[t + 1], (w, h))
        if H0 is None or H1 is None:
            skipped += 1
            continue
        M = image_motion(grays[t], grays[t + 1])
        if M is None:
            skipped += 1
            continue
        x_t = _project(np.linalg.inv(H0), P)
        x_next = None if x_t is None else _project(M, x_t)
        P2 = None if x_next is None else _project(H1, x_next)
        if P2 is None:
            skipped += 1
            continue
        d = float(np.mean(np.linalg.norm(P2 - P, axis=1)))
        if not math.isfinite(d):
            skipped += 1
            continue
        kind = "switch" if "keyframe" in (sources[t], sources[t + 1]) else "between"
        buckets[kind].append(d)
    return {"switch_m": float(np.mean(buckets["switch"])) if buckets["switch"] else None,
            "between_m": float(np.mean(buckets["between"])) if buckets["between"] else None,
            "pairs_switch": len(buckets["switch"]), "pairs_between": len(buckets["between"]),
            "pairs_skipped": skipped}


MIN_TRIPLE_AREA = 0.01       # smallest normalised triangle area of a 4-subset in general position
MIN_EXTENT = 0.01            # a 4-subset's largest pairwise distance, as a share of the scale


def _general_position(pts: np.ndarray, scale: float) -> bool:
    """True when no three of the 4 points are (near) collinear and they are not all close together.

    Each triple's area is divided by the squared extent (largest pairwise distance of the 4
    points) so it is scale free: 0 for three collinear points, 0.433 for an equilateral
    triangle spanning the extent. Calibrated on exact broadcast projections: a penalty box gives
    0.175 on the pitch and 0.046-0.049 in pixels; three halfway-line points plus any other
    point give exactly 0, and 3 px of click noise across a 340 px line gives about 0.0005.
    """
    ext = max(float(np.linalg.norm(pts[i] - pts[j])) for i, j in itertools.combinations(range(4), 2))
    if ext < MIN_EXTENT * scale:
        return False
    for i, j, k in itertools.combinations(range(4), 3):
        d1, d2 = pts[j] - pts[i], pts[k] - pts[i]
        if abs(d1[0] * d2[1] - d1[1] * d2[0]) / 2 / ext**2 < MIN_TRIPLE_AREA:
            return False
    return True


def _has_general_position_subset(pix: np.ndarray, pitch: np.ndarray, pix_scale: float,
                                 pitch_scale: float) -> bool:
    return any(_general_position(pix[list(c)], pix_scale)
               and _general_position(pitch[list(c)], pitch_scale)
               for c in itertools.combinations(range(len(pix)), 4))


def labelling_error_m(a: dict, b: dict, landmarks: dict, image_size) -> float | None:
    """Disagreement of two labellings of one frame, in model metres (not symmetric in a and b).

    Fits a homography to A on the landmarks both labelled (at least 4 with finite pixels), maps
    A's and B's pixels through it, and returns the median distance between the two sets. None
    when fewer than 4 landmarks are shared, when no 4 of them are in general position (no three
    near-collinear) both in pixels and on the pitch (the fit would be arbitrary, as with the
    halfway line plus one point), or when no finite distance remains.
    """
    from ..geometry.homography import Calibration, apply_homography

    shared = [n for n in sorted(set(a) & set(b))
              if np.isfinite(np.asarray(a[n], float)).all()
              and np.isfinite(np.asarray(b[n], float)).all()]
    if len(shared) < 4:
        return None
    pa = np.array([a[n] for n in shared], float)
    pb = np.array([b[n] for n in shared], float)
    pitch = np.array([landmarks[n] for n in shared], float)
    all_pitch = np.array(list(landmarks.values()), float)
    if not _has_general_position_subset(pa, pitch, math.hypot(*image_size),
                                        float(np.linalg.norm(np.ptp(all_pitch, axis=0)))):
        return None
    try:
        cal = Calibration.from_correspondences({n: tuple(a[n]) for n in shared}, landmarks,
                                               tuple(image_size))
    except np.linalg.LinAlgError:
        return None
    with np.errstate(invalid="ignore", over="ignore"):
        d = np.linalg.norm(apply_homography(cal.H, pa) - apply_homography(cal.H, pb), axis=1)
    d = d[np.isfinite(d)]
    return float(np.median(d)) if len(d) else None
