"""Our own auxiliary metrics. Official scores come from the evaluators (evaluators.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..geometry.sn_camera import ground_lines, project, scale_camera
from .adapters.calib import mirror

_SAMPLES = 60


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def _distances(lines: dict, cam: dict, width: int, height: int) -> list[float]:
    model = ground_lines()
    out: list[float] = []
    for name, pts in lines.items():
        if name not in model or not pts:
            continue
        a, b = (np.array(p) for p in model[name])
        w = np.linspace(0, 1, _SAMPLES)[:, None]
        poly = project(cam, np.c_[(1 - w) * a + w * b, np.zeros(_SAMPLES)])
        poly = poly[np.isfinite(poly).all(1)]           # drop the part behind the camera
        if len(poly) < 2:
            continue
        gt = np.array([[q["x"] * (width - 1), q["y"] * (height - 1)] for q in pts])
        d = np.min([_seg_dist(gt, poly[i], poly[i + 1]) for i in range(len(poly) - 1)], axis=0)
        out.extend(d.tolist())
    return out


def median_reprojection_px(gt_lines: dict, cam: dict, width: int = 960,
                           height: int = 540) -> float | None:
    """Median distance (px at width x height) from each labelled point of a straight
    ground-plane line to the camera's projection of that line, taking the better of the labels
    and their mirror as the official evaluator does. None if no such line is labelled."""
    c = scale_camera(cam, width)
    meds = [float(np.median(d)) for d in (_distances(gt_lines, c, width, height),
                                          _distances(mirror(gt_lines), c, width, height)) if d]
    return round(min(meds), 3) if meds else None


# ------------------------------------------------------------ full pipeline vs ground truth

_PLAYING = ("player", "goalkeeper")
_GT_SIDE = {"home": "left", "away": "right"}       # gt_to_state: left -> HOME, right -> AWAY
_SHAPE = ("centroid_x", "centroid_y", "width", "depth", "compactness")


def _playing(t: pd.DataFrame) -> pd.DataFrame:
    p = t[t["role"].astype(str).isin(_PLAYING)]
    return p[np.isfinite(p["x"].astype(float)) & np.isfinite(p["y"].astype(float))]


def match_people(gt: pd.DataFrame, pred: pd.DataFrame, frames,
                 max_dist: float = 2.0) -> pd.DataFrame:
    """Per frame, the Hungarian match of GT and predicted players (and keepers) on pitch
    distance, keeping pairs within `max_dist` metres."""
    from scipy.optimize import linear_sum_assignment

    g, p = _playing(gt), _playing(pred)
    gb, pb = dict(tuple(g.groupby("frame_idx"))), dict(tuple(p.groupby("frame_idx")))
    rows = []
    for f in frames:
        a, b = gb.get(f), pb.get(f)
        if a is None or b is None:
            continue
        d = np.hypot(a["x"].to_numpy(float)[:, None] - b["x"].to_numpy(float)[None],
                     a["y"].to_numpy(float)[:, None] - b["y"].to_numpy(float)[None])
        r, c = linear_sum_assignment(d)
        for i, j in zip(r, c):
            if d[i, j] <= max_dist:
                rows.append((f, a.index[i], b.index[j], float(d[i, j])))
    return pd.DataFrame(rows, columns=["frame_idx", "gt", "pred", "dist"])


def team_accuracy(gt: pd.DataFrame, pred: pd.DataFrame, matches: pd.DataFrame,
                  pred_sides: dict) -> float | None:
    """Share of matched players whose predicted side (our team through `pred_sides`) is the
    GT side. A HOME/AWAY label swap is not an error; a wrong side is."""
    ok = n = 0
    for gi, pi in zip(matches["gt"], matches["pred"]):
        side = _GT_SIDE.get(str(gt.at[gi, "team"]))
        if side is None:
            continue
        n += 1
        ok += pred_sides.get(str(pred.at[pi, "team"])) == side
    return round(ok / n, 4) if n else None


def _ball_centres(t: pd.DataFrame) -> dict:
    b = t[(t["role"].astype(str) == "ball") & ~t["interpolated"].astype(bool)]
    b = b[np.isfinite(b["bbox_x"].astype(float))]
    c = np.column_stack([b["bbox_x"] + b["bbox_w"] / 2, b["bbox_y"] + b["bbox_h"] / 2])
    return dict(zip(b["frame_idx"], c))


def ball_metrics(gt: pd.DataFrame, pred: pd.DataFrame, frames,
                 hit_px: float = 20.0) -> dict:
    """In the image (detection, not calibration): recall = share of GT-ball frames where our
    ball box centre is within `hit_px`; px median over frames where both have a box.
    Interpolated balls have no box and do not count."""
    g, p = _ball_centres(gt), _ball_centres(pred)
    gt_frames = [f for f in frames if f in g]
    d = [float(np.linalg.norm(g[f] - p[f])) for f in gt_frames if f in p]
    return {"ball_recall": round(sum(x <= hit_px for x in d) / len(gt_frames), 4)
            if gt_frames else None,
            "ball_px_median": round(float(np.median(d)), 2) if d else None}


def _shape(xy: np.ndarray) -> dict:
    c = xy.mean(axis=0)
    return {"centroid_x": c[0], "centroid_y": c[1], "width": np.ptp(xy[:, 1]),
            "depth": np.ptp(xy[:, 0]),
            "compactness": float(np.linalg.norm(xy - c, axis=1).mean())}


def shape_error(gt: pd.DataFrame, pred: pd.DataFrame, frames, pred_sides: dict,
                min_players: int = 6) -> dict:
    """Mean absolute error (m) of each side's shape (centroid, width, depth, compactness),
    per frame where both GT and prediction see at least `min_players` of that side."""
    g, p = _playing(gt), _playing(pred)
    g = g.assign(side=g["team"].astype(str).map(_GT_SIDE))
    p = p.assign(side=p["team"].astype(str).map(pred_sides))
    errs = {k: [] for k in _SHAPE}
    gb = dict(tuple(g.groupby(["frame_idx", "side"])))
    pb = dict(tuple(p.groupby(["frame_idx", "side"])))
    for f in frames:
        for side in ("left", "right"):
            a, b = gb.get((f, side)), pb.get((f, side))
            if a is None or b is None or len(a) < min_players or len(b) < min_players:
                continue
            sa = _shape(a[["x", "y"]].to_numpy(float))
            sb = _shape(b[["x", "y"]].to_numpy(float))
            for k in _SHAPE:
                errs[k].append(abs(sa[k] - sb[k]))
    out = {f"{k}_err": round(float(np.mean(v)), 3) if v else None for k, v in errs.items()}
    vals = [v for v in out.values() if v is not None]
    out["shape_error_m"] = round(float(np.mean(vals)), 3) if vals else None
    return out


def seconds_per_match_minute(seconds: float, frames: int, fps: float = 25.0) -> float:
    return round(seconds / (frames / fps / 60.0), 2) if frames else float("nan")
