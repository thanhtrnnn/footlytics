"""Our own auxiliary metrics. Official scores come from the evaluators (evaluators.py)."""
from __future__ import annotations

from collections import Counter

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
ROLE_CLASSES = ("player", "goalkeeper", "referee")
_GT_SIDE = {"home": "left", "away": "right"}       # gt_to_state: left -> HOME, right -> AWAY
_SHAPE = ("centroid_x", "centroid_y", "width", "depth", "compactness")
#: A predicted person within MATCH_M metres of a GT one is that person. Position error looks out
#: to POS_GATE_M, so a player we place 3 m off still has an error rather than none.
MATCH_M, POS_GATE_M = 2.0, 5.0


def _on_pitch(t: pd.DataFrame, roles=_PLAYING) -> pd.DataFrame:
    p = t[t["role"].astype(str).isin(roles)]
    return p[np.isfinite(p["x"].astype(float)) & np.isfinite(p["y"].astype(float))]


def share(ok: int, n: int) -> float | None:
    return round(ok / n, 4) if n else None


def match_people(gt: pd.DataFrame, pred: pd.DataFrame, frames, max_dist: float = MATCH_M,
                 roles=_PLAYING) -> pd.DataFrame:
    """Per frame, the Hungarian match of GT and predicted people of `roles` (players and
    keepers by default) on pitch distance, keeping pairs within `max_dist` metres. The
    assignment ignores `max_dist`: a tighter gate only drops pairs."""
    from scipy.optimize import linear_sum_assignment

    g, p = _on_pitch(gt, roles), _on_pitch(pred, roles)
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


def _side_hits(gt: pd.DataFrame, pred: pd.DataFrame, matches: pd.DataFrame,
               pred_sides: dict) -> tuple[int, int]:
    ok = n = 0
    for gi, pi in zip(matches["gt"], matches["pred"]):
        side = _GT_SIDE.get(str(gt.at[gi, "team"]))
        if side is None:
            continue
        n += 1
        ok += pred_sides.get(str(pred.at[pi, "team"])) == side
    return ok, n


def team_accuracy(gt: pd.DataFrame, pred: pd.DataFrame, matches: pd.DataFrame,
                  pred_sides: dict) -> float | None:
    """Share of matched players whose predicted side (our team through `pred_sides`) is the
    GT side. A HOME/AWAY label swap is not an error; a wrong side is."""
    return share(*_side_hits(gt, pred, matches, pred_sides))


def keeper_side_accuracy(gt: pd.DataFrame, pred: pd.DataFrame, matches: pd.DataFrame,
                         pred_sides: dict) -> tuple[int, int]:
    """`(ok, n)` over the matched GT goalkeepers: how many we put on their GT side. A keeper
    in the other team's colour cluster drags `team_sides` and can flip a whole clip. Raw, to
    pool across clips: a clip sees its keepers in few frames."""
    keepers = gt.loc[matches["gt"], "role"].astype(str).to_numpy() == "goalkeeper"
    return _side_hits(gt, pred, matches[keepers], pred_sides)


def role_counts(gt: pd.DataFrame, pred: pd.DataFrame,
                matches: pd.DataFrame) -> Counter[tuple[str, str]]:
    """Matched rows per (GT role, predicted role), over a match of `ROLE_CLASSES`."""
    return Counter(zip(gt.loc[matches["gt"], "role"].astype(str),
                       pred.loc[matches["pred"], "role"].astype(str)))


def role_summary(counts: dict[tuple[str, str], int]) -> dict:
    """The role numbers from `role_counts`, one clip's or pooled:

    - `keeper_recall`: share of matched GT goalkeepers we call goalkeepers.
    - `referee_recall`: share of matched GT referees we call referees.
    - `referee_precision`: share of the matched people we call referees who are referees.
    """
    gt_n, pred_n = Counter(), Counter()
    for (g, p), n in counts.items():
        gt_n[g] += n
        pred_n[p] += n
    hits = {r: counts.get((r, r), 0) for r in ("goalkeeper", "referee")}
    return {"keeper_recall": share(hits["goalkeeper"], gt_n["goalkeeper"]),
            "referee_recall": share(hits["referee"], gt_n["referee"]),
            "referee_precision": share(hits["referee"], pred_n["referee"])}


def position_summary(dists, gt_people: int) -> dict:
    """The position numbers from matched distances (m, players and keepers within
    `POS_GATE_M`) over `gt_people` located GT players and keepers. Both may be pooled across
    clips: a clip has no number of its own that averages correctly.

    - `pos_err_median_m`, `pos_err_p90_m`: how far a matched player sits from the truth.
    - `recall_2m`: share of GT people with a predicted one within `MATCH_M` (detection and
      mapping together, before teams).
    """
    found = sum(d <= MATCH_M for d in dists)
    return {"pos_err_median_m": round(float(np.median(dists)), 3) if len(dists) else None,
            "pos_err_p90_m": round(float(np.percentile(dists, 90)), 3) if len(dists) else None,
            "recall_2m": share(found, gt_people)}


def position_metrics(gt: pd.DataFrame, matches: pd.DataFrame, frames) -> dict:
    """`position_summary` of one clip from its `POS_GATE_M` match over `frames`. `_pos_dists`
    and `_gt_people` are the raw values to pool across clips; they are not metrics to store."""
    g = _on_pitch(gt)
    n = int(g["frame_idx"].isin(list(frames)).sum())
    d = matches["dist"].astype(float).tolist()
    return {**position_summary(d, n), "_pos_dists": d, "_gt_people": n}


def _ball_centres(t: pd.DataFrame) -> dict:
    b = t[(t["role"].astype(str) == "ball") & ~t["interpolated"].astype(bool)]
    b = b[np.isfinite(b["bbox_x"].astype(float))]
    c = np.column_stack([b["bbox_x"] + b["bbox_w"] / 2, b["bbox_y"] + b["bbox_h"] / 2])
    return dict(zip(b["frame_idx"], c))


BALL_HIT_PX = 20.0


def ball_summary(dists, gt_frames: int, hit_px: float = BALL_HIT_PX) -> dict:
    """The ball numbers from per-frame distances (px, GT centre to ours) over `gt_frames`
    GT-ball frames. The distances may be pooled across clips: a clip has no number of its own
    that averages correctly.

    - `ball_recall`: share of GT-ball frames where our box centre is within `hit_px`.
    - `ball_px_median_hits`: median distance over those hits (how well a found ball is placed).
    - `ball_wrong_share`: share of GT-ball frames where we have a box but farther than `hit_px`
      (the pipeline followed the wrong object). The rest are frames with no box at all.
    """
    hits = [x for x in dists if x <= hit_px]
    if not gt_frames:
        return {"ball_recall": None, "ball_px_median_hits": None, "ball_wrong_share": None}
    return {"ball_recall": round(len(hits) / gt_frames, 4),
            "ball_px_median_hits": round(float(np.median(hits)), 2) if hits else None,
            "ball_wrong_share": round((len(dists) - len(hits)) / gt_frames, 4)}


def ball_metrics(gt: pd.DataFrame, pred: pd.DataFrame, frames,
                 hit_px: float = BALL_HIT_PX) -> dict:
    """End to end, in the image: our ball box against the GT ball box on each GT-ball frame (see
    `ball_summary`). It judges the whole pipeline, not the detector alone: a frame without a
    calibration drops every detection, and the on-pitch filter and the ball selector work on
    pitch positions. Interpolated balls have no box and do not count as a box.

    `_dists` (the per-frame distances where we have a box) and `_gt_frames` (the GT-ball frame
    count) are the raw values to pool across clips; they are not metrics to store.
    """
    g, p = _ball_centres(gt), _ball_centres(pred)
    gt_frames = [f for f in frames if f in g]
    d = [float(np.linalg.norm(g[f] - p[f])) for f in gt_frames if f in p]
    return {**ball_summary(d, len(gt_frames), hit_px), "_dists": d, "_gt_frames": len(gt_frames)}


def _shape(xy: np.ndarray) -> dict:
    c = xy.mean(axis=0)
    return {"centroid_x": c[0], "centroid_y": c[1], "width": np.ptp(xy[:, 1]),
            "depth": np.ptp(xy[:, 0]),
            "compactness": float(np.linalg.norm(xy - c, axis=1).mean())}


def shape_error(gt: pd.DataFrame, pred: pd.DataFrame, frames, pred_sides: dict,
                min_players: int = 6) -> dict:
    """Mean absolute error (m) of each side's shape (centroid, width, depth, compactness),
    per frame where both GT and prediction see at least `min_players` of that side."""
    g, p = _on_pitch(gt), _on_pitch(pred)
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
