"""Flicker metrics: how steady a saved match state looks, measured from `tracks.parquet` alone.

HOTA scores identity and position against ground truth. It does not notice a radar that
shivers: a player who jumps a metre and back every few frames, a whole pitch that shifts
when a keyframe lands, ids that die and are reborn, a ball that flies 30 m in four frames. Those
make the product look broken and cost nothing on the scoreboard, so this module counts them.

Everything is computed on live-wide frames (replays and close-ups are not the tactical layer) and
on steps inside one shot. A step is the move from frame f-1 to frame f; it belongs to the
*keyframe* phase when `f % every == 0`, the frame where a fresh calibration lands, and to the
*between* phase otherwise. Step metrics are reported overall and per phase (`_kf`, `_between`):
a calibration that only misbehaves at its keyframes shows up there and not in the average.

The pass thresholds are `THRESHOLDS`, one table. `evaluate` marks each pass or fail; nothing
here raises on a fail, because today's outputs fail most of them and the point is to watch them
fall, not to block a run.
"""
from __future__ import annotations

import json
import operator
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..analytics.ball_link import BALL_MAX_SPEED, BALL_NOISE_M, ball_reach

PERSON = ("player", "goalkeeper")
#: Ball speed (m/s) above which a step is not a ball. Matches `schema.BALL_STEP_LIMIT_MS`.
BALL_STEP_MS = 40.0
#: Player step speeds (m/s) counted as impossible (Bolt peaked at ~12.4) and as a teleport.
PLAYER_STEP_MS = (12.0, 20.0)
#: A one-frame position error shared by every player (homography, camera) above this is visible.
COMMON_MODE_M = 0.5
STILL_WINDOW = 30            # frames
STILL_DRIFT_M = 0.5          # a window whose position moved less than this is a player standing still
PAIR_M, PAIR_FRAMES = 0.6, 10   # same-team players closer than this for this many frames: a duplicate
HOLE_FRAMES = 8              # a whole-frame radar hole longer than this, inside one shot
BLINK_FRAMES = 3             # 1..3 missing frames inside one id
TEAM_SIZE = 11
MAX_REFEREES = 4             # referee, two assistants, the fourth official
GAP_BINS = ((1, 1), (2, 5), (6, 15), (16, 30), (31, 90), (91, None))

#: metric -> (operator, limit). A check passes when `value <op> limit`.
THRESHOLDS: dict[str, tuple[str, float]] = {
    "player_step_gt12":      ("<=", 0.005),    # share of player steps implying > 12 m/s
    "player_step_gt20":      ("<=", 0.0005),   # ... > 20 m/s
    "ball_step_gt40":        ("<=", 0.002),    # share of ball steps (non-interpolated) > 40 m/s
    "ball_beyond_gate":      ("<=", 0),        # same-id ball steps beyond the link gate: count
    "common_mode_gt05":      ("<=", 0.002),    # share of frames with a common-mode step > 0.5 m
    "still_sigma_p50":       ("<=", 0.04),     # m
    "still_sigma_p90":       ("<=", 0.10),     # m
    "births_per_min":        ("<=", 15.0),
    "id_life_median_s":      (">=", 20.0),
    "frames_over_11_share":  ("<=", 0.0),      # frames with more than 11 on one team
}
_OPS = {"<=": operator.le, ">=": operator.ge}


# ---------------------------------------------------------------------- helpers


def _prepare(tracks: pd.DataFrame) -> pd.DataFrame:
    """Live-wide rows only, with the optional columns defaulted and roles as plain strings."""
    t = tracks
    if "live_wide" in t.columns:
        t = t[t["live_wide"].to_numpy(bool)]
    t = t.copy()
    t["role"] = t["role"].astype(str)
    t["team"] = t["team"].astype(str) if "team" in t.columns else "unknown"
    if "shot" not in t.columns:
        t["shot"] = 0
    if "interpolated" not in t.columns:
        t["interpolated"] = False
    t["interpolated"] = t["interpolated"].fillna(False).astype(bool)
    t["frame_idx"] = t["frame_idx"].astype(np.int64)
    t["track_id"] = t["track_id"].astype(np.int64)
    return t[t["track_id"] >= 0] if (t["track_id"] < 0).any() else t


def _share(mask: np.ndarray) -> float | None:
    return float(mask.mean()) if len(mask) else None


def _q(a: np.ndarray, q: float) -> float | None:
    return float(np.percentile(a, q)) if len(a) else None


def _phase_split(name: str, frames: np.ndarray, values: np.ndarray, every: int, fn) -> dict:
    """`fn(values)` overall, on keyframe-phase steps (`_kf`) and on the others (`_between`)."""
    kf = (frames % every) == 0
    return {name: fn(values), f"{name}_kf": fn(values[kf]), f"{name}_between": fn(values[~kf])}


def _steps(p: pd.DataFrame) -> pd.DataFrame:
    """Consecutive-frame steps of every track in `p`: frame (destination), dx, dy, d, shot."""
    p = p.sort_values(["track_id", "frame_idx"], kind="stable")
    tid, f, sh = (p[c].to_numpy() for c in ("track_id", "frame_idx", "shot"))
    x, y = p["x"].to_numpy(float), p["y"].to_numpy(float)
    ok = (tid[1:] == tid[:-1]) & (f[1:] - f[:-1] == 1) & (sh[1:] == sh[:-1])
    dx, dy = np.diff(x)[ok], np.diff(y)[ok]
    return pd.DataFrame({"track_id": tid[1:][ok], "frame": f[1:][ok], "dx": dx, "dy": dy,
                         "d": np.hypot(dx, dy)})


# ---------------------------------------------------------------------- the metric groups


def player_motion(p: pd.DataFrame, fps: float, every: int) -> dict:
    """Step speed shares > 12 and > 20 m/s, raw acceleration p99, and the common-mode step."""
    out: dict = {}
    s = _steps(p)
    v = s["d"].to_numpy() * fps
    out["player_steps"] = int(len(s))
    for lim in PLAYER_STEP_MS:
        out.update(_phase_split(f"player_step_gt{int(lim)}", s["frame"].to_numpy(), v > lim, every,
                                _share))
    # raw acceleration: change of the step vector between two consecutive steps of one track
    if len(s) > 1:
        tid, fr = s["track_id"].to_numpy(), s["frame"].to_numpy()
        two = (tid[1:] == tid[:-1]) & (fr[1:] - fr[:-1] == 1)
        ax = np.diff(s["dx"].to_numpy())[two] * fps * fps
        ay = np.diff(s["dy"].to_numpy())[two] * fps * fps
        out.update(_phase_split("player_accel_p99", fr[1:][two], np.hypot(ax, ay), every,
                                lambda a: _q(a, 99)))
    else:
        out.update(player_accel_p99=None, player_accel_p99_kf=None, player_accel_p99_between=None)
    # common mode: the median step vector across the players of one frame
    cm = (s.groupby("frame").agg(n=("dx", "size"), cx=("dx", "median"), cy=("dy", "median"))
            .query("n >= 5"))
    mag = np.hypot(cm["cx"].to_numpy(), cm["cy"].to_numpy())
    fr = cm.index.to_numpy()
    out["common_mode_frames"] = int(len(cm))
    out.update(_phase_split("common_mode_gt05", fr, mag > COMMON_MODE_M, every, _share))
    out.update(_phase_split("common_mode_p99", fr, mag, every, lambda a: _q(a, 99)))
    # duplicate video frames: nearly every matched player exactly where it was a frame ago
    frozen = (s.assign(z=s["d"] < 1e-3).groupby("frame").agg(n=("z", "size"), z=("z", "mean"))
                .query("n >= 5"))
    out["duplicate_frame_share"] = _share(frozen["z"].to_numpy() >= 0.9)
    return out


def still_jitter(p: pd.DataFrame) -> dict:
    """Detrended radial rms (m) of players standing still, in 30-frame windows of one track.

    A window is "still" when the median of its first 5 frames and of its last 5 are less than
    `STILL_DRIFT_M` apart. Nothing real moves a still player by centimetres, so this is the
    homography plus foot-point noise on its own.
    """
    p = p.sort_values(["track_id", "frame_idx"], kind="stable")
    tid, f, sh = (p[c].to_numpy() for c in ("track_id", "frame_idx", "shot"))
    x, y = p["x"].to_numpy(float), p["y"].to_numpy(float)
    brk = np.r_[True, (tid[1:] != tid[:-1]) | (f[1:] - f[:-1] != 1) | (sh[1:] != sh[:-1])]
    starts = np.flatnonzero(brk)
    ends = np.r_[starts[1:], len(f)]
    n = STILL_WINDOW
    t = np.arange(n, dtype=float)
    A = np.vstack([t, np.ones(n)]).T
    resid_proj = np.eye(n) - A @ np.linalg.pinv(A)            # removes a straight line
    wx, wy = [], []
    for a, b in zip(starts, ends):
        k = (b - a) // n
        if k:
            wx.append(x[a:a + k * n].reshape(k, n))
            wy.append(y[a:a + k * n].reshape(k, n))
    if not wx:
        return {"still_windows": 0, "still_sigma_p50": None, "still_sigma_p90": None}
    wx, wy = np.vstack(wx), np.vstack(wy)
    drift = np.hypot(np.median(wx[:, -5:], axis=1) - np.median(wx[:, :5], axis=1),
                     np.median(wy[:, -5:], axis=1) - np.median(wy[:, :5], axis=1))
    still = drift < STILL_DRIFT_M
    rx, ry = wx[still] @ resid_proj.T, wy[still] @ resid_proj.T
    sigma = np.sqrt((rx ** 2).mean(axis=1) + (ry ** 2).mean(axis=1))
    return {"still_windows_total": int(len(wx)), "still_windows": int(still.sum()),
            "still_sigma_p50": _q(sigma, 50), "still_sigma_p90": _q(sigma, 90)}


def ball_metrics(t: pd.DataFrame, live_frames: np.ndarray, fps: float, every: int) -> dict:
    """Ball step speed, same-id steps beyond the link gate, presence and gap lengths."""
    b = t[t["role"] == "ball"]
    b = (b.sort_values(["frame_idx", "det_conf"] if "det_conf" in b.columns else ["frame_idx"])
          .drop_duplicates("frame_idx", keep="last"))
    out: dict = {"ball_frames": int(len(b))}
    out["ball_presence"] = float(len(b) / len(live_frames)) if len(live_frames) else None
    det = b[~b["interpolated"]]
    out["ball_presence_detected"] = float(len(det) / len(live_frames)) if len(live_frames) else None
    out["ball_interpolated_share"] = float(b["interpolated"].mean()) if len(b) else None
    f = b["frame_idx"].to_numpy()
    x, y = b["x"].to_numpy(float), b["y"].to_numpy(float)
    sh, tid, it = b["shot"].to_numpy(), b["track_id"].to_numpy(), b["interpolated"].to_numpy()
    same_shot = sh[1:] == sh[:-1]
    df = f[1:] - f[:-1]
    d = np.hypot(np.diff(x), np.diff(y))
    # one-frame steps between two real detections
    one = (df == 1) & same_shot & ~(it[1:] | it[:-1])
    out["ball_steps"] = int(one.sum())
    out.update(_phase_split("ball_step_gt40", f[1:][one], d[one] * fps > BALL_STEP_MS, every,
                            _share))
    # same-id steps beyond the gate the selector uses (interpolated rows included: filling a gap
    # is itself a claim that the ball moved). Also against the old `35 m/s * dt + 1 m` gate.
    # (no shot condition: a ball id that carries across a cut is itself the defect, and rows
    # filled by the old `interpolate_ball` were all written with shot 0)
    same_id = (tid[1:] == tid[:-1]) & (df >= 1)
    dt = df / fps
    out["ball_beyond_gate"] = int((same_id & (d > ball_reach(dt, BALL_MAX_SPEED, BALL_NOISE_M))).sum())
    out["ball_beyond_gate_legacy"] = int((same_id & (d > BALL_MAX_SPEED * dt + 1.0)).sum())
    out["ball_beyond_gate_interpolated"] = int(
        (same_id & (d > ball_reach(dt, BALL_MAX_SPEED, BALL_NOISE_M)) & (it[1:] | it[:-1])).sum())
    # gaps: frames missing between two sightings in one shot
    gap = (df - 1)[(same_shot | it[1:] | it[:-1]) & (df > 1)]
    out["ball_gaps"] = int(len(gap))
    out["ball_gap_p50"], out["ball_gap_p90"], out["ball_gap_p99"] = (_q(gap, q) for q in (50, 90, 99))
    out["ball_gap_hist"] = {
        (f"{lo}" if lo == hi else f"{lo}-{hi}" if hi else f">{lo - 1}"):
        int(((gap >= lo) & (gap <= (hi if hi else np.inf))).sum()) for lo, hi in GAP_BINS}
    return out


def lifecycle(t: pd.DataFrame, fps: float, minutes: float) -> dict:
    """Births, id life, blinks, mass rebirths, over-full teams, duplicates, referees, holes."""
    out: dict = {}
    p = t[t["role"].isin(PERSON)]
    first_frame = int(t["frame_idx"].min())
    g = p.groupby("track_id")["frame_idx"]
    born = g.min()
    life = g.size() / fps
    pmin = len(p) / fps / 60 if len(p) else 0.0
    births = born[born > first_frame]
    out["person_ids"] = int(len(born))
    out["births_per_min"] = float(len(births) / minutes) if minutes else None
    out["id_life_median_s"] = float(life.median()) if len(life) else None
    out["id_life_p10_s"] = _q(life.to_numpy(), 10)
    # 1..3 missing frames inside an id (inside one shot): the dot blinks
    q = p.sort_values(["track_id", "frame_idx"], kind="stable")
    tid, f, sh = (q[c].to_numpy() for c in ("track_id", "frame_idx", "shot"))
    df = f[1:] - f[:-1]
    blink = (tid[1:] == tid[:-1]) & (sh[1:] == sh[:-1]) & (df >= 2) & (df <= BLINK_FRAMES + 1)
    out["blinks_per_player_min"] = float(blink.sum() / pmin) if pmin else None
    # frames where half the ids on the pitch were born that frame: a tracker reset
    per_frame = p.groupby("frame_idx").size()
    born_at = births.value_counts()
    nb = born_at.reindex(per_frame.index, fill_value=0)
    mass = (nb / per_frame >= 0.5) & (per_frame >= 4)
    out["mass_birth_frames"] = int(mass.sum())
    # more than 11 on one team in a frame
    tm = p[p["team"].isin(["home", "away"])].groupby(["frame_idx", "team"]).size().unstack(fill_value=0)
    over = (tm > TEAM_SIZE).any(axis=1) if len(tm) else pd.Series(dtype=bool)
    n_frames = per_frame.index.nunique()
    out["frames_over_11"] = int(over.sum())
    out["frames_over_11_share"] = float(over.sum() / n_frames) if n_frames else None
    out.update(pair_duplicates(p))
    # referees
    r = t[t["role"] == "referee"].groupby("frame_idx")["track_id"].nunique()
    out["frames_over_4_referees"] = int((r > MAX_REFEREES).sum())
    out["frames_over_4_referees_share"] = float((r > MAX_REFEREES).sum() / n_frames) if n_frames else None
    # whole-frame radar holes: no row at all for more than HOLE_FRAMES frames inside one shot
    fr = t.drop_duplicates("frame_idx").sort_values("frame_idx")
    ff, fs = fr["frame_idx"].to_numpy(), fr["shot"].to_numpy()
    hole = (np.diff(ff) - 1 > HOLE_FRAMES) & (fs[1:] == fs[:-1])
    out["radar_holes"] = int(hole.sum())
    out["radar_holes_per_min"] = float(hole.sum() / minutes) if minutes else None
    out["radar_hole_frames"] = int((np.diff(ff) - 1)[hole].sum())
    return out


def pair_duplicates(p: pd.DataFrame) -> dict:
    """Pairs of same-team players, different ids, within PAIR_M of each other for >= PAIR_FRAMES
    frames: one player the tracker has drawn twice."""
    q = p[p["team"].isin(["home", "away"])]
    if len(q) < 2:
        return {"duplicate_pairs": 0, "duplicate_pair_frames": 0}
    # a third coordinate that puts every frame 1000 m from the next, so one 3-d query finds
    # the within-frame pairs of the whole match at once
    pts = np.column_stack([q["x"].to_numpy(float), q["y"].to_numpy(float),
                           q["frame_idx"].to_numpy(float) * 1000.0])
    pairs = cKDTree(pts).query_pairs(PAIR_M, output_type="ndarray")
    if not len(pairs):
        return {"duplicate_pairs": 0, "duplicate_pair_frames": 0}
    tid, team = q["track_id"].to_numpy(), q["team"].to_numpy()
    a, b = pairs[:, 0], pairs[:, 1]
    keep = (tid[a] != tid[b]) & (team[a] == team[b])
    lo, hi = np.minimum(tid[a], tid[b])[keep], np.maximum(tid[a], tid[b])[keep]
    if not len(lo):
        return {"duplicate_pairs": 0, "duplicate_pair_frames": 0}
    n = pd.Series(1, index=pd.MultiIndex.from_arrays([lo, hi])).groupby(level=[0, 1]).sum()
    long = n[n >= PAIR_FRAMES]
    return {"duplicate_pairs": int(len(long)), "duplicate_pair_frames": int(long.sum())}


# ---------------------------------------------------------------------- entry points


def compute(tracks: pd.DataFrame, fps: float, every: int = 10) -> dict:
    """The flicker metrics of one state's observation table."""
    t = _prepare(tracks)
    out: dict = {"fps": float(fps), "every": int(every)}
    if t.empty:
        return {**out, "live_frames": 0, "minutes": 0.0}
    live_frames = np.sort(t["frame_idx"].unique())
    minutes = len(live_frames) / fps / 60
    out["live_frames"] = int(len(live_frames))
    out["minutes"] = float(minutes)
    p = t[t["role"].isin(PERSON)]
    out.update(player_motion(p, fps, every))
    out.update(still_jitter(p))
    out.update(ball_metrics(t, live_frames, fps, every))
    out.update(lifecycle(t, fps, minutes))
    return out


def evaluate(metrics: dict) -> dict:
    """Each `THRESHOLDS` entry as {value, op, limit, passed}; passed is None without a value."""
    checks = {}
    for name, (op, limit) in THRESHOLDS.items():
        v = metrics.get(name)
        checks[name] = {"value": v, "op": op, "limit": limit,
                        "passed": None if v is None else bool(_OPS[op](v, limit))}
    return checks


def estimate_fps(tracks: pd.DataFrame) -> float:
    """fps from timestamps when no meta.json is at hand: median frame step over time step."""
    t = tracks.drop_duplicates("frame_idx").sort_values("frame_idx")
    df, dt = np.diff(t["frame_idx"].to_numpy(float)), np.diff(t["timestamp"].to_numpy(float))
    ok = (df >= 1) & (dt > 0) & (df <= 5)
    if not ok.any():
        raise ValueError("cannot infer fps: pass it explicitly")
    return float(round(np.median(df[ok] / dt[ok]), 2))


def run_dir(out_dir: str | Path, every: int = 10, fps: float | None = None,
            write: bool = True) -> dict:
    """`compute` + `evaluate` on `<out_dir>/tracks.parquet`; writes `<out_dir>/flicker.json`."""
    d = Path(out_dir)
    tracks = pd.read_parquet(d / "tracks.parquet")
    if fps is None:
        meta = d / "meta.json"
        fps = json.loads(meta.read_text())["fps"] if meta.exists() else estimate_fps(tracks)
    metrics = compute(tracks, float(fps), every)
    checks = evaluate(metrics)
    scored = [c["passed"] for c in checks.values() if c["passed"] is not None]
    res = {"match": d.name, "metrics": metrics, "checks": checks,
           "passed": int(sum(scored)), "total": len(scored)}
    if write:
        (d / "flicker.json").write_text(json.dumps(res, indent=2, default=float))
    return res


def flat(metrics: dict, prefix: str = "flicker_") -> dict:
    """Numeric metrics under `prefix`, for a bench aggregate (nested values are left out)."""
    return {prefix + k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


# ---------------------------------------------------------------------- display

_FMT = {"player_step_gt12": "{:.2%}", "player_step_gt20": "{:.2%}", "ball_step_gt40": "{:.2%}",
        "common_mode_gt05": "{:.2%}", "frames_over_11_share": "{:.2%}"}


def _fmt(name: str, v) -> str:
    if v is None:
        return "-"
    if name in _FMT:
        return _FMT[name].format(v)
    return f"{v:.3g}" if isinstance(v, float) else str(v)


REPORTED = ("player_accel_p99", "common_mode_p99_kf", "common_mode_p99_between", "ball_presence",
            "ball_presence_detected", "ball_gap_p50", "ball_gap_p90", "ball_gap_p99",
            "blinks_per_player_min", "mass_birth_frames", "duplicate_pairs",
            "frames_over_4_referees", "radar_holes_per_min", "duplicate_frame_share",
            "ball_beyond_gate_legacy")


def render(results: list[dict]) -> str:
    """A fixed-width table: one column per match, the pass/fail checks first, then the rest."""
    names = [r["match"] for r in results]
    rows = [["metric", "limit"] + names]
    for name, (op, limit) in THRESHOLDS.items():
        cells = []
        for r in results:
            c = r["checks"][name]
            mark = {True: "PASS", False: "FAIL", None: ""}[c["passed"]]
            cells.append(f"{_fmt(name, c['value'])} {mark}".strip())
        rows.append([name, f"{op} {_fmt(name, limit)}"] + cells)
    rows.append(["passed"] + [""] + [f"{r['passed']}/{r['total']}" for r in results])
    rows.append(["reported (no limit)"] + [""] * (1 + len(results)))
    for name in REPORTED:
        rows.append([name, ""] + [_fmt(name, r["metrics"].get(name)) for r in results])
    w = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(row)).rstrip() for row in rows)
