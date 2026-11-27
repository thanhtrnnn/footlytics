"""How steady is the player state? Measurements, and an offline replay of the stabilisation.

The flicker of a Match State has three separable sources, each with its own measurement:

* a whole-frame jump (the camera): `common_mode`, the median step all players share in a frame,
  split by whether the frame is a calibration keyframe (`frame_idx % every == 0`);
* per-player jitter at rest: `still_sigma`, detrended rms of a near-stationary player;
* implausible motion: `step_stats`, the share of consecutive-frame steps implying > 12 / > 20 m/s,
  and the acceleration tail.

`replay` re-applies the camera smoothing to a *saved* state without the video: the image -> pitch
homography of every frame is recoverable exactly from the rows (pitch position = H(box foot), and a
frame carries a dozen of them), so it is re-estimated, smoothed (`geometry.camera_smooth`), and the
feet are projected through the smoothed cameras again. `retrack` runs the tracker over such rows
to count how many ids it needs.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import pandas as pd

from ..geometry.camera_smooth import smooth_homographies

PEOPLE = ("player", "goalkeeper", "referee")
PLAYERS = ("player", "goalkeeper")


def _prep(t: pd.DataFrame, roles) -> pd.DataFrame:
    return t[t["role"].astype(str).isin(roles) & (t["track_id"] >= 0)].sort_values(
        ["track_id", "frame_idx"], kind="stable")


def step_stats(t: pd.DataFrame, fps: float, roles=PLAYERS) -> dict:
    """Consecutive-frame steps of one id: share above 12 / 20 m/s, acceleration p99."""
    p = _prep(t, roles)
    g = p.groupby("track_id", sort=False)
    dx, dy, df = g["x"].diff(), g["y"].diff(), g["frame_idx"].diff()
    ok = (df == 1).to_numpy()
    v = np.hypot(dx, dy).to_numpy()[ok] * fps
    vx, vy = (dx * fps).to_numpy(), (dy * fps).to_numpy()
    same = (g["track_id"].shift(1) == p["track_id"]).to_numpy()
    # acceleration needs three consecutive frames of one id
    ax = np.r_[np.nan, np.diff(vx)] * fps
    ay = np.r_[np.nan, np.diff(vy)] * fps
    cons2 = ok & np.r_[False, ok[:-1]] & same & np.r_[False, same[:-1]]
    a = np.hypot(ax, ay)[cons2]
    a = a[np.isfinite(a)]
    return {"steps": int(len(v)), "gt12": float((v > 12).mean()), "gt20": float((v > 20).mean()),
            "step_p99_m": float(np.percentile(v, 99) / fps) if len(v) else float("nan"),
            "accel_p99": float(np.percentile(a, 99)) if len(a) else float("nan")}


def common_mode(t: pd.DataFrame, every: int = 10, roles=PEOPLE, min_tracks: int = 5) -> dict:
    """Per frame, the median step shared by all matched people; share > 0.5 m on keyframes vs not."""
    p = _prep(t, roles)
    g = p.groupby("track_id", sort=False)
    q = p.assign(dx=g["x"].diff(), dy=g["y"].diff(), df=g["frame_idx"].diff())
    q = q[q["df"] == 1]
    cm = q.groupby("frame_idx").agg(n=("dx", "size"), cx=("dx", "median"), cy=("dy", "median"))
    cm = cm[cm["n"] >= min_tracks]
    mag = np.hypot(cm["cx"], cm["cy"])
    key = (cm.index.to_numpy() % every) == 0
    return {"frames": int(len(cm)),
            "key_gt05": float((mag[key] > 0.5).mean()) if key.any() else float("nan"),
            "other_gt05": float((mag[~key] > 0.5).mean()) if (~key).any() else float("nan"),
            "all_gt05": float((mag > 0.5).mean()),
            "key_median_m": float(mag[key].median()), "other_median_m": float(mag[~key].median())}


def still_sigma(t: pd.DataFrame, W: int = 30, roles=PLAYERS, max_drift: float = 0.5) -> dict:
    """Jitter of near-stationary players: detrended rms radial residual over 30-frame windows."""
    p = _prep(t, roles)
    sig = []
    tt = np.arange(W)
    for _, g in p.groupby("track_id", sort=False):
        if len(g) < W:
            continue
        f = g["frame_idx"].to_numpy()
        x, y = g["x"].to_numpy(float), g["y"].to_numpy(float)
        brk = np.flatnonzero(np.diff(f) != 1) + 1
        for a, b in zip(np.r_[0, brk], np.r_[brk, len(f)]):
            for s in range(a, b - W + 1, W):
                xx, yy = x[s:s + W], y[s:s + W]
                drift = np.hypot(np.median(xx[-5:]) - np.median(xx[:5]),
                                 np.median(yy[-5:]) - np.median(yy[:5]))
                if drift < max_drift:
                    rx = xx - np.polyval(np.polyfit(tt, xx, 1), tt)
                    ry = yy - np.polyval(np.polyfit(tt, yy, 1), tt)
                    sig.append(np.sqrt((rx ** 2).mean() + (ry ** 2).mean()))
    sig = np.array(sig)
    return {"windows": int(len(sig)),
            "median_m": float(np.median(sig)) if len(sig) else float("nan"),
            "p90_m": float(np.percentile(sig, 90)) if len(sig) else float("nan")}


def id_stats(t: pd.DataFrame, fps: float, roles=PEOPLE, observed_only: bool = True) -> dict:
    """Ids born per tracked person-minute, and the median life of an id in seconds."""
    p = t[t["role"].astype(str).isin(roles) & (t["track_id"] >= 0)]
    if observed_only and "coasted" in p.columns:
        p = p[~p["coasted"].astype(bool)]
    n = p.groupby("track_id").size()
    return {"ids": int(len(n)), "person_minutes": float(len(p) / fps / 60.0),
            "births_per_person_min": float(len(n) / max(len(p) / fps / 60.0, 1e-9)),
            "median_life_s": float(np.median(n) / fps)}


def measure(t: pd.DataFrame, fps: float, every: int = 10) -> dict:
    return {**{f"step_{k}": v for k, v in step_stats(t, fps).items()},
            **{f"cm_{k}": v for k, v in common_mode(t, every).items()},
            **{f"still_{k}": v for k, v in still_sigma(t).items()},
            **{f"id_{k}": v for k, v in id_stats(t, fps).items()}}


# ------------------------------------------------------------------------------ replay of A

def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def recover_homographies(t: pd.DataFrame, min_n: int = 8, min_spread_px: float = 40.0
                         ) -> dict[int, np.ndarray]:
    """Per-frame image -> pitch homography from the saved rows: x, y = H(box foot)."""
    p = t[t["role"].astype(str).isin(PEOPLE + ("other", "ball")) & ~t["interpolated"].astype(bool)
          & t["bbox_w"].notna()]
    out: dict[int, np.ndarray] = {}
    fx = (p["bbox_x"] + p["bbox_w"] / 2).to_numpy(np.float64)
    fy = (p["bbox_y"] + p["bbox_h"]).to_numpy(np.float64)
    px, py = p["x"].to_numpy(np.float64), p["y"].to_numpy(np.float64)
    frames = p["frame_idx"].to_numpy()
    order = np.argsort(frames, kind="stable")
    frames = frames[order]
    cuts = np.flatnonzero(np.diff(frames)) + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(frames)]):
        if b - a < min_n:
            continue
        ii = order[a:b]
        src = np.c_[fx[ii], fy[ii]]
        if src.std(axis=0).min() < min_spread_px:
            continue
        H, _ = cv2.findHomography(src, np.c_[px[ii], py[ii]], 0)
        if H is not None and np.isfinite(H).all():
            out[int(frames[a])] = H
    return out


def replay_camera(t: pd.DataFrame, wh: tuple[int, int] = (1920, 1080), every: int = 10,
                  max_gap: int = 3, **kw) -> tuple[pd.DataFrame, dict]:
    """Re-project every row's box foot through the drift-corrected camera of its frame.

    `every`: the clip's keyframe cadence (the saved state does not record which keyframes were
    accepted; a step corrected where none was taken is ~0). Frames whose camera cannot be recovered (fewer than 8 rows) keep their rows; a camera is
    filled across at most `max_gap` such frames so that smoothing sees one segment.
    """
    Hs = recover_homographies(t)
    if not Hs:
        return t, {"frames_recovered": 0}
    n = int(t["frame_idx"].max()) + 1
    series: list[Optional[np.ndarray]] = [Hs.get(i) for i in range(n)]
    have = np.array([h is not None for h in series])
    # short holes: copy the nearer neighbour so the segment is not cut for want of a camera
    idx = np.flatnonzero(have)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            for k in range(a + 1, b):
                series[k] = series[a] if k - a <= b - k else series[b]
    fr_shot = t.drop_duplicates("frame_idx").set_index("frame_idx")["shot"].reindex(range(n))
    shot = fr_shot.ffill().bfill().to_numpy()
    breaks = np.r_[False, np.diff(shot) != 0]
    key = (np.arange(n) % every == 0) & have
    sm, info = smooth_homographies(series, wh[0], wh[1], key=key, breaks=breaks, **kw)
    out = t.copy()
    real = ~out["interpolated"].astype(bool).to_numpy() & out["bbox_w"].notna().to_numpy()
    f = out["frame_idx"].to_numpy()
    ok = real & np.array([sm[i] is not None for i in f])
    Hstack = np.zeros((n, 3, 3))
    for i, H in enumerate(sm):
        if H is not None:
            Hstack[i] = H
    feet = np.c_[(out["bbox_x"] + out["bbox_w"] / 2).to_numpy(np.float64),
                 (out["bbox_y"] + out["bbox_h"]).to_numpy(np.float64), np.ones(len(out))]
    ii = np.flatnonzero(ok)
    p = np.einsum("nij,nj->ni", Hstack[f[ii]], feet[ii])
    xy = p[:, :2] / p[:, 2:3]
    good = np.isfinite(xy).all(axis=1)
    x = out["x"].to_numpy(np.float64).copy()
    y = out["y"].to_numpy(np.float64).copy()
    x[ii[good]], y[ii[good]] = xy[good, 0], xy[good, 1]
    out["x"], out["y"] = x.astype(np.float32), y.astype(np.float32)
    return out, {"frames_recovered": len(Hs), "frames_smoothed": int(sum(
        1 for a, b in zip(series, sm) if a is not None and a is not b)),
        "segments": info.segments, "events": info.events,
        "breaks_jump": info.breaks_jump, "breaks_cap": info.breaks_cap,
        "max_step_m": info.max_step_m, "max_move_m": info.max_move_m,
        "rows_reprojected": int(good.sum())}


# ---------------------------------------------------------------------- association replay

def retrack(t: pd.DataFrame, fps: float, min_frames: int = 15, max_age: Optional[int] = None):
    """Run the tracker over the saved detections (no appearance: only geometry and boxes).

    The saved rows are the detections that survived the clip's own filters, so this is not
    the production id count; it is the same procedure on two sets of positions, which is what
    shows the effect of the camera. Returns rows with new ids, ids shorter than `min_frames` removed.
    """
    from ..perception.track import PitchTracker, TrackerConfig

    cfg = TrackerConfig(fps=fps)
    if max_age is not None:
        cfg.max_age = max_age
    tr = PitchTracker(cfg)
    p = t[t["role"].astype(str).isin(PEOPLE) & (t["track_id"] >= 0)].sort_values(
        ["frame_idx", "track_id"], kind="stable")
    rows = []
    fidx = p["frame_idx"].to_numpy()
    cuts = np.flatnonzero(np.diff(fidx)) + 1
    xy_all = p[["x", "y"]].to_numpy(float)
    box_all = p[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].to_numpy(float)
    role_all = p["role"].astype(str).to_numpy()
    conf_all = p["det_conf"].to_numpy(float)
    last = None
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(p)]):
        f = int(fidx[a])
        if last is not None and f - last > 1:
            for _ in range(min(f - last - 1, cfg.max_age + 1)):     # frames with no detections
                tr.update(np.empty((0, 2)), [])
        matches = tr.update(xy_all[a:b], list(role_all[a:b]), bboxes=box_all[a:b],
                            confs=conf_all[a:b])
        for di, tid in matches:
            rows.append((f, tid, a + di))
        last = f
    r = pd.DataFrame(rows, columns=["frame_idx", "new_id", "src"])
    n = r.groupby("new_id")["src"].transform("size")
    r = r[n >= min_frames]
    out = p.iloc[r["src"].to_numpy()].copy()
    out["track_id"] = r["new_id"].to_numpy()
    return out.reset_index(drop=True)


# ------------------------------------------------------------------- over-smoothing check

def _smooth_one(xy: np.ndarray, foot_row: float, fps: float, cfg) -> np.ndarray:
    """Run the player-state smoother over one synthetic track."""
    from ..perception.state_smooth import smooth_player_states

    n = len(xy)
    df = pd.DataFrame({
        "frame_idx": np.arange(n), "period": 1, "timestamp": np.arange(n) / fps, "track_id": 1,
        "role": "player", "team": "home", "jersey": np.nan,
        "bbox_x": 900.0 + 0.01 * np.arange(n),     # boxes that never repeat exactly
        "bbox_y": foot_row - 150.0, "bbox_w": 60.0, "bbox_h": 150.0,
        "det_conf": 0.9, "x": xy[:, 0], "y": xy[:, 1], "z": np.nan})
    out, _ = smooth_player_states(df, fps, cfg)
    return out[["x", "y"]].to_numpy(float)


def synthetic_response(fps: float = 30.0, foot_row_frac: float = 0.6, seed: int = 0,
                       n_noise: int = 40, cfg=None, frame_h: float = 1080.0) -> dict:
    """What the smoother does to a real manoeuvre: lag and attenuation, noise-free and noisy.

    Two truths: a sprint start (standing for 1 s, then 7 m/s^2 up to 8 m/s, then steady) and a sharp cut
    (6 m/s along x, then 6 m/s along y with the velocity change in one frame: a 90 degree turn).
    Noise is Gaussian at the measurement sigma the smoother itself assumes for that image row, so
    the noisy run is the case it is designed for. Reported per manoeuvre: the peak distance of the
    smoothed position from the truth (m) and the same as a lag in seconds at the manoeuvre's peak
    speed, the peak speed kept (share of the true peak, speed from one-frame differences of the
    smoothed positions), and, under noise, the rms error of the smoothed positions against that
    of the raw measurements.
    """
    from ..perception.state_smooth import StateSmoothConfig, measurement_sigma

    cfg = cfg or StateSmoothConfig()
    dt = 1.0 / fps
    t = np.arange(int(4.0 * fps)) * dt
    v = np.clip(7.0 * (t - 1.0), 0.0, 8.0)
    sprint = np.c_[np.cumsum(v) * dt, np.zeros_like(t)]
    k = len(t) // 2
    vel = np.zeros((len(t), 2))
    vel[:k, 0], vel[k:, 1] = 6.0, 6.0
    cut = np.cumsum(vel, axis=0) * dt
    row = foot_row_frac * frame_h
    sig = float(measurement_sigma(np.array([row]), frame_h, cfg)[0])
    rng = np.random.default_rng(seed)
    out: dict = {"meas_sigma_m": sig, "fps": fps}
    for name, truth in (("sprint", sprint), ("cut", cut)):
        clean = _smooth_one(truth, row, fps, cfg)
        err = np.linalg.norm(clean - truth, axis=1)
        sp = np.linalg.norm(np.diff(clean, axis=0), axis=1) * fps
        tr = np.linalg.norm(np.diff(truth, axis=0), axis=1) * fps
        rms_s, rms_r, peaks = [], [], []
        for _ in range(n_noise):
            z = truth + rng.normal(0.0, sig, truth.shape)
            s = _smooth_one(z, row, fps, cfg)
            rms_s.append(np.sqrt(np.mean(np.sum((s - truth) ** 2, axis=1))))
            rms_r.append(np.sqrt(np.mean(np.sum((z - truth) ** 2, axis=1))))
            peaks.append(np.linalg.norm(s - truth, axis=1).max())
        out[name] = {
            "noise_free_peak_err_m": float(err.max()),
            "noise_free_lag_s": float(err.max() / max(tr.max(), 1e-9)),
            "noise_free_peak_speed_kept": float(sp.max() / tr.max()),
            "noisy_rms_err_m": float(np.mean(rms_s)), "raw_rms_err_m": float(np.mean(rms_r)),
            "noisy_peak_err_m": float(np.mean(peaks)),
        }
    return out
