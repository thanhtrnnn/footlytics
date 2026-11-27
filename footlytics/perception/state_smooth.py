"""A smoothed pitch-space state for every tracked person, written into the Match State itself.

The tracker's Kalman state is used for association and then thrown away: the emitted `x, y` was
the raw measurement (bbox bottom centre through the homography), so every consumer (the web radar,
packs, analytics) got jitter and drew it. Smoothing in the renderer fixes one consumer; this fixes
the state.

Per person track, offline, a forward-backward (Rauch-Tung-Striebel) smoother with a constant-
velocity model replaces `x, y`; the measurement stays in `x_raw, y_raw` for audit.

* Measurement noise grows with distance from the camera. A foot-point error of a pixel is 0.23 m
  at image row 0.30H and 0.03 m at 0.9H (`depth_sensitivity`), so a far player is trusted less
  and a near one more. The two ends of the pitch need different smoothers; a single constant
  does one of them wrong.
* Robust. An innovation beyond `huber` standard deviations gets a proportionally larger variance
  (its pull falls as 1/d); beyond `gate` it is dropped. `reinit_after` dropped measurements in a
  row mean the track really moved (an identity swap, an uncorrected camera step): the filter
  restarts at the new position and the smoother does not average across the join.
* Segments break at shot changes and at gaps longer than `coast_s`. Inside a gap of at most
  `coast_s` the smoother interpolates and the frames are emitted with `coasted=True`: the player
  was occluded, not gone, so the radar should not blink him out. Past the last or before the first
  observation nothing is invented (offline, a track that ends has ended).

The filter runs for all segments at once, one numpy step per time step, so a 90-minute match
smooths in seconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

_PERSON_EXCLUDE = "ball"


@dataclass
class StateSmoothConfig:
    #: Sigma of the acceleration driving the constant-velocity model, m/s^2. A sprint start is
    #: ~7 m/s^2 and a cut ~9; 3 is the width of the typical, not the extreme.
    accel_noise: float = 3.0
    #: Floor of a measurement's sigma, metres (a sharp near-camera box).
    meas_floor: float = 0.10
    #: Foot-point noise of the detector box in pixels (at 1080 rows), turned into metres by
    #: `depth_sensitivity`.
    foot_px: float = 2.5
    #: Innovations beyond this many sigmas are down-weighted ~1/d; beyond `gate` dropped.
    huber: float = 2.5
    gate: float = 6.0
    #: This many dropped measurements in a row restart the filter at the new position.
    reinit_after: int = 5
    #: Manoeuvre adaptation (see `rts_adaptive`): a systematic residual of `adapt_z` (in sigmas,
    #: of the mean over `2*adapt_window+1` frames) multiplies the process variance by `adapt_boost`
    #: there; `adapt_passes` repeats at most.
    adapt_z: float = 3.0
    adapt_window: int = 2
    adapt_boost: float = 100.0
    adapt_passes: int = 2
    #: Camera noise the homography still carries is the same for everybody in the frame: the median
    #: step of the people in a frame (>= `cm_min_tracks` of them, steps under `cm_event_m`), summed
    #: over time, minus its own slow trend (Gaussian of `cm_sigma_s`), is that noise, because a
    #: team does not accelerate in a tenth of a second. It is taken off every measurement first.
    #: Off at 0 tracks. Steps beyond `cm_event_m` are a cut or a camera jump, not noise: left alone.
    common_mode: bool = True
    cm_min_tracks: int = 6
    cm_event_m: float = 3.0
    cm_sigma_s: float = 0.5
    #: Frames of a gap that are bridged (emitted, `coasted=True`), seconds. 0 turns coasting off.
    coast_s: float = 0.5
    #: Image height in pixels the boxes are in; None infers it from the boxes.
    frame_h: Optional[float] = None


def depth_sensitivity(v_frac: np.ndarray) -> np.ndarray:
    """Pitch metres of depth error per pixel of foot-row error, at a 1080-row image.

    A ground plane seen from a camera above it: depth ~ 1 / (v - v_horizon), so the error per
    pixel falls with the inverse square of the distance below the horizon. Fitted to the two
    measured points, 0.23 m/px at row 0.30H and 0.03 m/px at 0.9H (horizon just above the frame).
    """
    v = np.asarray(v_frac, float)
    return np.clip(0.0264 / (v + 0.039) ** 2, 0.02, 0.40)


def infer_frame_h(boxes_bottom: np.ndarray) -> float:
    """The standard frame height a set of foot rows fits in."""
    top = float(np.nanpercentile(boxes_bottom, 99.9)) if len(boxes_bottom) else 1080.0
    for h in (360, 480, 540, 576, 720, 1080, 1440, 2160):
        if top <= h * 1.01:
            return float(h)
    return top


def measurement_sigma(foot_row: np.ndarray, frame_h: float, cfg: StateSmoothConfig) -> np.ndarray:
    """Per-measurement sigma in metres from where the foot sits in the image."""
    v = np.clip(np.asarray(foot_row, float) / frame_h, 0.05, 1.2)
    sens = depth_sensitivity(v) * (1080.0 / frame_h)
    return np.sqrt(cfg.meas_floor ** 2 + (sens * cfg.foot_px) ** 2)


def rts_batch(z: np.ndarray, r: np.ndarray, obs: np.ndarray, seg_off: np.ndarray,
              seg_len: np.ndarray, dt: float, cfg: StateSmoothConfig,
              qscale: Optional[np.ndarray] = None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth many independent segments at once.

    `z` (N, 2) measurements and `r` (N,) sigmas for all segments laid end to end; `obs` (N,)
    False on frames without a measurement (a gap: predicted); each segment starts on an observed
    frame. `qscale` (N,) multiplies the process variance entering each frame (manoeuvres).
    Returns (xy_smoothed (N, 2), n_dropped (N,) bool, restart (N,) bool).
    """
    N = len(z)
    S = len(seg_off)
    xs = np.zeros((N, 2))
    dropped = np.zeros(N, bool)
    restart = np.zeros(N, bool)
    if N == 0:
        return xs, dropped, restart
    order = np.argsort(-seg_len, kind="stable")
    off, ln = seg_off[order].astype(np.int64), seg_len[order].astype(np.int64)
    maxlen = int(ln[0])
    n_active = np.searchsorted(-ln, -np.arange(maxlen + 1), side="left")   # segments longer than k
    q2 = cfg.accel_noise ** 2 * (np.ones(N) if qscale is None else np.asarray(qscale, float))
    dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4

    X = np.zeros((N, 4))          # filtered [px, py, vx, vy]
    P = np.zeros((N, 3))          # filtered [P00, P01, P11] (shared by both axes)
    Xp = np.zeros((N, 4))
    Pp = np.zeros((N, 3))
    px = np.zeros(S); py = np.zeros(S); vx = np.zeros(S); vy = np.zeros(S)
    p00 = np.zeros(S); p01 = np.zeros(S); p11 = np.zeros(S)
    rej = np.zeros(S, np.int64)
    P11_0 = 25.0
    # sorted-segment views of the inputs are gathered per step via flat indices
    for k in range(maxlen):
        m = int(n_active[k])
        i = off[:m] + k
        zk, rk, ok = z[i], r[i], obs[i]
        if k == 0:
            px[:m], py[:m] = zk[:, 0], zk[:, 1]
            vx[:m] = vy[:m] = 0.0
            p00[:m], p01[:m], p11[:m] = rk ** 2, 0.0, P11_0
            Xp[i] = np.c_[px[:m], py[:m], vx[:m], vy[:m]]
            Pp[i] = np.c_[p00[:m], p01[:m], p11[:m]]
            X[i], P[i] = Xp[i], Pp[i]
            continue
        a, b, c = p00[:m], p01[:m], p11[:m]
        px[:m] += vx[:m] * dt
        py[:m] += vy[:m] * dt
        qk = q2[i]
        n00 = a + 2 * dt * b + dt2 * c + qk * dt4 / 4
        n01 = b + dt * c + qk * dt3 / 2
        n11 = c + qk * dt2
        p00[:m], p01[:m], p11[:m] = n00, n01, n11
        Xp[i] = np.c_[px[:m], py[:m], vx[:m], vy[:m]]
        Pp[i] = np.c_[n00, n01, n11]
        # innovation of observed segments
        ix, iy = zk[:, 0] - px[:m], zk[:, 1] - py[:m]
        Rm = rk ** 2
        d = np.hypot(ix, iy) / np.sqrt(n00 + Rm)
        # Huber: variance grows with the excess so the pull falls ~ 1/d
        Reff = np.where(d > cfg.huber, Rm * (d / cfg.huber), Rm)
        use = ok & (d <= cfg.gate)
        drop = ok & ~use
        rej[:m] = np.where(drop, rej[:m] + 1, np.where(ok, 0, rej[:m]))
        dropped[i] = drop
        # restart: the track genuinely moved
        rs = drop & (rej[:m] >= cfg.reinit_after)
        Sv = n00 + Reff
        K0, K1 = n00 / Sv, n01 / Sv
        u = use
        px[:m] = np.where(u, px[:m] + K0 * ix, px[:m])
        py[:m] = np.where(u, py[:m] + K0 * iy, py[:m])
        vx[:m] = np.where(u, vx[:m] + K1 * ix, vx[:m])
        vy[:m] = np.where(u, vy[:m] + K1 * iy, vy[:m])
        p00[:m] = np.where(u, (1 - K0) * n00, n00)
        p01[:m] = np.where(u, (1 - K0) * n01, n01)
        p11[:m] = np.where(u, n11 - K1 * n01, n11)
        if rs.any():
            j = np.flatnonzero(rs)
            px[j], py[j] = zk[j, 0], zk[j, 1]
            vx[j] = vy[j] = 0.0
            p00[j], p01[j], p11[j] = rk[j] ** 2, 0.0, P11_0
            rej[j] = 0
            restart[i[j]] = True
        X[i] = np.c_[px[:m], py[:m], vx[:m], vy[:m]]
        P[i] = np.c_[p00[:m], p01[:m], p11[:m]]

    # backward pass
    Xs = X.copy()
    for k in range(maxlen - 2, -1, -1):
        m = int(n_active[k + 1])                 # segments that have a step k + 1
        i = off[:m] + k
        j = i + 1
        P00, P01, P11 = P[i, 0], P[i, 1], P[i, 2]
        q00, q01, q11 = Pp[j, 0], Pp[j, 1], Pp[j, 2]
        det = q00 * q11 - q01 * q01
        det = np.where(np.abs(det) < 1e-18, 1e-18, det)
        # C = (P F^T) Pp^-1, F = [[1, dt], [0, 1]]
        m00, m01 = P00 + dt * P01, P01
        m10, m11 = P01 + dt * P11, P11
        i00, i01, i11 = q11 / det, -q01 / det, q00 / det
        c00, c01 = m00 * i00 + m01 * i01, m00 * i01 + m01 * i11
        c10, c11 = m10 * i00 + m11 * i01, m10 * i01 + m11 * i11
        use = ~restart[j]                        # never smooth across a restart
        for ax in (0, 1):
            dp = Xs[j, ax] - Xp[j, ax]
            dv = Xs[j, 2 + ax] - Xp[j, 2 + ax]
            Xs[i, ax] = np.where(use, X[i, ax] + c00 * dp + c01 * dv, X[i, ax])
            Xs[i, 2 + ax] = np.where(use, X[i, 2 + ax] + c10 * dp + c11 * dv, X[i, 2 + ax])
    xs[:] = Xs[:, :2]
    return xs, dropped, restart


def rts_adaptive(z, r, obs, seg_off, seg_len, dt, cfg):
    """`rts_batch`, then again with a larger process noise wherever the first pass lagged.

    A constant-velocity model with a small acceleration noise is the right smoother for steady
    running and the wrong one for a sprint start or a turn: it rounds the corner, and the
    measurements keep pulling the same way until the gate calls them outliers. The smoothed
    residual shows it. Where its mean over a few frames is a systematic offset (not the noise,
    which averages out) the acceleration variance is raised by `boost` for those frames and the
    pass is repeated: the smoother is allowed to turn exactly where the data say it turned.
    """
    N = len(z)
    qs = np.ones(N)
    out = rts_batch(z, r, obs, seg_off, seg_len, dt, cfg)
    w = cfg.adapt_window
    ker = np.ones(2 * w + 1)
    for _ in range(cfg.adapt_passes):
        xs = out[0]
        res = np.where(obs[:, None], (z - xs) / r[:, None], 0.0)
        cnt = np.convolve(obs.astype(float), ker, mode="same")
        m = np.stack([np.convolve(res[:, a], ker, mode="same") for a in (0, 1)], axis=1)
        stat = np.hypot(m[:, 0], m[:, 1]) / np.sqrt(np.maximum(cnt, 1.0))
        flag = (stat > cfg.adapt_z) & (cnt >= 3)
        grown = np.convolve(flag.astype(float), ker, mode="same") > 0
        new = np.where(grown, cfg.adapt_boost, 1.0)
        if not (new > qs).any():
            break
        qs = np.maximum(qs, new)
        out = rts_batch(z, r, obs, seg_off, seg_len, dt, cfg, qs)
    return out


def _person_mask(df: pd.DataFrame) -> np.ndarray:
    return ((df["role"].astype(str) != _PERSON_EXCLUDE) & (df["track_id"] >= 0)).to_numpy()


def _frame_lookup(df: pd.DataFrame, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(shot id or -1, live) per frame index 0..n-1 from whatever rows that frame has."""
    shot = np.full(n, -1, np.int64)
    live = np.ones(n, bool)
    f = df.drop_duplicates("frame_idx")
    idx = f["frame_idx"].to_numpy().astype(np.int64)
    if "shot" in f.columns:
        shot[idx] = f["shot"].to_numpy().astype(np.int64)
    if "live_wide" in f.columns:
        live[idx] = f["live_wide"].to_numpy().astype(bool)
    return shot, live


def common_mode_offsets(P: pd.DataFrame, fps: float, n_frames: int, f_shot: np.ndarray,
                        cfg: StateSmoothConfig) -> np.ndarray:
    """(n_frames, 2) camera noise common to everyone in the frame, metres. See `common_mode`.

    `P` is sorted by (track_id, frame_idx). The slow trend of the summed median step is taken per
    shot, so a cut never leaks into its neighbours.
    """
    from scipy.ndimage import gaussian_filter1d

    tid = P["track_id"].to_numpy()
    fr = P["frame_idx"].to_numpy().astype(np.int64)
    xy = P[["x", "y"]].to_numpy(float)
    same = (tid[1:] == tid[:-1]) & (fr[1:] - fr[:-1] == 1)
    d = np.full((len(P), 2), np.nan)
    d[1:][same] = xy[1:][same] - xy[:-1][same]
    # a repeated video frame has a zero step that says nothing about the camera
    box = P[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].to_numpy(float)
    rep = np.zeros(len(P), bool)
    rep[1:] = (box[1:] == box[:-1]).all(axis=1) & same
    d[rep] = np.nan
    df = pd.DataFrame({"f": fr, "dx": d[:, 0], "dy": d[:, 1]}).dropna()
    g = df.groupby("f")
    cnt = g["dx"].size()
    med = g[["dx", "dy"]].median()
    med = med[cnt >= cfg.cm_min_tracks]
    m = np.zeros((n_frames, 2))
    mm = med.to_numpy().copy()
    mm[np.hypot(mm[:, 0], mm[:, 1]) > cfg.cm_event_m] = 0.0
    m[med.index.to_numpy()] = mm
    out = np.zeros((n_frames, 2))
    sig = max(cfg.cm_sigma_s * fps, 1.0)
    # one run per shot: contiguous frames, ignoring frames without rows (-1)
    known = np.flatnonzero(f_shot >= 0)
    if not len(known):
        return out
    sh = f_shot[known]
    cuts = np.flatnonzero(np.diff(sh) != 0) + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(known)]):
        lo, hi = int(known[a]), int(known[b - 1]) + 1
        M = np.cumsum(m[lo:hi], axis=0)
        out[lo:hi] = M - gaussian_filter1d(M, sig, axis=0, mode="nearest")
    return out


def smooth_player_states(tracks: pd.DataFrame, fps: float,
                         cfg: Optional[StateSmoothConfig] = None) -> tuple[pd.DataFrame, dict]:
    """Return `tracks` with people's `x, y` smoothed, `x_raw, y_raw` kept, coasted rows added.

    Rows of the ball and of untracked detections (track_id < 0) pass through unchanged
    (`x_raw = x`). Columns other than the pitch position are carried over; a coasted row holds its
    predecessor's box with `det_conf = 0`. Returns (table, report).
    """
    cfg = cfg or StateSmoothConfig()
    df = tracks.reset_index(drop=True)
    if "x_raw" not in df.columns:
        df["x_raw"] = df["x"]
        df["y_raw"] = df["y"]
    if "coasted" not in df.columns:
        df["coasted"] = False
    rep: dict = {"segments": 0, "rows": 0, "coasted_rows": 0, "dropped": 0, "restarts": 0,
           "split_at_jumps": 0, "repeated_frames": 0}
    person = _person_mask(df)
    if not person.any():
        return df, rep
    P = df[person].sort_values(["track_id", "frame_idx"], kind="stable").drop_duplicates(
        ["track_id", "frame_idx"], keep="first").reset_index(drop=True)
    other = df[~person]
    fh = cfg.frame_h or infer_frame_h((P["bbox_y"] + P["bbox_h"]).to_numpy(float))
    coast_n = int(round(cfg.coast_s * fps))

    tid = P["track_id"].to_numpy()
    fr = P["frame_idx"].to_numpy().astype(np.int64)
    n_frames = int(df["frame_idx"].max()) + coast_n + 2
    f_shot, f_live = _frame_lookup(df, n_frames)
    # A frame with no row at all (uncalibrated, nothing detected) is the camera's gap, not one
    # person's occlusion: nobody is carried through it.
    f_rows = np.zeros(n_frames, bool)
    f_rows[df["frame_idx"].to_numpy().astype(np.int64)] = True
    shot = f_shot[fr]
    # break where the id changes, the gap is too long to be an occlusion, the shot changes, or
    # a frame inside the gap is a different shot or a replay (no coasting through those)
    new = np.ones(len(P), bool)
    gapn = fr[1:] - fr[:-1] - 1
    brk = (tid[1:] != tid[:-1]) | (gapn > coast_n) | (shot[1:] != shot[:-1])
    inner = np.flatnonzero(~brk & (gapn > 0))
    if len(inner) and coast_n > 0:
        cols = np.arange(1, coast_n + 1)[None, :]
        fidx = fr[inner][:, None] + cols                       # candidate gap frames
        inside = cols <= gapn[inner][:, None]
        fidx = np.minimum(fidx, n_frames - 1)
        bad = inside & ((f_shot[fidx] != shot[inner][:, None]) & (f_shot[fidx] >= 0)
                        | ~f_live[fidx])
        brk[inner[bad.any(axis=1)]] = True
    new[1:] = brk

    z_all = P[["x", "y"]].to_numpy(float)
    z_raw = z_all
    foot = (P["bbox_y"] + P["bbox_h"]).to_numpy(float)
    sig = measurement_sigma(foot, fh, cfg)
    # A repeated video frame repeats every box exactly; it is the same observation, not a second
    # one, so it is not fed to the filter again (it still gets a smoothed row).
    box = P[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].to_numpy(float)
    dup = np.zeros(len(P), bool)
    dup[1:] = ((tid[1:] == tid[:-1]) & (fr[1:] - fr[:-1] == 1)
               & (box[1:] == box[:-1]).all(axis=1) & (z_raw[1:] == z_raw[:-1]).all(axis=1))
    if cfg.common_mode:
        off = common_mode_offsets(P, fps, n_frames, f_shot, cfg)
        z_all = z_raw - off[fr]
        rep["common_mode_rms_m"] = float(np.sqrt(np.mean(np.sum(off[np.unique(fr)] ** 2, axis=1))))
    finite = np.isfinite(z_all).all(axis=1) & ~dup
    rep_dups = int(dup.sum())

    splits = 0
    for attempt in range(3):
        starts = np.flatnonzero(new)
        seg_of = np.cumsum(new) - 1
        first_fr, last_fr = fr[starts], fr[np.r_[starts[1:] - 1, len(P) - 1]]
        seg_len = (last_fr - first_fr + 1).astype(np.int64)
        seg_off = np.r_[0, np.cumsum(seg_len)[:-1]]
        N = int(seg_len.sum())
        pos = seg_off[seg_of] + (fr - first_fr[seg_of])        # flat slot of each observed row
        z = np.zeros((N, 2))
        r = np.full(N, 1.0)
        obs = np.zeros(N, bool)
        z[pos] = np.nan_to_num(z_all)
        r[pos] = sig
        obs[pos] = finite
        xs, dropped, restart = rts_adaptive(z, r, obs, seg_off, seg_len, 1.0 / fps, cfg)
        if attempt == 2 or not restart.any():
            break
        # The track moved for real (a swap, an uncorrected camera step). The filter noticed only
        # after `reinit_after` dropped measurements, which by then belong to the new place: cut
        # the segment at the first of them and run again so they are smoothed with it.
        slot_row = np.full(N, -1, np.int64)
        slot_row[pos] = np.arange(len(P))
        rrows = slot_row[np.flatnonzero(restart)]
        cut = np.maximum(rrows - (cfg.reinit_after - 1), 0)
        cut = cut[(cut > 0) & (tid[cut] == tid[rrows]) & ~new[cut]]
        if not len(cut):
            break
        new[cut] = True
        splits += len(cut)

    Pm = P.copy()
    Pm["x_raw"] = P["x"].to_numpy()
    Pm["y_raw"] = P["y"].to_numpy()
    sm = xs[pos]
    # a segment shorter than 3 frames is not smoothed: nothing to average
    keep = (seg_len[seg_of] < 3) | ~obs[seg_off][seg_of] | ~np.isfinite(z_all).all(axis=1)
    Pm["x"] = np.where(keep, z_all[:, 0], sm[:, 0])
    Pm["y"] = np.where(keep, z_all[:, 1], sm[:, 1])
    Pm["coasted"] = False

    # coasted rows: frames inside a segment without a measurement
    gap = np.ones(N, bool)
    gap[pos] = False
    out = Pm
    if gap.any() and coast_n > 0:
        gi = np.flatnonzero(gap)
        gseg = np.searchsorted(seg_off, gi, side="right") - 1
        gfr = first_fr[gseg] + (gi - seg_off[gseg])
        gok = (seg_len[gseg] >= 3) & obs[seg_off][gseg] & f_rows[gfr]
        gi, gfr = gi[gok], gfr[gok]
        prev_slot = np.maximum.accumulate(np.where(~gap, np.arange(N), -1))[gi]
        slot_row = np.full(N, -1, np.int64)
        slot_row[pos] = np.arange(len(P))
        src = slot_row[prev_slot]
        C = P.iloc[src].copy().reset_index(drop=True)
        C["frame_idx"] = gfr.astype(P["frame_idx"].dtype)
        C["timestamp"] = P["timestamp"].to_numpy(float)[src] + (gfr - fr[src]) / fps
        C["x_raw"] = np.nan
        C["y_raw"] = np.nan
        C["x"], C["y"] = xs[gi, 0], xs[gi, 1]
        C["det_conf"] = 0.0
        C["coasted"] = True
        if "interpolated" in C.columns:
            C["interpolated"] = False
        if "live_wide" in C.columns:
            C["live_wide"] = f_live[gfr]
        out = pd.concat([Pm, C], ignore_index=True)
        rep["coasted_rows"] = int(len(C))
    out = pd.concat([out, other], ignore_index=True).sort_values(
        ["frame_idx", "track_id"], kind="stable").reset_index(drop=True)
    rep.update(segments=int(len(starts)), rows=int(len(P)), dropped=int(dropped[pos].sum()),
               restarts=int(restart.sum()), split_at_jumps=int(splits), repeated_frames=rep_dups)
    return out, rep


def is_smoothed(tracks: pd.DataFrame) -> bool:
    """Has this table's `x, y` already been through `smooth_player_states`?"""
    if "x_raw" not in tracks.columns or tracks.empty:
        return False
    p = tracks[_person_mask(tracks)]
    if p.empty:
        return False
    d = np.abs(p["x"].to_numpy(float) - p["x_raw"].to_numpy(float))
    return bool(np.nanmax(d, initial=0.0) > 1e-4) or bool(p["coasted"].any())
