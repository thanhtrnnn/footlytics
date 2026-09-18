"""Speed and acceleration from tracked positions.

Naive frame-differencing of positions is unusable: at 25 fps a 0.35 m position
error becomes a 8.75 m/s velocity error, which is larger than the signal. So we
smooth positions first, then differentiate, and we do it per track so that a
track ending never bleeds into the next one.

Distance in particular must be integrated from *smoothed* positions. Summing raw
frame-to-frame displacement accumulates every pixel of jitter into the total, and
it does so relentlessly -- there is no cancellation, because each step contributes
its absolute length. Measured on SoccerTrack v2's own hand-annotated ground truth,
whose boxes still wobble ~1.7 px per frame, that inflated distance by 1.9x versus
integrating smoothed positions.

Choosing the window
-------------------
Calibrated against the same ground truth, using the fact that footballers do not
run faster than about 9 m/s:

    window   median top speed   max top speed
    0.2 s        14.45 m/s        31.04 m/s     nonsense
    0.4 s         8.60 m/s        12.60 m/s     still faster than Usain Bolt
    0.8 s         7.34 m/s         8.67 m/s     physiologically sane
    1.2 s         6.96 m/s         8.23 m/s     sane, slightly blunted

Hence a 0.8 s default. It is a genuine trade-off: longer windows clip real
acceleration bursts, which is what sprint counts are built from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A player's top speed is ~10-12 m/s and their top acceleration ~8 m/s^2.
# Anything beyond these is a tracking artefact, not football.
MAX_HUMAN_SPEED = 12.0
MAX_HUMAN_ACCEL = 10.0


def _hampel(v: np.ndarray, window: int = 7, n_sigma: float = 3.0) -> np.ndarray:
    """Replace isolated outliers with the local median (Hampel filter).

    Tracking error is not Gaussian. Measured on SoccerTrack v2's own annotated
    ground truth, 6.5% of frames contain a physically impossible jump (>12 m/s)
    and 81% of those are single-frame spikes -- one bad box, then the track
    carries on as if nothing happened.

    A moving average is the wrong tool for that. Averaging does not remove a
    spike, it spreads it across the whole window, so one 10 m error becomes a
    smaller error on twenty frames. A median is immune to it: as long as
    outliers are a minority of the window, they contribute nothing at all.

    So: reject outliers first, then smooth. In that order.
    """
    if len(v) < window or window < 3:
        return v
    s = pd.Series(v)
    med = s.rolling(window, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=1).median()
    # 1.4826 makes the MAD a consistent estimator of sigma for normal data.
    sigma = 1.4826 * mad
    bad = (s - med).abs() > n_sigma * sigma.replace(0, np.nan)
    return s.where(~bad.fillna(False), med).to_numpy()


def _smooth(v: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average that keeps its length and handles short tracks."""
    if len(v) < 3 or window < 3:
        return v
    window = min(window | 1, len(v) if len(v) % 2 else len(v) - 1)
    if window < 3:
        return v
    pad = window // 2
    padded = np.concatenate([np.full(pad, v[0]), v, np.full(pad, v[-1])])
    kern = np.ones(window) / window
    return np.convolve(padded, kern, mode="valid")


def add_kinematics(tracks: pd.DataFrame, fps: float, smooth_s: float = 0.8,
                   clamp: bool = True) -> pd.DataFrame:
    """Fill the `speed` and `accel` columns, in m/s and m/s^2.

    `smooth_s` is the smoothing window in seconds; see the module docstring for
    how 0.8 s was chosen against real annotated football.
    """
    if tracks.empty:
        return tracks
    out = tracks.sort_values(["track_id", "frame_idx"], kind="stable").copy()
    win = max(int(round(smooth_s * fps)), 3)

    speeds = np.full(len(out), np.nan, np.float32)
    accels = np.full(len(out), np.nan, np.float32)

    for _, idx in out.groupby("track_id", observed=True).indices.items():
        if len(idx) < 3:
            continue
        f = out["frame_idx"].values[idx].astype(float)
        # Reject spikes, then smooth. Both steps are needed and the order matters.
        x = _smooth(_hampel(out["x"].values[idx].astype(float)), win)
        y = _smooth(_hampel(out["y"].values[idx].astype(float)), win)
        # Gradient against real frame numbers, so gaps where the track was
        # coasting do not masquerade as teleportation.
        dt = np.gradient(f) / fps
        dt[dt <= 0] = 1.0 / fps
        vx, vy = np.gradient(x) / dt, np.gradient(y) / dt
        sp = np.hypot(vx, vy)
        if clamp:
            sp = np.minimum(sp, MAX_HUMAN_SPEED)
        ac = np.gradient(_smooth(sp, win)) / dt
        if clamp:
            ac = np.clip(ac, -MAX_HUMAN_ACCEL, MAX_HUMAN_ACCEL)
        speeds[idx], accels[idx] = sp, ac

    out["speed"] = speeds
    out["accel"] = accels
    return out.sort_values(["frame_idx", "track_id"], kind="stable").reset_index(drop=True)


def smooth_positions(tracks: pd.DataFrame, fps: float, smooth_s: float = 0.4,
                     hampel_window: int = 7) -> pd.DataFrame:
    """Return a copy with x/y cleaned: spikes rejected, then lightly smoothed.

    For *display*. The stored MatchState keeps raw positions on purpose -- once
    you smooth in place you can never tell a tracking problem from a real one --
    but a radar drawn from raw coordinates jitters visibly, and that jitter is
    mostly single-frame annotation spikes rather than anything the player did.

    A shorter window than the kinematics default is used here: the eye tolerates
    a little noise far better than it tolerates a player gliding through a turn.
    """
    if tracks.empty:
        return tracks
    out = tracks.sort_values(["track_id", "frame_idx"], kind="stable").copy()
    win = max(int(round(smooth_s * fps)), 3)
    xs = out["x"].to_numpy(float).copy()
    ys = out["y"].to_numpy(float).copy()
    for _, idx in out.groupby("track_id", observed=True).indices.items():
        if len(idx) < 3:
            continue
        xs[idx] = _smooth(_hampel(xs[idx], hampel_window), win)
        ys[idx] = _smooth(_hampel(ys[idx], hampel_window), win)
    out["x"], out["y"] = xs.astype("float32"), ys.astype("float32")
    return out.sort_values(["frame_idx", "track_id"], kind="stable").reset_index(drop=True)


def physical_summary(state) -> pd.DataFrame:
    """Per-track distance, top speed and sprint count -- the fitness-coach table.

    Sprint thresholds follow the common senior-football convention
    (high-speed running >5.5 m/s, sprint >7.0 m/s). They are conventions, not
    physics: for youth football you should lower them, which is why they are
    arguments rather than constants.
    """
    return distance_summary(state.players, state.meta.fps)


def distance_summary(players: pd.DataFrame, fps: float,
                     hsr_thresh: float = 5.5, sprint_thresh: float = 7.0,
                     smooth_s: float = 0.8) -> pd.DataFrame:
    """Per-track distance, top speed and sprint count.

    Distance integrates SMOOTHED positions. Summing raw displacement was the
    original implementation and it overstated distance by 1.9x on real annotated
    football, because tracking jitter never cancels -- every wobble adds.

    `top_speed_ms` is the 99.5th percentile rather than the maximum, so one bad
    frame cannot define a player's sprint speed.

    Distances are for the clip processed. Do NOT scale them to 90 minutes: a
    4-minute passage is not a match, and players do not sustain its intensity.
    """
    win = max(int(round(smooth_s * fps)), 3)
    rows = []
    for tid, g in players.groupby("track_id", observed=True):
        g = g.sort_values("frame_idx")
        if len(g) < 3:
            continue
        x = _smooth(_hampel(g["x"].values.astype(float)), win)
        y = _smooth(_hampel(g["y"].values.astype(float)), win)
        steps = np.hypot(np.diff(x), np.diff(y))
        contiguous = np.diff(g["frame_idx"].values) == 1
        steps = steps[contiguous]                 # never bridge a tracking gap

        sp = g["speed"].values
        valid = ~np.isnan(sp)
        if valid.any():
            in_sprint = sp[valid] > sprint_thresh
            bursts = int(np.sum(np.diff(in_sprint.astype(int)) == 1)) + int(in_sprint[0])
            top = float(np.percentile(sp[valid], 99.5))
            step_sp = sp[valid][:len(steps)]
            hsr = float(steps[:len(step_sp)][step_sp > hsr_thresh].sum())
        else:
            bursts, top, hsr = 0, np.nan, 0.0

        rows.append({
            "track_id": int(tid),
            "team": g["team"].iloc[0],
            "role": g["role"].iloc[0],
            "jersey": g["jersey"].dropna().mode().iloc[0] if g["jersey"].notna().any() else np.nan,
            "frames": len(g),
            "minutes": len(g) / fps / 60,
            "distance_m": float(steps.sum()),
            "top_speed_ms": top,
            "hsr_m": hsr,
            "sprints": bursts,
        })
    return pd.DataFrame(rows).sort_values("distance_m", ascending=False).reset_index(drop=True)
