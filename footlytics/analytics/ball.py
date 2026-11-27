"""Ball trajectory post-processing.

The ball is lost far more often than a player is: it is small, fast, and spends
part of its time behind someone. `PitchTracker` coasts it on the Kalman filter
for a few frames, but once a track dies the frame simply has no ball row at all,
and every possession- or distance-to-ball statistic silently skips that frame.

Short gaps are the ones worth filling, and only by interpolating *between* two
real observations -- never by extrapolating past the last one, which invents a
ball flying off in whatever direction it was last seen going.

Carried over from the V0 prototype, where it worked on a wide one-row-per-frame
ball table. In Match State the ball is one row per frame in the long observation
table, so a gap is a *missing row* and filling it means inserting one.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from footlytics.analytics.ball_link import BALL_MAX_SPEED, BALL_NOISE_M, ball_reach
from footlytics.state.schema import Role, Team


def interpolate_ball(
    tracks: pd.DataFrame,
    fps: float,
    max_gap: int = 5,
    last_frame: Optional[int] = None,
    max_speed: float = BALL_MAX_SPEED,
    noise_m: float = BALL_NOISE_M,
) -> pd.DataFrame:
    """Insert ball rows for gaps of at most `max_gap` frames, linearly interpolated.

    Returns a new observation table. Inserted rows carry `interpolated=True`, a
    NaN `det_conf` and NaN bboxes -- nothing was detected, so there is no box and
    no confidence to report. Gaps longer than `max_gap`, and any stretch before
    the first or after the last sighting, are left empty.

    A gap is filled only when the two sightings around it are plausibly one ball in
    one continuous stretch of play: the same ball track id (a new id is a segment the
    tracker or `ball_link` declared unreachable from the previous one), the same period
    and shot, and a displacement the ball could have covered in the gap (`ball_reach`,
    the gate the selector uses). Otherwise "interpolating" would draw a ball flying
    between two different balls, at a speed no ball has.

    `max_gap <= 0` disables it and returns `tracks` unchanged.
    """
    if max_gap <= 0 or tracks.empty:
        return tracks

    is_ball = tracks["role"].astype(str) == Role.BALL.value
    ball = tracks[is_ball]
    if len(ball) < 2:
        return tracks

    # One ball per frame. Detector plus tracker can occasionally emit two; keep
    # the more confident, so the interpolation is anchored on the better track.
    ball = (ball.sort_values(["frame_idx", "det_conf"])
                .drop_duplicates("frame_idx", keep="last")
                .sort_values("frame_idx"))
    if last_frame is not None:
        ball = ball[ball["frame_idx"] <= int(last_frame)]
    if len(ball) < 2:
        return tracks

    f = ball["frame_idx"].to_numpy(np.int64)
    x, y = ball["x"].to_numpy(float), ball["y"].to_numpy(float)
    gap = np.diff(f) - 1
    ok = (gap >= 1) & (gap <= max_gap)
    for col in ("track_id", "period", "shot"):
        if col in ball.columns:
            v = ball[col].to_numpy()
            ok &= v[:-1] == v[1:]
    if "live_wide" in ball.columns:
        lw = ball["live_wide"].to_numpy(bool)
        ok &= lw[:-1] & lw[1:]
    dist = np.hypot(np.diff(x), np.diff(y))
    ok &= np.isfinite(dist) & (dist <= ball_reach((gap + 1) / float(fps), max_speed, noise_m))
    if not ok.any():
        return tracks

    left = np.flatnonzero(ok)                       # index of the sighting before each filled gap
    n = gap[left]
    src = np.repeat(left, n)                        # its left sighting, once per filled frame
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + 1   # 1..n inside each gap
    frac = k / (np.repeat(n, n) + 1.0)
    idx = f[src] + k
    new = pd.DataFrame({
        "frame_idx": idx.astype("int32"),
        "period": ball["period"].to_numpy()[src],
        "timestamp": idx / float(fps),
        "track_id": ball["track_id"].to_numpy()[src],
        "role": Role.BALL.value,
        "team": Team.UNKNOWN.value,
        "jersey": np.nan,
        "bbox_x": np.nan, "bbox_y": np.nan, "bbox_w": np.nan, "bbox_h": np.nan,
        "det_conf": np.nan,
        "x": x[src] + frac * (x[src + 1] - x[src]),
        "y": y[src] + frac * (y[src + 1] - y[src]),
        "z": 0.0,
        "speed": np.nan, "accel": np.nan,
        "interpolated": True,
    })
    for col in ("live_wide", "shot"):               # the same stretch of video as its neighbours
        if col in tracks.columns:
            new[col] = ball[col].to_numpy()[src]

    out = pd.concat([tracks, new], ignore_index=True)
    return out.sort_values(["frame_idx", "track_id"], kind="stable").reset_index(drop=True)


def ball_coverage(tracks: pd.DataFrame, frames: int, detected_only: bool = False) -> float:
    """Share of frames that have a ball position.

    `detected_only` counts only frames where the ball was actually seen, which is
    the number the detector should be judged on.
    """
    if not frames:
        return 0.0
    ball = tracks[tracks["role"].astype(str) == Role.BALL.value]
    if detected_only and "interpolated" in ball.columns:
        ball = ball[~ball["interpolated"].astype(bool)]
    return float(ball["frame_idx"].nunique() / frames)
