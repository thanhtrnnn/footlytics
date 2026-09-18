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

from footlytics.state.schema import Role, Team


def interpolate_ball(
    tracks: pd.DataFrame,
    fps: float,
    max_gap: int = 5,
    last_frame: Optional[int] = None,
) -> pd.DataFrame:
    """Insert ball rows for gaps of at most `max_gap` frames, linearly interpolated.

    Returns a new observation table. Inserted rows carry `interpolated=True`, a
    NaN `det_conf` and NaN bboxes -- nothing was detected, so there is no box and
    no confidence to report. Gaps longer than `max_gap`, and any stretch before
    the first or after the last sighting, are left empty.

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
                .set_index("frame_idx")
                .sort_index())

    first, last = int(ball.index.min()), int(ball.index.max())
    if last_frame is not None:
        last = min(last, int(last_frame))
    full = pd.RangeIndex(first, last + 1)
    missing = ~full.isin(ball.index)
    if not missing.any():
        return tracks

    # Only fill runs short enough to be a brief occlusion rather than a lost ball.
    run_id = pd.Series(missing, index=full)
    run_id = (run_id != run_id.shift()).cumsum()
    run_len = pd.Series(missing, index=full).groupby(run_id).transform("sum")
    fillable = pd.Series(missing, index=full) & (run_len <= max_gap)
    if not fillable.any():
        return tracks

    wide = ball.reindex(full)
    # limit_area="inside" is what keeps this an interpolation: no extrapolation
    # past the ends, even though reindex made room at neither.
    xy = wide[["x", "y"]].interpolate(method="linear", limit_area="inside")
    fillable &= xy["x"].notna()
    if not fillable.any():
        return tracks

    idx = full[fillable.to_numpy()]
    new = pd.DataFrame({
        "frame_idx": idx.astype("int32"),
        "period": wide["period"].ffill().reindex(idx).to_numpy(),
        "timestamp": idx.to_numpy() / float(fps),
        "track_id": wide["track_id"].ffill().reindex(idx).to_numpy(),
        "role": Role.BALL.value,
        "team": Team.UNKNOWN.value,
        "jersey": np.nan,
        "bbox_x": np.nan, "bbox_y": np.nan, "bbox_w": np.nan, "bbox_h": np.nan,
        "det_conf": np.nan,
        "x": xy.loc[idx, "x"].to_numpy(),
        "y": xy.loc[idx, "y"].to_numpy(),
        "z": 0.0,
        "speed": np.nan, "accel": np.nan,
        "interpolated": True,
    })

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
