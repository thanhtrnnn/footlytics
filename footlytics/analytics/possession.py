"""Possession inferred from Match State alone, so pressing and transitions run without an event feed.

`press_distance` and `transitions` take a possession table shaped like SoccerTrack's
ground truth (`ballStatus` per frame). This builds that table from positions:

`control`
    A frame where a home or away player is within `radius_m` of the ball. Frames
    where nobody is that close (a pass in flight, a loose ball) are free and never
    change possession by themselves.

`hold`
    Possession only changes hands once the other team has controlled the ball for
    `min_hold_s`, counted over their control frames with no touch by the incumbent
    in between. The change is then backdated to their first touch. Without the
    hold, every deflection on the way to a teammate would register as a turnover.

`gaps`
    The detector misses the ball in a quarter to half of the frames. A short gap
    carries the last team; a gap longer than `max_gap_s` is UNKNOWN rather than a
    guess, and the next team has to earn possession again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..state.schema import MatchState, Team

HOME, AWAY, BALLOUT, UNKNOWN = "HOME", "AWAY", "BALLOUT", "UNKNOWN"
_STATUS = {Team.HOME.value: HOME, Team.AWAY.value: AWAY}


def _control(state: MatchState, radius_m: float, out_margin_m: float) -> pd.Series:
    """frame_idx -> HOME / AWAY / BALLOUT for frames with a ball, None where it is free."""
    ball = state.ball.drop_duplicates("frame_idx").set_index("frame_idx")[["x", "y"]]
    half_l = state.meta.pitch_length / 2 + out_margin_m
    half_w = state.meta.pitch_width / 2 + out_margin_m
    p = state.players
    p = p[p["team"].isin(list(_STATUS))]
    p = p.join(ball, on="frame_idx", rsuffix="_ball", how="inner")
    p = p.assign(d=np.hypot(p["x"] - p["x_ball"], p["y"] - p["y_ball"]))
    nearest = p.loc[p.groupby("frame_idx")["d"].idxmin()].set_index("frame_idx")

    out = pd.Series(None, index=ball.index, dtype=object)
    near = nearest[nearest["d"] <= radius_m]
    out.loc[near.index] = near["team"].astype(str).map(_STATUS)
    outside = (ball["x"].abs() > half_l) | (ball["y"].abs() > half_w)
    out.loc[outside[outside].index] = BALLOUT
    return out


def infer_possession(state: MatchState, radius_m: float = 2.0, min_hold_s: float = 0.5,
                     max_gap_s: float = 2.0, out_margin_m: float = 1.0) -> pd.DataFrame:
    """Per-frame possession: frame_idx, period, timestamp, ballStatus.

    ballStatus is HOME / AWAY / BALLOUT as in SoccerTrack's ground truth, plus
    UNKNOWN before anyone has held the ball and across long detection gaps.
    """
    fps = state.meta.fps
    frames = (state.tracks.drop_duplicates("frame_idx")
                          .set_index("frame_idx")[["period", "timestamp"]].sort_index())
    control = _control(state, radius_m, out_margin_m)
    has_ball = frames.index.isin(control.index)
    hold_n = max(1, int(np.ceil(min_hold_s * fps)))
    max_gap_n = int(max_gap_s * fps)

    status = np.full(len(frames), UNKNOWN, dtype=object)
    current = UNKNOWN
    challenger, challenger_start, challenger_n = None, 0, 0
    gap = 0
    for i, f in enumerate(frames.index):
        if not has_ball[i]:
            gap += 1
            if gap > max_gap_n:
                # Too long to carry: the whole gap is unknown, and possession restarts.
                status[i - gap + 1:i + 1] = UNKNOWN
                current, challenger = UNKNOWN, None
            else:
                status[i] = current
            continue
        gap = 0
        c = control.loc[f]
        if c == BALLOUT:
            current, challenger = BALLOUT, None
        elif c is not None and c != current:
            if c != challenger:
                challenger, challenger_start, challenger_n = c, i, 0
            challenger_n += 1
            if challenger_n >= hold_n:
                status[challenger_start:i] = c
                current, challenger = c, None
        elif c is not None:
            challenger = None          # the incumbent touched it: the challenge is over
        status[i] = current

    out = frames.reset_index()
    out["ballStatus"] = status
    return out[["frame_idx", "period", "timestamp", "ballStatus"]]
