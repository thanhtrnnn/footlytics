"""Possession inferred from Match State alone, so pressing and transitions run without an event feed.

`press_distance` and `transitions` take a possession table shaped like SoccerTrack's
ground truth (`ballStatus` per frame). This builds that table from positions, in the
way tracking-data event detectors do (Vidal-Codina et al. 2022, arXiv 2202.00804):
a player has the ball while it is inside his possession zone, and a gain needs
evidence that the ball was touched, a change in its direction or speed.

`ballStatus`
    HOME / AWAY    one team controls the ball.
    CONTESTED      the ball is in reach of both teams and neither is clearly nearer, or a
                   player is nearest to the ball but no touch confirms he took it from the
                   holder. FIFA's fourth possession state ("in contest").
    BALLOUT        the ball is beyond the lines.
    UNKNOWN        no usable ball evidence: undetected, only interpolated, or loose for
                   longer than `max_gap_s`.

`control`
    A measured (not interpolated) ball frame with a player within `radius_m`. The nearer
    team controls it, unless both are in reach and closer together than `dominance_m`,
    which is a contest. Frames where nobody is in reach are free: a pass in flight keeps
    its team for up to `max_gap_s`.

`touch`
    A frame where the ball's velocity changes by at least `touch_dv_mps` over
    `TOUCH_WINDOW_S` either side and a player is in reach: the team of the nearer player.
    Possession only passes from a holder to the other team when that team has a touch
    while it is in control for `min_hold_s`, unless the holder has had nobody within
    `claim_radius_m` of the ball since the other team took control (a dribbled ball stays
    within about a stride of the carrier): then nobody contests it and the hold alone is
    enough. Taking over from nobody (after a restart, a loss of the ball, a contest) needs
    the hold alone.

The result carries two evidence columns for downstream turnover detection, `touch`
("HOME", "AWAY" or "") and `gain`, set on the first frame of each spell of possession to
how it was won: "touch", "unclaimed" (taken from a holder nobody was contesting it for) or
"hold" (taken from nobody). Every other column is as before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..state.schema import MatchState, Team

HOME, AWAY, BALLOUT, UNKNOWN, CONTESTED = "HOME", "AWAY", "BALLOUT", "UNKNOWN", "CONTESTED"
_STATUS = {Team.HOME.value: HOME, Team.AWAY.value: AWAY}

POSITION_ERROR_P90_M = 0.32
RELATIVE_ERROR_P90_M = POSITION_ERROR_P90_M * 2 ** 0.5
CONTROL_RADIUS_M = 1.0 + RELATIVE_ERROR_P90_M
DOMINANCE_MARGIN_M = RELATIVE_ERROR_P90_M
TOUCH_DV_MPS = 8.0
CLAIM_RADIUS_M = 2.0
TOUCH_WINDOW_S = 0.2

_NO_BALL, _FREE, _OUT, _CTRL_HOME, _CTRL_AWAY, _CONTEST = range(6)


def _frame_evidence(state: MatchState, radius_m: float, dominance_m: float,
                    out_margin_m: float) -> pd.DataFrame:
    """One row per tracked frame: ball xy, each team's nearest distance, and the raw frame class."""
    frames = (state.tracks.drop_duplicates("frame_idx").set_index("frame_idx")
                          [["period", "timestamp", "live_wide", "shot"]].sort_index())
    ball = (state.ball.drop_duplicates("frame_idx").set_index("frame_idx")
                      [["x", "y", "interpolated"]])
    ev = frames.join(ball.rename(columns={"x": "bx", "y": "by"}))
    ev.loc[~ev["live_wide"], ["bx", "by"]] = np.nan
    ev["interpolated"] = ev["interpolated"].fillna(False).astype(bool)

    p = state.players
    p = p[p["team"].isin(list(_STATUS))]
    p = p.join(ev[["bx", "by"]].dropna(), on="frame_idx", how="inner")
    ev["d_home"] = np.inf
    ev["d_away"] = np.inf
    if not p.empty:
        p = p.assign(d=np.hypot(p["x"] - p["bx"], p["y"] - p["by"]),
                     st=p["team"].astype(str).map(_STATUS))
        nearest = p.groupby(["frame_idx", "st"], observed=True)["d"].min().unstack()
        for team, col in ((HOME, "d_home"), (AWAY, "d_away")):
            if team in nearest.columns:
                ev.loc[nearest.index, col] = nearest[team].to_numpy()

    half_l = state.meta.pitch_length / 2 + out_margin_m
    half_w = state.meta.pitch_width / 2 + out_margin_m
    has_ball = ev["bx"].notna()
    out = has_ball & ((ev["bx"].abs() > half_l) | (ev["by"].abs() > half_w))
    measured = has_ball & ~ev["interpolated"]
    near_h, near_a = ev["d_home"] <= radius_m, ev["d_away"] <= radius_m
    contest = measured & near_h & near_a & ((ev["d_home"] - ev["d_away"]).abs() <= dominance_m)
    ctrl_h = measured & near_h & (ev["d_home"] < ev["d_away"]) & ~contest
    ctrl_a = measured & near_a & (ev["d_away"] < ev["d_home"]) & ~contest
    raw = np.full(len(ev), _FREE, dtype=np.int8)
    raw[~has_ball.to_numpy()] = _NO_BALL
    raw[ctrl_h.to_numpy()] = _CTRL_HOME
    raw[ctrl_a.to_numpy()] = _CTRL_AWAY
    raw[contest.to_numpy()] = _CONTEST
    raw[out.to_numpy()] = _OUT
    ev["raw"] = raw
    return ev


def _touches(ev: pd.DataFrame, fps: float, radius_m: float, dv_mps: float) -> np.ndarray:
    """Per frame of `ev`: "HOME" / "AWAY" where the ball's velocity changed with that team in reach."""
    k = max(1, round(TOUCH_WINDOW_S * fps))
    lo, hi = int(ev.index.min()), int(ev.index.max())
    full = ev.reindex(range(lo, hi + 1))
    live_shot = full["shot"].where(full["live_wide"].fillna(False).astype(bool))
    same_shot = (live_shot == live_shot.shift(k)) & (live_shot == live_shot.shift(-k))
    vx_in, vy_in = (full["bx"] - full["bx"].shift(k)) * fps / k, (full["by"] - full["by"].shift(k)) * fps / k
    vx_out, vy_out = (full["bx"].shift(-k) - full["bx"]) * fps / k, (full["by"].shift(-k) - full["by"]) * fps / k
    dv = np.hypot(vx_out - vx_in, vy_out - vy_in)
    d_home, d_away = full["d_home"].fillna(np.inf), full["d_away"].fillna(np.inf)
    reach = np.minimum(d_home, d_away) <= radius_m
    touch = (dv >= dv_mps) & same_shot & reach
    team = np.where(d_home <= d_away, HOME, AWAY)
    out = pd.Series(np.where(touch, team, ""), index=full.index)
    return out.reindex(ev.index).to_numpy()


def infer_possession(state: MatchState, radius_m: float = CONTROL_RADIUS_M, min_hold_s: float = 0.5,
                     max_gap_s: float = 2.0, out_margin_m: float = 2.0,
                     dominance_m: float = DOMINANCE_MARGIN_M, touch_dv_mps: float = TOUCH_DV_MPS,
                     claim_radius_m: float = CLAIM_RADIUS_M) -> pd.DataFrame:
    """Per-frame possession: frame_idx, period, timestamp, ballStatus, touch.

    A change of hands is backdated to the new team's first control frame. A frame
    carries the last status through a free ball or a detection gap of up to `max_gap_s`
    (counting frames missing from the table as gap); longer than that it is UNKNOWN.
    """
    fps = state.meta.fps
    ev = _frame_evidence(state, radius_m, dominance_m, out_margin_m)
    if ev.empty:
        return pd.DataFrame(columns=["frame_idx", "period", "timestamp", "ballStatus", "touch", "gain"])
    touch = _touches(ev, fps, radius_m, touch_dv_mps)
    hold_n = max(1, int(np.ceil(min_hold_s * fps)))
    max_gap_n = int(max_gap_s * fps)
    pre_n = max(1, round(TOUCH_WINDOW_S * fps))
    touch_cum = {team: np.concatenate([[0], np.cumsum(touch == team)]) for team in (HOME, AWAY)}

    def touched(team: str, a: int, b: int) -> bool:
        a = max(a, 0)
        return bool(touch_cum[team][b + 1] - touch_cum[team][a])

    frame_idx, raw = ev.index.to_numpy(), ev["raw"].to_numpy()
    claims = {HOME: (ev["d_home"] <= claim_radius_m).to_numpy(), AWAY: (ev["d_away"] <= claim_radius_m).to_numpy()}
    n = len(ev)
    status = np.full(n, UNKNOWN, dtype=object)
    gain = np.full(n, "", dtype=object)
    holder, display = None, UNKNOWN
    challenger, ch_start, ch_n = None, 0, 0
    contest_n, gap_frames, gap_rows = 0, 0, 0
    holder_claimed = -1
    for i in range(n):
        r = raw[i]
        if holder is not None and claims[holder][i]:
            holder_claimed = frame_idx[i]
        if i:
            gap_frames += int(frame_idx[i] - frame_idx[i - 1] - 1)
        no_evidence = r in (_NO_BALL, _FREE)
        if no_evidence:
            gap_frames += 1
            gap_rows += 1
        if gap_frames > max_gap_n:
            status[i - gap_rows + (1 if no_evidence else 0):i + (1 if no_evidence else 0)] = UNKNOWN
            holder, display, challenger, contest_n, gap_rows = None, UNKNOWN, None, 0, 0
        if no_evidence:
            status[i] = display
            continue
        gap_frames = gap_rows = 0
        if r == _OUT:
            holder, display, challenger, contest_n = None, BALLOUT, None, 0
        elif r == _CONTEST:
            contest_n += 1
            if contest_n >= hold_n:
                display = CONTESTED
        else:
            team = HOME if r == _CTRL_HOME else AWAY
            contest_n = 0
            if holder == team:
                challenger, display = None, team
            else:
                if challenger != team:
                    challenger, ch_start, ch_n = team, i, 0
                ch_n += 1
                if ch_n >= hold_n:
                    unclaimed = holder_claimed < frame_idx[ch_start]
                    touch_won = touched(team, ch_start - pre_n, i)
                    if holder is None or unclaimed or touch_won:
                        gain[ch_start] = "hold" if holder is None else "touch" if touch_won else "unclaimed"
                        holder, display, challenger, holder_claimed = team, team, None, frame_idx[i]
                        status[ch_start:i] = team
                    else:
                        display = CONTESTED
        status[i] = display

    out = ev.reset_index()[["frame_idx", "period", "timestamp"]]
    out["ballStatus"] = status
    out["touch"] = touch
    out["gain"] = gain
    return out
