"""Rule-based tactical sequences: the "Low block -- 22 sequences" an analyst searches for.

Each sequence is a stretch where one team's shape meets a label's conditions for
long enough, returned with the evidence that put it there so an analyst can mark
it right or wrong (docs/tactical-query-vision.md section 2). The thresholds are
first guesses to be tuned per club against those marks, not settled definitions.

Every figure is measured over the players the camera showed. `n_visible` and
`n_visible_opp` (players on the pitch in view, goalkeepers included, median over the
sequence) say how much of each team the measurement rests on; a broadcast frame
shows a median of about 14 of 22.

`low_block`
    Out of possession, or with the ball in the team's own third and nobody in control,
    the four deepest visible outfielders averaging at or behind `low_block_line` in the
    team's own attacking frame (-22.5 m, which is 30 m from its own goal line; the third
    starts 35 m out), compact, for `low_block_min_s`. Still-ball set-piece samples are
    skipped (`ball_still`). The second branch also admits a team nominally in possession
    while it is pinned in its own box (ball at or behind -25 m, centroid behind -25 m,
    line behind -30 m), and a run is dropped when the team had the ball for most of it.

`mid_block`
    Out of possession, the three deepest visible outfielders averaging in the middle
    third (SkillCorner's mid block), compact, with at least `mid_block_min_visible`
    outfielders in view, for `low_block_min_s`. A line read from the visible players can
    only be higher than the real one, so a shape with its deepest players off screen
    reads here as a mid block; the visibility floor and `n_visible` are the guard.

`high_press`
    Out of possession and play live (no dead ball), the ball in the opponent's defensive
    third and `high_press_n` outfielders of the pressing team within 10 m of it (the
    goalkeeper is not counted), for `high_press_min_s`. The back line is not tested:
    it is measured on the players in view, and when the camera shows only the
    opponent's box that is the forwards. `high_press_line` is still accepted; when
    given, the line is required only in frames with `mid_block_min_visible` outfielders
    in view.

`flank_overload`
    The team has the ball and the ball is in one flank channel of the opponent's half
    (wider than `flank_y`, a sixth of the pitch width from the centre line by default:
    the wing plus the half-space), with at least `overload_n` of the team's players in
    that channel against at most `overload_max_defenders` opponents, for `overload_min_s`.
    The counts reported are from the single sampled frame of the run with the widest
    margin, not maxima over the run. Samples where either team has fewer than
    `overload_min_visible` players in view are skipped: a defender off screen turns a
    fair fight into a false overload. "Right" is the attacker's right, assuming +y is the
    touchline nearer the main camera.

`build_up`
    In possession with the ball in the team's own third, for `build_up_min_s`.
    `n_opp_in_third` is how many opponents stood in that third (the median over the
    sequence): where the opposition is, not whether it presses. `n_pressing_ball` is
    how many opponents were pressing the ball (`tactics.press_on_ball`), the median over
    the sequence. `n_pressers` repeats `n_opp_in_third` for readers of the old column.
    Dead-ball samples are skipped: the ball has not moved for `still_s` (a goal
    kick being placed), or the players have stopped, their median speed over the
    last second under `stopped_mps` (a foul, an injury). Play is not on then.

`midfield_loss` and `turnover`
    A turnover (`tactics.transitions`: both teams held the ball for its minimum hold and
    the winner touched it) as a `loss_window_s` sequence of the team that lost it.
    `detect_turnovers` returns them for the whole pitch, labelled `turnover`, with `zone`
    (the third where the loser lost it, in its own attacking frame); `midfield_loss` is
    those with `zone == "middle_third"`. Evidence:
    `n_pressing_ball` (the most of the loser's outfielders closing in on the new carrier,
    at 2 m/s or more and within 3 m or, ahead of him, 9 m, in any sampled frame within
    `counter_press_s` of the loss: the dispossessed player standing beside the ball is not
    counter-pressing), `retreat_mps` (how fast its centroid moved back, negative = towards its own
    goal, over the players in view throughout the window, NaN when too few were),
    `gain_5s_m` (how far the winner moved the ball up the pitch in 5 s, the
    attacking-transition measure), and `ball_dist_m` (its second-nearest outfielder to the ball, mean
    from one second after the loss to `counter_press_s`; kept for old readers).
    `response` is `counter_press` when anyone was closing in, else `retreat` when
    `retreat_mps` is at most the `retreat_mps` threshold, `hold` when it is above it, and
    `unclear` when it could not be measured.

Only live wide-camera frames count (`MatchState.tracks["live_wide"]`, see `perception.shots`): replays,
close-ups and graphics are dropped, and a sequence never spans a cut (`shot` id change), however short.

The state must already be normalised (`tactics.normalise_direction`): home
attacks +x in every period.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..state.schema import MatchState, Team
from .tactics import (
    attacking_sign, goalkeepers, press_distance, press_on_ball, retreat_speed, team_block, transitions)

_OPPONENT_STATUS = {Team.HOME.value: "AWAY", Team.AWAY.value: "HOME"}
_OWN_STATUS = {Team.HOME.value: "HOME", Team.AWAY.value: "AWAY"}
_OTHER = {Team.HOME.value: Team.AWAY.value, Team.AWAY.value: Team.HOME.value}
COLUMNS = ["label", "team", "period", "start_s", "end_s", "start_frame", "end_frame",
           "duration_s", "def_line_x", "compactness_m", "n_within_10m",
           "flank", "n_attackers", "n_defenders", "in_possession_share",
           "n_pressers", "ball_dist_m", "retreat_mps", "response",
           "n_visible", "n_visible_opp", "zone", "n_opp_in_third", "n_pressing_ball",
           "gain_5s_m", "def_line3_x"]
_NAN = float("nan")


def detect_sequences(state: MatchState, possession: pd.DataFrame, sample_every: int = 5,
                     low_block_line: float = -22.5, low_block_compact_m: float = 14.0,
                     low_block_min_s: float = 8.0, mid_block_min_visible: int = 8,
                     high_press_line: float | None = None, high_press_n: int = 3,
                     high_press_min_s: float = 3.0,
                     flank_y: float | None = None, overload_n: int = 4,
                     overload_max_defenders: int = 3, overload_min_visible: int = 7,
                     overload_min_s: float = 3.0,
                     build_up_min_s: float = 4.0, build_up_min_centroid_x: float = -25.0,
                     still_s: float = 2.0, still_m: float = 1.0,
                     stopped_mps: float = 1.0,
                     loss_window_s: float = 6.0, counter_press_s: float = 3.0,
                     counter_press_m: float | None = None, retreat_mps: float = -1.0,
                     with_turnovers: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """One row per sequence: label, team, period, start/end, and mean evidence over it.

    `counter_press_m` is accepted for old profiles and ignored: counter-pressing is
    judged by `tactics.press_on_ball`, not by a distance. With `with_turnovers`, returns
    (sequences, every turnover in all thirds) so a caller needing both detects once.
    """
    fps = state.meta.fps
    frames = _frames_of(state)
    state = _live_only(state)
    opp_third = state.meta.pitch_length / 2 - state.meta.pitch_length / 3
    blk, pressure = _block_frame(state, possession, frames, sample_every, still_s, still_m, stopped_mps)
    if blk.empty:
        empty = pd.DataFrame(columns=COLUMNS)
        return (empty, empty) if with_turnovers else empty

    flanks = flank_counts(state, sample_every=sample_every,
                          flank_y=state.meta.pitch_width / 6 if flank_y is None else flank_y)
    blk = blk.merge(flanks, on=["frame_idx", "team"], how="left")
    blk["flank"] = blk["flank"].fillna("")
    for c in ("n_attackers", "n_defenders"):
        blk[c] = blk[c].fillna(0).astype(int)

    oop = blk["out_of_possession"]
    deep_scramble = (
        (~blk["in_possession"] & ((blk["ball_x_own"] <= -opp_third) | (blk["n_opp_in_third"] >= 3)))
        | (blk["in_possession"] & (blk["ball_x_own"] <= -25.0)
           & (blk["centroid_x"] < build_up_min_centroid_x) & (blk["def_line_x"] <= -30.0))
    )
    lb_gap = max(2 * sample_every, round(1.0 * fps))
    compact = blk["compactness_m"] <= low_block_compact_m
    seen_enough = blk["n_outfield"] >= mid_block_min_visible
    high_press_mask = (oop & ~blk["dead_ball"] & (blk["ball_x_own"] >= opp_third)
                       & (blk["n_within_10m"] >= high_press_n))
    if high_press_line is not None:
        high_press_mask &= seen_enough & (blk["def_line_x"] >= high_press_line)
    rules = {
        "low_block": ((oop | deep_scramble) & (blk["def_line_x"] <= low_block_line)
                      & compact & ~blk["ball_still"], low_block_min_s, lb_gap),
        "mid_block": (oop & seen_enough & (blk["def_line3_x"] > -opp_third)
                      & (blk["def_line3_x"] <= opp_third) & compact & ~blk["dead_ball"],
                      low_block_min_s, lb_gap),
        "high_press": (high_press_mask, high_press_min_s, 2 * sample_every),
        "flank_overload": ((blk["n_attackers"] >= overload_n)
                           & (blk["n_defenders"] <= overload_max_defenders)
                           & (blk["n_visible"] >= overload_min_visible)
                           & (blk["n_visible_opp"] >= overload_min_visible)
                           & blk["in_possession"] & ~blk["dead_ball"],
                           overload_min_s, 2 * sample_every),
        "build_up": (blk["in_possession"] & (blk["ball_x_own"] <= -opp_third)
                     & (blk["centroid_x"] >= build_up_min_centroid_x)
                     & ~blk["dead_ball"], build_up_min_s, 2 * sample_every),
    }
    rows = []
    for label, (mask, min_s, gap_f) in rules.items():
        rows += _runs(blk[mask], label, min_s, sample_every, fps, max_gap_frames=gap_f)
    turnovers = _turnover_rows(state, possession, blk, pressure, frames, sample_every, loss_window_s,
                               counter_press_s, retreat_mps, stopped_mps,
                               ball_still(state, still_s, still_m))
    if not turnovers.empty:
        rows += turnovers[turnovers["zone"] == "middle_third"].assign(label="midfield_loss").to_dict("records")
    out = pd.DataFrame(rows, columns=COLUMNS).sort_values(["start_s", "label"]).reset_index(drop=True)
    return (out, turnovers) if with_turnovers else out


def detect_turnovers(state: MatchState, possession: pd.DataFrame, sample_every: int = 5,
                     loss_window_s: float = 6.0, counter_press_s: float = 3.0,
                     retreat_mps: float = -1.0, stopped_mps: float = 1.0,
                     still_s: float = 2.0, still_m: float = 1.0) -> pd.DataFrame:
    """One `turnover` row per live turnover on the whole pitch, for the team that lost the ball.

    Same columns as `detect_sequences`; `zone` says which third the loser lost it in.
    """
    frames = _frames_of(state)
    state = _live_only(state)
    blk, pressure = _block_frame(state, possession, frames, sample_every, still_s, still_m, stopped_mps)
    return _turnover_rows(state, possession, blk, pressure, frames, sample_every, loss_window_s,
                          counter_press_s, retreat_mps, stopped_mps, ball_still(state, still_s, still_m))


def _frames_of(state: MatchState) -> pd.DataFrame:
    return state.tracks[["frame_idx", "live_wide", "shot"]].drop_duplicates("frame_idx").sort_values("frame_idx")


def _live_only(state: MatchState) -> MatchState:
    if state.tracks["live_wide"].all():
        return state
    return MatchState(state.meta, state.tracks[state.tracks["live_wide"]])


def _block_frame(state: MatchState, possession: pd.DataFrame, frames: pd.DataFrame, sample_every: int,
                 still_s: float, still_m: float, stopped_mps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per sampled frame and team: block shape, possession flags, pressure, dead-ball flags and visibility.

    Also returns the `press_distance` table (renamed so its `team` is the pressing team).
    `state` must already be live-only.
    """
    blk = team_block(state, sample_every=sample_every, min_outfield=4)
    pressure = press_distance(state, possession, sample_every=sample_every)
    if not pressure.empty:
        pressure = pressure.rename(columns={"pressing_team": "team"})
    if blk.empty:
        return blk, pressure
    blk["shot"] = blk["frame_idx"].map(frames.set_index("frame_idx")["shot"]).astype(int)

    status = possession.drop_duplicates("frame_idx").set_index("frame_idx")["ballStatus"]
    blk["out_of_possession"] = blk["frame_idx"].map(status) == blk["team"].map(_OPPONENT_STATUS)
    blk["in_possession"] = blk["frame_idx"].map(status) == blk["team"].map(_OWN_STATUS)

    if not pressure.empty:
        blk = blk.merge(pressure[["frame_idx", "team", "n_within_10m"]],
                        on=["frame_idx", "team"], how="left")
    else:
        blk["n_within_10m"] = pd.NA
    blk["n_within_10m"] = blk["n_within_10m"].fillna(0).astype(int)

    on_ball = press_on_ball(state, sample_every=sample_every)
    blk["opponent"] = blk["team"].map(_OTHER)
    blk = blk.merge(on_ball.rename(columns={"team": "opponent", "n_pressing": "n_pressing_ball"})
                    [["frame_idx", "opponent", "n_pressing_ball"]], on=["frame_idx", "opponent"], how="left")
    blk["n_pressing_ball"] = blk["n_pressing_ball"].fillna(0).astype(int)

    ball = state.ball.drop_duplicates("frame_idx").set_index("frame_idx")
    blk["ball_x_own"] = blk["frame_idx"].map(ball["x"]) * blk["team"].map(attacking_sign)
    third = state.meta.pitch_length / 2 - state.meta.pitch_length / 3
    blk["ball_still"] = blk["frame_idx"].map(ball_still(state, still_s, still_m)).fillna(False).astype(bool)
    blk["play_stopped"] = blk["frame_idx"].map(
        play_stopped(state, sample_every, stopped_mps)).fillna(False).astype(bool)
    blk["ball_active"] = blk["frame_idx"].map(ball_active(state, still_s)).fillna(False).astype(bool)
    blk["dead_ball"] = blk["ball_still"] | blk["play_stopped"]
    blk = blk.merge(own_third_opponents(state, sample_every, third), on=["frame_idx", "team"], how="left")
    blk["n_opp_in_third"] = blk["n_opp_in_third"].fillna(0).astype(int)
    blk = blk.merge(visible_counts(state, sample_every), on=["frame_idx", "team"], how="left")
    for c in ("n_visible", "n_visible_opp"):
        blk[c] = blk[c].fillna(0).astype(int)
    return blk, pressure


def visible_counts(state: MatchState, sample_every: int = 5) -> pd.DataFrame:
    """Per sampled frame and team: its players on the pitch in view (keeper included) and the opponent's."""
    p = state.players
    half_l, half_w = state.meta.pitch_length / 2, state.meta.pitch_width / 2
    p = p[(p["frame_idx"] % sample_every == 0) & p["team"].isin([Team.HOME.value, Team.AWAY.value])
          & (p["x"].abs() <= half_l) & (p["y"].abs() <= half_w)]
    n = p.groupby(["frame_idx", "team"], observed=True).size().unstack(fill_value=0)
    for team in (Team.HOME.value, Team.AWAY.value):
        if team not in n.columns:
            n[team] = 0
    rows = []
    for team in (Team.HOME.value, Team.AWAY.value):
        rows.append(pd.DataFrame({"frame_idx": n.index.astype(int), "team": team,
                                  "n_visible": n[team].to_numpy(),
                                  "n_visible_opp": n[_OTHER[team]].to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def flank_counts(state: MatchState, sample_every: int = 5, flank_y: float = 34.0 / 3) -> pd.DataFrame:
    """Per sampled frame and team: the flank the ball is in, and who is there.

    Taken in the team's own attacking frame (x and y times `attacking_sign`, the
    180-degree turn `normalise_direction` uses), so "right" is the attacker's right for
    both teams. The flank is the one the ball is in: the ball must be in the opponent's
    half and at least `flank_y` from the centre line; otherwise the frame has no flank
    and counts nothing. Counts are the players on the pitch in that channel in the
    opponent's half, the team's own (`n_attackers`) and the opponent's (`n_defenders`).
    """
    ball = state.ball.drop_duplicates("frame_idx").set_index("frame_idx")[["x", "y"]]
    p = state.players
    half_l, half_w = state.meta.pitch_length / 2, state.meta.pitch_width / 2
    p = p[(p["frame_idx"] % sample_every == 0) & p["team"].isin([Team.HOME.value, Team.AWAY.value])
          & (p["x"].abs() <= half_l) & (p["y"].abs() <= half_w)]
    rows = []
    for frame, g in p.groupby("frame_idx", observed=True):
        if frame not in ball.index:
            continue
        for team in (Team.HOME.value, Team.AWAY.value):
            sign = attacking_sign(team)
            bx, by = float(ball.loc[frame, "x"]) * sign, float(ball.loc[frame, "y"]) * sign
            if bx < 0 or abs(by) < flank_y:
                continue
            side = "right" if by > 0 else "left"
            x, y = g["x"].to_numpy() * sign, g["y"].to_numpy() * sign
            own = (g["team"] == team).to_numpy()
            zone = (x >= 0) & ((y >= flank_y) if side == "right" else (y <= -flank_y))
            rows.append({"frame_idx": int(frame), "team": team, "flank": side,
                         "n_attackers": int((zone & own).sum()), "n_defenders": int((zone & ~own).sum())})
    return pd.DataFrame(rows, columns=["frame_idx", "team", "flank", "n_attackers", "n_defenders"])


def ball_still(state: MatchState, still_s: float = 2.0, still_m: float = 1.0) -> pd.Series:
    """frame_idx -> True where the ball stayed within `still_m` over the last `still_s`.

    A dead ball (a foul, a goal kick being placed) sits still; in play it rarely
    does for two seconds. Frames with the ball seen in under half of the window
    are not called still: a detector gap is not a stoppage.
    """
    b = state.ball.drop_duplicates("frame_idx").set_index("frame_idx")[["x", "y"]].sort_index()
    if b.empty:
        return pd.Series(dtype=bool)
    b = b.reindex(range(int(b.index.min()), int(b.index.max()) + 1))
    n = max(2, round(still_s * state.meta.fps))
    roll = b.rolling(n, min_periods=n // 2 + 1)
    spread = (roll.max() - roll.min()).pow(2).sum(axis=1, min_count=2).pow(0.5)
    return (spread <= still_m) & roll.count()["x"].ge(n // 2 + 1)


def ball_active(state: MatchState, window_s: float = 2.0, min_spread_m: float = 3.0) -> pd.Series:
    """frame_idx -> True where the ball's 15%-85% spatial spread over `window_s` exceeds `min_spread_m`."""
    b = state.ball.drop_duplicates("frame_idx").set_index("frame_idx")[["x", "y"]].sort_index()
    if b.empty:
        return pd.Series(dtype=bool)
    b = b.reindex(range(int(b.index.min()), int(b.index.max()) + 1))
    n = max(4, round(window_s * state.meta.fps))
    mp = max(2, n // 4)
    qx = b["x"].rolling(n, min_periods=mp)
    qy = b["y"].rolling(n, min_periods=mp)
    spread = np.hypot(qx.quantile(0.85) - qx.quantile(0.15), qy.quantile(0.85) - qy.quantile(0.15))
    return (spread > min_spread_m).fillna(False)


def play_stopped(state: MatchState, sample_every: int = 5, stopped_mps: float = 1.0) -> pd.Series:
    """Sampled frame_idx -> True where the players' median speed over the last second is under
    `stopped_mps`: everyone walking or standing, as after a whistle.

    In play somebody is always running, so the median player stays above a
    jog; at a foul or an injury the whole pitch slows to a walk.
    """
    p = state.players
    p = p[p["team"].isin([Team.HOME.value, Team.AWAY.value])][["frame_idx", "track_id", "x", "y"]]
    lag = round(state.meta.fps)
    now = p[p["frame_idx"] % sample_every == 0]
    before = p.assign(frame_idx=p["frame_idx"] + lag)
    j = now.merge(before, on=["frame_idx", "track_id"], suffixes=("", "_1s"))
    if j.empty:
        return pd.Series(dtype=bool)
    speed = np.hypot(j["x"] - j["x_1s"], j["y"] - j["y_1s"]) * state.meta.fps / lag
    return speed.groupby(j["frame_idx"]).median() < stopped_mps


def own_third_opponents(state: MatchState, sample_every: int, third_x: float) -> pd.DataFrame:
    """Per sampled frame and team: opponents on the pitch inside the team's own third."""
    p = state.players
    half_l, half_w = state.meta.pitch_length / 2, state.meta.pitch_width / 2
    p = p[(p["frame_idx"] % sample_every == 0) & p["team"].isin([Team.HOME.value, Team.AWAY.value])
          & (p["x"].abs() <= half_l) & (p["y"].abs() <= half_w)]
    out = []
    for team in (Team.HOME.value, Team.AWAY.value):
        opp = p[p["team"] == _OTHER[team]]
        inside = opp[opp["x"] * attacking_sign(team) <= -third_x]
        n = inside.groupby("frame_idx", observed=True).size()
        out.append(pd.DataFrame({"frame_idx": n.index.astype(int), "team": team,
                                 "n_opp_in_third": n.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def _turnover_rows(state: MatchState, possession: pd.DataFrame, blk: pd.DataFrame,
                   pressure: pd.DataFrame, frames: pd.DataFrame, sample_every: int,
                   window_s: float, counter_press_s: float, retreat_mps: float,
                   stopped_mps: float, ball_still_s: pd.Series) -> pd.DataFrame:
    """A `turnover` row per live turnover, for the team that lost the ball."""
    fps = state.meta.fps
    turnovers = transitions(state, possession)
    if turnovers.empty:
        return pd.DataFrame(columns=COLUMNS)
    last_frame = int(state.tracks["frame_idx"].max())
    has_cuts = (~frames["live_wide"]).any() or frames["shot"].nunique() > 1
    shot_of = frames.set_index("frame_idx")["shot"] if has_cuts else None
    ball_df = state.ball.drop_duplicates("frame_idx").set_index("frame_idx")
    on_ball = press_on_ball(state, sample_every=sample_every)
    keepers = set(goalkeepers(state).values())
    out = []
    for t in turnovers.itertuples():
        team = _OTHER[t.won_by]
        f0 = int(t.frame_idx)
        end_bound = last_frame
        if has_cuts:
            s0 = int(shot_of.loc[f0])
            bad = frames.loc[(frames["frame_idx"] > f0) & (~frames["live_wide"] | (frames["shot"] != s0)),
                             "frame_idx"]
            if len(bad):
                end_bound = min(end_bound, int(bad.iloc[0]) - 1)
            in_shot = frames.loc[(frames["frame_idx"] >= f0) & (frames["frame_idx"] <= end_bound)
                                 & frames["live_wide"] & (frames["shot"] == s0), "frame_idx"]
            if len(in_shot):
                end_bound = min(end_bound, int(in_shot.max()))
        f1 = min(end_bound, f0 + round(window_s * fps) - 1)
        if f1 - f0 + 1 < counter_press_s * fps:
            continue  # cut off before the response can be judged
        b = blk[(blk["team"] == team) & blk["frame_idx"].between(f0, f1)] if len(blk) else blk
        if len(b) and "ball_still" in b.columns and b["ball_still"].any():
            continue
        if bool(ball_still_s.loc[f0:f1].any()):
            continue
        b_late = ball_df[(ball_df.index >= f0 + int(counter_press_s * fps)) & (ball_df.index <= f1)]
        if len(b_late) >= 5 and "speed" in b_late.columns:
            sp = b_late["speed"].dropna()
            if len(sp) >= 5 and float(sp.median()) < stopped_mps:
                continue
        pr = (pressure[(pressure["team"] == team) & pressure["frame_idx"].between(f0, f1)]
              if len(pressure) else pressure)
        early = (pr[pr["frame_idx"].between(f0 + fps, f0 + counter_press_s * fps - 1)]
                 if len(pr) else pr)
        ball_dist = float(early["second_nearest_m"].mean()) if len(early) else _NAN
        pressing = on_ball[(on_ball["team"] == team)
                           & on_ball["frame_idx"].between(f0, f0 + counter_press_s * fps)]
        n_closing = int(pressing["n_closing"].max()) if len(pressing) else 0
        retreat = retreat_speed(state, team, f0, f1, sample_every=sample_every, keepers=keepers)
        if n_closing > 0:
            response = "counter_press"
        elif np.isnan(retreat):
            response = "unclear"
        else:
            response = "retreat" if retreat <= retreat_mps else "hold"
        vis = _visible_in(b)
        out.append({
            "label": "turnover", "team": team, "period": int(t.period),
            "start_s": float(t.timestamp), "end_s": float(t.timestamp + (f1 - f0 + 1) / fps),
            "start_frame": f0, "end_frame": f1, "duration_s": float((f1 - f0 + 1) / fps),
            "def_line_x": float(b["def_line_x"].mean()) if len(b) else _NAN,
            "compactness_m": float(b["compactness_m"].mean()) if len(b) else _NAN,
            "n_within_10m": int(pr["n_within_10m"].max()) if len(pr) else 0,
            "flank": "", "n_attackers": 0, "n_defenders": 0, "in_possession_share": 0.0,
            "n_pressers": 0, "ball_dist_m": ball_dist, "retreat_mps": retreat,
            "response": response, "n_visible": vis[0], "n_visible_opp": vis[1],
            "zone": t.zone, "n_opp_in_third": 0, "n_pressing_ball": n_closing,
            "gain_5s_m": float(t.gain_5s_m), "def_line3_x": float(b["def_line3_x"].mean()) if len(b) else _NAN,
        })
    return pd.DataFrame(out, columns=COLUMNS)


def _visible_in(b: pd.DataFrame) -> tuple[int, int]:
    """Median own and opponent players in view over the sampled frames of `b`."""
    if b.empty or "n_visible" not in b.columns:
        return 0, 0
    return round(float(b["n_visible"].median())), round(float(b["n_visible_opp"].median()))


def _runs(hits: pd.DataFrame, label: str, min_s: float, step: int, fps: float,
          max_gap_frames: int | None = None) -> list[dict]:
    """Group consecutive samples per team and period into sequences of at least `min_s`.

    One missing sample is tolerated by default (`2 * step`), so a single noisy frame does not
    split a run in two. `max_gap_frames` overrides that gap budget within the same shot.
    A flank overload is also split by flank: a switch of play is two sequences, not one.
    """
    out = []
    gap = 2 * step if max_gap_frames is None else max_gap_frames
    keys = ["team", "period"] + (["flank"] if label == "flank_overload" else [])
    for key, g in hits.sort_values("frame_idx").groupby(keys):
        team, period = key[0], key[1]
        run_id = ((g["frame_idx"].diff() > gap) | (g["shot"].diff() != 0)).cumsum()
        for _, r in g.groupby(run_id):
            duration = (r["frame_idx"].iloc[-1] - r["frame_idx"].iloc[0] + step) / fps
            if duration < min_s:
                continue
            if label == "low_block" and (
                not (
                    bool(r["out_of_possession"].any())
                    or (float(r["n_opp_in_third"].median()) >= 3 and float(r["ball_x_own"].median()) <= -25.0)
                )
                or float(r["in_possession"].mean()) > 0.65
                or (
                    float(r["play_stopped"].mean()) > 0.55
                    and (
                        float(r["compactness_m"].mean()) <= 6.5
                        or float(r["ball_active"].mean()) < 0.30
                    )
                )
            ):
                continue
            snap = r.loc[(r["n_attackers"] - r["n_defenders"]).idxmax()] if label == "flank_overload" else None
            vis = ((int(snap["n_visible"]), int(snap["n_visible_opp"])) if snap is not None
                   else _visible_in(r))
            n_opp = round(float(r["n_opp_in_third"].median()))
            out.append({
                "label": label, "team": team, "period": int(period),
                "start_s": float(r["timestamp"].iloc[0]),
                "end_s": float(r["timestamp"].iloc[-1] + step / fps),
                "start_frame": int(r["frame_idx"].iloc[0]),
                "end_frame": int(r["frame_idx"].iloc[-1] + step - 1),
                "duration_s": float(duration),
                "def_line_x": float(r["def_line_x"].mean()),
                "compactness_m": float(r["compactness_m"].mean()),
                "n_within_10m": int(r["n_within_10m"].max()),
                "flank": key[2] if len(key) > 2 else "",
                "n_attackers": int(snap["n_attackers"]) if snap is not None else 0,
                "n_defenders": int(snap["n_defenders"]) if snap is not None else 0,
                "in_possession_share": float(r["in_possession"].mean()),
                "n_pressers": n_opp,
                "ball_dist_m": _NAN, "retreat_mps": _NAN, "response": "",
                "n_visible": vis[0], "n_visible_opp": vis[1], "zone": "",
                "n_opp_in_third": n_opp,
                "n_pressing_ball": round(float(r["n_pressing_ball"].median())),
                "gain_5s_m": _NAN, "def_line3_x": float(r["def_line3_x"].mean()),
            })
    return out


def ball_in_play_seconds(state: MatchState, possession: pd.DataFrame, sample_every: int = 5,
                         still_s: float = 2.0, still_m: float = 1.0, stopped_mps: float = 1.0) -> dict:
    """Seconds of live wide-camera footage with the ball in play, and how possession split them.

    A sampled live frame is in play unless the ball is beyond the lines (`BALLOUT`), has
    not moved for `still_s` (`ball_still`) or the players have stopped (`play_stopped`).
    `possession` maps each in-play second to HOME / AWAY / CONTESTED / UNKNOWN.
    """
    fps = state.meta.fps
    t = state.tracks[state.tracks["live_wide"]]
    sampled = np.sort(t.loc[t["frame_idx"] % sample_every == 0, "frame_idx"].unique())
    status = possession.drop_duplicates("frame_idx").set_index("frame_idx")["ballStatus"].reindex(sampled)
    live = MatchState(state.meta, t)
    dead = (pd.Series(sampled, index=sampled).map(ball_still(live, still_s, still_m)).fillna(False).astype(bool)
            | pd.Series(sampled, index=sampled).map(play_stopped(live, sample_every, stopped_mps)).fillna(False).astype(bool))
    in_play = (status != "BALLOUT") & ~dead
    step = sample_every / fps
    shares = status[in_play].fillna("UNKNOWN").value_counts()
    total = float(in_play.sum() * step)
    return {
        "live_s": float(len(sampled) * step),
        "ball_in_play_s": total,
        "possession_share": {k: float(shares.get(k, 0) * step / total) if total else _NAN
                             for k in ("HOME", "AWAY", "CONTESTED", "UNKNOWN")},
    }


def phase_time_shares(state: MatchState, possession: pd.DataFrame, seqs: pd.DataFrame,
                      sample_every: int = 5, **kwargs) -> dict:
    """Time in each detected phase, per team, as a share of ball-in-play time.

    Every count of sequences needs this denominator: "22 low blocks" means nothing across
    matches until it is seconds in the phase over seconds of live play. Phases overlap (a
    team can be in a low block and the other in build-up), so shares do not sum to one.
    `midfield_loss` windows are the seconds after a loss, not a phase of play, and are
    reported by count only.
    """
    play = ball_in_play_seconds(state, possession, sample_every, **kwargs)
    denom = play["ball_in_play_s"]
    phases = []
    if len(seqs):
        timed = seqs[seqs["label"] != "midfield_loss"]
        for (label, team), g in timed.groupby(["label", "team"]):
            secs = float(g["duration_s"].sum())
            phases.append({"label": label, "team": team, "n_sequences": int(len(g)), "seconds": secs,
                           "share_of_ball_in_play": secs / denom if denom else _NAN})
        for (label, team), g in seqs[seqs["label"] == "midfield_loss"].groupby(["label", "team"]):
            phases.append({"label": label, "team": team, "n_sequences": int(len(g)), "seconds": _NAN,
                           "share_of_ball_in_play": _NAN})
    return {"match_id": state.meta.match_id, "fps": state.meta.fps, "sample_every": sample_every,
            **play, "phases": phases}


def write_sequence_outputs(state: MatchState, possession: pd.DataFrame, out_dir: Path | str,
                           sample_every: int = 5, **detect_kwargs) -> pd.DataFrame:
    """Write `sequences.csv`, `turnovers.csv` (all thirds) and `phase_shares.json` to `out_dir`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seqs, turnovers = detect_sequences(state, possession, sample_every=sample_every,
                                       with_turnovers=True, **detect_kwargs)
    seqs.to_csv(out / "sequences.csv", index=False)
    turnovers.to_csv(out / "turnovers.csv", index=False)
    shares = phase_time_shares(state, possession, seqs, sample_every=sample_every)
    (out / "phase_shares.json").write_text(json.dumps(shares, indent=2))
    return seqs
