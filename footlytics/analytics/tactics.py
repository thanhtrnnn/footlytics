"""Tactical analysis over Match State: shape, block, pressing, transitions.

Pure functions of the state table. Nothing here knows whether the positions came
from our own perception, a commercial feed, or a dataset's ground truth -- which
is the point of the Match State contract, and why this work could start while
the detector is still at 74% recall.

Two conventions everything depends on:

`attacking direction`
    Teams change ends. Every metric that mentions "forward", "high" or "deep" is
    meaningless until direction is normalised, and the failure is silent -- a
    second half simply reports the mirror image of the truth. `normalise_direction`
    flips coordinates so both teams always attack +x.

`goalkeeper`
    The source has no role labels, so the keeper is identified as the outfield
    outlier: the player whose mean position is furthest from their own team's
    centroid, towards their own goal. Keepers must be excluded from shape and
    line-height metrics or they drag every defensive line 20 m too deep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..state.schema import MatchState, Role, Team

_OTHER_TEAM = {Team.HOME.value: Team.AWAY.value, Team.AWAY.value: Team.HOME.value}


# ------------------------------------------------------------------ direction

def infer_attacking_direction(state: MatchState) -> dict[tuple[int, str], int]:
    """(period, team) -> +1 if that team attacks +x, else -1.

    Decided by where each team's goalkeeper (the labelled one, else the extreme
    player) stands: a keeper is at the end his
    team defends, so the team attacks the other way. Using outfield mean position
    instead would fail during sustained pressure, when a whole team can sit in
    one half for minutes.

    The two teams always attack opposite ways. A broadcast view of one end shows
    one keeper; the other team's extreme player then stands at the same end, and
    judged alone would give both teams the same direction. So a labelled keeper
    decides for both teams, and without one the more extreme player does.
    """
    out: dict[tuple[int, str], int] = {}
    p = state.players
    p = p[p["team"].isin([Team.HOME.value, Team.AWAY.value])]
    for period, gp in p.groupby("period", observed=True):
        keeper_x, extreme_x = {}, {}
        for team, g in gp.groupby("team", observed=True):
            keeper = g[g["role"] == Role.GOALKEEPER.value]
            if not keeper.empty:
                keeper_x[str(team)] = float(keeper["x"].mean())
            mean_x = g.groupby("track_id", observed=True)["x"].mean()
            if not mean_x.empty:
                # Whichever end the extreme player sits at is the end defended.
                extreme_x[str(team)] = float(mean_x.loc[mean_x.abs().idxmax()])
        evidence = keeper_x or extreme_x
        if not evidence:
            continue
        if len(keeper_x) == 2:
            # Two keepers: each team by its own, as they are seen.
            for team, x in keeper_x.items():
                out[(int(period), team)] = -1 if x > 0 else 1
            continue
        team = max(evidence, key=lambda t: abs(evidence[t]))
        sign = -1 if evidence[team] > 0 else 1
        out[(int(period), team)] = sign
        other = _OTHER_TEAM[team]
        if other in extreme_x:
            out[(int(period), other)] = -sign
    return out


def normalise_direction(state: MatchState) -> MatchState:
    """Flip each period so the HOME team always attacks +x.

    One canonical frame for the whole state, not one per team. An earlier version
    flipped each team by its own attacking direction, which reads sensibly until
    you use it: the two teams end up in opposite coordinate frames, and the ball
    -- which belongs to neither -- is never flipped at all. Every metric mixing
    them silently breaks. It showed up as one team pressing at 3.7 m and the
    other at 16.6 m, and as teams losing ground after winning the ball.

    Per-team orientation is applied per metric instead, via `attacking_sign`.
    """
    d = infer_attacking_direction(state)
    t = state.tracks.copy()
    for period in t["period"].unique():
        sign = d.get((int(period), Team.HOME.value))
        if sign is None:
            sign = -d.get((int(period), Team.AWAY.value), 1)
        m = t["period"] == period
        t.loc[m, "x"] = t.loc[m, "x"] * sign
        t.loc[m, "y"] = t.loc[m, "y"] * sign
    return MatchState(state.meta, t)


def attacking_sign(team: str) -> int:
    """+1 if the team attacks +x in the normalised frame, else -1.

    After `normalise_direction`, home attacks +x and away attacks -x in every
    period. Multiply x by this to put a team in its own attacking frame, so that
    "forward" and "high" mean the same thing for both.
    """
    return 1 if str(team) == Team.HOME.value else -1


def goalkeepers(state: MatchState) -> dict[str, int]:
    """team -> track_id of its goalkeeper: the labelled one, else the most extreme mean x."""
    out = {}
    for team, g in state.players.groupby("team", observed=True):
        if team not in (Team.HOME.value, Team.AWAY.value):
            continue
        labelled = g[g["role"] == Role.GOALKEEPER.value]
        if not labelled.empty:
            out[str(team)] = int(labelled.groupby("track_id", observed=True).size().idxmax())
            continue
        mean_x = g.groupby("track_id", observed=True)["x"].mean()
        counts = g.groupby("track_id", observed=True).size()
        # Only consider players who were on the pitch long enough to judge.
        mean_x = mean_x[counts > counts.max() * 0.3]
        if len(mean_x):
            out[str(team)] = int(mean_x.abs().idxmax())
    return out


# --------------------------------------------------------------------- block

def team_block(state: MatchState, exclude_gk: bool = True,
               sample_every: int = 25, min_outfield: int = 6) -> pd.DataFrame:
    """Per-frame shape of each team: line height, width, depth, compactness.

    Every figure is over the outfielders the detector saw in that frame (`n_outfield`),
    not the whole team: a broadcast camera follows the ball, so the deepest visible
    player is often not the deepest player. A line read from a subset is an upper bound
    on the real line.

    `sample_every` frames, because these move on the scale of seconds and a
    135-minute match is 200k frames.
    """
    gks = goalkeepers(state) if exclude_gk else {}
    p = state.players
    p = p[p["frame_idx"] % sample_every == 0]
    rows = []
    for (frame, period, team), g in p.groupby(["frame_idx", "period", "team"], observed=True):
        if team not in (Team.HOME.value, Team.AWAY.value):
            continue
        gk = gks.get(str(team))
        outfield = g[g["track_id"] != gk] if gk is not None else g
        if len(outfield) < min_outfield:
            continue
        sign = attacking_sign(team)
        # Own attacking frame, so "defensive line" means the same for both teams.
        xs = np.sort(outfield["x"].to_numpy() * sign)
        ys = outfield["y"].to_numpy()
        rows.append({
            "frame_idx": int(frame), "period": int(period), "team": str(team),
            "timestamp": float(g["timestamp"].iloc[0]),
            "n_outfield": len(outfield),
            # Defensive line: the four deepest outfielders, not the single deepest,
            # which is usually one player caught upfield or dropping off.
            "def_line_x": float(xs[:4].mean()),
            # The three deepest, the line height SkillCorner's block definitions use.
            "def_line3_x": float(xs[:3].mean()),
            "fwd_line_x": float(xs[-2:].mean()),
            "depth_m": float(xs[-1] - xs[0]),
            "width_m": float(ys.max() - ys.min()),
            "centroid_x": float(outfield["x"].mean() * sign),
            "centroid_y": float(outfield["y"].mean()),
            # Compactness: mean distance to the team centroid. Lower = tighter.
            "compactness_m": float(np.hypot(outfield["x"] - outfield["x"].mean(),
                                            outfield["y"] - outfield["y"].mean()).mean()),
            "attacking_sign": sign,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ formation

#: Depth-band triples (defenders, midfielders, forwards). Three bands cannot tell a 4-2-3-1
#: from a 4-5-1 or a 4-1-4-1, so the triple carries one name.
FORMATION_TEMPLATES = {
    "4-4-2": (4, 4, 2), "4-3-3": (4, 3, 3), "4-5-1": (4, 5, 1),
    "3-5-2": (3, 5, 2), "5-3-2": (5, 3, 2),
    "3-4-3": (3, 4, 3), "5-4-1": (5, 4, 1),
}


def formation_over_time(state: MatchState, window_s: float = 300.0,
                        sample_every: int = 25) -> pd.DataFrame:
    """Classify each team's shape in rolling windows.

    Averaging positions over a window rather than reading a single frame: a
    formation is a tendency, and any one frame catches players mid-transition.
    Five minutes is long enough to be stable and short enough to catch a change.

    The classification is deliberately crude -- outfielders are split into three
    bands by depth and matched to the nearest template. It answers "roughly what
    shape", not "which system", and should be read as a label on a distribution,
    not a fact.
    """
    gks = goalkeepers(state)
    p = state.players
    p = p[p["frame_idx"] % sample_every == 0]
    fps = state.meta.fps
    rows = []
    for (period, team), g in p.groupby(["period", "team"], observed=True):
        if team not in (Team.HOME.value, Team.AWAY.value):
            continue
        gk = gks.get(str(team))
        g = g[g["track_id"] != gk] if gk is not None else g
        if g.empty:
            continue
        t0, t1 = g["timestamp"].min(), g["timestamp"].max()
        for start in np.arange(t0, t1, window_s):
            w = g[(g["timestamp"] >= start) & (g["timestamp"] < start + window_s)]
            if w.empty:
                continue
            # Mean position per player over the window, keeping the regulars.
            per = w.groupby("track_id", observed=True).agg(x=("x", "mean"), n=("x", "size"))
            per = per[per["n"] >= per["n"].max() * 0.5]
            if len(per) < 8:
                continue
            xs = np.sort(per["x"].to_numpy() * attacking_sign(team))
            # Split into three bands at the two largest gaps in depth.
            gaps = np.diff(xs)
            if len(gaps) < 2:
                continue
            cuts = np.sort(np.argsort(gaps)[-2:])
            bands = (cuts[0] + 1, cuts[1] - cuts[0], len(xs) - cuts[1] - 1)
            best = min(FORMATION_TEMPLATES.items(),
                       key=lambda kv: sum(abs(a - b) for a, b in zip(kv[1], bands)))
            rows.append({
                "period": int(period), "team": str(team),
                "start_s": float(start), "end_s": float(start + window_s),
                "n_players": len(per),
                "bands": "-".join(map(str, bands)),
                "formation": best[0],
                "template_distance": sum(abs(a - b) for a, b in zip(best[1], bands)),
                "def_line_x": float(xs[:4].mean()),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ pressing

#: Event labels that count as a defensive action for PPDA.
DEFENSIVE_ACTIONS = {"PLAYER SUCCESSFUL TACKLE", "BALL PLAYER BLOCK"}
#: Event labels that count as a pass attempt for PPDA.
PASS_ACTIONS = {"PASS", "HIGH PASS", "CROSS"}


def ppda(state: MatchState, events: pd.DataFrame, player_team: dict[int, str],
         press_zone: Optional[float] = None) -> pd.DataFrame:
    """Passes Allowed Per Defensive Action, per team per period. WHOLE PITCH.

    The standard pressing metric: how many passes the opposition completes before
    the pressing team makes a tackle or block. LOWER means more aggressive.

    The usual convention restricts it to the pressing team's attacking 60% of the
    pitch; otherwise a team defending its own box racks up defensive actions and
    looks like it is pressing high when it is doing the opposite. That restriction
    is NOT applied here: the 12-class event feed carries a time, not a pitch
    location, so the zone would have to be recovered from the ball in `state` at
    each event time. Passing `press_zone` raises rather than silently returning the
    unrestricted number under a restricted name.
    """
    if press_zone is not None:
        raise NotImplementedError(
            "press_zone needs event locations (or the ball position at each event); "
            "this feed has none, so ppda() is whole-pitch. See docstring.")
    ev = events.copy()
    ev["team_id"] = ev["player_id"].map(player_team)
    rows = []
    for period, g in ev.groupby("period"):
        for team in (Team.HOME.value, Team.AWAY.value):
            opp = Team.AWAY.value if team == Team.HOME.value else Team.HOME.value
            opp_passes = int(((g["team_id"] == opp) & g["label"].isin(PASS_ACTIONS)).sum())
            our_def = int(((g["team_id"] == team) & g["label"].isin(DEFENSIVE_ACTIONS)).sum())
            rows.append({
                "period": int(period), "team": team,
                "opp_passes": opp_passes, "def_actions": our_def,
                "ppda": (opp_passes / our_def) if our_def else np.nan,
            })
    return pd.DataFrame(rows)


def press_distance(state: MatchState, possession: pd.DataFrame,
                   sample_every: int = 25, exclude_gk: bool = True) -> pd.DataFrame:
    """How close the out-of-possession team gets to the ball, over time.

    A direct read on distance, not on pressure: it is undirected and ignores who is
    closing (`press_on_ball` is the pressure test). The pressing team's goalkeeper is
    left out; he stands near the ball whenever play is in his box without pressing it.
    """
    # A handful of matchTime values repeat, so frame_idx is not perfectly
    # unique; without deduplicating, every lookup returns a Series instead of a
    # value and comparisons raise deep inside the loop.
    ball = (state.ball.drop_duplicates("frame_idx")
                      .set_index("frame_idx")[["x", "y"]])
    poss = (possession.drop_duplicates("frame_idx")
                      .set_index("frame_idx")["ballStatus"])
    p = state.players
    if exclude_gk:
        p = p[~p["track_id"].isin(goalkeepers(state).values())]
    p = p[p["frame_idx"] % sample_every == 0]
    rows = []
    for frame, g in p.groupby("frame_idx", observed=True):
        status = poss.get(frame)
        if status not in ("HOME", "AWAY") or frame not in ball.index:
            continue
        b = ball.loc[frame]
        pressing_team = Team.AWAY.value if status == "HOME" else Team.HOME.value
        d = g[g["team"] == pressing_team]
        if d.empty:
            continue
        dist = np.sort(np.hypot(d["x"] - b["x"], d["y"] - b["y"]).to_numpy())
        rows.append({
            "frame_idx": int(frame),
            "timestamp": float(g["timestamp"].iloc[0]),
            "period": int(g["period"].iloc[0]),
            "in_possession": Team.HOME.value if status == "HOME" else Team.AWAY.value,
            "pressing_team": pressing_team,
            "nearest_m": float(dist[0]),
            "second_nearest_m": float(dist[1]) if len(dist) > 1 else np.nan,
            "n_within_5m": int((dist <= 5).sum()),
            "n_within_10m": int((dist <= 10).sum()),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------ ball pressure

PRESS_BACK_M = 3.0
PRESS_FRONT_M = 9.0
PRESS_CLOSING_MPS = 2.0
PRESS_LOOKBACK_S = 0.5


def press_on_ball(state: MatchState, sample_every: int = 5, exclude_gk: bool = True) -> pd.DataFrame:
    """Per sampled frame and team: how many of its outfielders are pressing the ball.

    "Pressing" is asked of the team as if the other team carried the ball. A player
    presses when he is within `PRESS_BACK_M` of the ball, or within `PRESS_FRONT_M` of it on
    the goal side of the ball (between the carrier and the goal he attacks) while closing on it
    at `PRESS_CLOSING_MPS` or more, measured over the last `PRESS_LOOKBACK_S`. Standing
    off, or arriving from behind, does not count. `n_closing` counts only the players
    closing at `PRESS_CLOSING_MPS` or more, in the same distance bands: a player who is
    simply standing next to the ball presses it but is not closing in on it. Columns:
    frame_idx, team, n_pressing, n_closing, nearest_m. Frames with no ball are absent.
    """
    fps = state.meta.fps
    lag = max(1, round(PRESS_LOOKBACK_S * fps))
    ball = state.ball.drop_duplicates("frame_idx").set_index("frame_idx")[["x", "y"]]
    p = state.players
    p = p[p["team"].isin([Team.HOME.value, Team.AWAY.value])]
    if exclude_gk:
        p = p[~p["track_id"].isin(goalkeepers(state).values())]
    before = p[["frame_idx", "track_id", "x", "y"]].assign(frame_idx=p["frame_idx"] + lag)
    now = p[p["frame_idx"] % sample_every == 0].join(ball, on="frame_idx", rsuffix="_ball", how="inner")
    now = now.merge(before, on=["frame_idx", "track_id"], how="left", suffixes=("", "_prev"))
    if now.empty:
        return pd.DataFrame(columns=["frame_idx", "team", "n_pressing", "n_closing", "nearest_m"])
    dx, dy = now["x_ball"] - now["x"], now["y_ball"] - now["y"]
    d = np.hypot(dx, dy)
    closing = ((now["x"] - now["x_prev"]) * dx + (now["y"] - now["y_prev"]) * dy) / d.where(d > 0) * fps / lag
    goal_x = (state.meta.pitch_length / 2) * now["team"].astype(str).map(_OTHER_TEAM).map(attacking_sign)
    ux, uy = goal_x - now["x_ball"], -now["y_ball"]
    on_goal_side = (-dx * ux - dy * uy) > 0
    closing_in = closing >= PRESS_CLOSING_MPS
    pressing = (d <= PRESS_BACK_M) | (on_goal_side & (d <= PRESS_FRONT_M) & closing_in)
    now = now.assign(d=d, pressing=pressing, closing=pressing & closing_in)
    out = now.groupby(["frame_idx", "team"], observed=True).agg(
        n_pressing=("pressing", "sum"), n_closing=("closing", "sum"), nearest_m=("d", "min")).reset_index()
    out["team"] = out["team"].astype(str)
    out[["n_pressing", "n_closing"]] = out[["n_pressing", "n_closing"]].astype(int)
    return out


def retreat_speed(state: MatchState, team: str, f0: int, f1: int, sample_every: int = 5,
                  min_players: int = 3, max_hole_s: float = 1.0,
                  keepers: set[int] | None = None) -> float:
    """How fast a team's block moved towards its own goal over frames f0..f1 (negative = back), in m/s.

    The slope of the centroid of the outfielders tracked in every sampled frame of the
    window. The centroid of whoever the camera shows moves when the camera pans, so a
    player who enters or leaves the view is left out. NaN when fewer than `min_players`
    stayed in view throughout, or the window has fewer than four samples or a hole longer
    than `max_hole_s`. `keepers` are the goalkeeper track ids to leave out (found from the
    state when not given).
    """
    fps = state.meta.fps
    t = state.tracks
    t = t[t["frame_idx"].between(f0, f1) & (t["frame_idx"] % sample_every == 0)]
    frames = np.sort(t["frame_idx"].unique())
    if len(frames) < 4 or np.diff(frames).max() > max_hole_s * fps:
        return float("nan")
    p = state.players
    p = p[(p["team"] == team) & p["frame_idx"].isin(frames)]
    p = p[~p["track_id"].isin(set(goalkeepers(state).values()) if keepers is None else keepers)]
    n_seen = p.groupby("track_id", observed=True)["frame_idx"].nunique()
    stayed = n_seen[n_seen == len(frames)].index
    if len(stayed) < min_players:
        return float("nan")
    c = p[p["track_id"].isin(stayed)].groupby("frame_idx").agg(x=("x", "mean"), t=("timestamp", "first"))
    return float(np.polyfit(c["t"], c["x"] * attacking_sign(team), 1)[0])


# --------------------------------------------------------------- transitions

def third_zone(x_own: float, third_x: float) -> str:
    """Which third a point is in, from the attacking frame of the team asking (+x = towards its goal)."""
    if x_own <= -third_x:
        return "own_third"
    return "final_third" if x_own >= third_x else "middle_third"


def transitions(state: MatchState, possession: pd.DataFrame,
                min_hold_s: float = 1.5, window_s: float = 5.0,
                max_gap_s: float = 2.0) -> pd.DataFrame:
    """Turnovers in every third, with what the team that won the ball did with it.

    A turnover only counts once the new team holds the ball for `min_hold_s`;
    without that, every deflection and contested touch registers as a transition
    and the count balloons into noise. Contiguous runs are split across long
    tracking gaps (> `max_gap_s`), and a handoff across > `max_gap_s` or any
    intervening BALLOUT dead-ball restart is not counted as a live turnover. A
    contested spell shorter than `max_gap_s` between the two runs is a loose ball
    being won, which is a turnover. When the possession table carries contact evidence the
    winner must have it: a `gain` of "touch" or "unclaimed" on his first frame (see
    `infer_possession`), or else a `touch` by him between the loser's last control and
    `max_gap_s` after his own first. A ball nobody touched did not change hands.

    `zone` is the third where the ball was won, from the point of view of the team
    that lost it (`final_third` = it lost the ball attacking). `gain_5s_m` (named after
    `window_s`) is how far the winner moved the ball towards the goal it attacks within
    the window, the attacking-transition measure.
    """
    fps = state.meta.fps
    poss = possession.sort_values("frame_idx").reset_index(drop=True)
    if poss.empty:
        return pd.DataFrame()

    max_gap_f = max_gap_s * fps
    poss["run"] = ((poss["ballStatus"] != poss["ballStatus"].shift())
                   | (poss["period"] != poss["period"].shift())
                   | (poss["frame_idx"].diff() > max_gap_f)).cumsum()
    runs = poss.groupby("run").agg(
        team=("ballStatus", "first"), start=("frame_idx", "min"),
        end=("frame_idx", "max"), period=("period", "first"),
        t0=("timestamp", "min")).reset_index(drop=True)
    runs["len_s"] = (runs["end"] - runs["start"]) / fps
    runs = runs[runs["team"].isin(["HOME", "AWAY"]) & (runs["len_s"] >= min_hold_s)].reset_index(drop=True)

    touches = ({team: poss.loc[poss["touch"] == team, "frame_idx"].to_numpy() for team in ("HOME", "AWAY")}
               if "touch" in poss.columns else None)
    gain_at = poss.set_index("frame_idx")["gain"] if "gain" in poss.columns else None
    ball = (state.ball.drop_duplicates("frame_idx").set_index("frame_idx")[["x", "y"]].sort_index())
    third_x = state.meta.pitch_length / 2 - state.meta.pitch_length / 3
    rows = []
    for i in range(1, len(runs)):
        prev, cur = runs.iloc[i - 1], runs.iloc[i]
        if prev["team"] == cur["team"] or prev["period"] != cur["period"]:
            continue
        if (cur["start"] - prev["end"]) / fps > max_gap_s:
            continue
        between = poss[(poss["frame_idx"] > prev["end"]) & (poss["frame_idx"] < cur["start"])]
        if (between["ballStatus"] == "BALLOUT").any():
            continue
        if touches is not None:
            mine = touches[cur["team"]]
            lo, hi = np.searchsorted(mine, [prev["end"], cur["start"] + max_gap_f + 1])
            confirmed = gain_at is not None and gain_at.get(cur["start"], "") in ("touch", "unclaimed")
            if hi <= lo and not confirmed:
                continue
        f0 = int(cur["start"]); f1 = int(f0 + window_s * fps)
        nearest = ball.index.searchsorted(f0)
        at = [k for k in (nearest - 1, nearest) if 0 <= k < len(ball) and abs(ball.index[k] - f0) <= fps]
        if not at:
            continue
        b0 = ball.iloc[min(at, key=lambda k: abs(ball.index[k] - f0))]
        later = ball[(ball.index > f0) & (ball.index <= f1)]
        winner = Team.HOME.value if cur["team"] == "HOME" else Team.AWAY.value
        sign = attacking_sign(winner)
        gained = float((later.iloc[-1]["x"] - b0["x"]) * sign) if len(later) else np.nan
        rows.append({
            "period": int(cur["period"]), "timestamp": float(cur["t0"]),
            "frame_idx": f0,
            "won_by": winner,
            # In the winning team's own attacking frame: negative = own half.
            "won_at_x": float(b0["x"] * sign),
            "won_at_y": float(b0["y"]),
            "zone": third_zone(float(-b0["x"] * sign), third_x),
            "hold_s": float(cur["len_s"]),
            f"gain_{int(window_s)}s_m": gained,
        })
    return pd.DataFrame(rows)
