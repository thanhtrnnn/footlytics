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


# ------------------------------------------------------------------ direction

def infer_attacking_direction(state: MatchState) -> dict[tuple[int, str], int]:
    """(period, team) -> +1 if that team attacks +x, else -1.

    Decided by where each team's goalkeeper stands: a keeper is at the end his
    team defends, so the team attacks the other way. Using outfield mean position
    instead would fail during sustained pressure, when a whole team can sit in
    one half for minutes.
    """
    out: dict[tuple[int, str], int] = {}
    p = state.players
    for (period, team), g in p.groupby(["period", "team"], observed=True):
        if team not in (Team.HOME.value, Team.AWAY.value):
            continue
        mean_x = g.groupby("track_id", observed=True)["x"].mean()
        if mean_x.empty:
            continue
        # The keeper is the extreme; whichever end he sits at is the end defended.
        gk_x = mean_x.loc[mean_x.abs().idxmax()]
        out[(int(period), str(team))] = -1 if gk_x > 0 else 1
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
    """team -> track_id of its goalkeeper (most extreme mean x)."""
    out = {}
    for team, g in state.players.groupby("team", observed=True):
        if team not in (Team.HOME.value, Team.AWAY.value):
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
               sample_every: int = 25) -> pd.DataFrame:
    """Per-frame shape of each team: line height, width, depth, compactness.

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
        if len(outfield) < 6:
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

FORMATION_TEMPLATES = {
    "4-4-2": (4, 4, 2), "4-3-3": (4, 3, 3), "4-2-3-1": (4, 5, 1),
    "3-5-2": (3, 5, 2), "5-3-2": (5, 3, 2), "4-5-1": (4, 5, 1),
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
                   sample_every: int = 25) -> pd.DataFrame:
    """How close the out-of-possession team gets to the ball, over time.

    A direct read on pressing that needs no event labels: when the opposition
    has the ball, the distance from the ball to the nearest defender is what
    "pressure" physically means.
    """
    # A handful of matchTime values repeat, so frame_idx is not perfectly
    # unique; without deduplicating, every lookup returns a Series instead of a
    # value and comparisons raise deep inside the loop.
    ball = (state.ball.drop_duplicates("frame_idx")
                      .set_index("frame_idx")[["x", "y"]])
    poss = (possession.drop_duplicates("frame_idx")
                      .set_index("frame_idx")["ballStatus"])
    p = state.players
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


# --------------------------------------------------------------- transitions

def transitions(state: MatchState, possession: pd.DataFrame,
                min_hold_s: float = 1.5, window_s: float = 5.0) -> pd.DataFrame:
    """Turnovers, and what each team did in the seconds afterwards.

    A turnover only counts once the new team holds the ball for `min_hold_s`;
    without that, every deflection and contested touch registers as a transition
    and the count balloons into noise.
    """
    fps = state.meta.fps
    poss = possession.sort_values("frame_idx").reset_index(drop=True)
    poss = poss[poss["ballStatus"].isin(["HOME", "AWAY"])].copy()
    if poss.empty:
        return pd.DataFrame()

    poss["run"] = (poss["ballStatus"] != poss["ballStatus"].shift()).cumsum()
    runs = poss.groupby("run").agg(
        team=("ballStatus", "first"), start=("frame_idx", "min"),
        end=("frame_idx", "max"), period=("period", "first"),
        t0=("timestamp", "min")).reset_index(drop=True)
    runs["len_s"] = (runs["end"] - runs["start"]) / fps
    runs = runs[runs["len_s"] >= min_hold_s].reset_index(drop=True)

    ball = (state.ball.drop_duplicates("frame_idx")
                      .set_index("frame_idx")[["x", "y"]])
    rows = []
    for i in range(1, len(runs)):
        prev, cur = runs.iloc[i - 1], runs.iloc[i]
        if prev["team"] == cur["team"]:
            continue
        f0 = int(cur["start"]); f1 = int(f0 + window_s * fps)
        if f0 not in ball.index:
            continue
        b0 = ball.loc[f0]
        later = ball[(ball.index > f0) & (ball.index <= f1)]
        gained = np.nan
        if len(later):
            b1 = later.iloc[-1]
            # Progress towards the goal the winning team attacks.
            sign = attacking_sign(Team.HOME.value if cur["team"] == "HOME"
                                  else Team.AWAY.value)
            gained = float((b1["x"] - b0["x"]) * sign)
        rows.append({
            "period": int(cur["period"]), "timestamp": float(cur["t0"]),
            "frame_idx": f0,
            "won_by": Team.HOME.value if cur["team"] == "HOME" else Team.AWAY.value,
            # In the winning team's own attacking frame: negative = own half.
            "won_at_x": float(b0["x"] * (1 if cur["team"] == "HOME" else -1)),
            "won_at_y": float(b0["y"]),
            "hold_s": float(cur["len_s"]),
            f"gain_{int(window_s)}s_m": gained,
        })
    return pd.DataFrame(rows)
