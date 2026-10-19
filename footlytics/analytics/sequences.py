"""Rule-based tactical sequences: the "Low block -- 22 sequences" an analyst searches for.

Each sequence is a stretch where one team's shape meets a label's conditions for
long enough, returned with the evidence that put it there so an analyst can mark
it right or wrong (docs/tactical-query-vision.md section 2). The thresholds are
first guesses to be tuned per club against those marks, not settled definitions.

`low_block`
    Out of possession, the four deepest outfielders at or behind `low_block_line`
    in the team's own attacking frame (-22.5 m: about the edge of their own
    third), compact, for `low_block_min_s`. Needs positions and possession only.

`high_press`
    Out of possession, the ball in the opponent's defensive third, the back line
    pushed up to `high_press_line`, and `high_press_n` players within 10 m of the
    ball, for `high_press_min_s`. Needs the ball, so it is only as good as ball
    coverage.

`flank_overload`
    At least `overload_n` of a team's players in one flank channel of the
    opponent's half (wider than `flank_y`, a sixth of the pitch width from the
    centre line by default: the wing plus the half-space) against at most
    `overload_max_defenders` opponents there, for `overload_min_s`. Positions
    only, no ball or possession, so it holds up while ball recall is low; the
    share of samples the team had the ball travels with it as evidence. A
    broadcast camera shows only part of the pitch, and a defender off screen
    turns a fair fight into a false overload, so samples where either team has
    fewer than `overload_min_visible` players in view are skipped. "Right" is
    the attacker's right, assuming +y is the touchline nearer the main camera.

Only live wide-camera frames count (`MatchState.tracks["live_wide"]`, see `perception.shots`): replays,
close-ups and graphics are dropped, and a sequence never spans a cut (`shot` id change), however short.

The state must already be normalised (`tactics.normalise_direction`): home
attacks +x in every period.
"""

from __future__ import annotations

import pandas as pd

from ..state.schema import MatchState, Team
from .tactics import attacking_sign, press_distance, team_block

_OPPONENT_STATUS = {Team.HOME.value: "AWAY", Team.AWAY.value: "HOME"}
_OWN_STATUS = {Team.HOME.value: "HOME", Team.AWAY.value: "AWAY"}
_OTHER = {Team.HOME.value: Team.AWAY.value, Team.AWAY.value: Team.HOME.value}
COLUMNS = ["label", "team", "period", "start_s", "end_s", "start_frame", "end_frame",
           "duration_s", "def_line_x", "compactness_m", "n_within_10m",
           "flank", "n_attackers", "n_defenders", "in_possession_share"]


def detect_sequences(state: MatchState, possession: pd.DataFrame, sample_every: int = 5,
                     low_block_line: float = -22.5, low_block_compact_m: float = 14.0,
                     low_block_min_s: float = 8.0,
                     high_press_line: float = 7.5, high_press_n: int = 3,
                     high_press_min_s: float = 3.0,
                     flank_y: float | None = None, overload_n: int = 4,
                     overload_max_defenders: int = 3, overload_min_visible: int = 7,
                     overload_min_s: float = 3.0) -> pd.DataFrame:
    """One row per sequence: label, team, period, start/end, and mean evidence over it."""
    fps = state.meta.fps
    shot_of = state.tracks.drop_duplicates("frame_idx").set_index("frame_idx")["shot"]
    if not state.tracks["live_wide"].all():
        state = MatchState(state.meta, state.tracks[state.tracks["live_wide"]])
    blk = team_block(state, sample_every=sample_every)
    if blk.empty:
        return pd.DataFrame(columns=COLUMNS)
    blk["shot"] = blk["frame_idx"].map(shot_of).astype(int)

    status = possession.drop_duplicates("frame_idx").set_index("frame_idx")["ballStatus"]
    blk["out_of_possession"] = blk["frame_idx"].map(status) == blk["team"].map(_OPPONENT_STATUS)
    blk["in_possession"] = blk["frame_idx"].map(status) == blk["team"].map(_OWN_STATUS)

    pressure = press_distance(state, possession, sample_every=sample_every)
    if not pressure.empty:
        pressure = pressure[["frame_idx", "pressing_team", "n_within_10m"]].rename(
            columns={"pressing_team": "team"})
        blk = blk.merge(pressure, on=["frame_idx", "team"], how="left")
    else:
        blk["n_within_10m"] = pd.NA
    blk["n_within_10m"] = blk["n_within_10m"].fillna(0).astype(int)

    ball_x = state.ball.drop_duplicates("frame_idx").set_index("frame_idx")["x"]
    blk["ball_x_own"] = blk["frame_idx"].map(ball_x) * blk["team"].map(attacking_sign)
    opp_third = state.meta.pitch_length / 2 - state.meta.pitch_length / 3

    flanks = flank_counts(state, sample_every=sample_every,
                          flank_y=state.meta.pitch_width / 6 if flank_y is None else flank_y)
    blk = blk.merge(flanks, on=["frame_idx", "team"], how="left")
    blk["flank"] = blk["flank"].fillna("")
    for c in ("n_attackers", "n_defenders", "n_visible", "n_visible_opp"):
        blk[c] = blk[c].fillna(0).astype(int)

    oop = blk["out_of_possession"]
    rules = {
        "low_block": (oop & (blk["def_line_x"] <= low_block_line)
                      & (blk["compactness_m"] <= low_block_compact_m), low_block_min_s),
        "high_press": (oop & (blk["def_line_x"] >= high_press_line)
                       & (blk["ball_x_own"] >= opp_third)
                       & (blk["n_within_10m"] >= high_press_n), high_press_min_s),
        "flank_overload": ((blk["n_attackers"] >= overload_n)
                           & (blk["n_defenders"] <= overload_max_defenders)
                           & (blk["n_visible"] >= overload_min_visible)
                           & (blk["n_visible_opp"] >= overload_min_visible), overload_min_s),
    }
    rows = []
    for label, (mask, min_s) in rules.items():
        rows += _runs(blk[mask], label, min_s, sample_every, fps)
    out = pd.DataFrame(rows, columns=COLUMNS)
    return out.sort_values(["start_s", "label"]).reset_index(drop=True)


def flank_counts(state: MatchState, sample_every: int = 5, flank_y: float = 34.0 / 3) -> pd.DataFrame:
    """Per sampled frame and team: its busiest flank in the opponent's half and who is there.

    Both counts are taken in the team's own attacking frame (x and y times
    `attacking_sign`, the 180-degree turn `normalise_direction` uses), so
    "right" is the attacker's right for both teams. Of the two flanks, the one
    with the larger attackers-minus-defenders margin is kept. Visible counts are
    players on the pitch in that frame.
    """
    p = state.players
    half_l, half_w = state.meta.pitch_length / 2, state.meta.pitch_width / 2
    # On the pitch only: substitutes waiting on the touchline are tracked too.
    p = p[(p["frame_idx"] % sample_every == 0) & p["team"].isin([Team.HOME.value, Team.AWAY.value])
          & (p["x"].abs() <= half_l) & (p["y"].abs() <= half_w)]
    rows = []
    for frame, g in p.groupby("frame_idx", observed=True):
        visible = g["team"].value_counts()
        for team in (Team.HOME.value, Team.AWAY.value):
            sign = attacking_sign(team)
            x, y = g["x"].to_numpy() * sign, g["y"].to_numpy() * sign
            own = (g["team"] == team).to_numpy()
            best = None
            for side, on_flank in (("right", y >= flank_y), ("left", y <= -flank_y)):
                zone = on_flank & (x >= 0)
                n_att, n_def = int((zone & own).sum()), int((zone & ~own).sum())
                if best is None or n_att - n_def > best[1] - best[2]:
                    best = (side, n_att, n_def)
            rows.append({"frame_idx": int(frame), "team": team, "flank": best[0],
                         "n_attackers": best[1], "n_defenders": best[2],
                         "n_visible": int(visible.get(team, 0)),
                         "n_visible_opp": int(visible.get(_OTHER[team], 0))})
    return pd.DataFrame(rows, columns=["frame_idx", "team", "flank", "n_attackers", "n_defenders",
                                       "n_visible", "n_visible_opp"])


def _runs(hits: pd.DataFrame, label: str, min_s: float, step: int, fps: float) -> list[dict]:
    """Group consecutive samples per team and period into sequences of at least `min_s`.

    One missing sample is tolerated, so a single noisy frame does not split a block in two.
    A flank overload is also split by flank: a switch of play is two sequences, not one.
    """
    out = []
    keys = ["team", "period"] + (["flank"] if label == "flank_overload" else [])
    for key, g in hits.sort_values("frame_idx").groupby(keys):
        team, period = key[0], key[1]
        run_id = ((g["frame_idx"].diff() > 2 * step) | (g["shot"].diff() != 0)).cumsum()
        for _, r in g.groupby(run_id):
            duration = (r["frame_idx"].iloc[-1] - r["frame_idx"].iloc[0] + step) / fps
            if duration < min_s:
                continue
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
                "n_attackers": int(r["n_attackers"].max()),
                "n_defenders": int(r["n_defenders"].max()),
                "in_possession_share": float(r["in_possession"].mean()),
            })
    return out
