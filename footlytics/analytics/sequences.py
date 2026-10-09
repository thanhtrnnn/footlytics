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

The state must already be normalised (`tactics.normalise_direction`): home
attacks +x in every period.
"""

from __future__ import annotations

import pandas as pd

from ..state.schema import MatchState, Team
from .tactics import attacking_sign, press_distance, team_block

_OPPONENT_STATUS = {Team.HOME.value: "AWAY", Team.AWAY.value: "HOME"}
COLUMNS = ["label", "team", "period", "start_s", "end_s", "start_frame", "end_frame",
           "duration_s", "def_line_x", "compactness_m", "n_within_10m"]


def detect_sequences(state: MatchState, possession: pd.DataFrame, sample_every: int = 5,
                     low_block_line: float = -22.5, low_block_compact_m: float = 14.0,
                     low_block_min_s: float = 8.0,
                     high_press_line: float = 7.5, high_press_n: int = 3,
                     high_press_min_s: float = 3.0) -> pd.DataFrame:
    """One row per sequence: label, team, period, start/end, and mean evidence over it."""
    fps = state.meta.fps
    blk = team_block(state, sample_every=sample_every)
    if blk.empty:
        return pd.DataFrame(columns=COLUMNS)

    status = possession.drop_duplicates("frame_idx").set_index("frame_idx")["ballStatus"]
    blk["out_of_possession"] = blk["frame_idx"].map(status) == blk["team"].map(_OPPONENT_STATUS)

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

    oop = blk["out_of_possession"]
    rules = {
        "low_block": (oop & (blk["def_line_x"] <= low_block_line)
                      & (blk["compactness_m"] <= low_block_compact_m), low_block_min_s),
        "high_press": (oop & (blk["def_line_x"] >= high_press_line)
                       & (blk["ball_x_own"] >= opp_third)
                       & (blk["n_within_10m"] >= high_press_n), high_press_min_s),
    }
    rows = []
    for label, (mask, min_s) in rules.items():
        rows += _runs(blk[mask], label, min_s, sample_every, fps)
    out = pd.DataFrame(rows, columns=COLUMNS)
    return out.sort_values(["start_s", "label"]).reset_index(drop=True)


def _runs(hits: pd.DataFrame, label: str, min_s: float, step: int, fps: float) -> list[dict]:
    """Group consecutive samples per team and period into sequences of at least `min_s`.

    One missing sample is tolerated, so a single noisy frame does not split a block in two.
    """
    out = []
    for (team, period), g in hits.sort_values("frame_idx").groupby(["team", "period"]):
        run_id = (g["frame_idx"].diff() > 2 * step).cumsum()
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
            })
    return out
