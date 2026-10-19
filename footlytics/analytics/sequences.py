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

`build_up`
    In possession with the ball in the team's own third, for `build_up_min_s`;
    `n_pressers` is how many opponents stood in that third (the median over the
    sequence), so an analyst can ask for build-up against a three-man press.
    Dead-ball samples are skipped: the ball has not moved for `still_s` (a goal
    kick being placed), or the players have stopped, their median speed over the
    last second under `stopped_mps` (a foul, an injury). Play is not on then.

`midfield_loss`
    A turnover won in the middle third (`tactics.transitions`, so both teams held
    the ball for its minimum hold), as a `loss_window_s` sequence of the team
    that lost it, with what that team did next: `ball_dist_m` (its second-nearest
    player to the ball, mean from one second after the loss to `counter_press_s`;
    the nearest is usually the player who just lost it) and `retreat_mps`
    (how fast its centroid moved back, negative = towards its own goal).
    `response` is `counter_press` when `ball_dist_m` is at most
    `counter_press_m`, else `retreat` when `retreat_mps` is at most
    `retreat_mps`, else `hold`.

The state must already be normalised (`tactics.normalise_direction`): home
attacks +x in every period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..state.schema import MatchState, Team
from .tactics import attacking_sign, press_distance, team_block, transitions

_OPPONENT_STATUS = {Team.HOME.value: "AWAY", Team.AWAY.value: "HOME"}
_OWN_STATUS = {Team.HOME.value: "HOME", Team.AWAY.value: "AWAY"}
_OTHER = {Team.HOME.value: Team.AWAY.value, Team.AWAY.value: Team.HOME.value}
COLUMNS = ["label", "team", "period", "start_s", "end_s", "start_frame", "end_frame",
           "duration_s", "def_line_x", "compactness_m", "n_within_10m",
           "flank", "n_attackers", "n_defenders", "in_possession_share",
           "n_pressers", "ball_dist_m", "retreat_mps", "response"]


def detect_sequences(state: MatchState, possession: pd.DataFrame, sample_every: int = 5,
                     low_block_line: float = -22.5, low_block_compact_m: float = 14.0,
                     low_block_min_s: float = 8.0,
                     high_press_line: float = 7.5, high_press_n: int = 3,
                     high_press_min_s: float = 3.0,
                     flank_y: float | None = None, overload_n: int = 4,
                     overload_max_defenders: int = 3, overload_min_visible: int = 7,
                     overload_min_s: float = 3.0,
                     build_up_min_s: float = 4.0, still_s: float = 2.0, still_m: float = 1.0,
                     stopped_mps: float = 1.0,
                     loss_window_s: float = 6.0, counter_press_s: float = 3.0,
                     counter_press_m: float = 6.0, retreat_mps: float = -1.0) -> pd.DataFrame:
    """One row per sequence: label, team, period, start/end, and mean evidence over it."""
    fps = state.meta.fps
    blk = team_block(state, sample_every=sample_every)
    if blk.empty:
        return pd.DataFrame(columns=COLUMNS)

    status = possession.drop_duplicates("frame_idx").set_index("frame_idx")["ballStatus"]
    blk["out_of_possession"] = blk["frame_idx"].map(status) == blk["team"].map(_OPPONENT_STATUS)
    blk["in_possession"] = blk["frame_idx"].map(status) == blk["team"].map(_OWN_STATUS)

    pressure = press_distance(state, possession, sample_every=sample_every)
    if not pressure.empty:
        pressure = pressure.rename(columns={"pressing_team": "team"})
        blk = blk.merge(pressure[["frame_idx", "team", "n_within_10m"]],
                        on=["frame_idx", "team"], how="left")
    else:
        blk["n_within_10m"] = pd.NA
    blk["n_within_10m"] = blk["n_within_10m"].fillna(0).astype(int)

    ball_x = state.ball.drop_duplicates("frame_idx").set_index("frame_idx")["x"]
    blk["ball_x_own"] = blk["frame_idx"].map(ball_x) * blk["team"].map(attacking_sign)
    opp_third = state.meta.pitch_length / 2 - state.meta.pitch_length / 3
    blk["dead_ball"] = (blk["frame_idx"].map(ball_still(state, still_s, still_m)).fillna(False)
                        | blk["frame_idx"].map(play_stopped(state, sample_every, stopped_mps))
                                          .fillna(False))
    blk = blk.merge(own_third_pressers(state, sample_every, opp_third),
                    on=["frame_idx", "team"], how="left")
    blk["n_pressers"] = blk["n_pressers"].fillna(0).astype(int)

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
        "build_up": (blk["in_possession"] & (blk["ball_x_own"] <= -opp_third)
                     & ~blk["dead_ball"], build_up_min_s),
    }
    rows = []
    for label, (mask, min_s) in rules.items():
        rows += _runs(blk[mask], label, min_s, sample_every, fps)
    rows += _midfield_losses(state, possession, blk, pressure, opp_third, loss_window_s,
                             counter_press_s, counter_press_m, retreat_mps)
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
    n = max(2, int(round(still_s * state.meta.fps)))
    roll = b.rolling(n, min_periods=n // 2 + 1)
    spread = (roll.max() - roll.min()).pow(2).sum(axis=1, min_count=2).pow(0.5)
    return (spread <= still_m) & roll.count()["x"].ge(n // 2 + 1)


def play_stopped(state: MatchState, sample_every: int = 5, stopped_mps: float = 1.0) -> pd.Series:
    """Sampled frame_idx -> True where the players' median speed over the last second is under
    `stopped_mps`: everyone walking or standing, as after a whistle.

    In play somebody is always running, so the median player stays above a
    jog; at a foul or an injury the whole pitch slows to a walk.
    """
    p = state.players
    p = p[p["team"].isin([Team.HOME.value, Team.AWAY.value])][["frame_idx", "track_id", "x", "y"]]
    lag = int(round(state.meta.fps))
    now = p[p["frame_idx"] % sample_every == 0]
    before = p.assign(frame_idx=p["frame_idx"] + lag)
    j = now.merge(before, on=["frame_idx", "track_id"], suffixes=("", "_1s"))
    if j.empty:
        return pd.Series(dtype=bool)
    speed = np.hypot(j["x"] - j["x_1s"], j["y"] - j["y_1s"]) * state.meta.fps / lag
    return speed.groupby(j["frame_idx"]).median() < stopped_mps


def own_third_pressers(state: MatchState, sample_every: int, third_x: float) -> pd.DataFrame:
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
                                 "n_pressers": n.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def _midfield_losses(state: MatchState, possession: pd.DataFrame, blk: pd.DataFrame,
                     pressure: pd.DataFrame, third_x: float, window_s: float,
                     counter_press_s: float, counter_press_m: float,
                     retreat_mps: float) -> list[dict]:
    """One sequence per turnover in the middle third, for the team that lost the ball."""
    fps = state.meta.fps
    turnovers = transitions(state, possession)
    if turnovers.empty:
        return []
    last_frame = int(state.tracks["frame_idx"].max())
    out = []
    for t in turnovers.itertuples():
        if abs(t.won_at_x) > third_x:
            continue
        team = _OTHER[t.won_by]
        f0 = int(t.frame_idx)
        f1 = min(last_frame, f0 + int(round(window_s * fps)) - 1)
        if f1 - f0 + 1 < counter_press_s * fps:
            continue  # cut off before the response can be judged
        b = blk[(blk["team"] == team) & blk["frame_idx"].between(f0, f1)]
        pr = (pressure[(pressure["team"] == team) & pressure["frame_idx"].between(f0, f1)]
              if not pressure.empty else pressure)
        early = (pr[pr["frame_idx"].between(f0 + fps, f0 + counter_press_s * fps - 1)]
                 if len(pr) else pr)
        ball_dist = float(early["second_nearest_m"].mean()) if len(early) else float("nan")
        retreat = (float(np.polyfit(b["timestamp"], b["centroid_x"], 1)[0])
                   if len(b) >= 2 else float("nan"))
        if ball_dist <= counter_press_m:
            response = "counter_press"
        elif retreat <= retreat_mps:
            response = "retreat"
        else:
            response = "hold"
        out.append({
            "label": "midfield_loss", "team": team, "period": int(t.period),
            "start_s": float(t.timestamp), "end_s": float(t.timestamp + (f1 - f0 + 1) / fps),
            "start_frame": f0, "end_frame": f1, "duration_s": float((f1 - f0 + 1) / fps),
            "def_line_x": float(b["def_line_x"].mean()) if len(b) else float("nan"),
            "compactness_m": float(b["compactness_m"].mean()) if len(b) else float("nan"),
            "n_within_10m": int(pr["n_within_10m"].max()) if len(pr) else 0,
            "flank": "", "n_attackers": 0, "n_defenders": 0, "in_possession_share": 0.0,
            "n_pressers": 0, "ball_dist_m": ball_dist, "retreat_mps": retreat,
            "response": response,
        })
    return out


def _runs(hits: pd.DataFrame, label: str, min_s: float, step: int, fps: float) -> list[dict]:
    """Group consecutive samples per team and period into sequences of at least `min_s`.

    One missing sample is tolerated, so a single noisy frame does not split a block in two.
    A flank overload is also split by flank: a switch of play is two sequences, not one.
    """
    out = []
    keys = ["team", "period"] + (["flank"] if label == "flank_overload" else [])
    for key, g in hits.sort_values("frame_idx").groupby(keys):
        team, period = key[0], key[1]
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
                "flank": key[2] if len(key) > 2 else "",
                "n_attackers": int(r["n_attackers"].max()),
                "n_defenders": int(r["n_defenders"].max()),
                "in_possession_share": float(r["in_possession"].mean()),
                "n_pressers": int(round(r["n_pressers"].median())),
                "ball_dist_m": float("nan"), "retreat_mps": float("nan"), "response": "",
            })
    return out
