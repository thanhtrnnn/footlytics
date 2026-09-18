"""Full-match tracking from SoccerTrack v2's tracker XML.

Why this and not our own pipeline. Running our perception over 90 minutes takes
~37 hours per half on an M2 Pro, and at 74% recall with 347 identity switches per
250 frames the result would be useless for situation analysis anyway -- you
cannot ask "who marked whom" of data that scrambles identity 1.4 times a frame.

This loader gives a full match of trustworthy state today, which is exactly what
the Match State contract exists for: analytics never learns where the state came
from, so the perception work and the analytics work proceed independently and
meet without changes on either side.

The XML carries three things the GSR JSONs also have but at 5.4 GB rather than
318 MB: per-frame positions for all 22 players, per-frame possession
(`ballStatus` is HOME / AWAY / BALLOUT), and stable player ids that join to the
team sheet for shirt numbers and teams.

Coordinates are normalised to [0, 1] over the pitch and are converted here to the
centre-origin metres the rest of the codebase uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from ..state.schema import MatchMeta, MatchState, Role, Team, TeamInfo

PITCH_L, PITCH_W = 105.0, 76.0
FPS = 25.0


def load_team_sheet(nodes_csv: str | Path) -> pd.DataFrame:
    """player_id -> team, shirt number, name (deduplicated)."""
    d = pd.read_csv(nodes_csv)
    cols = ["player_id", "back_number", "team_id", "team_name",
            "player_name", "player_last_name"]
    sheet = d[[c for c in cols if c in d.columns]].drop_duplicates("player_id")
    return sheet.set_index("player_id")


def load_full_match(
    xml_path: str | Path,
    nodes_csv: str | Path,
    match_id: str = "SOCCERTRACK-FULL",
    pitch_length: float = PITCH_L,
    pitch_width: float = PITCH_W,
    max_frames: Optional[int] = None,
    verbose: bool = True,
) -> tuple[MatchState, pd.DataFrame]:
    """Stream-parse the tracker XML into a MatchState plus a possession table.

    Streamed with iterparse and cleared as it goes: the file is 318 MB of XML and
    building a DOM for it costs several GB.
    """
    sheet = load_team_sheet(nodes_csv)
    # Provisional mapping by first appearance in the team sheet -- arbitrary, and
    # corrected below against the possession labels. Getting this backwards is
    # not a cosmetic error: it inverts every possession-based metric while
    # leaving them all looking plausible.
    team_ids = list(dict.fromkeys(sheet["team_id"].dropna().tolist()))
    team_of = {tid: (Team.HOME.value if i == 0 else Team.AWAY.value)
               for i, tid in enumerate(team_ids)}
    names = {tid: sheet.loc[sheet.team_id == tid, "team_name"].iloc[0]
             for tid in team_ids if (sheet.team_id == tid).any()}

    rows: list[tuple] = []
    poss: list[tuple] = []
    # This match has THREE periods, not two: FIRST_HALF, SECOND_HALF and
    # EXTRA_FIRST_HALF, ~45 minutes each, 135 minutes in total.
    period_map = {"FIRST_HALF": 1, "SECOND_HALF": 2,
                  "EXTRA_FIRST_HALF": 3, "EXTRA_SECOND_HALF": 4}
    n_frames = 0
    skipped = 0
    BALL_ID = -1

    # iterparse fires "end" for EVERY element, children included. Clearing a
    # <player> when its own end event arrives wipes its attributes before the
    # enclosing <frame> is ever read -- which silently yields zero players and a
    # table of NaNs rather than an error. Only the frame is cleared, and only
    # after it has been consumed; that releases its children too.
    context = ET.iterparse(str(xml_path), events=("start", "end"))
    _, root = next(context)
    for event, el in context:
        if event != "end" or el.tag != "frame":
            continue
        period = period_map.get(el.get("eventPeriod", "FIRST_HALF"), 1)
        ms = float(el.get("matchTime", 0))
        # frameNumber restarts each period -- 66% of values are duplicated across
        # the three -- so using it as the index silently stacks every period on
        # top of the others and yields 66 players per frame. matchTime is
        # monotonic over the whole match, so the frame index comes from that.
        fno = int(round(ms / (1000.0 / FPS)))
        status = el.get("ballStatus", "UNKNOWN")
        poss.append((fno, period, ms / 1000.0, status))

        # <player> and <ball> share a shape. The ball carries playerId="ball",
        # and a handful of player records later in the file have no id at all --
        # skipped rather than guessed, and counted so the loss is visible.
        for p in list(el.findall("player")) + list(el.findall("ball")):
            raw_id = p.get("playerId")
            is_ball = (p.tag == "ball") or (raw_id == "ball")
            if is_ball:
                pid = BALL_ID
            else:
                try:
                    pid = int(raw_id)
                except (TypeError, ValueError):
                    skipped += 1
                    continue
            loc = p.get("loc", "[0,0]").strip("[]").split(",")
            try:
                nx, ny = float(loc[0]), float(loc[1])
            except ValueError:
                skipped += 1
                continue
            try:
                sp = float(p.get("speed", "nan"))
            except ValueError:
                sp = np.nan
            rows.append((
                fno, period, ms / 1000.0, pid,
                nx * pitch_length - pitch_length / 2,
                ny * pitch_width - pitch_width / 2,
                sp,
            ))
        el.clear()
        root.clear()          # frames accumulate under the root otherwise
        n_frames += 1
        if verbose and n_frames % 20000 == 0:
            print(f"  parsed {n_frames:,} frames, {len(rows):,} observations", flush=True)
        if max_frames and n_frames >= max_frames:
            break

    df = pd.DataFrame(rows, columns=["frame_idx", "period", "timestamp",
                                     "player_id", "x", "y", "speed_src"])
    possession = pd.DataFrame(poss, columns=["frame_idx", "period", "timestamp", "ballStatus"])

    is_ball = df["player_id"] == -1
    df["team"] = df["player_id"].map(
        lambda p: team_of.get(sheet["team_id"].get(p), Team.UNKNOWN.value))
    df.loc[is_ball, "team"] = Team.UNKNOWN.value
    df["jersey"] = df["player_id"].map(lambda p: sheet["back_number"].get(p, np.nan))
    df.loc[is_ball, "jersey"] = np.nan
    df["role"] = np.where(is_ball, Role.BALL.value, Role.PLAYER.value)
    df["track_id"] = df["player_id"]
    df["z"] = np.nan
    for c in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"):
        df[c] = np.nan
    df["det_conf"] = 1.0

    # Anchor home/away against the data rather than the CSV's ordering: when the
    # XML says HOME is in possession, the home team must be the one nearer the
    # ball. Checked on a sample of frames; if it disagrees, the labels are
    # swapped. Caught a real inversion here -- home players sat 2.1 m from the
    # ball on frames labelled AWAY, and 5.0 m on frames labelled HOME.
    swapped = _possession_says_swapped(df, possession)
    if swapped:
        if verbose:
            print("[fullmatch] team labels inverted vs possession -- swapping")
        flip = {Team.HOME.value: Team.AWAY.value, Team.AWAY.value: Team.HOME.value}
        df["team"] = df["team"].map(lambda t: flip.get(t, t))
        team_of = {k: flip.get(v, v) for k, v in team_of.items()}
        team_ids = team_ids[::-1]

    home_id, away_id = (team_ids + [None, None])[:2]
    meta = MatchMeta(
        match_id=match_id, fps=FPS,
        pitch_length=pitch_length, pitch_width=pitch_width,
        source_type="fixed_panoramic",
        competition="SoccerTrack v2 (university, Japan)",
        home=TeamInfo(str(names.get(home_id, "Home")), "HOME"),
        away=TeamInfo(str(names.get(away_id, "Away")), "AWAY"),
        provenance={"source": "soccertrack tracker_box_data.xml"},
    )
    state = MatchState(meta, df)
    if verbose:
        if skipped:
            print(f"[fullmatch] skipped {skipped:,} unusable records")
        print(f"[fullmatch] {state}")
        print(f"[fullmatch] {df.player_id.nunique()} players, "
              f"periods {sorted(possession.period.unique())}")
    return state, possession


def _possession_says_swapped(df: pd.DataFrame, possession: pd.DataFrame,
                             n_samples: int = 300) -> bool:
    """Does the team labelled home actually hold the ball on HOME frames?"""
    ball = df[df["player_id"] == -1].drop_duplicates("frame_idx").set_index("frame_idx")
    poss = possession.drop_duplicates("frame_idx").set_index("frame_idx")["ballStatus"]
    players = df[df["player_id"] != -1]
    frames = [f for f in poss.index[::max(len(poss) // n_samples, 1)]
              if f in ball.index][:n_samples]
    agree = disagree = 0
    for f in frames:
        st = poss.get(f)
        if st not in ("HOME", "AWAY"):
            continue
        g = players[players["frame_idx"] == f]
        b = ball.loc[f]
        near = {}
        for team in (Team.HOME.value, Team.AWAY.value):
            d = g[g["team"] == team]
            if len(d):
                near[team] = float(np.min(np.hypot(d["x"] - b["x"], d["y"] - b["y"])))
        if len(near) < 2:
            continue
        closer = min(near, key=near.get)
        expected = Team.HOME.value if st == "HOME" else Team.AWAY.value
        agree += (closer == expected)
        disagree += (closer != expected)
    return disagree > agree


def load_events(events_json: str | Path, nodes_csv: Optional[str | Path] = None,
                first_half_end_ms: float = 2_695_000.0) -> pd.DataFrame:
    """Ball-action events: 12 classes, with acting team and player.

    `first_half_end_ms` comes from the match's padding_info.csv.
    """
    import json

    d = json.loads(Path(events_json).read_text())
    ev = pd.DataFrame(d["actions"])
    # `position` (ms into the match) is the reliable field. `gameTime` is not:
    # 68% of rows look like "1 - 0:01" and the rest are a bare "135:22" on a
    # different clock, so parsing it as period-plus-time fails on a third of the
    # data. Period is derived from the timestamp against the half boundaries.
    ev["ms"] = pd.to_numeric(ev["position"], errors="coerce")
    ev["timestamp"] = ev["ms"] / 1000.0
    ev["period"] = np.where(ev["ms"] <= first_half_end_ms, 1, 2)
    ev["frame_idx"] = (ev["timestamp"] * FPS).round().astype("Int64")
    ev["player_id"] = pd.to_numeric(ev.get("player_id"), errors="coerce").astype("Int64")
    if nodes_csv is not None:
        sheet = load_team_sheet(nodes_csv)
        ev["jersey"] = ev["player_id"].map(lambda p: sheet["back_number"].get(p, np.nan))
    return ev[["period", "timestamp", "frame_idx", "label", "team",
               "player_id"] + (["jersey"] if nodes_csv is not None else [])]
