"""analytics.ball.interpolate_ball, and the schema column it writes into."""
import numpy as np
import pandas as pd
import pytest

from footlytics.analytics.ball import ball_coverage, interpolate_ball
from footlytics.state.schema import MatchMeta, MatchState, Role, Team, TeamInfo

FPS = 25.0


def _ball_rows(frames):
    """One ball row per frame in `frames`, moving 1 m per frame along x."""
    return pd.DataFrame([dict(
        frame_idx=f, period=1, timestamp=f / FPS, track_id=99,
        role=Role.BALL.value, team=Team.UNKNOWN.value, jersey=np.nan,
        bbox_x=10.0, bbox_y=10.0, bbox_w=8.0, bbox_h=8.0, det_conf=0.5,
        x=float(f), y=0.0, z=0.0, speed=np.nan, accel=np.nan, interpolated=False,
    ) for f in frames])


def _player_rows(frames):
    return pd.DataFrame([dict(
        frame_idx=f, period=1, timestamp=f / FPS, track_id=1,
        role=Role.PLAYER.value, team=Team.HOME.value, jersey=np.nan,
        bbox_x=0.0, bbox_y=0.0, bbox_w=5.0, bbox_h=20.0, det_conf=0.9,
        x=0.0, y=5.0, z=np.nan, speed=np.nan, accel=np.nan, interpolated=False,
    ) for f in frames])


def test_short_gap_is_filled_linearly():
    tracks = _ball_rows([0, 1, 2, 6, 7])          # frames 3,4,5 missing: a 3-frame gap
    out = interpolate_ball(tracks, FPS, max_gap=5)
    ball = out[out["role"] == Role.BALL.value].set_index("frame_idx")

    assert sorted(ball.index) == [0, 1, 2, 3, 4, 5, 6, 7]
    # x advances 1 m/frame, so the filled frames must land on their own index
    assert ball.loc[[3, 4, 5], "x"].tolist() == pytest.approx([3.0, 4.0, 5.0])
    assert ball.loc[[3, 4, 5], "interpolated"].all()
    assert not ball.loc[[0, 1, 2, 6, 7], "interpolated"].any()
    # nothing was detected on a filled frame, so there is no box and no confidence
    assert ball.loc[[3, 4, 5], "det_conf"].isna().all()
    assert ball.loc[[3, 4, 5], "bbox_x"].isna().all()


def test_long_gap_is_left_alone():
    tracks = _ball_rows([0, 1, 10, 11])           # an 8-frame gap
    out = interpolate_ball(tracks, FPS, max_gap=5)
    assert sorted(out["frame_idx"].unique()) == [0, 1, 10, 11]


def test_does_not_extrapolate_past_the_last_sighting():
    tracks = pd.concat([_ball_rows([2, 3]), _player_rows(range(8))], ignore_index=True)
    out = interpolate_ball(tracks, FPS, max_gap=5)
    ball = out[out["role"] == Role.BALL.value]
    # frames 0-1 and 4-7 have no ball either side to interpolate between
    assert sorted(ball["frame_idx"].unique()) == [2, 3]


def test_max_gap_zero_disables():
    tracks = _ball_rows([0, 1, 4])
    assert interpolate_ball(tracks, FPS, max_gap=0).equals(tracks)


def test_players_are_untouched():
    tracks = pd.concat([_ball_rows([0, 4]), _player_rows([0, 2, 4])], ignore_index=True)
    out = interpolate_ball(tracks, FPS, max_gap=5)
    players = out[out["role"] == Role.PLAYER.value]
    assert sorted(players["frame_idx"].tolist()) == [0, 2, 4]   # gap at 1 and 3 left alone
    assert not players["interpolated"].any()


def test_interpolated_survives_a_matchstate_round_trip(tmp_path):
    """The column must come back as a real bool, and False where nothing was filled.

    `_coerce` casts every column to its schema dtype; a plain astype(bool) turns
    NaN into True, which would mark an untouched table as entirely interpolated.
    """
    tracks = interpolate_ball(_ball_rows([0, 1, 2, 5, 6]), FPS, max_gap=5)
    meta = MatchMeta(match_id="T", fps=FPS, home=TeamInfo("H"), away=TeamInfo("A"))
    state = MatchState(meta, tracks)

    assert state.tracks["interpolated"].dtype == bool
    assert state.tracks["interpolated"].sum() == 2           # frames 3 and 4
    assert not state.tracks["interpolated"].all()

    state.save(tmp_path / "s")
    back = MatchState.load(tmp_path / "s")
    assert back.tracks["interpolated"].dtype == bool
    assert back.tracks["interpolated"].sum() == 2


def test_missing_column_defaults_to_false_not_true():
    """A table built without the column at all must not come back all-True."""
    tracks = _ball_rows([0, 1, 2]).drop(columns=["interpolated"])
    meta = MatchMeta(match_id="T", fps=FPS, home=TeamInfo("H"), away=TeamInfo("A"))
    state = MatchState(meta, tracks)
    assert state.tracks["interpolated"].dtype == bool
    assert not state.tracks["interpolated"].any()


def test_ball_coverage_separates_detected_from_filled():
    filled = interpolate_ball(_ball_rows([0, 1, 2, 5, 6]), FPS, max_gap=5)
    assert ball_coverage(filled, frames=7) == pytest.approx(1.0)
    assert ball_coverage(filled, frames=7, detected_only=True) == pytest.approx(5 / 7)


# ---------------------------------------------------------------- never across a break


def _two_sightings(f0, f1, dx, **right):
    """Ball seen at frame f0 at x=0 and at f1 at x=dx (y=0); `right` overrides the second row."""
    rows = _ball_rows([f0, f1])
    rows["x"] = [0.0, dx]
    for k, v in right.items():
        rows.loc[1, k] = v
    return rows


def _filled(out):
    return sorted(out[(out["role"] == Role.BALL.value) & out["interpolated"]]["frame_idx"])


def test_a_new_ball_track_id_is_a_segment_break_and_is_not_bridged():
    # ball_link gave the second sighting a new id: it declared it unreachable from the first
    out = interpolate_ball(_two_sightings(0, 4, 2.0, track_id=100), FPS, max_gap=5)
    assert _filled(out) == []


def test_same_id_but_impossible_speed_is_not_bridged():
    # 29.7 m in 4 frames (the Hanoi 17288-17292 case) = 186 m/s
    out = interpolate_ball(_two_sightings(0, 4, 29.7), FPS, max_gap=5)
    assert _filled(out) == []
    # while a 31 m/s ball (inside the gate) over the same gap is filled
    out = interpolate_ball(_two_sightings(0, 4, 5.0), FPS, max_gap=5)
    assert _filled(out) == [1, 2, 3]


def test_not_bridged_across_a_shot_change_or_a_non_live_frame():
    base = _two_sightings(0, 4, 2.0)
    base["shot"], base["live_wide"] = 3, True
    assert _filled(interpolate_ball(base, FPS)) == [1, 2, 3]
    cut = base.copy()
    cut.loc[1, "shot"] = 4
    assert _filled(interpolate_ball(cut, FPS)) == []
    replay = base.copy()
    replay.loc[1, "live_wide"] = False
    assert _filled(interpolate_ball(replay, FPS)) == []
    period = base.copy()
    period.loc[1, "period"] = 2
    assert _filled(interpolate_ball(period, FPS)) == []


def test_filled_rows_carry_the_neighbours_shot_and_live_wide():
    base = _two_sightings(10, 13, 1.0)
    base["shot"], base["live_wide"] = 7, True
    out = interpolate_ball(base, FPS)
    new = out[out["interpolated"]]
    assert new["shot"].tolist() == [7, 7] and new["live_wide"].tolist() == [True, True]


def test_each_gap_is_judged_alone():
    # three sightings: the first gap is plausible, the second jumps 30 m
    rows = _ball_rows([0, 3, 6])
    rows["x"] = [0.0, 1.0, 31.0]
    out = interpolate_ball(rows, FPS, max_gap=5)
    assert _filled(out) == [1, 2]
