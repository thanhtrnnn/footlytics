"""Speed clamps (a person is not the ball) and `MatchState.validate`'s implied-speed check."""
import numpy as np
import pandas as pd
import pytest

from footlytics.analytics.kinematics import (MAX_BALL_SPEED, MAX_HUMAN_SPEED, add_kinematics)
from footlytics.state.schema import MatchMeta, MatchState, Role, Team, TeamInfo

FPS = 25.0


def _rows(role, track_id, xs, ys=None, frames=None, **extra):
    n = len(xs)
    frames = list(range(n)) if frames is None else list(frames)
    return pd.DataFrame(dict(
        frame_idx=frames, period=1, timestamp=np.asarray(frames) / FPS, track_id=track_id,
        role=role, team=Team.HOME.value if role != Role.BALL.value else Team.UNKNOWN.value,
        jersey=np.nan, bbox_x=0.0, bbox_y=0.0, bbox_w=5.0, bbox_h=5.0, det_conf=0.9,
        x=np.asarray(xs, float), y=np.zeros(n) if ys is None else np.asarray(ys, float),
        z=np.nan, speed=np.nan, accel=np.nan, interpolated=False, **extra))


def _state(rows, fps=FPS):
    return MatchState(MatchMeta(match_id="T", fps=fps, home=TeamInfo("H"), away=TeamInfo("A")),
                      pd.concat(rows, ignore_index=True))


def _problems(rows):
    # only the speed problems: a toy table trips the unrelated pitch / head-count checks
    return [p for p in _state(rows).validate() if "steps imply" in p or "speeds exceed" in p]


# ------------------------------------------------------------------ kinematics


def test_a_fast_ball_is_not_clamped_to_the_human_speed_limit():
    n = 80
    ball = _rows(Role.BALL.value, 99, 0.8 * np.arange(n))           # 20 m/s, steady
    runner = _rows(Role.PLAYER.value, 1, 0.8 * np.arange(n))        # the same, as a player
    out = add_kinematics(pd.concat([ball, runner], ignore_index=True), FPS)
    b = out[out["role"] == Role.BALL.value]["speed"]
    p = out[out["role"] == Role.PLAYER.value]["speed"]
    assert b.iloc[n // 2] == pytest.approx(20.0, abs=0.5)
    assert p.max() == pytest.approx(MAX_HUMAN_SPEED)                # still capped: that is a tracking artefact
    assert not (b == MAX_HUMAN_SPEED).any()


def test_the_ball_has_its_own_cap():
    ball = _rows(Role.BALL.value, 99, 2.0 * np.arange(80))          # 50 m/s: not a real ball either
    out = add_kinematics(ball, FPS)
    assert out["speed"].max() == pytest.approx(MAX_BALL_SPEED)
    assert out["accel"].abs().max() < 1000                          # not clipped to a player's 10


# ------------------------------------------------------------------ validate


def test_validate_reports_a_player_teleport_the_clamped_speed_column_hides():
    n = 100
    x = np.zeros(n)
    x[50:] = 20.0                                                   # 20 m in one frame = 500 m/s
    rows = _rows(Role.PLAYER.value, 1, x)
    st = _state([rows])
    st.tracks = add_kinematics(st.tracks, FPS)
    assert st.tracks["speed"].max() <= MAX_HUMAN_SPEED + 1e-6       # the old check can never fire
    msgs = [p for p in st.validate() if "steps imply" in p]
    assert len(msgs) == 1 and "player" in msgs[0] and "12.5 m/s" in msgs[0]


def test_validate_is_quiet_for_a_runner_and_a_driven_ball():
    n = 100
    runner = _rows(Role.PLAYER.value, 1, 0.3 * np.arange(n))        # 7.5 m/s
    ball = _rows(Role.BALL.value, 99, 1.2 * np.arange(n) - 60)      # 30 m/s
    assert _problems([runner, ball]) == []


def test_validate_has_separate_limits_for_players_and_ball():
    n = 100
    x = 1.0 * np.arange(n) - 50                                     # 25 m/s
    assert any("player" in p for p in _problems([_rows(Role.PLAYER.value, 1, x)]))
    assert _problems([_rows(Role.BALL.value, 99, x)]) == []         # fine for a ball
    fast = 2.0 * np.arange(n) - 100                                 # 50 m/s
    assert any("ball" in p and "40 m/s" in p for p in _problems([_rows(Role.BALL.value, 99, fast)]))


def test_one_ball_teleport_is_reported_but_one_player_blip_in_a_long_track_is_tolerated():
    n = 2000
    x = np.zeros(n)
    x[1000:] = 8.0
    assert any("ball" in p for p in _problems([_rows(Role.BALL.value, 99, x)]))
    assert _problems([_rows(Role.PLAYER.value, 1, x)]) == []        # 1 of 1999 steps = 0.05%


def test_validate_uses_elapsed_time_across_gaps_and_skips_shot_and_period_changes():
    # 4 m over 10 frames is 10 m/s, not 100 m/s
    gap = _rows(Role.PLAYER.value, 1, [0.0, 4.0], frames=[0, 10])
    assert _problems([gap]) == []
    # a jump across a shot change is a cut, not a teleport
    cut = _rows(Role.PLAYER.value, 1, [0.0, 30.0], shot=[1, 2], live_wide=True)
    assert _problems([cut]) == []
    same = _rows(Role.PLAYER.value, 1, [0.0, 30.0], shot=[1, 1], live_wide=True)
    assert any("player" in p for p in _problems([same]))
    # nor is a jump into a replay frame
    replay = _rows(Role.PLAYER.value, 1, [0.0, 30.0], shot=[1, 1], live_wide=[True, False])
    assert _problems([replay]) == []
    # nor across periods
    per = _rows(Role.PLAYER.value, 1, [0.0, 30.0])
    per["period"] = [1, 2]
    assert _problems([per]) == []
