"""possession.py: who has the ball, inferred from Match State alone (no event feed)."""
import numpy as np
import pandas as pd

from footlytics.analytics.possession import infer_possession
from footlytics.state.schema import MatchMeta, MatchState, Role, Team, TeamInfo

FPS = 25.0
H1, H2, A1 = (-10.0, 0.0), (10.0, 0.0), (0.0, 10.0)   # two home players, one away


def _state(ball_path, a1_path=None):
    """One row per frame for H1, H2, A1 (A1 may move), plus the ball where given.

    `ball_path[f]` is (x, y) or None for a frame with no ball detected.
    """
    rows = []
    for f, b in enumerate(ball_path):
        a1 = a1_path[f] if a1_path is not None else A1
        for tid, (x, y), team in ((1, H1, Team.HOME), (2, H2, Team.HOME), (3, a1, Team.AWAY)):
            rows.append(dict(frame_idx=f, period=1, timestamp=f / FPS, track_id=tid,
                             role=Role.PLAYER.value, team=team.value, x=x, y=y))
        if b is not None:
            rows.append(dict(frame_idx=f, period=1, timestamp=f / FPS, track_id=-1,
                             role=Role.BALL.value, team=Team.UNKNOWN.value, x=b[0], y=b[1]))
    meta = MatchMeta(match_id="P", fps=FPS, home=TeamInfo("H", "H"), away=TeamInfo("A", "A"))
    return MatchState(meta, pd.DataFrame(rows))


def _line(p, q, n):
    return [(p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t) for t in np.linspace(0, 1, n)]


def _status(poss):
    return poss.set_index("frame_idx")["ballStatus"]


def test_possession_survives_a_pass_between_teammates():
    # 1 s at H1's feet, 1 s in flight (nobody within reach), 1 s at H2's feet.
    path = [H1] * 25 + _line(H1, H2, 25) + [H2] * 25
    s = _status(infer_possession(_state(path)))
    assert (s == "HOME").all()


def test_a_brief_touch_by_the_other_team_does_not_flip_possession():
    # Ball passes A1 for 5 frames (0.2 s) on its way, then reaches H2.
    path = [H1] * 25 + [A1] * 5 + [H2] * 25
    s = _status(infer_possession(_state(path), min_hold_s=0.5))
    assert (s == "HOME").all()


def test_possession_flips_once_the_other_team_holds_the_ball():
    path = [H1] * 25 + [A1] * 25
    s = _status(infer_possession(_state(path), min_hold_s=0.5))
    assert (s.loc[:24] == "HOME").all()
    # Backdated to the first touch once the hold is confirmed.
    assert (s.loc[25:] == "AWAY").all()


def test_a_long_stretch_without_the_ball_is_unknown_not_carried():
    path = [H1] * 25 + [None] * 100 + [H1] * 25
    s = _status(infer_possession(_state(path), max_gap_s=2.0))
    assert (s.loc[:24] == "HOME").all()
    assert (s.loc[25:124] == "UNKNOWN").all()
    assert (s.loc[125:] == "HOME").all()


def test_a_short_detection_gap_carries_the_last_team():
    path = [H1] * 25 + [None] * 10 + [H1] * 25
    s = _status(infer_possession(_state(path), max_gap_s=2.0))
    assert (s == "HOME").all()


def test_ball_beyond_the_lines_is_ballout():
    path = [H1] * 25 + [(0.0, 40.0)] * 25        # 6 m past the near touchline (W = 68)
    s = _status(infer_possession(_state(path)))
    assert (s.loc[25:] == "BALLOUT").all()


def test_output_has_the_columns_press_distance_and_transitions_read():
    poss = infer_possession(_state([H1] * 10))
    assert {"frame_idx", "period", "timestamp", "ballStatus"} <= set(poss.columns)
    assert len(poss) == 10


def test_uncontrolled_free_ball_flight_past_max_gap_resets_to_unknown():
    # Home holds briefly (40 frames), then ball travels in flight (no player within 2 m) for
    # 60 frames (> 2.0 s at 25 fps) before away collects it. Flight must become UNKNOWN.
    far_away = (0.0, 20.0)
    path = [H1] * 40 + [(0.0, 30.0)] * 60 + [A1] * 40
    a1_path = [far_away] * 100 + [A1] * 40
    s = _status(infer_possession(_state(path, a1_path=a1_path), min_hold_s=0.5, max_gap_s=2.0))
    assert (s.loc[:39] == "HOME").all()
    assert (s.loc[40:99] == "UNKNOWN").all()
    assert (s.loc[100:] == "AWAY").all()


def test_deep_box_1v1_mark_does_not_flip_possession_to_defender():
    # Home attacker dribbles inside away's deep box (x_own_away <= -30 -> x >= +30), while an
    # away defender tracks at 1.0 m (closer than attacker's 2.5 m, inside 3.8 m contest radius).
    rows = []
    bx = 35.0
    for f in range(80):
        # First 40 frames: home clearly controls unchallenged (away far away).
        # Next 40 frames: away defender marks tighter (1.0 m) while home attacker stays at 2.5 m.
        hx = bx if f < 40 else bx - 2.5
        ax = 0.0 if f < 40 else bx + 1.0
        for tid, x, y, team in ((1, hx, 0.0, Team.HOME), (2, ax, 0.0, Team.AWAY)):
            rows.append(dict(frame_idx=f, period=1, timestamp=f / FPS, track_id=tid,
                             role=Role.PLAYER.value, team=team.value, x=x, y=y))
        rows.append(dict(frame_idx=f, period=1, timestamp=f / FPS, track_id=-1,
                         role=Role.BALL.value, team=Team.UNKNOWN.value, x=bx, y=0.0))
    st = MatchState(MatchMeta(match_id="B", fps=FPS, home=TeamInfo("H", "H"), away=TeamInfo("A", "A")),
                    pd.DataFrame(rows))
    s = _status(infer_possession(st))
    assert (s == "HOME").all()
