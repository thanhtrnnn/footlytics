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


def _scene(ball_path, players, interpolated=False):
    """players: {track_id: (team, [(x, y) per frame])}; ball_path: [(x, y) or None per frame]."""
    rows = []
    for f, b in enumerate(ball_path):
        for tid, (team, path) in players.items():
            x, y = path[f]
            rows.append(dict(frame_idx=f, period=1, timestamp=f / FPS, track_id=tid,
                             role=Role.PLAYER.value, team=team.value, x=x, y=y))
        if b is not None:
            rows.append(dict(frame_idx=f, period=1, timestamp=f / FPS, track_id=-1,
                             role=Role.BALL.value, team=Team.UNKNOWN.value, x=b[0], y=b[1],
                             interpolated=interpolated))
    meta = MatchMeta(match_id="P", fps=FPS, home=TeamInfo("H", "H"), away=TeamInfo("A", "A"))
    return MatchState(meta, pd.DataFrame(rows))


def _carried(n, speed_mps=3.0, home_off=(-1.0, 0.0), away_off=(-1.2, 0.3), still_frames=25):
    """Ball at home's feet for 1 s, then rolling at a constant speed with both teams alongside."""
    ball, home, away = [], [], []
    for f in range(n):
        bx = max(0, f - still_frames) * speed_mps / FPS
        ball.append((bx, 0.0))
        if f < still_frames:
            home.append((bx - 0.3, 0.0))
            away.append((15.0, 0.0))
        else:
            home.append((bx + home_off[0], home_off[1]))
            away.append((bx + away_off[0], away_off[1]))
    return ball, {1: (Team.HOME, home), 2: (Team.AWAY, away)}


def test_a_marker_almost_as_close_as_the_carrier_leaves_the_ball_contested():
    ball, players = _carried(100)                     # home 1.0 m, away 1.24 m behind the ball
    s = _status(infer_possession(_scene(ball, players)))
    assert (s.loc[:24] == "HOME").all()
    assert (s.loc[45:] == "CONTESTED").all()
    assert (s != "AWAY").all()


def test_a_nearer_defender_who_never_touches_the_ball_does_not_win_it():
    ball, players = _carried(100, home_off=(-1.25, 0.0), away_off=(-0.8, 0.0))
    s = _status(infer_possession(_scene(ball, players)))
    assert (s != "AWAY").all()
    assert (s.loc[45:] == "CONTESTED").all()


def _intercepted():
    """Home holds it 1 s, passes at 10 m/s, and an away player stops it 4 m out."""
    ball = [(0.0, 0.0)] * 25 + [(0.4 * k, 0.0) for k in range(1, 11)] + [(4.0, 0.0)] * 40
    home = [(-0.3, 0.0)] * len(ball)
    away = [(4.3, 0.0)] * len(ball)
    return ball, {1: (Team.HOME, home), 2: (Team.AWAY, away)}


def test_a_stopping_touch_by_the_defender_gives_him_the_ball():
    ball, players = _intercepted()
    s = _status(infer_possession(_scene(ball, players)))
    assert (s.loc[:31] == "HOME").all()
    assert (s.loc[32:] == "AWAY").all()


def test_a_defender_who_has_the_ball_to_himself_gets_it_without_a_touch():
    """The ball sits still and the home player has gone: nobody within a stride contests it."""
    ball = [(0.0, 0.0)] * 80
    home = [(-0.3, 0.0)] * 25 + [(30.0, 0.0)] * 55
    away = [(40.0, 0.0)] * 25 + [(0.8, 0.0)] * 55
    s = _status(infer_possession(_scene(ball, {1: (Team.HOME, home), 2: (Team.AWAY, away)})))
    assert (s.loc[:24] == "HOME").all()
    assert (s.loc[25:] == "AWAY").all()


def test_a_defender_with_the_carrier_still_within_a_stride_needs_a_touch_to_win_it():
    ball = [(0.0, 0.0)] * 80
    home = [(-0.3, 0.0)] * 25 + [(-1.8, 0.0)] * 55
    away = [(40.0, 0.0)] * 25 + [(0.8, 0.0)] * 55
    s = _status(infer_possession(_scene(ball, {1: (Team.HOME, home), 2: (Team.AWAY, away)})))
    assert (s.loc[:24] == "HOME").all()
    assert (s != "AWAY").all() and (s.loc[40:] == "CONTESTED").all()


def test_touch_column_names_the_team_whose_player_changed_the_balls_velocity():
    ball, players = _intercepted()
    poss = infer_possession(_scene(ball, players))
    touch = poss.set_index("frame_idx")["touch"]
    assert touch.loc[34] == "AWAY"
    assert touch.loc[5] == ""


def test_gain_column_says_how_each_spell_of_possession_was_won():
    ball, players = _intercepted()
    gain = infer_possession(_scene(ball, players)).set_index("frame_idx")["gain"]
    assert gain.loc[0] == "hold" and gain.loc[32] == "touch"
    assert (gain.drop([0, 32]) == "").all()

    ball = [(0.0, 0.0)] * 80
    home = [(-0.3, 0.0)] * 25 + [(30.0, 0.0)] * 55
    away = [(40.0, 0.0)] * 25 + [(0.8, 0.0)] * 55
    gain = infer_possession(_scene(ball, {1: (Team.HOME, home), 2: (Team.AWAY, away)})).set_index("frame_idx")["gain"]
    assert gain.loc[25] == "unclaimed"


def test_a_loose_ball_with_both_teams_in_reach_is_contested():
    ball = [(0.0, 0.0)] * 60
    players = {1: (Team.HOME, [(-1.0, 0.0)] * 60), 2: (Team.AWAY, [(1.1, 0.0)] * 60)}
    s = _status(infer_possession(_scene(ball, players)))
    assert (s.loc[15:] == "CONTESTED").all()
    assert (s.loc[:5] == "UNKNOWN").all()


def test_interpolated_ball_positions_are_not_evidence_of_control():
    ball = [(0.0, 0.0)] * 50
    players = {1: (Team.HOME, [(0.3, 0.0)] * 50)}
    s = _status(infer_possession(_scene(ball, players, interpolated=True)))
    assert (s == "UNKNOWN").all()


def test_touchline_dribble_within_two_metre_margin_is_not_ballout():
    # Far-touchline homography compression places an on-line ball ~1.5 m past y = -34 m
    # (kept by PipelineConfig.ball_margin_m = 2.0); it must remain in play rather than BALLOUT.
    path = [H1] * 25 + [(-10.0, -35.5)] * 25 + [H1] * 25
    s = _status(infer_possession(_state(path)))
    assert (s == "HOME").all()
