"""tactics.py: block lines, press counts, ball pressure, turnovers and formation templates."""
import numpy as np
import pandas as pd
import pytest

from footlytics.analytics.tactics import (
    FORMATION_TEMPLATES, press_distance, press_on_ball, retreat_speed, team_block, transitions)
from footlytics.state.schema import MatchMeta, MatchState, Role, Team, TeamInfo

FPS = 25.0


def _state(people, ball, n_frames):
    """people: [(track_id, team, role, path)] with path a (x, y) or a function of the frame index."""
    rows = []
    for f in range(n_frames):
        for tid, team, role, path in people:
            x, y = path(f) if callable(path) else path
            rows.append(dict(frame_idx=f, period=1, timestamp=f / FPS, track_id=tid, role=role.value,
                             team=team.value, x=x, y=y))
        bx, by = ball(f) if callable(ball) else ball
        rows.append(dict(frame_idx=f, period=1, timestamp=f / FPS, track_id=-1, role=Role.BALL.value,
                         team=Team.UNKNOWN.value, x=bx, y=by))
    meta = MatchMeta(match_id="T", fps=FPS, home=TeamInfo("H", "H"), away=TeamInfo("A", "A"))
    return MatchState(meta, pd.DataFrame(rows))


def _possession(statuses, touches=None, gains=None):
    rows = [{"frame_idx": f, "period": 1, "timestamp": f / FPS, "ballStatus": s,
             "touch": (touches or {}).get(f, ""), "gain": (gains or {}).get(f, "")}
            for f, s in enumerate(statuses)]
    return pd.DataFrame(rows)


def test_formation_templates_do_not_repeat_a_band_triple():
    assert len(set(FORMATION_TEMPLATES.values())) == len(FORMATION_TEMPLATES)


def test_team_block_reports_the_mean_of_the_three_deepest_outfielders():
    outfield = [(-30.0, 0.0), (-24.0, 5.0), (-18.0, -5.0), (-6.0, 0.0), (0.0, 10.0), (6.0, -10.0)]
    people = [(1, Team.HOME, Role.GOALKEEPER, (-50.0, 0.0))]
    people += [(2 + i, Team.HOME, Role.PLAYER, p) for i, p in enumerate(outfield)]
    st = _state(people, (0.0, 0.0), 1)
    row = team_block(st, sample_every=1, min_outfield=4).iloc[0]
    assert row["def_line3_x"] == pytest.approx(-24.0)
    assert row["def_line_x"] == pytest.approx(-19.5)


def test_press_distance_does_not_count_the_goalkeeper():
    people = [(1, Team.HOME, Role.PLAYER, (0.0, 0.0)),
              (10, Team.AWAY, Role.GOALKEEPER, (3.0, 0.0)),
              (11, Team.AWAY, Role.PLAYER, (30.0, 0.0)),
              (12, Team.AWAY, Role.PLAYER, (40.0, 0.0)),
              (13, Team.AWAY, Role.PLAYER, (8.0, 0.0))]
    st = _state(people, (0.5, 0.0), 1)
    poss = _possession(["HOME"])
    row = press_distance(st, poss, sample_every=1).iloc[0]
    assert row["n_within_5m"] == 0 and row["n_within_10m"] == 1
    assert row["nearest_m"] == pytest.approx(7.5)


def _pressure_scene(defender_path, carrier=(0.0, 0.0), n=30):
    """Home carries at the origin (attacking +x). One away defender follows `defender_path`."""
    people = [(1, Team.HOME, Role.PLAYER, carrier),
              (10, Team.AWAY, Role.GOALKEEPER, (50.0, 0.0)),
              (11, Team.AWAY, Role.PLAYER, defender_path)]
    return _state(people, carrier, n)


def test_a_defender_closing_in_front_of_the_carrier_is_pressing():
    # 8 m ahead of the carrier, running at him at 4 m/s.
    st = _pressure_scene(lambda f: (8.0 - 4.0 * f / FPS, 0.0))
    rows = press_on_ball(st, sample_every=5)
    away = rows[(rows["team"] == "away") & (rows["frame_idx"] == 20)].iloc[0]
    assert away["n_pressing"] == 1


def test_a_defender_standing_beside_the_carrier_presses_him_but_is_not_closing_in():
    rows = press_on_ball(_pressure_scene((2.0, 1.0)), sample_every=5)
    away = rows[rows["team"] == "away"]
    assert (away["n_pressing"] == 1).all() and (away["n_closing"] == 0).all()


def test_a_defender_running_at_the_carrier_is_closing_in():
    st = _pressure_scene(lambda f: (8.0 - 4.0 * f / FPS, 0.0))
    rows = press_on_ball(st, sample_every=5)
    away = rows[(rows["team"] == "away") & (rows["frame_idx"] == 20)].iloc[0]
    assert away["n_closing"] == 1


def test_a_defender_standing_off_in_front_is_not_pressing():
    st = _pressure_scene((6.0, 0.0))
    rows = press_on_ball(st, sample_every=5)
    assert (rows[rows["team"] == "away"]["n_pressing"] == 0).all()


def test_a_defender_within_three_metres_behind_the_carrier_is_pressing_but_not_at_five():
    close = press_on_ball(_pressure_scene((-2.5, 0.0)), sample_every=5)
    far = press_on_ball(_pressure_scene((-5.0, 0.0)), sample_every=5)
    assert (close[close["team"] == "away"]["n_pressing"] == 1).all()
    assert (far[far["team"] == "away"]["n_pressing"] == 0).all()


def test_the_goalkeeper_is_never_a_presser():
    people = [(1, Team.HOME, Role.PLAYER, (40.0, 0.0)),
              (10, Team.AWAY, Role.GOALKEEPER, (43.0, 0.0)),
              (11, Team.AWAY, Role.PLAYER, (-20.0, 0.0))]
    rows = press_on_ball(_state(people, (40.0, 0.0), 10), sample_every=5)
    assert (rows[rows["team"] == "away"]["n_pressing"] == 0).all()


def _two_teams(ball):
    people = [(1, Team.HOME, Role.PLAYER, (0.0, 0.0)), (2, Team.AWAY, Role.PLAYER, (5.0, 0.0))]
    return _state(people, ball, 200)


def _turnover_possession(touch_at=None, gain=None):
    status = ["HOME"] * 75 + ["AWAY"] * 125
    return _possession(status, {touch_at: "AWAY"} if touch_at is not None else None,
                       {75: gain} if gain else None)


def test_a_turnover_in_the_losers_final_third_is_reported_with_its_zone_and_gain():
    # Home loses it at x = +30 (its attacking third); away carries it 5 m towards -x over 5 s.
    ball = lambda f: (30.0 if f < 75 else 30.0 - (f - 75) / FPS, 0.0)
    t = transitions(_two_teams(ball), _turnover_possession(touch_at=74))
    assert len(t) == 1
    row = t.iloc[0]
    assert row["won_by"] == "away" and row["zone"] == "final_third"
    assert row["gain_5s_m"] == pytest.approx(4.96, abs=0.01)


def test_a_turnover_in_the_middle_third_has_the_middle_zone():
    t = transitions(_two_teams((2.0, 0.0)), _turnover_possession(touch_at=75))
    assert t.iloc[0]["zone"] == "middle_third"


def test_a_turnover_in_the_losers_own_third_has_the_own_zone():
    t = transitions(_two_teams((-30.0, 0.0)), _turnover_possession(touch_at=75))
    assert t.iloc[0]["zone"] == "own_third"


def test_no_turnover_without_a_touch_by_the_team_that_won_the_ball():
    t = transitions(_two_teams((2.0, 0.0)), _turnover_possession(touch_at=None))
    assert t.empty


def test_a_ball_taken_from_a_holder_nobody_contested_is_a_turnover_without_a_touch():
    t = transitions(_two_teams((2.0, 0.0)), _turnover_possession(gain="unclaimed"))
    assert len(t) == 1


def test_a_ball_picked_up_from_nobody_needs_a_touch_even_when_possession_changed_hands():
    assert transitions(_two_teams((2.0, 0.0)), _turnover_possession(gain="hold")).empty


def test_a_possession_table_without_touch_evidence_still_gives_turnovers():
    poss = _turnover_possession().drop(columns=["touch", "gain"])
    assert len(transitions(_two_teams((2.0, 0.0)), poss)) == 1


def test_retreat_speed_uses_only_players_in_view_throughout():
    people = [(1, Team.HOME, Role.PLAYER, lambda f: (-10.0 - 2.0 * f / FPS, 0.0)),
              (2, Team.HOME, Role.PLAYER, lambda f: (-12.0 - 2.0 * f / FPS, 5.0)),
              (3, Team.HOME, Role.PLAYER, lambda f: (-14.0 - 2.0 * f / FPS, -5.0))]
    st = _state(people, (0.0, 0.0), 100)
    # A fourth player enters the view late, far upfield: the visible centroid would jump forward.
    late = st.tracks[st.tracks["frame_idx"] >= 60].copy()
    late_player = late[late["track_id"] == 1].assign(track_id=9, x=30.0, y=2.0)
    st2 = MatchState(st.meta, pd.concat([st.tracks, late_player]))
    assert retreat_speed(st2, Team.HOME.value, 0, 99, sample_every=5) == pytest.approx(-2.0, abs=0.05)


def test_retreat_speed_is_unavailable_with_fewer_than_three_players_in_view_throughout():
    people = [(1, Team.HOME, Role.PLAYER, lambda f: (-10.0 - 2.0 * f / FPS, 0.0)),
              (2, Team.HOME, Role.PLAYER, lambda f: (-12.0 - 2.0 * f / FPS, 5.0))]
    st = _state(people, (0.0, 0.0), 100)
    assert np.isnan(retreat_speed(st, Team.HOME.value, 0, 99, sample_every=5))
