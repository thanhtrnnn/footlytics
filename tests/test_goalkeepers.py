"""Goalkeeper role: found by position, team from the end he defends."""
import pandas as pd

from footlytics.analytics.tactics import goalkeepers, infer_attacking_direction
from footlytics.perception.identity import (build_tracklets, goalkeeper_teams,
                                            goalkeeper_tracklets)
from footlytics.state.schema import MatchMeta, MatchState, Role, Team, TeamInfo

HOME, AWAY = Team.HOME.value, Team.AWAY.value
N = 50


def _rows(tid, team, role, xy):
    return [dict(frame_idx=f, period=1, timestamp=f / 25.0, track_id=tid, role=role,
                 team=team, x=xy(f)[0], y=xy(f)[1]) for f in range(N)]


def _state(rows):
    meta = MatchMeta(match_id="G", fps=25.0, home=TeamInfo("H", "H"), away=TeamInfo("A", "A"))
    return MatchState(meta, pd.DataFrame(rows))


def _clip():
    """Home defends the left. Keeper 1 stands in the left box but was clustered AWAY.
    Defender 2 (home) is in the box all clip but 4 m in front of the keeper."""
    P = Role.PLAYER.value
    rows = _rows(1, AWAY, P, lambda f: (-50.0, 1.0))
    rows += _rows(2, HOME, P, lambda f: (-46.0, 8.0))
    rows += _rows(3, HOME, P, lambda f: (-20.0, -10.0))
    rows += _rows(4, HOME, P, lambda f: (-5.0, 12.0))
    rows += _rows(5, AWAY, P, lambda f: (10.0, 0.0))
    rows += _rows(6, AWAY, P, lambda f: (25.0, -15.0))
    rows += _rows(7, AWAY, P, lambda f: (49.0, 0.0))
    return _state(rows)


def test_keeper_in_the_wrong_kit_cluster_is_found_and_given_the_defending_team():
    st = _clip()
    tl = build_tracklets(st)
    keepers = goalkeeper_tracklets(st.tracks, tl, 52.5, 34.0)
    assert keepers == {1: -1}
    for t in tl:
        if t.id == 1:
            t.team = AWAY
        elif t.id in (2, 3, 4):
            t.team = HOME
        else:
            t.team = AWAY
    assert goalkeeper_teams(st.tracks, tl, keepers) == {1: HOME}


def test_a_defender_who_lives_in_the_box_stays_a_player():
    st = _clip()
    keepers = goalkeeper_tracklets(st.tracks[st.tracks.track_id != 1], build_tracklets(st), 52.5, 34.0)
    assert 2 not in keepers


def test_short_tracklets_are_not_keepers():
    rows = _rows(1, HOME, Role.PLAYER.value, lambda f: (-50.0, 0.0))
    st = _state([r for r in rows if r["frame_idx"] < 20])
    assert goalkeeper_tracklets(st.tracks, build_tracklets(st), 52.5, 34.0) == {}


def test_direction_follows_the_labelled_keeper_not_the_extreme_player():
    """Away attacks the left? No: away's keeper is at +x so away attacks -x even though
    an away outfielder is far deeper at -x than the keeper is at +x."""
    P, G = Role.PLAYER.value, Role.GOALKEEPER.value
    rows = _rows(1, HOME, G, lambda f: (-50.0, 0.0))
    rows += _rows(2, HOME, P, lambda f: (-5.0, 0.0))
    rows += _rows(3, AWAY, G, lambda f: (40.0, 0.0))
    rows += _rows(4, AWAY, P, lambda f: (-52.0, 5.0))
    st = _state(rows)
    d = infer_attacking_direction(st)
    assert d == {(1, HOME): 1, (1, AWAY): -1}
    assert goalkeepers(st) == {HOME: 1, AWAY: 3}


def test_without_keeper_roles_the_extreme_player_heuristic_still_applies():
    P = Role.PLAYER.value
    rows = _rows(1, HOME, P, lambda f: (-50.0, 0.0)) + _rows(2, HOME, P, lambda f: (-5.0, 0.0))
    assert infer_attacking_direction(_state(rows)) == {(1, HOME): 1}
