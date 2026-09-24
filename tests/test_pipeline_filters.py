"""Per-frame filters in pipeline.radar and identity.apply_to_state."""
import numpy as np
import pandas as pd

from footlytics.geometry.pitch import DEFAULT_PITCH
from footlytics.perception.identity import apply_to_state, build_tracklets
from footlytics.pipeline.radar import on_pitch
from footlytics.state.schema import MatchMeta, MatchState, Role, Team, TeamInfo

BALL, PLAYER = Role.BALL.value, Role.PLAYER.value


def test_ball_gets_its_own_margin():
    """Measured: the ball beside the far touchline projects to y = -37.5 on a 34 m
    half-width. The 2 m player margin deleted it; a person there is still dropped."""
    xy = np.array([[35.7, -37.5], [35.7, -37.5], [0.0, -60.0], [0.0, 0.0], [np.nan, 0.0]])
    roles = [BALL, PLAYER, BALL, PLAYER, BALL]
    keep = on_pitch(xy, roles, DEFAULT_PITCH, margin_m=2.0, ball_margin_m=10.0)
    assert keep.tolist() == [True, False, False, True, False]


def test_equal_margins_reduce_to_the_old_filter():
    xy = np.array([[35.7, -37.5], [0.0, 0.0]])
    keep = on_pitch(xy, [BALL, PLAYER], DEFAULT_PITCH, margin_m=2.0, ball_margin_m=2.0)
    assert keep.tolist() == [False, True]


def _state():
    rows = []
    for f in range(5):
        for tid, role in ((1, PLAYER), (2, PLAYER), (9, BALL)):
            rows.append(dict(frame_idx=f, period=1, timestamp=f / 25.0, track_id=tid, role=role,
                             team=Team.UNKNOWN.value, x=float(tid), y=0.0))
    return MatchState(MatchMeta(match_id="t", fps=25.0, home=TeamInfo("H"), away=TeamInfo("A")),
                      pd.DataFrame(rows))


def test_apply_to_state_carries_role():
    """The pipeline marks official tracklets Role.REFEREE; apply_to_state used to write
    team only, so officials stayed 'player' and were counted in MatchState.players."""
    state = _state()
    tracklets = build_tracklets(state)
    by_id = {t.id: t for t in tracklets}
    by_id[2].team, by_id[2].role = Team.OFFICIAL.value, Role.REFEREE.value
    by_id[1].team = Team.HOME.value

    out = apply_to_state(state, tracklets)
    roles = out.tracks.groupby("track_id", observed=True)["role"].first().astype(str).to_dict()
    assert roles == {1: PLAYER, 2: Role.REFEREE.value, 9: BALL}      # ball untouched
    assert set(out.players["track_id"]) == {1}


def test_spare_ball_beyond_the_touchline_is_not_a_candidate():
    """Measured: spare balls by the far touchline project to y = -37.5 and -38.0."""
    xy = np.array([[35.8, -37.5], [1.8, -38.0], [10.0, 5.0]])
    keep = on_pitch(xy, [BALL, BALL, BALL], DEFAULT_PITCH, margin_m=2.0, ball_margin_m=2.0)
    assert keep.tolist() == [False, False, True]


def test_choose_ball_prefers_the_reachable_candidate_over_the_confident_one():
    from footlytics.pipeline.radar import choose_ball

    xy = np.array([[40.0, 10.0], [0.5, 0.0]])        # a far, confident one; a near one
    conf = np.array([0.9, 0.4])
    last = np.array([0.0, 0.0])
    assert choose_ball(xy, conf, last, frames_since=1, fps=25.0) == 1
    # nothing reachable: no ball this frame rather than a 40 m teleport
    assert choose_ball(xy[:1], conf[:1], last, frames_since=1, fps=25.0) is None
    # no recent sighting: most confident
    assert choose_ball(xy, conf, None, frames_since=0, fps=25.0) == 0
    assert choose_ball(xy, conf, last, frames_since=100, fps=25.0) == 0
