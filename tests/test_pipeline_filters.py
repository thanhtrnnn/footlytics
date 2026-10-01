"""Per-frame filters in pipeline.radar and identity.apply_to_state."""
import numpy as np
import pandas as pd

from footlytics.geometry.pitch import DEFAULT_PITCH
from footlytics.perception.identity import apply_to_state, build_tracklets
from footlytics.pipeline.radar import on_pitch
from footlytics.state.schema import MatchMeta, MatchState, Role, Team, TeamInfo

BALL, PLAYER = Role.BALL.value, Role.PLAYER.value


def test_ball_gets_its_own_margin():
    """`on_pitch` mechanics: the ball and people can be held to different margins."""
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


def test_selector_lets_go_of_a_still_ball_when_another_moves():
    """Measured: a second ball lay still in the box for 57 s on the 3-minute clip and
    continuity alone held it for 45% of all sightings."""
    from footlytics.pipeline.radar import BallSelector

    sel = BallSelector(fps=25.0)
    still = np.array([42.7, 14.3])
    picks = []
    for f in range(150):
        cands = [still]
        if f >= 100:                       # the match ball comes into view, moving
            cands.append(np.array([20.0 + 0.5 * (f - 100), 0.0]))
        p = sel.pick(f, np.array(cands), np.array([0.9] + [0.5] * (len(cands) - 1)))
        picks.append(None if p is None else tuple(np.array(cands)[p].round(1)))
    assert picks[0] == (42.7, 14.3)                    # nothing else there: hold it
    assert all(pk[1] == 0.0 for pk in picks[102:])     # then follow the moving ball
    assert sel.n_released >= 1


def test_selector_never_starts_on_a_still_ball():
    from footlytics.pipeline.radar import BallSelector

    sel = BallSelector(fps=25.0, reacquire_s=0.2)
    for f in range(60):                                # a decoy sits still for 2.4 s
        sel.pick(f, np.array([[42.7, 14.3]]), np.array([0.9]))
    sel.last_xy, sel.last_frame = None, -10**9         # fresh start (e.g. after a cut)
    p = sel.pick(60, np.array([[42.7, 14.3], [0.0, 0.0]]), np.array([0.9, 0.3]))
    assert p == 1


def test_a_still_ball_with_the_taker_beside_it_is_kept():
    """A match ball waiting for a restart is still, but not alone."""
    from footlytics.pipeline.radar import BallSelector

    sel = BallSelector(fps=25.0)
    ball = np.array([[42.7, 14.3]])
    taker, far = np.array([[43.5, 14.0]]), np.array([[0.0, 0.0]])
    picks_taker = [sel.pick(f, ball, np.array([0.9]), people_xy=taker) for f in range(100)]
    assert all(p == 0 for p in picks_taker)
    sel2 = BallSelector(fps=25.0)
    picks_alone = [sel2.pick(f, ball, np.array([0.9]), people_xy=far) for f in range(100)]
    assert picks_alone[0] == 0 and picks_alone[-1] is None


def test_touchline_people_are_taken_out_of_the_teams():
    """Measured on the 3-minute clip: a dozen tracklets sat on the touchlines, all
    labelled 'away' players -- an assistant referee running the line, ball boys and
    staff standing beside it."""
    from footlytics.perception.identity import touchline_people

    rows = []
    for f in range(50):
        rows += [
            dict(frame_idx=f, track_id=1, x=-10.0 + 0.4 * f, y=34.3),   # runs the touchline
            dict(frame_idx=f, track_id=2, x=10.0, y=-35.0),             # stands beside it
            dict(frame_idx=f, track_id=3, x=0.2 * f, y=33.8 if f < 5 else 20.0),  # throw-in, back
            dict(frame_idx=f, track_id=4, x=53.0, y=0.0),               # behind the goal line
        ]
    rows += [dict(frame_idx=f, track_id=5, x=0.0, y=-34.5) for f in range(10)]  # too short
    tracks = pd.DataFrame(rows)
    st = MatchState(MatchMeta(match_id="t", fps=25.0, home=TeamInfo("H"), away=TeamInfo("A")),
                    tracks.assign(period=1, timestamp=tracks.frame_idx / 25.0, role=PLAYER,
                                  team=Team.UNKNOWN.value))
    got = touchline_people(st.tracks, build_tracklets(st), 52.5, 34.0)
    assert got == {1: Role.REFEREE.value, 2: Role.OTHER.value, 4: Role.OTHER.value}
