"""sequences.py: rule-based tactical sequences (low block, high press) over a normalised Match State."""
import numpy as np
import pandas as pd
import pytest

from footlytics.analytics.sequences import detect_sequences
from footlytics.state.schema import MatchMeta, MatchState, Role, Team, TeamInfo

FPS = 25.0


def _shape(x0, x1, y0=-20.0, y1=20.0, n=10):
    """n outfield positions spread over a rectangle, back to front."""
    xs = np.repeat(np.linspace(x0, x1, 3), 4)[:n]
    ys = np.tile(np.linspace(y0, y1, 4), 3)[:n]
    return list(zip(xs, ys))


def _at(v, f):
    return v(f) if callable(v) else v


def _match(home_shape, away_shape, ball, status, seconds):
    """A normalised state (home attacks +x) for `seconds`, plus its possession table.

    Shapes, ball and status are fixed values or functions of the frame index.
    Keepers stand on their goal lines so `goalkeepers()` finds them and the
    block metrics exclude them.
    """
    rows, poss = [], []
    for f in range(int(seconds * FPS)):
        t = f / FPS
        people = [(1, (-50.0, 0.0), Team.HOME)]
        people += [(2 + i, p, Team.HOME) for i, p in enumerate(_at(home_shape, f))]
        people += [(20, (50.0, 0.0), Team.AWAY)]
        people += [(21 + i, p, Team.AWAY) for i, p in enumerate(_at(away_shape, f))]
        for tid, (x, y), team in people:
            rows.append(dict(frame_idx=f, period=1, timestamp=t, track_id=tid,
                             role=Role.PLAYER.value, team=team.value, x=x, y=y))
        bx, by = _at(ball, f)
        rows.append(dict(frame_idx=f, period=1, timestamp=t, track_id=-1,
                         role=Role.BALL.value, team=Team.UNKNOWN.value, x=bx, y=by))
        poss.append(dict(frame_idx=f, period=1, timestamp=t, ballStatus=_at(status, f)))
    meta = MatchMeta(match_id="S", fps=FPS, home=TeamInfo("H", "H"), away=TeamInfo("A", "A"))
    return MatchState(meta, pd.DataFrame(rows)), pd.DataFrame(poss)


# Home sits deep and tight in its own half while away has the ball.
LOW_HOME = _shape(-42.0, -28.0, -12.0, 12.0)
AWAY_SPREAD = _shape(-20.0, 10.0, -30.0, 30.0)


def test_low_block_is_found_for_the_team_out_of_possession():
    st, poss = _match(LOW_HOME, AWAY_SPREAD, ball=(-15.0, 0.0), status="AWAY", seconds=12)
    seq = detect_sequences(st, poss)
    lb = seq[seq["label"] == "low_block"]
    assert len(lb) == 1
    row = lb.iloc[0]
    assert row["team"] == Team.HOME.value
    assert row["start_s"] <= 1.0 and row["end_s"] >= 11.0
    assert row["def_line_x"] <= -22.5          # evidence travels with the sequence


def test_no_low_block_while_the_team_has_the_ball():
    st, poss = _match(LOW_HOME, AWAY_SPREAD, ball=(-15.0, 0.0), status="HOME", seconds=12)
    seq = detect_sequences(st, poss)
    assert (seq["label"] != "low_block").all() if len(seq) else True


def test_a_low_block_shorter_than_the_minimum_is_not_a_sequence():
    st, poss = _match(LOW_HOME, AWAY_SPREAD, ball=(-15.0, 0.0), status="AWAY", seconds=5)
    seq = detect_sequences(st, poss, low_block_min_s=8.0)
    assert len(seq) == 0


def test_high_press_needs_bodies_around_the_ball_in_the_opponents_third():
    # Away keeps the ball deep in its own third (home's attacking third, x > +17.5);
    # home's back line has pushed up and four home players are within 10 m of the ball.
    ball = (38.0, 0.0)
    press = [(34.0, -4.0), (34.0, 4.0), (42.0, -5.0), (42.0, 5.0)]
    rest = [(20.0, -15.0), (20.0, 15.0), (10.0, -20.0), (10.0, -5.0), (10.0, 5.0), (10.0, 20.0)]
    away_deep = _shape(25.0, 45.0, -25.0, 25.0)
    st, poss = _match(press + rest, away_deep, ball=ball, status="AWAY", seconds=6)
    seq = detect_sequences(st, poss)
    hp = seq[seq["label"] == "high_press"]
    assert len(hp) == 1 and hp.iloc[0]["team"] == Team.HOME.value
    assert hp.iloc[0]["n_within_10m"] >= 3


def test_no_high_press_when_the_ball_is_in_midfield():
    press = [(-2.0, -4.0), (-2.0, 4.0), (6.0, -5.0), (6.0, 5.0)]
    rest = [(-10.0, -15.0), (-10.0, 15.0), (-20.0, -20.0), (-20.0, -5.0), (-20.0, 5.0), (-20.0, 20.0)]
    st, poss = _match(press + rest, _shape(-5.0, 30.0), ball=(2.0, 0.0), status="AWAY", seconds=6)
    seq = detect_sequences(st, poss)
    assert (seq["label"] != "high_press").all() if len(seq) else True


def test_sequence_columns():
    st, poss = _match(LOW_HOME, AWAY_SPREAD, ball=(-15.0, 0.0), status="AWAY", seconds=12)
    seq = detect_sequences(st, poss)
    assert {"label", "team", "period", "start_s", "end_s", "start_frame", "end_frame",
            "def_line_x", "compactness_m", "n_within_10m"} <= set(seq.columns)


# Home pushes five players into its right flank (y > 0 in its attacking frame) in
# the away half, against two away defenders there; everyone else stays central.
HOME_OVERLOAD = [(22.0, 20.0), (26.0, 28.0), (30.0, 16.0), (34.0, 24.0), (28.0, 31.0),
                 (5.0, -5.0), (5.0, 5.0), (-10.0, -8.0), (-10.0, 8.0), (-20.0, 0.0)]
AWAY_THIN_FLANK = [(30.0, 22.0), (35.0, 18.0), (35.0, -15.0), (35.0, -5.0), (35.0, 5.0),
                   (25.0, -8.0), (25.0, 2.0), (15.0, -3.0), (15.0, 6.0), (5.0, 0.0)]


def test_flank_overload_is_found_with_its_side_and_counts():
    st, poss = _match(HOME_OVERLOAD, AWAY_THIN_FLANK, ball=(28.0, 25.0), status="HOME", seconds=6)
    seq = detect_sequences(st, poss)
    ov = seq[seq["label"] == "flank_overload"]
    assert len(ov) == 1
    row = ov.iloc[0]
    assert row["team"] == Team.HOME.value and row["flank"] == "right"
    assert row["n_attackers"] == 5 and row["n_defenders"] == 2
    assert row["start_s"] <= 1.0 and row["end_s"] >= 5.0


def test_flank_overload_uses_each_teams_own_attacking_frame():
    # Mirror the whole picture: away now attacks -x, so its right flank is y < 0.
    mirror = lambda ps: [(-x, -y) for x, y in ps]
    st, poss = _match(mirror(AWAY_THIN_FLANK), mirror(HOME_OVERLOAD), ball=(-28.0, -25.0),
                      status="AWAY", seconds=6)
    ov = detect_sequences(st, poss).query("label == 'flank_overload'")
    assert len(ov) == 1
    assert ov.iloc[0]["team"] == Team.AWAY.value and ov.iloc[0]["flank"] == "right"


def test_no_flank_overload_when_the_defence_matches_the_numbers():
    away = [(30.0, 22.0), (35.0, 18.0), (25.0, 27.0), (32.0, 30.0)] + AWAY_THIN_FLANK[4:]
    st, poss = _match(HOME_OVERLOAD, away, ball=(28.0, 25.0), status="HOME", seconds=6)
    seq = detect_sequences(st, poss)
    assert (seq["label"] != "flank_overload").all() if len(seq) else True


def test_no_flank_overload_in_the_teams_own_half():
    home = [(-x, y) for x, y in HOME_OVERLOAD]
    away = [(-x, y) for x, y in AWAY_THIN_FLANK]
    st, poss = _match(home, away, ball=(-28.0, 25.0), status="HOME", seconds=6)
    ov = detect_sequences(st, poss).query("label == 'flank_overload'")
    assert (ov["team"] != Team.HOME.value).all()


def test_no_flank_overload_when_the_camera_shows_too_few_defenders():
    # Broadcast view: only a handful of away players are on screen, so "two
    # defenders on the flank" may just be the ones the camera caught.
    st, poss = _match(HOME_OVERLOAD, AWAY_THIN_FLANK[:4], ball=(28.0, 25.0), status="HOME", seconds=6)
    seq = detect_sequences(st, poss)
    assert (seq["label"] != "flank_overload").all() if len(seq) else True


def test_a_brief_flank_overload_is_not_a_sequence():
    st, poss = _match(HOME_OVERLOAD, AWAY_THIN_FLANK, ball=(28.0, 25.0), status="HOME", seconds=2)
    seq = detect_sequences(st, poss, overload_min_s=3.0)
    assert (seq["label"] != "flank_overload").all() if len(seq) else True


def test_substitutes_on_the_touchline_are_not_attackers():
    # A stoppage: three of the five "attackers" stand just off the pitch (y > 34 m).
    home = [(22.0, 20.0), (26.0, 34.5), (30.0, 35.0), (34.0, 34.8)] + HOME_OVERLOAD[4:]
    st, poss = _match(home, AWAY_THIN_FLANK, ball=(28.0, 25.0), status="HOME", seconds=6)
    seq = detect_sequences(st, poss, overload_min_visible=5)
    assert (seq["label"] != "flank_overload").all() if len(seq) else True


# Build-up: home plays it around its own third (x < -17.5) while three away players press there.
HOME_BUILD = [(-40.0, -20.0), (-40.0, 20.0), (-35.0, -8.0), (-35.0, 8.0), (-15.0, -25.0),
              (-15.0, 25.0), (-12.0, 0.0), (5.0, -10.0), (5.0, 10.0), (15.0, 0.0)]
AWAY_PRESS_3 = [(-30.0, -10.0), (-30.0, 10.0), (-25.0, 0.0), (-5.0, -20.0), (-5.0, 0.0),
                (-5.0, 20.0), (10.0, -15.0), (10.0, 15.0), (20.0, -5.0), (20.0, 5.0)]


def _jog(shape):
    """Every player circles his spot at about 2 m/s: play is on."""
    return lambda f: [(x + 1.5 * np.sin(f / FPS * 1.5 + i), y + 1.5 * np.cos(f / FPS * 1.5 + i))
                      for i, (x, y) in enumerate(shape)]


def _passing(x0, dx=4.0):
    """The ball moving between defenders: it never stands still for a second."""
    return lambda f: (x0 + dx * np.sin(f / FPS * 2.0), 10.0 * np.cos(f / FPS * 2.0))


def test_build_up_is_found_with_the_number_of_pressers():
    st, poss = _match(_jog(HOME_BUILD), _jog(AWAY_PRESS_3), ball=_passing(-35.0), status="HOME",
                      seconds=6)
    bu = detect_sequences(st, poss).query("label == 'build_up'")
    assert len(bu) == 1
    assert bu.iloc[0]["team"] == Team.HOME.value and bu.iloc[0]["n_pressers"] == 3


def test_build_up_uses_each_teams_own_third():
    """Mirror the pitch: away builds up at +x and home presses there."""
    mirror = lambda shape: [(-x, -y) for x, y in shape]
    st, poss = _match(_jog(mirror(AWAY_PRESS_3)), _jog(mirror(HOME_BUILD)), ball=_passing(35.0),
                      status="AWAY", seconds=6)
    bu = detect_sequences(st, poss).query("label == 'build_up'")
    assert len(bu) == 1
    assert bu.iloc[0]["team"] == Team.AWAY.value and bu.iloc[0]["n_pressers"] == 3


def test_no_build_up_while_the_ball_stands_still():
    """A goal kick being placed: the players move but the ball does not."""
    st, poss = _match(_jog(HOME_BUILD), _jog(AWAY_PRESS_3), ball=(-35.0, 0.0), status="HOME",
                      seconds=6)
    assert (detect_sequences(st, poss)["label"] != "build_up").all()


def test_no_build_up_while_the_players_have_stopped():
    """A foul: the ball rolls on, but every player stands."""
    st, poss = _match(HOME_BUILD, AWAY_PRESS_3, ball=_passing(-35.0), status="HOME", seconds=6)
    assert (detect_sequences(st, poss)["label"] != "build_up").all()


def test_no_build_up_once_the_ball_leaves_the_own_third():
    st, poss = _match(_jog(HOME_BUILD), _jog(AWAY_PRESS_3), ball=_passing(-5.0), status="HOME",
                      seconds=6)
    assert (detect_sequences(st, poss)["label"] != "build_up").all()


# Midfield loss: home has the ball around the halfway line for 3 s, then away wins it there.
LOSS_F = int(3 * FPS)
HOME_MID = [(-20.0, -20.0), (-20.0, 0.0), (-20.0, 20.0), (-5.0, -15.0), (-5.0, 15.0),
            (2.0, -6.0), (2.0, 6.0), (10.0, -20.0), (10.0, 20.0), (15.0, 0.0)]
AWAY_MID = [(30.0, -20.0), (30.0, 0.0), (30.0, 20.0), (12.0, -10.0), (12.0, 10.0),
            (5.0, 0.0), (0.0, -25.0), (0.0, 25.0), (-10.0, -8.0), (-10.0, 8.0)]


def _ball_mid(f):
    """Home moves it about the centre spot; after the loss away carries it slowly forward (-x)."""
    if f < LOSS_F:
        return (2.0 * np.sin(f / FPS * 3.0), 2.0 * np.cos(f / FPS * 3.0))
    return (-1.5 * (f - LOSS_F) / FPS, 0.0)


def _loss(home_shape):
    status = lambda f: "HOME" if f < LOSS_F else "AWAY"
    return _match(home_shape, AWAY_MID, ball=_ball_mid, status=status, seconds=10)


def test_midfield_loss_followed_by_a_counter_press():
    """Three home players close the new ball carrier down at once."""
    def home(f):
        if f < LOSS_F:
            return HOME_MID
        bx, by = _ball_mid(f)
        chasers = [(bx + 2.0, by - 2.0), (bx + 2.0, by + 2.0), (bx + 3.5, by)]
        return HOME_MID[:7] + chasers
    seq = detect_sequences(*_loss(home)).query("label == 'midfield_loss'")
    assert len(seq) == 1
    row = seq.iloc[0]
    assert row["team"] == Team.HOME.value and row["response"] == "counter_press"
    assert row["ball_dist_m"] <= 5.0 and abs(row["start_s"] - 3.0) <= 0.5


def test_midfield_loss_followed_by_a_retreat():
    """Nobody goes to the ball; the whole block drops 2 m every second."""
    far = [(x - 25.0, y) for x, y in HOME_MID]
    def home(f):
        drop = 2.0 * max(0.0, (f - LOSS_F) / FPS)
        return far if f < LOSS_F else [(x - drop, y) for x, y in far]
    seq = detect_sequences(*_loss(home)).query("label == 'midfield_loss'")
    assert len(seq) == 1
    row = seq.iloc[0]
    assert row["response"] == "retreat" and row["retreat_mps"] == pytest.approx(-2.0, abs=0.3)


def test_a_loss_in_the_own_third_is_not_a_midfield_loss():
    status = lambda f: "HOME" if f < LOSS_F else "AWAY"
    deep = lambda f: (-35.0 + np.sin(f / FPS * 3.0), np.cos(f / FPS * 3.0))
    st, poss = _match(HOME_BUILD, AWAY_PRESS_3, ball=deep, status=status, seconds=10)
    assert (detect_sequences(st, poss)["label"] != "midfield_loss").all()
