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
            rows.append({"frame_idx": f, "period": 1, "timestamp": t, "track_id": tid,
                         "role": Role.PLAYER.value, "team": team.value, "x": x, "y": y})
        bx, by = _at(ball, f)
        rows.append({"frame_idx": f, "period": 1, "timestamp": t, "track_id": -1,
                     "role": Role.BALL.value, "team": Team.UNKNOWN.value, "x": bx, "y": by})
        poss.append({"frame_idx": f, "period": 1, "timestamp": t, "ballStatus": _at(status, f)})
    meta = MatchMeta(match_id="S", fps=FPS, home=TeamInfo("H", "H"), away=TeamInfo("A", "A"))
    return MatchState(meta, pd.DataFrame(rows)), pd.DataFrame(poss)


# Home sits deep and tight in its own half while away has the ball.
LOW_HOME = _shape(-42.0, -28.0, -12.0, 12.0)
AWAY_SPREAD = _shape(-20.0, 10.0, -30.0, 30.0)


def test_low_block_is_found_for_the_team_out_of_possession():
    st, poss = _match(LOW_HOME, AWAY_SPREAD, ball=_passing(-15.0), status="AWAY", seconds=12)
    seq = detect_sequences(st, poss)
    lb = seq[seq["label"] == "low_block"]
    assert len(lb) == 1
    row = lb.iloc[0]
    assert row["team"] == Team.HOME.value
    assert row["start_s"] <= 1.0 and row["end_s"] >= 11.0
    assert row["def_line_x"] <= -22.5          # evidence travels with the sequence


def test_no_low_block_while_the_ball_stands_still():
    """A set-piece free kick: defenders hold a compact line, but the ball is dead still."""
    st, poss = _match(LOW_HOME, AWAY_SPREAD, ball=(-15.0, 0.0), status="AWAY", seconds=12)
    assert (detect_sequences(st, poss)["label"] != "low_block").all()


def test_no_low_block_while_the_team_has_the_ball():
    st, poss = _match(LOW_HOME, AWAY_SPREAD, ball=(-15.0, 0.0), status="HOME", seconds=12)
    seq = detect_sequences(st, poss)
    assert (seq["label"] != "low_block").all() if len(seq) else True


def test_a_low_block_shorter_than_the_minimum_is_not_a_sequence():
    st, poss = _match(LOW_HOME, AWAY_SPREAD, ball=(-15.0, 0.0), status="AWAY", seconds=5)
    seq = detect_sequences(st, poss, low_block_min_s=8.0)
    assert len(seq) == 0


HP_PRESS = [(34.0, -4.0), (34.0, 4.0), (42.0, -5.0), (42.0, 5.0)]
HP_REST = [(20.0, -15.0), (20.0, 15.0), (10.0, -20.0), (10.0, -5.0), (10.0, 5.0), (10.0, 20.0)]
HP_AWAY_DEEP = _shape(25.0, 45.0, -25.0, 25.0)
_pressed_ball = lambda f: (38.0 + 2.0 * np.sin(f / FPS * 2.0), 3.0 * np.cos(f / FPS * 2.0))


def test_high_press_needs_bodies_around_the_ball_in_the_opponents_third():
    # Away keeps the ball deep in its own third (home's attacking third, x > +17.5);
    # four home players are within 10 m of the ball while play is live.
    st, poss = _match(_jog(HP_PRESS + HP_REST), _jog(HP_AWAY_DEEP), ball=_pressed_ball,
                      status="AWAY", seconds=6)
    seq = detect_sequences(st, poss)
    hp = seq[seq["label"] == "high_press"]
    assert len(hp) == 1 and hp.iloc[0]["team"] == Team.HOME.value
    assert hp.iloc[0]["n_within_10m"] >= 3


def test_no_high_press_at_a_dead_ball():
    """A throw-in or free kick in the opponent's third: bodies around a ball that does not move."""
    st, poss = _match(HP_PRESS + HP_REST, HP_AWAY_DEEP, ball=(38.0, 0.0), status="AWAY", seconds=6)
    assert (detect_sequences(st, poss)["label"] != "high_press").all()


def test_a_line_height_demand_only_counts_when_most_of_the_team_is_in_view():
    """Camera on the opponent's box: only the four forwards are visible, so their 'line' is high."""
    st, poss = _match(_jog(HP_PRESS), _jog(HP_AWAY_DEEP), ball=_pressed_ball,
                      status="AWAY", seconds=6)
    assert (detect_sequences(st, poss, high_press_line=7.5)["label"] != "high_press").all()
    assert (detect_sequences(st, poss)["label"] == "high_press").any()


def test_no_high_press_when_the_ball_is_in_midfield():
    press = [(-2.0, -4.0), (-2.0, 4.0), (6.0, -5.0), (6.0, 5.0)]
    rest = [(-10.0, -15.0), (-10.0, 15.0), (-20.0, -20.0), (-20.0, -5.0), (-20.0, 5.0), (-20.0, 20.0)]
    st, poss = _match(press + rest, _shape(-5.0, 30.0), ball=(2.0, 0.0), status="AWAY", seconds=6)
    seq = detect_sequences(st, poss)
    assert (seq["label"] != "high_press").all() if len(seq) else True


def test_sequence_columns():
    st, poss = _match(LOW_HOME, AWAY_SPREAD, ball=_passing(-15.0), status="AWAY", seconds=12)
    seq = detect_sequences(st, poss)
    assert {"label", "team", "period", "start_s", "end_s", "start_frame", "end_frame",
            "def_line_x", "compactness_m", "n_within_10m"} <= set(seq.columns)


# Home pushes five players into its right flank (y > 0 in its attacking frame) in
# the away half, against two away defenders there; everyone else stays central.
HOME_OVERLOAD = [(22.0, 20.0), (26.0, 28.0), (30.0, 16.0), (34.0, 24.0), (28.0, 31.0),
                 (5.0, -5.0), (5.0, 5.0), (-10.0, -8.0), (-10.0, 8.0), (-20.0, 0.0)]
AWAY_THIN_FLANK = [(30.0, 22.0), (35.0, 18.0), (35.0, -15.0), (35.0, -5.0), (35.0, 5.0),
                   (25.0, -8.0), (25.0, 2.0), (15.0, -3.0), (15.0, 6.0), (5.0, 0.0)]


def _flank_ball(sign=1.0):
    """The ball circulating on the attacker's right wing, well inside the flank channel."""
    return lambda f: (sign * (28.0 + 3.0 * np.sin(f / FPS * 2.0)), sign * (22.0 + 3.0 * np.cos(f / FPS * 2.0)))


def test_flank_overload_is_found_with_its_side_and_counts():
    st, poss = _match(_jog(HOME_OVERLOAD), _jog(AWAY_THIN_FLANK), ball=_flank_ball(),
                      status="HOME", seconds=6)
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
    st, poss = _match(_jog(mirror(AWAY_THIN_FLANK)), _jog(mirror(HOME_OVERLOAD)),
                      ball=_flank_ball(-1.0), status="AWAY", seconds=6)
    ov = detect_sequences(st, poss).query("label == 'flank_overload'")
    assert len(ov) == 1
    assert ov.iloc[0]["team"] == Team.AWAY.value and ov.iloc[0]["flank"] == "right"


def test_no_flank_overload_during_stoppage_or_out_of_possession():
    # Injury/foul stoppage: players stand still around a downed player on the wing.
    st_stop, poss_stop = _match(HOME_OVERLOAD, AWAY_THIN_FLANK, ball=(28.0, 25.0),
                                status="HOME", seconds=6)
    assert (detect_sequences(st_stop, poss_stop)["label"] != "flank_overload").all()

    # Pressing trap: opponent holds possession 100% of the window (not an attacking overload).
    st_oop, poss_oop = _match(_jog(HOME_OVERLOAD), _jog(AWAY_THIN_FLANK), ball=_flank_ball(),
                              status="AWAY", seconds=6)
    assert (detect_sequences(st_oop, poss_oop)["label"] != "flank_overload").all()


def test_no_flank_overload_while_the_ball_is_somewhere_else():
    """Five attackers stand in the channel, but play is in the centre of the pitch."""
    st, poss = _match(_jog(HOME_OVERLOAD), _jog(AWAY_THIN_FLANK), ball=_passing(28.0),
                      status="HOME", seconds=6)
    assert (detect_sequences(st, poss)["label"] != "flank_overload").all()


def test_no_flank_overload_while_the_ball_is_contested():
    st, poss = _match(_jog(HOME_OVERLOAD), _jog(AWAY_THIN_FLANK), ball=_flank_ball(),
                      status="CONTESTED", seconds=6)
    assert (detect_sequences(st, poss)["label"] != "flank_overload").all()


def test_flank_overload_counts_come_from_one_frame_not_the_maxima_of_the_run():
    # First 3 s: six attackers against three defenders in the channel. Last 3 s: four against none.
    # Maxima over the run would read 6 v 3; the widest single frame is 4 v 0.
    out = [(5.0, -5.0), (5.0, 5.0), (-10.0, 0.0), (-20.0, 0.0)]
    in_zone = [(22.0, 20.0), (26.0, 28.0), (30.0, 16.0), (34.0, 24.0), (28.0, 31.0), (24.0, 14.0)]
    home = lambda f: _jog(in_zone[:6] + out[:4] if f < 75 else in_zone[:4] + out)(f)
    away_in = [(30.0, 22.0), (35.0, 18.0), (25.0, 27.0)]
    away_out = [(35.0, -15.0), (35.0, -5.0), (35.0, 5.0), (25.0, -8.0), (25.0, 2.0), (15.0, -3.0), (15.0, 6.0)]
    away = lambda f: _jog(away_in + away_out if f < 75 else away_out + [(0.0, 0.0)] * 3 + [(2.0, 0.0)])(f)
    st, poss = _match(home, away, ball=_flank_ball(), status="HOME", seconds=6)
    ov = detect_sequences(st, poss).query("label == 'flank_overload'")
    assert len(ov) == 1
    assert (ov.iloc[0]["n_attackers"], ov.iloc[0]["n_defenders"]) == (4, 0)


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
    """Three home players run at the new ball carrier from 8 m away."""
    def home(f):
        if f < LOSS_F:
            return HOME_MID
        bx, by = _ball_mid(f)
        left = max(0.0, 1.0 - (f - LOSS_F) / (1.5 * FPS))
        chasers = [(bx + 2.0 + 6.0 * left, by - 2.0), (bx + 2.0 + 6.0 * left, by + 2.0),
                   (bx + 2.5 + 6.0 * left, by)]
        return HOME_MID[:7] + chasers
    seq = detect_sequences(*_loss(home)).query("label == 'midfield_loss'")
    assert len(seq) == 1
    row = seq.iloc[0]
    assert row["team"] == Team.HOME.value and row["response"] == "counter_press"
    assert row["n_pressing_ball"] == 3 and abs(row["start_s"] - 3.0) <= 0.5


def test_the_dispossessed_player_walking_beside_the_new_carrier_is_not_a_counter_press():
    ball = lambda f: (5.0 - 1.5 * f / FPS, 0.0)
    home = lambda f: HOME_MID[:9] + [(5.0 - 1.5 * f / FPS + 2.0, -1.0)]
    st, poss = _match(home, AWAY_MID, ball=ball, status=lambda f: "HOME" if f < LOSS_F else "AWAY", seconds=10)
    row = detect_sequences(st, poss).query("label == 'midfield_loss'").iloc[0]
    assert row["response"] != "counter_press" and row["n_pressing_ball"] == 0


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


def test_transitions_do_not_bridge_long_unknown_gaps_or_ballout():
    from footlytics.analytics.tactics import transitions

    def status_gap(f):
        if f < 50:
            return "HOME"
        if f < 130:                                    # 3.2 s UNKNOWN stoppage (> 2.0 s)
            return "UNKNOWN"
        return "AWAY"

    st, poss = _match(HOME_MID, AWAY_MID, ball=_ball_mid, status=status_gap, seconds=8)
    assert transitions(st, poss).empty

    def status_out(f):
        if f < 50:
            return "HOME"
        if f < 60:                                     # brief BALLOUT dead-ball restart (0.4 s)
            return "BALLOUT"
        return "AWAY"

    st2, poss2 = _match(HOME_MID, AWAY_MID, ball=_ball_mid, status=status_out, seconds=8)
    assert transitions(st2, poss2).empty


def test_no_build_up_when_team_centroid_is_pinned_in_own_box():
    pinned_home = _shape(-42.0, -28.0, -15.0, 15.0)
    st, poss = _match(_jog(pinned_home), _jog(AWAY_PRESS_3), ball=_passing(-35.0), status="HOME",
                      seconds=6)
    assert (detect_sequences(st, poss)["label"] != "build_up").all()


def test_no_midfield_loss_when_ball_stops_after_foul():
    def stopped_ball(f):
        if f < LOSS_F:
            return (2.0 * np.sin(f / FPS * 3.0), 2.0 * np.cos(f / FPS * 3.0))
        if f < LOSS_F + 25:
            return (-1.5 * (f - LOSS_F) / FPS, 0.0)
        return (-1.5, 0.0)
    st, poss = _match(HOME_MID, AWAY_MID, ball=stopped_ball,
                      status=lambda f: "HOME" if f < LOSS_F else "AWAY", seconds=10)
    from footlytics.analytics.kinematics import add_kinematics
    st = MatchState(st.meta, add_kinematics(st.tracks, FPS))
    assert (detect_sequences(st, poss)["label"] != "midfield_loss").all()


def test_low_block_survives_brief_box_scramble_and_bridges_short_gap():
    def status(f):
        # Away attacks home's box; a 2 s defender touch / UNKNOWN scramble in the middle
        # must not split a 12 s low block into two pieces under 8 s.
        if 100 <= f < 130:
            return "HOME"
        if 130 <= f < 150:
            return "UNKNOWN"
        return "AWAY"

    st, poss = _match(LOW_HOME, AWAY_SPREAD, ball=_passing(-35.0), status=status, seconds=12)
    lb = detect_sequences(st, poss).query("label == 'low_block'")
    assert len(lb) == 1 and lb.iloc[0]["team"] == Team.HOME.value


def test_build_up_works_when_broadcast_zoom_shows_four_outfielders():
    zoom_home = HOME_BUILD[3:7]   # 4 outfielders + goalkeeper in view (centroid >= -25 m)
    st, poss = _match(_jog(zoom_home), _jog(AWAY_PRESS_3), ball=_passing(-22.0), status="HOME",
                      seconds=6)
    bu = detect_sequences(st, poss).query("label == 'build_up'")
    assert len(bu) == 1 and bu.iloc[0]["team"] == Team.HOME.value


def test_no_low_block_on_set_piece_wall_with_occluded_or_jittering_ball():
    # Direct free-kick wall: 10 defenders stand shoulder-to-shoulder (compactness <= 6 m),
    # players are stopped waiting for the whistle, and the ball either jitters ~1.8 m in
    # homography projection or is occluded behind legs.
    wall_home = _shape(-38.0, -33.0, -6.0, 6.0)
    jitter_ball = lambda f: (-24.0 + 0.9 * np.sin(f * 0.7), 0.9 * np.cos(f * 0.9))
    st_jit, poss_jit = _match(wall_home, AWAY_SPREAD, ball=jitter_ball, status="AWAY", seconds=12)
    assert (detect_sequences(st_jit, poss_jit)["label"] != "low_block").all()

    # Same stationary defensive block with the ball completely occluded behind players.
    no_ball_tracks = st_jit.tracks[st_jit.tracks["role"] != Role.BALL.value].copy()
    st_occ = MatchState(st_jit.meta, no_ball_tracks)
    assert (detect_sequences(st_occ, poss_jit)["label"] != "low_block").all()


def test_build_up_separates_opponents_in_the_third_from_opponents_pressing_the_ball():
    """Three opponents stand in the third, 12 m from a ball that is going nowhere near them."""
    ball = lambda f: (-38.0 + np.sin(f / FPS * 2.0), np.cos(f / FPS * 2.0))
    st, poss = _match(_jog(HOME_BUILD), _jog(AWAY_PRESS_3), ball=ball, status="HOME", seconds=6)
    bu = detect_sequences(st, poss).query("label == 'build_up'")
    assert len(bu) == 1
    row = bu.iloc[0]
    assert row["n_opp_in_third"] == 3 and row["n_pressers"] == 3
    assert row["n_pressing_ball"] == 0


MID_HOME = _shape(-14.0, -4.0, -12.0, 12.0)


def test_mid_block_is_found_when_the_deepest_three_hold_the_middle_third():
    st, poss = _match(_jog(MID_HOME), _jog(AWAY_SPREAD), ball=_passing(-5.0), status="AWAY", seconds=12)
    seq = detect_sequences(st, poss)
    mb = seq[seq["label"] == "mid_block"]
    assert len(mb) == 1 and mb.iloc[0]["team"] == Team.HOME.value
    assert mb.iloc[0]["def_line3_x"] == pytest.approx(-14.0, abs=1.6)
    assert (seq["label"] != "low_block").all()


def test_a_deep_block_is_not_also_a_mid_block():
    st, poss = _match(_jog(LOW_HOME), _jog(AWAY_SPREAD), ball=_passing(-15.0), status="AWAY", seconds=12)
    assert (detect_sequences(st, poss)["label"] != "mid_block").all()


def test_no_mid_block_when_only_a_few_outfielders_are_in_view():
    st, poss = _match(_jog(MID_HOME[:5]), _jog(AWAY_SPREAD), ball=_passing(-5.0), status="AWAY", seconds=12)
    assert (detect_sequences(st, poss)["label"] != "mid_block").all()


def test_every_sequence_says_how_many_players_it_rests_on():
    st, poss = _match(_jog(LOW_HOME), _jog(AWAY_SPREAD), ball=_passing(-15.0), status="AWAY", seconds=12)
    row = detect_sequences(st, poss).query("label == 'low_block'").iloc[0]
    assert (row["n_visible"], row["n_visible_opp"]) == (11, 11)


def _loss_at(x, home_shape=HOME_MID, away_shape=AWAY_MID):
    """Home holds the ball at x for 3 s, away takes it and carries it 1.5 m/s towards -x."""
    ball = lambda f: ((x + 2.0 * np.sin(f / FPS * 3.0), 2.0 * np.cos(f / FPS * 3.0)) if f < LOSS_F
                      else (x - 1.5 * (f - LOSS_F) / FPS, 0.0))
    return _match(home_shape, away_shape, ball=ball, status=lambda f: "HOME" if f < LOSS_F else "AWAY",
                  seconds=10)


def test_turnovers_are_reported_in_every_third_with_their_zone():
    from footlytics.analytics.sequences import detect_turnovers
    for x, zone in ((-30.0, "own_third"), (0.0, "middle_third"), (30.0, "final_third")):
        t = detect_turnovers(*_loss_at(x))
        assert list(t["label"]) == ["turnover"] and list(t["zone"]) == [zone], x


def test_only_the_middle_third_turnovers_are_midfield_losses():
    labels = {x: list(detect_sequences(*_loss_at(x))["label"]) for x in (-30.0, 0.0, 30.0)}
    assert "midfield_loss" in labels[0.0]
    assert "midfield_loss" not in labels[-30.0] and "midfield_loss" not in labels[30.0]
    assert all("turnover" not in v for v in labels.values())


def test_a_midfield_loss_carries_the_winners_five_second_gain():
    row = detect_sequences(*_loss_at(0.0)).query("label == 'midfield_loss'").iloc[0]
    assert row["gain_5s_m"] == pytest.approx(7.3, abs=0.3)
    assert row["zone"] == "middle_third"


def test_a_team_standing_off_the_new_ball_carrier_is_not_counter_pressing():
    """A home player waits 5 m ahead of the carrier, not closing: the old second-nearest test called this a press."""
    def home(f):
        if f < LOSS_F:
            return HOME_MID
        bx, by = _ball_mid(f)
        return HOME_MID[:9] + [(bx - 5.0, by)]
    row = detect_sequences(*_loss(home)).query("label == 'midfield_loss'").iloc[0]
    assert row["response"] != "counter_press" and row["n_pressing_ball"] == 0


def test_the_response_is_unclear_when_too_few_players_stayed_in_view_to_judge_a_retreat():
    far = [(x - 25.0, y) for x, y in HOME_MID]

    def home(f):
        return far if f < LOSS_F + 50 else far[:2]
    row = detect_sequences(*_loss(home)).query("label == 'midfield_loss'").iloc[0]
    assert row["response"] == "unclear" and np.isnan(row["retreat_mps"])


def test_phase_shares_divide_phase_time_by_ball_in_play_time(tmp_path):
    import json

    from footlytics.analytics.sequences import phase_time_shares, write_sequence_outputs
    st, poss = _match(_jog(LOW_HOME), _jog(AWAY_SPREAD), ball=_passing(-15.0), status="AWAY", seconds=12)
    seq = detect_sequences(st, poss)
    shares = phase_time_shares(st, poss, seq)
    assert shares["ball_in_play_s"] == pytest.approx(12.0, abs=0.5)
    low = next(p for p in shares["phases"] if p["label"] == "low_block")
    assert low["team"] == "home" and low["share_of_ball_in_play"] == pytest.approx(1.0, abs=0.1)
    assert shares["possession_share"]["AWAY"] == pytest.approx(1.0)

    write_sequence_outputs(st, poss, tmp_path)
    assert json.loads((tmp_path / "phase_shares.json").read_text())["ball_in_play_s"] == shares["ball_in_play_s"]
    assert (tmp_path / "sequences.csv").is_file() and (tmp_path / "turnovers.csv").is_file()
