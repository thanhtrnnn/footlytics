"""Pipeline metrics against GSR ground truth, on synthetic Match States."""
import numpy as np
import pandas as pd
import pytest

from footlytics.bench.metrics import (ball_metrics, match_people, position_metrics,
                                      position_summary, seconds_per_match_minute, shape_error,
                                      team_accuracy)


def tracks(rows):
    cols = ["frame_idx", "track_id", "role", "team", "x", "y", "bbox_x", "bbox_y", "bbox_w",
            "bbox_h", "interpolated"]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])


def people(frame, xs, team, start_id, dx=0.0):
    return [(frame, start_id + i, "player", team, x + dx, 5.0 * i, 0, 0, 10, 20, False)
            for i, x in enumerate(xs)]


LEFT_X, RIGHT_X = [-30, -25, -20, -15, -10, -5], [5, 10, 15, 20, 25, 30]


def gt_frame(f):
    return people(f, LEFT_X, "home", 1) + people(f, RIGHT_X, "away", 101)


def test_match_people_pairs_within_the_distance():
    gt = tracks(gt_frame(0))
    pred = tracks(people(0, LEFT_X, "away", 1, dx=0.5) + people(0, RIGHT_X, "home", 101, dx=9))
    m = match_people(gt, pred, [0], max_dist=2.0)
    assert len(m) == 6 and (m["dist"] <= 2.0).all()


def test_the_gate_drops_pairs_but_never_changes_them():
    """The assignment ignores the gate, so the 2 m pairs are exactly the 5 m pairs within 2 m:
    team accuracy reads them off the one 5 m match."""
    gt = tracks(gt_frame(0))
    pred = tracks(people(0, LEFT_X, "home", 1, dx=3.0) + people(0, RIGHT_X, "away", 101, dx=0.5))
    near = match_people(gt, pred, [0], max_dist=5.0)
    close = match_people(gt, pred, [0], max_dist=2.0)
    assert sorted(near["dist"]) == [0.5] * 6 + [3.0] * 6 and len(close) == 6
    pd.testing.assert_frame_equal(near[near["dist"] <= 2.0].reset_index(drop=True), close)


def test_position_summary_median_p90_and_recall_over_gt_people():
    assert position_summary([0.5, 1.0, 1.5, 3.0, 4.5], 6) == {
        "pos_err_median_m": 1.5, "pos_err_p90_m": 3.9, "recall_2m": 0.5}
    assert position_summary([], 4) == {
        "pos_err_median_m": None, "pos_err_p90_m": None, "recall_2m": 0.0}
    assert position_summary([], 0) == {
        "pos_err_median_m": None, "pos_err_p90_m": None, "recall_2m": None}


def test_position_metrics_charge_every_located_gt_player_in_the_scored_frames():
    """Frame 1 predicts no left player: its six are misses. A referee, a player with no pitch
    position and a frame outside the scored ones are not GT people."""
    nan = np.nan
    gt = tracks(gt_frame(0) + gt_frame(1) + gt_frame(2)
                + [(0, 50, "referee", "official", 0.0, 30.0, 0, 0, 10, 20, False),
                   (1, 51, "player", "home", nan, nan, 0, 0, 10, 20, False)])
    pred = tracks(people(0, LEFT_X, "home", 1, dx=3.0) + people(0, RIGHT_X, "away", 101, dx=0.5)
                  + people(1, RIGHT_X, "away", 101, dx=0.5))
    p = position_metrics(gt, match_people(gt, pred, [0, 1], max_dist=5.0), [0, 1])
    assert p["_gt_people"] == 24 and sorted(p["_pos_dists"]) == [0.5] * 12 + [3.0] * 6
    assert (p["pos_err_median_m"], p["pos_err_p90_m"], p["recall_2m"]) == (0.5, 3.0, 0.5)


def test_team_accuracy_ignores_a_label_swap_but_not_a_wrong_side():
    gt = tracks(gt_frame(0))
    pred = tracks(people(0, LEFT_X, "away", 1) + people(0, RIGHT_X, "home", 101))
    m = match_people(gt, pred, [0])
    assert team_accuracy(gt, pred, m, {"away": "left", "home": "right"}) == 1.0
    assert team_accuracy(gt, pred, m, {"away": "right", "home": "left"}) == 0.0
    assert team_accuracy(gt, pred, m.iloc[:0], {}) is None


def ball(f, dx, dy, interpolated=False):
    """A predicted ball box `dx`, `dy` px from the GT box at (100, 100) (centre (105, 105))."""
    nan = np.nan
    return (f, 9, "ball", "unknown", 0, 0,
            nan if interpolated else 100 + dx, nan if interpolated else 100 + dy,
            nan if interpolated else 10, nan if interpolated else 10, interpolated)


def test_ball_metrics_split_hits_from_a_wrong_object_a_missing_box_and_an_interpolated_ball():
    gt = tracks([(f, 0, "ball", "unknown", 0, 0, 100, 100, 10, 10, False) for f in range(6)])
    pred = tracks([ball(0, 3, 4),                     # 5 px: hit
                   ball(1, 3, 0),                     # 3 px: hit
                   ball(2, 60, 0),                    # 60 px: a box, on the wrong object
                   ball(3, 0, 0, interpolated=True),  # no box: neither a hit nor a wrong object
                   ball(5, 25, 0)])                   # 25 px: just over hit_px, wrong
    # frame 4 has no predicted ball at all
    b = ball_metrics(gt, pred, range(6), hit_px=20.0)
    assert b["ball_recall"] == pytest.approx(2 / 6, abs=1e-4)
    assert b["ball_px_median_hits"] == pytest.approx(4.0)       # median of 5 and 3, hits only
    assert b["ball_wrong_share"] == pytest.approx(2 / 6, abs=1e-4)
    assert "ball_px_median" not in b
    assert sorted(b["_dists"]) == pytest.approx([3.0, 5.0, 25.0, 60.0])  # raw, for pooling
    assert b["_gt_frames"] == 6


def test_ball_metrics_without_gt_balls_or_hits_have_no_median():
    gt = tracks([(f, 0, "ball", "unknown", 0, 0, 100, 100, 10, 10, False) for f in range(2)])
    wrong = tracks([ball(0, 90, 0), ball(1, 90, 0)])
    b = ball_metrics(gt, wrong, range(2))
    assert b["ball_recall"] == 0.0 and b["ball_wrong_share"] == 1.0
    assert b["ball_px_median_hits"] is None
    none = ball_metrics(tracks(people(0, LEFT_X, "home", 1)), wrong, [0])
    assert none["ball_recall"] is None and none["ball_wrong_share"] is None
    assert none["ball_px_median_hits"] is None and none["_dists"] == [] and none["_gt_frames"] == 0


def test_shape_error_is_zero_for_identical_shapes_under_a_label_swap():
    gt = tracks(gt_frame(0) + gt_frame(1))
    pred = tracks(people(0, LEFT_X, "away", 1) + people(0, RIGHT_X, "home", 101)
                  + people(1, LEFT_X, "away", 1) + people(1, RIGHT_X, "home", 101))
    e = shape_error(gt, pred, [0, 1], {"away": "left", "home": "right"})
    assert e["shape_error_m"] == pytest.approx(0.0)
    shifted = pred.copy()
    shifted["x"] = shifted["x"] + 2.0
    assert shape_error(gt, shifted, [0, 1], {"away": "left", "home": "right"})["centroid_x_err"] \
        == pytest.approx(2.0)


def test_shape_error_without_enough_players_is_none():
    gt = tracks(people(0, LEFT_X[:3], "home", 1))
    assert shape_error(gt, gt, [0], {"home": "left"})["shape_error_m"] is None


def test_seconds_per_match_minute():
    assert seconds_per_match_minute(60.0, 750, 25.0) == pytest.approx(120.0)
