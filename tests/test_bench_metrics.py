"""Pipeline metrics against GSR ground truth, on synthetic Match States."""
import numpy as np
import pandas as pd
import pytest

from footlytics.bench.metrics import (ball_metrics, match_people, seconds_per_match_minute,
                                      shape_error, team_accuracy)


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


def test_team_accuracy_ignores_a_label_swap_but_not_a_wrong_side():
    gt = tracks(gt_frame(0))
    pred = tracks(people(0, LEFT_X, "away", 1) + people(0, RIGHT_X, "home", 101))
    m = match_people(gt, pred, [0])
    assert team_accuracy(gt, pred, m, {"away": "left", "home": "right"}) == 1.0
    assert team_accuracy(gt, pred, m, {"away": "right", "home": "left"}) == 0.0
    assert team_accuracy(gt, pred, m.iloc[:0], {}) is None


def test_ball_metrics_use_image_boxes_and_skip_interpolated_balls():
    gt = tracks([(f, 0, "ball", "unknown", 0, 0, 100, 100, 10, 10, False) for f in range(4)])
    pred = tracks([(0, 9, "ball", "unknown", 0, 0, 103, 104, 10, 10, False),   # 5 px: hit
                   (1, 9, "ball", "unknown", 0, 0, 160, 100, 10, 10, False),   # 60 px: miss
                   (2, 9, "ball", "unknown", 0, 0, np.nan, np.nan, np.nan, np.nan, True)])
    b = ball_metrics(gt, pred, range(4), hit_px=20.0)
    assert b["ball_recall"] == pytest.approx(1 / 4)
    assert b["ball_px_median"] == pytest.approx(np.median([5.0, 60.0]))


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
