"""Trajectory linking of ball candidates, its pipeline flag, and the ball-loss diagnostic."""
import numpy as np
import pandas as pd
import pytest

from footlytics.analytics.ball_link import link_ball_candidates
from footlytics.bench.metrics import ball_loss, ball_loss_summary
from footlytics.pipeline.radar import (BallSelector, PipelineConfig, homography_mapper,
                                       run_frames)
from footlytics.state.schema import MatchMeta, Role
from tests.test_pipeline_frames import FakeDetector, H, frames

FPS = 25.0


def arr(*rows):
    return np.array(rows, float).reshape(-1, 2)


def test_empty_input_and_frames_without_candidates():
    assert link_ball_candidates([], [], [], FPS) == []
    assert link_ball_candidates([0, 1], [arr(), arr()], [np.empty(0), np.empty(0)], FPS) == [None, None]


def test_a_faint_ball_that_keeps_moving_beats_a_bright_one_that_appears_first():
    # frame 0: a bright false positive (a boot, 20 m away) and the faint true ball; afterwards
    # only the true ball, drifting 0.3 m a frame. The greedy selector takes the bright one,
    # finds the true ball unreachable, and loses a second; the path follows the true one from
    # the second frame (the first frame's bright box costs one frame, not a second).
    n = 30
    xy = [arr([20, 0], [0, 0])] + [arr([0.3 * t, 0]) for t in range(1, n)]
    conf = [np.array([0.6, 0.15])] + [np.array([0.15])] * (n - 1)
    frames_ = list(range(n))

    picks = link_ball_candidates(frames_, xy, conf, FPS)
    assert picks[1:] == [0] * (n - 1)

    sel = BallSelector(FPS)
    greedy = [sel.pick(f, x, c) for f, x, c in zip(frames_, xy, conf)]
    assert greedy[0] == 0 and sum(g is None for g in greedy) >= 8


def test_a_lone_bright_detection_that_would_break_the_track_twice_is_ignored():
    n = 20
    xy = [arr([0.3 * t, 0]) for t in range(n)]
    conf = [np.array([0.2])] * n
    xy[10] = arr([0.3 * 10, 0], [25, 10])
    conf[10] = np.array([0.2, 0.4])
    picks = link_ball_candidates(list(range(n)), xy, conf, FPS)
    assert picks[10] == 0 and all(p == 0 for p in picks if p is not None)


def test_gaps_within_the_limit_keep_the_track_and_longer_ones_start_a_new_one():
    xy = [arr([0, 0]), arr([0.3, 0]), arr([30, 0])]
    conf = [np.array([0.3])] * 3
    # a long silence before the third: it is a new sighting, not a continuation, but still taken
    assert link_ball_candidates([0, 1, 100], xy, conf, FPS, break_cost=0.1) == [0, 0, 0]
    # right after the second it is 30 m in 1 frame: unreachable, a break costing more than it earns
    assert link_ball_candidates([0, 1, 2], xy, conf, FPS, break_cost=0.5) == [0, 0, None]


def test_decoys_cannot_be_picked_and_a_still_ball_loses_to_a_moving_one():
    xy = [arr([0, 0], [10, 0])] * 3
    conf = [np.array([0.5, 0.45])] * 3
    f = [0, 1, 2]
    assert link_ball_candidates(f, xy, conf, FPS, penalty=[np.array([1.0, 0.0])] * 3) == [1, 1, 1]
    assert link_ball_candidates(f, xy, conf, FPS, penalty=[np.array([0.5, 0.0])] * 3) == [1, 1, 1]
    assert link_ball_candidates(f, xy, conf, FPS) == [0, 0, 0]
    # nothing else moves: a held still ball stays
    assert link_ball_candidates(f, [arr([0, 0])] * 3, [np.array([0.5])] * 3, FPS,
                                penalty=[np.array([0.5])] * 3) == [0, 0, 0]


def test_one_pick_per_frame_and_frames_may_be_skipped():
    xy = [arr([0, 0]), arr(), arr([0.6, 0])]
    conf = [np.array([0.3]), np.empty(0), np.array([0.3])]
    assert link_ball_candidates([0, 1, 2], xy, conf, FPS) == [0, None, 0]


# ----------------------------------------------------------------------------- pipeline

def _run(n=12, **kw):
    cfg = PipelineConfig(verbose=False, montage=False, use_appearance=False, stitch=False,
                         touchline_people=False, ball_max_gap=0, **kw)
    return run_frames(frames(n), FPS, homography_mapper([H] * n), FakeDetector(n),
                      MatchMeta(match_id="t", fps=FPS), cfg)


def test_ball_link_is_off_by_default():
    assert PipelineConfig().ball_link is False


def test_ball_link_writes_the_ball_without_the_tracker_and_leaves_people_alone():
    n = 12
    base, _ = _run(n)
    state, report = _run(n, ball_link=True)
    ball = state.tracks[state.tracks["role"] == Role.BALL.value]
    assert sorted(ball["frame_idx"]) == list(range(n))
    assert ball["track_id"].nunique() == 1 and report["ball_link_segments"] == 1
    assert report["ball_link_frames"] == n
    assert ball["det_conf"].eq(0.5).all() and ball["bbox_w"].eq(6).all()
    # the same positions as the greedy path gives
    ref = base.tracks[base.tracks["role"] == Role.BALL.value].sort_values("frame_idx")
    np.testing.assert_allclose(ball.sort_values("frame_idx")[["x", "y"]], ref[["x", "y"]])
    people = state.tracks[state.tracks["role"] != Role.BALL.value]
    assert people["track_id"].nunique() == 3
    assert not set(ball["track_id"]) & set(people["track_id"])


def test_keep_ball_candidates_records_raw_boxes_and_whether_they_survived():
    n = 6
    Hs = [H if t != 2 else None for t in range(n)]       # frame 2 is uncalibrated
    cfg = PipelineConfig(verbose=False, montage=False, use_appearance=False, stitch=False,
                         touchline_people=False, keep_ball_candidates=True)
    _, report = run_frames(frames(n), FPS, homography_mapper(Hs), FakeDetector(n),
                           MatchMeta(match_id="t", fps=FPS), cfg)
    cands = report["_ball_candidates"]
    assert sorted(cands) == list(range(n))
    boxes, ok = cands[0]
    assert boxes.shape == (1, 5) and bool(ok[0])
    assert not cands[2][1][0]


# ----------------------------------------------------------------------------- bench metric

def _gt(frames_xy):
    rows = [dict(frame_idx=f, role="ball", interpolated=False, bbox_x=x - 3, bbox_y=y - 3,
                 bbox_w=6.0, bbox_h=6.0) for f, (x, y) in frames_xy.items()]
    return pd.DataFrame(rows)


def test_ball_loss_separates_detector_misses_from_filtered_candidates():
    gt = _gt({0: (100, 100), 1: (100, 100), 2: (100, 100), 3: (100, 100)})
    box = lambda x, y: np.array([[x - 3, y - 3, 6, 6, 0.4]])
    cands = {
        0: (box(105, 100), np.array([True])),     # found and kept
        1: (box(105, 100), np.array([False])),    # found, then dropped (off pitch / uncalibrated)
        2: (box(300, 300), np.array([True])),     # only a wrong object
        # frame 3: nothing detected
    }
    c = ball_loss(gt, cands, range(4))["_ball_loss"]
    assert (c["gt"], c["detected"], c["on_pitch"]) == (4, 2, 1)
    assert ball_loss_summary(c) == {"ball_det_recall": 0.5, "ball_cand_recall": 0.25}
    assert ball_loss_summary(ball_loss(gt, {}, range(4))["_ball_loss"])["ball_det_recall"] == 0.0
    assert ball_loss_summary(None)["ball_det_recall"] is None
