"""Identity inside a continuous shot: scene-jump compensation, cut-aware stitching, keeper slots,
the per-segment identity report, and the capture/replay seam they are measured through."""
import numpy as np
import pandas as pd
import pytest

from footlytics.perception.identity import Tracklet, merge_same_role, stitch_tracklets
from footlytics.perception.track import PitchTracker, TrackerConfig
from footlytics.pipeline.quality import segment_identity
from footlytics.pipeline.radar import (Observation, PipelineConfig, homography_mapper,
                                       load_observations, run_frames, run_observations,
                                       save_observations)
from footlytics.state.schema import MatchMeta, Role

from .test_pipeline_frames import FakeDetector, H, frames


# ------------------------------------------------------------------ scene jump

def _scene(motion_comp: bool, jump_at: int = 20, n: int = 30):
    """Twelve players walking; from `jump_at` the calibration is off by a position-dependent
    1-2 m while the camera pans 40 px a frame (so the pictures stay consistent)."""
    rng = np.random.default_rng(1)
    p0 = np.c_[rng.uniform(-30, 30, 12), rng.uniform(-20, 20, 12)]
    v = rng.normal(0, 1.5, (12, 2))
    tr = PitchTracker(TrackerConfig(fps=25.0, motion_comp=motion_comp))
    ids = []
    for t in range(n):
        p = p0 + v * t / 25.0
        xy = p.copy()
        if t >= jump_at:
            xy += np.c_[-1.4 + 0.03 * p[:, 0], 1.5 * np.tanh(p[:, 0] / 10.0)]
        feet = np.c_[10 * p[:, 0] + 40 * t + 900, 10 * p[:, 1] + 500]
        boxes = np.c_[feet[:, 0] - 10, feet[:, 1] - 80, np.full(12, 20.0), np.full(12, 80.0)]
        m = dict(tr.update(xy, [Role.PLAYER.value] * 12, bboxes=boxes))
        ids.append([m[i] for i in range(12)])
    return np.array(ids)


def test_a_calibration_jump_breaks_every_track_unless_compensated():
    plain, comp = _scene(False), _scene(True)
    assert len(np.unique(plain)) > 12                                    # reborn at the jump
    assert len(np.unique(comp)) == 12                                    # one id each, throughout


def test_compensation_leaves_ordinary_motion_alone():
    tr = PitchTracker(TrackerConfig(fps=25.0, motion_comp=True))
    rng = np.random.default_rng(2)
    p0 = np.c_[rng.uniform(-30, 30, 12), rng.uniform(-20, 20, 12)]
    for t in range(40):
        xy = p0 + np.array([0.12, 0.0]) * t
        boxes = np.c_[10 * xy[:, 0] + 900, 10 * xy[:, 1] + 420, np.full(12, 20.0), np.full(12, 80.0)]
        tr.update(xy, [Role.PLAYER.value] * 12, bboxes=boxes)
    assert tr.n_comp == 0 and len(tr.tracks) == 12


# -------------------------------------------------------------------- stitching

def _tl(i, start, end, x0, x1, shot=None, vx=0.0):
    return Tracklet(id=i, track_ids=[i], start=start, end=end, start_xy=np.array([x0, 0.0]),
                    end_xy=np.array([x1, 0.0]), exit_velocity=np.array([vx, 0.0]),
                    n_frames=end - start + 1, shot=shot, shot_end=shot)


def test_stitching_does_not_cross_a_cut_when_told_not_to():
    a, b = _tl(1, 0, 100, 0, 5, shot=0), _tl(2, 150, 300, 6, 9, shot=1)
    assert len(stitch_tracklets([a, b], 25.0, across_shots=True)) == 1
    out = stitch_tracklets([_tl(1, 0, 100, 0, 5, shot=0), _tl(2, 150, 300, 6, 9, shot=1)],
                           25.0)
    assert len(out) == 2
    same = stitch_tracklets([_tl(1, 0, 100, 0, 5, shot=3), _tl(2, 150, 300, 6, 9, shot=3)],
                            25.0)
    assert len(same) == 1 and same[0].track_ids == [1, 2]


def test_a_long_gap_is_bridged_only_up_to_max_gap():
    a, b = _tl(1, 0, 100, 0, 5), _tl(2, 300, 400, 20, 25)            # 8 s apart
    assert len(stitch_tracklets([a, b], 25.0, max_gap_s=2.0)) == 2
    assert len(stitch_tracklets([_tl(1, 0, 100, 0, 5), _tl(2, 300, 400, 20, 25)], 25.0,
                                max_gap_s=10.0)) == 1


def test_exit_velocity_is_not_extrapolated_over_a_long_gap():
    """Running east at 8 m/s, then 6 s unseen: the fragment near where he was lost wins, not the
    one 48 m east that a full-length coast would land on."""
    a = _tl(1, 0, 100, 0, 5, vx=8.0)
    near, far = _tl(2, 250, 400, 7, 9), _tl(3, 250, 400, 53, 55)
    out = stitch_tracklets([a, near, far], 25.0, max_gap_s=10.0, coast_s=1.0)
    joined = next(t for t in out if 1 in t.track_ids)
    assert joined.track_ids == [1, 2]


def test_team_veto_refuses_opposite_kits():
    a, b = _tl(1, 0, 100, 0, 5), _tl(2, 110, 200, 6, 9)
    a.team_score, b.team_score = -0.6, 0.6
    assert len(stitch_tracklets([a, b], 25.0)) == 2
    a2, b2 = _tl(1, 0, 100, 0, 5), _tl(2, 110, 200, 6, 9)
    a2.team_score, b2.team_score = -0.6, 0.6
    assert len(stitch_tracklets([a2, b2], 25.0, team_veto=0.0)) == 1
    a3, b3 = _tl(1, 0, 100, 0, 5), _tl(2, 110, 200, 6, 9)
    a3.team_score, b3.team_score = -0.1, 0.1               # a doubtful lean is not a veto
    assert len(stitch_tracklets([a3, b3], 25.0)) == 1


# ----------------------------------------------------------------- keeper slots

def test_one_keeper_per_slot_across_cuts_but_overlap_means_two_people():
    k1, k2 = _tl(1, 0, 100, -50, -50), _tl(2, 500, 700, -49, -49)
    dup = _tl(3, 50, 150, -51, -51)                                  # overlaps k1: a second person
    other_end = _tl(4, 0, 100, 50, 50)
    p = _tl(5, 0, 100, -20, -20)
    slot = {1: (-1, "home"), 2: (-1, "home"), 3: (-1, "home"), 4: (1, "away")}
    out = merge_same_role([k1, k2, dup, other_end, p], slot)
    by_ids = sorted(sorted(t.track_ids) for t in out)
    assert by_ids == [[1, 2], [3], [4], [5]]


# --------------------------------------------------------------- segment report

def _rows(tid, frames_, shot, role=Role.PLAYER.value, live=True):
    return [dict(frame_idx=f, track_id=tid, role=role, shot=shot, live_wide=live) for f in frames_]


def test_segment_identity_counts_ids_per_visible_player_inside_each_shot():
    rows = []
    for k in range(10):                                  # shot 0: ten players, one id each, 5 s
        rows += _rows(k, range(0, 125), 0)
    rows += _rows(99, range(0, 125), 0, role=Role.GOALKEEPER.value)
    for k in range(10):                                  # shot 1: the same ten, fragmented in two
        rows += _rows(100 + k, range(200, 262), 1) + _rows(200 + k, range(262, 325), 1)
    rows += _rows(300, range(400, 500), 2, live=False)   # not live: ignored
    out = segment_identity(pd.DataFrame(rows), 25.0)
    assert out["live_segments"] == 2
    assert out["ids_per_segment_mean"] == pytest.approx((11 + 20) / 2)
    assert out["ids_per_player_mean"] == pytest.approx((1.0 * 5 + 2.0 * 5) / 10)
    assert out["keeper_ids_total"] == 1 and out["keeper_ids_per_segment_max"] == 1
    assert out["median_track_seconds"] == pytest.approx(2.5, abs=0.1)


def test_segment_identity_is_empty_without_shot_columns():
    assert segment_identity(pd.DataFrame(_rows(1, range(100), 0)).drop(columns=["shot"]), 25.0) == {}


# ---------------------------------------------------------- capture and replay

def test_replaying_a_capture_gives_the_same_tracks(tmp_path):
    n = 12
    cfg = dict(verbose=False, montage=False, stitch=False, touchline_people=False)
    captured: list[Observation] = []
    meta = MatchMeta(match_id="t", fps=25.0)
    state, report = run_frames(frames(n), 25.0, homography_mapper([H] * n), FakeDetector(n), meta,
                               PipelineConfig(**cfg), capture=captured)
    assert len(captured) == n and captured[0].desc is not None
    save_observations(tmp_path / "c.pkl", captured, 25.0, False)
    obs, fps, has_ref = load_observations(tmp_path / "c.pkl")
    again, rep2 = run_observations(obs, fps, MatchMeta(match_id="t", fps=25.0),
                                   PipelineConfig(**cfg), has_referee_class=has_ref)
    a, b = state.tracks.reset_index(drop=True), again.tracks.reset_index(drop=True)
    assert len(a) == len(b)
    assert (a["track_id"].to_numpy() == b["track_id"].to_numpy()).all()
    assert np.allclose(a[["x", "y"]].to_numpy(float), b[["x", "y"]].to_numpy(float))
    assert rep2["tracks_created"] == report["tracks_created"]


def test_one_frame_false_alarms_do_not_become_track_ids():
    rng = np.random.default_rng(3)
    obs = []
    for t in range(60):
        xy = np.c_[np.arange(5) * 8.0 - 16 + 0.05 * t, np.zeros(5)]
        dets = np.c_[1000 + 60 * np.arange(5), np.full(5, 400.0), np.full(5, 20.0), np.full(5, 60.0),
                     np.full(5, 0.9), np.zeros(5)]
        if t == 30:                                      # one doubled box that lasts a single frame
            xy = np.vstack([xy, [[0.0, 20.0]]])
            dets = np.vstack([dets, [1500, 400, 20, 60, 0.4, 0]])
        obs.append(Observation(dets, [Role.PLAYER.value] * len(dets), xy,
                               np.tile(np.eye(1, 4), (len(dets), 1))))
    for keep, expect in ((1, 6), (5, 5)):
        st, _ = run_observations(obs, 25.0, MatchMeta(match_id="t", fps=25.0),
                                 PipelineConfig(verbose=False, montage=False, stitch=False,
                                                touchline_people=False, goalkeepers=False,
                                                min_track_frames=keep))
        assert st.tracks["track_id"].nunique() == expect
