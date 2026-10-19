"""run_frames: the pipeline over an iterator of frames with a per-frame pitch mapper."""
import numpy as np
import pytest

from footlytics.pipeline.radar import PipelineConfig, homography_mapper, run_frames
from footlytics.state.schema import MatchMeta, Role


class FakeDetector:
    """Three people standing still and a ball that moves; one class map like the x6 weights."""
    roles = {0: Role.PLAYER.value, 1: Role.BALL.value}

    def __init__(self, n_frames):
        self.t = 0

    def detect(self, frame):
        t = int(frame[0, 0, 0])
        people = [[100 + 60 * k, 300, 20, 60, 0.9, 0] for k in range(3)]
        ball = [[200 + 2 * t, 340, 6, 6, 0.5, 1]]
        return np.array(people + ball, float)

    def role_of(self, c):
        return self.roles[int(c)]


def frames(n, h=480, w=640):
    out = []
    for t in range(n):
        f = np.full((h, w, 3), 90, np.uint8)
        f[0, 0] = t                      # all channels: the pipeline hands the detector RGB
        out.append(f)
    return out


# image (px) -> pitch (m): 10 cm per pixel, origin at image (320, 240)
H = np.array([[0.1, 0, -32.0], [0, 0.1, -24.0], [0, 0, 1.0]])


def test_homography_mapper_maps_feet_and_handles_gaps():
    m = homography_mapper([H, None])
    dets = np.array([[100, 300, 20, 60, 0.9, 0]], float)
    assert m(0, None, dets) == pytest.approx(np.array([[0.1 * 110 - 32, 0.1 * 360 - 24]]))
    assert m(0, None, dets[:0]).shape == (0, 2)
    assert m(1, None, dets) is None and m(5, None, dets) is None


def test_run_frames_tracks_people_and_ball_with_a_homography_per_frame():
    n = 12
    state, report = run_frames(frames(n), 25.0, homography_mapper([H] * n), FakeDetector(n),
                               MatchMeta(match_id="t", fps=25.0),
                               PipelineConfig(verbose=False, montage=False, use_appearance=False,
                                              stitch=False, touchline_people=False))
    assert report["frames_processed"] == n and report["frames_calibrated"] == n
    assert report["calib_success_rate"] == 1.0 and "_montage" not in report
    people = state.tracks[state.tracks["role"] != Role.BALL.value]
    assert people["track_id"].nunique() == 3          # three steady tracks
    assert set(state.tracks["frame_idx"]) == set(range(n))


def test_uncalibrated_frames_lose_their_detections_but_the_run_goes_on():
    n = 12
    Hs = [H if t % 3 else None for t in range(n)]         # every third frame uncalibrated
    state, report = run_frames(frames(n), 25.0, homography_mapper(Hs), FakeDetector(n),
                               MatchMeta(match_id="t", fps=25.0),
                               PipelineConfig(verbose=False, montage=False, use_appearance=False,
                                              stitch=False, touchline_people=False,
                                              ball_max_gap=0))
    assert report["frames_uncalibrated"] == 4 and report["frames_calibrated"] == 8
    seen = set(state.tracks.loc[~state.tracks["interpolated"], "frame_idx"])
    assert not seen & {0, 3, 6, 9}


def _broadcast_frames(n_wide, n_other, h=480, w=640):
    out = frames(n_wide + n_other, h, w)
    for t, f in enumerate(out):
        if t < n_wide:
            f[1:, :] = (60, 140, 50)                      # grass, so the shot filter calls it wide
    return out


def test_broadcast_run_flags_frames_that_are_not_live_wide_and_keeps_their_rows():
    n = 150
    cfg = PipelineConfig(verbose=False, montage=False, use_appearance=False, stitch=False,
                         touchline_people=False)
    state, report = run_frames(_broadcast_frames(75, 75), 25.0, homography_mapper([H] * n),
                               FakeDetector(n), MatchMeta(match_id="t", fps=25.0, source_type="broadcast"), cfg)
    per = state.tracks.groupby("frame_idx")[["live_wide", "shot"]].first()
    assert per.loc[:74, "live_wide"].all() and not per.loc[75:, "live_wide"].any()
    assert per.loc[0, "shot"] != per.loc[149, "shot"]
    assert report["live_wide_share"] == pytest.approx(0.5) and report["replays"] == 0


def test_fixed_rig_runs_are_untouched_by_the_shot_filter():
    n = 12
    state, report = run_frames(_broadcast_frames(0, n), 25.0, homography_mapper([H] * n), FakeDetector(n),
                               MatchMeta(match_id="t", fps=25.0, source_type="fixed_panoramic"),
                               PipelineConfig(verbose=False, montage=False, use_appearance=False,
                                              stitch=False, touchline_people=False))
    assert state.tracks["live_wide"].all() and "live_wide_share" not in report
