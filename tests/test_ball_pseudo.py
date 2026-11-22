import numpy as np

from footlytics.perception.ball_pseudo import TrackConfig, build_tracks, label_rows


def _frames(n, fn):
    return [np.array(fn(t), float).reshape(-1, 6) for t in range(n)]


def test_smooth_ball_becomes_a_track_and_noise_does_not():
    # ball moves 10 px/frame; a one-frame false positive and a still false blob elsewhere
    def f(t):
        rows = [[100 + 10 * t, 200 + 2 * t, 14, 14, 0.9, 1]]
        if t == 5:
            rows.append([900, 600, 14, 14, 0.95, 1])
        if t % 2 == 0:
            rows.append([1500, 100, 14, 14, 0.3, 1])
        return rows
    dets = _frames(30, f)
    tr = build_tracks(dets)
    assert len(tr) == 1 and len(tr[0].frames) == 30
    lab = label_rows(tr, dets)
    assert set(lab) == set(range(30))
    assert lab[5]["rivals"] == 1 and lab[4]["rivals"] == 0


def test_gap_is_interpolated_and_long_gap_ends_the_track():
    def f(t):
        if t in (10, 11):                      # 2-frame gap: bridged
            return []
        if 20 <= t <= 24:                      # 5-frame gap: the track ends at 19
            return []
        return [[100 + 8 * t, 300, 12, 12, 0.8, 1]]
    tr = build_tracks(_frames(40, f))
    long = max(tr, key=lambda t: len(t.frames))
    assert long.frames[0] == 0 and long.frames[-1] == 19
    assert long.interp.sum() == 2
    assert abs(long.boxes[10][0] - (100 + 80)) < 1e-6


def test_jumpy_off_pitch_and_still_tracks_are_rejected():
    jumpy = _frames(20, lambda t: [[100 + (40 if t % 2 else 0) + 3 * t, 300 + (25 if t % 3 == 0 else 0),
                                    12, 12, 0.7, 1]])
    assert build_tracks(jumpy) == []
    off = _frames(20, lambda t: [[100 + 8 * t, 300, 12, 12, 0.9, 0]])
    assert build_tracks(off) == []
    still = _frames(20, lambda t: [[500, 300, 12, 12, 0.7, 1]])
    assert build_tracks(still) == []
    sure = _frames(20, lambda t: [[500, 300, 12, 12, 0.95, 1]])
    assert len(build_tracks(sure)) == 1


def test_grows_backwards_with_the_right_velocity():
    # seed (highest score) in the middle, weaker candidates either side
    def f(t):
        return [[100 + 12 * t, 300, 12, 12, 0.95 if t == 15 else 0.4 + 0.1 * (t % 2), 1]]
    cfg = TrackConfig(min_mean_score=0.4, min_first_share=0.0)
    tr = build_tracks(_frames(30, f), cfg)
    assert len(tr) == 1 and len(tr[0].frames) == 30
