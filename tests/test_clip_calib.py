"""Clip calibration: keyframes, the gate, bridging, and the overlay mask."""
import numpy as np
import pytest

from footlytics.geometry.calib_nn import CameraFit
from footlytics.geometry.clip_calib import ClipCalibrator, static_overlay_mask
from footlytics.geometry.sn_camera import ground_homography, make_camera

# a broadcast camera scaled to tiny 16x9 frames, so the bridged camera is recoverable
CAM = make_camera(0, 80, 0, 12.5, (0, 55, -15), 16, 9)
H0 = np.linalg.inv(ground_homography(CAM))
H0 /= H0[2, 2]
STEP = 0.01                                            # px of pan per frame


def frame(t):
    f = np.zeros((9, 16, 3), np.uint8)
    f[0, 0, 0] = t
    return f


def shift(dx):
    return np.array([[1.0, 0, dx * STEP], [0, 1, 0], [0, 0, 1]])


class FakeTracker:
    """Anchor->t is a small pan of (t - t_anchor) steps; frames in LOST are untrackable."""
    LOST: set = set()

    def __init__(self, anchor, mask=None):
        self.t0 = int(anchor[0, 0, 0])

    def update(self, f, mask=None):
        t = int(f[0, 0, 0])
        return None if t in self.LOST else shift(t - self.t0)


def fits(rep_errs):
    def fit_keyframes(chunk, keys):
        return {k: (None if rep_errs.get(int(chunk[k][0, 0, 0])) is None
                    else CameraFit(CAM, rep_errs[int(chunk[k][0, 0, 0])], H0)) for k in keys}
    return fit_keyframes


def run(rep_errs, n=60, lost=(), every=25):
    FakeTracker.LOST = set(lost)
    cc = ClipCalibrator(fits(rep_errs), every=every, gate_px=25.0, tracker_factory=FakeTracker,
                        chunk=30)
    return cc, cc.run(frame(t) for t in range(n))


def test_keyframes_anchor_and_frames_between_are_bridged():
    cc, out = run({0: 4.0, 25: 6.0, 50: 5.0})
    assert [o.source for o in out[:3]] == ["keyframe", "bridged", "bridged"]
    assert np.allclose(out[3].H_img_to_pitch, H0 @ np.linalg.inv(shift(3)) /
                       (H0 @ np.linalg.inv(shift(3)))[2, 2])
    assert out[3].cam is not None                       # camera recovered from the homography
    assert cc.keyframes == 3 and cc.keyframes_accepted == 3


def test_a_keyframe_over_the_gate_is_rejected_and_bridging_continues():
    cc, out = run({0: 4.0, 25: 40.0, 50: 5.0})
    assert out[25].source == "bridged" and out[25].rep_err == 40.0
    assert out[30].source == "bridged"
    assert cc.keyframes_accepted == 2


def test_frames_before_the_first_accepted_keyframe_stay_uncalibrated():
    _, out = run({0: 99.0, 25: 5.0, 50: 5.0})
    assert all(o.source == "none" and o.H_img_to_pitch is None for o in out[:25])
    assert out[25].source == "keyframe"


def test_a_lost_track_leaves_frames_uncalibrated_until_the_next_keyframe():
    _, out = run({0: 4.0, 25: None, 50: 5.0}, lost=range(10, 50))
    assert out[9].source == "bridged" and out[10].source == "none" and out[49].source == "none"
    assert out[50].source == "keyframe"


def test_overlay_mask_finds_a_static_logo_on_a_moving_picture():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (120, 200), np.uint8)
    frames = []
    for t in range(20):
        g = np.roll(base, 3 * t, axis=1)
        g[5:20, 5:40] = 200                           # a score bug that never moves
        frames.append(np.dstack([g, g, g]))
    m = static_overlay_mask(frames)
    assert (m[5:20, 5:40] == 0).all()
    assert (m == 255).mean() > 0.8


def test_overlay_mask_on_a_static_camera_falls_back_to_a_top_band():
    f = np.random.default_rng(1).integers(0, 255, (100, 100, 3), np.uint8)
    m = static_overlay_mask([f] * 10)
    assert (m[:12] == 0).all() and (m[12:] == 255).all()
