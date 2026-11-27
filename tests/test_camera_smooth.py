"""Camera drift correction: stable parameterisation, steps handed back, motion and cuts untouched."""
import numpy as np

from footlytics.geometry.camera_smooth import (
    anchor_points, correct_drift, corner_dist, homography_from_params, params_from_homography,
    smooth_homographies, step_height,
)
from footlytics.geometry.sn_camera import ground_homography, make_camera

W, H = 1920.0, 1080.0


def cam_H(pan, tilt=72.0, focal=2200.0):
    cam = make_camera(pan, tilt, 0.0, focal, [-18.0, 52.0, -22.0], W, H)
    Hm = np.linalg.inv(ground_homography(cam))
    return Hm / Hm[2, 2]


def pitch_of(Hm, pts):
    q = np.c_[pts, np.ones(len(pts))] @ Hm.T
    return q[:, :2] / q[:, 2:3]


PROBE = np.array([(u, v) for v in (0.4 * H, 0.6 * H, 0.9 * H) for u in (0.15 * W, 0.5 * W, 0.85 * W)])


def probe_err(Ha, Hb):
    return np.linalg.norm(pitch_of(Ha, PROBE) - pitch_of(Hb, PROBE), axis=1).max()


def pan_series(n, step=0.05, start=30.0):
    return [cam_H(start + step * t) for t in range(n)]


def sawtooth(truth, every=10, drift=(0.8, -0.5)):
    """A chain: exact motion, drift growing since the last keyframe, reset at each keyframe."""
    out, key = [], []
    for t, Ht in enumerate(truth):
        P = params_from_homography(Ht, W, H)
        P = P + np.array(drift * 4) * ((t % every) / every)
        out.append(homography_from_params(P, W, H))
        key.append(t % every == 0)
    return out, key


def test_params_round_trip_is_exact():
    Hm = cam_H(35.0)
    P = params_from_homography(Hm, W, H)
    assert probe_err(Hm, homography_from_params(P, W, H)) < 1e-6
    assert np.allclose(pitch_of(Hm, anchor_points(W, H)).ravel(), P)


def test_anchor_on_horizon_is_rejected():
    Hm = cam_H(35.0)
    Hm[2, :] = [0, 1e-3, 0]
    Hm[2, 2] = -0.5 * H * 1e-3                               # horizon in the middle of the quad
    assert params_from_homography(Hm, W, H) is None
    assert params_from_homography(None, W, H) is None
    assert params_from_homography(np.full((3, 3), np.nan), W, H) is None


def test_a_steady_pan_comes_back_unchanged():
    truth = pan_series(120)
    sm, info = smooth_homographies(truth, W, H, key=[t % 10 == 0 for t in range(120)], sigma=3.0)
    assert max(probe_err(a, b) for a, b in zip(truth, sm)) < 0.02       # no lag, no bias on a ramp
    assert info.segments == 1


def test_step_height_discounts_the_pan():
    f = np.arange(30.0)
    X = np.outer(f, np.arange(1, 9) * 0.1)                  # a pan: every coordinate ramps
    X[15:] += 0.7
    e = step_height(f, X, 15)
    assert np.allclose(e, 0.7, atol=1e-9)
    X2 = np.outer(f, np.arange(1, 9) * 0.1)
    assert np.abs(step_height(f, X2, 15)).max() < 1e-9


def test_the_sawtooth_of_a_drifting_chain_is_removed():
    truth = pan_series(200)
    seen, key = sawtooth(truth)

    def worst_step(Hs):
        return max(probe_err(Hs[i], Hs[i + 1]) for i in range(25, 175))

    sm, info = smooth_homographies(seen, W, H, key=key, sigma=0.0)
    assert worst_step(seen) > 0.8
    assert worst_step(sm) < 0.25 * worst_step(seen)
    # and the camera is closer to the truth, not just smoother
    err_seen = np.mean([probe_err(seen[i], truth[i]) for i in range(25, 175)])
    err_sm = np.mean([probe_err(sm[i], truth[i]) for i in range(25, 175)])
    assert err_sm < 0.3 * err_seen
    assert info.events >= 15


def test_a_cut_is_not_smeared():
    a, b = pan_series(60, start=20.0), pan_series(60, start=75.0)
    seq = a + b                                              # another shot, tens of metres away
    key = [t % 10 == 0 for t in range(120)]
    assert corner_dist(params_from_homography(a[-1], W, H)[None],
                       params_from_homography(b[0], W, H)[None])[0] > 15
    sm, info = smooth_homographies(seq, W, H, key=key, sigma=3.0)
    assert info.breaks_jump == 1 and info.segments == 2
    for i in (56, 57, 58, 59, 60, 61, 62, 63):               # frames either side of the cut
        assert probe_err(sm[i], seq[i]) < 0.1, i


def test_a_step_too_big_to_be_drift_is_not_spread():
    a, b = pan_series(60, start=20.0), pan_series(60, start=33.0)
    seq = a + b
    jump = corner_dist(params_from_homography(a[-1], W, H)[None],
                       params_from_homography(b[0], W, H)[None])[0]
    P = np.array([params_from_homography(h, W, H) for h in seq])
    key = np.zeros(120, bool)
    key[60] = True
    out, info = correct_drift(P, np.ones(120, bool), key, break_jump_m=1e9, cap_m=jump / 2, sigma=0.0)
    assert info.breaks_cap == 1 and info.events == 0
    assert np.allclose(out, P)


def test_caller_breaks_and_missing_frames_split_segments():
    seq = pan_series(80)
    key = [t % 10 == 0 for t in range(80)]
    _, info = smooth_homographies(seq, W, H, key=key, breaks=[i == 40 for i in range(80)])
    assert info.segments == 2
    seq2 = list(seq)
    seq2[30] = None
    sm2, info2 = smooth_homographies(seq2, W, H, key=key)
    assert sm2[30] is None and info2.segments == 2


def test_repeated_video_frames_keep_their_first_frames_camera():
    truth = pan_series(100, step=0.4)
    seen, key = sawtooth(truth)
    for t in range(1, 100, 3):                                # every 3rd frame repeats the one before
        seen[t] = seen[t - 1]
    sm, _ = smooth_homographies(seen, W, H, key=key, sigma=3.0)
    for t in range(1, 100, 3):
        assert probe_err(sm[t], sm[t - 1]) < 1e-6, t


def test_nothing_to_correct_is_the_identity():
    seq = pan_series(30)
    sm, _ = smooth_homographies(seq, W, H, sigma=0.0)
    assert all(a is b for a, b in zip(seq, sm))
