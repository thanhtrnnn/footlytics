"""ClipCalibrator on a camera that is a camera: keyframe smoothing, far-field gate, keypoint floor.

(`test_clip_calib.py` uses a 16x9-pixel toy whose far field is meaningless; it switches these
stages off and tests the gate and the bridging.)
"""
import numpy as np

from footlytics.geometry.calib_nn import CameraFit
from footlytics.geometry.camera_smooth import homography_from_params, params_from_homography
from footlytics.geometry.clip_calib import ClipCalibrator
from footlytics.geometry.sn_camera import ground_homography, make_camera

FW, FH = 192, 108
RCAM = make_camera(35.0, 72.0, 0.0, 220.0, [-18.0, 52.0, -22.0], FW, FH)
PROBE = np.array([(u * FW, v * FH) for v in (0.4, 0.6, 0.9) for u in (0.15, 0.5, 0.85)])


def real_H(pan):
    cam = make_camera(pan, 72.0, 0.0, 220.0, [-18.0, 52.0, -22.0], FW, FH)
    Hm = np.linalg.inv(ground_homography(cam))
    return Hm / Hm[2, 2]


def truth(t):
    return 30.0 + 0.04 * t


def real_frame(t):
    f = np.zeros((FH, FW, 3), np.uint8)
    f[0, 0, 0] = t
    return f


def pitch_of(Hm, pts):
    q = np.c_[pts, np.ones(len(pts))] @ Hm.T
    return q[:, :2] / q[:, 2:3]


class DriftingTracker:
    """Bridges exactly along the true pan, starting from whatever the anchor keyframe said."""
    KEY: dict = {}

    def __init__(self, anchor, mask=None):
        self.t0 = int(anchor[0, 0, 0])

    def update(self, f, mask=None):
        t = int(f[0, 0, 0])
        Hk = DriftingTracker.KEY[self.t0]
        d = (params_from_homography(real_H(truth(t)), FW, FH)
             - params_from_homography(real_H(truth(self.t0)), FW, FH))
        Hb = homography_from_params(params_from_homography(Hk, FW, FH) + d, FW, FH)
        return np.linalg.inv(np.linalg.inv(Hk) @ Hb)


def keyfits(offsets, n_keypoints=None):
    def fit_keyframes(chunk, keys):
        out = {}
        for k in keys:
            t = int(chunk[k][0, 0, 0])
            P = params_from_homography(real_H(truth(t)), FW, FH) + offsets.get(t, 0.0)
            Hk = homography_from_params(P, FW, FH)
            DriftingTracker.KEY[t] = Hk
            fit = CameraFit(RCAM, 3.0, Hk)
            if n_keypoints is not None:
                fit.n_keypoints = n_keypoints
            out[k] = fit
        return out
    return fit_keyframes


def calibrate(offsets, n=120, n_keypoints=None, **kw):
    cc = ClipCalibrator(keyfits(offsets, n_keypoints), every=10, gate_px=25.0,
                        tracker_factory=DriftingTracker, chunk=30, **kw)
    return cc, cc.run(real_frame(t) for t in range(n))


def worst_step(out, lo=15, hi=105):
    return max(np.linalg.norm(pitch_of(out[i].H_img_to_pitch, PROBE)
                              - pitch_of(out[i + 1].H_img_to_pitch, PROBE), axis=1).max()
               for i in range(lo, hi))


def test_smoothing_softens_the_step_at_every_keyframe():
    rng = np.random.default_rng(3)
    offs = {t: rng.normal(0, 0.5, 8) for t in range(0, 120, 10)}      # each keyframe ~0.5 m off
    _, raw = calibrate(offs, smooth=False)
    cc, smooth = calibrate(offs)
    assert worst_step(raw) > 0.5
    assert worst_step(smooth) < 0.5 * worst_step(raw)
    assert cc.smooth_info.segments == 1
    assert smooth[15].cam is not None and smooth[15].source == raw[15].source


def test_far_field_disagreement_rejects_a_keyframe_the_lower_image_cannot_see():
    # agrees on the pitch where players stand, metres out at the far touchline
    far_only = np.zeros(8)
    far_only[:2] = (6.0, 0.0)                     # far-left anchor moves 6 m
    kw = dict(n=40, smooth=False, bridge_max_m=3.0)
    _, out_bad = calibrate({10: far_only}, bridge_far_max_m=3.0, **kw)
    _, out_ok = calibrate({10: np.zeros(8)}, bridge_far_max_m=3.0, **kw)
    _, out_old = calibrate({10: far_only}, bridge_far_max_m=None, **kw)
    assert out_bad[10].source == "bridged"        # refused
    assert out_ok[10].source == "keyframe"
    assert out_old[10].source == "keyframe"       # the lower-image-only check let it through


def test_a_fit_with_too_few_keypoints_is_not_trusted():
    cc, out = calibrate({}, n=40, smooth=False, n_keypoints=4)
    assert cc.keyframes_accepted == 0 and cc.keyframes_rejected_keypoints == 4
    assert all(o.source == "none" for o in out)
    cc2, out2 = calibrate({}, n=40, smooth=False, n_keypoints=7)
    assert cc2.keyframes_accepted == 4 and out2[10].source == "keyframe"
    cc3, _ = calibrate({}, n=40, smooth=False)               # does not say: not penalised
    assert cc3.keyframes_accepted == 4
