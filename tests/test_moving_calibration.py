"""MovingCalibration: a per-frame image->pitch mapping for a camera that moves.

The anchor Calibration (their DLT + optional TPS) stays untouched; detections in
frame t are first warped back into anchor-frame pixels with inv(H_anchor_to_t).
"""
import numpy as np

from footlytics.geometry.camera_motion import MovingCalibration
from footlytics.geometry.homography import Calibration, apply_homography
from footlytics.geometry.pitch import DEFAULT_PITCH

W, H_PX = 1280, 720


def _anchor_calibration():
    lm = DEFAULT_PITCH.landmarks()
    # a plausible pitch->image homography for a tactical cam on one touchline
    H_pi = np.array([[9.0, -2.5, 640.0], [0.6, 5.0, 380.0], [0.0, -0.004, 1.0]])
    use = ["corner_LT", "corner_RT", "corner_LB", "corner_RB", "halfway_T", "halfway_B",
           "pen_spot_L", "pen_spot_R", "centre_circle_T", "centre_circle_B"]
    img = {n: tuple(apply_homography(H_pi, [lm[n]])[0]) for n in use}
    return Calibration.from_correspondences(img, lm, (W, H_PX)), H_pi, lm


def test_feet_map_through_camera_motion_back_to_pitch():
    cal, H_pi, lm = _anchor_calibration()
    mc = MovingCalibration(cal)
    truth = np.array([[10.0, 5.0], [-30.0, -20.0], [40.0, 25.0]])
    feet_anchor = apply_homography(H_pi, truth)
    # camera pans and zooms: anchor px -> frame t px
    H_a_t = np.array([[1.05, 0.0, 30.0], [0.0, 1.05, -12.0], [0.0, 0.0, 1.0]])
    feet_t = apply_homography(H_a_t, feet_anchor)
    bboxes = np.column_stack([feet_t[:, 0] - 10, feet_t[:, 1] - 40, np.full(3, 20.0), np.full(3, 40.0)])
    out = mc.feet_to_pitch(bboxes, H_a_t)
    assert np.allclose(out, truth, atol=0.05)


def test_identity_motion_equals_anchor_calibration():
    cal, H_pi, lm = _anchor_calibration()
    mc = MovingCalibration(cal)
    bboxes = np.array([[600.0, 300.0, 20.0, 40.0], [200.0, 500.0, 20.0, 40.0]])
    assert np.allclose(mc.feet_to_pitch(bboxes, np.eye(3)), cal.feet_to_pitch(bboxes))
