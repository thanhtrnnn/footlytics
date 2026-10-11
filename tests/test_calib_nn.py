"""Calibrator: PnLCalib's solver recovers a known camera; the heads calibrate a real frame."""
import json
import os
import zipfile

import numpy as np
import pytest

from footlytics.geometry.sn_camera import make_camera, project

CODE = os.getenv("FOOTLYTICS_SOCCERMASTER_CODE")
WEIGHTS = os.getenv("FOOTLYTICS_SOCCERMASTER_WEIGHTS")
SNCAL_ZIP = os.getenv("FOOTLYTICS_SNCAL_ZIP")
needs_code = pytest.mark.skipif(not CODE, reason="set FOOTLYTICS_SOCCERMASTER_CODE")


def synthetic_keypoints(solver, cam, width, height):
    kp = {}
    for i in range(1, 31):
        uv = project(cam, [solver.keypoint_world(i)])[0]
        if np.isfinite(uv).all() and 0 <= uv[0] < width and 0 <= uv[1] < height:
            kp[i] = {"x": uv[0] / width, "y": uv[1] / height, "p": 1.0}
    return kp


@needs_code
@pytest.mark.parametrize("voting", [True, False])
def test_solver_recovers_a_known_camera(voting):
    from footlytics.geometry.calib_nn import PnlSolver

    solver = PnlSolver(CODE)
    cam = make_camera(-18, 77, 0.5, 1900, (-8, 52, -16), 1920, 1080)
    kp = synthetic_keypoints(solver, cam, 1920, 1080)
    assert len(kp) >= 6
    fit = solver.solve(kp, {}, 1920, 1080, voting=voting, refine_lines=False)
    assert fit is not None and fit.rep_err < 2.0
    for k, tol in (("pan_degrees", 0.5), ("tilt_degrees", 0.5)):
        assert fit.cam[k] == pytest.approx(cam[k], abs=tol)
    assert fit.cam["x_focal_length"] == pytest.approx(1900, rel=0.03)
    assert "rotation_matrix" not in fit.cam
    assert np.isfinite(fit.H_img_to_pitch @ np.array([960.0, 540.0, 1.0])).all()


@needs_code
def test_solve_many_reports_each_job():
    from footlytics.geometry.calib_nn import PnlSolver, solve_many

    solver = PnlSolver(CODE)
    cam = make_camera(10, 78, 0, 1700, (5, 54, -15), 1920, 1080)
    good = (synthetic_keypoints(solver, cam, 1920, 1080), {}, 1920, 1080)
    out = solve_many(CODE, [good, ({}, {}, 1920, 1080)], workers=1, refine_lines=False)
    assert out[0][0] is not None and out[0][1] is None
    assert out[1] == (None, None)                    # no keypoints: no camera, not an error


@pytest.mark.skipif(not (CODE and WEIGHTS and SNCAL_ZIP),
                    reason="set FOOTLYTICS_SOCCERMASTER_CODE/_WEIGHTS and FOOTLYTICS_SNCAL_ZIP")
def test_calibrates_a_real_sn_calibration_frame():
    import cv2

    from footlytics.bench.metrics import median_reprojection_px
    from footlytics.geometry.calib_nn import SoccerMasterCalibrator

    z = zipfile.ZipFile(SNCAL_ZIP)
    img = cv2.imdecode(np.frombuffer(z.read("valid/00007.jpg"), np.uint8), cv2.IMREAD_COLOR)
    gt = json.loads(z.read("valid/00007.json"))
    cal = SoccerMasterCalibrator(CODE, WEIGHTS)
    fit = cal.fit([img[:, :, ::-1]])[0]
    assert fit is not None
    assert median_reprojection_px(gt, fit.cam) < 15.0
