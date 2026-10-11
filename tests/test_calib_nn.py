"""Calibrator: PnLCalib's solver recovers a known camera; the heads calibrate a real frame."""
import importlib
import json
import os
import sys
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
def test_keypoint_world_is_centre_origin():
    from footlytics.geometry.calib_nn import PnlSolver

    solver = PnlSolver(CODE)
    assert solver.keypoint_world(1) == (-52.5, -34.0, 0.0)    # PnLCalib kp 1: the (0, 0) corner
    assert solver.keypoint_world(3) == (52.5, -34.0, 0.0)
    assert solver.keypoint_world(12)[2] == -2.44               # crossbar end
    assert solver.keypoint_world(13)[2] == 0.0                 # its post foot


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
    for k, tol in (("pan_degrees", 0.5), ("tilt_degrees", 0.5), ("roll_degrees", 0.5)):
        assert fit.cam[k] == pytest.approx(cam[k], abs=tol)
    assert fit.cam["position_meters"] == pytest.approx(cam["position_meters"], abs=0.5)
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

    # 00532 shows the penalty area clearly (22 keypoints, 10 lines on the host, median 1.8 px); this is a wiring check, not a difficulty benchmark.
    z = zipfile.ZipFile(SNCAL_ZIP)
    img = cv2.imdecode(np.frombuffer(z.read("valid/00532.jpg"), np.uint8), cv2.IMREAD_COLOR)
    gt = json.loads(z.read("valid/00532.json"))
    cal = SoccerMasterCalibrator(CODE, WEIGHTS)
    fit = cal.fit([img[:, :, ::-1]])[0]
    assert fit is not None
    assert median_reprojection_px(gt, fit.cam) < 15.0


def test_fit_records_solver_errors_per_frame():
    import torch

    from footlytics.geometry.calib_nn import CameraFit, SoccerMasterCalibrator

    ok = CameraFit({"x_focal_length": 1.0}, 0.5, np.eye(3))

    class StubSolver:
        def decode(self, kp, ln, kp_threshold, line_threshold):
            return [({}, {}), ({}, {})]

        def solve(self, kp, ln, w, h, **kw):
            if self.calls:
                raise ZeroDivisionError("x")
            self.calls += 1
            return ok

        calls = 0

    cal = object.__new__(SoccerMasterCalibrator)
    cal.heatmaps = lambda frames: (torch.zeros(2, 58, 4, 4), torch.zeros(2, 24, 4, 4))
    cal.solver = StubSolver()
    cal.last_errors = {0: "stale"}
    out = cal.fit([np.zeros((8, 8, 3), np.uint8)] * 2)
    assert out == {0: ok, 1: None}
    assert cal.last_errors == {1: "ZeroDivisionError: x"}


@pytest.fixture
def isolated_imports(monkeypatch):
    """Leave sys.path and sys.modules as found: later tests import the real checkout."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    saved = dict(sys.modules)
    yield
    sys.modules.clear()
    sys.modules.update(saved)
    importlib.invalidate_caches()


def test_checkout_packages_win_over_an_installed_regular_package(tmp_path, isolated_imports):
    from footlytics.geometry.lines_nn import use_soccermaster_checkout

    site, checkout = tmp_path / "site", tmp_path / "checkout"
    (site / "data").mkdir(parents=True)
    (site / "data" / "__init__.py").write_text('WHO = "site"\n')
    (checkout / "data" / "pnlcalib_utils").mkdir(parents=True)
    (checkout / "data" / "pnlcalib_utils" / "utils_calib.py").write_text('WHO = "checkout"\n')
    sys.modules.pop("data", None)
    sys.path.insert(0, str(site))
    assert importlib.import_module("data").WHO == "site"

    use_soccermaster_checkout(checkout)
    use_soccermaster_checkout(checkout)                       # idempotent

    assert sys.path[0] == str(checkout) and sys.path.count(str(checkout)) == 1
    assert list(sys.modules["data"].__path__) == [str(checkout / "data")]
    assert importlib.import_module("data.pnlcalib_utils.utils_calib").WHO == "checkout"
    assert sys.modules["data.pnlcalib_utils.utils_calib"].__file__.startswith(str(checkout))


def test_solve_many_survives_a_worker_that_dies():
    from footlytics.geometry.calib_nn import solve_many
    from tests.calib_job_helpers import die_on_width_666

    jobs = [({}, {}, 100 + i if i != 3 else 666, 10) for i in range(8)]
    out = solve_many("unused", jobs, workers=2, _job=die_on_width_666)
    assert len(out) == len(jobs)
    assert out[3][0] is None and out[3][1].startswith("BrokenProcessPool")
    assert all(r == (None, None) or r[1].startswith("BrokenProcessPool") for r in out)
