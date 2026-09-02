import json

import numpy as np

from footlytics.calibration import load_calibration, static_homography
from footlytics.homography import image_to_pitch
from footlytics.pitch import PITCH_VERTICES_M


def _project(H, pts):
    pts_h = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return pts_h[:, :2] / pts_h[:, 2:3]


def test_load_and_fit_static_calibration(tmp_path):
    H_true = np.array([[8.0, -3.0, 200.0], [0.5, 4.0, 100.0], [0.0, -0.01, 1.0]])
    verts = np.asarray(PITCH_VERTICES_M)
    ids = [14, 15, 16, 18, 21, 25, 30, 31]  # 1-based roboflow vertex ids
    img = _project(H_true, verts[[i - 1 for i in ids]])
    calib = {"points": [{"vertex": i, "x": float(p[0]), "y": float(p[1])} for i, p in zip(ids, img)]}
    f = tmp_path / "calib.json"
    f.write_text(json.dumps(calib))
    img_pts, pitch_pts = load_calibration(f)
    assert img_pts.shape == (8, 2) and pitch_pts.shape == (8, 2)
    H, err = static_homography(f)
    assert err < 0.05
    assert np.allclose(image_to_pitch(H, img), verts[[i - 1 for i in ids]], atol=0.05)


def test_calibration_accepts_named_landmarks(tmp_path):
    calib = {"points": [
        {"vertex": "halfway_top", "x": 10, "y": 10},
        {"vertex": "halfway_bottom", "x": 10, "y": 700},
        {"vertex": "right_corner_top", "x": 1200, "y": 10},
        {"vertex": "right_corner_bottom", "x": 1200, "y": 700},
    ]}
    f = tmp_path / "calib.json"
    f.write_text(json.dumps(calib))
    img_pts, pitch_pts = load_calibration(f)
    assert np.allclose(pitch_pts[0], [52.5, 0]) or np.allclose(pitch_pts[0], [52.5, 68])
    assert pitch_pts[:, 0].max() == 105.0
