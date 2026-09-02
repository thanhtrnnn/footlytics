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


def test_anchor_frame_composition():
    """H_t = H_anchor @ inv(H_anchor_to_t) with H_anchor_to_t = H_0_to_t @ inv(H_0_to_anchor)."""
    from footlytics.calibration import compose_from_anchor

    H_anchor = np.array([[2.0, 0.1, 5.0], [0.0, 1.5, -3.0], [0.0, 0.001, 1.0]])
    motions = {0: np.eye(3)}
    step = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    for t in range(1, 5):
        motions[t] = step @ motions[t - 1]
    homs = compose_from_anchor(H_anchor, motions, anchor_frame=2)
    assert np.allclose(homs[2], H_anchor)
    # frame 3 is one step after the anchor: image_3 = step(image_2), so H_3 = H_anchor @ inv(step)
    assert np.allclose(homs[3], H_anchor @ np.linalg.inv(step))
    assert np.allclose(homs[0], H_anchor @ step @ step)


def test_calibration_json_anchor_frame_defaults_to_zero(tmp_path):
    from footlytics.calibration import anchor_frame_of

    f = tmp_path / "c.json"
    f.write_text(json.dumps({"points": []}))
    assert anchor_frame_of(f) == 0
    f.write_text(json.dumps({"points": [], "frame": 750}))
    assert anchor_frame_of(f) == 750
