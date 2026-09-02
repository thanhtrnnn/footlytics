import numpy as np

from footlytics.homography import PITCH_LENGTH_M, PITCH_WIDTH_M, image_to_pitch
from footlytics.pitch import PITCH_VERTICES_M, homography_from_keypoints


def test_pitch_vertices_are_32_points_within_pitch_in_metres():
    v = np.asarray(PITCH_VERTICES_M)
    assert v.shape == (32, 2)
    assert v[:, 0].min() == 0 and abs(v[:, 0].max() - PITCH_LENGTH_M) < 1e-6
    assert v[:, 1].min() == 0 and abs(v[:, 1].max() - PITCH_WIDTH_M) < 1e-6
    # centre spot neighbours: vertices 31 and 32 are on the halfway line's centre circle
    assert abs(v[30, 1] - PITCH_WIDTH_M / 2) < 1e-6 and abs(v[31, 1] - PITCH_WIDTH_M / 2) < 1e-6


def _project(H, pts):
    pts_h = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return pts_h[:, :2] / pts_h[:, 2:3]


def test_homography_recovered_from_confident_keypoints_only():
    rng = np.random.default_rng(0)
    # a plausible pitch->image homography (tilted camera)
    H_true = np.array([[8.0, -3.0, 200.0], [0.5, 4.0, 100.0], [0.0, -0.01, 1.0]])
    verts = np.asarray(PITCH_VERTICES_M)
    img = _project(H_true, verts)
    conf = np.full(32, 0.9)
    # make 20 keypoints "invisible": low conf and garbage coords
    hidden = rng.choice(32, 20, replace=False)
    conf[hidden] = 0.1
    img[hidden] = rng.uniform(0, 1000, size=(20, 2))
    H, err, n_used = homography_from_keypoints(img, conf, min_conf=0.5)
    assert n_used == 12
    assert err < 0.05
    back = image_to_pitch(H, img[conf >= 0.5])
    assert np.allclose(back, verts[conf >= 0.5], atol=0.05)


def test_too_few_keypoints_gives_none():
    img = np.zeros((32, 2))
    conf = np.zeros(32)
    conf[:3] = 0.9
    H, err, n_used = homography_from_keypoints(img, conf, min_conf=0.5)
    assert H is None and n_used == 3
