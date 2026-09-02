import cv2
import numpy as np

from footlytics.camera_motion import AnchoredCamera


def _textured(seed=0, h=360, w=640):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 90, np.uint8)
    for _ in range(150):
        x, y = rng.integers(0, w), rng.integers(0, h)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(3, 10)), tuple(int(v) for v in rng.integers(100, 255, 3)), -1)
    for _ in range(15):
        p, q = rng.integers(0, [w, h]), rng.integers(0, [w, h])
        cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in q), (235, 235, 235), 2)
    return img


def _warp(img, H):
    return cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))


def _proj(H, p):
    q = np.hstack([p, np.ones((len(p), 1))]) @ H.T
    return q[:, :2] / q[:, 2:3]


def test_cut_invalidates_and_return_re_anchors():
    base = _textured(0)
    other = _textured(7)  # an unrelated shot (bench close-up)
    cam = AnchoredCamera(base)
    assert cam.valid and np.allclose(cam.H_anchor_to_t, np.eye(3))
    pan = np.array([[1.001, 0.0, 4.0], [0.0, 1.001, 1.0], [0.0, 0.0, 1.0]])
    H = np.eye(3)
    for _ in range(4):
        H = pan @ H
        cam.update(_warp(base, H))
    assert cam.valid
    pts = np.array([[100, 100], [500, 120], [300, 250], [560, 330]], dtype=np.float64)
    assert np.abs(_proj(cam.H_anchor_to_t, pts) - _proj(H, pts)).max() < 2.0
    # cut to another shot: must become invalid, not hallucinate
    for _ in range(3):
        cam.update(other)
    assert not cam.valid
    # return to a shifted view of the pitch: global re-anchoring recovers within tolerance
    H_back = np.array([[1.0, 0.0, 25.0], [0.0, 1.0, -8.0], [0.0, 0.0, 1.0]])
    cam.update(_warp(base, H_back))
    assert cam.valid
    assert np.abs(_proj(cam.H_anchor_to_t, pts) - _proj(H_back, pts)).max() < 3.0


def test_periodic_reanchor_limits_drift():
    base = _textured(1)
    cam = AnchoredCamera(base, reanchor_every=5)
    pan = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    H = np.eye(3)
    for _ in range(30):
        H = pan @ H
        cam.update(_warp(base, H))
    pts = np.array([[100, 100], [500, 120], [300, 250]], dtype=np.float64)
    assert cam.valid
    assert np.abs(_proj(cam.H_anchor_to_t, pts) - _proj(H, pts)).max() < 2.0
    assert cam.n_reanchors >= 5
