import cv2
import numpy as np

from footlytics.camera_motion import CameraMotion


def _textured_frame(seed=0, h=360, w=640):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 90, np.uint8)
    for _ in range(80):  # random bright blobs and lines as trackable texture
        x, y = rng.integers(0, w), rng.integers(0, h)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(3, 9)), tuple(int(v) for v in rng.integers(120, 255, 3)), -1)
    for _ in range(12):
        p, q = rng.integers(0, [w, h]), rng.integers(0, [w, h])
        cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in q), (230, 230, 230), 2)
    return img


def _warp(img, H):
    return cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))


def test_accumulated_homography_tracks_small_pan_and_zoom():
    base = _textured_frame()
    cm = CameraMotion()
    cm.update(base, mask=None)
    assert np.allclose(cm.H_0_to_t, np.eye(3))
    # simulate a slow pan (3 px/frame) plus slight zoom over 5 frames
    H_total = np.eye(3)
    for k in range(1, 6):
        step = np.array([[1.002, 0.0, 3.0], [0.0, 1.002, 0.5], [0.0, 0.0, 1.0]])
        H_total = step @ H_total
        cm.update(_warp(base, H_total), mask=None)
    est = cm.H_0_to_t
    # compare by mapping a grid of points
    pts = np.array([[100, 100], [500, 100], [300, 250], [550, 320]], dtype=np.float64)
    def proj(H, p):
        q = np.hstack([p, np.ones((len(p), 1))]) @ H.T
        return q[:, :2] / q[:, 2:3]
    assert np.abs(proj(est, pts) - proj(H_total, pts)).max() < 2.0


def test_static_frames_keep_identity_and_mask_excludes_regions():
    base = _textured_frame(seed=1)
    cm = CameraMotion()
    cm.update(base)
    mask = np.full(base.shape[:2], 255, np.uint8)
    mask[:, :320] = 0  # only right half trackable
    cm.update(base.copy(), mask=mask)
    assert np.abs(cm.H_0_to_t - np.eye(3)).max() < 1e-2
    assert cm.n_tracked > 20
