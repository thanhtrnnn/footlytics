import numpy as np

from footlytics.homography import PITCH_LENGTH_M, PITCH_WIDTH_M, fit_homography, image_to_pitch


def _synthetic_pairs():
    # Simulated image corners of a pitch seen from a broadcast camera (px) -> pitch metres.
    img = np.array([[100, 600], [1180, 600], [900, 150], [380, 150]], dtype=float)
    pitch = np.array([[0, 0], [PITCH_LENGTH_M, 0], [PITCH_LENGTH_M, PITCH_WIDTH_M], [0, PITCH_WIDTH_M]], dtype=float)
    return img, pitch


def test_pitch_dimensions():
    assert PITCH_LENGTH_M == 105.0
    assert PITCH_WIDTH_M == 68.0


def test_corners_map_to_pitch_corners():
    img, pitch = _synthetic_pairs()
    H, err = fit_homography(img, pitch)
    assert H.shape == (3, 3)
    assert err < 1e-6
    out = image_to_pitch(H, img)
    assert np.allclose(out, pitch, atol=1e-6)


def test_needs_at_least_four_points():
    img, pitch = _synthetic_pairs()
    H, err = fit_homography(img[:3], pitch[:3])
    assert H is None
    assert np.isnan(err)


def test_outlier_rejected_by_ransac():
    img, pitch = _synthetic_pairs()
    # add 4 more consistent points (midpoints of sides) and one outlier
    H_true, _ = fit_homography(img, pitch)
    extra_pitch = np.array([[52.5, 0], [52.5, 68], [0, 34], [105, 34]], dtype=float)
    Hinv = np.linalg.inv(H_true)
    extra_img = image_to_pitch(Hinv, extra_pitch)
    img_all = np.vstack([img, extra_img, [[640, 360]]])
    pitch_all = np.vstack([pitch, extra_pitch, [[10, 10]]])  # last pair is garbage
    H, err = fit_homography(img_all, pitch_all, ransac_thresh_m=2.0)
    out = image_to_pitch(H, img)
    assert np.allclose(out, pitch, atol=0.5)
