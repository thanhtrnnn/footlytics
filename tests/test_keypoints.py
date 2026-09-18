"""geometry.keypoints: the 32-vertex layout and the homography fitted from it.

`PitchKeypointDetector` itself needs a YOLO-pose model, so only the geometry is
covered here -- that is where the mistakes live anyway.
"""
import cv2
import numpy as np
import pytest

from footlytics.geometry.keypoints import (
    PITCH_VERTICES_M,
    homography_from_keypoints,
    pitch_vertices,
)
from footlytics.geometry.pitch import DEFAULT_PITCH, Pitch


def test_vertices_are_centre_origin_and_span_the_pitch():
    """Origin at the centre spot, not a corner. Getting this wrong fits cleanly
    and puts every player half a pitch from where they are."""
    v = PITCH_VERTICES_M
    assert v.shape == (32, 2)
    assert v[:, 0].min() == pytest.approx(-DEFAULT_PITCH.half_l)
    assert v[:, 0].max() == pytest.approx(DEFAULT_PITCH.half_l)
    assert v[:, 1].min() == pytest.approx(-DEFAULT_PITCH.half_w)
    assert v[:, 1].max() == pytest.approx(DEFAULT_PITCH.half_w)
    # the halfway line sits on x = 0, and the centre spot is a vertex
    assert (np.abs(v[:, 0]) < 1e-9).sum() >= 3


def test_vertices_follow_the_pitch_dimensions():
    v = pitch_vertices(Pitch(length=100.0, width=64.0))
    assert v[:, 0].min() == pytest.approx(-50.0)
    assert v[:, 1].max() == pytest.approx(32.0)


def test_landmarks_agree_with_the_pitch_model():
    """The two vertex sets are different conventions over the same pitch, so the
    corners at least must coincide."""
    lm = DEFAULT_PITCH.landmarks()
    v = PITCH_VERTICES_M
    for name in ("corner_LT", "corner_RB"):
        d = np.linalg.norm(v - np.asarray(lm[name]), axis=1).min()
        assert d < 1e-6, f"{name} is not among the 32 vertices"


def _synthetic_view(vertices):
    """Project the pitch into a plausible 1280x720 camera view and return (pixels, H_true)."""
    src = np.array([[-52.5, -34.0], [52.5, -34.0], [52.5, 34.0], [-52.5, 34.0]], np.float32)
    dst = np.array([[120, 620], [1160, 620], [980, 210], [300, 210]], np.float32)   # perspective
    H_pitch_to_img = cv2.getPerspectiveTransform(src, dst)
    px = cv2.perspectiveTransform(np.asarray(vertices, np.float64).reshape(-1, 1, 2),
                                  H_pitch_to_img).reshape(-1, 2)
    return px, np.linalg.inv(H_pitch_to_img)


def test_recovers_the_mapping_from_confident_keypoints():
    px, _ = _synthetic_view(PITCH_VERTICES_M)
    conf = np.full(32, 0.9)
    conf[:6] = 0.1                                  # a few vertices not found

    cal, err, n_used = homography_from_keypoints(px, conf, image_size=(1280, 720))
    assert n_used == 26
    assert err < 0.05
    back = cal.image_to_pitch(px)
    assert np.abs(back - PITCH_VERTICES_M).max() < 0.1


def test_ransac_rejects_a_stray_keypoint():
    """One vertex predicted in completely the wrong place must not drag the fit."""
    px, _ = _synthetic_view(PITCH_VERTICES_M)
    px[7] = [40.0, 60.0]
    cal, err, _ = homography_from_keypoints(px, np.full(32, 0.9), image_size=(1280, 720))
    assert err < 0.5
    good = np.delete(np.arange(32), 7)
    back = cal.image_to_pitch(px[good])
    assert np.abs(back - PITCH_VERTICES_M[good]).max() < 1.0


def test_too_few_confident_keypoints_returns_none():
    px, _ = _synthetic_view(PITCH_VERTICES_M)
    conf = np.zeros(32)
    conf[:3] = 0.9
    cal, err, n_used = homography_from_keypoints(px, conf, image_size=(1280, 720))
    assert cal is None and n_used == 3 and np.isnan(err)


def test_wrong_keypoint_count_is_rejected_loudly():
    with pytest.raises(ValueError, match="expected 32"):
        homography_from_keypoints(np.zeros((10, 2)), np.ones(10), image_size=(1280, 720))
