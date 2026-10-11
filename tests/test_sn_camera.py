"""SoccerNet camera convention: rotation, projection, homography <-> camera."""
import numpy as np
import pytest

from footlytics.geometry.sn_camera import (HALF_L, HALF_W, camera_from_homography,
                                           ground_homography, ground_lines, make_camera,
                                           pan_tilt_roll, project, rotation, scale_camera)

BROADCAST = dict(pan=0.0, tilt=80.0, roll=0.0, focal=1500.0, position=(0.0, 55.0, -15.0),
                 width=1920, height=1080)


def cam(**kw):
    a = {**BROADCAST, **kw}
    return make_camera(a["pan"], a["tilt"], a["roll"], a["focal"], a["position"], a["width"],
                       a["height"])


@pytest.mark.parametrize("ptr", [(0, 80, 0), (-25, 72, 1.5), (30, 85, -2)])
def test_rotation_roundtrip(ptr):
    assert pan_tilt_roll(rotation(*ptr)) == pytest.approx(ptr, abs=1e-9)


def test_broadcast_camera_sees_the_pitch_the_right_way_up():
    c = cam()
    centre, far, near, left = project(c, [[0, 0, 0], [0, -HALF_W, 0], [0, HALF_W, 0], [-20, 0, 0]])
    assert centre[0] == pytest.approx(960)
    assert far[1] < near[1]                 # far touchline higher in the image
    assert left[0] < 960                    # left goal on the left
    bar, foot = project(c, [[-HALF_L, 0, -2.44], [-HALF_L, 0, 0]])
    assert bar[1] < foot[1]                 # z negative is up


def test_points_behind_the_camera_are_nan():
    assert np.isnan(project(cam(), [[0, 80, 0]])).all()


def test_ground_homography_matches_projection():
    c = cam(pan=-12, tilt=76, roll=0.8)
    xy = np.random.default_rng(0).uniform([-50, -30], [50, 30], (20, 2))
    H = ground_homography(c)
    h = np.c_[xy, np.ones(len(xy))] @ H.T
    assert h[:, :2] / h[:, 2:] == pytest.approx(project(c, np.c_[xy, np.zeros(len(xy))]), abs=1e-6)


@pytest.mark.parametrize("kw", [{}, dict(pan=-20, tilt=70, roll=1), dict(pan=15, focal=2600,
                                                                     position=(-10, 60, -22))])
def test_camera_from_homography_roundtrip(kw):
    c = cam(**kw)
    back = camera_from_homography(ground_homography(c), c["principal_point"][0] * 2,
                                  c["principal_point"][1] * 2)
    for k in ("pan_degrees", "tilt_degrees", "roll_degrees", "x_focal_length"):
        assert back[k] == pytest.approx(c[k], abs=1e-4)
    assert back["position_meters"] == pytest.approx(c["position_meters"], abs=1e-3)


def test_scale_camera_keeps_angles_and_scales_pixels():
    s = scale_camera(cam(), 960)
    assert s["principal_point"] == [480.0, 270.0] and s["x_focal_length"] == 750.0
    assert s["pan_degrees"] == cam()["pan_degrees"]


def test_ground_lines_cover_the_ground_plane_classes():
    lines = ground_lines()
    assert len(lines) == 17 and "Big rect. left top" in lines and "Circle central" not in lines
    pts = np.array([p for seg in lines.values() for p in seg])
    assert (np.abs(pts[:, 0]) <= HALF_L).all() and (np.abs(pts[:, 1]) <= HALF_W).all()
    assert lines["Side line top"][0][1] == -HALF_W     # top = far = y negative
