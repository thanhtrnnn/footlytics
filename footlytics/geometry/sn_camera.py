"""SoccerNet camera parameters: the full pinhole camera that SN-Calibration scores.

World frame (SoccerNet's; GSR and PnLCalib use it too): metres from the centre spot, x along
the length, y negative towards the far ("top") touchline, z negative UP (a crossbar sits at
z = -2.44). A world point X goes to the camera frame as R (X - position), with
R = (Rz(pan) Rx(tilt) Rz(roll))^T, then to pixels through the focal lengths and principal point.
Distortion is always zero in what we write.

Our implementation of the published convention; the evaluator's code is never imported.
"""
from __future__ import annotations

import numpy as np

HALF_L, HALF_W = 52.5, 34.0
_PA_DEPTH, _PA_HALF_W = 16.5, 20.16
_GA_DEPTH, _GA_HALF_W = 5.5, 9.16


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation(pan: float, tilt: float, roll: float) -> np.ndarray:
    """World -> camera rotation from SoccerNet's pan/tilt/roll in degrees."""
    p, t, r = np.radians([pan, tilt, roll])
    return (_rz(p) @ _rx(t) @ _rz(r)).T


def pan_tilt_roll(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of `rotation`; of the two solutions, the one with less roll (operators keep it
    near zero), as SoccerNet does."""
    o = np.asarray(R).T
    best = None
    t0 = float(np.arccos(np.clip(o[2, 2], -1.0, 1.0)))
    for tilt in (t0, -t0):
        s = 1.0 if np.sin(tilt) > 0 else -1.0
        pan = np.arctan2(s * o[0, 2], -s * o[1, 2])
        roll = np.arctan2(s * o[2, 0], s * o[2, 1])
        if best is None or abs(roll) < abs(best[2]):
            best = (pan, tilt, roll)
    return tuple(float(np.degrees(v)) for v in best)


def make_camera(pan: float, tilt: float, roll: float, focal: float, position, width: float,
                height: float) -> dict:
    """A SoccerNet camera dict, as the evaluator reads it."""
    return {"pan_degrees": float(pan), "tilt_degrees": float(tilt), "roll_degrees": float(roll),
            "position_meters": [float(v) for v in position],
            "x_focal_length": float(focal), "y_focal_length": float(focal),
            "principal_point": [float(width) / 2, float(height) / 2],
            "radial_distortion": [0.0] * 6, "tangential_distortion": [0.0] * 2,
            "thin_prism_distortion": [0.0] * 4}


def intrinsics(cam: dict) -> np.ndarray:
    cx, cy = cam["principal_point"]
    return np.array([[cam["x_focal_length"], 0.0, cx], [0.0, cam["y_focal_length"], cy],
                     [0.0, 0.0, 1.0]])


def _pose(cam: dict) -> tuple[np.ndarray, np.ndarray]:
    R = rotation(cam["pan_degrees"], cam["tilt_degrees"], cam["roll_degrees"])
    return R, np.asarray(cam["position_meters"], float)


def project(cam: dict, pts3d) -> np.ndarray:
    """World points (N, 3) -> pixels (N, 2); NaN for points at or behind the camera."""
    R, C = _pose(cam)
    X = (np.asarray(pts3d, float).reshape(-1, 3) - C) @ R.T
    out = np.full((len(X), 2), np.nan)
    ok = X[:, 2] > 1e-3
    K = intrinsics(cam)
    uv = X[ok] @ K.T
    out[ok] = uv[:, :2] / uv[:, 2:]
    return out


def ground_homography(cam: dict) -> np.ndarray:
    """Pitch plane (x, y, z=0) -> image pixels."""
    R, C = _pose(cam)
    return intrinsics(cam) @ np.column_stack([R[:, 0], R[:, 1], -R @ C])


def camera_from_homography(H: np.ndarray, width: float, height: float) -> dict | None:
    """A camera from a pitch -> image homography, assuming square pixels and the principal
    point at the image centre (as PnLCalib does). None when the homography admits no camera."""
    cx, cy = width / 2, height / 2
    M = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]]) @ np.asarray(H, float)
    cands = []                       # (|denominator|, f^2) from the two orthonormality constraints
    d1 = M[2, 0] * M[2, 1]
    if abs(d1) > 1e-12:
        cands.append((abs(d1), -(M[0, 0] * M[0, 1] + M[1, 0] * M[1, 1]) / d1))
    d2 = M[2, 0] ** 2 - M[2, 1] ** 2
    if abs(d2) > 1e-12:
        cands.append((abs(d2), (M[0, 1] ** 2 + M[1, 1] ** 2 - M[0, 0] ** 2 - M[1, 0] ** 2) / d2))
    cands = [c for c in cands if c[1] > 0]
    if not cands:
        return None
    f = float(np.sqrt(max(cands)[1]))
    A = np.diag([1 / f, 1 / f, 1.0]) @ M
    lam = 2.0 / (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
    r1, r2, t = lam * A[:, 0], lam * A[:, 1], lam * A[:, 2]
    if t[2] < 0:                     # the pitch must be in front of the camera
        r1, r2, t = -r1, -r2, -t
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        return None
    pan, tilt, roll = pan_tilt_roll(R)
    return make_camera(pan, tilt, roll, f, -R.T @ t, width, height)


def scale_camera(cam: dict, width: float) -> dict:
    """The same camera at another image width (the image size is 2 x principal point)."""
    k = width / (2 * cam["principal_point"][0])
    return {**cam, "x_focal_length": cam["x_focal_length"] * k,
            "y_focal_length": cam["y_focal_length"] * k,
            "principal_point": [cam["principal_point"][0] * k, cam["principal_point"][1] * k]}


def ground_lines() -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    """The straight ground-plane line classes of SN-Calibration, as world segments."""
    L, W = HALF_L, HALF_W
    lines = {"Side line top": ((-L, -W), (L, -W)), "Side line bottom": ((-L, W), (L, W)),
             "Side line left": ((-L, -W), (-L, W)), "Side line right": ((L, -W), (L, W)),
             "Middle line": ((0.0, -W), (0.0, W))}
    for side, s in (("left", -1), ("right", 1)):
        for name, depth, hw in (("Big rect.", _PA_DEPTH, _PA_HALF_W),
                                ("Small rect.", _GA_DEPTH, _GA_HALF_W)):
            x0, x1 = s * L, s * (L - depth)
            lines[f"{name} {side} top"] = ((x0, -hw), (x1, -hw))
            lines[f"{name} {side} main"] = ((x1, -hw), (x1, hw))
            lines[f"{name} {side} bottom"] = ((x0, hw), (x1, hw))
    return lines
