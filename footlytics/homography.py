"""Image-to-pitch homography fitting and mapping.

Pitch coordinate frame: metres, origin at bottom-left corner, x along the length (0..105),
y along the width (0..68).
"""
from __future__ import annotations

import cv2
import numpy as np

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


def fit_homography(
    image_pts: np.ndarray, pitch_pts: np.ndarray, ransac_thresh_m: float = 2.0
) -> tuple[np.ndarray | None, float]:
    """Fit H mapping image pixels -> pitch metres.

    Returns (H, median reprojection error in metres). H is None and error NaN when fewer
    than 4 point pairs are given or fitting fails.
    """
    image_pts = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    pitch_pts = np.asarray(pitch_pts, dtype=np.float64).reshape(-1, 2)
    if len(image_pts) < 4 or len(image_pts) != len(pitch_pts):
        return None, float("nan")
    method = cv2.RANSAC if len(image_pts) > 4 else 0
    H, mask = cv2.findHomography(image_pts, pitch_pts, method, ransac_thresh_m)
    if H is None:
        return None, float("nan")
    inliers = mask.ravel().astype(bool) if mask is not None else np.ones(len(image_pts), bool)
    proj = image_to_pitch(H, image_pts[inliers])
    err = float(np.median(np.linalg.norm(proj - pitch_pts[inliers], axis=1)))
    return H, err


def image_to_pitch(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply homography H to Nx2 points."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)
