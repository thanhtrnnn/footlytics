"""Static (per-clip) manual calibration from a JSON file of image points <-> pitch landmarks.

Use when the camera is fixed (tactical cam) or as a fallback when automatic keypoints fail.
JSON format:
    {"points": [{"vertex": 14, "x": 431.0, "y": 152.0}, {"vertex": "right_corner_top", "x": ..., "y": ...}]}
`vertex` is either a 1-based roboflow pitch vertex id (1..32) or a named landmark below.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from footlytics.homography import PITCH_LENGTH_M, PITCH_WIDTH_M, fit_homography
from footlytics.pitch import PITCH_VERTICES_M

L, W = PITCH_LENGTH_M, PITCH_WIDTH_M
# Pitch frame: x along length (0 = left goal line), y along width (0 = top touchline as seen
# from the main camera side, W = bottom touchline). "top"/"bottom" follow the roboflow vertex order.
NAMED_LANDMARKS: dict[str, tuple[float, float]] = {
    "left_corner_top": PITCH_VERTICES_M[0],
    "left_corner_bottom": PITCH_VERTICES_M[5],
    "halfway_top": PITCH_VERTICES_M[13],
    "centre_circle_top": PITCH_VERTICES_M[14],
    "centre_circle_bottom": PITCH_VERTICES_M[15],
    "halfway_bottom": PITCH_VERTICES_M[16],
    "centre_spot": (L / 2, W / 2),
    "centre_circle_left": PITCH_VERTICES_M[30],
    "centre_circle_right": PITCH_VERTICES_M[31],
    "right_box_top_left": PITCH_VERTICES_M[17],
    "right_goalbox_top_left": PITCH_VERTICES_M[18],
    "right_goalbox_bottom_left": PITCH_VERTICES_M[19],
    "right_box_bottom_left": PITCH_VERTICES_M[20],
    "right_penalty_spot": PITCH_VERTICES_M[21],
    "right_goalbox_top_right": PITCH_VERTICES_M[22],
    "right_goalbox_bottom_right": PITCH_VERTICES_M[23],
    "right_corner_top": PITCH_VERTICES_M[24],
    "right_box_top_right": PITCH_VERTICES_M[25],
    "right_goal_top": PITCH_VERTICES_M[26],
    "right_goal_bottom": PITCH_VERTICES_M[27],
    "right_box_bottom_right": PITCH_VERTICES_M[28],
    "right_corner_bottom": PITCH_VERTICES_M[29],
    "left_box_top_left": PITCH_VERTICES_M[1],
    "left_box_bottom_left": PITCH_VERTICES_M[4],
    "left_box_top_right": PITCH_VERTICES_M[9],
    "left_box_bottom_right": PITCH_VERTICES_M[12],
}


def _resolve(vertex: int | str) -> tuple[float, float]:
    if isinstance(vertex, int) or (isinstance(vertex, str) and vertex.isdigit()):
        i = int(vertex)
        if not 1 <= i <= 32:
            raise ValueError(f"vertex id out of range: {i}")
        return PITCH_VERTICES_M[i - 1]
    if vertex in NAMED_LANDMARKS:
        return NAMED_LANDMARKS[vertex]
    raise ValueError(f"unknown landmark: {vertex}")


def load_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(Path(path).read_text())
    pts = data["points"]
    img = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float64)
    pitch = np.array([_resolve(p["vertex"]) for p in pts], dtype=np.float64)
    return img, pitch


def static_homography(path: str | Path, ransac_thresh_m: float = 1.5) -> tuple[np.ndarray, float]:
    img, pitch = load_calibration(path)
    H, err = fit_homography(img, pitch, ransac_thresh_m=ransac_thresh_m)
    if H is None:
        raise ValueError("calibration needs at least 4 non-degenerate points")
    return H, err


def anchor_frame_of(path: str | Path) -> int:
    """Frame index the landmarks were annotated on (JSON key "frame", default 0)."""
    return int(json.loads(Path(path).read_text()).get("frame", 0))


def compose_from_anchor(H_anchor: np.ndarray, motions: dict[int, np.ndarray], anchor_frame: int) -> dict[int, np.ndarray]:
    """Given per-frame accumulated camera motion H_0_to_t, return image_t -> pitch homographies.

    H_t = H_anchor @ inv(H_anchor_to_t),  H_anchor_to_t = H_0_to_t @ inv(H_0_to_anchor).
    """
    if anchor_frame not in motions:
        raise KeyError(f"anchor frame {anchor_frame} not in motions")
    inv_anchor = np.linalg.inv(motions[anchor_frame])
    out: dict[int, np.ndarray] = {}
    for t, H_0_to_t in motions.items():
        H_anchor_to_t = H_0_to_t @ inv_anchor
        out[t] = H_anchor @ np.linalg.inv(H_anchor_to_t)
    return out
