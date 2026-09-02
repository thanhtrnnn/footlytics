"""Overlay the pitch model on chosen frames using the anchored camera, and list invalid frame runs.

Usage: python scripts/drift_check.py CLIP CALIB_JSON OUT_DIR FRAME [FRAME ...]
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from footlytics.calibration import anchor_frame_of, static_homography
from footlytics.camera_motion import AnchoredCamera
from footlytics.homography import image_to_pitch
from footlytics.ingest import iter_frames
from footlytics.pipeline import _read_frame
from footlytics.pitch import PITCH_VERTICES_M

EDGES = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8), (10, 11), (11, 12), (12, 13), (14, 15), (15, 16), (16, 17),
         (18, 19), (19, 20), (20, 21), (23, 24), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (1, 14), (2, 10),
         (3, 7), (4, 8), (5, 13), (6, 17), (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30)]


def draw_model(frame, H_img_to_pitch):
    Hi = np.linalg.inv(H_img_to_pitch)
    vis = frame.copy()
    V = image_to_pitch(Hi, np.asarray(PITCH_VERTICES_M))
    for a, b in EDGES:
        p, q = V[a - 1], V[b - 1]
        if np.all(np.abs([p, q]) < 6000):
            cv2.line(vis, tuple(p.astype(int)), tuple(q.astype(int)), (255, 0, 255), 2)
    ang = np.linspace(0, 2 * np.pi, 90)
    circ = np.c_[52.5 + 9.15 * np.cos(ang), 34 + 9.15 * np.sin(ang)]
    cv2.polylines(vis, [image_to_pitch(Hi, circ).astype(np.int32).reshape(-1, 1, 2)], True, (255, 0, 255), 2)
    return vis


def main():
    clip, calib, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    wanted = {int(x) for x in sys.argv[4:]}
    out.mkdir(parents=True, exist_ok=True)
    H_anchor, _ = static_homography(calib)
    anchor = anchor_frame_of(calib)
    cam = AnchoredCamera(_read_frame(clip, anchor), start_at_anchor=(anchor == 0))
    invalid = []
    for idx, fr in iter_frames(clip):
        H_a_t = cam.update(fr)
        if H_a_t is None:
            invalid.append(idx)
            continue
        if idx in wanted:
            cv2.imwrite(str(out / f"drift_{idx}.jpg"), draw_model(fr, H_anchor @ np.linalg.inv(H_a_t)))
    runs, s, p = [], None, None
    for i in invalid:
        if s is None:
            s = p = i
        elif i == p + 1:
            p = i
        else:
            runs.append((s, p)); s = p = i
    if s is not None:
        runs.append((s, p))
    print(f"invalid frames: {len(invalid)}; runs: {runs}; reanchors: {cam.n_reanchors}")


if __name__ == "__main__":
    main()
