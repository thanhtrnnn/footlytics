"""Frame-to-frame camera motion as an accumulated homography (KLT + RANSAC).

Use: H_t(image_t -> pitch) = H_0(image_0 -> pitch) @ inv(H_0_to_t).
Features are re-detected every frame on the previous image, restricted by an optional mask
(e.g. exclude player boxes). Falls back to the previous motion when tracking is weak.
"""
from __future__ import annotations

import cv2
import numpy as np

_FEATURE_PARAMS = dict(maxCorners=600, qualityLevel=0.01, minDistance=8, blockSize=7)
_LK_PARAMS = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


class CameraMotion:
    def __init__(self, min_tracked: int = 12, ransac_thresh_px: float = 2.0):
        self._prev_gray: np.ndarray | None = None
        self.H_0_to_t = np.eye(3)
        self.n_tracked = 0
        self._min_tracked = min_tracked
        self._ransac = ransac_thresh_px

    def update(self, frame: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self._prev_gray is None:
            self._prev_gray = gray
            self.n_tracked = 0
            return self.H_0_to_t
        p0 = cv2.goodFeaturesToTrack(self._prev_gray, mask=mask, **_FEATURE_PARAMS)
        step = None
        if p0 is not None and len(p0) >= self._min_tracked:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, p0, None, **_LK_PARAMS)
            ok = st.ravel() == 1
            q0, q1 = p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)
            if len(q0) >= self._min_tracked:
                step, inl = cv2.findHomography(q0, q1, cv2.RANSAC, self._ransac)
                self.n_tracked = int(inl.sum()) if inl is not None else 0
                if step is None or self.n_tracked < self._min_tracked:
                    step = None
        if step is not None:
            self.H_0_to_t = step @ self.H_0_to_t
        else:
            self.n_tracked = 0  # keep previous accumulated motion
        self._prev_gray = gray
        return self.H_0_to_t
