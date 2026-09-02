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


class AnchoredCamera:
    """Camera motion relative to one anchor frame, robust to shot cuts.

    Per frame: KLT step from the previous frame (fast). Periodically, and whenever KLT fails
    (few inliers -> likely a cut or a blur), re-anchor by global SIFT matching to the anchor
    frame; if that fails too the frame is marked invalid. `H_anchor_to_t` maps anchor pixels
    to current pixels; use H_t = H_anchor @ inv(H_anchor_to_t).
    """

    def __init__(self, anchor_bgr: np.ndarray, mask: np.ndarray | None = None, reanchor_every: int = 50,
                 min_klt_inliers: int = 12, min_klt_ratio: float = 0.5, min_global_inliers: int = 30,
                 min_global_ratio: float = 0.5, start_at_anchor: bool = True):
        self._anchor_gray = cv2.cvtColor(anchor_bgr, cv2.COLOR_BGR2GRAY) if anchor_bgr.ndim == 3 else anchor_bgr
        self._sift = cv2.SIFT_create(nfeatures=3000)
        self._kp_a, self._des_a = self._sift.detectAndCompute(self._anchor_gray, mask)
        self._matcher = cv2.BFMatcher()
        self._reanchor_every = reanchor_every
        self._min_klt, self._min_klt_ratio = min_klt_inliers, min_klt_ratio
        self._min_glob, self._min_glob_ratio = min_global_inliers, min_global_ratio
        self.H_anchor_to_t = np.eye(3)
        self.valid = start_at_anchor
        self._prev_gray = self._anchor_gray if start_at_anchor else None
        self._count = 0
        self.n_reanchors = 0
        self.n_tracked = 0

    def _klt_step(self, gray: np.ndarray, mask: np.ndarray | None) -> np.ndarray | None:
        p0 = cv2.goodFeaturesToTrack(self._prev_gray, mask=mask, **_FEATURE_PARAMS)
        if p0 is None or len(p0) < self._min_klt:
            return None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, p0, None, **_LK_PARAMS)
        ok = st.ravel() == 1
        q0, q1 = p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)
        if len(q0) < self._min_klt:
            return None
        step, inl = cv2.findHomography(q0, q1, cv2.RANSAC, 2.0)
        n = int(inl.sum()) if inl is not None else 0
        self.n_tracked = n
        if step is None or n < self._min_klt or n / len(q0) < self._min_klt_ratio:
            return None
        return step

    def _global(self, gray: np.ndarray, mask: np.ndarray | None) -> np.ndarray | None:
        kp, des = self._sift.detectAndCompute(gray, mask)
        if des is None or self._des_a is None or len(kp) < 8:
            return None
        matches = self._matcher.knnMatch(self._des_a, des, k=2)
        good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < self._min_glob:
            return None
        src = np.float32([self._kp_a[g.queryIdx].pt for g in good])
        dst = np.float32([kp[g.trainIdx].pt for g in good])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        n = int(inl.sum()) if inl is not None else 0
        if H is None or n < self._min_glob or n / len(good) < self._min_glob_ratio:
            return None
        return H

    def update(self, frame: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        self._count += 1
        step = self._klt_step(gray, mask) if (self.valid and self._prev_gray is not None) else None
        if step is not None:
            self.H_anchor_to_t = step @ self.H_anchor_to_t
        else:
            self.valid = False
        if not self.valid or self._count % self._reanchor_every == 0:
            Hg = self._global(gray, mask)
            if Hg is not None:
                self.H_anchor_to_t = Hg
                self.valid = True
                self.n_reanchors += 1
        self._prev_gray = gray
        return self.H_anchor_to_t if self.valid else None
