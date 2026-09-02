"""Team classification from jersey colour."""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans

H_BINS, S_BINS = 16, 4


def jersey_feature(crop_bgr: np.ndarray) -> np.ndarray:
    """L1-normalised hue/saturation histogram of the torso region of a player crop."""
    h, w = crop_bgr.shape[:2]
    torso = crop_bgr[int(0.15 * h) : int(0.55 * h) or 1, int(0.2 * w) : int(0.8 * w) or 1]
    if torso.size == 0:
        torso = crop_bgr
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [H_BINS, S_BINS], [0, 180, 0, 256]).ravel()
    total = hist.sum()
    return hist / total if total > 0 else hist


class TeamClassifier:
    """Two-cluster KMeans over jersey features. Labels are 0 and 1."""

    def __init__(self, seed: int = 0):
        self._km = KMeans(n_clusters=2, n_init=10, random_state=seed)

    def fit(self, crops: list[np.ndarray]) -> "TeamClassifier":
        X = np.stack([jersey_feature(c) for c in crops])
        self._km.fit(X)
        return self

    def predict(self, crops: list[np.ndarray]) -> list[int]:
        X = np.stack([jersey_feature(c) for c in crops])
        return [int(v) for v in self._km.predict(X)]
