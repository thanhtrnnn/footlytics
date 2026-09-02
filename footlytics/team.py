"""Team classification from jersey colour.

Feature: H/S/V histogram of torso pixels with grass pixels masked out. Classifier: KMeans with
up to three clusters; the two largest clusters are teams 0 and 1, the remaining cluster is
"ref" (referees, or whoever wears a third colour).
"""
from __future__ import annotations

import numpy as np
import cv2
from sklearn.cluster import KMeans

H_BINS, S_BINS, V_BINS = 8, 4, 4
GRASS_H = (35, 85)  # OpenCV hue range for green turf
GRASS_MIN_S = 50
MIN_KIT_PIXELS = 12


def _torso(crop_bgr: np.ndarray) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    t = crop_bgr[int(0.15 * h) : max(int(0.55 * h), 1), int(0.2 * w) : max(int(0.8 * w), 1)]
    return t if t.size else crop_bgr


def jersey_feature(crop_bgr: np.ndarray) -> np.ndarray:
    """L1-normalised HSV histogram of non-grass torso pixels."""
    hsv = cv2.cvtColor(_torso(crop_bgr), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hch, sch = hsv[:, 0], hsv[:, 1]
    grass = (hch >= GRASS_H[0]) & (hch <= GRASS_H[1]) & (sch >= GRASS_MIN_S)
    kit = hsv[~grass]
    if len(kit) < MIN_KIT_PIXELS:
        kit = hsv
    hist, _ = np.histogramdd(
        kit.astype(np.float64), bins=(H_BINS, S_BINS, V_BINS), range=((0, 180), (0, 256), (0, 256))
    )
    hist = hist.ravel()
    total = hist.sum()
    return hist / total if total > 0 else hist


class TeamClassifier:
    """KMeans(<=3) over jersey features -> labels 0, 1 (teams) or "ref"."""

    def __init__(self, seed: int = 0, max_clusters: int = 3):
        self._seed = seed
        self._max_clusters = max_clusters
        self._km: KMeans | None = None
        self._label_map: dict[int, int | str] = {}

    def fit(self, crops: list[np.ndarray]) -> "TeamClassifier":
        X = np.stack([jersey_feature(c) for c in crops])
        k = int(min(self._max_clusters, len(np.unique(X.round(6), axis=0)), len(X)))
        k = max(k, 1)
        self._km = KMeans(n_clusters=k, n_init=10, random_state=self._seed).fit(X)
        sizes = np.bincount(self._km.labels_, minlength=k)
        order = np.argsort(-sizes)
        self._label_map = {int(c): "ref" for c in range(k)}
        for team, c in enumerate(order[:2]):
            self._label_map[int(c)] = team
        return self

    def predict(self, crops: list[np.ndarray]) -> list[int | str]:
        if self._km is None:
            raise RuntimeError("fit first")
        X = np.stack([jersey_feature(c) for c in crops])
        return [self._label_map[int(v)] for v in self._km.predict(X)]
