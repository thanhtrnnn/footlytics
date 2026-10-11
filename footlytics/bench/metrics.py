"""Our own auxiliary metrics. Official scores come from the evaluators (evaluators.py)."""
from __future__ import annotations

import numpy as np

from ..geometry.sn_camera import ground_lines, project, scale_camera
from .adapters.calib import mirror

_SAMPLES = 60


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def _distances(lines: dict, cam: dict, width: int, height: int) -> list[float]:
    model = ground_lines()
    out: list[float] = []
    for name, pts in lines.items():
        if name not in model or not pts:
            continue
        a, b = (np.array(p) for p in model[name])
        w = np.linspace(0, 1, _SAMPLES)[:, None]
        poly = project(cam, np.c_[(1 - w) * a + w * b, np.zeros(_SAMPLES)])
        poly = poly[np.isfinite(poly).all(1)]           # drop the part behind the camera
        if len(poly) < 2:
            continue
        gt = np.array([[q["x"] * (width - 1), q["y"] * (height - 1)] for q in pts])
        d = np.min([_seg_dist(gt, poly[i], poly[i + 1]) for i in range(len(poly) - 1)], axis=0)
        out.extend(d.tolist())
    return out


def median_reprojection_px(gt_lines: dict, cam: dict, width: int = 960,
                           height: int = 540) -> float | None:
    """Median distance (px at width x height) from each labelled point of a straight
    ground-plane line to the camera's projection of that line, taking the better of the labels
    and their mirror as the official evaluator does. None if no such line is labelled."""
    c = scale_camera(cam, width)
    meds = [float(np.median(d)) for d in (_distances(gt_lines, c, width, height),
                                          _distances(mirror(gt_lines), c, width, height)) if d]
    return round(min(meds), 3) if meds else None
