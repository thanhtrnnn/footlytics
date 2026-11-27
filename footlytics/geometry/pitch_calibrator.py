"""Experimental broadcast pitch keypoint & line calibrator (permissive code dependencies).

An optional calibration backend under evaluation alongside `ext/SoccerMaster` + `PnLCalib`
(`soccermaster` remains the default backend):
1. FIFA `105 x 68 m` 3D pitch keypoint (41 landmarks: 37 ground + 4 crossbar) and
   17 straight ground-line geometry catalog written directly against `sn_camera.py`.
2. `Apache2CameraSolver`: RANSAC homography initializer with line-intersection
   completion -> pinhole decomposition (`sn_camera.camera_from_homography`) ->
   joint 3D keypoint + point-to-line nonlinear Levenberg-Marquardt refinement
   (`scipy.optimize.least_squares` with robust `soft_l1` loss). Tested on synthetic cameras.
3. `PitchCalibNet` & `Apache2PitchCalibrator`: ResNet-FPN dual-head heatmap model
   (`torchvision` BSD-3-Clause + `safetensors` Apache-2.0 architecture; untrained/unscored on
   real broadcast footage until trained weights are provided) implementing the
   `fit(frames_rgb, which=..., ...)` -> `dict[int, CameraFit | None]` interface used
   by `ClipCalibrator`, `pipeline/match.py`, `pipeline/clip.py`, and `bench/runner.py`.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares

from .calib_nn import CameraFit, valid_camera_fit
from .sn_camera import (
    HALF_L,
    HALF_W,
    camera_from_homography,
    ground_homography,
    ground_lines,
    make_camera,
    project,
)

_PA_DEPTH = 16.5
_PA_HALF_W = 20.16
_GA_DEPTH = 5.5
_GA_HALF_W = 9.16
_PEN_SPOT_DIST = 11.0
_CIRCLE_R = 9.15
_GOAL_HALF_W = 3.66
_GOAL_HEIGHT = 2.44
_ARC_Y = math.sqrt(_CIRCLE_R**2 - (_PA_DEPTH - _PEN_SPOT_DIST) ** 2)


def build_pitch_keypoints_3d() -> tuple[list[str], np.ndarray]:
    """Return `(names, coords_3d [K, 3])` in SoccerNet world metres (`z < 0` is UP)."""
    L, W = HALF_L, HALF_W
    pa_x = L - _PA_DEPTH
    ga_x = L - _GA_DEPTH
    r = _CIRCLE_R
    d45 = r / math.sqrt(2.0)

    items: list[tuple[str, tuple[float, float, float]]] = [
        # Outer corners & touchline/halfway intersections (6)
        ("corner_LT", (-L, -W, 0.0)),
        ("corner_LB", (-L, W, 0.0)),
        ("corner_RT", (L, -W, 0.0)),
        ("corner_RB", (L, W, 0.0)),
        ("halfway_T", (0.0, -W, 0.0)),
        ("halfway_B", (0.0, W, 0.0)),
        # Centre spot + 8 points around the 9.15 m centre circle (9)
        ("centre_spot", (0.0, 0.0, 0.0)),
        ("circle_T", (0.0, -r, 0.0)),
        ("circle_B", (0.0, r, 0.0)),
        ("circle_L", (-r, 0.0, 0.0)),
        ("circle_R", (r, 0.0, 0.0)),
        ("circle_LT", (-d45, -d45, 0.0)),
        ("circle_LB", (-d45, d45, 0.0)),
        ("circle_RT", (d45, -d45, 0.0)),
        ("circle_RB", (d45, d45, 0.0)),
        # Left penalty & goal areas + penalty spot & arc intersections (11)
        ("pa_LT_goal", (-L, -_PA_HALF_W, 0.0)),
        ("pa_LB_goal", (-L, _PA_HALF_W, 0.0)),
        ("pa_LT_front", (-pa_x, -_PA_HALF_W, 0.0)),
        ("pa_LB_front", (-pa_x, _PA_HALF_W, 0.0)),
        ("ga_LT_goal", (-L, -_GA_HALF_W, 0.0)),
        ("ga_LB_goal", (-L, _GA_HALF_W, 0.0)),
        ("ga_LT_front", (-ga_x, -_GA_HALF_W, 0.0)),
        ("ga_LB_front", (-ga_x, _GA_HALF_W, 0.0)),
        ("pen_spot_L", (-L + _PEN_SPOT_DIST, 0.0, 0.0)),
        ("pen_arc_LT", (-pa_x, -_ARC_Y, 0.0)),
        ("pen_arc_LB", (-pa_x, _ARC_Y, 0.0)),
        # Right penalty & goal areas + penalty spot & arc intersections (11)
        ("pa_RT_goal", (L, -_PA_HALF_W, 0.0)),
        ("pa_RB_goal", (L, _PA_HALF_W, 0.0)),
        ("pa_RT_front", (pa_x, -_PA_HALF_W, 0.0)),
        ("pa_RB_front", (pa_x, _PA_HALF_W, 0.0)),
        ("ga_RT_goal", (L, -_GA_HALF_W, 0.0)),
        ("ga_RB_goal", (L, _GA_HALF_W, 0.0)),
        ("ga_RT_front", (ga_x, -_GA_HALF_W, 0.0)),
        ("ga_RB_front", (ga_x, _GA_HALF_W, 0.0)),
        ("pen_spot_R", (L - _PEN_SPOT_DIST, 0.0, 0.0)),
        ("pen_arc_RT", (pa_x, -_ARC_Y, 0.0)),
        ("pen_arc_RB", (pa_x, _ARC_Y, 0.0)),
        # Goalpost crossbars (4 keypoints at z = -2.44 m)
        ("crossbar_LT", (-L, -_GOAL_HALF_W, -_GOAL_HEIGHT)),
        ("crossbar_LB", (-L, _GOAL_HALF_W, -_GOAL_HEIGHT)),
        ("crossbar_RT", (L, -_GOAL_HALF_W, -_GOAL_HEIGHT)),
        ("crossbar_RB", (L, _GOAL_HALF_W, -_GOAL_HEIGHT)),
    ]
    names = [k for k, _ in items]
    coords = np.array([v for _, v in items], dtype=np.float64)
    return names, coords


KEYPOINT_NAMES, PITCH_KEYPOINTS_3D = build_pitch_keypoints_3d()
KEYPOINT_INDEX: dict[str, int] = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
NUM_PITCH_KEYPOINTS = len(KEYPOINT_NAMES)  # 41

_RAW_LINES = ground_lines()
LINE_NAMES: list[str] = list(_RAW_LINES.keys())
LINE_INDEX: dict[str, int] = {name: i for i, name in enumerate(LINE_NAMES)}
NUM_PITCH_LINES = len(LINE_NAMES)  # 17
PITCH_LINES_2D: np.ndarray = np.array(
    [[a, b] for a, b in _RAW_LINES.values()], dtype=np.float64
)  # [17, 2, 2]

# Pairs of straight pitch lines whose geometric intersection defines a canonical ground keypoint
LINE_INTERSECTION_KP: dict[tuple[str, str], str] = {
    ("Side line top", "Side line left"): "corner_LT",
    ("Side line bottom", "Side line left"): "corner_LB",
    ("Side line top", "Side line right"): "corner_RT",
    ("Side line bottom", "Side line right"): "corner_RB",
    ("Side line top", "Middle line"): "halfway_T",
    ("Side line bottom", "Middle line"): "halfway_B",
    ("Big rect. left top", "Side line left"): "pa_LT_goal",
    ("Big rect. left bottom", "Side line left"): "pa_LB_goal",
    ("Big rect. left top", "Big rect. left main"): "pa_LT_front",
    ("Big rect. left bottom", "Big rect. left main"): "pa_LB_front",
    ("Small rect. left top", "Side line left"): "ga_LT_goal",
    ("Small rect. left bottom", "Side line left"): "ga_LB_goal",
    ("Small rect. left top", "Small rect. left main"): "ga_LT_front",
    ("Small rect. left bottom", "Small rect. left main"): "ga_LB_front",
    ("Big rect. right top", "Side line right"): "pa_RT_goal",
    ("Big rect. right bottom", "Side line right"): "pa_RB_goal",
    ("Big rect. right top", "Big rect. right main"): "pa_RT_front",
    ("Big rect. right bottom", "Big rect. right main"): "pa_RB_front",
    ("Small rect. right top", "Side line right"): "ga_RT_goal",
    ("Small rect. right bottom", "Side line right"): "ga_RB_goal",
    ("Small rect. right top", "Small rect. right main"): "ga_RT_front",
    ("Small rect. right bottom", "Small rect. right main"): "ga_RB_front",
}


def _line_intersection_2d(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> tuple[float, float] | None:
    """Intersection of infinite lines (p1,p2) and (p3,p4), or None if near-parallel."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    n1 = x1 * y2 - y1 * x2
    n2 = x3 * y4 - y3 * x4
    ix = (n1 * (x3 - x4) - (x1 - x2) * n2) / denom
    iy = (n1 * (y3 - y4) - (y1 - y2) * n2) / denom
    if not (math.isfinite(ix) and math.isfinite(iy)):
        return None
    return (float(ix), float(iy))


def complete_keypoints_from_lines(
    keypoints: dict[int, tuple[float, float, float]],
    lines: dict[int, tuple[tuple[float, float], tuple[float, float], float]],
    width: int,
    height: int,
) -> dict[int, tuple[float, float, float]]:
    """Synthesize missing ground keypoints by intersecting pairs of detected pitch lines."""
    out = dict(keypoints)
    margin = 0.25 * max(width, height)
    for (la_name, lb_name), kp_name in LINE_INTERSECTION_KP.items():
        kp_idx = KEYPOINT_INDEX[kp_name]
        if kp_idx in out:
            continue
        ia = LINE_INDEX[la_name]
        ib = LINE_INDEX[lb_name]
        if ia not in lines or ib not in lines:
            continue
        a1, a2, sa = lines[ia]
        b1, b2, sb = lines[ib]
        pt = _line_intersection_2d(a1, a2, b1, b2)
        if pt is None:
            continue
        if -margin <= pt[0] <= width + margin and -margin <= pt[1] <= height + margin:
            out[kp_idx] = (pt[0], pt[1], 0.75 * min(sa, sb))
    return out


class Apache2CameraSolver:
    """Homography + pinhole camera Levenberg-Marquardt solver (no external PnLCalib code)."""

    def solve(
        self,
        keypoints: dict[int, tuple[float, float, float]],
        lines: dict[int, tuple[tuple[float, float], tuple[float, float], float]],
        width: int,
        height: int,
        *,
        ransac_px: float = 10.0,
        refine_lines: bool = True,
        max_nfev: int = 80,
    ) -> CameraFit | None:
        """Estimate a SoccerNet-compatible `CameraFit` from pixel keypoints & lines."""
        merged_kp = complete_keypoints_from_lines(keypoints, lines, width, height)

        ground_world: list[tuple[float, float]] = []
        ground_img: list[tuple[float, float]] = []
        ground_w: list[float] = []
        all_ids: list[int] = []
        all_img: list[tuple[float, float]] = []
        all_w: list[float] = []

        for kp_id, (u, v, score) in merged_kp.items():
            if not (0 <= kp_id < NUM_PITCH_KEYPOINTS):
                continue
            if not (math.isfinite(u) and math.isfinite(v) and score > 0.0):
                continue
            wx, wy, wz = PITCH_KEYPOINTS_3D[kp_id]
            all_ids.append(kp_id)
            all_img.append((float(u), float(v)))
            all_w.append(float(score))
            if abs(wz) < 1e-6:
                ground_world.append((float(wx), float(wy)))
                ground_img.append((float(u), float(v)))
                ground_w.append(float(score))

        if len(ground_world) < 4:
            return None

        src_g = np.array(ground_world, dtype=np.float64)
        dst_g = np.array(ground_img, dtype=np.float64)

        # Try RANSAC across two thresholds if >= 5 ground points, else exact DLT
        best_cam: dict[str, Any] | None = None
        best_err = float("inf")
        thresholds = [ransac_px, max(4.0, ransac_px * 0.5)] if len(src_g) >= 5 else [0.0]

        for thr in thresholds:
            if thr > 0.0 and len(src_g) >= 5:
                H_p2i, _ = cv2.findHomography(src_g, dst_g, cv2.RANSAC, thr)
            else:
                H_p2i, _ = cv2.findHomography(src_g, dst_g, 0)
            if H_p2i is None or not np.isfinite(H_p2i).all():
                continue
            cam0 = camera_from_homography(H_p2i, float(width), float(height))
            if cam0 is None:
                continue
            proj0 = project(cam0, PITCH_KEYPOINTS_3D[all_ids])
            if not np.isfinite(proj0).all():
                continue
            err0 = float(np.median(np.linalg.norm(proj0 - np.array(all_img), axis=1)))
            if err0 < best_err:
                best_err = err0
                best_cam = cam0

        if best_cam is None:
            return None

        # Refine 7 pinhole parameters jointly against 3D keypoints + line endpoints
        kp_pts3d = PITCH_KEYPOINTS_3D[all_ids]
        kp_uv = np.array(all_img, dtype=np.float64)
        kp_weights = np.sqrt(np.clip(np.array(all_w, dtype=np.float64), 0.1, 1.0))

        line_items: list[tuple[np.ndarray, np.ndarray, float]] = []
        if refine_lines and lines:
            for lid, (e1, e2, sc) in lines.items():
                if 0 <= lid < NUM_PITCH_LINES and sc > 0.0:
                    seg = PITCH_LINES_2D[lid]
                    p3 = np.array([[seg[0, 0], seg[0, 1], 0.0], [seg[1, 0], seg[1, 1], 0.0]], dtype=np.float64)
                    obs_ends = np.array([e1, e2], dtype=np.float64)
                    line_items.append((p3, obs_ends, math.sqrt(min(1.0, max(0.1, float(sc))))))

        x0 = np.array(
            [
                best_cam["pan_degrees"],
                best_cam["tilt_degrees"],
                best_cam["roll_degrees"],
                best_cam["x_focal_length"],
                best_cam["position_meters"][0],
                best_cam["position_meters"][1],
                best_cam["position_meters"][2],
            ],
            dtype=np.float64,
        )
        lb = np.array([-180.0, 20.0, -10.0, 0.45 * width, -120.0, -150.0, -100.0], dtype=np.float64)
        ub = np.array([180.0, 89.5, 10.0, 5.0 * width, 120.0, 150.0, -2.0], dtype=np.float64)
        x0 = np.clip(x0, lb + 1e-3, ub - 1e-3)

        def residuals(params: np.ndarray) -> np.ndarray:
            cam = make_camera(
                params[0], params[1], params[2], params[3], params[4:7], float(width), float(height)
            )
            proj = project(cam, kp_pts3d)
            res_kp = np.where(np.isfinite(proj), proj - kp_uv, 200.0)
            res_list = [(res_kp * kp_weights[:, None]).ravel()]

            if line_items:
                line_res: list[float] = []
                for p3, obs_ends, lw in line_items:
                    pr = project(cam, p3)
                    if not np.isfinite(pr).all():
                        line_res.extend([100.0 * lw, 100.0 * lw])
                        continue
                    v = pr[1] - pr[0]
                    norm_v = float(np.hypot(v[0], v[1]))
                    if norm_v < 1e-3:
                        line_res.extend([100.0 * lw, 100.0 * lw])
                        continue
                    nx, ny = -v[1] / norm_v, v[0] / norm_v
                    for ep in obs_ends:
                        d = (ep[0] - pr[0, 0]) * nx + (ep[1] - pr[0, 1]) * ny
                        line_res.append(0.65 * lw * float(d))
                res_list.append(np.array(line_res, dtype=np.float64))
            return np.concatenate(res_list)

        try:
            opt = least_squares(
                residuals,
                x0,
                bounds=(lb, ub),
                loss="soft_l1",
                f_scale=6.0,
                max_nfev=max_nfev,
            )
            p_opt = opt.x
            cam_final = make_camera(
                p_opt[0], p_opt[1], p_opt[2], p_opt[3], p_opt[4:7], float(width), float(height)
            )
        except Exception:  # noqa: BLE001
            cam_final = best_cam

        proj_final = project(cam_final, kp_pts3d)
        if not np.isfinite(proj_final).all():
            return None
        rep_err = float(np.mean(np.linalg.norm(proj_final - kp_uv, axis=1)))
        try:
            H_inv = np.linalg.inv(ground_homography(cam_final))
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(H_inv).all() or abs(float(H_inv[2, 2])) < 1e-12:
            return None
        H_inv = H_inv / H_inv[2, 2]
        fit = CameraFit(
            cam=cam_final,
            rep_err=rep_err,
            H_img_to_pitch=H_inv,
            n_keypoints=len(all_ids),
        )
        if not valid_camera_fit(fit, width, height):
            return None
        return fit


def decode_keypoint_heatmaps(
    kp_hm: np.ndarray,
    width: int,
    height: int,
    *,
    threshold: float = 0.30,
) -> dict[int, tuple[float, float, float]]:
    """Decode `[K, Hh, Wh]` sigmoid heatmaps -> `{kp_id: (u_px, v_px, score)}` with sub-pixel peak."""
    k_cnt, hh, wh = kp_hm.shape
    sx = width / float(wh)
    sy = height / float(hh)
    out: dict[int, tuple[float, float, float]] = {}
    for k in range(min(k_cnt, NUM_PITCH_KEYPOINTS)):
        hm = kp_hm[k]
        flat_idx = int(np.argmax(hm))
        score = float(hm.flat[flat_idx])
        if score < threshold:
            continue
        r, c = divmod(flat_idx, wh)
        # 1D parabolic log-Gaussian refinement along row & col for sub-pixel peak accuracy
        r_sub, c_sub = float(r), float(c)
        if 0 < r < hh - 1:
            lm = math.log(max(float(hm[r - 1, c]), 1e-7))
            l0 = math.log(max(float(hm[r, c]), 1e-7))
            lp = math.log(max(float(hm[r + 1, c]), 1e-7))
            denom = lm - 2.0 * l0 + lp
            if abs(denom) > 1e-6:
                r_sub += float(np.clip(0.5 * (lm - lp) / denom, -0.5, 0.5))
        if 0 < c < wh - 1:
            lm = math.log(max(float(hm[r, c - 1]), 1e-7))
            l0 = math.log(max(float(hm[r, c]), 1e-7))
            lp = math.log(max(float(hm[r, c + 1]), 1e-7))
            denom = lm - 2.0 * l0 + lp
            if abs(denom) > 1e-6:
                c_sub += float(np.clip(0.5 * (lm - lp) / denom, -0.5, 0.5))
        out[k] = ((c_sub + 0.5) * sx, (r_sub + 0.5) * sy, score)
    return out


def decode_line_heatmaps(
    line_hm: np.ndarray,
    width: int,
    height: int,
    *,
    threshold: float = 0.45,
    min_pixels: int = 8,
) -> dict[int, tuple[tuple[float, float], tuple[float, float], float]]:
    """Decode `[L, Hh, Wh]` line heatmaps -> `{line_id: ((u1,v1), (u2,v2), score)}` via weighted PCA."""
    l_cnt, hh, wh = line_hm.shape
    sx = width / float(wh)
    sy = height / float(hh)
    out: dict[int, tuple[tuple[float, float], tuple[float, float], float]] = {}
    for li in range(min(l_cnt, NUM_PITCH_LINES)):
        hm = line_hm[li]
        ys, xs = np.where(hm >= threshold)
        if len(xs) < min_pixels:
            continue
        w = hm[ys, xs].astype(np.float64)
        us = (xs.astype(np.float64) + 0.5) * sx
        vs = (ys.astype(np.float64) + 0.5) * sy
        pts = np.column_stack([us, vs])
        mean = (pts * w[:, None]).sum(axis=0) / w.sum()
        centered = pts - mean
        cov = (centered.T * w) @ centered / w.sum()
        eigvals, eigvecs = np.linalg.eigh(cov)
        if eigvals[1] < 25.0:  # span too tiny to define a reliable line direction
            continue
        axis = eigvecs[:, 1]
        proj = centered @ axis
        p_lo = mean + float(np.quantile(proj, 0.05)) * axis
        p_hi = mean + float(np.quantile(proj, 0.95)) * axis
        out[li] = ((float(p_lo[0]), float(p_lo[1])), (float(p_hi[0]), float(p_hi[1])), float(w.mean()))
    return out


def render_heatmap_targets(
    cam: dict[str, Any],
    src_w: int,
    src_h: int,
    hm_w: int,
    hm_h: int,
    *,
    kp_sigma: float = 2.2,
    line_thickness: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Render ground-truth `[41, hm_h, hm_w]` keypoint & `[17, hm_h, hm_w]` line training heatmaps."""
    sx = hm_w / float(src_w)
    sy = hm_h / float(src_h)

    kp_targets = np.zeros((NUM_PITCH_KEYPOINTS, hm_h, hm_w), dtype=np.float32)
    uv_all = project(cam, PITCH_KEYPOINTS_3D)
    yy, xx = np.meshgrid(np.arange(hm_h, dtype=np.float32), np.arange(hm_w, dtype=np.float32), indexing="ij")
    inv_2s2 = 1.0 / (2.0 * kp_sigma * kp_sigma)
    for k in range(NUM_PITCH_KEYPOINTS):
        u, v = uv_all[k]
        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        if not (0.0 <= u < src_w and 0.0 <= v < src_h):
            continue
        cx = u * sx - 0.5
        cy = v * sy - 0.5
        kp_targets[k] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) * inv_2s2)

    line_targets = np.zeros((NUM_PITCH_LINES, hm_h, hm_w), dtype=np.float32)
    for li in range(NUM_PITCH_LINES):
        a, b = PITCH_LINES_2D[li]
        ts = np.linspace(0.0, 1.0, 40)
        seg3d = np.column_stack(
            [a[0] + ts * (b[0] - a[0]), a[1] + ts * (b[1] - a[1]), np.zeros_like(ts)]
        )
        uv_seg = project(cam, seg3d)
        ok = (
            np.isfinite(uv_seg).all(axis=1)
            & (uv_seg[:, 0] >= -20)
            & (uv_seg[:, 0] <= src_w + 20)
            & (uv_seg[:, 1] >= -20)
            & (uv_seg[:, 1] <= src_h + 20)
        )
        if ok.sum() < 2:
            continue
        pts = np.round(np.column_stack([uv_seg[ok, 0] * sx, uv_seg[ok, 1] * sy])).astype(np.int32)
        mask = np.zeros((hm_h, hm_w), dtype=np.uint8)
        cv2.polylines(mask, [pts.reshape(-1, 1, 2)], False, 255, line_thickness, cv2.LINE_AA)
        line_targets[li] = mask.astype(np.float32) / 255.0

    return kp_targets, line_targets


@dataclass(frozen=True)
class PitchCalibConfig:
    backbone: str = "resnet34"
    input_size: tuple[int, int] = (512, 288)  # (width, height) 16:9 broadcast native aspect
    fpn_channels: int = 128
    num_keypoints: int = NUM_PITCH_KEYPOINTS
    num_lines: int = NUM_PITCH_LINES


def build_pitch_calib_net(cfg: PitchCalibConfig, *, pretrained_backbone: bool = False):
    """Build the ResNet-FPN dual-head pitch keypoint+line model (`torchvision` BSD-3-Clause)."""
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torchvision.models import ResNet34_Weights, ResNet50_Weights, resnet34, resnet50

    class PitchCalibNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if cfg.backbone == "resnet50":
                w = ResNet50_Weights.DEFAULT if pretrained_backbone else None
                bb = resnet50(weights=w)
                c_dims = (256, 512, 1024, 2048)
            elif cfg.backbone == "resnet34":
                w = ResNet34_Weights.DEFAULT if pretrained_backbone else None
                bb = resnet34(weights=w)
                c_dims = (64, 128, 256, 512)
            else:
                raise ValueError(f"unsupported backbone {cfg.backbone!r}; use 'resnet34' or 'resnet50'")

            self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
            self.layer1 = bb.layer1
            self.layer2 = bb.layer2
            self.layer3 = bb.layer3
            self.layer4 = bb.layer4

            ch = cfg.fpn_channels
            self.lat4 = nn.Conv2d(c_dims[3], ch, 1)
            self.lat3 = nn.Conv2d(c_dims[2], ch, 1)
            self.lat2 = nn.Conv2d(c_dims[1], ch, 1)
            self.lat1 = nn.Conv2d(c_dims[0], ch, 1)
            self.smooth = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            )
            self.kp_head = nn.Conv2d(ch, cfg.num_keypoints, 1)
            self.line_head = nn.Conv2d(ch, cfg.num_lines, 1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            s = self.stem(x)
            c1 = self.layer1(s)
            c2 = self.layer2(c1)
            c3 = self.layer3(c2)
            c4 = self.layer4(c3)

            p4 = self.lat4(c4)
            p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
            p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
            p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
            feat = self.smooth(p1)
            return self.kp_head(feat), self.line_head(feat)

    return PitchCalibNet()


def save_pitch_calibrator(model, cfg: PitchCalibConfig, out_dir: Path | str) -> Path:
    """Save `PitchCalibNet` weights (`model.safetensors`) + `config.json`."""
    from safetensors.torch import save_file

    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    cfg_dict = {
        "backbone": cfg.backbone,
        "input_size": list(cfg.input_size),
        "fpn_channels": cfg.fpn_channels,
        "num_keypoints": cfg.num_keypoints,
        "num_lines": cfg.num_lines,
        "code_license": "Apache-2.0",
    }
    (dst / "config.json").write_text(json.dumps(cfg_dict, indent=2) + "\n", encoding="utf-8")
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(dst / "model.safetensors"))
    return dst


class Apache2PitchCalibrator:
    """Optional ResNet-FPN broadcast pitch calibrator matching `SoccerMasterCalibrator.fit()`."""

    def __init__(
        self,
        weights_dir: Path | str,
        *,
        device: str | None = None,
        batch: int = 8,
    ) -> None:
        import torch
        from safetensors.torch import load_file

        wdir = Path(weights_dir)
        cfg_raw = json.loads((wdir / "config.json").read_text(encoding="utf-8"))
        self.cfg = PitchCalibConfig(
            backbone=str(cfg_raw.get("backbone", "resnet34")),
            input_size=(int(cfg_raw["input_size"][0]), int(cfg_raw["input_size"][1])),
            fpn_channels=int(cfg_raw.get("fpn_channels", 128)),
            num_keypoints=int(cfg_raw.get("num_keypoints", NUM_PITCH_KEYPOINTS)),
            num_lines=int(cfg_raw.get("num_lines", NUM_PITCH_LINES)),
        )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        self.batch = max(1, int(batch))
        self.model = build_pitch_calib_net(self.cfg, pretrained_backbone=False)
        state = load_file(str(wdir / "model.safetensors"))
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.solver = Apache2CameraSolver()
        self.last_errors: dict[int, str] = {}

        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def heatmaps(self, frames_rgb: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Run batched ResNet-FPN inference -> `(kp_probs [N, K, Hh, Wh], line_probs [N, L, Hh, Wh])`."""
        import torch

        frames = list(frames_rgb)
        if not frames:
            raise ValueError("pass at least one frame")
        in_w, in_h = self.cfg.input_size
        kps, lns = [], []
        with torch.no_grad():
            for i in range(0, len(frames), self.batch):
                chunk = frames[i : i + self.batch]
                arr = np.stack(
                    [cv2.resize(f, (in_w, in_h), interpolation=cv2.INTER_LINEAR) for f in chunk],
                    axis=0,
                ).astype(np.float32) / 255.0
                arr = np.transpose(arr, (0, 3, 1, 2))
                arr = (arr - self._mean) / self._std
                x = torch.from_numpy(arr).to(self.device)
                kp_logits, ln_logits = self.model(x)
                kps.append(torch.sigmoid(kp_logits).cpu().numpy())
                lns.append(torch.sigmoid(ln_logits).cpu().numpy())
        return np.concatenate(kps, axis=0), np.concatenate(lns, axis=0)

    def fit(
        self,
        frames_rgb: Sequence[np.ndarray],
        which: Sequence[int] | None = None,
        kp_threshold: float = 0.30,
        line_threshold: float = 0.45,
        **solve_kw: Any,
    ) -> dict[int, CameraFit | None]:
        """Drop-in replacement for `SoccerMasterCalibrator.fit()`."""
        self.last_errors = {}
        frames = list(frames_rgb)
        if not frames:
            return {}
        h, w = frames[0].shape[:2]
        idxs = list(range(len(frames)) if which is None else which)
        if not idxs:
            return {}

        kp_hms, ln_hms = self.heatmaps([frames[i] for i in idxs])
        ransac_px = float(solve_kw.get("ransac", 10.0) or 10.0)
        refine_lines = bool(solve_kw.get("refine_lines", True))

        out: dict[int, CameraFit | None] = {}
        for pos, frame_idx in enumerate(idxs):
            try:
                kps = decode_keypoint_heatmaps(kp_hms[pos], w, h, threshold=kp_threshold)
                lns = decode_line_heatmaps(ln_hms[pos], w, h, threshold=line_threshold)
                out[frame_idx] = self.solver.solve(
                    kps, lns, w, h, ransac_px=ransac_px, refine_lines=refine_lines
                )
            except Exception as e:  # noqa: BLE001
                out[frame_idx] = None
                self.last_errors[frame_idx] = f"{type(e).__name__}: {e}"
        return out
