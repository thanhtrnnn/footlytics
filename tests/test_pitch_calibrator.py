"""Unit & integration tests for the ResNet-FPN pitch calibrator and camera solver."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from footlytics.geometry.clip_calib import ClipCalibrator
from footlytics.geometry.homography import apply_homography
from footlytics.geometry.pitch_calibrator import (
    KEYPOINT_INDEX,
    LINE_INDEX,
    NUM_PITCH_KEYPOINTS,
    NUM_PITCH_LINES,
    PITCH_KEYPOINTS_3D,
    PITCH_LINES_2D,
    Apache2CameraSolver,
    Apache2PitchCalibrator,
    PitchCalibConfig,
    build_pitch_calib_net,
    complete_keypoints_from_lines,
    decode_keypoint_heatmaps,
    decode_line_heatmaps,
    render_heatmap_targets,
    save_pitch_calibrator,
)
from footlytics.geometry.sn_camera import make_camera, project


def _broadcast_camera(width: int = 1280, height: int = 720, pan: float = 12.0) -> dict:
    return make_camera(
        pan=pan,
        tilt=68.0,
        roll=0.5,
        focal=1550.0,
        position=[0.0, 62.0, -21.0],
        width=width,
        height=height,
    )


def test_solver_recovers_broadcast_camera_from_noisy_keypoints_and_lines() -> None:
    w, h = 1280, 720
    cam_gt = _broadcast_camera(w, h, pan=14.0)
    uv = project(cam_gt, PITCH_KEYPOINTS_3D)
    rng = np.random.default_rng(42)

    kps: dict[int, tuple[float, float, float]] = {}
    for idx, (u, v) in enumerate(uv):
        if np.isfinite(u) and np.isfinite(v) and 15 <= u <= w - 15 and 15 <= v <= h - 15:
            kps[idx] = (float(u + rng.normal(0.0, 0.8)), float(v + rng.normal(0.0, 0.8)), 0.92)
    assert len(kps) >= 8

    lines: dict[int, tuple[tuple[float, float], tuple[float, float], float]] = {}
    for lid, seg in enumerate(PITCH_LINES_2D):
        p3 = np.array([[seg[0, 0], seg[0, 1], 0.0], [seg[1, 0], seg[1, 1], 0.0]])
        pr = project(cam_gt, p3)
        if np.isfinite(pr).all() and np.all((pr[:, 0] >= -50) & (pr[:, 0] <= w + 50)):
            lines[lid] = ((float(pr[0, 0]), float(pr[0, 1])), (float(pr[1, 0]), float(pr[1, 1])), 0.85)

    solver = Apache2CameraSolver()
    fit = solver.solve(kps, lines, w, h, refine_lines=True)
    assert fit is not None
    assert fit.n_keypoints is not None and fit.n_keypoints >= len(kps)
    assert fit.rep_err < 2.5

    # Verify metric accuracy across a test grid projected onto the pitch plane
    test_world = np.array([[-30.0, -15.0], [-10.0, 0.0], [0.0, 10.0], [20.0, -12.0], [35.0, 5.0]])
    test_uv = project(cam_gt, np.column_stack([test_world, np.zeros(len(test_world))]))
    in_view = np.isfinite(test_uv).all(axis=1) & (test_uv[:, 0] >= 0) & (test_uv[:, 0] <= w)
    pred_world = apply_homography(fit.H_img_to_pitch, test_uv[in_view])
    err_m = np.linalg.norm(pred_world - test_world[in_view], axis=1)
    assert float(np.median(err_m)) < 0.25


def test_line_intersection_completion_synthesizes_missing_corners() -> None:
    w, h = 1280, 720
    cam_gt = _broadcast_camera(w, h, pan=-18.0)

    def proj_line(name: str) -> tuple[tuple[float, float], tuple[float, float], float]:
        seg = PITCH_LINES_2D[LINE_INDEX[name]]
        pr = project(cam_gt, np.array([[seg[0, 0], seg[0, 1], 0.0], [seg[1, 0], seg[1, 1], 0.0]]))
        return ((float(pr[0, 0]), float(pr[0, 1])), (float(pr[1, 0]), float(pr[1, 1])), 0.9)

    lines = {
        LINE_INDEX["Big rect. left top"]: proj_line("Big rect. left top"),
        LINE_INDEX["Big rect. left main"]: proj_line("Big rect. left main"),
    }
    completed = complete_keypoints_from_lines({}, lines, w, h)
    kp_idx = KEYPOINT_INDEX["pa_LT_front"]
    assert kp_idx in completed
    gt_uv = project(cam_gt, PITCH_KEYPOINTS_3D[[kp_idx]])[0]
    err_px = float(np.hypot(completed[kp_idx][0] - gt_uv[0], completed[kp_idx][1] - gt_uv[1]))
    assert err_px < 1e-3


def test_heatmap_render_decode_and_solve_roundtrip() -> None:
    w, h = 1280, 720
    cam_gt = _broadcast_camera(w, h, pan=8.0)
    kp_hm, ln_hm = render_heatmap_targets(cam_gt, w, h, 128, 72)
    assert kp_hm.shape == (NUM_PITCH_KEYPOINTS, 72, 128)
    assert ln_hm.shape == (NUM_PITCH_LINES, 72, 128)

    kps = decode_keypoint_heatmaps(kp_hm, w, h, threshold=0.35)
    lns = decode_line_heatmaps(ln_hm, w, h, threshold=0.40)
    assert len(kps) >= 6
    assert len(lns) >= 3

    fit = Apache2CameraSolver().solve(kps, lns, w, h)
    assert fit is not None
    assert fit.rep_err < 5.0
    assert fit.n_keypoints is not None and fit.n_keypoints >= len(kps)
    assert abs(fit.cam["pan_degrees"] - 8.0) < 0.5


def test_solver_records_exact_n_keypoints_with_four_keypoints() -> None:
    w, h = 1280, 720
    cam_gt = _broadcast_camera(w, h, pan=10.0)
    names = ("centre_spot", "halfway_T", "pa_RT_front", "pa_RB_front")
    kps: dict[int, tuple[float, float, float]] = {}
    for name in names:
        idx = KEYPOINT_INDEX[name]
        u, v = project(cam_gt, PITCH_KEYPOINTS_3D[[idx]])[0]
        kps[idx] = (float(u), float(v), 0.95)
    fit = Apache2CameraSolver().solve(kps, {}, w, h, refine_lines=False)
    assert fit is not None
    assert fit.n_keypoints == 4
    assert abs(fit.cam["tilt_degrees"] - 68.0) < 1.0


def test_pitch_calib_net_safetensors_checkpoint_and_clip_calibrator(tmp_path: Path) -> None:
    cfg = PitchCalibConfig(backbone="resnet34", input_size=(256, 144), fpn_channels=64)
    net = build_pitch_calib_net(cfg, pretrained_backbone=False)
    ckpt_dir = save_pitch_calibrator(net, cfg, tmp_path / "apache2-calib")
    assert (ckpt_dir / "config.json").is_file()
    assert (ckpt_dir / "model.safetensors").is_file()

    cal = Apache2PitchCalibrator(ckpt_dir, device="cpu", batch=2)
    frames = [np.zeros((180, 320, 3), dtype=np.uint8) for _ in range(3)]
    kp_hm, ln_hm = cal.heatmaps(frames)
    assert kp_hm.shape == (3, NUM_PITCH_KEYPOINTS, 36, 64)
    assert ln_hm.shape == (3, NUM_PITCH_LINES, 36, 64)

    # Verify drop-in fit() contract + ClipCalibrator wiring without touching SoccerMaster/ultralytics
    fits = cal.fit(frames, which=[0, 2], kp_threshold=0.99, line_threshold=0.99)
    assert set(fits.keys()) == {0, 2}
    cc = ClipCalibrator(lambda buf, keys: cal.fit([f[:, :, ::-1] for f in buf], which=keys), every=2)
    out = cc.run(frames)
    assert len(out) == 3

    # Verify zero ultralytics / SoccerMaster imports in an isolated fresh interpreter
    res = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import footlytics.geometry.pitch_calibrator as pc; "
                "assert 'ultralytics' not in sys.modules; "
                "assert 'data.pnlcalib_utils' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
