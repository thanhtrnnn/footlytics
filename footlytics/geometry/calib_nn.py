"""Per-frame camera calibration: SoccerMaster's keypoint and line heads, solved by PnLCalib.

One pass through the backbone's early layers feeds both heads, frame by frame
(`SoccerMasterLines` loads the backbone and the lines head; this adds the keypoints head). Heatmaps are decoded and
solved with PnLCalib's code as vendored in the SoccerMaster repository. That repository has no
licence (internal R&D), so it is imported from `code_dir` at runtime and never copied here. The
output is the full SoccerNet camera (`sn_camera`), which SN-Calibration scores, plus the
image->pitch homography the pipeline uses.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .lines_nn import use_soccermaster_checkout
from .sn_camera import ground_homography

NUM_KEYPOINTS = 58                 # 57 pitch keypoints + background
#: PnLCalib's published inference thresholds (tuned in the benchmark, plan 2 task 7)
DEFAULT_KP_THRESHOLD = 0.3434
DEFAULT_LINE_THRESHOLD = 0.7867
_CROSSBAR = {12, 15, 16, 19}       # keypoints 2.44 m up


@dataclass
class CameraFit:
    cam: dict                      # SoccerNet camera at the source resolution
    rep_err: float                 # solver's keypoint reprojection error, source px
    H_img_to_pitch: np.ndarray
    n_keypoints: int | None = None


def valid_camera_fit(fit: Optional[CameraFit], width: int, height: int) -> bool:
    """Reject physically impossible PnLCalib fits (e.g. 4-point overfitting behind the camera)."""
    if fit is None or not np.isfinite(fit.rep_err):
        return False
    cam = fit.cam
    if all(k in cam for k in ("x_focal_length", "position_meters", "tilt_degrees", "roll_degrees")):
        fw = float(cam["x_focal_length"]) / max(width, 1)
        z = float(cam["position_meters"][2])
        tilt = float(cam["tilt_degrees"])
        roll = float(cam["roll_degrees"])
        if not (0.45 <= fw <= 5.0 and -100.0 <= z <= -2.0 and 20.0 <= tilt <= 89.5 and abs(roll) <= 10.0):
            return False
    H = np.asarray(fit.H_img_to_pitch, float)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        return False
    if float(H[2] @ np.array([width / 2.0, height / 2.0, 1.0])) < 0:
        H = -H
    for v in np.linspace(0.45 * height, 0.95 * height, 3):
        for u in np.linspace(0.1 * width, 0.9 * width, 5):
            p = H @ np.array([u, v, 1.0])
            wv = float(p[2])
            if not np.isfinite(wv) or wv <= 1e-9:
                return False
            J = np.array([
                [(H[0, 0] * wv - p[0] * H[2, 0]) / wv ** 2,
                 (H[0, 1] * wv - p[0] * H[2, 1]) / wv ** 2],
                [(H[1, 0] * wv - p[1] * H[2, 0]) / wv ** 2,
                 (H[1, 1] * wv - p[1] * H[2, 1]) / wv ** 2],
            ])
            if not np.isfinite(J).all():
                return False
            sv = np.linalg.svd(J, compute_uv=False)
            if float(np.linalg.det(J)) <= 0 or float(sv[0] / max(sv[1], 1e-12)) > 25.0:
                return False
    cp = H @ np.array([width / 2.0, height / 2.0, 1.0])
    if not np.isfinite(cp).all() or cp[2] <= 1e-9:
        return False
    cx, cy = cp[:2] / cp[2]
    return bool(abs(cx) <= 150.0 and abs(cy) <= 100.0)


def _plain(v):
    """numpy arrays and scalars -> nested lists of plain floats (a camera dict is JSON)."""
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return float(v)


def _peaks(hm, min_dist: int, k: int, pad_with_ones: bool, scale: int = 2):
    """Heatmaps [B, C, H, W] -> [B, C, k, 3] (u, v, score) on the CPU: PnLCalib's max-pool
    decoders (`get_keypoints_from_heatmap_batch_maxpool` and `_l`) without their Python loops.

    A pixel is a peak when it equals the max over its (2 * min_dist + 1)-square window; the k
    highest peaks per channel come back, best first (a channel with fewer peaks is filled with
    zero-score, non-peak pixels, which the thresholds drop). `pad_with_ones` is the keypoint
    flavour: the window is padded with 1.0, so nothing within `min_dist` of the border can be a
    peak; the line flavour pads with -inf. (u, v) = (column, row) * `scale`: the heatmaps are
    half the image size. Runs on the heatmaps' device; the result is tiny, so it is moved
    to the CPU."""
    import torch
    import torch.nn.functional as F

    b, c, h, w = hm.shape
    kernel = 2 * min_dist + 1
    if pad_with_ones:
        pooled = F.max_pool2d(F.pad(hm, (min_dist,) * 4, mode="constant", value=1.0), kernel,
                              stride=1, padding=0)
    else:
        pooled = F.max_pool2d(hm, kernel, stride=1, padding=min_dist)
    scores, idx = torch.topk((hm * (pooled == hm)).reshape(b, c, -1), k, sorted=True)
    rows = torch.div(idx, w, rounding_mode="floor")
    cols = idx % w
    out = torch.stack([cols * scale, rows * scale, scores], dim=-1)
    return out.float().cpu()


class PnlSolver:
    """PnLCalib's decoding and solving, imported from the SoccerMaster checkout."""

    def __init__(self, code_dir: str | Path):
        use_soccermaster_checkout(code_dir)
        from data.pnlcalib_utils import utils_calib, utils_heatmap

        self._calib, self._hm = utils_calib, utils_heatmap

    def keypoint_world(self, kp_id: int) -> tuple[float, float, float]:
        """PnLCalib keypoint `kp_id` (1-based) in SoccerNet world metres. PnLCalib's tables are
        already centre-origin (`utils_calib` shifts them by 52.5, 34), as SoccerNet's are."""
        c = self._calib
        wp = (c.keypoint_world_coords_2D[kp_id - 1] if kp_id <= 57
              else c.keypoint_aux_world_coords_2D[kp_id - 58])
        return wp[0], wp[1], -2.44 if kp_id in _CROSSBAR else 0.0

    def decode(self, kp_hm, line_hm, kp_threshold: float = DEFAULT_KP_THRESHOLD,
               line_threshold: float = DEFAULT_LINE_THRESHOLD) -> list[tuple[dict, dict]]:
        """Heatmaps [T, 58, h, w] and [T, 24, h, w] (torch, any device) -> per frame (keypoints,
        lines), normalised to [0, 1]. Peaks are found by `_peaks` (PnLCalib's decoders,
        vectorised); PnLCalib's `coords_to_dict` and `complete_keypoints` do the rest."""
        hm = self._hm
        kp = hm.coords_to_dict(_peaks(kp_hm[:, :-1], min_dist=15, k=1, pad_with_ones=True),
                               threshold=kp_threshold)
        ln = hm.coords_to_dict(_peaks(line_hm[:, :-1], min_dist=10, k=2, pad_with_ones=False),
                               threshold=line_threshold)
        side = 2 * kp_hm.shape[-1]          # coordinates come back at twice the heatmap size
        return [hm.complete_keypoints(k, l, w=side, h=side, normalize=True)
                for k, l in zip(kp, ln)]

    @staticmethod
    def _fit_from_cam(cam, err, n_keypoints: int | None = None) -> Optional[CameraFit]:
        if cam is None or err is None or not np.isfinite(err):
            return None
        cam_p = {k: _plain(v) for k, v in cam.items() if k != "rotation_matrix"}
        try:
            H = np.linalg.inv(ground_homography(cam_p))
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(H).all() or abs(float(H[2, 2])) < 1e-12:
            return None
        return CameraFit(cam_p, float(err), H / H[2, 2], n_keypoints=n_keypoints)

    def solve(self, keypoints: dict, lines: dict, width: int, height: int, voting: bool = True,
              mode: str = "full", ransac: float = 0.0,
              refine_lines: bool = True) -> Optional[CameraFit]:
        """A camera from normalised detections; `voting` runs PnLCalib's 18-setting search,
        otherwise the single (`mode`, `ransac`) setting. None when nothing fits."""
        c = self._calib.FramebyFrameCalib(iwidth=width, iheight=height, denormalize=True)
        c.update(copy.deepcopy(keypoints), copy.deepcopy(lines))
        n_kp = len(keypoints)
        if voting:
            res = c.heuristic_voting(refine_lines=refine_lines)
            cam, err = (res["cam_params"], res["rep_err"]) if res else (None, None)
        else:
            cam, err = c.get_cam_params(mode=mode, use_ransac=ransac, refine_w_lines=refine_lines)
        fit = self._fit_from_cam(cam, err, n_keypoints=n_kp)
        if not voting and mode == "full" and ransac > 0 and (
            fit is None or fit.rep_err > 25.0 or not valid_camera_fit(fit, width, height)
        ):
            c2 = self._calib.FramebyFrameCalib(iwidth=width, iheight=height, denormalize=True)
            c2.update(copy.deepcopy(keypoints), copy.deepcopy(lines))
            try:
                cam2, err2 = c2.get_cam_params(mode="ground_plane", use_ransac=0.0,
                                              refine_w_lines=refine_lines)
                fit2 = self._fit_from_cam(cam2, err2, n_keypoints=n_kp)
                if fit2 is not None and fit2.rep_err <= 25.0 and valid_camera_fit(fit2, width, height):
                    fit = fit2
            except Exception:  # noqa: BLE001
                pass
        if fit is not None and not valid_camera_fit(fit, width, height):
            return None
        return fit


_WORKER: dict = {}


def _solve_job(args) -> tuple[Optional[CameraFit], Optional[str]]:
    code_dir, kp, ln, w, h, kw = args
    if _WORKER.get("code_dir") != code_dir:
        _WORKER.update(code_dir=code_dir, solver=PnlSolver(code_dir))
    try:
        return _WORKER["solver"].solve(kp, ln, w, h, **kw), None
    except Exception as e:  # noqa: BLE001 - degenerate detections; the caller counts them
        return None, f"{type(e).__name__}: {e}"


def solve_many(code_dir: str | Path, jobs: Sequence[tuple[dict, dict, int, int]],
               workers: int = 1, _job=_solve_job,
               **solve_kw) -> list[tuple[Optional[CameraFit], Optional[str]]]:
    """Solve (keypoints, lines, width, height) jobs, on `workers` CPU processes: the solver,
    not the network, is the slow part (about 5 s a frame with voting). A worker that dies (OOM
    kill, a crash in cv2) breaks the pool; every job without a result then comes back as
    `(None, "BrokenProcessPool: ...")` rather than hanging the run."""
    args = [(str(code_dir), k, l, w, h, solve_kw) for k, l, w, h in jobs]
    if workers <= 1:
        return [_job(a) for a in args]
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    def broken(e):
        return None, f"BrokenProcessPool: {e}"

    out: list = []
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = []
        for a in args:
            try:
                futures.append(pool.submit(_job, a))
            except BrokenProcessPool as e:      # died while we were still submitting
                futures.append(e)
        for f in futures:
            if isinstance(f, BrokenProcessPool):
                out.append(broken(f))
                continue
            try:
                out.append(f.result())
            except BrokenProcessPool as e:
                out.append(broken(e))
    return out


class SoccerMasterCalibrator:
    """Backbone + lines head + keypoints head, decoded and solved per frame."""

    def __init__(self, code_dir: str | Path, weights_dir: str | Path,
                 device: Optional[str] = None, batch: int = 8):
        import torch

        from .lines_nn import SoccerMasterLines

        self.batch = max(1, int(batch))
        self.code_dir = Path(code_dir)
        self.lines = SoccerMasterLines(code_dir, weights_dir, device=device)
        self.solver = PnlSolver(code_dir)
        from models.keypoints_detection import KeypointsDetection

        self.kp_head = KeypointsDetection(backbone_num_channels=[1024],
                                          num_keypoints=NUM_KEYPOINTS, backbone_type="video")
        state = torch.load(Path(weights_dir) / "KeypointsDetection.pt", map_location="cpu",
                           weights_only=True)
        res = self.kp_head.load_state_dict(state, strict=False)
        if res.missing_keys:
            raise RuntimeError(f"keypoints head is missing {len(res.missing_keys)} weights, "
                               f"e.g. {res.missing_keys[:3]}")
        self.kp_head.to(device=self.lines.device, dtype=self.lines.dtype).eval()
        self.last_errors: dict[int, str] = {}

    def heatmaps(self, frames_rgb: Sequence[np.ndarray]):
        """Any number of RGB frames -> (kp, lines) heatmaps [n, 58, 256, 256] and
        [n, 24, 256, 256], float, on the model's device.

        Only the early layers run, in batches of at most `self.batch` frames. That is exact:
        SoccerMaster's backbone builds blocks 0 .. temporal_start_layer-1 without temporal
        attention (spatial attention per frame), and both heads read only
        `local_features_early`, the output of those blocks, taken before the temporal
        embedding and the temporal layers (which feed only the global features the heads do
        not use). A frame's heatmaps therefore depend on that frame alone: no 30-frame clip,
        no padding, and the later 8 layers are skipped. Equal to the full 30-frame path within
        float noise (`test_early_layer_heatmaps_equal_the_30_frame_path`)."""
        import torch

        frames = list(frames_rgb)
        if not frames:
            raise ValueError("pass at least one frame")
        bb = self.lines.backbone
        kps, lns = [], []
        with torch.no_grad():
            for i in range(0, len(frames), self.batch):
                x = self.lines._preprocess(frames[i:i + self.batch])[0]     # [n, 3, 512, 512]
                n = x.shape[0]
                f = bb.vision_model_embedding(x)
                for blk in bb.encoder_blocks[:bb.temporal_start_layer]:
                    f = blk(f, 1, n)
                feats = {"global_features": None, "local_features": f[None]}
                kps.append(self.kp_head(feats, metas=None)["pred_keypoints_heatmap"][0])
                lns.append(self.lines.head(feats, metas=None)["pred_lines_heatmap"][0])
        return torch.cat(kps).float(), torch.cat(lns).float()

    def decode(self, frames_rgb: Sequence[np.ndarray], kp_threshold: float = DEFAULT_KP_THRESHOLD,
               line_threshold: float = DEFAULT_LINE_THRESHOLD) -> list[tuple[dict, dict]]:
        kp, ln = self.heatmaps(frames_rgb)
        return self.solver.decode(kp, ln, kp_threshold, line_threshold)

    def fit(self, frames_rgb: Sequence[np.ndarray], which: Optional[Sequence[int]] = None,
            kp_threshold: float = DEFAULT_KP_THRESHOLD,
            line_threshold: float = DEFAULT_LINE_THRESHOLD,
            **solve_kw) -> dict[int, Optional[CameraFit]]:
        """Cameras for the frames in `which` (default all), keyed by index into `frames_rgb`.
        Heatmaps are computed only for those frames: they are independent of each other (see
        `heatmaps`), so the rest of a chunk costs nothing. A frame whose detections the solver
        cannot use gets None; if the solver raised, the reason is in `self.last_errors[frame]`
        (reset on every call)."""
        self.last_errors = {}
        frames = list(frames_rgb)
        h, w = frames[0].shape[:2]
        which = list(range(len(frames)) if which is None else which)
        if not which:
            return {}
        kp, ln = self.heatmaps([frames[i] for i in which])
        decoded = self.solver.decode(kp, ln, kp_threshold, line_threshold)
        out: dict[int, Optional[CameraFit]] = {}
        for i, (k, l) in zip(which, decoded):
            try:
                out[i] = self.solver.solve(k, l, w, h, **solve_kw)
            except Exception as e:  # noqa: BLE001 - degenerate detections: no camera this frame
                out[i] = None
                self.last_errors[i] = f"{type(e).__name__}: {e}"
        return out
