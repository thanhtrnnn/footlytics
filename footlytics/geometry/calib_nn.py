"""Per-frame camera calibration: SoccerMaster's keypoint and line heads, solved by PnLCalib.

One backbone pass over up to 30 consecutive frames feeds both heads (`SoccerMasterLines`
loads the backbone and the lines head; this adds the keypoints head). Heatmaps are decoded and
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


def _plain(v):
    """numpy arrays and scalars -> nested lists of plain floats (a camera dict is JSON)."""
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return float(v)


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
        """Heatmaps [T, 58, h, w] and [T, 24, h, w] (torch) -> per frame (keypoints, lines),
        normalised to [0, 1]."""
        hm = self._hm
        kp = hm.coords_to_dict(hm.get_keypoints_from_heatmap_batch_maxpool(kp_hm[:, :-1]),
                               threshold=kp_threshold)
        ln = hm.coords_to_dict(hm.get_keypoints_from_heatmap_batch_maxpool_l(line_hm[:, :-1]),
                               threshold=line_threshold)
        side = 2 * kp_hm.shape[-1]          # coordinates come back at twice the heatmap size
        return [hm.complete_keypoints(k, l, w=side, h=side, normalize=True)
                for k, l in zip(kp, ln)]

    def solve(self, keypoints: dict, lines: dict, width: int, height: int, voting: bool = True,
              mode: str = "full", ransac: float = 0.0,
              refine_lines: bool = True) -> Optional[CameraFit]:
        """A camera from normalised detections; `voting` runs PnLCalib's 18-setting search,
        otherwise the single (`mode`, `ransac`) setting. None when nothing fits."""
        c = self._calib.FramebyFrameCalib(iwidth=width, iheight=height, denormalize=True)
        c.update(copy.deepcopy(keypoints), copy.deepcopy(lines))
        if voting:
            res = c.heuristic_voting(refine_lines=refine_lines)
            cam, err = (res["cam_params"], res["rep_err"]) if res else (None, None)
        else:
            cam, err = c.get_cam_params(mode=mode, use_ransac=ransac, refine_w_lines=refine_lines)
        if cam is None or err is None or not np.isfinite(err):
            return None
        cam = {k: _plain(v) for k, v in cam.items() if k != "rotation_matrix"}
        H = np.linalg.inv(ground_homography(cam))
        return CameraFit(cam, float(err), H / H[2, 2])


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
                 device: Optional[str] = None):
        import torch

        from .lines_nn import SoccerMasterLines

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
        """Up to 30 consecutive RGB frames (a still is repeated) -> (kp, lines) heatmaps on CPU."""
        import torch

        from .lines_nn import NUM_FRAMES

        frames = list(frames_rgb)
        if not 0 < len(frames) <= NUM_FRAMES:
            raise ValueError(f"pass 1 to {NUM_FRAMES} frames, got {len(frames)}")
        n = len(frames)
        x = self.lines._preprocess(frames + [frames[-1]] * (NUM_FRAMES - n))
        with torch.no_grad():
            local, pooled = self.lines.backbone(x)
            feats = {"global_features": pooled, "local_features": local}
            kp = self.kp_head(feats, metas=None)["pred_keypoints_heatmap"][0, :n]
            ln = self.lines.head(feats, metas=None)["pred_lines_heatmap"][0, :n]
        return kp.float().cpu(), ln.float().cpu()

    def decode(self, frames_rgb: Sequence[np.ndarray], kp_threshold: float = DEFAULT_KP_THRESHOLD,
               line_threshold: float = DEFAULT_LINE_THRESHOLD) -> list[tuple[dict, dict]]:
        kp, ln = self.heatmaps(frames_rgb)
        return self.solver.decode(kp, ln, kp_threshold, line_threshold)

    def fit(self, frames_rgb: Sequence[np.ndarray], which: Optional[Sequence[int]] = None,
            kp_threshold: float = DEFAULT_KP_THRESHOLD,
            line_threshold: float = DEFAULT_LINE_THRESHOLD,
            **solve_kw) -> dict[int, Optional[CameraFit]]:
        """Heatmaps for all frames (temporal context), cameras for `which` (default all).
        A frame whose detections the solver cannot use gets None; if the solver raised, the
        reason is in `self.last_errors[frame]` (reset on every call)."""
        self.last_errors = {}
        frames = list(frames_rgb)
        h, w = frames[0].shape[:2]
        kp, ln = self.heatmaps(frames)
        which = list(range(len(frames)) if which is None else which)
        if not which:
            return {}
        decoded = self.solver.decode(kp[which], ln[which], kp_threshold, line_threshold)
        out: dict[int, Optional[CameraFit]] = {}
        for i, (k, l) in zip(which, decoded):
            try:
                out[i] = self.solver.solve(k, l, w, h, **solve_kw)
            except Exception as e:  # noqa: BLE001 - degenerate detections: no camera this frame
                out[i] = None
                self.last_errors[i] = f"{type(e).__name__}: {e}"
        return out
