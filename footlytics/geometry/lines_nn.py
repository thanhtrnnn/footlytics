"""SoccerMaster's learned pitch-line detector, run locally on Apple Silicon.

RESULT: it runs, and on this footage it does not work either.

Two separate findings, and they should not be confused.

*The infrastructure works.* An M2 Pro runs this fine -- 32 GB of unified memory,
MPS, 272 ms per frame, and both checkpoints load with zero missing or unexpected
keys. An earlier claim in this project that the head "needs a GPU" was simply
wrong: an Apple laptop is a perfectly good place to run it, and this module is
reusable for the KeypointsDetection head too, whose weights are already alongside.

*What the head actually predicts.* Not a drawn line. For each named line it
emits two Gaussian blobs at that line's two ENDPOINTS -- confirmed in
`generate_gaussian_array_vectorized_l`, which writes a blob at (x_1, y_1) and
(x_2, y_2) per class with sigma=2. Endpoints give line equations, and line
correspondences give the homography. Reading the blobs as failed line
segmentation was my mistake, and it nearly produced a wrong conclusion.

*The detection is genuinely weak here, on the correct reading.* Verified against
a real broadcast frame, which the model handles beautifully: 9 lines at 0.70-0.86
with both endpoints found and coherent geometry -- the two goalpost classes come
out as vertical segments on the actual posts, the crossbar joins their tops, the
six-yard box traces correctly. The same model on the SoccerTrack night panorama
finds 4 lines at 0.21-0.47, mostly a single endpoint pinned to the image border,
and draws "Side line left" diagonally across the middle of the pitch. Feeding the
un-cropped 2:1 panorama is worse still: every class falls under 0.01.

That broadcast test also settles a question worth settling: the integration is
correct. The weakness is the footage, not the wiring.

The likely reason is the same for this and for the classical method in
`lines.py`: heavy fisheye curvature, floodlit night contrast, and full-pitch
framing put this footage far outside the broadcast distribution both were built
for. Worth re-testing on daytime footage with clear markings, where it may well
work -- the infrastructure to try is now here. It is not something to depend on.

Classical line detection failed on hard footage (see `lines.py` for the four
approaches that did not work and why). This is the alternative: SoccerMaster's
`LinesDetection` head on its SigLIP2-large backbone, which predicts one
segmentation heatmap per *named* pitch line. Knowing which line is which is the
part classical vision could not deliver -- a bright curve is a bright curve, but
"Side line bottom" is a constraint you can calibrate against.

It does not need a datacentre. The backbone is 1.4 GB and inference on a handful
of frames fits comfortably in an M2 Pro's unified memory via PyTorch's MPS
backend. Earlier in this project I said it "needs a GPU"; that was wrong, and
worth correcting -- an Apple laptop is a perfectly good place to run this.

Weights (ungated):
    xleprime/SoccerMaster -> backbone.pt (1.4 GB), LinesDetection.pt (33 MB)

The base SigLIP2 checkpoint is NOT needed. `backbone.pt` holds all 473 tensors
and 358.8M parameters of the fine-tuned tower -- patch embedding, position
embedding, all 24 encoder blocks, the temporal embedding and the head -- so the
architecture can be built from the config alone and the pretrained weights
loaded straight over it. That saves a 1.6 GB download of weights which would be
immediately overwritten.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

#: SoccerNet's line vocabulary, in the order the head emits channels.
#: Taken verbatim from SoccerMaster's own `data/pnlcalib_utils/utils_lines.py`
#: rather than reconstructed, because a wrong order silently mislabels every
#: line and the failure looks like a bad detector.
LINE_CLASSES = [
    "Big rect. left bottom", "Big rect. left main", "Big rect. left top",
    "Big rect. right bottom", "Big rect. right main", "Big rect. right top",
    "Goal left crossbar", "Goal left post left", "Goal left post right",
    "Goal right crossbar", "Goal right post left", "Goal right post right",
    "Middle line",
    "Side line bottom", "Side line left", "Side line right", "Side line top",
    "Small rect. left bottom", "Small rect. left main", "Small rect. left top",
    "Small rect. right bottom", "Small rect. right main", "Small rect. right top",
]

IMAGE_SIZE = 512
NUM_FRAMES = 30          # the temporal embedding is trained at this length
NUM_LINES = 24


@dataclass
class LineDetection:
    """Heatmaps and extracted points for one frame."""
    heatmaps: np.ndarray                       # (num_lines, 256, 256) in [0, 1]
    frame_size: tuple[int, int]                # (width, height) of the source

    def points(self, name: str, threshold: float = 0.5,
               max_points: int = 400) -> np.ndarray:
        """Pixel coordinates of one named line, in the ORIGINAL frame's scale."""
        if name not in LINE_CLASSES:
            raise KeyError(f"unknown line {name!r}; expected one of {LINE_CLASSES}")
        hm = self.heatmaps[LINE_CLASSES.index(name)]
        ys, xs = np.nonzero(hm >= threshold)
        if not len(xs):
            return np.empty((0, 2))
        w, h = self.frame_size
        pts = np.column_stack([xs / hm.shape[1] * w, ys / hm.shape[0] * h])
        if len(pts) > max_points:
            pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]
        return pts

    def strength(self, name: str) -> float:
        return float(self.heatmaps[LINE_CLASSES.index(name)].max())

    def present(self, threshold: float = 0.5) -> dict[str, float]:
        """Which lines the model thinks it can see, strongest first."""
        out = {n: self.strength(n) for n in LINE_CLASSES}
        return {k: v for k, v in sorted(out.items(), key=lambda kv: -kv[1]) if v >= threshold}


class SoccerMasterLines:
    """Loads the backbone and the lines head, and runs them on frames."""

    #: SigLIP2-large-patch16-512 vision tower, as read from the published config
    #: and confirmed against the shapes inside backbone.pt.
    SIGLIP_CONFIG = dict(
        hidden_size=1024, intermediate_size=4096, num_hidden_layers=24,
        num_attention_heads=16, patch_size=16, image_size=512, num_channels=3,
    )

    def __init__(
        self,
        code_dir: str | Path,
        weights_dir: str | Path,
        device: Optional[str] = None,
        dtype: str = "float32",
        temporal_start_layer: int = 16,
    ):
        import torch

        self.code_dir = Path(code_dir)
        if str(self.code_dir) not in sys.path:
            sys.path.insert(0, str(self.code_dir))

        if device is None:
            device = ("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.torch = torch
        # MPS is missing a few ops; the fallback keeps them on the CPU instead
        # of crashing. Set before any tensor touches the device.
        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        import models.soccer_master as sm
        from models.soccer_master import VisionBackbone
        from models.lines_detection import LinesDetection
        from transformers import SiglipVisionConfig, SiglipVisionModel

        cfg = SiglipVisionConfig(**self.SIGLIP_CONFIG)

        # VisionBackbone insists on `from_pretrained`, which would pull 1.6 GB of
        # weights that backbone.pt overwrites in full a moment later. Build the
        # architecture from the config instead; the random init never survives.
        # transformers 5 flattened SiglipVisionModel: what used to live under
        # `.vision_model` is now the model itself. SoccerMaster was written
        # against transformers 4 and reaches for `.vision_model`, so hand it a
        # shim rather than pinning an old transformers for one attribute.
        class _Shim:
            def __init__(self, inner):
                self.vision_model = inner

        orig_from_pretrained = SiglipVisionModel.from_pretrained
        orig_cfg_from_pretrained = SiglipVisionConfig.from_pretrained
        try:
            sm.SiglipVisionModel.from_pretrained = staticmethod(
                lambda *a, **k: _Shim(SiglipVisionModel(cfg)))
            sm.SiglipVisionConfig.from_pretrained = staticmethod(lambda *a, **k: cfg)
            self.backbone = VisionBackbone(
                ckpt_path="(built from config)", num_frames=NUM_FRAMES,
                temporal_start_layer=temporal_start_layer,
            )
        finally:
            sm.SiglipVisionModel.from_pretrained = orig_from_pretrained
            sm.SiglipVisionConfig.from_pretrained = orig_cfg_from_pretrained
        w = Path(weights_dir)
        state = torch.load(w / "backbone.pt", map_location="cpu", weights_only=True)
        target = getattr(self.backbone, "vision_model", self.backbone)
        missing = target.load_state_dict(state, strict=False)
        print(f"[lines_nn] backbone: {len(missing.missing_keys)} missing, "
              f"{len(missing.unexpected_keys)} unexpected keys")

        self.head = LinesDetection(
            backbone_num_channels=[1024], num_lines=NUM_LINES, backbone_type="video",
        )
        hstate = torch.load(w / "LinesDetection.pt", map_location="cpu", weights_only=True)
        res = self.head.load_state_dict(hstate, strict=False)
        print(f"[lines_nn] head: {len(res.missing_keys)} missing, "
              f"{len(res.unexpected_keys)} unexpected keys")

        # transformers 4's SiglipEncoderLayer returned a 1-tuple; transformers 5
        # returns the tensor itself. SoccerMaster does `self.encoder(x, mask)[0]`,
        # which in v5 silently slices the batch dimension instead of unpacking --
        # a 3-D [B*T, N, D] activation becomes 2-D and attention fails deep in
        # the stack with an unhelpful IndexError. Restore the tuple contract.
        #
        # Applied AFTER loading weights: wrapping first would rename every
        # encoder key and the checkpoint would no longer match.
        from torch import nn as _nn

        class _TupleOut(_nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, *a, **k):
                out = self.inner(*a, **k)
                return out if isinstance(out, tuple) else (out,)

        for blk in self.backbone.encoder_blocks:
            blk.encoder = _TupleOut(blk.encoder)

        td = getattr(torch, dtype)
        self.backbone.to(device=device, dtype=td).eval()
        self.head.to(device=device, dtype=td).eval()
        self.dtype = td

    # ------------------------------------------------------------------ run

    def _preprocess(self, frames: Sequence[np.ndarray]):
        """RGB uint8 frames -> normalised tensor [1, T, 3, 512, 512]."""
        import cv2
        import torch

        out = []
        for f in frames:
            r = cv2.resize(f, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            out.append(r.astype(np.float32) / 255.0)
        x = np.stack(out)
        # SigLIP normalisation: mean 0.5, std 0.5.
        x = (x - 0.5) / 0.5
        t = torch.from_numpy(x).permute(0, 3, 1, 2)[None]
        return t.to(device=self.device, dtype=self.dtype)

    def detect(self, frames: Sequence[np.ndarray]) -> list[LineDetection]:
        """Run on up to NUM_FRAMES RGB frames. Fewer are padded by repetition,
        because the temporal embedding is sized for a fixed clip length."""
        import torch

        frames = list(frames)
        if not frames:
            raise ValueError("no frames given")
        h, w = frames[0].shape[:2]
        n = len(frames)
        if n < NUM_FRAMES:
            frames = frames + [frames[-1]] * (NUM_FRAMES - n)
        elif n > NUM_FRAMES:
            raise ValueError(f"pass at most {NUM_FRAMES} frames per call")

        x = self._preprocess(frames)
        with torch.no_grad():
            # VisionBackbone returns (local_features, pooled). Only the wrapper
            # class assembles the dict the heads expect, and that wrapper also
            # drags in a text tower we have no use for -- so build it here.
            local, pooled = self.backbone(x)
            feats = {"global_features": pooled, "local_features": local}
            out = self.head(feats, metas=None)
        hm = out["pred_lines_heatmap"][0].float().cpu().numpy()   # [T, L, 256, 256]
        return [LineDetection(hm[i], (w, h)) for i in range(n)]


# --------------------------------------------------------------------------
# Tiled inference
# --------------------------------------------------------------------------

def tile_boxes(width: int, height: int, aspect: float = 16 / 9,
               overlap: float = 0.35, rows: int = 1) -> list[tuple[int, int, int, int]]:
    """Overlapping broadcast-shaped crops covering a panorama."""
    tile_h = height // rows
    tile_w = int(round(tile_h * aspect))
    if tile_w > width:
        tile_w, tile_h = width, int(round(width / aspect))
    step = max(int(tile_w * (1 - overlap)), 1)
    xs = list(range(0, max(width - tile_w, 0) + 1, step)) or [0]
    if xs[-1] + tile_w < width:
        xs.append(width - tile_w)
    ys = [r * tile_h for r in range(rows)]
    return [(x, y, min(x + tile_w, width), min(y + tile_h, height))
            for y in ys for x in xs]


def detect_tiled(model: "SoccerMasterLines", frame: np.ndarray,
                 aspect: float = 16 / 9, overlap: float = 0.35, rows: int = 1,
                 brighten: float = 1.0, out_size: int = 512,
                 roi: Optional[tuple[int, int, int, int]] = None,
                 verbose: bool = True) -> LineDetection:
    """Run line detection on broadcast-shaped tiles and merge the heatmaps.

    The same lesson as the player detector, for the same reason. SoccerMaster is
    trained on broadcast frames, so it expects a pitch that fills a roughly 16:9
    view. Squashing a 2:1 full-pitch panorama into the model's square input put
    every line class under 0.01 activation -- the model saw nothing at all. The
    identical frame, cropped to 16:9, reached 0.93.

    Brightening helps on floodlit night footage, where the painted lines carry
    very little contrast: it lifted the strongest line from 0.23 to 0.47 here.
    It is a preprocessing choice, not a correction, so it is off by default.
    """
    import cv2

    h, w = frame.shape[:2]
    # Tiles must frame the PITCH, not the whole image. Tiling the full frame
    # height on a panorama keeps the sky and the stands in every crop, which
    # leaves each tile at roughly the same 2:1 shape that defeated the model in
    # the first place -- the aspect only becomes broadcast-like once the
    # non-pitch bands are excluded.
    rx0, ry0, rx1, ry1 = roi if roi else (0, 0, w, h)
    boxes = [(x0 + rx0, y0 + ry0, x1 + rx0, y1 + ry0)
             for x0, y0, x1, y1 in tile_boxes(rx1 - rx0, ry1 - ry0, aspect, overlap, rows)]
    acc = np.zeros((NUM_LINES, out_size, out_size), np.float32)
    counts = np.zeros((out_size, out_size), np.float32) + 1e-6

    for i, (x0, y0, x1, y1) in enumerate(boxes):
        crop = frame[y0:y1, x0:x1]
        if brighten != 1.0:
            crop = cv2.convertScaleAbs(crop, alpha=brighten, beta=10)
        det = model.detect([crop] * NUM_FRAMES)[0]
        hm = det.heatmaps                                    # (L, 256, 256)
        # Place this tile's heatmap into the full-frame canvas.
        tw = int(round((x1 - x0) / w * out_size))
        th = int(round((y1 - y0) / h * out_size))
        tx = int(round(x0 / w * out_size))
        ty = int(round(y0 / h * out_size))
        tw, th = max(tw, 1), max(th, 1)
        resized = np.stack([cv2.resize(c, (tw, th)) for c in hm])
        sl = (slice(ty, min(ty + th, out_size)), slice(tx, min(tx + tw, out_size)))
        acc[:, sl[0], sl[1]] = np.maximum(
            acc[:, sl[0], sl[1]], resized[:, :sl[0].stop - sl[0].start,
                                          :sl[1].stop - sl[1].start])
        counts[sl] += 1
        if verbose:
            print(f"  tile {i + 1}/{len(boxes)} [{x0},{y0},{x1},{y1}] "
                  f"max {hm.max():.3f}", flush=True)

    return LineDetection(acc, (w, h))


def virtual_pinhole_view(
    frame: np.ndarray, K: np.ndarray, D: np.ndarray, Knew: np.ndarray, H: np.ndarray,
    target_pitch_xy: tuple[float, float], fov_deg: float = 38.0,
    out_size: tuple[int, int] = (1280, 720),
):
    """Synthesise a broadcast-like view of one part of the pitch, from a panorama.

    This resolves a real contradiction in using a broadcast-trained model on a
    fixed panoramic camera: a broadcast camera NEVER sees the whole pitch, so the
    model was built to calibrate a partial view frame by frame, from whatever
    lines happen to be visible. Handing it an entire pitch squashed into a square
    is not a hard version of its task; it is a different task.

    So we build the view it expects. Choosing a virtual pinhole camera at the same
    optical centre, rotated to face a chosen pitch location with a narrow field of
    view, gives a rectilinear zoomed window -- lines straight, one region filling
    the frame. Several such windows across the pitch then yield line
    correspondences everywhere, which a single panoramic frame cannot.

    Measured on the SoccerTrack night match this lifted the best line response
    from 0.47 to 0.63, but inspection showed the endpoint landing nowhere near the
    penalty area it was labelled with. The technique is sound; that pitch simply
    has almost no visible markings under floodlights, and no view synthesis
    recovers signal that is not in the pixels.

    `target_pitch_xy` is in corner-origin pitch metres; `H` maps pitch to
    undistorted image, as shipped with SoccerTrack v2.
    """
    import cv2

    h = np.array([target_pitch_xy[0], target_pitch_xy[1], 1.0]) @ H.T
    u = h[:2] / h[2]
    d = np.linalg.inv(Knew) @ np.array([u[0], u[1], 1.0])
    d = d / np.linalg.norm(d)

    z = np.array([0.0, 0.0, 1.0])
    a = np.cross(d, z)
    s = float(np.linalg.norm(a))
    c = float(d @ z)
    if s < 1e-9:
        R = np.eye(3)
    else:
        ax = a / s
        Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = np.eye(3) + s * Kx + (1 - c) * (Kx @ Kx)

    f = out_size[0] / (2 * np.tan(np.radians(fov_deg) / 2))
    P = np.array([[f, 0, out_size[0] / 2], [0, f, out_size[1] / 2], [0, 0, 1.0]])
    mx, my = cv2.fisheye.initUndistortRectifyMap(K, D, R, P, out_size, cv2.CV_32FC1)
    return cv2.remap(frame, mx, my, cv2.INTER_LINEAR)
