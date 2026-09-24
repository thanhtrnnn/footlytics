"""Player and ball detection.

The wrinkle with a full-pitch panoramic view is scale. A broadcast close-up
gives you a 300 px player; a 4K view of the whole pitch gives you ~40 px, and
the far touchline is worse. Feeding a 3840x2160 frame to a detector at
imgsz=1280 downscales 3x, leaving a 13 px player -- below what the model can
reliably fire on, and the ball effectively disappears.

So we tile: run the detector on overlapping crops at native resolution and merge
in global coordinates. Slower, but it is the difference between finding 22
players and finding 14.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ..state.schema import Role

# How SoccerNet-style class names map onto our roles. Matched case-insensitively
# against substrings of the model's own class names, most specific first.
_ROLE_PATTERNS: list[tuple[str, str]] = [
    ("goalkeeper", Role.GOALKEEPER.value),
    ("keeper", Role.GOALKEEPER.value),
    ("referee", Role.REFEREE.value),
    ("ref", Role.REFEREE.value),
    ("ball", Role.BALL.value),
    ("player", Role.PLAYER.value),
    ("person", Role.PLAYER.value),
]

DET_COLS = ("x", "y", "w", "h", "conf", "cls")


def map_class_names(names: dict[int, str]) -> dict[int, str]:
    """Build class-id -> role from a model's own `names`, warning on surprises."""
    out: dict[int, str] = {}
    for cid, name in names.items():
        low = str(name).lower()
        for pat, role in _ROLE_PATTERNS:
            if pat in low:
                out[int(cid)] = role
                break
        else:
            out[int(cid)] = Role.OTHER.value
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    """Greedy NMS on xywh boxes. Returns indices to keep, best score first."""
    if len(boxes) == 0:
        return np.empty(0, dtype=int)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    area = np.maximum(boxes[:, 2], 0) * np.maximum(boxes[:, 3], 0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / np.maximum(area[i] + area[rest] - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=int)


def tile_grid(width: int, height: int, tile: int = 1280, overlap: float = 0.25
              ) -> list[tuple[int, int, int, int]]:
    """Overlapping tiles covering the frame, as (x0, y0, x1, y1).

    Overlap must comfortably exceed one player's width, or a player straddling a
    seam is cut in half in both tiles and detected in neither.
    """
    if tile >= width and tile >= height:
        return [(0, 0, width, height)]
    step = max(int(tile * (1 - overlap)), 1)
    xs = list(range(0, max(width - tile, 0) + 1, step)) or [0]
    ys = list(range(0, max(height - tile, 0) + 1, step)) or [0]
    if xs[-1] + tile < width:
        xs.append(width - tile)
    if ys[-1] + tile < height:
        ys.append(height - tile)
    return [(x, y, min(x + tile, width), min(y + tile, height)) for y in ys for x in xs]


@dataclass
class DetectorConfig:
    tile: int = 1280           # 0 or negative disables tiling
    overlap: float = 0.25
    #: Measured end to end on real footage, not argued from first principles.
    #:
    #: The gate was briefly set to 0.10 on the reasoning that recall matters more
    #: than precision for tracking -- a missed player breaks a track, whereas a
    #: false positive would be filtered by the tracker's min_hits rule. That was
    #: wrong, and measuring it (scripts/compare_conf.py, 250 real frames) showed
    #: the opposite:
    #:
    #:     conf   tracklets  recall  precision  ID switches  players/frame
    #:     0.10       133      74%       77%        347           21
    #:     0.25        78      60%       94%        179           14
    #:     0.35        60      53%       97%        147           12
    #:
    #: False positives do not get quietly filtered. They spawn tracks, compete
    #: in the association step, and fragment identity worse than the missing
    #: players do -- 0.10 produced nearly twice the identity switches of 0.25.
    #:
    #: 0.25 is the compromise: identity clean enough to watch, at 14 of 22
    #: players seen. None of these is good enough for analysis, which is the
    #: real finding -- the detector has to improve, and no gate tunes around it.
    #:
    #: Two other things that seemed obvious and measured the other way:
    #: brightening the frame loses 16 points of recall, and smaller tiles lose 7.
    conf: float = 0.25
    ball_conf: float = 0.10    # the ball is small and faint; let it through
    iou_merge: float = 0.55
    imgsz: int = 1280
    max_det: int = 300
    device: str = "cuda"
    half: bool = True
    #: Optional single-class ball model run on the full frame at a higher resolution. On a
    #: 720p tactical cam the finetuned people detector sees the ball in ~30% of frames; a
    #: dedicated model at 1920 px saw it in 95% (measured on a public 3-minute clip). When it
    #: fires, its boxes replace the ball rows.
    ball_weights: Optional[str] = None
    ball_imgsz: int = 1920
    #: Ball candidates kept per frame, most confident first. Not 1: the ball model also
    #: finds the spare balls beside the pitch, and a still, unoccluded spare ball usually
    #: out-scores the match ball -- keeping only the best box put the "ball" beyond the far
    #: touchline in 87% of frames of the 30 s clip. The pipeline picks one per frame after
    #: it knows where each candidate is on the pitch (`pipeline.radar.choose_ball`).
    max_balls: int = 5


def _bgr(rgb: np.ndarray) -> np.ndarray:
    """RGB -> BGR for Ultralytics, which treats a numpy image as BGR (as cv2 reads it)
    and flips it to RGB itself. The package passes RGB everywhere -- kit descriptors
    need it -- and handing that straight to the model swapped red and blue: on the
    30 s tactical-cam clip the people model found 2-4 fewer players per frame."""
    return np.ascontiguousarray(rgb[..., ::-1])


class Detector:
    """Thin wrapper over an Ultralytics model, adding tiling and role mapping."""

    def __init__(self, weights: str | Path, cfg: DetectorConfig | None = None):
        from ultralytics import YOLO           # imported lazily: heavy, GPU-side
        self.cfg = cfg or DetectorConfig()
        self.model = YOLO(str(weights))
        self.names: dict[int, str] = dict(self.model.names)
        self.roles: dict[int, str] = map_class_names(self.names)
        unknown = {self.names[c] for c, r in self.roles.items() if r == Role.OTHER.value}
        if unknown:
            print(f"[detect] classes not recognised as football roles: {sorted(unknown)} "
                  f"-- edit _ROLE_PATTERNS if one of these is a player/ball class")
        print(f"[detect] class map: { {self.names[c]: r for c, r in self.roles.items()} }")
        self.ball_model = YOLO(str(self.cfg.ball_weights)) if self.cfg.ball_weights else None
        ball_classes = [c for c, r in self.roles.items() if r == Role.BALL.value]
        self._ball_cls = ball_classes[0] if ball_classes else None
        if self.ball_model is not None and self._ball_cls is None:
            # the main model has no ball class: give the dedicated model its own id
            self._ball_cls = max(self.names) + 1
            self.names[self._ball_cls] = "ball"
            self.roles[self._ball_cls] = Role.BALL.value

    # ------------------------------------------------------------------ run

    def _raw(self, images: list[np.ndarray]) -> list[np.ndarray]:
        res = self.model.predict(
            [_bgr(im) for im in images], imgsz=self.cfg.imgsz, conf=min(self.cfg.conf, self.cfg.ball_conf),
            iou=0.7, max_det=self.cfg.max_det, device=self.cfg.device,
            half=self.cfg.half, verbose=False,
        )
        out = []
        for r in res:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append(np.empty((0, 6), np.float32)); continue
            xyxy = b.xyxy.cpu().numpy()
            arr = np.column_stack([
                xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1],
                b.conf.cpu().numpy(), b.cls.cpu().numpy(),
            ]).astype(np.float32)
            out.append(arr)
        return out

    def detect(self, frame: np.ndarray, batch_tiles: int = 8) -> np.ndarray:
        """Detect on one RGB frame. Returns (N, 6) of x, y, w, h, conf, cls."""
        h, w = frame.shape[:2]
        tiles = ([(0, 0, w, h)] if self.cfg.tile <= 0
                 else tile_grid(w, h, self.cfg.tile, self.cfg.overlap))

        dets: list[np.ndarray] = []
        for i in range(0, len(tiles), batch_tiles):
            chunk = tiles[i:i + batch_tiles]
            crops = [frame[y0:y1, x0:x1] for x0, y0, x1, y1 in chunk]
            for (x0, y0, _, _), d in zip(chunk, self._raw(crops)):
                if len(d):
                    d = d.copy(); d[:, 0] += x0; d[:, 1] += y0
                    dets.append(d)

        if not dets:
            return self._with_dedicated_ball(frame, np.empty((0, 6), np.float32))
        d = np.vstack(dets)

        # Per-class confidence gate: the ball gets a lower bar than people.
        is_ball = np.array([self.roles.get(int(c)) == Role.BALL.value for c in d[:, 5]])
        gate = np.where(is_ball, self.cfg.ball_conf, self.cfg.conf)
        d = d[d[:, 4] >= gate]
        if not len(d):
            return self._with_dedicated_ball(frame, np.empty((0, 6), np.float32))

        # NMS per class, so a player standing on the ball doesn't suppress it.
        keep: list[int] = []
        for c in np.unique(d[:, 5]):
            idx = np.flatnonzero(d[:, 5] == c)
            keep.extend(idx[nms(d[idx, :4], d[idx, 4], self.cfg.iou_merge)])
        d = d[np.array(sorted(keep), dtype=int)]

        # Keep the `max_balls` most confident ball candidates; the pipeline picks one.
        ball_rows = np.array([self.roles.get(int(c)) == Role.BALL.value for c in d[:, 5]])
        if ball_rows.sum() > self.cfg.max_balls:
            bi = np.flatnonzero(ball_rows)
            drop = set(bi[np.argsort(-d[bi, 4])[self.cfg.max_balls:]].tolist())
            d = d[[i for i in range(len(d)) if i not in drop]]
        return self._with_dedicated_ball(frame, d)

    def _with_dedicated_ball(self, frame: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Replace the ball rows with the dedicated model's best boxes when it fires."""
        if self.ball_model is None:
            return d
        res = self.ball_model.predict(
            _bgr(frame), imgsz=self.cfg.ball_imgsz, conf=self.cfg.ball_conf, device=self.cfg.device,
            half=self.cfg.half, verbose=False,
        )[0]
        b = res.boxes
        if b is None or len(b) == 0:
            return d
        conf = b.conf.cpu().numpy()
        top = np.argsort(-conf)[:self.cfg.max_balls]
        xyxy = b.xyxy.cpu().numpy()[top]
        row = np.column_stack([
            xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1],
            conf[top], np.full(len(top), float(self._ball_cls)),
        ]).astype(np.float32)
        not_ball = np.array([self.roles.get(int(c)) != Role.BALL.value for c in d[:, 5]], bool) if len(d) else np.zeros(0, bool)
        return np.vstack([d[not_ball], row]) if len(d) else row

    def role_of(self, cls_id: float) -> str:
        return self.roles.get(int(cls_id), Role.OTHER.value)
