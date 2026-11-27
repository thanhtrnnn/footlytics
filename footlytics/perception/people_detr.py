"""Multi-class football people detector on RT-DETRv2 (Apache-2.0 architecture and base weights),
an option under evaluation to replace the AGPL Ultralytics YOLOv8x6 people model. YOLOv8x6 remains
the default (`DetectorConfig.people_detector`).

Fine-tuned by `scripts/train_people_detr.py` from `PekingU/rtdetr_v2_r50vd` on SoccerNet GSR train.
Indices `0..3` are trained as `player`, `goalkeeper`, `referee`, and `other` (seeded from COCO's
pretrained `person` prior at index `0`). At pipeline inference time (`collapse_roles=True` default),
off-pitch `other` detections (`cls == 3`) are dropped and on-pitch person roles (`0, 1, 2`) are
emitted as `player` (`0`) so whole-clip trajectory role resolution (`split_officials`,
`goalkeeper_tracklets`, `team_scores_by_track`) decides officials and keepers without confusing a
dark-kit team for referees.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..state.schema import Role

PEOPLE_NAMES: dict[int, str] = {
    0: Role.PLAYER.value,
    1: Role.GOALKEEPER.value,
    2: Role.REFEREE.value,
    3: Role.OTHER.value,
}
PERSON_ONLY_NAMES: dict[int, str] = {0: Role.PLAYER.value}
NUM_PEOPLE_CLASSES = 4


def infer_people_batch(
    model,
    crops_rgb: list[np.ndarray],
    device,
    thr: float = 0.25,
    max_n: int = 300,
    imgsz: int = 1280,
    collapse_roles: bool = False,
) -> list[np.ndarray]:
    """Batch of RGB uint8 images -> list of `(N, 6)` float32 arrays `[x, y, w, h, conf, cls]`."""
    if not crops_rgb:
        return []
    import torch

    resized: list[np.ndarray] = []
    orig_sizes: list[tuple[int, int]] = []
    proc_sizes: list[tuple[int, int]] = []
    for im in crops_rgb:
        oh, ow = im.shape[:2]
        orig_sizes.append((oh, ow))
        if imgsz > 0 and max(oh, ow) > imgsz:
            scale_in = imgsz / float(max(oh, ow))
            nh = max(1, round(oh * scale_in))
            nw = max(1, round(ow * scale_in))
            rim = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            nh, nw = oh, ow
            rim = im
        proc_sizes.append((nh, nw))
        resized.append(rim)

    max_h = max(h for h, _ in proc_sizes)
    max_w = max(w for _, w in proc_sizes)
    pad_h = -(-max_h // 32) * 32
    pad_w = -(-max_w // 32) * 32

    x = torch.zeros((len(resized), 3, pad_h, pad_w), dtype=torch.float32, device=device)
    for i, (rim, (nh, nw)) in enumerate(zip(resized, proc_sizes)):
        t = torch.from_numpy(np.ascontiguousarray(rim)).to(device).permute(2, 0, 1).float() / 255.0
        x[i, :, :nh, :nw] = t

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
        out = model(pixel_values=x)

    probs = out.logits[:, :, :NUM_PEOPLE_CLASSES].float().sigmoid()
    boxes_all = out.pred_boxes.float()
    pad_scale = np.array([pad_w, pad_h, pad_w, pad_h], dtype=np.float32)

    results: list[np.ndarray] = []
    for i, ((oh, ow), (nh, nw)) in enumerate(zip(orig_sizes, proc_sizes)):
        conf_t, cls_t = probs[i].max(dim=-1)
        keep_mask = conf_t >= thr
        if collapse_roles:
            keep_mask = keep_mask & (cls_t != 3)
        keep = torch.nonzero(keep_mask).flatten()
        if len(keep) == 0:
            results.append(np.empty((0, 6), dtype=np.float32))
            continue
        keep = keep[conf_t[keep].argsort(descending=True)][:max_n]
        b = boxes_all[i, keep].cpu().numpy() * pad_scale
        sx = float(ow) / float(nw)
        sy = float(oh) / float(nh)
        b[:, 0] *= sx
        b[:, 2] *= sx
        b[:, 1] *= sy
        b[:, 3] *= sy
        s = conf_t[keep].cpu().numpy()
        c = (
            np.zeros(len(keep), dtype=np.float32)
            if collapse_roles
            else cls_t[keep].cpu().numpy().astype(np.float32)
        )
        x0 = np.clip(b[:, 0] - b[:, 2] / 2.0, 0.0, float(ow))
        y0 = np.clip(b[:, 1] - b[:, 3] / 2.0, 0.0, float(oh))
        x1 = np.clip(b[:, 0] + b[:, 2] / 2.0, 0.0, float(ow))
        y1 = np.clip(b[:, 1] + b[:, 3] / 2.0, 0.0, float(oh))
        bw = x1 - x0
        bh = y1 - y0
        valid = (bw > 1.0) & (bh > 1.0)
        if not np.any(valid):
            results.append(np.empty((0, 6), dtype=np.float32))
            continue
        arr = np.column_stack([x0[valid], y0[valid], bw[valid], bh[valid], s[valid], c[valid]])
        results.append(arr.astype(np.float32))
    return results


def infer_people(
    model,
    rgb: np.ndarray,
    device,
    thr: float = 0.25,
    max_n: int = 300,
    imgsz: int = 1280,
    collapse_roles: bool = False,
) -> np.ndarray:
    """Single RGB uint8 frame -> `(N, 6)` float32 `[x, y, w, h, conf, cls]`, best score first."""
    return infer_people_batch(
        model, [rgb], device, thr=thr, max_n=max_n, imgsz=imgsz, collapse_roles=collapse_roles
    )[0]


class PeopleDetr:
    """Loads a `save_pretrained` RT-DETRv2 people model directory (Apache-2.0, zero Ultralytics)."""

    def __init__(
        self,
        path: str,
        device: str = "cuda",
        thr: float = 0.25,
        max_n: int = 300,
        imgsz: int = 1280,
        collapse_roles: bool = True,
    ):
        from transformers import RTDetrV2ForObjectDetection

        self.device = device
        self.thr = thr
        self.max_n = max_n
        self.imgsz = imgsz
        self.collapse_roles = collapse_roles
        self.names: dict[int, str] = dict(PERSON_ONLY_NAMES if collapse_roles else PEOPLE_NAMES)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(path).to(device).eval()

    def predict_batch(
        self,
        crops_rgb: list[np.ndarray],
        thr: float | None = None,
        max_n: int | None = None,
    ) -> list[np.ndarray]:
        return infer_people_batch(
            self.model,
            crops_rgb,
            self.device,
            thr=self.thr if thr is None else thr,
            max_n=self.max_n if max_n is None else max_n,
            imgsz=self.imgsz,
            collapse_roles=self.collapse_roles,
        )

    def __call__(
        self,
        rgb: np.ndarray,
        thr: float | None = None,
        max_n: int | None = None,
    ) -> np.ndarray:
        return infer_people(
            self.model,
            rgb,
            self.device,
            thr=self.thr if thr is None else thr,
            max_n=self.max_n if max_n is None else max_n,
            imgsz=self.imgsz,
            collapse_roles=self.collapse_roles,
        )
