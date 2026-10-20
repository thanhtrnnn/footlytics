"""Football detector on RT-DETRv2 (Apache-2.0 code and weights), a licence-clean alternative to
the AGPL YOLO ball model.

Fine-tuned by `scripts/train_ball_detr.py` from `PekingU/rtdetr_v2_r50vd` on SoccerNet GSR train.
The COCO head is kept and only its "sports ball" class (32) is used. The network is fully
convolutional, so a 1080p frame goes in whole (padded to a multiple of 32) at native resolution:
the ball is ~18 px and downscaling it is what the YOLO model loses.
"""
from __future__ import annotations

import numpy as np

BALL_CLS = 32


def infer(model, rgb: np.ndarray, device, thr: float = 0.05, max_n: int = 15) -> list[tuple]:
    """One RGB uint8 frame -> [(x, y, w, h, score)] in pixels, best first."""
    import torch

    h, w = rgb.shape[:2]
    H, W = -(-h // 32) * 32, -(-w // 32) * 32
    x = torch.zeros(1, 3, H, W, dtype=torch.float32, device=device)
    x[0, :, :h, :w] = torch.from_numpy(np.ascontiguousarray(rgb)).to(device).permute(2, 0, 1).float() / 255.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
        out = model(pixel_values=x)
    score = out.logits[0, :, BALL_CLS].float().sigmoid()
    keep = torch.nonzero(score >= thr).flatten()
    keep = keep[score[keep].argsort(descending=True)][:max_n]
    if not len(keep):
        return []
    b = out.pred_boxes[0, keep].float().cpu().numpy() * np.array([W, H, W, H], np.float32)
    s = score[keep].cpu().numpy()
    return [(float(cx - bw / 2), float(cy - bh / 2), float(bw), float(bh), float(c))
            for (cx, cy, bw, bh), c in zip(b, s)]


class BallDetr:
    """Loads a `save_pretrained` directory; `__call__(rgb)` -> (N, 6) x, y, w, h, conf, cls."""

    def __init__(self, path: str, device: str = "cuda", thr: float = 0.05, max_n: int = 15):
        from transformers import RTDetrV2ForObjectDetection

        self.device, self.thr, self.max_n = device, thr, max_n
        self.model = RTDetrV2ForObjectDetection.from_pretrained(path).to(device).eval()

    def __call__(self, rgb: np.ndarray, cls_id: float, thr: float | None = None,
                 max_n: int | None = None) -> np.ndarray:
        c = infer(self.model, rgb, self.device, self.thr if thr is None else thr,
                  self.max_n if max_n is None else max_n)
        return np.array([[x, y, w, h, s, cls_id] for x, y, w, h, s in c], np.float32).reshape(-1, 6)
