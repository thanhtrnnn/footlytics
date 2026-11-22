import numpy as np
import pytest

torch = pytest.importorskip("torch")

from footlytics.perception.ball_detr import BALL_CLS, infer


class _Out:
    def __init__(self, logits, boxes):
        self.logits, self.pred_boxes = logits, boxes


class _FakeModel:
    """Two queries: a confident ball at (100, 50) 20x20 px and a weak one."""
    def __call__(self, pixel_values):
        _, _, H, W = pixel_values.shape
        assert H % 32 == 0 and W % 32 == 0
        logits = torch.full((1, 2, 80), -10.0)
        logits[0, 0, BALL_CLS], logits[0, 1, BALL_CLS] = 3.0, -1.0
        boxes = torch.tensor([[[110 / W, 60 / H, 20 / W, 20 / H], [500 / W, 400 / H, 12 / W, 12 / H]]])
        return _Out(logits, boxes)


def test_infer_pads_to_32_and_returns_pixel_boxes_best_first():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    c = infer(_FakeModel(), frame, "cpu", thr=0.05)
    assert [round(v) for v in c[0][:4]] == [100, 50, 20, 20] and c[0][4] > c[1][4]
    assert len(infer(_FakeModel(), frame, "cpu", thr=0.5)) == 1
    assert len(infer(_FakeModel(), frame, "cpu", thr=0.05, max_n=1)) == 1
