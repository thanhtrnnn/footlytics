"""Tests for Apache-2.0 RT-DETRv2 multi-class people detector (`people_detr.py` & `Detector`)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from footlytics.perception.detect import Detector, DetectorConfig
from footlytics.perception.people_detr import (
    PEOPLE_NAMES,
    PERSON_ONLY_NAMES,
    infer_people,
    infer_people_batch,
)
from footlytics.state.schema import Role


class _Out:
    def __init__(self, logits, boxes):
        self.logits = logits
        self.pred_boxes = boxes


class _FakePeopleModel:
    """Four queries: player (0), goalkeeper (1), off-pitch other (3), and weak noise."""

    def __call__(self, pixel_values):
        bsz, _, h, w = pixel_values.shape
        assert h % 32 == 0 and w % 32 == 0
        logits = torch.full((bsz, 4, 80), -10.0)
        logits[:, 0, 0] = 3.5   # player (~0.97)
        logits[:, 1, 1] = 2.2   # goalkeeper (~0.90)
        logits[:, 2, 3] = 2.0   # other (~0.88)
        logits[:, 3, 2] = -2.5  # weak referee (~0.076)
        boxes = torch.tensor(
            [[[200 / w, 300 / h, 40 / w, 90 / h],
              [80 / w, 260 / h, 36 / w, 88 / h],
              [600 / w, 350 / h, 34 / w, 84 / h],
              [500 / w, 400 / h, 30 / w, 70 / h]]] * bsz,
            dtype=torch.float32,
        )
        return _Out(logits, boxes)


def test_infer_people_pads_to_32_rescales_and_maps_roles():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Multi-role mode (collapse_roles=False) keeps player, goalkeeper, and other
    dets_native = infer_people(_FakePeopleModel(), frame, "cpu", thr=0.25, imgsz=0, collapse_roles=False)
    assert dets_native.shape == (3, 6)
    assert int(dets_native[0, 5]) == 0 and PEOPLE_NAMES[int(dets_native[0, 5])] == Role.PLAYER.value
    assert int(dets_native[1, 5]) == 1 and PEOPLE_NAMES[int(dets_native[1, 5])] == Role.GOALKEEPER.value
    assert int(dets_native[2, 5]) == 3 and PEOPLE_NAMES[int(dets_native[2, 5])] == Role.OTHER.value
    assert [round(float(v)) for v in dets_native[0, :4]] == [180, 255, 40, 90]

    # Collapsed mode (collapse_roles=True) drops 'other' (cls 3) and maps on-pitch people to cls 0
    dets_collapsed = infer_people(_FakePeopleModel(), frame, "cpu", thr=0.25, imgsz=1280, collapse_roles=True)
    assert dets_collapsed.shape == (2, 6)
    assert (dets_collapsed[:, 5] == 0).all()
    assert [round(float(v)) for v in dets_collapsed[0, :4]] == [270, 382, 60, 135]


def test_infer_people_batch_handles_variable_tile_sizes():
    c1 = np.zeros((720, 1280, 3), dtype=np.uint8)
    c2 = np.zeros((500, 640, 3), dtype=np.uint8)
    batch_out = infer_people_batch(_FakePeopleModel(), [c1, c2], "cpu", thr=0.25, collapse_roles=True)
    assert len(batch_out) == 2
    assert batch_out[0].shape == (2, 6)
    assert batch_out[1].shape == (2, 6)


def test_detector_rtdetr_people_mode_runs_without_ultralytics(monkeypatch):
    class _StubPeopleDetr:
        def __init__(self, path: str, device: str, thr: float, max_n: int, imgsz: int = 1280):
            self.names = dict(PERSON_ONLY_NAMES)
            self.model = _FakePeopleModel()
            self.device = "cpu"
            self.thr = thr
            self.max_n = max_n
            self.imgsz = imgsz

        def predict_batch(self, crops, thr=None, max_n=None):
            return infer_people_batch(
                self.model, crops, "cpu", thr=thr or self.thr, max_n=max_n or self.max_n,
                imgsz=self.imgsz, collapse_roles=True,
            )

    import footlytics.perception.people_detr as pmod

    monkeypatch.setattr(pmod, "PeopleDetr", _StubPeopleDetr)
    det = Detector(
        "weights/people-rtdetrv2/best",
        DetectorConfig(tile=0, conf=0.25, device="cpu", half=False, people_detector="rtdetr"),
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = det.detect(frame)
    assert out.shape == (2, 6)
    assert det.role_of(out[0, 5]) == Role.PLAYER.value
    assert Role.REFEREE.value not in set(det.roles.values())
