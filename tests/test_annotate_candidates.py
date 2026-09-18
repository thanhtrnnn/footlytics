import numpy as np

from footlytics.geometry.annotate import draw_candidates


def test_draw_candidates_marks_points_and_leaves_input_untouched():
    frame = np.zeros((100, 160, 3), np.uint8)
    cands = [{"id": 0, "x": 40.0, "y": 50.0}, {"id": 1, "x": 120.0, "y": 20.0}]
    out = draw_candidates(frame, cands)
    assert out.shape == frame.shape
    assert frame.max() == 0                                  # input not modified
    assert out[50, 40].max() > 0 and out[20, 120].max() > 0  # a mark at each candidate
