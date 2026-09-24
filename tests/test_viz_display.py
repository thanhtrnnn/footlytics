"""Display-only fixes in viz: pitch orientation and bridging short track gaps."""
import numpy as np
import pandas as pd

from footlytics.state.schema import Role, Team
from footlytics.viz.radar import draw_pitch, hold_gaps


def _rows(tid, frames, role=Role.PLAYER.value):
    return pd.DataFrame([dict(frame_idx=f, timestamp=f / 25.0, track_id=tid, role=role,
                              team=Team.HOME.value, x=float(f), y=0.0, bbox_x=10.0 * f,
                              bbox_y=0.0, bbox_w=5.0, bbox_h=20.0) for f in frames])


def test_short_gap_is_bridged_long_gap_is_not():
    t = _rows(1, [0, 1, 4, 5, 30])            # gap of 2 (frames 2-3), gap of 24 (6-29)
    out = hold_gaps(t, max_gap=12)
    frames = sorted(out.loc[out.track_id == 1, "frame_idx"])
    assert frames == [0, 1, 2, 3, 4, 5, 30]
    filled = out[out.frame_idx.isin([2, 3])].sort_values("frame_idx")
    assert filled["x"].tolist() == [2.0, 3.0]
    assert filled["bbox_x"].tolist() == [20.0, 30.0]


def test_ball_and_zero_are_left_alone():
    t = pd.concat([_rows(1, [0, 3]), _rows(9, [0, 3], role=Role.BALL.value)])
    assert len(hold_gaps(t, max_gap=0)) == len(t)
    out = hold_gaps(t, max_gap=12)
    assert sorted(out.loc[out.track_id == 9, "frame_idx"]) == [0, 3]


def test_touchline_T_is_drawn_at_the_top():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    draw_pitch(ax)
    lo, hi = ax.get_ylim()
    assert lo > hi                      # y grows downward: negative y (T) on top
    plt.close(fig)
