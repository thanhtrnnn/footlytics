"""Overlay rendering (boxes, ids, team colours) onto video frames."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

TEAM_COLOURS = {"0": (0, 0, 255), "1": (255, 128, 0), "ref": (0, 255, 255), "unknown": (200, 200, 200)}
BALL_COLOUR = (0, 255, 255)


def draw_frame(frame: np.ndarray, rows: pd.DataFrame, ball_xy: tuple[float, float] | None) -> np.ndarray:
    out = frame.copy()
    for r in rows.itertuples(index=False):
        colour = TEAM_COLOURS.get(str(r.team), TEAM_COLOURS["unknown"])
        x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(out, str(int(r.track_id)), (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    if ball_xy is not None and not np.isnan(ball_xy[0]):
        cv2.circle(out, (int(ball_xy[0]), int(ball_xy[1])), 6, BALL_COLOUR, 2)
    return out


def write_overlay_video(frames_iter, tracks: pd.DataFrame, ball: pd.DataFrame, out_path: str | Path, fps: float) -> None:
    """frames_iter yields (idx, frame). tracks has track_id/x1..y2/team per frame; ball has frame/ball_x_m/ball_y_m."""
    by_frame = {f: g for f, g in tracks.groupby("frame")}
    ball_by_frame = ball.set_index("frame")[["ball_x_m", "ball_y_m"]].to_dict("index") if len(ball) else {}
    writer = None
    for idx, frame in frames_iter:
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        rows = by_frame.get(idx, tracks.iloc[0:0])
        b = ball_by_frame.get(idx)
        ball_xy = (b["ball_x_m"], b["ball_y_m"]) if b else None
        writer.write(draw_frame(frame, rows, ball_xy))
    if writer is not None:
        writer.release()
