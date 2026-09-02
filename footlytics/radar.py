"""2D pitch (radar) rendering of the game-state table."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from footlytics.homography import PITCH_LENGTH_M, PITCH_WIDTH_M

SCALE = 8  # px per metre
MARGIN = 3  # metres
TEAM_COLOURS = {"0": (0, 0, 255), "1": (255, 128, 0), "ref": (0, 255, 255), "unknown": (200, 200, 200)}


def _to_px(x_m: float, y_m: float) -> tuple[int, int]:
    return int((x_m + MARGIN) * SCALE), int((y_m + MARGIN) * SCALE)


def draw_pitch() -> np.ndarray:
    w, h = int((PITCH_LENGTH_M + 2 * MARGIN) * SCALE), int((PITCH_WIDTH_M + 2 * MARGIN) * SCALE)
    img = np.full((h, w, 3), (60, 140, 60), np.uint8)
    white = (255, 255, 255)
    cv2.rectangle(img, _to_px(0, 0), _to_px(PITCH_LENGTH_M, PITCH_WIDTH_M), white, 2)
    cv2.line(img, _to_px(52.5, 0), _to_px(52.5, PITCH_WIDTH_M), white, 2)
    cv2.circle(img, _to_px(52.5, 34), int(9.15 * SCALE), white, 2)
    for x0, sgn in ((0, 1), (PITCH_LENGTH_M, -1)):
        cv2.rectangle(img, _to_px(x0, 34 - 20.16), _to_px(x0 + sgn * 16.5, 34 + 20.16), white, 2)
        cv2.rectangle(img, _to_px(x0, 34 - 9.16), _to_px(x0 + sgn * 5.5, 34 + 9.16), white, 2)
        cv2.circle(img, _to_px(x0 + sgn * 11, 34), 3, white, -1)
    return img


def draw_radar_frame(rows: pd.DataFrame, ball_xy: tuple[float, float] | None = None) -> np.ndarray:
    img = draw_pitch()
    for r in rows.itertuples(index=False):
        if np.isnan(r.x_m) or np.isnan(r.y_m):
            continue
        cv2.circle(img, _to_px(r.x_m, r.y_m), 7, TEAM_COLOURS.get(str(r.team), TEAM_COLOURS["unknown"]), -1)
        cv2.putText(img, str(int(r.player_id)), _to_px(r.x_m + 0.8, r.y_m - 0.8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    if ball_xy is not None and not np.isnan(ball_xy[0]):
        cv2.circle(img, _to_px(*ball_xy), 5, (255, 255, 255), -1)
        cv2.circle(img, _to_px(*ball_xy), 5, (0, 0, 0), 1)
    return img


def write_radar_video(gs: pd.DataFrame, out_path: str | Path, fps: float) -> None:
    frames = sorted(gs["frame"].unique())
    if not frames:
        return
    sample = draw_pitch()
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (sample.shape[1], sample.shape[0]))
    by_frame = {f: g for f, g in gs.groupby("frame")}
    for f in range(int(frames[0]), int(frames[-1]) + 1):
        g = by_frame.get(f)
        if g is None:
            writer.write(sample)
            continue
        b = g.iloc[0]
        writer.write(draw_radar_frame(g, (b["ball_x_m"], b["ball_y_m"])))
    writer.release()
