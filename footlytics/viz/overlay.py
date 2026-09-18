"""Video plus radar, side by side -- the view that lets a human check the data.

A radar on its own is unfalsifiable: dots move plausibly whether or not they
correspond to the right players. Putting the source frame above it, with each
tracked box drawn and labelled, makes every error visible at a glance -- a
swapped identity, a missed player, a homography that puts someone in the wrong
part of the pitch.

This is also the surface a correcting operator works on, which is how commercial
systems actually reach usable identity.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..state.schema import MatchState, Role, Team
from .radar import TEAM_COLORS, draw_pitch
from ..geometry.pitch import Pitch


def _rgb(c) -> tuple[int, int, int]:
    if isinstance(c, str):
        c = c.lstrip("#")
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    return tuple(int(255 * v) for v in c[:3])


def render_side_by_side(
    state: MatchState,
    video_path: str,
    out_path: str,
    start: int = 0,
    end: Optional[int] = None,
    width: int = 1280,
    crop: Optional[tuple[int, int, int, int]] = None,
    smooth_display: bool = True,
    fps: Optional[float] = None,
    trail_frames: int = 20,
    color_by: str = "track",
    draw_boxes: bool = True,
    progress: bool = True,
) -> str:
    """Stack the source video (with boxes) above the 2D radar.

    `crop` is (x0, y0, x1, y1) in source pixels. A full-pitch panorama is mostly
    sky and stands; cropping to the playing area makes 40-pixel players actually
    visible at any sane output size.

    `smooth_display` cleans the radar's positions for display only -- single
    frame annotation spikes are rejected and the path lightly smoothed. The
    stored MatchState keeps raw coordinates, because once you smooth in place
    you can no longer tell a tracking fault from a real movement.
    """
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import imageio.v2 as imageio

    if smooth_display:
        from ..analytics.kinematics import smooth_positions
        state = MatchState(state.meta, smooth_positions(state.tracks, state.meta.fps))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    full_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cx0, cy0, cx1, cy1 = crop if crop else (0, 0, full_w, full_h)
    src_w, src_h = cx1 - cx0, cy1 - cy0
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = min(end if end is not None else state.n_frames, n_src)
    out_fps = fps or state.meta.fps

    scale = width / src_w
    vid_h = int(round(src_h * scale))

    pitch = Pitch(state.meta.pitch_length, state.meta.pitch_width)
    radar_h = int(width * (pitch.width + 6) / (pitch.length + 6))
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, radar_h / dpi), dpi=dpi)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.patch.set_facecolor("#0d1117")
    draw_pitch(ax, pitch)

    people = state.tracks[state.tracks["role"].isin(
        [Role.PLAYER.value, Role.GOALKEEPER.value, Role.REFEREE.value])]
    by_frame = {f: g for f, g in people.groupby("frame_idx")}
    ball_by_frame = {f: g for f, g in state.ball.groupby("frame_idx")}
    max_p = int(people.groupby("frame_idx").size().max()) if len(people) else 0

    dots = ax.scatter([], [], s=150, zorder=5, edgecolors="white", linewidths=1.4)
    ball_dot = ax.scatter([], [], s=80, c="white", edgecolors="black",
                          linewidths=1.1, zorder=8)
    labels = [ax.text(0, 0, "", ha="center", va="center", fontsize=6.5,
                      color="white", weight="bold", zorder=6) for _ in range(max_p)]
    trails = [ax.plot([], [], lw=1.3, alpha=0.45, zorder=3)[0] for _ in range(max_p)]

    def color_of(row):
        if color_by == "track":
            return cm.tab20(int(row["track_id"]) % 20)
        return TEAM_COLORS.get(str(row["team"]), TEAM_COLORS[Team.UNKNOWN.value])

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    with imageio.get_writer(out_path, fps=out_fps, macro_block_size=1, quality=8) as w:
        for fi in range(start, end):
            ok, bgr = cap.read()
            if not ok:
                break
            frame = cv2.resize(bgr[cy0:cy1, cx0:cx1], (width, vid_h))[:, :, ::-1].copy()

            g = by_frame.get(fi)
            cols = []
            if g is not None and len(g):
                cols = [color_of(r) for _, r in g.iterrows()]
                if draw_boxes:
                    for (_, r), c in zip(g.iterrows(), cols):
                        x0 = int((r["bbox_x"] - cx0) * scale)
                        y0 = int((r["bbox_y"] - cy0) * scale)
                        x1 = int((r["bbox_x"] + r["bbox_w"] - cx0) * scale)
                        y1 = int((r["bbox_y"] + r["bbox_h"] - cy0) * scale)
                        col = _rgb(c)
                        cv2.rectangle(frame, (x0, y0), (x1, y1), col, 1)
                        lab = (f"{int(r['jersey'])}" if not np.isnan(r["jersey"])
                               else f"{int(r['track_id'])}")
                        cv2.putText(frame, lab, (x0, max(y0 - 3, 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)

                dots.set_offsets(g[["x", "y"]].to_numpy())
                dots.set_facecolor(cols)
            else:
                dots.set_offsets(np.empty((0, 2)))

            for k, t in enumerate(labels):
                if g is not None and k < len(g):
                    r = g.iloc[k]
                    t.set_position((r["x"], r["y"]))
                    t.set_text(f"{int(r['jersey'])}" if not np.isnan(r["jersey"])
                               else f"{int(r['track_id'])}")
                else:
                    t.set_text("")

            for k, ln in enumerate(trails):
                if trail_frames > 0 and g is not None and k < len(g):
                    tid = int(g.iloc[k]["track_id"])
                    h = people[(people["track_id"] == tid)
                               & (people["frame_idx"] <= fi)
                               & (people["frame_idx"] > fi - trail_frames)]
                    ln.set_data(h["x"].to_numpy(), h["y"].to_numpy())
                    ln.set_color(cols[k] if k < len(cols) else "white")
                else:
                    ln.set_data([], [])

            b = ball_by_frame.get(fi)
            ball_dot.set_offsets(b[["x", "y"]].to_numpy()
                                 if b is not None and len(b) else np.empty((0, 2)))

            fig.canvas.draw()
            radar = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            if radar.shape[1] != width:
                radar = np.array(
                    cv2.resize(radar, (width, int(radar.shape[0] * width / radar.shape[1]))))

            combined = np.vstack([frame, radar])
            if combined.shape[0] % 2:
                combined = combined[:-1]
            w.append_data(combined)
            if progress and (fi - start) % 100 == 0:
                print(f"  side-by-side {fi - start}/{end - start}", end="\r", flush=True)

    cap.release(); plt.close(fig)
    if progress:
        print(f"  written to {out_path}                      ")
    return out_path
