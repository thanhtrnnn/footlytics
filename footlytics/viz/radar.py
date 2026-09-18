"""The 2D tactical radar -- the thing you actually show a coach.

Deliberately matplotlib rather than a bespoke renderer: the output needs to be
droppable into a PDF report, a slide and a video, and this is the demo, not the
product surface. A WebGL version belongs in the coach app later.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..geometry.pitch import (
    Pitch, DEFAULT_PITCH, CENTRE_CIRCLE_R, PENALTY_AREA_DEPTH, PENALTY_AREA_WIDTH,
    GOAL_AREA_DEPTH, GOAL_AREA_WIDTH, PENALTY_SPOT_DIST, GOAL_WIDTH,
)
from ..state.schema import MatchState, Role, Team

PITCH_GREEN = "#1b7a3d"
LINE_WHITE = "#e9f5ec"
TEAM_COLORS = {
    Team.HOME.value: "#e63946",
    Team.AWAY.value: "#1d7fe0",
    Team.OFFICIAL.value: "#f4d35e",
    Team.UNKNOWN.value: "#9aa5ad",
}


def draw_pitch(ax, pitch: Pitch = DEFAULT_PITCH, facecolor: str = PITCH_GREEN,
               line: str = LINE_WHITE, lw: float = 1.4, pad: float = 3.0):
    """Draw pitch markings to scale from the pitch model itself."""
    from matplotlib.patches import Arc, Circle, Rectangle

    L, W = pitch.half_l, pitch.half_w
    # Draw the grass as an explicit patch rather than relying on
    # ax.set_facecolor: ax.axis("off") below hides the axes patch, so the face
    # colour silently never appears and the pitch renders on the figure's
    # background instead.
    ax.set_facecolor(facecolor)
    ax.add_patch(Rectangle((-L - pad, -W - pad), 2 * (L + pad), 2 * (W + pad),
                           facecolor=facecolor, ec="none", zorder=0))
    ax.add_patch(Rectangle((-L, -W), 2 * L, 2 * W, fill=False, ec=line, lw=lw, zorder=2))
    ax.plot([0, 0], [-W, W], color=line, lw=lw, zorder=2)
    ax.add_patch(Circle((0, 0), CENTRE_CIRCLE_R, fill=False, ec=line, lw=lw, zorder=2))
    ax.add_patch(Circle((0, 0), 0.35, color=line, zorder=2))

    for sx in (-1, 1):
        pa_x, pa_y = L - PENALTY_AREA_DEPTH, PENALTY_AREA_WIDTH / 2
        ga_x, ga_y = L - GOAL_AREA_DEPTH, GOAL_AREA_WIDTH / 2
        ax.add_patch(Rectangle((min(sx * L, sx * pa_x), -pa_y), PENALTY_AREA_DEPTH,
                               2 * pa_y, fill=False, ec=line, lw=lw, zorder=2))
        ax.add_patch(Rectangle((min(sx * L, sx * ga_x), -ga_y), GOAL_AREA_DEPTH,
                               2 * ga_y, fill=False, ec=line, lw=lw, zorder=2))
        spot = sx * (L - PENALTY_SPOT_DIST)
        ax.add_patch(Circle((spot, 0), 0.3, color=line, zorder=2))
        # Only the arc outside the penalty area is drawn, hence the angle maths.
        half = np.degrees(np.arccos(
            np.clip((PENALTY_AREA_DEPTH - PENALTY_SPOT_DIST) / CENTRE_CIRCLE_R, -1, 1)))
        centre_ang = 0.0 if sx < 0 else 180.0
        ax.add_patch(Arc((spot, 0), 2 * CENTRE_CIRCLE_R, 2 * CENTRE_CIRCLE_R,
                         theta1=centre_ang - half, theta2=centre_ang + half,
                         ec=line, lw=lw, zorder=2))
        ax.add_patch(Rectangle((sx * L if sx > 0 else sx * L - 2.0, -GOAL_WIDTH / 2),
                               2.0, GOAL_WIDTH, fill=False, ec=line, lw=lw * 1.4, zorder=2))

    ax.set_xlim(-L - pad, L + pad)
    ax.set_ylim(-W - pad, W + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def plot_frame(state: MatchState, frame_idx: int, ax=None,
               show_ids: bool = True, show_velocity: bool = True,
               trail_frames: int = 0, color_by: str = "team"):
    """Render one frame of the match onto a pitch.

    `color_by` is "team" or "track". Use "track" for data that genuinely has no
    team labels -- colouring by a guessed team is worse than showing none,
    because a wrong 11/11 split looks exactly as convincing as a right one.
    """
    import matplotlib.pyplot as plt

    pitch = Pitch(state.meta.pitch_length, state.meta.pitch_width)
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax, pitch)

    if trail_frames > 0:
        past = state.players[
            (state.players["frame_idx"] < frame_idx)
            & (state.players["frame_idx"] >= frame_idx - trail_frames)
        ]
        for tid, g in past.groupby("track_id", observed=True):
            g = g.sort_values("frame_idx")
            import matplotlib.cm as cm
            col = (cm.tab20(int(tid) % 20) if color_by == "track"
                   else TEAM_COLORS.get(str(g["team"].iloc[0]),
                                        TEAM_COLORS[Team.UNKNOWN.value]))
            ax.plot(g["x"], g["y"], color=col, alpha=0.45, lw=1.6, zorder=3)

    f = state.frame(frame_idx)
    people = f[f["role"].isin([Role.PLAYER.value, Role.GOALKEEPER.value, Role.REFEREE.value])]
    import matplotlib.cm as cm
    for _, r in people.iterrows():
        if color_by == "track":
            col = cm.tab20(int(r["track_id"]) % 20)
        else:
            col = TEAM_COLORS.get(str(r["team"]), TEAM_COLORS[Team.UNKNOWN.value])
        is_gk = r["role"] == Role.GOALKEEPER.value
        ax.scatter([r["x"]], [r["y"]], s=210 if is_gk else 170, color=col, zorder=5,
                   edgecolors="white", linewidths=1.6, marker="s" if is_gk else "o")
        if show_ids:
            label = (f"{int(r['jersey'])}" if not np.isnan(r["jersey"])
                     else f"{int(r['track_id'])}")
            ax.text(r["x"], r["y"], label, ha="center", va="center",
                    fontsize=7, color="white", weight="bold", zorder=6)
        if show_velocity and not np.isnan(r.get("speed", np.nan)) and r["speed"] > 1.0:
            # 1 second of travel at the current speed, in the direction of motion.
            g = state.tracks[(state.tracks["track_id"] == r["track_id"])
                             & (state.tracks["frame_idx"].between(frame_idx - 5, frame_idx))]
            if len(g) >= 2:
                dx = g["x"].iloc[-1] - g["x"].iloc[0]
                dy = g["y"].iloc[-1] - g["y"].iloc[0]
                n = np.hypot(dx, dy)
                if n > 1e-6:
                    ax.arrow(r["x"], r["y"], dx / n * r["speed"], dy / n * r["speed"],
                             color="white", alpha=0.75, width=0.12,
                             head_width=0.7, length_includes_head=True, zorder=4)

    b = f[f["role"] == Role.BALL.value]
    if len(b):
        ax.scatter(b["x"], b["y"], s=95, c="white", edgecolors="black",
                   linewidths=1.3, zorder=8, marker="o")

    t = frame_idx / state.meta.fps
    ax.set_title(
        f"{state.meta.home.name}  vs  {state.meta.away.name}"
        f"      {int(t // 60):02d}:{int(t % 60):02d}   (frame {frame_idx})",
        color="white", fontsize=12, pad=10,
    )
    return ax


def render_video(state: MatchState, out_path: str, start: int = 0, end: Optional[int] = None,
                 step: int = 1, fps: Optional[float] = None, dpi: int = 100,
                 trail_frames: int = 12, progress: bool = True,
                 color_by: str = "team", figsize=(12, 8)) -> str:
    """Render a range of frames to an mp4 radar clip.

    The pitch and every marker are created once and then *updated* per frame.
    Clearing the axes and redrawing from scratch, as the first version did,
    re-creates dozens of patches per frame and makes a 90-minute render
    impractical; artist reuse is roughly an order of magnitude faster and is the
    difference between a demo and a product.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import imageio.v2 as imageio

    end = end if end is not None else state.n_frames
    frames = list(range(start, end, step))
    out_fps = fps if fps is not None else state.meta.fps / step
    pitch = Pitch(state.meta.pitch_length, state.meta.pitch_width)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("#0d1117")
    draw_pitch(ax, pitch)

    # Pre-index by frame once: repeated boolean masks over a 1.5M-row frame are
    # the other half of the cost.
    people = state.tracks[state.tracks["role"].isin(
        [Role.PLAYER.value, Role.GOALKEEPER.value, Role.REFEREE.value])]
    by_frame = {f: g for f, g in people.groupby("frame_idx")}
    ball_by_frame = {f: g for f, g in state.ball.groupby("frame_idx")}

    max_players = int(people.groupby("frame_idx").size().max()) if len(people) else 0
    dots = ax.scatter([], [], s=170, zorder=5, edgecolors="white", linewidths=1.5)
    ball_dot = ax.scatter([], [], s=90, c="white", edgecolors="black",
                          linewidths=1.2, zorder=8)
    labels = [ax.text(0, 0, "", ha="center", va="center", fontsize=7,
                      color="white", weight="bold", zorder=6)
              for _ in range(max_players)]
    trails = [ax.plot([], [], lw=1.5, alpha=0.45, zorder=3)[0]
              for _ in range(max_players)]
    title = ax.set_title("", color="white", fontsize=12, pad=10)

    def color_of(row):
        if color_by == "track":
            return cm.tab20(int(row["track_id"]) % 20)
        return TEAM_COLORS.get(str(row["team"]), TEAM_COLORS[Team.UNKNOWN.value])

    with imageio.get_writer(out_path, fps=out_fps, macro_block_size=1,
                            quality=8) as w:
        for i, fi in enumerate(frames):
            g = by_frame.get(fi)
            if g is None or g.empty:
                xy, cols = np.empty((0, 2)), []
            else:
                xy = g[["x", "y"]].to_numpy()
                cols = [color_of(r) for _, r in g.iterrows()]
            dots.set_offsets(xy if len(xy) else np.empty((0, 2)))
            if len(cols):
                dots.set_facecolor(cols)

            for k, t in enumerate(labels):
                if g is not None and k < len(g):
                    r = g.iloc[k]
                    lab = (f"{int(r['jersey'])}" if not np.isnan(r["jersey"])
                           else f"{int(r['track_id'])}")
                    t.set_position((r["x"], r["y"])); t.set_text(lab)
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
            ball_dot.set_offsets(b[["x", "y"]].to_numpy() if b is not None and len(b)
                                 else np.empty((0, 2)))

            t_s = fi / state.meta.fps
            title.set_text(f"{state.meta.home.name}  vs  {state.meta.away.name}"
                           f"      {int(t_s // 60):02d}:{int(t_s % 60):02d}")
            fig.canvas.draw()
            w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            if progress and i % 100 == 0:
                print(f"  radar {i}/{len(frames)}", end="\r", flush=True)

    plt.close(fig)
    if progress:
        print(f"  radar written to {out_path}                    ")
    return out_path
