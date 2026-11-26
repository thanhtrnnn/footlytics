#!/usr/bin/env python3
"""Build the 1080p@30fps Footlytics Founder Demo Video Showcase.

Renders a self-contained H.264 MP4 (`1920x1080`, `yuv420p`, `+faststart`) from
real processed LPBank V.League 1 2025/26 MatchStates, opposition pack clips, and
live workspace/dossier surfaces:

- Act I   (0:00 - 0:13.0): Editorial hook + live 1080p pitch calibration & tracking reveal
- Act II  (0:13 - 0:45.5): All 5 V1 tactical sequence engines across 3 matches & 2 stadiums
- Act III (0:45.5 - 0:58.5): Interactive Web Tactical Workspace & Opposition Scouting Pack
- Act IV  (0:58.5 - 1:07.5): Verified V1 precision, Apache-2.0 benchmarks, & unit economics
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT_DISPLAY = ROOT / "web" / "assets" / "bricolage-grotesque.ttf"
FONT_MONO_CANDIDATES = (
    Path("/System/Library/Fonts/SFNSMono.ttf"),
    Path("/System/Library/Fonts/Menlo.ttc"),
    Path("/System/Library/Fonts/Supplemental/Courier New Bold.ttf"),
)

W, H = 1920, 1080
FPS = 30

# Brand palette (RGB for Pillow, converted to BGR where needed)
C_BG_DEEP = (11, 24, 19)
C_BG_STAGE = (16, 38, 29)
C_CARD = (20, 46, 36)
C_CARD_DARK = (13, 29, 23)
C_RULE = (52, 84, 69)
C_PAPER = (242, 241, 233)
C_MUTED = (176, 194, 180)
C_SIGNAL = (229, 240, 108)  # #e5f06c signature chartreuse
C_CYAN = (128, 201, 204)    # #80c9cc away / secondary accent
C_CORAL = (251, 113, 133)   # #fb7185 home / alert accent
C_GREEN = (34, 197, 94)
C_AMBER = (245, 158, 11)
C_INK = (20, 34, 25)


def rgb_to_bgr(c: tuple[int, int, int]) -> tuple[int, int, int]:
    return (c[2], c[1], c[0])


def ease_out_cubic(t: float) -> float:
    x = float(np.clip(t, 0.0, 1.0))
    return 1.0 - (1.0 - x) ** 3


def ease_in_out_cubic(t: float) -> float:
    x = float(np.clip(t, 0.0, 1.0))
    return 4.0 * x * x * x if x < 0.5 else 1.0 - ((-2.0 * x + 2.0) ** 3) / 2.0


class FontBank:
    """Cached Bricolage Grotesque variable font + system monospace loader."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, int, int], ImageFont.FreeTypeFont] = {}
        self._mono_path: Path | None = next((p for p in FONT_MONO_CANDIDATES if p.exists()), None)

    def display(self, size: int, weight: int = 600, width: int = 100) -> ImageFont.FreeTypeFont:
        key = ("display", size, weight, width)
        if key not in self._cache:
            f = ImageFont.truetype(str(FONT_DISPLAY), size)
            opsz = int(np.clip(size, 12, 96))
            f.set_variation_by_axes([opsz, int(np.clip(weight, 200, 800)), int(np.clip(width, 75, 100))])
            self._cache[key] = f
        return self._cache[key]

    def mono(self, size: int) -> ImageFont.FreeTypeFont:
        key = ("mono", size, 0, 0)
        if key not in self._cache:
            if self._mono_path is not None:
                self._cache[key] = ImageFont.truetype(str(self._mono_path), size)
            else:
                self._cache[key] = self.display(size, weight=500)
        return self._cache[key]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    cur = words[0]
    for w in words[1:]:
        cand = f"{cur} {w}"
        bbox = draw.textbbox((0, 0), cand, font=font)
        if bbox[2] - bbox[0] <= max_w:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def format_match_clock(elapsed_s: float, period: int) -> str:
    s = max(0, round(float(elapsed_s)))
    half = 45 * 60
    if period == 1:
        if s <= half:
            m, sec = divmod(s, 60)
            return f"H1 {m:02d}:{sec:02d}"
        m, sec = divmod(s - half, 60)
        return f"H1 45+{m:02d}:{sec:02d}"
    if s <= half:
        m, sec = divmod(half + s, 60)
        return f"H2 {m:02d}:{sec:02d}"
    m, sec = divmod(s - half, 60)
    return f"H2 90+{m:02d}:{sec:02d}"


@dataclass(frozen=True)
class SequenceShowcaseSpec:
    step_idx: int
    code: str
    title_en: str
    title_vi: str
    match_id: str
    match_title: str
    stadium: str
    clip_path: Path
    clip_t0: float
    clip_start_state_s: float
    seq_start_s: float
    seq_end_s: float
    period: int
    period_start_s: float
    focus_team: str  # 'home' or 'away'
    home_code: str
    away_code: str
    rule_text: str
    insight_text: str
    badge_text: str
    flank: str = ""
    summary_stats: tuple[tuple[str, str, float], ...] = ()


def build_sequence_specs() -> list[SequenceShowcaseSpec]:
    packs = ROOT / "data" / "out" / "packs"
    return [
        SequenceShowcaseSpec(
            step_idx=0,
            code="high_press",
            title_en="High Press",
            title_vi="Pressing tầm cao",
            match_id="hanoi_pvf_cand_2526",
            match_title="Hà Nội FC vs PVF-CAND",
            stadium="Sân Hàng Đẫy · Hà Nội",
            clip_path=packs / "hanoi_2match_pack/clips/01_high_press/001_hanoi_pvf_cand_2526_P1_0393s_high_press_HANOI.mp4",
            clip_t0=2.8,
            clip_start_state_s=390.0,
            seq_start_s=393.0,
            seq_end_s=397.0,
            period=1,
            period_start_s=145.0,
            focus_team="home",
            home_code="HANOI",
            away_code="PVFCAND",
            rule_text="Rule: Out-of-possession defensive line > +8.0 m into opponent half with >= 4 players closing within 10 m of the ball.",
            insight_text="Hà Nội FC pushes 6 players inside a 10 m pressing trap on the left touchline (+28.7 m line height), forcing an immediate turnover.",
            badge_text="HANG DAY STADIUM · LIVE 30 HZ",
            summary_stats=(
                ("Defensive Line Height", "+28.7 m", 0.84),
                ("Pressers Within 10m", "6 players", 0.86),
                ("Team Compactness", "10.6 m", 0.78),
            ),
        ),
        SequenceShowcaseSpec(
            step_idx=1,
            code="build_up",
            title_en="Build-up vs Press",
            title_vi="Triển khai bóng sân nhà",
            match_id="pvf_cand_hanoi_2526",
            match_title="PVF-CAND vs Hà Nội FC",
            stadium="Sân PVF · Hưng Yên",
            clip_path=packs / "pvf_cand_hanoi_2526_verified/clips/02_build_up/007_pvf_cand_hanoi_2526_P1_2818s_build_up_PVFCAND.mp4",
            clip_t0=2.6,
            clip_start_state_s=2815.17,
            seq_start_s=2818.17,
            seq_end_s=2824.33,
            period=1,
            period_start_s=160.0,
            focus_team="home",
            home_code="PVFCAND",
            away_code="HANOI",
            rule_text="Rule: Controlled own-third possession (x <= -17.5 m for >= 4.0 s) progressing cleanly into midfield under opponent pressure.",
            insight_text="PVF-CAND absorbs 3 pressing Hà Nội forwards in the right defensive third (-17.7 m) and progresses cleanly up the far touchline.",
            badge_text="HUMAN VERIFIED · PVF STADIUM",
            summary_stats=(
                ("Opponent Pressers in 3rd", "3 forwards", 0.65),
                ("Starting Line Depth", "-17.7 m", 0.58),
                ("Sequence Duration", "6.2 s", 0.72),
            ),
        ),
        SequenceShowcaseSpec(
            step_idx=2,
            code="low_block",
            title_en="Compact Low Block",
            title_vi="Khối phòng ngự thấp",
            match_id="hanoi_pvf_cand_2526",
            match_title="Hà Nội FC vs PVF-CAND",
            stadium="Sân Hàng Đẫy · Hà Nội",
            clip_path=packs / "pvfcand_3match_opposition_pack/clips/03_low_block/060_hanoi_pvf_cand_2526_P1_0884s_low_block_PVFCAND.mp4",
            clip_t0=3.8,
            clip_start_state_s=881.5,
            seq_start_s=884.5,
            seq_end_s=935.67,
            period=1,
            period_start_s=145.0,
            focus_team="away",
            home_code="HANOI",
            away_code="PVFCAND",
            rule_text="Rule: Out-of-possession block anchored <= -22.0 m with vertical compactness <= 14.5 m sustained across open-play circulation.",
            insight_text="PVF-CAND organizes a disciplined two-bank 4-4-2 low block (-33.8 m line, 10.7 m compactness) for 51.2 seconds against Hà Nội circulation.",
            badge_text="51.2S SUSTAINED BLOCK · HANG DAY",
            summary_stats=(
                ("Block Line Depth", "-33.8 m", 0.82),
                ("Vertical Compactness", "10.7 m", 0.85),
                ("Opponents in Final 3rd", "6 attackers", 0.75),
            ),
        ),
        SequenceShowcaseSpec(
            step_idx=3,
            code="midfield_loss",
            title_en="Midfield Loss & Counter-Press",
            title_vi="Mất bóng giữa sân & Phản ứng 3s",
            match_id="hanoi_pvf_cand_2526",
            match_title="Hà Nội FC vs PVF-CAND",
            stadium="Sân Hàng Đẫy · Hà Nội",
            clip_path=packs / "pvfcand_3match_opposition_pack/clips/04_midfield_loss/137_hanoi_pvf_cand_2526_P1_0427s_midfield_loss_PVFCAND.mp4",
            clip_t0=2.5,
            clip_start_state_s=424.17,
            seq_start_s=427.17,
            seq_end_s=433.17,
            period=1,
            period_start_s=145.0,
            focus_team="away",
            home_code="HANOI",
            away_code="PVFCAND",
            rule_text="Rule: Possession turnover inside middle third (|x| <= 17.5 m) classified over the next 3.0 s into Counter-Press, Hold, or Retreat.",
            insight_text="Immediate counter-press after center-circle turnover: 5 players swarm the ball zone with 2nd nearest defender closing to 5.4 m.",
            badge_text="3.0S TRANSITION CLASSIFIER",
            summary_stats=(
                ("3s Transition Reaction", "COUNTER-PRESS", 0.90),
                ("2nd Defender to Ball", "5.4 m", 0.78),
                ("Players Within 10m", "5 players", 0.74),
            ),
        ),
        SequenceShowcaseSpec(
            step_idx=4,
            code="flank_overload",
            title_en="Flank Overload",
            title_vi="Quá tải hành lang biên",
            match_id="pvf_cand_cahn_2526",
            match_title="PVF-CAND vs Công An Hà Nội",
            stadium="Sân PVF · Hưng Yên (Floodlights)",
            clip_path=packs / "pvfcand_3match_opposition_pack/clips/05_flank_overload/192_pvf_cand_cahn_2526_P1_1884s_flank_overload_PVFCAND.mp4",
            clip_t0=2.5,
            clip_start_state_s=1881.5,
            seq_start_s=1884.5,
            seq_end_s=1889.0,
            period=1,
            period_start_s=125.0,
            focus_team="home",
            home_code="PVFCAND",
            away_code="CAHN",
            flank="right",
            rule_text="Rule: Attacking numerical superiority (>= 4 attackers and >= +1 overload) concentrated in wide channel (|y| > 18.0 m) in opponent half.",
            insight_text="Against Công An Hà Nội under floodlights, PVF-CAND isolates a 5v3 right-channel overload (+7.4 m line) to drive into the final third.",
            badge_text="PVF STADIUM · FLOODLIGHT MATCH",
            summary_stats=(
                ("Right Channel Ratio", "5 v 3 (+2)", 0.88),
                ("Attacking Support Line", "+7.4 m", 0.68),
                ("Players Within 10m", "5 players", 0.74),
            ),
        ),
    ]


def draw_wordmark(draw: ImageDraw.ImageDraw, x: int, y: int, fonts: FontBank, size: int = 30) -> int:
    f = fonts.display(size, weight=750)
    draw.text((x, y), "FOOTLYTICS", font=f, fill=C_PAPER)
    w = draw.textbbox((x, y), "FOOTLYTICS", font=f)[2] - x
    draw.text((x + w + 2, y), ".", font=f, fill=C_SIGNAL)
    return w + 16


def render_supersampled_tactical_radar(
    frame_rows: pd.DataFrame | None,
    trail_rows: pd.DataFrame | None,
    *,
    out_w: int,
    out_h: int,
    home_code: str,
    away_code: str,
    pattern_code: str,
    focus_team: str,
    flank: str,
    fonts: FontBank,
) -> np.ndarray:
    """Render a 2x-supersampled 2D tactical pitch radar with pattern-specific geometry."""
    scale = 2
    rw, rh = out_w * scale, out_h * scale
    canvas = np.full((rh, rw, 3), (18, 48, 34), dtype=np.uint8)  # BGR deep turf

    top_bar = 26 * scale
    pad_x, pad_y = 14 * scale, 12 * scale
    pw = rw - 2 * pad_x
    ph = rh - top_bar - 2 * pad_y
    pitch_l, pitch_w = 105.0, 68.0
    hl, hw = pitch_l / 2.0, pitch_w / 2.0

    def to_px(xm: float, ym: float) -> tuple[int, int]:
        u = round(pad_x + (xm + hl) / pitch_l * pw)
        v = round(top_bar + pad_y + (ym + hw) / pitch_w * ph)
        return int(np.clip(u, pad_x, rw - pad_x)), int(np.clip(v, top_bar + pad_y, rh - pad_y))

    # Alternating subtle mowing stripes
    n_stripes = 10
    for i in range(n_stripes):
        if i % 2 == 0:
            sx0 = round(pad_x + i * pw / n_stripes)
            sx1 = round(pad_x + (i + 1) * pw / n_stripes)
            cv2.rectangle(canvas, (sx0, top_bar + pad_y), (sx1, top_bar + pad_y + ph), (22, 55, 39), -1)

    # Determine focus team players and attack direction in pitch coords
    focus_df = pd.DataFrame()
    ball_xy: tuple[float, float] | None = None
    if frame_rows is not None and not frame_rows.empty:
        valid = frame_rows[np.isfinite(frame_rows["x"]) & np.isfinite(frame_rows["y"])]
        b_rows = valid[valid["role"] == "ball"]
        if not b_rows.empty:
            ball_xy = (float(b_rows["x"].iloc[0]), float(b_rows["y"].iloc[0]))
        pl = valid[valid["role"].isin(["player", "goalkeeper"])]
        focus_df = pl[pl["team"] == focus_team]

    # Tactical geometry overlays BEFORE pitch lines & dots
    overlay = canvas.copy()
    if pattern_code == "flank_overload":
        # Highlight wide channel |y| > 18m
        y_band = (18.0, 34.0) if flank != "left" else (-34.0, -18.0)
        # Pick side where more focus players sit if ambiguous
        if not focus_df.empty:
            n_pos = int((focus_df["y"] > 15.0).sum())
            n_neg = int((focus_df["y"] < -15.0).sum())
            y_band = (18.0, 34.0) if n_pos >= n_neg else (-34.0, -18.0)
        p1 = to_px(-hl, y_band[0])
        p2 = to_px(hl, y_band[1])
        cv2.rectangle(overlay, (min(p1[0], p2[0]), min(p1[1], p2[1])), (max(p1[0], p2[0]), max(p1[1], p2[1])), rgb_to_bgr(C_SIGNAL), -1)
        cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0.0, dst=canvas)
        cv2.rectangle(canvas, (min(p1[0], p2[0]), min(p1[1], p2[1])), (max(p1[0], p2[0]), max(p1[1], p2[1])), rgb_to_bgr(C_SIGNAL), 2, cv2.LINE_AA)

    elif pattern_code == "low_block" and len(focus_df) >= 4:
        pts_m = focus_df[["x", "y"]].to_numpy(dtype=np.float32)
        # Drop goalkeeper outlier if far behind, keep compact outfield hull
        pts_px = np.array([to_px(float(x), float(y)) for x, y in pts_m], dtype=np.int32)
        hull = cv2.convexHull(pts_px)
        cv2.fillConvexPoly(overlay, hull, rgb_to_bgr(C_SIGNAL))
        cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0.0, dst=canvas)
        cv2.polylines(canvas, [hull], True, rgb_to_bgr(C_SIGNAL), 2, cv2.LINE_AA)

    elif pattern_code == "build_up" and not focus_df.empty:
        mean_x = float(focus_df["x"].mean())
        own_third = (-hl, -17.5) if mean_x < 0 else (17.5, hl)
        p1, p2 = to_px(own_third[0], -hw), to_px(own_third[1], hw)
        cv2.rectangle(overlay, (min(p1[0], p2[0]), min(p1[1], p2[1])), (max(p1[0], p2[0]), max(p1[1], p2[1])), rgb_to_bgr(C_CYAN), -1)
        cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0.0, dst=canvas)

    elif pattern_code == "midfield_loss":
        p1, p2 = to_px(-17.5, -hw), to_px(17.5, hw)
        cv2.rectangle(overlay, (p1[0], p1[1]), (p2[0], p2[1]), rgb_to_bgr(C_AMBER), -1)
        cv2.addWeighted(overlay, 0.14, canvas, 0.86, 0.0, dst=canvas)

    # Pitch markings
    line_col = (182, 212, 190)
    tl, br = to_px(-hl, -hw), to_px(hl, hw)
    cv2.rectangle(canvas, tl, br, line_col, 2, cv2.LINE_AA)
    cv2.line(canvas, to_px(0.0, -hw), to_px(0.0, hw), line_col, 2, cv2.LINE_AA)
    cx, cy = to_px(0.0, 0.0)
    cv2.circle(canvas, (cx, cy), max(6, round(9.15 / pitch_l * pw)), line_col, 2, cv2.LINE_AA)
    for sign in (-1.0, 1.0):
        pa1 = to_px(sign * hl, -20.16)
        pa2 = to_px(sign * (hl - 16.5), 20.16)
        cv2.rectangle(canvas, (min(pa1[0], pa2[0]), min(pa1[1], pa2[1])), (max(pa1[0], pa2[0]), max(pa1[1], pa2[1])), line_col, 2, cv2.LINE_AA)

    # High-press 10m pressing ring + defending line laser
    if pattern_code in ("high_press", "midfield_loss") and ball_xy is not None:
        bu, bv = to_px(ball_xy[0], ball_xy[1])
        r10 = round(10.0 / pitch_l * pw)
        ring_over = canvas.copy()
        cv2.circle(ring_over, (bu, bv), r10, rgb_to_bgr(C_SIGNAL), -1, cv2.LINE_AA)
        cv2.addWeighted(ring_over, 0.18, canvas, 0.82, 0.0, dst=canvas)
        cv2.circle(canvas, (bu, bv), r10, rgb_to_bgr(C_SIGNAL), 2, cv2.LINE_AA)

    if pattern_code in ("high_press", "low_block") and len(focus_df) >= 4:
        xs = np.sort(focus_df["x"].to_numpy(dtype=float))
        # Outfield deepest line (2nd deepest player, skipping keeper)
        mean_x = float(xs.mean())
        line_x = float(xs[1]) if mean_x < 0 else float(xs[-2])
        lu, _ = to_px(line_x, 0.0)
        cv2.line(canvas, (lu, top_bar + pad_y), (lu, top_bar + pad_y + ph), rgb_to_bgr(C_SIGNAL), 3, cv2.LINE_AA)

    # Motion trails
    home_bgr = rgb_to_bgr(C_CYAN)
    away_bgr = rgb_to_bgr(C_CORAL)
    if trail_rows is not None and not trail_rows.empty:
        valid_t = trail_rows[np.isfinite(trail_rows["x"]) & np.isfinite(trail_rows["y"])]
        for tid, tg in valid_t.groupby("track_id", observed=True):
            if int(tid) < 0 or len(tg) < 2:
                continue
            role = str(tg["role"].iloc[-1])
            tm = str(tg["team"].iloc[-1])
            if role == "ball":
                col = rgb_to_bgr(C_SIGNAL)
                thick = 3
            else:
                col = home_bgr if tm == "home" else (away_bgr if tm == "away" else (180, 180, 180))
                thick = 2
            pts = np.array([to_px(float(x), float(y)) for x, y in zip(tg["x"], tg["y"])], np.int32)
            cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, col, thick, cv2.LINE_AA)

    # Player & ball markers
    if frame_rows is not None and not frame_rows.empty:
        valid = frame_rows[np.isfinite(frame_rows["x"]) & np.isfinite(frame_rows["y"])]
        for r in valid.itertuples():
            u, v = to_px(float(r.x), float(r.y))
            if r.role == "ball":
                cv2.circle(canvas, (u, v), 11, rgb_to_bgr(C_SIGNAL), 2, cv2.LINE_AA)
                cv2.circle(canvas, (u, v), 6, (255, 255, 255), -1, cv2.LINE_AA)
            elif r.role in ("player", "goalkeeper"):
                col = home_bgr if r.team == "home" else (away_bgr if r.team == "away" else (185, 185, 185))
                cv2.circle(canvas, (u, v), 8, col, -1, cv2.LINE_AA)
                cv2.circle(canvas, (u, v), 8, (250, 250, 250), 2, cv2.LINE_AA)

    # Downsample 2x -> 1x with INTER_AREA for crisp anti-aliasing
    small = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_AREA)

    # Crisp header bar & border via Pillow
    pil_im = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil_im)
    dr.rectangle((0, 0, out_w - 1, 25), fill=C_BG_DEEP)
    f_leg = fonts.display(12, weight=650)
    cx_cur = 10
    dr.ellipse((cx_cur, 8, cx_cur + 8, 16), fill=C_CYAN)
    dr.text((cx_cur + 12, 5), home_code, font=f_leg, fill=C_PAPER)
    cx_cur += 16 + (dr.textbbox((0, 0), home_code, font=f_leg)[2]) + 14
    dr.ellipse((cx_cur, 8, cx_cur + 8, 16), fill=C_CORAL)
    dr.text((cx_cur + 12, 5), away_code, font=f_leg, fill=C_PAPER)
    cx_cur += 16 + (dr.textbbox((0, 0), away_code, font=f_leg)[2]) + 14
    dr.ellipse((cx_cur, 8, cx_cur + 8, 16), fill=C_SIGNAL)
    dr.text((cx_cur + 12, 5), "BALL · 105x68M", font=f_leg, fill=C_SIGNAL)
    dr.rectangle((0, 0, out_w - 1, out_h - 1), outline=C_RULE, width=2)
    return cv2.cvtColor(np.asarray(pil_im), cv2.COLOR_RGB2BGR)


def estimate_smoothed_homographies(tdf: pd.DataFrame, start_f: int, end_f: int) -> dict[int, np.ndarray]:
    """Compute temporally smoothed pitch->image homographies from tracked player footpoints."""
    out: dict[int, np.ndarray] = {}
    last_h: np.ndarray | None = None
    sub = tdf[(tdf["frame_idx"] >= start_f - 10) & (tdf["frame_idx"] <= end_f + 5)]
    by_f = {int(k): g for k, g in sub.groupby("frame_idx", observed=True)}
    for f in range(start_f - 10, end_f + 5):
        g = by_f.get(f)
        h_mat: np.ndarray | None = None
        if g is not None:
            pl = g[
                g["role"].isin(["player", "goalkeeper"])
                & np.isfinite(g["x"])
                & np.isfinite(g["y"])
                & np.isfinite(g["bbox_x"])
                & np.isfinite(g["bbox_y"])
                & np.isfinite(g["bbox_w"])
                & np.isfinite(g["bbox_h"])
            ]
            if len(pl) >= 6:
                src = pl[["x", "y"]].to_numpy(dtype=np.float64)
                dst_u = (pl["bbox_x"] + 0.5 * pl["bbox_w"]).to_numpy(dtype=np.float64)
                dst_v = (pl["bbox_y"] + pl["bbox_h"]).to_numpy(dtype=np.float64)
                dst = np.column_stack([dst_u, dst_v])
                cand, _ = cv2.findHomography(src, dst, cv2.RANSAC, 12.0)
                if cand is not None and np.isfinite(cand).all() and abs(cand[2, 2]) > 1e-6:
                    cand = cand / cand[2, 2]
                    h_mat = cand
        if h_mat is not None:
            if last_h is None:
                last_h = h_mat
            else:
                last_h = 0.82 * last_h + 0.18 * h_mat
                last_h = last_h / last_h[2, 2]
        if last_h is not None and f >= start_f:
            out[f] = last_h.copy()
    return out


def draw_projected_pitch_overlay(canvas_bgr: np.ndarray, h_mat: np.ndarray, reveal_frac: float) -> None:
    """Project 105x68m pitch markings & tactical 10.5m grid onto 1080p grass via homography."""
    if reveal_frac <= 0.01:
        return
    overlay = canvas_bgr.copy()
    max_x_reveal = -52.5 + 105.0 * ease_out_cubic(reveal_frac)

    def proj_poly(pts_xy: list[tuple[float, float]]) -> np.ndarray | None:
        arr = np.array(pts_xy, dtype=np.float64).reshape(-1, 1, 2)
        im_pts = cv2.perspectiveTransform(arr, h_mat).reshape(-1, 2)
        if not np.isfinite(im_pts).all():
            return None
        # Filter wildly offscreen explosions
        if np.any(np.abs(im_pts) > 8000):
            return None
        return np.round(im_pts).astype(np.int32)

    # Subtle 10.5m vertical zone lines
    for xm in np.linspace(-42.0, 42.0, 9):
        if xm > max_x_reveal:
            continue
        ys = np.linspace(-34.0, 34.0, 25)
        poly = proj_poly([(float(xm), float(y)) for y in ys])
        if poly is not None:
            cv2.polylines(overlay, [poly.reshape(-1, 1, 2)], False, rgb_to_bgr(C_CYAN), 1, cv2.LINE_AA)

    # Touchlines & halfway line
    for y_edge in (-34.0, 34.0):
        xs = [x for x in np.linspace(-52.5, 52.5, 40) if x <= max_x_reveal]
        if len(xs) >= 2:
            poly = proj_poly([(float(x), y_edge) for x in xs])
            if poly is not None:
                cv2.polylines(overlay, [poly.reshape(-1, 1, 2)], False, rgb_to_bgr(C_SIGNAL), 2, cv2.LINE_AA)

    if max_x_reveal >= 0.0:
        poly_mid = proj_poly([(0.0, float(y)) for y in np.linspace(-34.0, 34.0, 30)])
        if poly_mid is not None:
            cv2.polylines(overlay, [poly_mid.reshape(-1, 1, 2)], False, rgb_to_bgr(C_SIGNAL), 2, cv2.LINE_AA)
        # Center circle
        ang = np.linspace(0.0, 2.0 * np.pi, 64)
        cc_pts = [(9.15 * float(np.cos(a)), 9.15 * float(np.sin(a))) for a in ang if 9.15 * np.cos(a) <= max_x_reveal]
        if len(cc_pts) >= 4:
            poly_cc = proj_poly(cc_pts)
            if poly_cc is not None:
                cv2.polylines(overlay, [poly_cc.reshape(-1, 1, 2)], False, rgb_to_bgr(C_SIGNAL), 2, cv2.LINE_AA)

    # Left penalty box & penalty arc
    for sign in (-1.0, 1.0):
        x_goal = sign * 52.5
        x_box = sign * 36.0
        if min(x_goal, x_box) <= max_x_reveal:
            box_pts = [(x_goal, -20.16), (x_box, -20.16), (x_box, 20.16), (x_goal, 20.16)]
            poly_b = proj_poly(box_pts)
            if poly_b is not None:
                cv2.polylines(overlay, [poly_b.reshape(-1, 1, 2)], False, rgb_to_bgr(C_SIGNAL), 2, cv2.LINE_AA)

    alpha = 0.42 * min(1.0, reveal_frac * 1.4)
    cv2.addWeighted(overlay, alpha, canvas_bgr, 1.0 - alpha, 0.0, dst=canvas_bgr)


def render_act1_frames(fonts: FontBank) -> list[np.ndarray]:
    """Render Act I (13.0s = 390 frames): Hero Title Card + Live 1080p Calibration/Tracking."""
    frames: list[np.ndarray] = []
    raw_video = ROOT / "data" / "clips" / "vn_cahcm_hanoi_40m_10min.mp4"
    tdf = pd.read_parquet(ROOT / "data" / "out" / "vleague_10min_rtdetr_vl" / "tracks.parquet")

    start_raw_f = 9190
    end_raw_f = 9525
    homos = estimate_smoothed_homographies(tdf, start_raw_f, end_raw_f)
    sub_tdf = tdf[(tdf["frame_idx"] >= start_raw_f - 30) & (tdf["frame_idx"] <= end_raw_f + 5)]
    by_frame = {int(k): g for k, g in sub_tdf.groupby("frame_idx", observed=True)}

    cap = cv2.VideoCapture(str(raw_video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_raw_f)
    raw_cache: dict[int, np.ndarray] = {}
    for rf in range(start_raw_f, end_raw_f + 1):
        ok, fr = cap.read()
        if not ok:
            break
        if fr.shape[:2] != (H, W):
            fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
        raw_cache[rf] = fr
    cap.release()

    # Scene 1A: 0:00 - 0:04.5 (135 frames) — Hero Opening Card over atmospheric live broadcast
    n_1a = 135
    for i in range(n_1a):
        rf = start_raw_f + int(i * 25 / 30)
        bg = raw_cache.get(rf, next(iter(raw_cache.values()))).copy()
        # Deep forest atmospheric grade
        tint = np.full_like(bg, rgb_to_bgr(C_BG_DEEP))
        cv2.addWeighted(bg, 0.22, tint, 0.78, 0.0, dst=bg)

        t_in = ease_out_cubic(min(1.0, i / 20.0))
        y_shift = round((1.0 - t_in) * 24)

        pil = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(pil)

        # Top rule & wordmark
        draw_wordmark(dr, 88, 66 + y_shift, fonts, size=34)
        dr.text(
            (320, 74 + y_shift),
            "FOUNDER PRODUCT SHOWCASE  ·  LPBANK V.LEAGUE 1 2025/26",
            font=fonts.mono(15),
            fill=C_SIGNAL,
        )
        dr.line((88, 122 + y_shift, W - 88, 122 + y_shift), fill=C_RULE, width=2)

        # Hero headline
        f_h1 = fonts.display(74, weight=650)
        dr.text((88, 198 + y_shift), "Broadcast Video to", font=f_h1, fill=C_PAPER)
        dr.text((88, 284 + y_shift), "Tactical Intelligence —", font=f_h1, fill=C_PAPER)
        dr.text((88, 370 + y_shift), "Delivered Overnight.", font=f_h1, fill=C_SIGNAL)

        # Vietnamese + English thesis
        f_sub = fonts.display(25, weight=450)
        sub_lines = [
            "Từ một luồng hình truyền hình V.League duy nhất đến dữ liệu tọa độ 105×68m,",
            "5 mẫu hình chiến thuật tự động và hồ sơ đối thủ sẵn sàng cho Ban Huấn luyện.",
        ]
        for li, ln in enumerate(sub_lines):
            dr.text((88, 496 + li * 38 + y_shift), ln, font=f_sub, fill=C_MUTED)

        # Bottom 3 ruled proof columns
        t_cards = ease_out_cubic(max(0.0, min(1.0, (i - 8) / 22.0)))
        if t_cards > 0.05:
            y_base = 690 + round((1.0 - t_cards) * 18)
            dr.line((88, y_base, W - 88, y_base), fill=C_RULE, width=2)
            metrics = [
                ("549,300 FRAMES", "3 full 90-min V.League 2025/26 matches processed end-to-end at 30 Hz"),
                ("96.3% PRECISION", "100.0% on decided sequences (52/52) across 55 visually audited clips"),
                ("$3.66 / MATCH", "4.58 GPUh compute (~$2.58 on Apache-2.0 RT-DETRv2) · <8h overnight SLA"),
            ]
            col_w = (W - 176) // 3
            for ci, (big, desc) in enumerate(metrics):
                cx0 = 88 + ci * col_w
                if ci > 0:
                    dr.line((cx0 - 24, y_base + 24, cx0 - 24, y_base + 185), fill=C_RULE, width=1)
                dr.text((cx0, y_base + 28), f"0{ci + 1}", font=fonts.mono(14), fill=C_SIGNAL)
                dr.text((cx0, y_base + 56), big, font=fonts.display(44, weight=700), fill=C_PAPER)
                for dli, dln in enumerate(wrap_text(dr, desc, fonts.display(18, weight=450), col_w - 48)):
                    dr.text((cx0, y_base + 118 + dli * 26), dln, font=fonts.display(18, weight=450), fill=C_MUTED)

        frames.append(cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR))

    # Scene 1B: 0:04.5 - 0:13.0 (255 frames = 8.5s) — Live 1080p Perception & Metric Projection
    n_1b = 255
    base_rf = start_raw_f + int(n_1a * 25 / 30)
    for i in range(n_1b):
        sec = i / FPS
        rf = min(end_raw_f, base_rf + int(i * 25 / 30))
        canvas = raw_cache.get(rf, raw_cache[max(raw_cache)]).copy()
        f_rows = by_frame.get(rf)

        # Phase 1 (0.0s+): Homography pitch calibration lines sweep across broadcast grass
        h_mat = homos.get(rf)
        if h_mat is not None:
            reveal = min(1.0, sec / 1.8)
            draw_projected_pitch_overlay(canvas, h_mat, reveal)

        # Phase 2 (1.8s+): Player foot ellipses + team classification + ball halo
        if sec >= 1.6 and f_rows is not None and not f_rows.empty:
            track_alpha = ease_out_cubic(min(1.0, (sec - 1.6) / 0.7))
            ov = canvas.copy()
            for r in f_rows.itertuples():
                has_bb = (
                    np.isfinite(r.bbox_x)
                    and np.isfinite(r.bbox_y)
                    and np.isfinite(r.bbox_w)
                    and np.isfinite(r.bbox_h)
                )
                if r.role in ("player", "goalkeeper") and has_bb:
                    cx = round(float(r.bbox_x + 0.5 * r.bbox_w))
                    cy = round(float(r.bbox_y + r.bbox_h))
                    rx = max(14, round(float(r.bbox_w) * 0.46))
                    ry = max(6, round(rx * 0.38))
                    col = rgb_to_bgr(C_CORAL if r.team == "home" else C_CYAN)
                    cv2.ellipse(ov, (cx, cy), (rx, ry), 0, -15, 205, col, 3, cv2.LINE_AA)
                elif r.role == "referee" and has_bb and float(r.bbox_y) < 960:
                    cx = round(float(r.bbox_x + 0.5 * r.bbox_w))
                    cy = round(float(r.bbox_y + r.bbox_h))
                    cv2.ellipse(ov, (cx, cy), (14, 6), 0, 0, 360, rgb_to_bgr(C_AMBER), 2, cv2.LINE_AA)
                elif r.role == "ball":
                    bx_by: tuple[int, int] | None = None
                    if has_bb:
                        bx_by = (round(float(r.bbox_x + 0.5 * r.bbox_w)), round(float(r.bbox_y + 0.5 * r.bbox_h)))
                    elif h_mat is not None and np.isfinite(r.x) and np.isfinite(r.y):
                        pt = cv2.perspectiveTransform(
                            np.array([[[float(r.x), float(r.y)]]], dtype=np.float64), h_mat
                        ).reshape(2)
                        if np.isfinite(pt).all() and 0 <= pt[0] < W and 0 <= pt[1] < H:
                            bx_by = (round(float(pt[0])), round(float(pt[1])))
                    if bx_by is not None:
                        bx, by = bx_by
                        cv2.circle(ov, (bx, by), 22, rgb_to_bgr(C_SIGNAL), 3, cv2.LINE_AA)
                        cv2.line(ov, (bx - 30, by), (bx - 14, by), rgb_to_bgr(C_SIGNAL), 2, cv2.LINE_AA)
                        cv2.line(ov, (bx + 14, by), (bx + 30, by), rgb_to_bgr(C_SIGNAL), 2, cv2.LINE_AA)
            cv2.addWeighted(ov, track_alpha, canvas, 1.0 - track_alpha, 0.0, dst=canvas)

        # Phase 3 (4.2s+): Live 2D Metric Match State dock slides in bottom-right
        if sec >= 4.0:
            slide = ease_out_cubic(min(1.0, (sec - 4.0) / 0.65))
            rw, rh = 540, 352
            rx0 = round(W - 56 - rw + (1.0 - slide) * (rw + 80))
            ry0 = H - 52 - rh
            trail_sub = sub_tdf[(sub_tdf["frame_idx"] >= rf - 38) & (sub_tdf["frame_idx"] <= rf)]
            radar_bgr = render_supersampled_tactical_radar(
                f_rows,
                trail_sub,
                out_w=rw,
                out_h=rh,
                home_code="CA TP.HCM",
                away_code="HA NOI FC",
                pattern_code="",
                focus_team="away",
                flank="",
                fonts=fonts,
            )
            x_clip_end = min(W, rx0 + rw)
            if rx0 < W and x_clip_end > max(0, rx0):
                sx0 = max(0, -rx0)
                dx0 = max(0, rx0)
                w_vis = x_clip_end - dx0
                canvas[ry0 : ry0 + rh, dx0 : dx0 + w_vis] = radar_bgr[:, sx0 : sx0 + w_vis]

        # Clean bottom-left stage callout card (never covers FPT scorebug at top-left!)
        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(pil, "RGBA")

        if sec < 2.0:
            step_tag = "STAGE 01 / 03  ·  HOMOGRAPHY PITCH CALIBRATION"
            step_title = "Single Broadcast Camera -> 105×68m Metric Pitch"
            step_meta = "V.League < 1.0m accuracy: 87.5% (dev) / 76.2% (frozen) · Thống Nhất Stadium 1080p"
        elif sec < 4.3:
            step_tag = "STAGE 02 / 03  ·  APACHE-2.0 PERCEPTION & IDENTITY"
            step_title = "RT-DETRv2 People + Ball Detection & Team Separation"
            step_meta = "GS-HOTA 60.49 (Apache-2.0) / 63.10 (YOLOv8x6) · Team ID 92.8% · Ball Recall 70.5%"
        else:
            step_tag = "STAGE 03 / 03  ·  CONTINUOUS 30 HZ MATCH STATE"
            step_title = "Synchronized 2D Trajectories, Speeds & Possession"
            step_meta = "Every player & ball projected to meters (x, y, vx, vy) -> ready for tactical queries"

        card_x0, card_y0, card_x1, card_y1 = 56, H - 196, 1140, H - 52
        dr.rounded_rectangle((card_x0, card_y0, card_x1, card_y1), radius=10, fill=(11, 24, 19, 228), outline=C_RULE + (255,), width=2)
        dr.rectangle((card_x0, card_y0, card_x0 + 6, card_y1), fill=C_SIGNAL + (255,))
        dr.text((card_x0 + 26, card_y0 + 18), step_tag, font=fonts.mono(14), fill=C_SIGNAL)
        dr.text((card_x0 + 26, card_y0 + 46), step_title, font=fonts.display(30, weight=650), fill=C_PAPER)
        dr.text((card_x0 + 26, card_y0 + 96), step_meta, font=fonts.display(17, weight=450), fill=C_MUTED)

        frames.append(cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR))

    # Cross-dissolve 12 frames between Scene 1A and Scene 1B
    fade_n = 12
    blended = frames[: n_1a - fade_n]
    for k in range(fade_n):
        a = ease_in_out_cubic((k + 1) / (fade_n + 1))
        f1 = frames[n_1a - fade_n + k]
        f2 = frames[n_1a + k]
        blended.append(cv2.addWeighted(f2, a, f1, 1.0 - a, 0.0))
    blended.extend(frames[n_1a + fade_n :])
    return blended


def build_act2_static_chrome(
    spec: SequenceShowcaseSpec,
    all_specs: list[SequenceShowcaseSpec],
    match_seqs: pd.DataFrame,
    fonts: FontBank,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Pre-render the 1920x1080 static chrome + full-match timeline for one Act II pattern."""
    pil = Image.new("RGB", (W, H), C_BG_DEEP)
    dr = ImageDraw.Draw(pil)

    # Top Header Bar (y=0..82)
    dr.rectangle((0, 0, W, 82), fill=C_CARD_DARK)
    dr.line((0, 82, W, 82), fill=C_RULE, width=2)
    w_mark = draw_wordmark(dr, 56, 24, fonts, size=26)
    dr.text((56 + w_mark + 12, 31), "V1 TACTICAL SEQUENCE ENGINE", font=fonts.mono(13), fill=C_MUTED)

    # 5-Step Pattern Stepper on Top Right
    px_cursor = W - 56
    step_labels = [
        (4, "05 FLANK OVERLOAD"),
        (3, "04 MIDFIELD LOSS"),
        (2, "03 LOW BLOCK"),
        (1, "02 BUILD-UP"),
        (0, "01 HIGH PRESS"),
    ]
    f_step = fonts.mono(12)
    for idx, txt in step_labels:
        tw = dr.textbbox((0, 0), txt, font=f_step)[2]
        pw_pill = tw + 24
        x0 = px_cursor - pw_pill
        active = idx == spec.step_idx
        fill = C_SIGNAL if active else C_BG_STAGE
        fg = C_INK if active else C_MUTED
        outline = C_SIGNAL if active else C_RULE
        dr.rounded_rectangle((x0, 23, px_cursor, 59), radius=6, fill=fill, outline=outline, width=1)
        dr.text((x0 + 12, 33), txt, font=f_step, fill=fg)
        px_cursor = x0 - 10

    # Video Stage Bezel (left: x=56..1304, y=104..773 -> 1248 x 669)
    vx0, vy0, vw, vh = 56, 104, 1248, 669
    dr.rectangle((vx0 - 2, vy0 - 2, vx0 + vw + 1, vy0 + vh + 1), outline=C_RULE, width=2)

    # Right Intelligence Panel (x=1330..1864, y=104..773 -> 534 x 669)
    rx0, ry0, rx1, ry1 = 1330, 104, 1864, 773
    dr.rounded_rectangle((rx0, ry0, rx1, ry1), radius=10, fill=C_CARD_DARK, outline=C_RULE, width=2)

    dr.text((rx0 + 28, ry0 + 24), f"ENGINE 0{spec.step_idx + 1} / 05  ·  {spec.badge_text}", font=fonts.mono(12), fill=C_SIGNAL)
    title_sz = 36
    while title_sz > 26 and (dr.textbbox((0, 0), spec.title_en, font=fonts.display(title_sz, weight=700))[2]) > (rx1 - rx0 - 56):
        title_sz -= 2
    dr.text((rx0 + 28, ry0 + 52 + (36 - title_sz) // 2), spec.title_en, font=fonts.display(title_sz, weight=700), fill=C_PAPER)
    dr.text((rx0 + 28, ry0 + 98), spec.title_vi, font=fonts.display(21, weight=550), fill=C_SIGNAL)
    dr.text((rx0 + 28, ry0 + 132), f"{spec.match_title}  ·  {spec.stadium}", font=fonts.display(16, weight=450), fill=C_MUTED)

    dr.line((rx0 + 28, ry0 + 170, rx1 - 28, ry0 + 170), fill=C_RULE, width=1)

    # Static labels for 3 telemetry bars (y = ry0 + 188 .. ry0 + 408)
    for bi, (m_label, m_val, _) in enumerate(spec.summary_stats):
        by = ry0 + 190 + bi * 74
        dr.text((rx0 + 28, by), m_label.upper(), font=fonts.mono(12), fill=C_MUTED)
        vw_txt = dr.textbbox((0, 0), m_val, font=fonts.mono(17))[2]
        dr.text((rx1 - 28 - vw_txt, by - 3), m_val, font=fonts.mono(17), fill=C_PAPER)
        dr.rounded_rectangle((rx0 + 28, by + 28, rx1 - 28, by + 40), radius=5, fill=C_BG_DEEP, outline=C_RULE, width=1)

    dr.line((rx0 + 28, ry0 + 422, rx1 - 28, ry0 + 422), fill=C_RULE, width=1)

    # Rule + Tactical Insight wrapped cleanly inside 478px width
    f_body = fonts.display(15, weight=450)
    f_insight = fonts.display(16, weight=550)
    max_tw = (rx1 - rx0) - 56
    cy_txt = ry0 + 438
    for ln in wrap_text(dr, spec.rule_text, f_body, max_tw)[:3]:
        dr.text((rx0 + 28, cy_txt), ln, font=f_body, fill=C_MUTED)
        cy_txt += 22
    cy_txt += 12
    dr.text((rx0 + 28, cy_txt), "SCOUTING TAKEAWAY", font=fonts.mono(11), fill=C_SIGNAL)
    cy_txt += 20
    for ln in wrap_text(dr, spec.insight_text, f_insight, max_tw)[:4]:
        dr.text((rx0 + 28, cy_txt), ln, font=f_insight, fill=C_PAPER)
        cy_txt += 23

    # Bottom Full-Match 90-Minute Timeline Card (x=56..1864, y=796..1036 -> 1808 x 240)
    tx0, ty0, tx1, ty1 = 56, 796, 1864, 1036
    dr.rounded_rectangle((tx0, ty0, tx1, ty1), radius=10, fill=C_CARD_DARK, outline=C_RULE, width=2)

    n_total_match = len(match_seqs)
    n_same_label = int((match_seqs["label"] == spec.code).sum())
    dr.text(
        (tx0 + 28, ty0 + 20),
        f"FULL 90-MINUTE MATCH INDEX  ·  {spec.match_title.upper()}",
        font=fonts.mono(13),
        fill=C_SIGNAL,
    )
    stat_right = f"{n_total_match} total sequences indexed in match  ·  {n_same_label} '{spec.code}' detections"
    sr_w = dr.textbbox((0, 0), stat_right, font=fonts.mono(13))[2]
    dr.text((tx1 - 28 - sr_w, ty0 + 20), stat_right, font=fonts.mono(13), fill=C_MUTED)

    # Timeline track bar (x = tx0+90 .. tx1-44)
    bar_x0, bar_x1 = tx0 + 150, tx1 - 40
    bar_w = bar_x1 - bar_x0
    max_s = max(6000.0, float(match_seqs["end_s"].max()) + 60.0)

    lanes = [
        ("high_press", "HIGH PRESS"),
        ("build_up", "BUILD-UP"),
        ("low_block", "LOW BLOCK"),
        ("midfield_loss", "MIDFIELD LOSS"),
        ("flank_overload", "FLANK OVERLOAD"),
    ]
    for li, (l_code, l_name) in enumerate(lanes):
        ly = ty0 + 64 + li * 31
        is_cur = l_code == spec.code
        dr.text((tx0 + 28, ly - 5), l_name, font=fonts.mono(11), fill=C_SIGNAL if is_cur else C_MUTED)
        dr.line((bar_x0, ly + 3, bar_x1, ly + 3), fill=(36, 62, 50) if is_cur else (24, 42, 34), width=6 if is_cur else 3)

        l_rows = match_seqs[match_seqs["label"] == l_code]
        for r in l_rows.itertuples():
            sx = bar_x0 + round(float(r.start_s) / max_s * bar_w)
            ex = max(sx + 4, bar_x0 + round(float(r.end_s) / max_s * bar_w))
            col = C_CYAN if is_cur else (72, 104, 88)
            dr.rectangle((sx, ly - 2, ex, ly + 8), fill=col)

    # Highlight active sequence tick on its lane
    act_lane_y = ty0 + 64 + spec.step_idx * 31
    asx = bar_x0 + round(spec.seq_start_s / max_s * bar_w)
    aex = max(asx + 8, bar_x0 + round(spec.seq_end_s / max_s * bar_w))
    dr.rectangle((asx - 2, act_lane_y - 6, aex + 2, act_lane_y + 12), fill=C_SIGNAL)

    # Kickoff / HT / FT markers
    for label_m, sec_m in (("KICKOFF", spec.period_start_s), ("HALF-TIME", 2950.0), ("90:00+", max_s - 120.0)):
        mx = bar_x0 + round(max(0.0, min(1.0, sec_m / max_s)) * bar_w)
        dr.text((mx - 24, ty1 - 24), label_m, font=fonts.mono(10), fill=C_MUTED)

    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR), (bar_x0, bar_x1, int(max_s), act_lane_y)


def render_act2_frames(fonts: FontBank) -> list[np.ndarray]:
    """Render Act II (32.5s = 975 frames): All 5 V1 Tactical Engines across 3 matches."""
    specs = build_sequence_specs()
    frames_per_clip = 195  # 6.5s @ 30fps

    # Cache tracks & sequence tables per match
    tracks_cache: dict[str, dict[int, pd.DataFrame]] = {}
    tracks_df_cache: dict[str, pd.DataFrame] = {}
    seqs_cache: dict[str, pd.DataFrame] = {}
    for sp in specs:
        if sp.match_id not in seqs_cache:
            mdir = ROOT / "data" / "out" / sp.match_id
            seqs_cache[sp.match_id] = pd.read_csv(mdir / "sequences.csv")
            tdf = pd.read_parquet(
                mdir / "tracks.parquet",
                columns=["frame_idx", "timestamp", "track_id", "role", "team", "x", "y"],
            )
            tracks_df_cache[sp.match_id] = tdf
            tracks_cache[sp.match_id] = {}

    clip_segments: list[list[np.ndarray]] = []
    vx0, vy0, vw, vh = 56, 104, 1248, 669
    rx0, ry0, rx1 = 1330, 104, 1864

    for sp in specs:
        base_bgr, (bar_x0, bar_x1, max_s, act_lane_y) = build_act2_static_chrome(
            sp, specs, seqs_cache[sp.match_id], fonts
        )
        tdf = tracks_df_cache[sp.match_id]
        s_min = sp.clip_start_state_s + sp.clip_t0 - 3.0
        s_max = sp.clip_start_state_s + sp.clip_t0 + (frames_per_clip / FPS) + 1.0
        win_tdf = tdf[(tdf["timestamp"] >= s_min) & (tdf["timestamp"] <= s_max)]
        by_fidx = {int(k): g for k, g in win_tdf.groupby("frame_idx", observed=True)}

        cap = cv2.VideoCapture(str(sp.clip_path))
        clip_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start_cf = round(sp.clip_t0 * clip_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_cf)

        seg_frames: list[np.ndarray] = []
        last_vid_crop: np.ndarray | None = None
        for i in range(frames_per_clip):
            ok, vfr = cap.read()
            if ok and vfr is not None:
                # Crop out top 34px ASCII HUD banner -> 1280x686 -> resize cleanly to 1248x669
                crop = vfr[34:, :, :]
                last_vid_crop = cv2.resize(crop, (vw, vh), interpolation=cv2.INTER_AREA)
            elif last_vid_crop is None:
                last_vid_crop = np.zeros((vh, vw, 3), dtype=np.uint8)

            canvas = base_bgr.copy()
            vid_stage = last_vid_crop.copy()

            cur_state_s = sp.clip_start_state_s + sp.clip_t0 + (i / FPS)
            cur_fidx = round(cur_state_s * 30.0)
            f_rows = by_fidx.get(cur_fidx)
            if f_rows is None:
                f_rows = by_fidx.get(cur_fidx - 1)
            if f_rows is None:
                f_rows = by_fidx.get(cur_fidx + 1)
            trail_rows = win_tdf[(win_tdf["frame_idx"] >= cur_fidx - 45) & (win_tdf["frame_idx"] <= cur_fidx)]

            # Replace old 336x218 PiP slot cleanly with 2x-supersampled Tactical Geometry Radar
            # Old PiP after 34px top crop & 0.975 scale sits at [902..1231, 438..652]
            pip_x0, pip_y0, pip_w, pip_h = 894, 430, 342, 226
            radar_pip = render_supersampled_tactical_radar(
                f_rows,
                trail_rows,
                out_w=pip_w,
                out_h=pip_h,
                home_code=sp.home_code,
                away_code=sp.away_code,
                pattern_code=sp.code,
                focus_team=sp.focus_team,
                flank=sp.flank,
                fonts=fonts,
            )
            vid_stage[pip_y0 : pip_y0 + pip_h, pip_x0 : pip_x0 + pip_w] = radar_pip
            canvas[vy0 : vy0 + vh, vx0 : vx0 + vw] = vid_stage

            # Dynamic telemetry bars + live match clock pill + playhead marker
            bar_anim = ease_out_cubic(min(1.0, (i + 1) / 24.0))
            for bi, (_, _, fill_ratio) in enumerate(sp.summary_stats):
                by = ry0 + 190 + bi * 74
                bx_start = rx0 + 30
                bx_max = rx1 - 30
                fill_w = round((bx_max - bx_start) * fill_ratio * bar_anim)
                if fill_w > 4:
                    col_b = rgb_to_bgr(C_SIGNAL if bi == 0 else C_CYAN)
                    cv2.rectangle(canvas, (bx_start, by + 30), (bx_start + fill_w, by + 38), col_b, -1)

            # Live timeline playhead
            play_x = bar_x0 + round(min(1.0, cur_state_s / max_s) * (bar_x1 - bar_x0))
            cv2.circle(canvas, (play_x, act_lane_y + 3), 7, rgb_to_bgr(C_SIGNAL), -1, cv2.LINE_AA)
            cv2.circle(canvas, (play_x, act_lane_y + 3), 11, rgb_to_bgr(C_SIGNAL), 1, cv2.LINE_AA)

            # Bottom-left live match clock badge on video (avoids top-left FPT TV scorebug)
            in_seq = (sp.seq_start_s - 0.25) <= cur_state_s <= (sp.seq_end_s + 0.35)
            clk_txt = format_match_clock(cur_state_s - sp.period_start_s, sp.period)
            pil_ov = Image.fromarray(cv2.cvtColor(canvas[vy0 + vh - 58 : vy0 + vh - 14, vx0 + 16 : vx0 + 360], cv2.COLOR_BGR2RGB))
            dr_ov = ImageDraw.Draw(pil_ov, "RGBA")
            dr_ov.rounded_rectangle((0, 0, 343, 43), radius=7, fill=(11, 24, 19, 224), outline=C_RULE + (255,), width=1)
            dot_col = C_SIGNAL if in_seq else C_MUTED
            dr_ov.ellipse((14, 15, 26, 27), fill=dot_col + (255,))
            status_lbl = "LIVE SEQUENCE" if in_seq else "PRE/POST CONTEXT"
            dr_ov.text((36, 13), f"{status_lbl}  ·  {clk_txt}", font=fonts.mono(13), fill=C_PAPER)
            canvas[vy0 + vh - 58 : vy0 + vh - 14, vx0 + 16 : vx0 + 360] = cv2.cvtColor(np.asarray(pil_ov), cv2.COLOR_RGB2BGR)

            seg_frames.append(canvas)
        cap.release()
        clip_segments.append(seg_frames)

    # Join all 5 segments with crisp 9-frame (0.3s) cross-dissolves
    fade = 9
    merged: list[np.ndarray] = []
    for idx, seg in enumerate(clip_segments):
        if idx == 0:
            merged.extend(seg[:-fade])
        else:
            prev_tail = clip_segments[idx - 1][-fade:]
            for k in range(fade):
                a = ease_in_out_cubic((k + 1) / (fade + 1))
                merged.append(cv2.addWeighted(seg[k], a, prev_tail[k], 1.0 - a, 0.0))
            if idx < len(clip_segments) - 1:
                merged.extend(seg[fade:-fade])
            else:
                merged.extend(seg[fade:])
    return merged


def render_act3_frames(
    workspace_png: Path,
    dossier_png: Path,
    fonts: FontBank,
) -> list[np.ndarray]:
    """Render Act III (13.0s = 390 frames): Live Analyst Workspace + Scouting Pack Delivery."""
    ws_img = Image.open(workspace_png).convert("RGB")
    ds_img = Image.open(dossier_png).convert("RGB")

    # Scene 3A (195 frames = 6.5s): Interactive Web Tactical Workspace
    n_3a = 195
    frames_3a: list[np.ndarray] = []
    for i in range(n_3a):
        prog = i / max(1, n_3a - 1)
        zoom = 1.0 + 0.035 * ease_in_out_cubic(prog)
        pil = Image.new("RGB", (W, H), C_BG_DEEP)
        dr = ImageDraw.Draw(pil)

        # Header
        dr.rectangle((0, 0, W, 82), fill=C_CARD_DARK)
        dr.line((0, 82, W, 82), fill=C_RULE, width=2)
        w_m = draw_wordmark(dr, 56, 24, fonts, size=26)
        dr.text((56 + w_m + 12, 31), "ANALYST DELIVERY  ·  01 / 02  INTERACTIVE TACTICAL WORKSPACE", font=fonts.mono(13), fill=C_SIGNAL)

        # Left narrative column (x=56..570)
        dr.text((56, 132), "footlytics workspace serve", font=fonts.mono(14), fill=C_SIGNAL)
        f_h2 = fonts.display(44, weight=700)
        dr.text((56, 166), "Query 354 Sequences", font=f_h2, fill=C_PAPER)
        dr.text((56, 218), "in One Browser.", font=f_h2, fill=C_SIGNAL)

        desc = (
            "Analysts filter across clubs, opponents, stadiums, defensive-line height, "
            "and pressing intensity — jumping straight to synchronized video + 2D radar."
        )
        cy = 294
        for ln in wrap_text(dr, desc, fonts.display(18, weight=450), 470):
            dr.text((56, cy), ln, font=fonts.display(18, weight=450), fill=C_MUTED)
            cy += 28

        bullets = [
            ("MULTI-MATCH FILTERING", "354 sequences (58.4 min) indexed across 3 V.League matches"),
            ("SYNCHRONIZED PLAYBACK", "HTML5 video locked frame-for-frame with 105×68m 2D radar"),
            ("KEYBOARD AUDIT & EXPORT", "R / W / U persistence to marks.json + 1-click Sportscode XML"),
        ]
        cy += 26
        for bt, bd in bullets:
            dr.line((56, cy, 520, cy), fill=C_RULE, width=1)
            dr.text((56, cy + 14), bt, font=fonts.mono(12), fill=C_SIGNAL)
            for li, bln in enumerate(wrap_text(dr, bd, fonts.display(16, weight=450), 464)):
                dr.text((56, cy + 36 + li * 22), bln, font=fonts.display(16, weight=450), fill=C_PAPER)
            cy += 98

        # Right framed browser window (x=568..1864, y=118..1016 -> 1296 x 898)
        bx0, by0, bw_box, bh_box = 568, 118, 1296, 898
        dr.rounded_rectangle((bx0, by0, bx0 + bw_box, by0 + bh_box), radius=12, fill=(14, 19, 24), outline=C_RULE, width=2)
        # Browser title bar
        dr.rounded_rectangle((bx0 + 2, by0 + 2, bx0 + bw_box - 2, by0 + 38), radius=10, fill=(21, 28, 36))
        for di, dcol in enumerate(((239, 68, 68), (245, 158, 11), (34, 197, 94))):
            dr.ellipse((bx0 + 18 + di * 18, by0 + 14, bx0 + 28 + di * 18, by0 + 24), fill=dcol)
        dr.text((bx0 + 88, by0 + 12), "http://127.0.0.1:8765  —  Footlytics Tactical Query Workspace", font=fonts.mono(12), fill=C_MUTED)

        # Zoomed crop of workspace screenshot
        sw, sh = ws_img.size
        cw = round(sw / zoom)
        ch = round(sh / zoom)
        cx_off = round((sw - cw) * 0.35 * prog)
        cy_off = round((sh - ch) * 0.20 * prog)
        cropped = ws_img.crop((cx_off, cy_off, cx_off + cw, cy_off + ch)).resize((bw_box - 8, bh_box - 44), Image.Resampling.LANCZOS)
        pil.paste(cropped, (bx0 + 4, by0 + 40))

        frames_3a.append(cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR))

    # Scene 3B (195 frames = 6.5s): Printable Vietnamese Opposition Dossier & NLE Pack
    n_3b = 195
    frames_3b: list[np.ndarray] = []
    for i in range(n_3b):
        prog = i / max(1, n_3b - 1)
        pil = Image.new("RGB", (W, H), C_BG_DEEP)
        dr = ImageDraw.Draw(pil)

        dr.rectangle((0, 0, W, 82), fill=C_CARD_DARK)
        dr.line((0, 82, W, 82), fill=C_RULE, width=2)
        w_m = draw_wordmark(dr, 56, 24, fonts, size=26)
        dr.text((56 + w_m + 12, 31), "ANALYST DELIVERY  ·  02 / 02  OPPOSITION SCOUTING PACK", font=fonts.mono(13), fill=C_SIGNAL)

        # Left: Printable Vietnamese Dossier preview with gentle vertical scroll
        dx0, dy0, dw_box, dh_box = 56, 118, 1080, 898
        dr.rounded_rectangle((dx0, dy0, dx0 + dw_box, dy0 + dh_box), radius=12, fill=(255, 255, 255), outline=C_RULE, width=2)
        sw, sh = ds_img.size
        # Crop central content column of dossier_pvfcand_1080p.png (x=280..1640) for crisp readability
        c_left, c_right = round(sw * 0.14), round(sw * 0.86)
        crop_w = c_right - c_left
        crop_h = round(crop_w * (dh_box / dw_box))
        max_scroll = max(0, sh - crop_h)
        y_sc = round(max_scroll * 0.65 * ease_in_out_cubic(prog))
        d_crop = ds_img.crop((c_left, y_sc, c_right, min(sh, y_sc + crop_h))).resize((dw_box - 8, dh_box - 8), Image.Resampling.LANCZOS)
        pil.paste(d_crop, (dx0 + 4, dy0 + 4))

        # Right: Self-contained pack deliverables spec (x=1172..1864)
        rx0 = 1172
        dr.text((rx0, 132), "footlytics pack export --club PVFCAND", font=fonts.mono(13), fill=C_SIGNAL)
        dr.text((rx0, 166), "Self-Contained Club", font=fonts.display(42, weight=700), fill=C_PAPER)
        dr.text((rx0, 216), "Opposition Pack.", font=fonts.display(42, weight=700), fill=C_SIGNAL)

        deliverables = [
            ("dossier.html / PDF", "Vietnamese scouting report synthesizing 195 PVF-CAND sequences across 3 matches."),
            ("sportscode.xml", "1-click timeline import into Hudl Sportscode, LongoMatch, and Nacsport."),
            ("clips/01..05/*.mp4", "Stratified H.264 video clips across all 5 labels with 2D tactical radar PiP."),
            ("playlist.csv", "Full spreadsheet index with kickoff-relative match clocks (H1 04:08, H2 69:38)."),
        ]
        cy = 296
        for fname, fdesc in deliverables:
            dr.rounded_rectangle((rx0, cy, 1864, cy + 156), radius=8, fill=C_CARD_DARK, outline=C_RULE, width=1)
            dr.text((rx0 + 22, cy + 18), fname, font=fonts.mono(16), fill=C_SIGNAL)
            for li, fln in enumerate(wrap_text(dr, fdesc, fonts.display(17, weight=450), 646)):
                dr.text((rx0 + 22, cy + 52 + li * 25), fln, font=fonts.display(17, weight=450), fill=C_PAPER)
            cy += 174

        frames_3b.append(cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR))

    fade = 12
    out = frames_3a[:-fade]
    for k in range(fade):
        a = ease_in_out_cubic((k + 1) / (fade + 1))
        out.append(cv2.addWeighted(frames_3b[k], a, frames_3a[-fade + k], 1.0 - a, 0.0))
    out.extend(frames_3b[fade:])
    return out


def render_act4_frames(fonts: FontBank) -> list[np.ndarray]:
    """Render Act IV (9.0s = 270 frames): Verified Benchmarks, Unit Economics & Pilot Close."""
    n_frames = 270
    frames: list[np.ndarray] = []

    columns = [
        (
            "01 · SEQUENCE PRECISION",
            "96.3%",
            "100.0% on decided clips (52 / 52)",
            "55 sequences visually audited frame-by-frame across 4 V.League matches and 3 stadiums (clearing >= 80% V1 gate).",
        ),
        (
            "02 · COMMERCIAL-CLEAN AI",
            "60.49",
            "Apache-2.0 GS-HOTA (63.10 YOLOv8x6)",
            "100% Apache-2.0 RT-DETRv2 people + ball detector: 92.8% team accuracy, 99.1% keeper side, 70.5% ball recall.",
        ),
        (
            "03 · UNIT ECONOMICS",
            "$3.66",
            "Per 90-min match (~$2.58 RT-DETRv2)",
            "4.58 GPUh baseline (~3.2 GPUh on RT-DETRv2) · < 8h overnight delivery from broadcast MP4 to full Analyst Pack.",
        ),
        (
            "04 · V.LEAGUE 2025/26 PROOF",
            "354",
            "Tactical sequences across 3 full matches",
            "549,300 frames processed across Hà Nội FC, PVF-CAND, and Công An Hà Nội at Hàng Đẫy & PVF Stadiums.",
        ),
    ]

    for i in range(n_frames):
        t_head = ease_out_cubic(min(1.0, i / 18.0))
        pil = Image.new("RGB", (W, H), C_BG_DEEP)
        dr = ImageDraw.Draw(pil)

        # Top header
        y_off = round((1.0 - t_head) * 18)
        draw_wordmark(dr, 88, 64 + y_off, fonts, size=34)
        dr.text(
            (320, 72 + y_off),
            "VERIFIED BENCHMARKS  ·  UNIT ECONOMICS  ·  PILOT READINESS",
            font=fonts.mono(14),
            fill=C_SIGNAL,
        )
        dr.line((88, 120 + y_off, W - 88, 120 + y_off), fill=C_RULE, width=2)

        dr.text((88, 162 + y_off), "Proven on Real V.League Broadcasts.", font=fonts.display(62, weight=700), fill=C_PAPER)
        dr.text((88, 238 + y_off), "Ready for Club Analyst Pilots.", font=fonts.display(62, weight=700), fill=C_SIGNAL)

        # 4 Ruled Proof Columns
        grid_y0 = 368
        dr.line((88, grid_y0, W - 88, grid_y0), fill=C_RULE, width=2)
        col_w = (W - 176) // 4
        for ci, (tag, big, sub, body) in enumerate(columns):
            stagger = ease_out_cubic(max(0.0, min(1.0, (i - 6 - ci * 4) / 20.0)))
            if stagger <= 0.02:
                continue
            cx = 88 + ci * col_w
            cy_s = round((1.0 - stagger) * 20)
            if ci > 0:
                dr.line((cx - 20, grid_y0 + 24, cx - 20, grid_y0 + 410), fill=C_RULE, width=1)
            dr.text((cx, grid_y0 + 28 + cy_s), tag, font=fonts.mono(13), fill=C_SIGNAL)
            dr.text((cx, grid_y0 + 66 + cy_s), big, font=fonts.display(72, weight=750), fill=C_PAPER)
            dr.text((cx, grid_y0 + 162 + cy_s), sub, font=fonts.display(18, weight=650), fill=C_SIGNAL)
            for li, bln in enumerate(wrap_text(dr, body, fonts.display(17, weight=450), col_w - 38)):
                dr.text((cx, grid_y0 + 206 + li * 26 + cy_s), bln, font=fonts.display(17, weight=450), fill=C_MUTED)

        # Bottom Phase 3 Pilot Banner
        by0 = 860
        dr.rounded_rectangle((88, by0, W - 88, 996), radius=10, fill=C_CARD_DARK, outline=C_SIGNAL, width=2)
        dr.text((124, by0 + 26), "NEXT MILESTONE  ·  PHASE 3 CLUB ANALYST DISCOVERY", font=fonts.mono(13), fill=C_SIGNAL)
        dr.text(
            (124, by0 + 56),
            "Live Opposition Packs & Interactive Tactical Workspace prepared for Hà Nội FC & PVF-CAND.",
            font=fonts.display(26, weight=600),
            fill=C_PAPER,
        )

        frames.append(cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR))
    return frames


def stitch_acts_and_encode(acts: list[list[np.ndarray]], out_mp4: Path) -> int:
    """Cross-dissolve between acts and stream BGR24 frames into libx264 MP4."""
    fade = 14
    all_frames: list[np.ndarray] = []
    for idx, act in enumerate(acts):
        if idx == 0:
            all_frames.extend(act[:-fade])
        else:
            prev_tail = acts[idx - 1][-fade:]
            for k in range(fade):
                a = ease_in_out_cubic((k + 1) / (fade + 1))
                all_frames.append(cv2.addWeighted(act[k], a, prev_tail[k], 1.0 - a, 0.0))
            if idx < len(acts) - 1:
                all_frames.extend(act[fade:-fade])
            else:
                all_frames.extend(act[fade:])

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for fr in all_frames:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    stderr_bytes = proc.stderr.read() if proc.stderr is not None else b""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with code {rc}: {stderr_bytes.decode(errors='replace')[-800:]}")
    return len(all_frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "out" / "showcase" / "footlytics_founder_showcase.mp4",
        help="Output 1080p MP4 path",
    )
    parser.add_argument(
        "--workspace-shot",
        type=Path,
        required=True,
        help="Path to 1080p workspace screenshot PNG",
    )
    parser.add_argument(
        "--dossier-shot",
        type=Path,
        required=True,
        help="Path to 1080p dossier screenshot PNG",
    )
    parser.add_argument(
        "--copy-to",
        type=Path,
        default=None,
        help="Optional second destination path (e.g. artifact folder)",
    )
    args = parser.parse_args()

    fonts = FontBank()
    print("[1/4] Rendering Act I (Hook + Live 1080p Calibration & Tracking)...")
    act1 = render_act1_frames(fonts)
    print(f"      Act I frames: {len(act1)}")

    print("[2/4] Rendering Act II (All 5 V1 Tactical Engines Across 3 Matches)...")
    act2 = render_act2_frames(fonts)
    print(f"      Act II frames: {len(act2)}")

    print("[3/4] Rendering Act III (Interactive Workspace & Scouting Pack)...")
    act3 = render_act3_frames(args.workspace_shot, args.dossier_shot, fonts)
    print(f"      Act III frames: {len(act3)}")

    print("[4/4] Rendering Act IV (Verified Benchmarks & Unit Economics)...")
    act4 = render_act4_frames(fonts)
    print(f"      Act IV frames: {len(act4)}")

    total = stitch_acts_and_encode([act1, act2, act3, act4], args.out)
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"Done! Wrote {args.out} ({total} frames, {total / FPS:.1f}s @ {FPS}fps, {size_mb:.1f} MB)")

    if args.copy_to is not None:
        args.copy_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.out, args.copy_to)
        print(f"Copied showcase video to {args.copy_to}")


if __name__ == "__main__":
    main()
