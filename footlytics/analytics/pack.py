"""Phase 2 Analyst Opposition & Post-Match Pack Exporter (`footlytics pack export`).

Turns one or more processed match directories (`sequences.csv`, optional `marks.json`,
`tracks.parquet`, `quality.json`, `meta.json`) into a self-contained club analyst package:

1. `playlist.csv`       -- NLE / spreadsheet sequence playlist with tactical metrics & verdicts.
2. `sportscode.xml`     -- One-click import file for Hudl Sportscode / LongoMatch / Nacsport.
3. `dossier.md`         -- Vietnamese + technical tactical scouting dossier & unit-economics summary.
4. `dossier.html`       -- Standalone print-ready HTML dossier (Cmd+P -> PDF, zero external deps).
5. `clips/<folder>/*.mp4` -- Optional H.264 video clips (`-pre_roll_s` .. `+post_roll_s`) with
                             pure-OpenCV 2D tactical radar picture-in-picture (PiP) and HUD banner.
6. `pack_manifest.json` -- Machine-readable pack summary and unit economics.
"""
from __future__ import annotations

import html
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from ..state.schema import MatchState, Role
from ..viz.radar import hold_gaps
from .query import MatchEntry, discover_matches
from .review import load_marks, sequence_ids

PACK_FOLDERS: dict[str, str] = {
    "high_press": "01_high_press",
    "build_up": "02_build_up",
    "low_block": "03_low_block",
    "midfield_loss": "04_midfield_loss",
    "flank_overload": "05_flank_overload",
}

LABEL_ORDER: tuple[str, ...] = (
    "high_press",
    "build_up",
    "low_block",
    "midfield_loss",
    "flank_overload",
)

LABEL_VI: dict[str, str] = {
    "high_press": "Pressing tam cao (High Press)",
    "build_up": "Trien khai bong san nha (Build-up vs Press)",
    "low_block": "Khoi phong ngu thap (Low Block)",
    "midfield_loss": "Mat bong khu vuc giua san (Midfield Loss)",
    "flank_overload": "Qua tai hanh lang bien (Flank Overload)",
}

LABEL_DISPLAY_VI: dict[str, str] = {
    "high_press": "Pressing tầm cao (High Press)",
    "build_up": "Triển khai bóng sân nhà (Build-up)",
    "low_block": "Khối phòng ngự thấp (Low Block)",
    "midfield_loss": "Mất bóng giữa sân (Midfield Loss)",
    "flank_overload": "Quá tải hành lang biên (Flank Overload)",
}


def format_clock(seconds: float) -> str:
    """Format match-state / video seconds as `MM:SS` or `HH:MM:SS`."""
    s = max(0, round(float(seconds)))
    hh, rem = divmod(s, 3600)
    mm, ss = divmod(rem, 60)
    return f"{hh}:{mm:02d}:{ss:02d}" if hh else f"{mm:02d}:{ss:02d}"


def resolve_sequence_marks(
    marks: dict[str, Any] | None,
    sequences: pd.DataFrame,
    max_shift_frames: int = 50,
) -> list[tuple[str, str, str]]:
    """Return `(sequence_id, verdict, note)` per row in `sequences` (`verdict=''` if unmarked)."""
    sids = sequence_ids(sequences)
    if not marks:
        return [(sid, "", "") for sid in sids]

    seq_marks: dict[str, dict[str, str]] = marks.get("sequences", {})
    used = set(sids) & set(seq_marks)
    parsed: list[tuple[str, str, str, int, int]] = []
    for k in seq_marks:
        if k in used:
            continue
        parts = k.rsplit("-", 3)
        if len(parts) == 4:
            try:
                parsed.append((k, parts[0], parts[1], int(parts[2]), int(parts[3])))
            except ValueError:
                pass

    out: list[tuple[str, str, str]] = []
    for sid, r in zip(sids, sequences.itertuples()):
        m = seq_marks.get(sid)
        if m is None:
            r_start, r_end = int(r.start_frame), int(r.end_frame)
            cands = [
                (abs(mf - r_start), k)
                for k, lbl, tm, per, mf in parsed
                if k not in used
                and lbl == r.label
                and tm == str(r.team)
                and per == int(r.period)
                and (r_start <= mf <= r_end or abs(mf - r_start) <= max_shift_frames)
            ]
            if cands:
                _, k = min(cands)
                used.add(k)
                m = seq_marks[k]
        if m is not None:
            out.append((sid, str(m.get("verdict", "")), str(m.get("note", ""))))
        else:
            out.append((sid, "", ""))
    return out


def evidence_summary(row: pd.Series | dict[str, Any]) -> str:
    """Compact human-readable tactical metric string for HUD banners, XML tags, and tables."""
    lbl = str(row.get("label", ""))
    def_x = row.get("def_line_x")
    comp = row.get("compactness_m")
    if lbl == "low_block":
        parts = []
        if pd.notna(def_x):
            parts.append(f"def_line {float(def_x):+.1f}m")
        if pd.notna(comp):
            parts.append(f"compact {float(comp):.1f}m")
        if pd.notna(row.get("n_pressers")):
            parts.append(f"opp_in_3rd {int(row['n_pressers'])}")
        return ", ".join(parts)
    if lbl == "high_press":
        parts = []
        if pd.notna(def_x):
            parts.append(f"def_line {float(def_x):+.1f}m")
        if pd.notna(row.get("n_within_10m")):
            parts.append(f"{int(row['n_within_10m'])} within 10m")
        return ", ".join(parts)
    if lbl == "build_up":
        parts = []
        if pd.notna(row.get("n_pressers")):
            parts.append(f"vs {int(row['n_pressers'])} pressers")
        if pd.notna(def_x):
            parts.append(f"def_line {float(def_x):+.1f}m")
        return ", ".join(parts)
    if lbl == "midfield_loss":
        parts = []
        resp = str(row.get("response") or "")
        if resp:
            parts.append(resp)
        if pd.notna(row.get("ball_dist_m")):
            parts.append(f"2nd_def {float(row['ball_dist_m']):.1f}m")
        if pd.notna(row.get("retreat_mps")):
            parts.append(f"line_vel {float(row['retreat_mps']):+.1f}m/s")
        return ", ".join(parts)
    if lbl == "flank_overload":
        flk = str(row.get("flank") or "")
        att = int(row["n_attackers"]) if pd.notna(row.get("n_attackers")) else 0
        dfd = int(row["n_defenders"]) if pd.notna(row.get("n_defenders")) else 0
        return f"{flk} flank {att}v{dfd}".strip()
    return ""


def _resolve_video_for_entry(
    entry: MatchEntry,
    marks_doc: dict[str, Any] | None,
    video: Path | None,
    video_dir: Path | None,
    single_match: bool,
) -> Path | None:
    if video is not None and single_match:
        return Path(video)
    candidates: list[Path] = []
    if marks_doc and marks_doc.get("video"):
        vname = str(marks_doc["video"])
        if video_dir is not None:
            candidates.append(Path(video_dir) / vname)
        candidates.append(entry.dir / vname)
    if video_dir is not None:
        for ext in (".mp4", ".mkv", ".mov", ".m4v"):
            candidates.append(Path(video_dir) / f"{entry.match_id}{ext}")
            candidates.append(Path(video_dir) / f"{entry.dir.name}{ext}")
    for c in candidates:
        if c.is_file():
            return c
    return None


def draw_radar_pip(
    canvas_bgr: np.ndarray,
    frame_rows: pd.DataFrame | None,
    trail_rows: pd.DataFrame | None,
    *,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    pip_w: int = 336,
    pip_h: int = 218,
    margin: int = 18,
    home_label: str = "HOME",
    away_label: str = "AWAY",
) -> None:
    """Composite a fast anti-aliased 2D pitch radar picture-in-picture onto `canvas_bgr`."""
    fh, fw = canvas_bgr.shape[:2]
    if fw < pip_w + 2 * margin or fh < pip_h + 2 * margin:
        return
    x0 = fw - pip_w - margin
    y0 = fh - pip_h - margin

    roi = canvas_bgr[y0 : y0 + pip_h, x0 : x0 + pip_w]
    bg = np.full_like(roi, (24, 58, 28), dtype=np.uint8)
    cv2.addWeighted(bg, 0.84, roi, 0.16, 0.0, dst=roi)
    cv2.rectangle(roi, (0, 0), (pip_w - 1, pip_h - 1), (210, 225, 210), 1)

    pad_x, pad_y, top_bar = 14, 14, 22
    pw = pip_w - 2 * pad_x
    ph = pip_h - pad_y - (pad_y + top_bar)
    half_l, half_w = pitch_length / 2.0, pitch_width / 2.0

    def to_px(px_m: float, py_m: float) -> tuple[int, int]:
        u = round(pad_x + (px_m + half_l) / pitch_length * pw)
        v = round(top_bar + pad_y + (py_m + half_w) / pitch_width * ph)
        return int(np.clip(u, 2, pip_w - 3)), int(np.clip(v, top_bar + 2, pip_h - 3))

    # Pitch markings
    tl, br = to_px(-half_l, -half_w), to_px(half_l, half_w)
    cv2.rectangle(roi, tl, br, (185, 210, 185), 1, cv2.LINE_AA)
    mid_t, mid_b = to_px(0.0, -half_w), to_px(0.0, half_w)
    cv2.line(roi, mid_t, mid_b, (185, 210, 185), 1, cv2.LINE_AA)
    cx, cy = to_px(0.0, 0.0)
    rad_px = max(4, round(9.15 / pitch_length * pw))
    cv2.circle(roi, (cx, cy), rad_px, (185, 210, 185), 1, cv2.LINE_AA)
    for sign in (-1.0, 1.0):
        pa1 = to_px(sign * half_l, -20.16)
        pa2 = to_px(sign * (half_l - 16.5), 20.16)
        cv2.rectangle(
            roi,
            (min(pa1[0], pa2[0]), min(pa1[1], pa2[1])),
            (max(pa1[0], pa2[0]), max(pa1[1], pa2[1])),
            (170, 198, 170),
            1,
            cv2.LINE_AA,
        )

    # Legend header bar (BGR: home = amber/cyan, away = coral/red, ball = yellow)
    home_bgr = (255, 195, 55)
    away_bgr = (80, 90, 255)
    cv2.rectangle(roi, (0, 0), (pip_w, top_bar), (15, 20, 16), -1)
    cv2.circle(roi, (12, 11), 4, home_bgr, -1, cv2.LINE_AA)
    cv2.putText(roi, home_label[:8], (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.circle(roi, (105, 11), 4, away_bgr, -1, cv2.LINE_AA)
    cv2.putText(roi, away_label[:8], (113, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.circle(roi, (202, 11), 4, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(roi, "BALL", (210, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (235, 235, 235), 1, cv2.LINE_AA)

    if trail_rows is not None and not trail_rows.empty:
        for tid, tg in trail_rows.groupby("track_id", observed=True):
            if int(tid) < 0 or len(tg) < 2:
                continue
            tm = str(tg["team"].iloc[-1])
            col = home_bgr if tm == "home" else (away_bgr if tm == "away" else (180, 180, 180))
            pts = np.array([to_px(float(x), float(y)) for x, y in zip(tg["x"], tg["y"])], np.int32)
            cv2.polylines(roi, [pts.reshape(-1, 1, 2)], False, col, 1, cv2.LINE_AA)

    if frame_rows is not None and not frame_rows.empty:
        for r in frame_rows.itertuples():
            if not (np.isfinite(r.x) and np.isfinite(r.y)):
                continue
            u, v = to_px(float(r.x), float(r.y))
            if r.role == Role.BALL.value:
                cv2.circle(roi, (u, v), 5, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(roi, (u, v), 5, (10, 10, 10), 1, cv2.LINE_AA)
            elif r.role in (Role.PLAYER.value, Role.GOALKEEPER.value):
                col = home_bgr if r.team == "home" else (away_bgr if r.team == "away" else (190, 190, 190))
                cv2.circle(roi, (u, v), 5, col, -1, cv2.LINE_AA)
                cv2.circle(roi, (u, v), 5, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_hud_banner(
    canvas_bgr: np.ndarray,
    *,
    club: str,
    opponent: str,
    period: int,
    label: str,
    evidence: str,
    in_window: bool,
    cur_s: float,
) -> None:
    fw = canvas_bgr.shape[1]
    bar_h = 34
    roi = canvas_bgr[:bar_h, :]
    dark = np.full_like(roi, (14, 18, 22), dtype=np.uint8)
    cv2.addWeighted(dark, 0.78, roi, 0.22, 0.0, dst=roi)
    badge_col = (45, 185, 85) if in_window else (95, 105, 115)
    badge_txt = "LIVE SEQ" if in_window else "CONTEXT"
    cv2.rectangle(canvas_bgr, (12, 6), (98, 28), badge_col, -1)
    cv2.putText(canvas_bgr, badge_txt, (19, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
    title = f"{club} vs {opponent}  |  P{period} {format_clock(cur_s)}  |  {label.upper()}"
    if evidence:
        title += f"  ({evidence})"
    cv2.putText(canvas_bgr, title[:115], (110, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
    if in_window:
        cv2.line(canvas_bgr, (0, bar_h - 1), (fw - 1, bar_h - 1), badge_col, 2)


def export_sequence_clip(
    video_path: Path,
    out_mp4: Path,
    *,
    seq_start_s: float,
    seq_end_s: float,
    clip_start_s: float,
    clip_end_s: float,
    video_offset_s: float = 0.0,
    state: MatchState | None = None,
    club: str = "HOME",
    opponent: str = "AWAY",
    home_name: str = "HOME",
    away_name: str = "AWAY",
    period: int = 1,
    label: str = "",
    evidence: str = "",
    width: int = 1280,
    max_out_fps: float = 30.0,
) -> Path:
    """Cut `[clip_start_s, clip_end_s]` with HUD banner and optional 2D radar PiP into H.264 `.mp4`."""
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    stride = max(1, round(src_fps / max_out_fps)) if src_fps > max_out_fps * 1.15 else 1
    out_fps = src_fps / stride
    out_w = int(width) + (int(width) % 2)
    out_h = round(src_h * (out_w / max(src_w, 1)))
    if out_h % 2:
        out_h -= 1

    v_start_s = max(0.0, clip_start_s + video_offset_s)
    v_end_s = max(v_start_s + 0.2, clip_end_s + video_offset_s)
    f0 = max(0, round(v_start_s * src_fps))
    f1 = round(v_end_s * src_fps)
    if total_frames > 0:
        f1 = min(total_frames, f1)

    state_fps = float(state.meta.fps) if state is not None else out_fps
    by_frame: dict[int, pd.DataFrame] = {}
    slice_people: pd.DataFrame | None = None
    if state is not None and not state.tracks.empty:
        sf0 = max(0, round(clip_start_s * state_fps) - 20)
        sf1 = round(clip_end_s * state_fps) + 20
        sub = state.tracks[(state.tracks["frame_idx"] >= sf0) & (state.tracks["frame_idx"] <= sf1)]
        if not sub.empty:
            held = hold_gaps(sub, 8)
            by_frame = {int(f): g for f, g in held.groupby("frame_idx", observed=True)}
            slice_people = held[held["role"].isin([Role.PLAYER.value, Role.GOALKEEPER.value])]

    ffmpeg_bin = shutil.which("ffmpeg")
    proc: subprocess.Popen[bytes] | None = None
    writer: cv2.VideoWriter | None = None
    try:
        if ffmpeg_bin:
            proc = subprocess.Popen(
                [
                    ffmpeg_bin,
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "-s",
                    f"{out_w}x{out_h}",
                    "-r",
                    f"{out_fps:.4f}",
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(out_mp4),
                ],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_mp4), fourcc, out_fps, (out_w, out_h))

        if f0 > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        pos = f0
        rel = 0
        while pos < f1:
            ok, bgr = cap.read()
            if not ok:
                break
            if rel % stride == 0:
                cur_match_s = (pos / src_fps) - video_offset_s
                canvas = (
                    cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
                    if (bgr.shape[1], bgr.shape[0]) != (out_w, out_h)
                    else bgr.copy()
                )
                in_seq = seq_start_s <= cur_match_s <= seq_end_s
                _draw_hud_banner(
                    canvas,
                    club=club,
                    opponent=opponent,
                    period=period,
                    label=label,
                    evidence=evidence,
                    in_window=in_seq,
                    cur_s=cur_match_s,
                )
                if state is not None:
                    s_idx = round(cur_match_s * state_fps)
                    f_rows = by_frame.get(s_idx)
                    t_rows = (
                        slice_people[
                            (slice_people["frame_idx"] <= s_idx)
                            & (slice_people["frame_idx"] > s_idx - 15)
                        ]
                        if slice_people is not None and not slice_people.empty
                        else None
                    )
                    draw_radar_pip(
                        canvas,
                        f_rows,
                        t_rows,
                        pitch_length=state.meta.pitch_length,
                        pitch_width=state.meta.pitch_width,
                        home_label=home_name,
                        away_label=away_name,
                    )
                if proc is not None and proc.stdin is not None:
                    proc.stdin.write(canvas.tobytes())
                elif writer is not None:
                    writer.write(canvas)
            pos += 1
            rel += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if proc is not None:
            if proc.stdin is not None:
                proc.stdin.close()
            err = proc.stderr.read().decode(errors="replace") if proc.stderr is not None else ""
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"ffmpeg exited {rc}: {err.strip()[-400:]}")
    return out_mp4


def write_sportscode_xml(playlist: pd.DataFrame, out_xml: Path) -> Path:
    """Write a Hudl Sportscode / LongoMatch / Nacsport compatible XML timeline."""
    root = ET.Element("file")
    all_inst = ET.SubElement(root, "ALL_INSTANCES")

    for idx, r in enumerate(playlist.itertuples(), 1):
        inst = ET.SubElement(all_inst, "instance")
        ET.SubElement(inst, "ID").text = str(idx)
        ET.SubElement(inst, "start").text = f"{float(r.clip_start_s):.2f}"
        ET.SubElement(inst, "end").text = f"{float(r.clip_end_s):.2f}"
        club_tag = str(r.club or r.side).upper()
        ET.SubElement(inst, "code").text = f"{club_tag} - {r.label}"

        tags = [
            ("Match", str(r.match_id)),
            ("Club", str(r.club or r.side)),
            ("Opponent", str(r.opponent or "")),
            ("Period", f"P{int(r.period)}"),
            ("Tactical Label", str(r.label)),
            ("Verdict", str(r.verdict or "unmarked")),
            ("Evidence", str(r.evidence_summary or "")),
        ]
        if r.note:
            tags.append(("Analyst Note", str(r.note)))
        for grp, txt in tags:
            if not txt:
                continue
            lbl_el = ET.SubElement(inst, "label")
            ET.SubElement(lbl_el, "group").text = grp
            ET.SubElement(lbl_el, "text").text = txt

    ET.indent(root, space="  ")
    out_xml = Path(out_xml)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    out_xml.write_bytes(xml_bytes + b"\n")
    return out_xml


def _tactical_profile(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-label tactical aggregate across all exported sequences."""
    profiles: list[dict[str, Any]] = []
    for lbl in LABEL_ORDER:
        sub = df[df["label"] == lbl] if not df.empty else df
        if sub.empty:
            continue
        item: dict[str, Any] = {
            "label": lbl,
            "display_vi": LABEL_DISPLAY_VI.get(lbl, lbl),
            "count": len(sub),
            "total_duration_s": round(float(sub["duration_s"].sum()), 1),
            "mean_duration_s": round(float(sub["duration_s"].mean()), 1),
            "key_metric": "",
        }
        if lbl == "low_block":
            dx = float(sub["def_line_x"].dropna().mean()) if sub["def_line_x"].notna().any() else float("nan")
            cp = float(sub["compactness_m"].dropna().mean()) if sub["compactness_m"].notna().any() else float("nan")
            item["mean_def_line_x"] = round(dx, 1)
            item["mean_compactness_m"] = round(cp, 1)
            item["key_metric"] = f"Line phòng ngự TB {dx:+.1f} m, độ cô đặc {cp:.1f} m"
        elif lbl == "build_up":
            pr = sub["n_pressers"].dropna()
            vs3 = int((pr >= 3).sum())
            mean_pr = float(pr.mean()) if len(pr) else 0.0
            item["vs_3plus_pressers"] = vs3
            item["key_metric"] = f"{vs3}/{len(sub)} tình huống trước >= 3 cầu thủ pressing (TB {mean_pr:.1f} người)"
        elif lbl == "high_press":
            dx = float(sub["def_line_x"].dropna().mean()) if sub["def_line_x"].notna().any() else float("nan")
            n10 = float(sub["n_within_10m"].dropna().mean()) if sub["n_within_10m"].notna().any() else 0.0
            item["key_metric"] = f"Line dâng cao TB {dx:+.1f} m, TB {n10:.1f} cầu thủ áp sát trong 10 m"
        elif lbl == "midfield_loss":
            vc = sub["response"].fillna("").value_counts()
            cp_n = int(vc.get("counter_press", 0))
            rt_n = int(vc.get("retreat", 0))
            hd_n = int(vc.get("hold", 0))
            n = len(sub)
            item["counter_press_share"] = round(cp_n / n, 2)
            item["retreat_share"] = round(rt_n / n, 2)
            item["key_metric"] = (
                f"Phản ứng sau mất bóng: Counter-press {cp_n} ({round(100*cp_n/n)}%), "
                f"Lùi khối {rt_n} ({round(100*rt_n/n)}%), Giữ cự ly {hd_n} ({round(100*hd_n/n)}%)"
            )
        elif lbl == "flank_overload":
            vc = sub["flank"].fillna("").value_counts()
            item["key_metric"] = f"Cánh phải: {int(vc.get('right', 0))}, Cánh trái: {int(vc.get('left', 0))}"
        profiles.append(item)
    return profiles


def _takeaways_vi(target_title: str, n_matches: int, profiles: list[dict[str, Any]], df: pd.DataFrame) -> list[str]:
    by_lbl = {p["label"]: p for p in profiles}
    bullets: list[str] = []
    if "low_block" in by_lbl:
        p = by_lbl["low_block"]
        bullets.append(
            f"**Tổ chức khối thấp ({p['count']} tình huống / {p['total_duration_s']:.0f}s qua {n_matches} trận):** "
            f"{target_title} lùi khối phòng ngự với chiều cao hàng thủ trung bình `{p.get('mean_def_line_x', 0):+.1f} m` "
            f"và cự ly đội hình `{p.get('mean_compactness_m', 0):.1f} m` (trung bình `{p['mean_duration_s']:.1f}s`/nhịp)."
        )
    if "build_up" in by_lbl:
        p = by_lbl["build_up"]
        bullets.append(
            f"**Triển khai bóng từ sân nhà ({p['count']} tình huống):** "
            f"`{p.get('vs_3plus_pressers', 0)}/{p['count']}` pha build-up diễn ra dưới áp lực `>= 3` cầu thủ đối phương ở 1/3 sân nhà."
        )
    if "midfield_loss" in by_lbl:
        p = by_lbl["midfield_loss"]
        bullets.append(
            f"**Chuyển đổi trạng thái khi mất bóng giữa sân ({p['count']} tình huống):** {p['key_metric']}."
        )
    if "high_press" in by_lbl:
        p = by_lbl["high_press"]
        bullets.append(
            f"**Thời điểm dâng cao pressing ({p['count']} tình huống):** {p['key_metric']}."
        )
    if "flank_overload" in by_lbl:
        p = by_lbl["flank_overload"]
        bullets.append(
            f"**Tập trung quân số hành lang biên ({p['count']} tình huống):** {p['key_metric']}."
        )
    if not bullets and df.empty:
        bullets.append("Không có tình huống nào khớp bộ lọc đã chọn.")
    return bullets


def _render_dossier_markdown(
    *,
    title: str,
    matches_meta: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    takeaways: list[str],
    playlist: pd.DataFrame,
    economics: dict[str, Any],
) -> str:
    lines: list[str] = [
        f"# BÁO CÁO PHÂN TÍCH CHIẾN THUẬT — {title}",
        "",
        f"- **Số trận phân tích:** `{len(matches_meta)}`",
        (
            f"- **Tổng số tình huống xuất gói:** `{len(playlist)}` "
            f"(đã xác thực `right`: `{economics['verified_right_sequences']}`)"
        ),
        (
            f"- **Thời gian xử lý GPU & Chi phí quy đổi:** `{economics['total_compute_hours']:.2f} giờ GPU` "
            f"(ước tính `${economics['estimated_compute_usd']:.2f} USD` @ `${economics['gpu_hourly_usd']:.2f}/h`)"
            + (
                f", QA nội bộ `{economics['operator_minutes']:.0f} phút`"
                if economics.get("operator_minutes") is not None
                else ""
            )
        ),
        "",
        "## 1. Danh sách trận đấu & Chất lượng dữ liệu",
        "",
        "| Trận đấu (`match_id`) | Cặp đấu | FPS xử lý | Tỷ lệ Calib | Phủ sóng bóng | Tình huống xuất |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for m in matches_meta:
        csr = f"{100 * m['calib_success_rate']:.1f}%" if m.get("calib_success_rate") is not None else "n/a"
        bcov = f"{100 * m['ball_coverage']:.1f}%" if m.get("ball_coverage") is not None else "n/a"
        fps_s = f"{m['fps']:.1f}" if m.get("fps") else "n/a"
        lines.append(
            f"| `{m['match_id']}` | {m['home']} vs {m['away']} | {fps_s} | {csr} | {bcov} | {m['exported_count']} |"
        )

    lines += [
        "",
        "## 2. Tổng hợp mẫu hình chiến thuật (5 Nhãn V1)",
        "",
        "| Mẫu hình chiến thuật | Số lượng | Tổng thời lượng (s) | TB / tình huống (s) | Chỉ số đặc trưng |",
        "|---|---:|---:|---:|---|",
    ]
    for p in profiles:
        lines.append(
            f"| {p['display_vi']} | {p['count']} | {p['total_duration_s']:.1f} | "
            f"{p['mean_duration_s']:.1f} | {p['key_metric']} |"
        )

    lines += ["", "## 3. Điểm nhấn chiến thuật tự động", ""]
    for b in takeaways:
        lines.append(f"- {b}")

    lines += [
        "",
        "## 4. Danh mục tình huống & Video Clips",
        "",
        "| # | Trận | CLB | Hiệp & Đồng hồ | Nhãn | Thời lượng | Thông số | Video Clip | Ghi chú |",
        "|---:|---|---|---|---|---:|---|---|---|",
    ]
    for r in playlist.itertuples():
        clip_cell = f"[Xem clip]({r.clip_path})" if r.clip_path else "—"
        note_clean = str(r.note or "").replace("|", "/")
        lines.append(
            f"| {int(r.clip_index)} | `{r.match_id}` | {r.club or r.side} | "
            f"H{int(r.period)} `{r.clock}` | `{r.label}` | {float(r.duration_s):.1f}s | "
            f"{r.evidence_summary} | {clip_cell} | {note_clean} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_dossier_html(
    *,
    title: str,
    matches_meta: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    takeaways: list[str],
    playlist: pd.DataFrame,
    economics: dict[str, Any],
) -> str:
    esc = html.escape
    cards_html = "".join(
        f"<div class='card'><div class='k'>{esc(p['display_vi'])}</div>"
        f"<div class='v'>{p['count']} <span class='sub'>({p['total_duration_s']:.0f}s)</span></div>"
        f"<div class='m'>{esc(p['key_metric'])}</div></div>"
        for p in profiles
    )
    def _fmt_match_row(m: dict[str, Any]) -> str:
        fps_cell = f"{m['fps']:.1f}" if m.get("fps") else "n/a"
        csr_cell = f"{100 * m['calib_success_rate']:.1f}%" if m.get("calib_success_rate") is not None else "n/a"
        bcov_cell = f"{100 * m['ball_coverage']:.1f}%" if m.get("ball_coverage") is not None else "n/a"
        return (
            f"<tr><td><code>{esc(str(m['match_id']))}</code></td>"
            f"<td>{esc(m['home'])} vs {esc(m['away'])}</td>"
            f"<td class='num'>{fps_cell}</td>"
            f"<td class='num'>{csr_cell}</td>"
            f"<td class='num'>{bcov_cell}</td>"
            f"<td class='num'>{m['exported_count']}</td></tr>"
        )

    def _fmt_seq_row(r: Any) -> str:
        clip_link = f'<a href="{esc(str(r.clip_path))}">MP4 + Radar</a>' if r.clip_path else "—"
        return (
            f"<tr><td class='num'>{int(r.clip_index)}</td>"
            f"<td><code>{esc(str(r.match_id))}</code></td>"
            f"<td><strong>{esc(str(r.club or r.side))}</strong></td>"
            f"<td>H{int(r.period)} <code>{esc(str(r.clock))}</code></td>"
            f"<td><span class='pill'>{esc(str(r.label))}</span></td>"
            f"<td class='num'>{float(r.duration_s):.1f}s</td>"
            f"<td>{esc(str(r.evidence_summary))}</td>"
            f"<td>{clip_link}</td>"
            f"<td>{esc(str(r.note or ''))}</td></tr>"
        )

    matches_rows = "".join(_fmt_match_row(m) for m in matches_meta)
    takeaway_items = "".join(
        f"<li>{esc(t.replace('**', '').replace('`', ''))}</li>" for t in takeaways
    )
    seq_rows = "".join(_fmt_seq_row(r) for r in playlist.itertuples())
    qa_str = (
        f" &middot; QA nội bộ: {economics['operator_minutes']:.0f} phút"
        if economics.get("operator_minutes") is not None
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Báo cáo Chiến thuật — {esc(title)}</title>
<style>
  :root {{ --bg:#f8fafc; --ink:#0f172a; --muted:#475569; --line:#e2e8f0; --brand:#0f766e; --card:#ffffff; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:var(--bg); color:var(--ink); margin:0; padding:32px; line-height:1.5; }}
  .wrap {{ max-width:1120px; margin:0 auto; }}
  header {{ background:#0f172a; color:#f8fafc; padding:24px 28px; border-radius:10px; margin-bottom:24px; }}
  header h1 {{ margin:0 0 8px 0; font-size:22px; letter-spacing:0.2px; }}
  header .meta {{ color:#cbd5e1; font-size:13px; }}
  h2 {{ font-size:16px; text-transform:uppercase; letter-spacing:0.6px; color:#1e293b; border-bottom:2px solid var(--line); padding-bottom:6px; margin-top:28px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; margin:14px 0; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px 16px; }}
  .card .k {{ font-size:12px; font-weight:600; color:var(--muted); }}
  .card .v {{ font-size:24px; font-weight:700; margin:4px 0; color:var(--brand); }}
  .card .v .sub {{ font-size:13px; font-weight:500; color:var(--muted); }}
  .card .m {{ font-size:12px; color:#334155; }}
  ul.takeaways {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:16px 20px 16px 36px; margin:12px 0; }}
  ul.takeaways li {{ margin:6px 0; font-size:14px; }}
  table {{ width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; font-size:13px; }}
  th, td {{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
  th {{ background:#f1f5f9; font-weight:600; color:#334155; }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .pill {{ display:inline-block; padding:2px 7px; border-radius:4px; background:#ccfbf1; color:#115e59; font-weight:600; font-size:11px; }}
  a {{ color:var(--brand); font-weight:600; text-decoration:none; }}
  a:hover {{ text-decoration:underline; }}
  @media print {{ body {{ background:#fff; padding:0; }} header {{ border-radius:0; }} tr {{ page-break-inside:avoid; }} }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>BÁO CÁO PHÂN TÍCH CHIẾN THUẬT — {esc(title)}</h1>
    <div class="meta">Số trận: {len(matches_meta)} &middot; Tình huống xuất: {len(playlist)} (đã xác thực right: {economics['verified_right_sequences']}) &middot; Compute: {economics['total_compute_hours']:.2f} GPUh (~${economics['estimated_compute_usd']:.2f} USD){qa_str}</div>
  </header>

  <h2>1. Tổng hợp mẫu hình chiến thuật</h2>
  <div class="grid">{cards_html}</div>

  <h2>2. Điểm nhấn chiến thuật tự động</h2>
  <ul class="takeaways">{takeaway_items}</ul>

  <h2>3. Danh sách trận đấu &amp; Chất lượng dữ liệu</h2>
  <table>
    <thead><tr><th>Trận đấu</th><th>Cặp đấu</th><th class="num">FPS</th><th class="num">Tỷ lệ Calib</th><th class="num">Phủ sóng bóng</th><th class="num">Tình huống</th></tr></thead>
    <tbody>{matches_rows}</tbody>
  </table>

  <h2>4. Danh mục tình huống &amp; Video Clips</h2>
  <table>
    <thead><tr><th class="num">#</th><th>Trận</th><th>CLB</th><th>Đồng hồ</th><th>Nhãn</th><th class="num">Thời lượng</th><th>Thông số</th><th>Clip</th><th>Ghi chú</th></tr></thead>
    <tbody>{seq_rows}</tbody>
  </table>
</div>
</body>
</html>
"""


def export_pack(
    roots: Sequence[Path | str] | Path | str,
    out_dir: Path | str,
    *,
    club: str | None = None,
    team: str | None = None,
    label: str | None = None,
    verified_only: bool = False,
    video: Path | None = None,
    video_dir: Path | None = None,
    export_clips: bool = True,
    pre_roll_s: float = 3.0,
    post_roll_s: float = 2.0,
    max_clips: int | None = None,
    gpu_hourly_usd: float = 0.80,
    operator_minutes: float | None = None,
) -> dict[str, Any]:
    """Build an Analyst Opposition / Match Pack in `out_dir` and return its manifest dict."""
    entries = discover_matches(roots)
    if not entries:
        raise ValueError("no match directories containing sequences.csv found")
    if video is not None and len(entries) > 1:
        raise ValueError("--video applies to a single match; use --video-dir for multi-match packs")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    matches_meta: list[dict[str, Any]] = []
    total_compute_s = 0.0
    match_videos: dict[str, tuple[Path, float, MatchEntry]] = {}

    for entry in entries:
        seq_df = pd.read_csv(entry.sequences_path)
        marks_path = entry.dir / "marks.json"
        marks_doc = load_marks(marks_path) if marks_path.is_file() else None
        resolved_marks = resolve_sequence_marks(marks_doc, seq_df)

        quality_path = entry.dir / "quality.json"
        report_path = entry.dir / "report.json"
        q_doc = json.loads(quality_path.read_text()) if quality_path.is_file() else {}
        r_doc = json.loads(report_path.read_text()) if report_path.is_file() else {}
        comp_s = float(
            q_doc.get("compute_seconds")
            or r_doc.get("compute_seconds")
            or q_doc.get("seconds")
            or r_doc.get("seconds")
            or 0.0
        )
        total_compute_s += comp_s

        vid_path = _resolve_video_for_entry(
            entry, marks_doc, video, video_dir, single_match=(len(entries) == 1)
        )
        vid_offset = float(marks_doc.get("video_offset_s", 0.0)) if marks_doc else 0.0
        if vid_path is not None:
            match_videos[entry.match_id] = (vid_path, vid_offset, entry)

        exported_for_match = 0
        for (_, row), (sid, verdict, note) in zip(seq_df.iterrows(), resolved_marks):
            side = str(row["team"])
            row_club = entry.home if side == "home" else (entry.away if side == "away" else "")
            row_opp = entry.away if side == "home" else (entry.home if side == "away" else "")

            if club is not None and row_club.lower() != club.lower():
                continue
            if team is not None and side.lower() != team.lower():
                continue
            if label is not None and str(row["label"]) != label:
                continue
            if verified_only:
                if verdict != "right":
                    continue
            else:
                if verdict in ("wrong", "unsure"):
                    continue

            s0, s1 = float(row["start_s"]), float(row["end_s"])
            c0 = max(0.0, s0 - pre_roll_s)
            c1 = s1 + post_roll_s
            exported_for_match += 1
            all_rows.append(
                {
                    "sequence_id": sid,
                    "match_id": entry.match_id,
                    "club": row_club,
                    "opponent": row_opp,
                    "side": side,
                    "period": int(row["period"]),
                    "clock": f"{format_clock(s0)}-{format_clock(s1)}",
                    "start_s": round(s0, 2),
                    "end_s": round(s1, 2),
                    "clip_start_s": round(c0, 2),
                    "clip_end_s": round(c1, 2),
                    "duration_s": round(float(row["duration_s"]), 2),
                    "label": str(row["label"]),
                    "verdict": verdict,
                    "note": note,
                    "evidence_summary": evidence_summary(row),
                    "def_line_x": row.get("def_line_x"),
                    "compactness_m": row.get("compactness_m"),
                    "n_within_10m": row.get("n_within_10m"),
                    "n_pressers": row.get("n_pressers"),
                    "flank": row.get("flank"),
                    "n_attackers": row.get("n_attackers"),
                    "n_defenders": row.get("n_defenders"),
                    "in_possession_share": row.get("in_possession_share"),
                    "ball_dist_m": row.get("ball_dist_m"),
                    "retreat_mps": row.get("retreat_mps"),
                    "response": row.get("response"),
                    "clip_path": "",
                }
            )

        matches_meta.append(
            {
                "match_id": entry.match_id,
                "home": entry.home or "HOME",
                "away": entry.away or "AWAY",
                "fps": entry.fps,
                "calib_success_rate": q_doc.get("calib_success_rate", r_doc.get("calib_success_rate")),
                "ball_coverage": q_doc.get("ball_coverage"),
                "compute_seconds": round(comp_s, 1),
                "exported_count": exported_for_match,
            }
        )

    playlist = pd.DataFrame(all_rows)
    if not playlist.empty:
        rank = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
        playlist["_ord"] = playlist["label"].map(lambda x: rank.get(str(x), 99))
        playlist = (
            playlist.sort_values(["_ord", "match_id", "start_s"], kind="stable")
            .drop(columns=["_ord"])
            .reset_index(drop=True)
        )
        if max_clips is not None and max_clips >= 0:
            playlist = playlist.iloc[: int(max_clips)].copy()
        playlist.insert(0, "clip_index", range(1, len(playlist) + 1))
    else:
        playlist = pd.DataFrame(
            columns=[
                "clip_index", "sequence_id", "match_id", "club", "opponent", "side", "period",
                "clock", "start_s", "end_s", "clip_start_s", "clip_end_s", "duration_s",
                "label", "verdict", "note", "evidence_summary", "clip_path",
            ]
        )

    clips_exported = 0
    if export_clips and not playlist.empty and match_videos:
        loaded_states: dict[str, MatchState | None] = {}
        clip_paths: list[str] = []
        for r in playlist.itertuples():
            mv = match_videos.get(str(r.match_id))
            if mv is None:
                clip_paths.append("")
                continue
            vid_path, vid_offset, m_entry = mv
            if m_entry.match_id not in loaded_states:
                loaded_states[m_entry.match_id] = (
                    MatchState.load(m_entry.dir) if m_entry.tracks_path is not None else None
                )
            folder = PACK_FOLDERS.get(str(r.label), f"99_{r.label}")
            fname = (
                f"{int(r.clip_index):03d}_{r.match_id}_P{int(r.period)}_"
                f"{round(float(r.start_s)):04d}s_{r.label}_{r.club or r.side}.mp4"
            )
            rel_mp4 = Path("clips") / folder / fname
            export_sequence_clip(
                vid_path,
                out_path / rel_mp4,
                seq_start_s=float(r.start_s),
                seq_end_s=float(r.end_s),
                clip_start_s=float(r.clip_start_s),
                clip_end_s=float(r.clip_end_s),
                video_offset_s=vid_offset,
                state=loaded_states[m_entry.match_id],
                club=str(r.club or r.side).upper(),
                opponent=str(r.opponent or ("AWAY" if r.side == "home" else "HOME")).upper(),
                home_name=m_entry.home or "HOME",
                away_name=m_entry.away or "AWAY",
                period=int(r.period),
                label=str(r.label),
                evidence=str(r.evidence_summary or ""),
            )
            clip_paths.append(rel_mp4.as_posix())
            clips_exported += 1
        playlist["clip_path"] = clip_paths

    csv_cols = [
        "clip_index", "sequence_id", "match_id", "club", "opponent", "side", "period",
        "clock", "start_s", "end_s", "clip_start_s", "clip_end_s", "duration_s",
        "label", "verdict", "note", "evidence_summary", "clip_path",
    ]
    playlist[csv_cols].to_csv(out_path / "playlist.csv", index=False)
    write_sportscode_xml(playlist, out_path / "sportscode.xml")

    profiles = _tactical_profile(playlist)
    title = (
        club.upper()
        if club
        else (
            f"{matches_meta[0]['home']} vs {matches_meta[0]['away']}"
            if len(matches_meta) == 1
            else f"{len(matches_meta)} Matches"
        )
    )
    takeaways = _takeaways_vi(title, len(matches_meta), profiles, playlist)
    compute_hours = total_compute_s / 3600.0
    economics = {
        "matches_count": len(matches_meta),
        "sequences_exported": len(playlist),
        "verified_right_sequences": int((playlist["verdict"] == "right").sum()) if len(playlist) else 0,
        "clips_exported": clips_exported,
        "total_compute_seconds": round(total_compute_s, 1),
        "total_compute_hours": round(compute_hours, 3),
        "gpu_hourly_usd": float(gpu_hourly_usd),
        "estimated_compute_usd": round(compute_hours * float(gpu_hourly_usd), 2),
        "operator_minutes": operator_minutes,
    }

    md_text = _render_dossier_markdown(
        title=title,
        matches_meta=matches_meta,
        profiles=profiles,
        takeaways=takeaways,
        playlist=playlist,
        economics=economics,
    )
    html_text = _render_dossier_html(
        title=title,
        matches_meta=matches_meta,
        profiles=profiles,
        takeaways=takeaways,
        playlist=playlist,
        economics=economics,
    )
    (out_path / "dossier.md").write_text(md_text)
    (out_path / "dossier.html").write_text(html_text)

    manifest = {
        "title": title,
        "club_filter": club,
        "team_filter": team,
        "label_filter": label,
        "verified_only": verified_only,
        "economics": economics,
        "matches": matches_meta,
        "profiles": profiles,
        "files": {
            "playlist_csv": str(out_path / "playlist.csv"),
            "sportscode_xml": str(out_path / "sportscode.xml"),
            "dossier_md": str(out_path / "dossier.md"),
            "dossier_html": str(out_path / "dossier.html"),
        },
    }
    (out_path / "pack_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
