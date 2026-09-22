"""One machine-readable quality report per run.

Built from the MatchState and the pipeline report, written next to `tracks.parquet` as
`quality.json` + `quality.md`. These are the numbers the feasibility blockers (B8-B12) are
scored on, so they stay stable across runs and repos.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..state.schema import MatchState, Role


def compute_quality(state: MatchState, report: dict, stride: int = 1, expected_players: int = 22) -> dict:
    tr = state.tracks
    players = tr[tr["role"].isin([Role.PLAYER.value, Role.GOALKEEPER.value, Role.REFEREE.value])]
    fps = float(state.meta.fps) if state.meta.fps else 25.0   # pipeline stores the effective (strided) fps
    frames = int(report.get("frames_processed") or tr["frame_idx"].nunique())
    per_frame = players.groupby("frame_idx")["track_id"].nunique() if len(players) else pd.Series(dtype=int)
    first = int(tr["frame_idx"].min()) if len(tr) else 0
    births = players.groupby("track_id")["frame_idx"].min() if len(players) else pd.Series(dtype=int)
    ball = tr.loc[tr["role"] == Role.BALL.value]
    ball_frames = ball["frame_idx"].nunique()
    # Frames where the ball was actually seen. `ball_coverage` counts gap-filled
    # frames too (analytics.ball.interpolate_ball), so the detector has to be
    # judged on this one or a change to the interpolation looks like a change to
    # the detector -- the PR-3 measurement (30.5% -> 94.9%) is this number.
    seen = ball[~ball["interpolated"].astype(bool)] if "interpolated" in ball.columns else ball
    ball_frames_detected = seen["frame_idx"].nunique()
    match_minutes = frames / fps / 60.0 if fps else float("nan")   # frames are at the effective fps
    seconds = float(report.get("seconds", float("nan")))
    q = {
        "frames": frames,
        "stride": stride,
        "expected_players": expected_players,
        "mean_players_per_frame": float(per_frame.mean()) if len(per_frame) else 0.0,
        "pct_frames_ge_18_players": float((per_frame >= 18).sum() / frames) if frames else 0.0,
        "pct_frames_ge_20_players": float((per_frame >= 20).sum() / frames) if frames else 0.0,
        "unique_track_ids": int(players["track_id"].nunique()) if len(players) else 0,
        "track_births_after_first_frame": int((births > first).sum()) if len(births) else 0,
        "id_switch_rate_per_player_per_minute": (
            float((births > first).sum() / expected_players / match_minutes) if match_minutes else float("nan")),
        "ball_coverage": float(ball_frames / frames) if frames else 0.0,
        "ball_coverage_detected": float(ball_frames_detected / frames) if frames else 0.0,
        "wall_seconds": seconds,
        "seconds_per_match_minute": float(seconds / match_minutes) if match_minutes else float("nan"),
    }
    for k in ("frames_calibrated", "frames_uncalibrated", "calib_success_rate", "reanchors",
              "tracklets_before_stitch", "tracklets_after_stitch", "median_players_per_frame",
              "dropped_off_pitch", "detections", "team_diagnosis",
              "ball_frames_detected", "ball_frames_interpolated", "ball_dropped_off_pitch",
              "officials_split"):
        if k in report:
            v = report[k]
            q[k] = v["summary"] if isinstance(v, dict) and "summary" in v else v
    q["validation"] = list(report.get("validation", []))
    return q


def write_quality(q: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "quality.json").write_text(json.dumps(q, indent=2, default=str))
    lines = ["# Quality report", ""]
    for k, v in q.items():
        if k == "validation":
            continue
        lines.append(f"- **{k}**: {v:.3f}" if isinstance(v, float) else f"- **{k}**: {v}")
    lines += ["", "## validate()", ""] + ([f"- {p}" for p in q["validation"]] or ["- no problems found"])
    (out / "quality.md").write_text("\n".join(lines) + "\n")
