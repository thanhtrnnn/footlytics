"""Quality metrics over a game-state table (answers blockers B8-B12)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def compute_quality(gs: pd.DataFrame, fps: float, wall_seconds: float, expected_players: int = 22) -> dict:
    per_frame = gs.groupby("frame")["player_id"].nunique()
    n_frames = int(per_frame.shape[0])
    first_frame = int(gs["frame"].min())
    births = gs.groupby("player_id")["frame"].min()
    ball_frames = gs.groupby("frame")["ball_x_m"].apply(lambda s: s.notna().any())
    calib_frames = gs.groupby("frame")["calib_ok"].all()
    match_minutes = n_frames / fps / 60.0
    return {
        "frames": n_frames,
        "expected_players": expected_players,
        "mean_players_per_frame": float(per_frame.mean()),
        "pct_frames_ge_20_players": float((per_frame >= 20).mean()),
        "pct_frames_ge_18_players": float((per_frame >= 18).mean()),
        "unique_track_ids": int(gs["player_id"].nunique()),
        "track_births_after_first_frame": int((births > first_frame).sum()),
        "id_switch_rate_per_player_per_minute": float((births > first_frame).sum() / expected_players / match_minutes) if match_minutes > 0 else float("nan"),
        "ball_coverage": float(ball_frames.mean()),
        "calib_success_rate": float(calib_frames.mean()),
        "wall_seconds": float(wall_seconds),
        "seconds_per_match_minute": float(wall_seconds / match_minutes) if match_minutes > 0 else float("nan"),
    }


def write_quality(q: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quality.json").write_text(json.dumps(q, indent=2))
    lines = ["# Quality report", ""] + [f"- **{k}**: {v:.3f}" if isinstance(v, float) else f"- **{k}**: {v}" for k, v in q.items()]
    (out_dir / "quality.md").write_text("\n".join(lines) + "\n")
