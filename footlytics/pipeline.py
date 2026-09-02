"""End-to-end Stage 0 pipeline: video -> tracks -> teams -> game state -> overlay + quality."""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from footlytics.ball import interpolate_ball
from footlytics.gamestate import build_gamestate, validate_gamestate
from footlytics.ingest import iter_frames, video_info
from footlytics.quality import compute_quality, write_quality
from footlytics.render import write_overlay_video
from footlytics.team import TeamClassifier
from footlytics.track import DEFAULT_TRACKER, track_video

CROPS_PER_TRACK = 6


def _collect_crops(path: Path, persons: pd.DataFrame, max_frames: int | None) -> dict[int, list[np.ndarray]]:
    wanted = persons.groupby("track_id").head(CROPS_PER_TRACK)
    by_frame = {f: g for f, g in wanted.groupby("frame")}
    crops: dict[int, list[np.ndarray]] = defaultdict(list)
    last = int(wanted["frame"].max()) if len(wanted) else -1
    for idx, frame in iter_frames(path, max_frames):
        if idx > last:
            break
        g = by_frame.get(idx)
        if g is None:
            continue
        h, w = frame.shape[:2]
        for r in g.itertuples(index=False):
            x1, y1 = max(0, int(r.x1)), max(0, int(r.y1))
            x2, y2 = min(w, int(r.x2)), min(h, int(r.y2))
            if x2 - x1 >= 4 and y2 - y1 >= 8:
                crops[int(r.track_id)].append(frame[y1:y2, x1:x2])
    return crops


def assign_teams(path: Path, persons: pd.DataFrame, max_frames: int | None) -> dict[int, int | str]:
    crops = _collect_crops(path, persons, max_frames)
    all_crops = [c for cs in crops.values() for c in cs]
    if len(crops) < 2 or len(all_crops) < 2:
        return {tid: "unknown" for tid in persons["track_id"].unique()}
    clf = TeamClassifier().fit(all_crops)
    teams: dict[int, int | str] = {}
    for tid, cs in crops.items():
        votes = Counter(clf.predict(cs))
        teams[tid] = votes.most_common(1)[0][0]
    for tid in persons["track_id"].unique():
        teams.setdefault(int(tid), "unknown")
    return teams


def ball_table(tracks: pd.DataFrame, n_frames: int, max_gap: int = 5) -> pd.DataFrame:
    balls = tracks[tracks["cls"] == "sports ball"].copy()
    balls["ball_x_m"] = (balls["x1"] + balls["x2"]) / 2
    balls["ball_y_m"] = (balls["y1"] + balls["y2"]) / 2
    best = balls.sort_values("conf", ascending=False).drop_duplicates("frame")[["frame", "ball_x_m", "ball_y_m"]]
    full = pd.DataFrame({"frame": np.arange(n_frames)}).merge(best, on="frame", how="left")
    return interpolate_ball(full, max_gap=max_gap)


def run_pipeline(clip: str | Path, out_dir: str | Path, max_frames: int | None = None,
                 model_name: str = "yolo11n.pt", device: str | None = None,
                 tracker: str = DEFAULT_TRACKER) -> dict:
    clip, out_dir = Path(clip), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = video_info(clip)
    fps = info["fps"]
    t0 = time.perf_counter()
    tracks = track_video(clip, max_frames=max_frames, model_name=model_name, device=device, tracker=tracker)
    persons = tracks[(tracks["cls"] == "person") & (tracks["track_id"] >= 0)].copy()
    n_frames = (max_frames if max_frames is not None else info["frames"])
    n_frames = int(min(n_frames, tracks["frame"].max() + 1)) if len(tracks) else 0
    teams = assign_teams(clip, persons, max_frames)
    ball = ball_table(tracks, n_frames)
    gs = build_gamestate(persons, teams, ball, fps=fps, homographies=None)
    validate_gamestate(gs)
    wall = time.perf_counter() - t0
    gs.to_parquet(out_dir / "gamestate.parquet", index=False)
    gs.to_csv(out_dir / "gamestate.csv", index=False)
    persons["team"] = persons["track_id"].map(teams)
    write_overlay_video(iter_frames(clip, max_frames), persons, ball, out_dir / "overlay.mp4", fps)
    quality = compute_quality(gs, fps=fps, wall_seconds=wall)
    quality["clip"] = str(clip)
    quality["model"] = model_name
    quality["tracker"] = tracker
    write_quality(quality, out_dir)
    return {"gamestate": gs, "quality": quality, "teams": teams}
