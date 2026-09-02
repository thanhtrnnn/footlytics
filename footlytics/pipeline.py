"""End-to-end Stage 0 pipeline: video -> tracks -> teams -> game state -> overlay + quality."""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from footlytics.ball import interpolate_ball
from footlytics.calibration import anchor_frame_of, compose_from_anchor, static_homography
from footlytics.camera_motion import AnchoredCamera
from footlytics.detect import BallDetector
from footlytics.homography import image_to_pitch
from footlytics.radar import write_radar_video
from footlytics.gamestate import build_gamestate, filter_off_pitch, validate_gamestate
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


def _read_frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"cannot read frame {index} of {path}")
    return frame


def _player_mask(shape: tuple[int, int], g: pd.DataFrame | None) -> np.ndarray:
    mask = np.full(shape, 255, np.uint8)
    if g is not None:
        for r in g.itertuples(index=False):
            cv2.rectangle(mask, (int(r.x1) - 6, int(r.y1) - 6), (int(r.x2) + 6, int(r.y2) + 6), 0, -1)
    return mask


def track_camera(path: Path, tracks: pd.DataFrame, H_anchor: np.ndarray, max_frames: int | None,
                 anchor_frame: int = 0) -> tuple[dict[int, np.ndarray], dict]:
    """Per-frame image->pitch homographies via anchored camera motion (KLT + periodic SIFT re-anchor).

    Frames where the camera cannot be related to the anchor (shot cuts) are absent from the result.
    """
    by_frame = {f: g for f, g in tracks.groupby("frame")}
    anchor_bgr = _read_frame(path, anchor_frame)
    cam = AnchoredCamera(anchor_bgr, mask=_player_mask(anchor_bgr.shape[:2], by_frame.get(anchor_frame)),
                         start_at_anchor=(anchor_frame == 0))
    hom: dict[int, np.ndarray] = {}
    tracked: list[int] = []
    n = 0
    for idx, frame in iter_frames(path, max_frames):
        n += 1
        H_a_t = cam.update(frame, mask=_player_mask(frame.shape[:2], by_frame.get(idx)))
        if H_a_t is not None:
            hom[idx] = H_anchor @ np.linalg.inv(H_a_t)
        tracked.append(cam.n_tracked)
    stats = {"frames_total": n, "frames_calibrated": len(hom), "reanchors": cam.n_reanchors,
             "camera_motion_tracked_mean": float(np.mean(tracked)) if tracked else 0.0}
    return hom, stats


def detect_ball_video(path: Path, weights: str | Path, max_frames: int | None, imgsz: int = 1920) -> pd.DataFrame:
    """Dedicated ball detector pass -> rows frame, x1, y1, x2, y2, conf, cls='sports ball'."""
    det = BallDetector(weights, imgsz=imgsz)
    rows = []
    for idx, frame in iter_frames(path, max_frames):
        res = det.detect(frame)
        if res is not None:
            x, y, c = res
            rows.append({"frame": idx, "track_id": -1, "x1": x, "y1": y, "x2": x, "y2": y, "conf": c, "cls": "sports ball"})
    return pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf", "cls"])


def ball_table(tracks: pd.DataFrame, n_frames: int, max_gap: int = 5, homographies: dict[int, np.ndarray] | None = None) -> pd.DataFrame:
    balls = tracks[tracks["cls"] == "sports ball"].copy()
    balls["ball_x_m"] = (balls["x1"] + balls["x2"]) / 2
    balls["ball_y_m"] = (balls["y1"] + balls["y2"]) / 2
    best = balls.sort_values("conf", ascending=False).drop_duplicates("frame")[["frame", "ball_x_m", "ball_y_m"]].copy()
    if homographies:
        for i, r in best.iterrows():
            H = homographies.get(int(r["frame"]))
            if H is not None:
                m = image_to_pitch(H, np.array([[r["ball_x_m"], r["ball_y_m"]]]))[0]
                best.loc[i, ["ball_x_m", "ball_y_m"]] = m
            else:
                best.loc[i, ["ball_x_m", "ball_y_m"]] = np.nan
    full = pd.DataFrame({"frame": np.arange(n_frames)}).merge(best, on="frame", how="left")
    return interpolate_ball(full, max_gap=max_gap)


def run_pipeline(clip: str | Path, out_dir: str | Path, max_frames: int | None = None,
                 model_name: str = "yolo11n.pt", device: str | None = None,
                 tracker: str = DEFAULT_TRACKER, calib: str | Path | None = None,
                 ball_weights: str | Path | None = None, imgsz: int = 1280) -> dict:
    clip, out_dir = Path(clip), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = video_info(clip)
    fps = info["fps"]
    t0 = time.perf_counter()
    tracks = track_video(clip, max_frames=max_frames, model_name=model_name, device=device, tracker=tracker, imgsz=imgsz)
    persons = tracks[(tracks["cls"] == "person") & (tracks["track_id"] >= 0)].copy()
    n_frames = (max_frames if max_frames is not None else info["frames"])
    n_frames = int(min(n_frames, tracks["frame"].max() + 1)) if len(tracks) else 0
    teams = assign_teams(clip, persons, max_frames)
    homographies, cam_stats = None, {}
    if calib is not None:
        H0, calib_err = static_homography(calib)
        homographies, cam_stats = track_camera(clip, tracks, H0, max_frames, anchor_frame=anchor_frame_of(calib))
        persons = persons[persons["frame"].isin(homographies.keys())]  # drop frames without a valid camera
    ball_source = tracks
    if ball_weights is not None:
        dedicated = detect_ball_video(clip, ball_weights, max_frames)
        # dedicated detections replace COCO "sports ball" rows; COCO kept only on frames the model missed
        coco_ball = tracks[(tracks["cls"] == "sports ball") & ~tracks["frame"].isin(dedicated["frame"])]
        ball_source = pd.concat([dedicated, coco_ball], ignore_index=True)
    ball = ball_table(ball_source, n_frames, homographies=homographies)
    gs = build_gamestate(persons, teams, ball, fps=fps, homographies=homographies)
    n_before = gs["player_id"].nunique()
    gs = filter_off_pitch(gs, margin_m=2.0)
    persons = persons[persons["track_id"].isin(gs["player_id"].unique())]
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
    quality["ball_model"] = str(ball_weights) if ball_weights is not None else "coco"
    quality["imgsz"] = imgsz
    if calib is not None:
        quality["calibration"] = str(calib)
        quality["calibration_err_m"] = float(calib_err)
        quality["tracks_dropped_off_pitch"] = int(n_before - gs["player_id"].nunique())
        quality.update(cam_stats)
        quality["calib_success_rate"] = cam_stats["frames_calibrated"] / max(cam_stats["frames_total"], 1)
        write_radar_video(gs, out_dir / "radar.mp4", fps)
    write_quality(quality, out_dir)
    return {"gamestate": gs, "quality": quality, "teams": teams}
