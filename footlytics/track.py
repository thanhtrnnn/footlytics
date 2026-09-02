"""Detection + tracking with ultralytics YOLO and ByteTrack."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO

from footlytics.ingest import iter_frames

TRACK_COLUMNS = ["frame", "track_id", "x1", "y1", "x2", "y2", "conf", "cls"]
COCO_PERSON, COCO_SPORTS_BALL = 0, 32


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def track_video(
    path: str | Path,
    max_frames: int | None = None,
    model_name: str = "yolo11n.pt",
    device: str | None = None,
    conf: float = 0.25,
    imgsz: int = 1280,
    tracker: str = "bytetrack.yaml",
    classes: tuple[int, ...] = (COCO_PERSON, COCO_SPORTS_BALL),
) -> pd.DataFrame:
    """Run YOLO tracking frame by frame. Detections without a track id get track_id -1."""
    model = YOLO(model_name)
    device = device or default_device()
    rows: list[dict] = []
    for idx, frame in iter_frames(path, max_frames):
        res = model.track(
            frame, persist=True, classes=list(classes), conf=conf, imgsz=imgsz,
            tracker=tracker, device=device, verbose=False,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else [-1] * len(xyxy)
        for (x1, y1, x2, y2), c, k, tid in zip(xyxy, confs, clss, ids):
            rows.append({"frame": idx, "track_id": int(tid), "x1": float(x1), "y1": float(y1),
                         "x2": float(x2), "y2": float(y2), "conf": float(c), "cls": model.names[int(k)]})
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)
