"""Run the upstream pipeline on a moving-camera clip through the AnchoredCamera bridge.

Usage: python scripts/run_moving_cam.py CLIP CALIB_JSON OUT_DIR [--anchor-frame N] [--stride N]
       [--max-frames N] [--weights W] [--device mps|cuda|cpu] [--no-camera]
Writes OUT_DIR/{tracks.parquet, meta.json, report.json}.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, ".")
import cv2
import numpy as np
from footlytics.geometry.camera_motion import AnchoredCamera
from footlytics.geometry.homography import Calibration
from footlytics.geometry.pitch import DEFAULT_PITCH
from footlytics.perception.detect import Detector, DetectorConfig
from footlytics.pipeline.radar import PipelineConfig, run
from footlytics.state.schema import MatchMeta, TeamInfo


def read_frame(path, idx):
    cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, f = cap.read(); cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {idx}")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip"); ap.add_argument("calib"); ap.add_argument("out")
    ap.add_argument("--anchor-frame", type=int, default=0); ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--weights", default="weights/yolo_v8x6_finetuned.pt"); ap.add_argument("--device", default="mps")
    ap.add_argument("--no-camera", action="store_true")
    a = ap.parse_args()
    lm = DEFAULT_PITCH.landmarks()
    cal = Calibration.load(a.calib, lm)
    print("calibration:", cal.verdict(lm)[1])
    camera = None if a.no_camera else AnchoredCamera(read_frame(a.clip, a.anchor_frame), start_at_anchor=(a.anchor_frame == 0))
    det = Detector(a.weights, DetectorConfig(tile=0, imgsz=1280, device=a.device, half=(a.device == "cuda"), conf=0.25))
    meta = MatchMeta(match_id=Path(a.clip).stem, fps=25.0, source_type="tactical_cam",
                     home=TeamInfo("Brazil", "BRA"), away=TeamInfo("France", "FRA"))
    t0 = time.time()
    state, report = run(a.clip, cal, det, meta, PipelineConfig(stride=a.stride, max_frames=a.max_frames, verbose=True), camera=camera)
    out = Path(a.out); state.save(out)
    clean = {k: v for k, v in report.items() if not k.startswith("_")}
    clean["wall_seconds"] = time.time() - t0
    clean["seconds_per_match_minute"] = clean["wall_seconds"] / (report["frames_processed"] * a.stride / 25.0 / 60.0)
    p = state.players
    clean["pct_inside_pitch"] = float(((p.x.abs() <= 54.5) & (p.y.abs() <= 36)).mean()) if len(p) else 0.0
    (out / "report.json").write_text(json.dumps(clean, indent=2, default=str))
    print(json.dumps(clean, indent=2, default=str))


if __name__ == "__main__":
    main()
