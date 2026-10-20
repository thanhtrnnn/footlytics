"""Run the detector and the calibration once on GSR clips and pickle what the ball selector sees.

Per frame: every detection (x, y, w, h, conf, is_ball) and its pitch position (None when the frame
has no calibration), plus the GT ball box in the image. `scripts/replay_ball_select.py` then runs
the selection offline in seconds, so the rules can be tuned without the GPU.

Run on the bench host (needs CUDA and the weights). Which clips is the caller's job: tune on
the train clips the detector held out (`--split train --clips SNGS-165 ...`), score once on valid.

    CUDA_MPS_PIPE_DIRECTORY=/tmp/iec-nomps python scripts/capture_ball_candidates.py \
        --split train --clips SNGS-165 SNGS-166 ... --out cache/ball_cands_train.pkl
"""
from __future__ import annotations

import argparse
import pickle
import time
import zipfile
from pathlib import Path

import numpy as np

from footlytics.bench import runner
from footlytics.bench.adapters.gsr import gt_to_state
from footlytics.bench.paths import BenchPaths
from footlytics.bench.runner import (TaskContext, _check_alignment, _clip_calibration, _detector,
                                     _frame_names, _imread, _labels)
from footlytics.pipeline.radar import homography_mapper
from footlytics.state.schema import Role


def capture_clip(ctx, z, det, split: str, seq: str, every: int) -> dict:
    labels = _labels(ctx.paths, split, seq)
    names = _frame_names(z, split, seq, ctx.paths)
    res, calib_s, _ = _clip_calibration(ctx, z, split, seq, {}, every, 25.0)
    _check_alignment(seq, names, len(res))
    mapper = homography_mapper([fc.H_img_to_pitch for fc in res])
    frames = {}
    for i, n in enumerate(names):
        bgr = _imread(z.read(n))
        dets = det.detect(bgr[:, :, ::-1])
        xy = mapper(i, bgr, dets)
        is_ball = np.array([det.role_of(c) == Role.BALL.value for c in dets[:, 5]], bool)
        frames[i] = (dets[:, :5].astype(np.float32), is_ball,
                     None if xy is None else np.asarray(xy, np.float32))
    gt = gt_to_state(labels).tracks
    b = gt[gt["role"].astype(str) == "ball"]
    b = b[np.isfinite(b["bbox_x"].astype(float))]
    gt_ball = {int(r.frame_idx): (float(r.bbox_x), float(r.bbox_y), float(r.bbox_w), float(r.bbox_h))
               for r in b.itertuples(index=False)}
    return {"n_frames": len(names), "fps": 25.0, "frames": frames, "gt_ball": gt_ball,
            "calib_ok": [fc.H_img_to_pitch is not None for fc in res], "calib_s": calib_s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "valid"])
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=10, help="calibration keyframe spacing")
    ap.add_argument("--ball-conf", type=float, default=0.05)
    ap.add_argument("--max-balls", type=int, default=15)
    a = ap.parse_args()
    paths = BenchPaths.from_env()
    ctx = TaskContext(paths, Path(a.out).parent, lambda m: print(m, flush=True))
    det = _detector(paths, {"ball_detector": "rtdetr", "ball_conf": a.ball_conf,
                            "max_balls": a.max_balls})
    out = {"split": a.split, "ball_conf": a.ball_conf, "max_balls": a.max_balls, "clips": {}}
    with zipfile.ZipFile(paths.gsr_zip(a.split)) as z:
        for seq in a.clips:
            t0 = time.time()
            out["clips"][seq] = capture_clip(ctx, z, det, a.split, seq, a.every)
            print(f"{seq}: {out['clips'][seq]['n_frames']} frames, "
                  f"{len(out['clips'][seq]['gt_ball'])} with GT ball, {time.time() - t0:.0f} s",
                  flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(out, f)
    print("saved", a.out, Path(a.out).stat().st_size // 1024, "KiB")


if __name__ == "__main__":
    main()
