"""Run the whole pipeline on one real broadcast video, calibrating every frame.

The path the GSR benchmark task proves on GPU: SoccerMaster keypoints/lines on every
`every`-th frame, `AnchoredCamera` bridging between them, frames that fail calibration left
as gaps. Then detection, tracking, kit clustering and goalkeeper roles, possession and
tactical sequences, written where `footlytics review serve` can read them.

Two passes over the video (calibrate, then detect) so no frame list is held in memory.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ..analytics.possession import infer_possession
from ..analytics.sequences import detect_sequences
from ..analytics.tactics import normalise_direction
from ..geometry.clip_calib import ClipCalibrator, FrameCalib, static_overlay_mask
from .quality import compute_quality, write_quality
from .radar import PipelineConfig, homography_mapper, run_frames
from ..state.schema import MatchMeta, TeamInfo


def read_frames(video: Path) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                return
            yield f
    finally:
        cap.release()


def sample_frames(video: Path, k: int = 25) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in sorted(set(np.linspace(0, max(n - 1, 0), min(k, max(n, 1))).astype(int).tolist())):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            out.append(f)
    cap.release()
    return out


def calibrate_video(video: Path, calibrator, every: int = 10, gate_px: float = 25.0,
                    log=print) -> tuple[list[FrameCalib], dict]:
    """Per-frame calibration of the whole video, with the bench's settings (`bench/runner.py`)."""
    from ..bench import runner

    cfg: dict = {}
    thr = runner._thresholds(cfg)
    solve_kw = {"voting": False, "mode": "full", "ransac": 10.0, "refine_lines": True}
    errors = 0

    def fit(chunk, keys):
        nonlocal errors
        fits = calibrator.fit([f[:, :, ::-1] for f in chunk], which=keys, **thr, **solve_kw)
        errors += len(getattr(calibrator, "last_errors", {}))
        return fits

    mask = static_overlay_mask(sample_frames(video))
    cc = ClipCalibrator(fit, every=every, gate_px=gate_px, tracker_factory=runner.TRACKER)
    t0 = time.perf_counter()
    res = cc.run(read_frames(video), mask=mask)
    stats = {"calib_seconds": time.perf_counter() - t0, "keyframes": cc.keyframes,
             "keyframes_accepted": cc.keyframes_accepted, "solver_errors": errors,
             "sources": {s: sum(r.source == s for r in res) for s in ("keyframe", "bridged", "none")}}
    log(f"[calibration] {stats}")
    return res, stats


def run_video(video: Path, out: Path, paths, every: int = 10, gate_px: float = 25.0,
              home: str = "HOME", away: str = "AWAY", log=print) -> dict:
    """Calibrate, detect, track, then write tracks, quality and sequences into `out`."""
    from ..bench import runner

    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    calibrator = runner._calibrator(paths)
    det = runner._detector(paths, {})
    calib, cstats = calibrate_video(video, calibrator, every, gate_px, log)

    t0 = time.perf_counter()
    meta = MatchMeta(match_id=video.stem, fps=fps, source_type="broadcast",
                     home=TeamInfo(home, home[:3].upper()), away=TeamInfo(away, away[:3].upper()))
    state, report = run_frames(read_frames(video), fps,
                               homography_mapper([c.H_img_to_pitch for c in calib]), det, meta,
                               PipelineConfig(verbose=False, montage=False), source=str(video))
    run_s = time.perf_counter() - t0

    state.save(out)
    clean = {k: v for k, v in report.items() if not k.startswith("_")}
    seconds = cstats["calib_seconds"] + run_s
    clean.update(wall_seconds=seconds, calib_seconds=cstats["calib_seconds"], pipeline_seconds=run_s)
    (out / "report.json").write_text(json.dumps(clean, indent=2, default=str))
    q = compute_quality(state, {**clean, "seconds": seconds})
    q["calibration"] = cstats
    write_quality(q, out)
    (out / "calib_sources.json").write_text(json.dumps([c.source for c in calib]))

    norm = normalise_direction(state)
    poss = infer_possession(norm)
    seqs = detect_sequences(norm, poss)
    seqs.to_csv(out / "sequences.csv", index=False)
    poss.to_csv(out / "possession.csv", index=False)
    log(f"[done] {seconds:.0f} s, {len(seqs)} sequences")
    return q
