#!/usr/bin/env python3
"""Score a MatchState against Alfheim's ZXY body-sensor ground truth.

This is the measurement that matters. Everything else in this repo is checked
against synthetic data I wrote myself, which proves the code does what I
intended but says nothing about whether it works on football. ZXY gives real
positions in real metres, at 20 Hz, for real players.

Alignment
---------
Video chunk filenames carry the wall-clock time the chunk starts, and every ZXY
row carries a wall-clock timestamp. So frame k sits at

    t = video_start + k / fps

and we take the nearest sensor reading within a tolerance. That is what makes
frame-accurate scoring possible without any manual syncing.

What is measured
----------------
`position error`   Distance in metres between each matched track and its
                   ground-truth player. This is the number that tells you
                   whether the calibration is good enough for offside (needs
                   <0.5 m) or only for heatmaps (2 m is fine).
`recall`           Fraction of sensor-tracked players we actually found. Low
                   recall means the detector, not the tracker.
`id consistency`   Whether one ground-truth player keeps mapping to one
                   track_id. This is the honest version of the ID-switch count.

What cannot be measured here
----------------------------
Only Tromsø players wore sensors, so roughly half the players on the pitch have
no ground truth. Unmatched detections are therefore NOT false positives -- they
are mostly the opposition. Precision is unmeasurable on this dataset; do not
report it.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def align(state_tracks: pd.DataFrame, zxy: pd.DataFrame, video_start: datetime,
          fps: float, tol_s: float = 0.05) -> pd.DataFrame:
    """Attach the nearest ground-truth timestamp to each processed frame."""
    frames = np.sort(state_tracks["frame_idx"].unique())
    want = pd.DataFrame({
        "frame_idx": frames,
        "ts": [video_start + timedelta(seconds=float(f) / fps) for f in frames],
    })
    z = zxy.sort_values("ts")
    merged = pd.merge_asof(want.sort_values("ts"), z[["ts"]].drop_duplicates(),
                           on="ts", direction="nearest",
                           tolerance=pd.Timedelta(seconds=tol_s))
    return merged.dropna(subset=["ts"])


def score(state_tracks: pd.DataFrame, zxy: pd.DataFrame, video_start: datetime,
          fps: float, max_match_dist: float = 5.0, tol_s: float = 0.05) -> dict:
    """Hungarian-match tracks to sensor players per frame, then summarise."""
    z = zxy.copy()
    z["ts"] = pd.to_datetime(z["ts"])
    z = z.sort_values("ts")

    errors: list[float] = []
    recalls: list[float] = []
    gt_to_track: dict[int, int] = {}
    switches = 0
    matched_pairs = 0
    frames_scored = 0

    for frame_idx, g in state_tracks.groupby("frame_idx"):
        t = video_start + timedelta(seconds=float(frame_idx) / fps)
        lo, hi = t - timedelta(seconds=tol_s), t + timedelta(seconds=tol_s)
        window = z[(z["ts"] >= lo) & (z["ts"] <= hi)]
        if window.empty:
            continue
        # one reading per player: the one closest in time
        window = (window.assign(_dt=(window["ts"] - t).abs())
                        .sort_values("_dt").drop_duplicates("tag_id"))
        gt_xy = window[["x_pitch", "y_pitch"]].to_numpy()
        gt_ids = window["tag_id"].to_numpy().astype(int)
        det_xy = g[["x", "y"]].to_numpy()
        det_ids = g["track_id"].to_numpy().astype(int)
        if not len(gt_xy) or not len(det_xy):
            continue

        frames_scored += 1
        d = np.linalg.norm(det_xy[:, None, :] - gt_xy[None, :, :], axis=-1)
        cost = np.where(d <= max_match_dist, d, 1e6)
        ri, ci = linear_sum_assignment(cost)
        n_matched = 0
        for a, b in zip(ri, ci):
            if cost[a, b] >= 1e6:
                continue
            n_matched += 1
            matched_pairs += 1
            errors.append(float(d[a, b]))
            gid, tid = int(gt_ids[b]), int(det_ids[a])
            if gid in gt_to_track and gt_to_track[gid] != tid:
                switches += 1
            gt_to_track[gid] = tid
        recalls.append(n_matched / len(gt_xy))

    if not errors:
        return {"error": "nothing matched -- check video_start, fps, or the calibration"}

    e = np.array(errors)
    return {
        "frames_scored": frames_scored,
        "matched_pairs": matched_pairs,
        "recall_mean": float(np.mean(recalls)),
        "err_median_m": float(np.median(e)),
        "err_mean_m": float(e.mean()),
        "err_p90_m": float(np.percentile(e, 90)),
        "err_p99_m": float(np.percentile(e, 99)),
        "id_switches": switches,
        "gt_players_seen": len(gt_to_track),
    }


def verdict(s: dict) -> str:
    if "error" in s:
        return s["error"]
    m = s["err_median_m"]
    if m < 0.5:
        band = "good enough for line height and offside-adjacent work"
    elif m < 1.5:
        band = "good enough for formation, heatmaps and pressing; NOT for offside"
    elif m < 3.0:
        band = "only good enough for coarse zonal summaries"
    else:
        band = "not usable -- recalibrate before trusting anything downstream"
    r = ("detector is finding nearly everyone" if s["recall_mean"] > 0.9
         else "detector is missing players -- fix detection before tuning the tracker")
    return f"median error {m:.2f} m: {band}. Recall {s['recall_mean']:.0%}: {r}"


def _self_test() -> int:
    """Validate the scoring logic itself against a known-error synthetic case."""
    print("self-test: injecting a known 0.80 m bias and 20% dropout\n")
    rng = np.random.default_rng(0)
    start = datetime(2013, 11, 3, 18, 1, 14)
    fps, n_frames, n_players = 30.0, 300, 11

    zrows, trows = [], []
    pos = np.column_stack([rng.uniform(-40, 40, n_players),
                           rng.uniform(-25, 25, n_players)])
    vel = rng.normal(0, 1.2, (n_players, 2))
    for f in range(n_frames):
        t = start + timedelta(seconds=f / fps)
        pos += vel / fps
        for i in range(n_players):
            zrows.append({"ts": t, "tag_id": i + 1,
                          "x_pitch": pos[i, 0], "y_pitch": pos[i, 1]})
            if rng.random() < 0.20:            # 20% of players missed
                continue
            off = rng.normal(0, 1, 2); off /= np.linalg.norm(off)
            trows.append({"frame_idx": f, "track_id": 500 + i,
                          "x": pos[i, 0] + off[0] * 0.80,
                          "y": pos[i, 1] + off[1] * 0.80})

    s = score(pd.DataFrame(trows), pd.DataFrame(zrows), start, fps)
    for k, v in s.items():
        print(f"  {k:18s} {v}")
    print("\n  " + verdict(s))
    ok = abs(s["err_median_m"] - 0.80) < 0.05 and abs(s["recall_mean"] - 0.80) < 0.03
    print(f"\nself-test {'PASSED' if ok else 'FAILED'} "
          f"(expected ~0.80 m error, ~80% recall)")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", help="directory saved by MatchState.save()")
    ap.add_argument("--zxy", help="parquet written by fetch_alfheim.py")
    ap.add_argument("--video-start", help='e.g. "2013-11-03 18:01:14.248366"')
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test or not a.state:
        return _self_test()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from footlytics.state.schema import MatchState

    st = MatchState.load(a.state)
    z = pd.read_parquet(a.zxy)
    start = datetime.fromisoformat(a.video_start)
    s = score(st.players, z, start, a.fps)
    for k, v in s.items():
        print(f"{k:18s} {v}")
    print("\n" + verdict(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
