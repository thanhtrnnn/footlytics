"""Before / after flicker measurements on saved match states, no video, no GPU.

    python scripts/stability_report.py DATA_OUT_DIR match1 [match2 ...] [--frames LO HI] [--retrack]

Each match is `DATA_OUT_DIR/<match>/{tracks.parquet,meta.json}` as `MatchState.save` wrote it.
Stages: `saved` (as written), `camera` (homographies recovered from the rows, smoothed across
keyframes, feet re-projected), `camera+state` (then the per-id RTS state with coasting).
`--retrack` also reruns the tracker over the rows of `saved` and `camera` (geometry only, no
appearance), which shows the effect of the camera step on id births without the video.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from footlytics.pipeline import stability as st
from footlytics.perception.state_smooth import StateSmoothConfig, smooth_player_states

ROWS = [
    ("steps > 12 m/s", "step_gt12", "{:.2%}"), ("steps > 20 m/s", "step_gt20", "{:.2%}"),
    ("accel p99 m/s^2", "step_accel_p99", "{:.0f}"),
    ("common-mode > 0.5 m, keyframe frames", "cm_key_gt05", "{:.2%}"),
    ("common-mode > 0.5 m, other frames", "cm_other_gt05", "{:.2%}"),
    ("common-mode > 0.5 m, all frames", "cm_all_gt05", "{:.2%}"),
    ("common-mode median step, keyframe / other (m)", None, None),
    ("still sigma median (m)", "still_median_m", "{:.3f}"), ("still sigma p90 (m)", "still_p90_m", "{:.3f}"),
    ("ids born per person-minute", "id_births_per_person_min", "{:.2f}"),
    ("median id life (s)", "id_median_life_s", "{:.1f}"),
]


def load(root: Path, match: str, lo, hi):
    t = pd.read_parquet(root / match / "tracks.parquet")
    for c in ("role", "team"):
        t[c] = t[c].astype(str)
    fps = json.loads((root / match / "meta.json").read_text())["fps"]
    if lo is not None:
        t = t[(t.frame_idx >= lo) & (t.frame_idx < hi)]
    return t.reset_index(drop=True), float(fps)


def table(cols: dict[str, dict]) -> str:
    names = list(cols)
    out = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label, key, fmt in ROWS:
        if key is None:
            vals = [f"{m['cm_key_median_m']:.3f} / {m['cm_other_median_m']:.3f}" for m in cols.values()]
        else:
            vals = [fmt.format(m[key]) for m in cols.values()]
        out.append(f"| {label} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("matches", nargs="*")
    ap.add_argument("--synthetic", action="store_true",
                    help="also print the over-smoothing check on a synthetic sprint and cut")
    ap.add_argument("--frames", nargs=2, type=int)
    ap.add_argument("--retrack", action="store_true")
    ap.add_argument("--every", type=int, default=10, help="keyframe cadence of the clip")
    a = ap.parse_args()
    lo, hi = a.frames if a.frames else (None, None)
    if a.synthetic:
        print("\n### over-smoothing check (30 fps; sprint 7 m/s^2 to 8 m/s, 90 degree cut at 6 m/s)")
        print("| image row | manoeuvre | peak error (m) | lag (s) | peak speed kept | rms err smoothed / raw (m) |")
        print("|---|---|---|---|---|---|")
        for row in (0.35, 0.6, 0.9):
            r = st.synthetic_response(30.0, row)
            for name in ("sprint", "cut"):
                m = r[name]
                print(f"| {row:.2f}H (sigma {r['meas_sigma_m']:.2f} m) | {name} | {m['noise_free_peak_err_m']:.2f} | "
                      f"{m['noise_free_lag_s']:.3f} | {m['noise_free_peak_speed_kept']:.0%} | "
                      f"{m['noisy_rms_err_m']:.2f} / {m['raw_rms_err_m']:.2f} |")
    for m in a.matches:
        t0 = time.time()
        t, fps = load(a.root, m, lo, hi)
        saved = st.measure(t, fps)
        cam, info = st.replay_camera(t, every=a.every)
        camera = st.measure(cam, fps)
        full, rep = smooth_player_states(cam, fps, StateSmoothConfig())
        final = st.measure(full, fps)
        cols = {"saved": saved, "camera": camera, "camera + state": final}
        print(f"\n### {m}  fps={fps:g}  rows={len(t):,}  frames {lo}..{hi}  ({time.time() - t0:.0f} s)")
        print(f"camera replay: {info}")
        print(f"state smoothing: {rep}")
        print(table(cols))
        if a.retrack:
            for name, tt in (("saved positions", t), ("camera-smoothed positions", cam)):
                r = st.retrack(tt, fps)
                s = st.id_stats(r, fps)
                print(f"retrack on {name}: ids {s['ids']:,}  births/person-min "
                      f"{s['births_per_person_min']:.2f}  median life {s['median_life_s']:.1f} s")


if __name__ == "__main__":
    main()
