"""Re-run tracking, stitching and identity on a captured clip, on the CPU, and print how well
identity holds inside each live-wide segment.

    footlytics run-video clip.mp4 --out runs/x --capture clip.pkl      # once, on the GPU host
    python scripts/replay_real.py clip.pkl                              # seconds, anywhere
    python scripts/replay_real.py clip.pkl -t max_age=60 -p stitch=false

`-t k=v` overrides a `TrackerConfig` field, `-p k=v` a `PipelineConfig` field. `--save DIR`
writes the resulting state (tracks.parquet) for `review serve`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from footlytics.perception.track import TrackerConfig
from footlytics.pipeline.quality import compute_quality
from footlytics.pipeline.radar import PipelineConfig, load_observations, run_observations
from footlytics.state.schema import MatchMeta, Role


def _kv(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = json.loads(v) if v[:1] in "0123456789-[{\"" or v in ("true", "false", "null") else v
    return out


def replay(pkl: str | Path, tracker: dict | None = None, pipeline: dict | None = None):
    obs, fps, has_ref = load_observations(pkl)
    pcfg = PipelineConfig(**{"verbose": False, "montage": False, **(pipeline or {})})
    tcfg = TrackerConfig(fps=fps / pcfg.stride, **(tracker or {}))
    meta = MatchMeta(match_id=Path(pkl).stem, fps=fps, source_type="broadcast")
    t0 = time.time()
    state, report = run_observations(obs, fps, meta, pcfg, tcfg, has_referee_class=has_ref)
    q = compute_quality(state, {**{k: v for k, v in report.items() if not k.startswith("_")},
                                "seconds": time.time() - t0})
    return state, report, q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("-t", "--tracker", action="append")
    ap.add_argument("-p", "--pipeline", action="append")
    ap.add_argument("--save", type=Path)
    a = ap.parse_args()
    state, report, q = replay(a.pkl, _kv(a.tracker), _kv(a.pipeline))
    tr = state.tracks
    pl = tr[tr["role"].isin([Role.PLAYER.value, Role.GOALKEEPER.value])]
    keys = ("unique_track_ids", "tracklets_before_stitch", "tracklets_after_stitch",
            "median_players_per_frame", "calib_success_rate")
    print({k: (round(q[k], 3) if isinstance(q[k], float) else q[k]) for k in keys if k in q})
    print("tracks created:", report["tracks_created"], " goalkeepers:", report["goalkeepers"],
          " player/keeper ids:", pl["track_id"].nunique())
    for k, v in q.items():
        if k.startswith("seg_"):
            print(f"  {k[4:]:28s} {v:.2f}" if isinstance(v, float) else f"  {k[4:]:28s} {v}")
    if a.save:
        state.save(a.save)


if __name__ == "__main__":
    main()
