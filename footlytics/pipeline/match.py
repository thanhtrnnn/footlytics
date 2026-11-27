"""Chunked, resumable full-match (90+ min) broadcast video pipeline.

Processes a full broadcast recording (1st half, optional 2nd half, 50/60/25 fps) in bounded
time chunks (default 10 minutes), checkpointing each chunk (`tracks.parquet`, `done.json`, and
optional `obs.pkl`) so an interrupted overnight GPU run resumes without redoing finished chunks,
and `--replay-only` re-runs tracking, identity, possession, and tactical sequences across the
entire match on CPU in ~1-2 minutes.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..analytics.possession import infer_possession
from ..analytics.sequences import write_sequence_outputs
from ..analytics.tactics import infer_attacking_direction, normalise_direction
from ..geometry.clip_calib import ClipCalibrator, FrameCalib, static_overlay_mask
from ..state.schema import MatchMeta, MatchState, Team, TeamInfo
from .quality import compute_quality, write_quality
from .radar import (
    PipelineConfig,
    homography_mapper,
    load_observations,
    run_frames,
    run_observations,
    save_observations,
)


@dataclass(frozen=True)
class ChunkSpec:
    """One contiguous slice of source video belonging to a single match period."""

    chunk_id: str
    period: int
    start_frame: int
    end_frame: int          # exclusive source frame index
    start_s: float
    end_s: float


def parse_time_s(token: str) -> float:
    """Parse seconds (`'2700'`, `'45.5'`) or clock strings (`'45:00'`, `'1:05:30'`)."""
    s = token.strip()
    if not s:
        raise ValueError("empty time token")
    if ":" not in s:
        val = float(s)
        if val < 0 or not np.isfinite(val):
            raise ValueError(f"invalid non-negative seconds: {token!r}")
        return val
    parts = s.split(":")
    if len(parts) == 2:
        mm, ss = int(parts[0]), float(parts[1])
        val = mm * 60.0 + ss
    elif len(parts) == 3:
        hh, mm, ss = int(parts[0]), int(parts[1]), float(parts[2])
        val = hh * 3600.0 + mm * 60.0 + ss
    else:
        raise ValueError(f"invalid time format {token!r}; expected SS, MM:SS, or HH:MM:SS")
    if val < 0 or not np.isfinite(val):
        raise ValueError(f"invalid non-negative time: {token!r}")
    return val


def parse_window(spec: str) -> tuple[float, float]:
    """Parse `'start-end'` (or `'start..end'`, or `'start:end'` for plain seconds) into seconds."""
    raw = spec.strip()
    if ".." in raw:
        left, right = raw.split("..", 1)
    elif "-" in raw:
        left, right = raw.split("-", 1)
    elif raw.count(":") == 1:
        left, right = raw.split(":", 1)
    else:
        raise ValueError(
            f"invalid window {spec!r}; use 'START-END' (e.g. '05:00-52:00' or '300-3120')"
        )
    t0, t1 = parse_time_s(left), parse_time_s(right)
    if t1 <= t0:
        raise ValueError(f"window end ({t1}) must be greater than start ({t0}) in {spec!r}")
    return t0, t1


def choose_stride(src_fps: float, target_fps: float = 25.0) -> int:
    """Choose integer frame stride bringing `src_fps` closest to `target_fps` (>= 1)."""
    if src_fps <= 0 or target_fps <= 0:
        return 1
    return max(1, round(src_fps / target_fps))


def plan_match_chunks(
    total_frames: int,
    src_fps: float,
    period_windows: Sequence[tuple[int, float, float]] | None = None,
    chunk_minutes: float = 10.0,
) -> list[ChunkSpec]:
    """Partition period windows into bounded chunks aligned to source frames."""
    if total_frames <= 0 or src_fps <= 0:
        raise ValueError(f"invalid video metadata: total_frames={total_frames}, src_fps={src_fps}")
    if chunk_minutes <= 0:
        raise ValueError("chunk_minutes must be positive")

    total_s = total_frames / src_fps
    windows = (
        list(period_windows)
        if period_windows
        else [(1, 0.0, total_s)]
    )
    chunk_frames = max(1, round(chunk_minutes * 60.0 * src_fps))
    specs: list[ChunkSpec] = []
    for period, s0, s1 in windows:
        f0 = max(0, round(s0 * src_fps))
        f1 = min(total_frames, round(s1 * src_fps))
        if f1 <= f0:
            raise ValueError(
                f"period {period} window [{s0}, {s1}] produces empty frame range [{f0}, {f1})"
            )
        idx = 0
        cur = f0
        while cur < f1:
            nxt = min(f1, cur + chunk_frames)
            # Absorb a tiny trailing sliver (< 20% of a chunk) into the current chunk
            if f1 - nxt > 0 and (f1 - nxt) < max(int(0.2 * chunk_frames), 25):
                nxt = f1
            specs.append(
                ChunkSpec(
                    chunk_id=f"p{int(period)}_c{idx:02d}",
                    period=int(period),
                    start_frame=cur,
                    end_frame=nxt,
                    start_s=cur / src_fps,
                    end_s=nxt / src_fps,
                )
            )
            cur = nxt
            idx += 1
    return specs


def read_frame_range(
    video: Path, start_frame: int, end_frame: int, stride: int = 1
) -> Iterator[np.ndarray]:
    """Stream `[start_frame, end_frame)` from `video`, yielding every `stride`-th BGR frame."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    try:
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        pos = start_frame
        rel = 0
        while pos < end_frame:
            ok, frame = cap.read()
            if not ok:
                return
            if rel % stride == 0:
                yield frame
            pos += 1
            rel += 1
    finally:
        cap.release()


def sample_window_frames(
    video: Path, chunks: Sequence[ChunkSpec], k: int = 25
) -> list[np.ndarray]:
    """Sample up to `k` frames across the planned match chunks for `static_overlay_mask`."""
    if not chunks:
        return []
    indices: list[int] = []
    per = max(1, k // len(chunks))
    for ch in chunks:
        pts = np.linspace(ch.start_frame, max(ch.end_frame - 1, ch.start_frame), per + 2)[1:-1]
        indices.extend(round(float(x)) for x in pts)
    uniq = sorted(set(indices))
    if len(uniq) > k:
        pick = np.linspace(0, len(uniq) - 1, k).astype(int)
        uniq = [uniq[i] for i in pick]

    cap = cv2.VideoCapture(str(video))
    out: list[np.ndarray] = []
    try:
        for i in uniq:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if ok:
                out.append(f)
    finally:
        cap.release()
    return out


def calibrate_chunk(
    frames_iter: Iterator[np.ndarray],
    calibrator,
    mask: np.ndarray | None,
    every: int = 10,
    gate_px: float = 25.0,
) -> tuple[list[FrameCalib], dict]:
    """Per-frame calibration over one streamed chunk."""
    from ..bench import runner

    thr = runner._thresholds({})
    solve_kw = {"voting": False, "mode": "full", "ransac": 10.0, "refine_lines": True}
    errors = 0

    def fit(chunk, keys):
        nonlocal errors
        fits = calibrator.fit([f[:, :, ::-1] for f in chunk], which=keys, **thr, **solve_kw)
        errors += len(getattr(calibrator, "last_errors", {}))
        return fits

    cc = ClipCalibrator(fit, every=every, gate_px=gate_px, tracker_factory=runner.TRACKER)
    t0 = time.perf_counter()
    res = cc.run(frames_iter, mask=mask)
    stats = {
        "calib_seconds": time.perf_counter() - t0,
        "keyframes": cc.keyframes,
        "keyframes_accepted": cc.keyframes_accepted,
        "solver_errors": errors,
        "sources": {s: sum(r.source == s for r in res) for s in ("keyframe", "bridged", "none")},
    }
    return res, stats


def _chunk_home_dir(st: MatchState) -> int | None:
    dirs = infer_attacking_direction(st)
    for p in sorted({int(x) for x in st.tracks["period"].unique()}):
        if (p, Team.HOME.value) in dirs:
            return int(dirs[(p, Team.HOME.value)])
        if (p, Team.AWAY.value) in dirs:
            return -int(dirs[(p, Team.AWAY.value)])
    return None


def assemble_match_states(
    chunk_states: Sequence[tuple[ChunkSpec, MatchState]],
    eff_fps: float,
    match_id: str,
    home: str = "HOME",
    away: str = "AWAY",
    align_teams: bool = True,
    home_defends: str | None = None,
    provenance: dict | None = None,
) -> MatchState:
    """Concatenate per-chunk MatchStates into one monotonic, multi-period MatchState.

    When `align_teams=True`, resolves per-chunk k-means HOME/AWAY label flips so `HOME` attacks
    the same direction throughout each half and switches ends across odd/even periods.
    Pass `home_defends='left'` or `'right'` (Period 1 defending goal) to pin `HOME` to the
    physical club defending that side at 1st-half kickoff instead of k-means cluster 0.
    """
    if not chunk_states:
        raise ValueError("no chunk states to assemble")

    anchor_period: int | None = None
    anchor_dir: int | None = None
    if home_defends is not None:
        side = home_defends.strip().lower()
        if side not in ("left", "right"):
            raise ValueError(f"home_defends must be 'left' or 'right', got {home_defends!r}")
        anchor_period = 1
        anchor_dir = 1 if side == "left" else -1
    swap_map = {Team.HOME.value: Team.AWAY.value, Team.AWAY.value: Team.HOME.value}

    parts: list[pd.DataFrame] = []
    tid_offset = 0
    shot_offset = 0
    for spec, st in chunk_states:
        df = st.tracks.copy()
        if df.empty:
            continue
        if align_teams:
            h_dir = _chunk_home_dir(st)
            if h_dir is not None:
                if anchor_dir is None:
                    anchor_period, anchor_dir = spec.period, h_dir
                else:
                    same_end = (spec.period % 2) == (int(anchor_period) % 2)
                    expected = anchor_dir if same_end else -anchor_dir
                    if h_dir == -expected:
                        df["team"] = [swap_map.get(str(t), str(t)) for t in df["team"]]
        frame_base = round(spec.start_s * eff_fps)
        df["frame_idx"] = df["frame_idx"].astype("int32") + int(frame_base)
        df["timestamp"] = df["frame_idx"].astype(float) / float(eff_fps)
        df["period"] = np.int8(spec.period)

        tids = df["track_id"].to_numpy(dtype=np.int64)
        pos_mask = tids >= 0
        if pos_mask.any():
            max_tid = int(tids[pos_mask].max())
            tids[pos_mask] = tids[pos_mask] + tid_offset
            tid_offset += max_tid + 1
        df["track_id"] = tids.astype("int32")

        if "shot" in df.columns:
            shots = df["shot"].to_numpy(dtype=np.int64)
            max_shot = int(shots.max()) if len(shots) else 0
            df["shot"] = (shots + shot_offset).astype("int32")
            shot_offset += max_shot + 1

        parts.append(df)

    period_starts = {
        str(p): round(min(spec.start_s for spec, _ in chunk_states if spec.period == p), 3)
        for p in sorted({int(spec.period) for spec, _ in chunk_states})
    }
    meta = MatchMeta(
        match_id=match_id,
        fps=eff_fps,
        source_type="broadcast",
        home=TeamInfo(home, home[:3].upper()),
        away=TeamInfo(away, away[:3].upper()),
        provenance={**(provenance or {}), "period_start_s": period_starts},
    )
    merged = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["frame_idx", "track_id"], kind="stable")
        .reset_index(drop=True)
        if parts
        else chunk_states[0][1].tracks.iloc[:0].copy()
    )
    return MatchState(meta, merged)


CHUNK_FINGERPRINT_VERSION = 1
_GEOMETRY_KEYS = ("video", "period", "start_frame", "end_frame", "stride")


def _video_identity(video: Path) -> dict:
    """Name, size and a head+tail content hash: cheap, and stable across copies and touches."""
    size = video.stat().st_size
    h = hashlib.sha256()
    window = 1 << 20
    with open(video, "rb") as f:
        h.update(f.read(window))
        if size > 2 * window:
            f.seek(-window, 2)
            h.update(f.read(window))
    return {"name": video.name, "size": size, "sha256_head_tail": h.hexdigest()[:24]}


def _chunk_fingerprint(spec: ChunkSpec, video_id: dict, stride: int, **config) -> dict:
    """Everything that determines a chunk's `tracks.parquet`."""
    return {
        "version": CHUNK_FINGERPRINT_VERSION,
        "video": video_id,
        "period": spec.period,
        "start_frame": spec.start_frame,
        "end_frame": spec.end_frame,
        "stride": stride,
        **config,
    }


def _fingerprint_diff(old: dict | None, new: dict) -> list[str]:
    if not isinstance(old, dict):
        return ["no fingerprint"]
    keys = sorted(set(old) | set(new))
    return [f"{k}: {old.get(k)!r} -> {new.get(k)!r}" for k in keys if old.get(k) != new.get(k)]


def _provenance(fingerprints: Sequence[dict | None]) -> dict:
    """Detector and calibrator that produced the chunks, as recorded in their fingerprints."""
    seen: list[dict] = []
    for fp in fingerprints:
        if not isinstance(fp, dict) or "detector" not in fp:
            entry = {"detector": "unknown", "calibrator": "unknown"}
        else:
            entry = {"detector": fp["detector"], "calibrator": fp["calibrator"]}
        if entry not in seen:
            seen.append(entry)
    if len(seen) == 1:
        return seen[0]
    return {"detector": "mixed", "calibrator": "mixed", "variants": seen}


def run_match(
    video: Path,
    out: Path,
    paths=None,
    *,
    period_windows: Sequence[tuple[int, float, float]] | None = None,
    chunk_minutes: float = 10.0,
    target_fps: float = 25.0,
    stride: int | None = None,
    every: int = 10,
    gate_px: float = 25.0,
    home: str = "HOME",
    away: str = "AWAY",
    home_defends: str | None = None,
    people_detector: str = "yolo",
    people_weights_dir: str | None = None,
    ball_link: bool = True,
    ball_detector: str = "rtdetr",
    ball_weights_dir: str | None = None,
    calib_backend: str = "soccermaster",
    calib_weights_dir: str | None = None,
    tactical_profile: str | Path | object | None = None,
    save_obs: bool = True,
    replay_only: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    """Run or resume a full match in checkpointed chunks and write unified MatchState + sequences."""
    out = Path(out)
    chunks_root = out / "chunks"
    chunks_root.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    use_stride = stride if stride is not None else choose_stride(src_fps, target_fps)
    eff_fps = src_fps / use_stride
    specs = plan_match_chunks(
        total_frames, src_fps, period_windows=period_windows, chunk_minutes=chunk_minutes
    )
    log(
        f"[match] {video.name}: {total_frames:,} frames @ {src_fps:.2f} fps -> "
        f"stride {use_stride} ({eff_fps:.2f} Hz), {len(specs)} chunks"
    )

    pcfg = PipelineConfig(
        stride=1,  # read_frame_range already applies `use_stride` before calibration & detection
        verbose=False,
        montage=False,
        ball_link=ball_link,
        ball_link_break_cost=3.0 if ball_detector == "rtdetr" else 0.25,
    )

    calibrator = None
    det = None
    mask = None

    def ensure_models():
        nonlocal calibrator, det, mask
        if det is not None:
            return
        from ..bench import runner
        from ..bench.paths import BenchPaths

        bpaths = paths or BenchPaths.from_env()
        cal_cfg: dict = {"backend": calib_backend}
        if calib_weights_dir:
            cal_cfg["weights_dir"] = calib_weights_dir
        calibrator = runner._get_calibrator(bpaths, cal_cfg)
        det_cfg: dict = {}
        if people_detector == "rtdetr":
            det_cfg["people_detector"] = "rtdetr"
            if people_weights_dir:
                det_cfg["people_weights_dir"] = people_weights_dir
        elif people_detector != "yolo":
            raise ValueError(
                f"unknown people_detector {people_detector!r}; expected 'yolo' or 'rtdetr'"
            )
        if ball_detector == "rtdetr":
            det_cfg["ball_detector"] = "rtdetr"
            det_cfg["ball_conf"] = 0.15
            if ball_weights_dir:
                det_cfg["ball_weights_dir"] = ball_weights_dir
        elif ball_detector != "yolo":
            raise ValueError(f"unknown ball_detector {ball_detector!r}; expected 'yolo' or 'rtdetr'")
        det = runner._detector(bpaths, det_cfg)
        mask = static_overlay_mask(sample_window_frames(video, specs))

    video_id = _video_identity(video)
    model_config = {
        "every": every,
        "gate_px": gate_px,
        "ball_link": ball_link,
        "detector": {
            "people": {"backend": people_detector, "weights_dir": people_weights_dir},
            "ball": {"backend": ball_detector, "weights_dir": ball_weights_dir},
        },
        "calibrator": {"backend": calib_backend, "weights_dir": calib_weights_dir},
    }

    chunk_states: list[tuple[ChunkSpec, MatchState]] = []
    chunk_fingerprints: list[dict | None] = []
    all_calib_sources: list[str] = []
    calib_unknown_chunks: list[str] = []
    total_calib_s = 0.0
    total_pipe_s = 0.0
    replay_s = 0.0
    chunks_reused = 0
    t_match0 = time.perf_counter()

    for idx, spec in enumerate(specs, 1):
        cdir = chunks_root / spec.chunk_id
        done_path = cdir / "done.json"
        obs_path = cdir / "obs.pkl"

        fingerprint = _chunk_fingerprint(spec, video_id, use_stride, **model_config)
        info = json.loads(done_path.read_text()) if done_path.is_file() else {}
        old_fp = info.get("fingerprint")
        sources_path = cdir / "calib_sources.json"

        if replay_only:
            if not obs_path.is_file():
                raise FileNotFoundError(
                    f"replay_only requested, missing chunk observations: {obs_path}"
                )
            if isinstance(old_fp, dict):
                bad = [k for k in _GEOMETRY_KEYS if old_fp.get(k) != fingerprint[k]]
                if bad:
                    raise ValueError(
                        f"replay_only: saved observations for {spec.chunk_id} do not match "
                        f"this run's video/windows/stride ({', '.join(bad)} differ); "
                        "re-run without --replay-only"
                    )
            else:
                log(f"[match] {spec.chunk_id}: no fingerprint, cannot verify observations match this run")
            obs, ofps, has_ref = load_observations(obs_path)
            cmeta = MatchMeta(
                match_id=f"{video.stem}_{spec.chunk_id}",
                fps=ofps,
                source_type="broadcast",
                home=TeamInfo(home, home[:3].upper()),
                away=TeamInfo(away, away[:3].upper()),
            )
            t0 = time.perf_counter()
            st, rep = run_observations(obs, ofps, cmeta, pcfg, has_referee_class=has_ref)
            pipe_s = time.perf_counter() - t0
            cdir.mkdir(parents=True, exist_ok=True)
            st.save(cdir)
            replay_s += pipe_s
            total_calib_s += float(info.get("calib_seconds", 0.0))
            total_pipe_s += float(info.get("pipeline_seconds", 0.0))
            if sources_path.is_file():
                all_calib_sources.extend(json.loads(sources_path.read_text()))
            else:
                calib_unknown_chunks.append(spec.chunk_id)
            chunk_states.append((spec, st))
            chunk_fingerprints.append(old_fp)
            log(f"[match] [{idx}/{len(specs)}] replayed {spec.chunk_id} in {pipe_s:.1f}s")
            continue

        if done_path.is_file() and (cdir / "tracks.parquet").is_file():
            diff = _fingerprint_diff(old_fp, fingerprint)
            if not diff:
                st = MatchState.load(cdir)
                chunk_states.append((spec, st))
                chunk_fingerprints.append(old_fp)
                total_calib_s += float(info.get("calib_seconds", 0.0))
                total_pipe_s += float(info.get("pipeline_seconds", 0.0))
                if sources_path.is_file():
                    all_calib_sources.extend(json.loads(sources_path.read_text()))
                else:
                    calib_unknown_chunks.append(spec.chunk_id)
                chunks_reused += 1
                log(f"[match] [{idx}/{len(specs)}] reused checkpoint {spec.chunk_id}")
                continue
            reason = "has no fingerprint (older checkpoint)" if diff == ["no fingerprint"] else "; ".join(diff)
            log(f"[match] [{idx}/{len(specs)}] checkpoint {spec.chunk_id} is stale ({reason}); re-running")
            done_path.unlink(missing_ok=True)
            obs_path.unlink(missing_ok=True)

        ensure_models()
        cdir.mkdir(parents=True, exist_ok=True)
        log(
            f"[match] [{idx}/{len(specs)}] running {spec.chunk_id} "
            f"(period {spec.period}, {spec.start_s:.0f}s..{spec.end_s:.0f}s)"
        )
        calib, cstats = calibrate_chunk(
            read_frame_range(video, spec.start_frame, spec.end_frame, use_stride),
            calibrator,
            mask=mask,
            every=every,
            gate_px=gate_px,
        )
        cmeta = MatchMeta(
            match_id=f"{video.stem}_{spec.chunk_id}",
            fps=eff_fps,
            source_type="broadcast",
            home=TeamInfo(home, home[:3].upper()),
            away=TeamInfo(away, away[:3].upper()),
        )
        captured = [] if save_obs else None
        t0 = time.perf_counter()
        st, rep = run_frames(
            read_frame_range(video, spec.start_frame, spec.end_frame, use_stride),
            eff_fps,
            homography_mapper([c.H_img_to_pitch for c in calib]),
            det,
            cmeta,
            pcfg,
            source=f"{video}:{spec.chunk_id}",
            capture=captured,
        )
        pipe_s = time.perf_counter() - t0
        if save_obs and captured is not None:
            save_observations(obs_path, captured, eff_fps, "referee" in set(det.roles.values()))

        st.save(cdir)
        sources = [c.source for c in calib]
        (cdir / "calib_sources.json").write_text(json.dumps(sources))
        clean_rep = {k: v for k, v in rep.items() if not k.startswith("_")}
        clean_rep.update(
            calib_seconds=cstats["calib_seconds"],
            pipeline_seconds=pipe_s,
            calibration=cstats,
        )
        (cdir / "report.json").write_text(json.dumps(clean_rep, indent=2, default=str))
        done_info = {
            "chunk_id": spec.chunk_id,
            "period": spec.period,
            "start_frame": spec.start_frame,
            "end_frame": spec.end_frame,
            "stride": use_stride,
            "calib_seconds": cstats["calib_seconds"],
            "pipeline_seconds": pipe_s,
            "fingerprint": fingerprint,
        }
        tmp_done = cdir / "done.json.tmp"
        tmp_done.write_text(json.dumps(done_info, indent=2))
        tmp_done.replace(done_path)

        total_calib_s += cstats["calib_seconds"]
        total_pipe_s += pipe_s
        all_calib_sources.extend(sources)
        chunk_states.append((spec, st))
        chunk_fingerprints.append(fingerprint)
        log(
            f"[match] [{idx}/{len(specs)}] finished {spec.chunk_id} "
            f"(calib {cstats['calib_seconds']:.0f}s + pipe {pipe_s:.0f}s)"
        )

    provenance = _provenance(chunk_fingerprints)
    full_state = assemble_match_states(
        chunk_states,
        eff_fps=eff_fps,
        match_id=video.stem,
        home=home,
        away=away,
        home_defends=home_defends,
        provenance=provenance,
    )
    full_state.save(out)
    if all_calib_sources:
        (out / "calib_sources.json").write_text(json.dumps(all_calib_sources))

    wall_s = time.perf_counter() - t_match0
    total_compute_s = total_calib_s + total_pipe_s
    frames_proc = sum(
        len(range(s.start_frame, s.end_frame, use_stride)) for s in specs
    )
    if calib_unknown_chunks or not all_calib_sources:
        log(
            "[match] WARNING: no calibration record for "
            f"{', '.join(calib_unknown_chunks) or 'any chunk'}; calibration figures reported as unknown"
        )
        n_uncal = frames_cal = calib_rate = None
    else:
        n_uncal = sum(src == "none" for src in all_calib_sources)
        frames_cal = frames_proc - n_uncal
        calib_rate = frames_cal / max(frames_proc, 1)
    match_report = {
        "frames_processed": frames_proc,
        "frames_calibrated": frames_cal,
        "frames_uncalibrated": n_uncal,
        "calib_success_rate": calib_rate,
        "source_fps": src_fps,
        "stride": use_stride,
        "effective_fps": eff_fps,
        "chunks_total": len(specs),
        "chunks_reused": chunks_reused,
        "replayed": replay_only,
        "replay_seconds": replay_s,
        "provenance": provenance,
        "calib_seconds": total_calib_s,
        "pipeline_seconds": total_pipe_s,
        "compute_seconds": total_compute_s,
        "wall_seconds": wall_s,
    }
    (out / "report.json").write_text(json.dumps(match_report, indent=2, default=str))
    q = compute_quality(full_state, {**match_report, "seconds": total_compute_s})
    q.update(
        chunks_total=len(specs),
        chunks_reused=chunks_reused,
        provenance=provenance,
        compute_seconds=total_compute_s,
        wall_seconds=wall_s,
    )
    write_quality(q, out)

    norm = normalise_direction(full_state)
    poss = infer_possession(norm)
    seq_kwargs: dict = {}
    if tactical_profile is not None:
        from ..analytics.profiles import TacticalProfile, load_profile, save_profile

        prof = (
            tactical_profile
            if isinstance(tactical_profile, TacticalProfile)
            else load_profile(tactical_profile)
        )
        seq_kwargs = prof.to_detect_kwargs()
        save_profile(prof, out / "tactical_profile.yaml")
    seqs = write_sequence_outputs(norm, poss, out, **seq_kwargs)
    poss.to_csv(out / "possession.csv", index=False)
    log(
        f"[match] done -> {out} ({len(specs)} chunks, {chunks_reused} reused, "
        f"{len(seqs)} sequences, {total_compute_s:.1f}s compute)"
    )
    return q
