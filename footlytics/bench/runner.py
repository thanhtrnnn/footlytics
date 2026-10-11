"""Suite loop on the host: tasks -> adapters -> evaluators -> one result row.

A suite is a YAML list of typed tasks (`suites/<name>.yaml`); each kind is a function
registered in `KINDS`. A clip that fails is recorded with its reason and the task goes on; a
task that fails is recorded and the suite goes on. Any failure makes the run `partial`.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import traceback
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml

from . import datasets
from ..geometry.calib_nn import DEFAULT_KP_THRESHOLD, DEFAULT_LINE_THRESHOLD, solve_many
from ..geometry.camera_motion import AnchoredCamera
from ..geometry.clip_calib import ClipCalibrator, static_overlay_mask
from ..geometry.sn_camera import scale_camera
from .adapters.calib import gsr_line_frames, write_camera
from .adapters.gsr import check_version, gt_to_state, state_to_predictions
from .evaluators import run_gsr, run_sncalib
from .host import ResourceError, contention, require_disk, require_vram, torch_info
from .metrics import median_reprojection_px
from .paths import BenchPaths

SUITES_DIR = Path(__file__).parent / "suites"
DONE = ("complete", "partial", "failed")
#: Room a run needs for its predictions and evaluator output (the disk is shared).
RUN_DISK_BYTES = 2_000_000_000
EVALUATORS = ("sn-trackeval", "sn-calibration")
#: The official calibration evaluator does not rescale a predicted camera: it reads the image
#: size off the camera's principal point and scores the GT at 960x540, so every camera is
#: written at this width whatever the frames' own size.
EVAL_WIDTH = 960
KINDS: dict[str, Callable[[dict, "TaskContext"], dict]] = {}


@dataclass
class TaskContext:
    paths: BenchPaths
    work: Path
    log: Callable[[str], None]


def kind(name: str):
    def register(fn):
        KINDS[name] = fn
        return fn
    return register


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    print(f"[bench {_now()}] {msg}", flush=True)


def load_suite(name: str) -> dict:
    return yaml.safe_load((SUITES_DIR / f"{name}.yaml").read_text())


def write_status(run_dir: Path, state: str, **extra) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / "status.json.tmp"
    tmp.write_text(json.dumps({"state": state, "updated": _now(), **extra}))
    tmp.replace(run_dir / "status.json")


def evaluate_check(check: dict | None, aggregate: dict) -> dict | None:
    if not check:
        return None
    value = aggregate.get(check["metric"])
    out = {**check, "value": value}
    if value is None:
        return {**out, "passed": False}
    if "min" in check:
        return {**out, "passed": value >= check["min"]}
    delta = value - check["expected"]
    return {**out, "delta": round(delta, 3), "passed": abs(delta) <= check["tolerance"]}


def run_status(tasks: dict) -> str:
    states = [t["status"] for t in tasks.values()]
    if states and all(s == "complete" for s in states):
        return "complete"
    if states and all(s == "failed" for s in states):
        return "failed"
    return "partial"


def _commit(repo: Path) -> str | None:
    if not (repo / ".git").exists():
        return None
    p = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                       text=True)
    return p.stdout.strip() or None


def preflight(paths: BenchPaths) -> dict:
    """Refuse to start below the disk floor, or on a torch other than the one setup pinned
    (it lives in a shared ~/.local). Records the evaluator commits: they are part of what a
    number means."""
    require_disk(paths.root, RUN_DISK_BYTES)
    info: dict = {"evaluators": {name: _commit(paths.ext / name) for name in EVALUATORS}}
    if not paths.torch_pin.exists():
        return info
    info.update(torch_info())
    pinned = paths.torch_pin.read_text().strip()
    if info["torch"] != pinned or not info["cuda"]:
        raise ResourceError(f"torch changed: pinned {pinned} with CUDA, found {info['torch']} "
                            f"cuda={info['cuda']}; rerun `footlytics bench setup`")
    return info


def prune_runs(paths: BenchPaths, keep: int = 5) -> None:
    """Keep the bulky work/ of the last `keep` runs; rows and logs stay."""
    works = sorted(paths.runs.glob("*/work"), key=lambda p: p.parent.name)
    for w in works[:-keep] if keep else works:
        shutil.rmtree(w, ignore_errors=True)


def run_suite(suite: str, run_id: str, paths: BenchPaths, git_sha: str, host_label: str,
              suite_spec: dict | None = None) -> dict:
    run_dir = paths.run_dir(run_id)
    write_status(run_dir, "running")
    row = {"run_id": run_id, "suite": suite, "git_sha": git_sha, "host": host_label,
           "started_at": _now(), "contention": {"start": contention()}, "tasks": {}}
    try:
        row["env"] = preflight(paths)
        spec = suite_spec or load_suite(suite)
    except Exception as e:  # noqa: BLE001 - recorded, not raised: the poller must see it
        row.update(status="failed", error=f"{type(e).__name__}: {e}", finished_at=_now())
        return _finish(run_dir, row, paths)
    for task in spec["tasks"]:
        tid = task["id"]
        write_status(run_dir, "running", task=tid)
        _log(f"task {tid} ({task['kind']})")
        t0 = time.time()
        try:
            out = KINDS[task["kind"]](task, TaskContext(paths, run_dir / "work" / tid, _log))
            res = {"kind": task["kind"], **out,
                   "status": "partial" if out.get("failed") else "complete"}
        except Exception as e:  # noqa: BLE001
            res = {"kind": task["kind"], "status": "failed", "error": f"{type(e).__name__}: {e}",
                   "traceback": traceback.format_exc()[-2000:]}
            _log(f"task {tid} failed: {res['error']}")
        res["seconds"] = round(time.time() - t0, 1)
        res["check"] = evaluate_check(task.get("check"), res.get("aggregate") or {})
        row["tasks"][tid] = res
    row["status"] = run_status(row["tasks"])
    row["finished_at"] = _now()
    return _finish(run_dir, row, paths)


def _finish(run_dir: Path, row: dict, paths: BenchPaths) -> dict:
    row["contention"]["end"] = contention()
    (run_dir / "result.json").write_text(json.dumps(row, indent=2, default=str))
    write_status(run_dir, row["status"])
    prune_runs(paths)
    _log(f"run {row['run_id']}: {row['status']}")
    return row


# ---------------------------------------------------------------- task kinds

def _resolve(paths: BenchPaths, split: str, wanted) -> list[str]:
    have = datasets.clips(paths, split)
    if not have:
        raise FileNotFoundError(f"no GSR {split} labels under {paths.gsr_labels / split}; "
                                f"run `footlytics bench fetch gsr-{split}`")
    return have if wanted == "all" else list(wanted)


def _labels(paths: BenchPaths, split: str, seq: str) -> dict:
    """One clip's ground truth, parsed and version-checked (raises for that clip alone)."""
    labels = json.loads((paths.gsr_labels / split / seq / "Labels-GameState.json").read_text())
    check_version(labels)
    return labels


def _score(ctx: TaskContext, split: str, tracker: str, seqs: list[str]) -> dict:
    """GS-HOTA jersey off (primary) and on (official) for one tracker folder."""
    per: dict[str, dict] = {s: {} for s in seqs}
    agg: dict[str, float] = {}
    for jersey in (False, True):
        tag = "on" if jersey else "off"
        s = run_gsr(ctx.paths.eval_python, ctx.paths.trackeval, ctx.paths.gsr_labels,
                    ctx.work / "trackers", split, tracker, seqs, ctx.work / f"eval-jersey-{tag}",
                    use_jersey=jersey)
        metrics = ("hota", "deta", "assa") if not jersey else ("hota",)
        for seq, m in s.clips.items():
            per.setdefault(seq, {}).update({f"{k}_jersey_{tag}": m[k] for k in metrics})
        agg.update({f"{k}_jersey_{tag}": s.combined[k] for k in metrics})
    return {"clips": per, "aggregate": agg}


@kind("gsr_gt_roundtrip")
def gsr_gt_roundtrip(task: dict, ctx: TaskContext) -> dict:
    """Ground truth -> Match State -> predictions. Scores ~100 or the adapters are wrong."""
    split = task["split"]
    out_dir = ctx.work / "trackers" / f"SoccerNetGS-{split}" / "footlytics"
    out_dir.mkdir(parents=True, exist_ok=True)
    failed, done = {}, []
    for seq in _resolve(ctx.paths, split, task.get("clips", "all")):
        try:
            pred = state_to_predictions(gt_to_state(_labels(ctx.paths, split, seq)), split,
                                        use_jersey=True)
            (out_dir / f"{seq}.json").write_text(json.dumps(pred))
            done.append(seq)
        except Exception as e:  # noqa: BLE001
            failed[seq] = f"{type(e).__name__}: {e}"
    if not done:
        raise RuntimeError(f"no clip converted: {failed}")
    ctx.log(f"converted {len(done)} clips, {len(failed)} failed")
    return {**_score(ctx, split, "footlytics", done), "failed": failed}


@kind("gsr_rescore")
def gsr_rescore(task: dict, ctx: TaskContext) -> dict:
    """Score someone else's published predictions (a zip of `<tracker>/<SEQ>.json`)."""
    split, tracker = task["split"], task.get("tracker", "tracklab")
    out_dir = ctx.work / "trackers" / f"SoccerNetGS-{split}" / tracker
    out_dir.mkdir(parents=True, exist_ok=True)
    failed, done = {}, []
    with zipfile.ZipFile(ctx.paths.data / task["predictions"]) as z:
        names = set(z.namelist())
        for seq in _resolve(ctx.paths, split, task.get("clips", "all")):
            member = f"{tracker}/{seq}.json"
            if member not in names:
                failed[seq] = f"no predictions in {task['predictions']}"
                continue
            # one bad clip must fail alone, not take the whole evaluator call with it
            try:
                _labels(ctx.paths, split, seq)
                raw = z.read(member)
                if not isinstance(json.loads(raw).get("predictions"), list):
                    raise ValueError(f"{member} has no predictions list")
            except Exception as e:  # noqa: BLE001
                failed[seq] = f"{type(e).__name__}: {e}"
                continue
            (out_dir / f"{seq}.json").write_bytes(raw)
            done.append(seq)
    if not done:
        raise RuntimeError(f"no predictions for any {split} clip in {task['predictions']}")
    return {**_score(ctx, split, tracker, done), "failed": failed}


# ---------------------------------------------------------------- calibration kinds

VRAM_MIB = 10_000
TRACKER = AnchoredCamera
_CALIBRATORS: dict = {}


def _calibrator(paths: BenchPaths):
    """One SoccerMaster model per process (1.4 GB of weights; thresholds are per call)."""
    key = (str(paths.soccermaster_code), str(paths.weights))
    if key not in _CALIBRATORS:
        from ..geometry.calib_nn import SoccerMasterCalibrator

        _CALIBRATORS[key] = SoccerMasterCalibrator(paths.soccermaster_code, paths.weights,
                                                   device="cuda")
    return _CALIBRATORS[key]


def _thresholds(cfg: dict) -> dict:
    return {"kp_threshold": cfg.get("kp_threshold", DEFAULT_KP_THRESHOLD),
            "line_threshold": cfg.get("line_threshold", DEFAULT_LINE_THRESHOLD)}


def _imread(raw: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("unreadable image")
    return img


def _write_prediction(path: Path, cam: dict) -> None:
    write_camera(path, scale_camera(cam, EVAL_WIDTH))


def _sncal_frames(z: zipfile.ZipFile, split: str, spec) -> list[str]:
    have = sorted(m.group(1) for n in z.namelist()
                  if (m := re.match(rf"^{split}/(\d{{5}})\.json$", n)))
    if spec == "all":
        return have
    if isinstance(spec, dict):
        drop = set(spec.get("exclude", []))
        return [f for f in have
                if int(f) % spec["every"] == spec.get("offset", 0) and f not in drop]
    missing = sorted(set(spec) - set(have))
    if missing:
        raise KeyError(f"frames not in {split}: {missing[:5]}")
    return list(spec)


def _game(info: dict) -> str:
    return "|".join(str(info.get(k, "")).strip() for k in ("league", "season", "match", "date"))


def _copy_subset(src_gt: Path, src_pred: Path, dst: Path, split: str, frames: list[str]) -> None:
    for kind_, src, pat in (("gt", src_gt, "{}.json"), ("pred", src_pred, "camera_{}.json")):
        d = dst / kind_ / split
        d.mkdir(parents=True, exist_ok=True)
        for f in frames:
            p = src / pat.format(f)
            if p.exists():
                shutil.copy(p, d / p.name)


@kind("sncalib")
def sncalib(task: dict, ctx: TaskContext) -> dict:
    """Stills from SN-Calibration: one camera per frame, scored by the official evaluator."""
    split, cfg = task["split"], task.get("calibrator", {})
    require_vram(task.get("vram_mib", VRAM_MIB))
    gt_dir, pred_dir = ctx.work / "gt" / split, ctx.work / "pred" / split
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    cal = _calibrator(ctx.paths)
    t0 = time.time()
    failed, jobs, ids, gts = {}, [], [], {}
    with zipfile.ZipFile(ctx.paths.calib_zip(split)) as z:
        frames = _sncal_frames(z, split, task.get("frames", "all"))
        names = set(z.namelist())
        info = (json.loads(z.read(f"{split}/match_info.json"))
                if f"{split}/match_info.json" in names else {})
        for f in frames:
            try:
                raw = z.read(f"{split}/{f}.json")
                (gt_dir / f"{f}.json").write_bytes(raw)        # the evaluator's denominator
                gts[f] = json.loads(raw)
                img = _imread(z.read(f"{split}/{f}.jpg"))
                (kp, ln), = cal.decode([img[:, :, ::-1]], **_thresholds(cfg))
            except Exception as e:  # noqa: BLE001
                failed[f] = f"{type(e).__name__}: {e}"
                continue
            jobs.append((kp, ln, img.shape[1], img.shape[0]))
            ids.append(f)
    ctx.log(f"decoded {len(ids)} frames, solving on {task.get('workers', 8)} processes")
    solve_kw = {"voting": cfg.get("voting", True), "refine_lines": cfg.get("refine_lines", True)}
    results = solve_many(ctx.paths.soccermaster_code, jobs, workers=task.get("workers", 8),
                         **solve_kw)
    per, errors = {}, 0
    for f, (fit, err) in zip(ids, results):
        errors += err is not None
        if fit is None:
            per[f] = {"calibrated": 0}
            continue
        _write_prediction(pred_dir / f"camera_{f}.json", fit.cam)
        per[f] = {"calibrated": 1, "rep_err": round(fit.rep_err, 2),
                  "median_px": median_reprojection_px(gts[f], fit.cam)}
    s = run_sncalib(ctx.paths.eval_python, ctx.paths.sn_calibration, ctx.work / "gt",
                    ctx.work / "pred", split, ctx.work / "eval")
    med = [c["median_px"] for c in per.values() if c.get("median_px") is not None]
    agg = {"score": s.score, "jac5": s.jac, "completeness": s.completeness,
           "median_px": round(float(np.median(med)), 2) if med else None,
           "seconds_per_frame": round((time.time() - t0) / max(len(frames), 1), 2),
           "frames": len(frames), "solver_errors": errors}
    games = {f: _game(info.get(f"{f}.jpg", {})) for f in frames}
    listed = set(task.get("leakage_games", []))
    if not listed:
        leakage = {"status": "unverified", "games": len(set(games.values()))}
    else:
        keep = [f for f in frames if games[f] not in listed]
        leakage = {"status": "checked", "excluded_games": sorted(listed),
                   "frames_kept": len(keep)}
        if keep:
            _copy_subset(gt_dir, pred_dir, ctx.work / "clean", split, keep)
            sub = run_sncalib(ctx.paths.eval_python, ctx.paths.sn_calibration,
                              ctx.work / "clean" / "gt", ctx.work / "clean" / "pred", split,
                              ctx.work / "eval-clean")
            leakage["score_non_overlap"] = sub.score
    return {"clips": per, "aggregate": agg, "failed": failed, "leakage": leakage}


@kind("gsr_calibration")
def gsr_calibration(task: dict, ctx: TaskContext) -> dict:
    """GSR clips calibrated frame by frame (keyframes + tracker), scored on the GSR pitch-line
    labels by the SN-Calibration evaluator."""
    split, cfg = task["split"], task.get("calibrator", {})
    every, gate = task.get("every", 25), float(task.get("gate_px", 25.0))
    require_vram(task.get("vram_mib", VRAM_MIB))
    cal = _calibrator(ctx.paths)
    thr = _thresholds(cfg)
    solve_kw = {"voting": False, "mode": cfg.get("mode", "full"),
                "ransac": float(cfg.get("ransac", 10.0)),
                "refine_lines": cfg.get("refine_lines", True)}
    solver_errors = 0

    def fit_keyframes(chunk, keys):
        nonlocal solver_errors
        fits = cal.fit([f[:, :, ::-1] for f in chunk], which=keys, **thr, **solve_kw)
        solver_errors += len(getattr(cal, "last_errors", {}))
        return fits

    per, failed = {}, {}
    all_gt, all_pred = ctx.work / "all" / "gt" / split, ctx.work / "all" / "pred" / split
    all_gt.mkdir(parents=True, exist_ok=True)
    all_pred.mkdir(parents=True, exist_ok=True)
    t0, n_frames = time.time(), 0
    with zipfile.ZipFile(ctx.paths.gsr_zip(split)) as z:
        for seq in _resolve(ctx.paths, split, task.get("clips", "all")):
            try:
                solver_errors = 0
                truth = gsr_line_frames(_labels(ctx.paths, split, seq))
                names = sorted(n for n in z.namelist()
                               if n.startswith(f"{seq}/img1/") and n.endswith(".jpg"))
                if not names:
                    raise FileNotFoundError(f"no frames for {seq} in {ctx.paths.gsr_zip(split)}")
                pick = sorted(set(np.linspace(0, len(names) - 1, min(25, len(names)))
                                  .astype(int).tolist()))
                mask = static_overlay_mask([_imread(z.read(names[i])) for i in pick])
                cc = ClipCalibrator(fit_keyframes, every=every, gate_px=gate,
                                    tracker_factory=TRACKER)
                res = cc.run((_imread(z.read(n)) for n in names), mask=mask)
                n_frames += len(res)
                gt_d, pred_d = (ctx.work / "clips" / seq / k / split for k in ("gt", "pred"))
                gt_d.mkdir(parents=True, exist_ok=True)
                pred_d.mkdir(parents=True, exist_ok=True)
                calibrated = 0
                for fc in res:
                    if fc.frame_idx not in truth:
                        continue
                    name = f"{seq}-{fc.frame_idx + 1:06d}"
                    for d in (gt_d, all_gt):
                        (d / f"{name}.json").write_text(json.dumps(truth[fc.frame_idx]))
                    if fc.cam is not None:
                        calibrated += 1
                        for d in (pred_d, all_pred):
                            _write_prediction(d / f"camera_{name}.json", fc.cam)
                s = run_sncalib(ctx.paths.eval_python, ctx.paths.sn_calibration,
                                gt_d.parent, pred_d.parent, split, ctx.work / "eval" / seq)
                per[seq] = {"score": s.score, "jac5": s.jac, "completeness": s.completeness,
                            "calibrated_share": round(calibrated / max(len(truth), 1), 4),
                            "keyframes": cc.keyframes, "keyframes_accepted": cc.keyframes_accepted,
                            "solver_errors": solver_errors}
                ctx.log(f"{seq}: score {s.score:.1f}, {calibrated}/{len(truth)} labelled "
                        f"frames calibrated, {solver_errors} solver errors")
            except Exception as e:  # noqa: BLE001
                failed[seq] = f"{type(e).__name__}: {e}"
    if not per:
        raise RuntimeError(f"no clip calibrated: {failed}")
    s = run_sncalib(ctx.paths.eval_python, ctx.paths.sn_calibration, all_gt.parent,
                    all_pred.parent, split, ctx.work / "eval" / "all")
    keys = sum(c["keyframes"] for c in per.values())
    agg = {"score": s.score, "jac5": s.jac, "completeness": s.completeness,
           "calibrated_share": round(float(np.mean([c["calibrated_share"] for c in per.values()])), 4),
           "keyframe_accept_rate": round(sum(c["keyframes_accepted"] for c in per.values())
                                         / max(keys, 1), 4),
           "solver_errors": sum(c["solver_errors"] for c in per.values()),
           "seconds_per_frame": round((time.time() - t0) / max(n_frames, 1), 3)}
    return {"clips": per, "aggregate": agg, "failed": failed}
