"""Suite loop on the host: tasks -> adapters -> evaluators -> one result row.

A suite is a YAML list of typed tasks (`suites/<name>.yaml`); each kind is a function
registered in `KINDS`. A clip that fails is recorded with its reason and the task goes on; a
task that fails is recorded and the suite goes on. Any failure makes the run `partial`.
"""
from __future__ import annotations

import json
import shutil
import time
import traceback
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml

from . import datasets
from .adapters.gsr import gt_to_state, state_to_predictions
from .evaluators import run_gsr
from .host import ResourceError, contention, torch_info
from .paths import BenchPaths

SUITES_DIR = Path(__file__).parent / "suites"
DONE = ("complete", "partial", "failed")
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


def preflight(paths: BenchPaths) -> dict:
    """Refuse to run on a torch other than the one setup pinned (it lives in a shared ~/.local)."""
    if not paths.torch_pin.exists():
        return {}
    info = torch_info()
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
            labels = json.loads((ctx.paths.gsr_labels / split / seq / "Labels-GameState.json")
                                .read_text())
            pred = state_to_predictions(gt_to_state(labels), split, use_jersey=True)
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
            (out_dir / f"{seq}.json").write_bytes(z.read(member))
            done.append(seq)
    if not done:
        raise RuntimeError(f"no predictions for any {split} clip in {task['predictions']}")
    return {**_score(ctx, split, tracker, done), "failed": failed}
