"""One row per run in bench/results/results.jsonl; SCOREBOARD.md is regenerated from it.

Both files are committed: the history of the numbers is part of the repository. A `partial`
run (any clip or task failed) is shown but never compared with complete runs.
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "bench" / "results"
HISTORY = 20


def load_results(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def append_result(path: str | Path, row: dict) -> bool:
    p = Path(path)
    if any(r["run_id"] == row["run_id"] for r in load_results(p)):
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return True


#: The number a task kind shows on the scoreboard when its suite gives it no check. `score` is
#: JaC@5 x completeness.
PRIMARY = {"gsr_pipeline": "hota_jersey_off", "gsr_gt_roundtrip": "hota_jersey_off",
           "gsr_rescore": "hota_jersey_on", "sncalib": "score", "gsr_calibration": "score"}


def headline(task: dict) -> tuple[str, float | None, bool | None]:
    """The task's check metric, else its kind's `PRIMARY`, else its first aggregate key."""
    if task.get("status") == "failed":
        return "error", None, None
    check = task.get("check") or {}
    agg = task.get("aggregate") or {}
    metric = check.get("metric") or PRIMARY.get(task.get("kind")) or next(iter(agg), "")
    return metric, agg.get(metric), check.get("passed")


def _cell(task: dict | None) -> str:
    if task is None:
        return ""
    metric, value, passed = headline(task)
    if value is None:
        return metric
    mark = {True: " ✓", False: " ✗"}.get(passed, "")
    return f"{metric} {value:.2f}{mark}"


def render_scoreboard(rows: list[dict]) -> str:
    out = ["# Benchmark scoreboard", "",
           "Regenerated from `results.jsonl` by `footlytics bench scoreboard`; do not edit.",
           "`partial` runs had a failed clip or task and are not comparable with complete runs.",
           ""]
    for suite in sorted({r["suite"] for r in rows}):
        runs = sorted((r for r in rows if r["suite"] == suite),
                      key=lambda r: r["finished_at"], reverse=True)[:HISTORY]
        task_ids = list(dict.fromkeys(t for r in runs for t in r["tasks"]))
        out += [f"## {suite}", "",
                "| finished (UTC) | commit | status | host | " + " | ".join(task_ids) + " |",
                "|---" * (4 + len(task_ids)) + "|"]
        for r in runs:
            when = r["finished_at"].replace("T", " ")[:16]
            cells = [_cell(r["tasks"].get(t)) for t in task_ids]
            out.append(f"| {when} | {r['git_sha'][:7]} | {r['status']} | {r['host']} | "
                       + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def write_scoreboard(results_dir: str | Path = RESULTS_DIR) -> Path:
    d = Path(results_dir)
    md = d / "SCOREBOARD.md"
    md.write_text(render_scoreboard(load_results(d / "results.jsonl")))
    return md
