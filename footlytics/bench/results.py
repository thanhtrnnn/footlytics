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
           "gsr_rescore": "hota_jersey_on", "sncalib": "score", "gsr_calibration": "score",
           "vleague": "share_under_1m", "flicker": "flicker_passed"}


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


#: (aggregate key, column title, format) of the flicker table. Reported next to the headline
#: numbers, never gating: `flicker.THRESHOLDS` says what passes, and a run that fails them is
#: still a complete run.
FLICKER_COLUMNS = (("flicker_player_step_gt12", "player >12 m/s", "{:.2%}"),
                   ("flicker_player_step_gt20", "player >20 m/s", "{:.2%}"),
                   ("flicker_ball_step_gt40", "ball >40 m/s", "{:.2%}"),
                   ("flicker_ball_beyond_gate", "ball beyond gate", "{:.0f}"),
                   ("flicker_common_mode_gt05", "common-mode >0.5 m", "{:.2%}"),
                   ("flicker_still_sigma_p50", "still sigma p50 m", "{:.3f}"),
                   ("flicker_still_sigma_p90", "still sigma p90 m", "{:.3f}"),
                   ("flicker_births_per_min", "births/min", "{:.1f}"),
                   ("flicker_id_life_median_s", "id life s", "{:.1f}"),
                   ("flicker_frames_over_11_share", ">11 on a team", "{:.2%}"))


def _flicker_section(runs: list[dict]) -> list[str]:
    """The flicker metrics of the newest run's tasks that carry them, as a table (or nothing)."""
    for r in runs:
        tasks = {t: v for t, v in r["tasks"].items()
                 if (v.get("aggregate") or {}).get("flicker_total")}
        if not tasks:
            continue
        out = ["Flicker (reported, not gating), newest run with it:", "",
               "| task | passed | " + " | ".join(c[1] for c in FLICKER_COLUMNS) + " |",
               "|---" * (2 + len(FLICKER_COLUMNS)) + "|"]
        for t, v in tasks.items():
            a = v["aggregate"]
            cells = [c[2].format(a[c[0]]) if a.get(c[0]) is not None else "-" for c in FLICKER_COLUMNS]
            out.append(f"| {t} | {a['flicker_passed']}/{a['flicker_total']} | " + " | ".join(cells) + " |")
        return out + [""]
    return []


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
        out += _flicker_section(runs)
    return "\n".join(out)


def write_scoreboard(results_dir: str | Path = RESULTS_DIR) -> Path:
    d = Path(results_dir)
    md = d / "SCOREBOARD.md"
    md.write_text(render_scoreboard(load_results(d / "results.jsonl")))
    return md
