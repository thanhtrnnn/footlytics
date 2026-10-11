"""Official evaluators, run as subprocesses in their own venv.

We do not reimplement GS-HOTA: our output is written in the SoccerNet format and scored by
sn-trackeval itself (pinned in host_setup.sh). Never import it; it lives in venv-eval.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class EvaluatorError(RuntimeError):
    """The evaluator failed; the message carries the end of its output."""


@dataclass
class GsrScores:
    clips: dict[str, dict]
    combined: dict


_PERCENT = {"hota": "HOTA___AUC", "deta": "DetA___AUC", "assa": "AssA___AUC",
            "loca": "LocA___AUC", "idf1": "IDF1"}
_COUNT = {"dets": "Dets", "gt_dets": "GT_Dets"}


def _row(r: dict) -> dict:
    out = {k: round(100 * float(r[c]), 3) for k, c in _PERCENT.items()}
    out.update({k: int(float(r[c])) for k, c in _COUNT.items()})
    return out


def parse_detailed(path: str | Path) -> GsrScores:
    rows = list(csv.DictReader(Path(path).open()))
    clips = {r["seq"]: _row(r) for r in rows if r["seq"] != "COMBINED"}
    combined = [_row(r) for r in rows if r["seq"] == "COMBINED"]
    if not combined:
        raise EvaluatorError(f"no COMBINED row in {path}")
    return GsrScores(clips, combined[0])


def trackeval_command(python: Path, trackeval_dir: Path, gt_root: Path, trackers_root: Path,
                      split: str, tracker: str, seqmap: Path, use_jersey: bool) -> list[str]:
    # No --OUTPUT_FOLDER: its default is None, which the script's argparse turns into a list
    # and then crashes on. Results land next to the predictions instead (`run_gsr` moves them).
    flags = {
        "--GT_FOLDER": gt_root, "--TRACKERS_FOLDER": trackers_root, "--TRACKER_SUB_FOLDER": "",
        "--TRACKERS_TO_EVAL": tracker, "--SPLIT_TO_EVAL": split, "--SEQMAP_FILE": seqmap,
        "--USE_JERSEY_NUMBERS": use_jersey,
        "--USE_PARALLEL": False, "--PRINT_CONFIG": False, "--TIME_PROGRESS": False,
        "--PLOT_CURVES": False,
    }
    cmd = [str(python), str(trackeval_dir / "scripts" / "run_soccernet_gs.py")]
    for k, v in flags.items():
        cmd += [k, str(v)]
    return cmd


def run_gsr(python: Path, trackeval_dir: Path, gt_root: Path, trackers_root: Path, split: str,
            tracker: str, seqs: list[str], out_dir: Path, use_jersey: bool,
            run=subprocess.run) -> GsrScores:
    """GS-HOTA for `seqs`, predictions at `trackers_root/SoccerNetGS-<split>/<tracker>/`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seqmap = out_dir / "seqmap.json"
    seqmap.write_text(json.dumps([{"name": s, "nframes": 750} for s in seqs]))  # nframes unused
    cmd = trackeval_command(python, trackeval_dir, gt_root, trackers_root, split, tracker,
                            seqmap, use_jersey)
    pred_dir = Path(trackers_root) / f"SoccerNetGS-{split}" / tracker
    outputs = ("*_detailed.csv", "*_summary.txt")
    for pattern in outputs:                      # a stale CSV must not pass for this run's
        for f in pred_dir.glob(pattern):
            f.unlink()
    p = run(cmd, capture_output=True, text=True, cwd=str(trackeval_dir))
    (out_dir / "evaluator.log").write_text(f"{' '.join(cmd)}\n\n{p.stdout}\n{p.stderr}")
    tail = "\n".join((p.stdout + "\n" + p.stderr).strip().splitlines()[-15:])
    if p.returncode != 0:
        raise EvaluatorError(f"sn-trackeval exited {p.returncode}:\n{tail}")
    for pattern in outputs:
        for f in pred_dir.glob(pattern):
            shutil.move(str(f), out_dir / f.name)
    csv_path = out_dir / "person_detailed.csv"
    if not csv_path.exists():
        raise EvaluatorError(f"sn-trackeval exited 0 but wrote no person_detailed.csv in "
                             f"{pred_dir}:\n{tail}")
    return parse_detailed(csv_path)
