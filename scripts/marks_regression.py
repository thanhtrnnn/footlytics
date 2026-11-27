"""Re-run sequence detection on a saved match directory and score it against analyst marks.

    uv run python scripts/marks_regression.py <match_dir> [--out sequences.csv] [--json report.json]

`<match_dir>` needs `tracks.parquet` and `marks.json` (the review tool's output). Each
marked sequence is looked up in the fresh detections by label, team and period; a
sequence is "matched" when a detection of that label overlaps the marked start frame by
`--tol-s` seconds either side. Missed sequences are matched when a detection of the same
label and team overlaps at least half of the shorter of the two intervals.

This is a regression check on a handful of labelled examples, not a precision estimate.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from footlytics.analytics.possession import infer_possession
from footlytics.analytics.sequences import detect_sequences
from footlytics.analytics.tactics import normalise_direction
from footlytics.state.schema import MatchState

_KEY = re.compile(r"^(?P<label>[a-z_]+)-(?P<team>home|away)-(?P<period>\d+)-(?P<frame>-?\d+)$")


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def score(seqs: pd.DataFrame, marks: dict, fps: float, tol_s: float = 2.0) -> pd.DataFrame:
    rows = []
    for key, m in marks.get("sequences", {}).items():
        k = _KEY.match(key)
        label, team, period, frame = k["label"], k["team"], int(k["period"]), int(k["frame"])
        t = frame / fps
        hit = seqs[(seqs["label"] == label) & (seqs["team"] == team) & (seqs["period"] == period)
                   & (seqs["end_s"] >= t - tol_s) & (seqs["start_s"] <= t + tol_s)]
        rows.append({"kind": "marked", "key": key, "verdict": m["verdict"], "label": label,
                     "detected_now": len(hit) > 0,
                     "now": (f"{hit.iloc[0]['start_s']:.1f}-{hit.iloc[0]['end_s']:.1f}s" if len(hit) else "")})
    for m in marks.get("missed", []):
        s = seqs[(seqs["label"] == m["label"]) & (seqs["team"] == m["team"].lower())]
        dur = m["end_s"] - m["start_s"]
        best = 0.0
        for r in s.itertuples():
            ov = _overlap(m["start_s"], m["end_s"], r.start_s, r.end_s)
            best = max(best, ov / max(1e-9, min(dur, r.end_s - r.start_s)))
        rows.append({"kind": "missed", "key": f"{m['label']}-{m['team'].lower()}-{m['start_s']:.0f}-{m['end_s']:.0f}s",
                     "verdict": "missed", "label": m["label"], "detected_now": best >= 0.5,
                     "now": f"best overlap {best:.2f}"})
    return pd.DataFrame(rows)


def summarise(table: pd.DataFrame) -> dict:
    marked, missed = table[table["kind"] == "marked"], table[table["kind"] == "missed"]
    return {
        "right_kept": int(((marked["verdict"] == "right") & marked["detected_now"]).sum()),
        "right_total": int((marked["verdict"] == "right").sum()),
        "wrong_dropped": int(((marked["verdict"] == "wrong") & ~marked["detected_now"]).sum()),
        "wrong_total": int((marked["verdict"] == "wrong").sum()),
        "unsure_detected": int(((marked["verdict"] == "unsure") & marked["detected_now"]).sum()),
        "unsure_total": int((marked["verdict"] == "unsure").sum()),
        "missed_recovered": int(missed["detected_now"].sum()),
        "missed_total": int(len(missed)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("match_dir", type=Path)
    ap.add_argument("--out", type=Path, help="write the fresh sequences here")
    ap.add_argument("--json", type=Path, help="write the per-sequence table and summary here")
    ap.add_argument("--tol-s", type=float, default=2.0)
    args = ap.parse_args()

    state = normalise_direction(MatchState.load(args.match_dir))
    poss = infer_possession(state)
    seqs = detect_sequences(state, poss)
    if args.out:
        seqs.to_csv(args.out, index=False)
    print(seqs["label"].value_counts().to_string())
    marks_path = args.match_dir / "marks.json"
    if not marks_path.is_file():
        return
    table = score(seqs, json.loads(marks_path.read_text()), state.meta.fps, args.tol_s)
    print(table.to_string(index=False))
    summary = summarise(table)
    print(json.dumps(summary))
    if args.json:
        args.json.write_text(json.dumps({"table": table.to_dict("records"), "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
