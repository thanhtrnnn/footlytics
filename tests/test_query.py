"""Tests for L5 multi-match DuckDB tactical query API and CLI."""
import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from footlytics.analytics.query import discover_matches, search_sequences, summarize_sequences
from footlytics.analytics.sequences import COLUMNS
from footlytics.cli import app


def _write_match(
    parent: Path,
    folder: str,
    rows: list[dict],
    *,
    match_id: str | None = None,
    home: str = "Home",
    away: str = "Away",
    fps: float = 25.0,
    with_meta: bool = True,
) -> Path:
    d = parent / folder
    d.mkdir(parents=True, exist_ok=True)
    full_rows = []
    for r in rows:
        base = {c: None for c in COLUMNS}
        base.update(r)
        full_rows.append(base)
    pd.DataFrame(full_rows, columns=COLUMNS).to_csv(d / "sequences.csv", index=False)
    if with_meta:
        meta = {
            "match_id": match_id or folder,
            "fps": fps,
            "home": {"name": home, "short_name": home[:3].upper()},
            "away": {"name": away, "short_name": away[:3].upper()},
        }
        (d / "meta.json").write_text(json.dumps(meta))
    return d


@pytest.fixture
def multi_match_root(tmp_path: Path) -> Path:
    _write_match(
        tmp_path,
        "match_a",
        [
            dict(
                label="build_up", team="away", period=1,
                start_s=80.0, end_s=85.0, start_frame=2000, end_frame=2124,
                duration_s=5.0, def_line_x=-20.0, compactness_m=15.0,
                n_within_10m=0, flank="", n_attackers=1, n_defenders=0,
                in_possession_share=1.0, n_pressers=3, response="",
            ),
            dict(
                label="low_block", team="away", period=1,
                start_s=268.0, end_s=277.4, start_frame=6700, end_frame=6934,
                duration_s=9.4, def_line_x=-36.4, compactness_m=10.3,
                n_within_10m=6, flank="", n_attackers=0, n_defenders=0,
                in_possession_share=0.0, n_pressers=7, response="",
            ),
        ],
        match_id="cahcm-hanoi",
        home="CAHCM",
        away="HANOI",
    )
    _write_match(
        tmp_path,
        "match_b",
        [
            dict(
                label="midfield_loss", team="home", period=1,
                start_s=120.0, end_s=126.0, start_frame=3000, end_frame=3149,
                duration_s=6.0, def_line_x=-4.0, compactness_m=17.0,
                n_within_10m=2, flank="", n_attackers=0, n_defenders=0,
                in_possession_share=0.0, n_pressers=0,
                ball_dist_m=4.5, retreat_mps=-0.2, response="counter_press",
            ),
            dict(
                label="flank_overload", team="home", period=2,
                start_s=3100.0, end_s=3104.5, start_frame=77500, end_frame=77612,
                duration_s=4.5, def_line_x=5.0, compactness_m=12.0,
                n_within_10m=1, flank="right", n_attackers=4, n_defenders=2,
                in_possession_share=0.9, n_pressers=1, response="",
            ),
            dict(
                label="low_block", team="away", period=2,
                start_s=3400.0, end_s=3412.0, start_frame=85000, end_frame=85299,
                duration_s=12.0, def_line_x=-31.0, compactness_m=9.8,
                n_within_10m=5, flank="", n_attackers=0, n_defenders=0,
                in_possession_share=0.0, n_pressers=4, response="",
            ),
        ],
        match_id="danang-haiphong",
        home="DANANG",
        away="HAIPHONG",
    )
    return tmp_path


def test_discover_matches_and_empty_folder(tmp_path: Path, multi_match_root: Path):
    entries = discover_matches(multi_match_root)
    assert [e.match_id for e in entries] == ["cahcm-hanoi", "danang-haiphong"]
    assert entries[0].home == "CAHCM" and entries[1].away == "HAIPHONG"

    # Passing direct match folders deduplicates properly
    entries2 = discover_matches([multi_match_root / "match_a", multi_match_root])
    assert len(entries2) == 2

    empty = tmp_path / "empty_dir"
    empty.mkdir()
    assert discover_matches(empty) == []
    assert summarize_sequences(empty).empty
    assert search_sequences(empty).empty


def test_discover_fallback_when_no_meta(tmp_path: Path):
    _write_match(tmp_path, "no_meta_match", [], with_meta=False)
    entries = discover_matches(tmp_path)
    assert len(entries) == 1
    assert entries[0].match_id == "no_meta_match"
    assert summarize_sequences(tmp_path).empty


def test_summarize_sequences_by_match_and_overall(multi_match_root: Path):
    by_m = summarize_sequences(multi_match_root, by_match=True)
    assert set(by_m["match_id"]) == {"cahcm-hanoi", "danang-haiphong"}
    assert len(by_m) == 5

    overall = summarize_sequences(multi_match_root, by_match=False)
    lb = overall[(overall["team"] == "away") & (overall["label"] == "low_block")]
    assert len(lb) == 1
    assert lb.iloc[0]["count"] == 2
    assert lb.iloc[0]["total_duration_s"] == pytest.approx(21.4)
    assert lb.iloc[0]["mean_duration_s"] == pytest.approx(10.7)


def test_search_sequences_filters_sort_and_limit(multi_match_root: Path):
    lb = search_sequences(
        multi_match_root, label="low_block", sort_by="duration_s", descending=True
    )
    assert len(lb) == 2
    assert lb["match_id"].tolist() == ["danang-haiphong", "cahcm-hanoi"]

    top1 = search_sequences(
        multi_match_root, label="low_block", sort_by="duration_s", descending=True, limit=1
    )
    assert len(top1) == 1 and top1.iloc[0]["match_id"] == "danang-haiphong"

    win = search_sequences(multi_match_root, start_s=270.0, end_s=3102.0)
    assert len(win) == 2
    assert set(win["label"]) == {"low_block", "flank_overload"}

    cp = search_sequences(multi_match_root, response="counter_press")
    assert len(cp) == 1 and cp.iloc[0]["match_id"] == "danang-haiphong"

    fl = search_sequences(multi_match_root, flank="right", period=2)
    assert len(fl) == 1 and fl.iloc[0]["label"] == "flank_overload"

    pr = search_sequences(multi_match_root, min_pressers=4, min_duration=10.0)
    assert len(pr) == 1 and pr.iloc[0]["match_id"] == "danang-haiphong"

    with pytest.raises(ValueError, match="unknown sort_by"):
        search_sequences(multi_match_root, sort_by="nope")


def test_cli_query_summary_search_and_exports(tmp_path: Path, multi_match_root: Path):
    runner = CliRunner()

    res_sum = runner.invoke(app, ["query", "summary", str(multi_match_root), "--overall", "--json"])
    assert res_sum.exit_code == 0
    payload = json.loads(res_sum.stdout)
    assert any(r["label"] == "low_block" and r["count"] == 2 for r in payload)

    out_csv = tmp_path / "out.csv"
    out_json = tmp_path / "out.json"

    res_csv = runner.invoke(
        app,
        ["query", "search", str(multi_match_root), "--label", "low_block", "--out", str(out_csv)],
    )
    assert res_csv.exit_code == 0
    assert out_csv.exists() and len(pd.read_csv(out_csv)) == 2

    res_json = runner.invoke(
        app,
        [
            "query", "search", str(multi_match_root),
            "--match", "cahcm-hanoi", "--team", "away", "--out", str(out_json), "--json",
        ],
    )
    assert res_json.exit_code == 0
    exported = json.loads(out_json.read_text())
    assert len(exported) == 2
    assert all(r["match_id"] == "cahcm-hanoi" for r in exported)


def test_reruns_sharing_a_match_id_stay_distinct(tmp_path: Path):
    row = dict(
        label="low_block", team="away", period=1, start_s=10.0, end_s=20.0, start_frame=250,
        end_frame=500, duration_s=10.0, def_line_x=-30.0, compactness_m=10.0, n_within_10m=3,
        flank="", n_attackers=0, n_defenders=0, in_possession_share=0.0, n_pressers=2, response="",
    )
    _write_match(tmp_path, "run_a", [row], match_id="vn_match", home="A", away="B")
    _write_match(tmp_path, "run_b", [row, {**row, "start_s": 40.0, "end_s": 50.0}], match_id="vn_match", home="A", away="B")
    _write_match(tmp_path, "solo", [row], match_id="other", home="C", away="D")

    entries = discover_matches(tmp_path)

    assert sorted(e.match_id for e in entries) == ["other", "vn_match@run_a", "vn_match@run_b"]
    per_match = summarize_sequences(tmp_path)
    assert per_match.groupby("match_id")["count"].sum().to_dict() == {"other": 1, "vn_match@run_a": 1, "vn_match@run_b": 2}
