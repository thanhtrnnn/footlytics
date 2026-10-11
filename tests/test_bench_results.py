"""results.jsonl is append-only and idempotent; SCOREBOARD.md is derived from it."""
from footlytics.bench.results import append_result, headline, load_results, render_scoreboard


def row(run_id, status="complete", hota=99.97, passed=True, finished="2026-10-12T10:03:00Z"):
    return {"run_id": run_id, "suite": "harness", "git_sha": "abcdef1234567", "host": "iec4090_2",
            "started_at": "2026-10-12T10:00:00Z", "finished_at": finished, "status": status,
            "tasks": {"gsr-gt-roundtrip-valid": {
                "kind": "gsr_gt_roundtrip", "status": "complete",
                "aggregate": {"hota_jersey_off": hota, "hota_jersey_on": hota},
                "check": {"metric": "hota_jersey_off", "value": hota, "passed": passed}}}}


def test_append_is_idempotent(tmp_path):
    p = tmp_path / "results.jsonl"
    assert append_result(p, row("r1")) is True
    assert append_result(p, row("r1")) is False
    assert [r["run_id"] for r in load_results(p)] == ["r1"]


def test_headline_prefers_the_check_metric():
    assert headline(row("r1")["tasks"]["gsr-gt-roundtrip-valid"]) == ("hota_jersey_off", 99.97, True)
    assert headline({"status": "failed", "error": "boom"}) == ("error", None, None)


def task(kind, aggregate, check=None):
    return {"kind": kind, "status": "complete", "aggregate": aggregate, "check": check}


def test_headline_is_the_kinds_primary_metric_not_the_first_stored_key():
    """Rows are stored with sorted keys, so the first key is an accident of the alphabet."""
    gsr = {"assa_jersey_off": 45.77, "hota_jersey_off": 30.12, "hota_jersey_on": 23.09}
    calib = {"calibrated_share": 1.0, "completeness": 76.67, "score": 54.4}
    assert headline(task("gsr_pipeline", gsr)) == ("hota_jersey_off", 30.12, None)
    assert headline(task("gsr_gt_roundtrip", gsr)) == ("hota_jersey_off", 30.12, None)
    assert headline(task("gsr_rescore", gsr)) == ("hota_jersey_on", 23.09, None)
    assert headline(task("sncalib", calib)) == ("score", 54.4, None)
    assert headline(task("gsr_calibration", calib)) == ("score", 54.4, None)
    assert headline(task("gsr_pipeline", gsr, {"metric": "assa_jersey_off", "passed": True})) \
        == ("assa_jersey_off", 45.77, True)                             # a check still wins
    assert headline(task("new_kind", {"b": 2.0, "a": 1.0})) == ("b", 2.0, None)
    md = render_scoreboard([{**row("r1"), "tasks": {"gp": task("gsr_pipeline", gsr)}}])
    assert "| hota_jersey_off 30.12 |" in md


def test_scoreboard_newest_first_and_marks_partial():
    md = render_scoreboard([row("r1", finished="2026-10-12T10:00:00Z"),
                            row("r2", status="partial", hota=50.0, passed=False,
                                finished="2026-10-13T10:00:00Z")])
    assert "## harness" in md
    lines = [l for l in md.splitlines() if l.startswith("| 2026")]
    assert "partial" in lines[0] and "50.00 ✗" in lines[0]
    assert "99.97 ✓" in lines[1]
    assert "abcdef1" in lines[1]
