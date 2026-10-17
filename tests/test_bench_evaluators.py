"""sn-trackeval wrapper: command line, CSV parsing, failure reporting."""
import json
import os
import subprocess
from pathlib import Path

import pytest

from footlytics.bench.adapters.gsr import gt_to_state, state_to_predictions
from footlytics.bench.evaluators import EvaluatorError, parse_detailed, run_gsr, trackeval_command
from tests.bench_fixtures import make_labels

FIXTURE = Path(__file__).parent / "fixtures" / "bench" / "person_detailed.csv"


def test_parse_detailed_reads_percent_scores():
    s = parse_detailed(FIXTURE)
    assert set(s.clips) == {"SNGS-117"}
    assert s.combined["hota"] == pytest.approx(24.163, abs=1e-3)
    assert s.combined["deta"] == pytest.approx(11.296, abs=1e-3)
    assert s.combined["assa"] == pytest.approx(51.823, abs=1e-3)
    assert s.combined["loca"] == pytest.approx(88.031, abs=1e-3)
    assert s.combined["idf1"] == pytest.approx(21.142, abs=1e-3)
    assert s.combined["gt_dets"] == 6847


def test_command_switches_jersey_and_pins_split(tmp_path):
    cmd = trackeval_command(Path("/py"), Path("/te"), Path("/gt"), Path("/tr"), "valid", "fl",
                            tmp_path / "seqmap.json", use_jersey=False)
    assert cmd[:2] == ["/py", "/te/scripts/run_soccernet_gs.py"]
    flags = dict(zip(cmd[2::2], cmd[3::2]))
    assert flags["--USE_JERSEY_NUMBERS"] == "False" and flags["--SPLIT_TO_EVAL"] == "valid"
    assert flags["--TRACKER_SUB_FOLDER"] == "" and flags["--TRACKERS_TO_EVAL"] == "fl"
    assert flags["--USE_TEAMS"] == "True"
    assert "--OUTPUT_FOLDER" not in flags          # the script's argparse breaks on it
    off = trackeval_command(Path("/py"), Path("/te"), Path("/gt"), Path("/tr"), "valid", "fl",
                            tmp_path / "seqmap.json", use_jersey=False, use_teams=False)
    assert dict(zip(off[2::2], off[3::2]))["--USE_TEAMS"] == "False"


def test_run_gsr_with_teams_off_tells_the_evaluator(tmp_path):
    """The teams-off pass reaches sn-trackeval as `--USE_TEAMS False` and is parsed like any."""
    pred = tmp_path / "tr" / "SoccerNetGS-valid" / "fl"
    pred.mkdir(parents=True)
    seen = []

    def fake(cmd, **kw):
        seen.append(dict(zip(cmd[2::2], cmd[3::2])))
        (pred / "person_detailed.csv").write_text(FIXTURE.read_text())
        return subprocess.CompletedProcess(cmd, 0, stdout="done", stderr="")
    s = run_gsr(Path("/py"), Path("/te"), tmp_path, tmp_path / "tr", "valid", "fl",
                ["SNGS-117"], tmp_path / "out", use_jersey=False, use_teams=False, run=fake)
    assert seen[0]["--USE_TEAMS"] == "False" and seen[0]["--USE_JERSEY_NUMBERS"] == "False"
    assert s.combined["hota"] == pytest.approx(24.163, abs=1e-3)
    assert (tmp_path / "out" / "person_detailed.csv").exists()


def test_evaluator_failure_carries_its_output(tmp_path):
    def fake(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, stdout="loading\n", stderr="GT file not found")
    with pytest.raises(EvaluatorError, match="GT file not found"):
        run_gsr(Path("/py"), Path("/te"), tmp_path, tmp_path, "valid", "fl", ["SNGS-990"],
                tmp_path / "out", use_jersey=False, run=fake)


def test_evaluator_without_csv_is_an_error(tmp_path):
    def fake(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout="done", stderr="")
    with pytest.raises(EvaluatorError, match="(?s)person_detailed.csv.*done"):
        run_gsr(Path("/py"), Path("/te"), tmp_path, tmp_path, "valid", "fl", ["SNGS-990"],
                tmp_path / "out", use_jersey=False, run=fake)


TE = os.getenv("FOOTLYTICS_TRACKEVAL")
EVAL_PY = os.getenv("FOOTLYTICS_EVAL_PYTHON")


@pytest.mark.skipif(not (TE and EVAL_PY), reason="set FOOTLYTICS_TRACKEVAL and FOOTLYTICS_EVAL_PYTHON")
def test_ground_truth_roundtrip_scores_100(tmp_path):
    labels = make_labels(n_frames=5)
    gt = tmp_path / "gt" / "valid" / "SNGS-990"
    gt.mkdir(parents=True)
    (gt / "Labels-GameState.json").write_text(json.dumps(labels))
    pred = tmp_path / "tr" / "SoccerNetGS-valid" / "fl"
    pred.mkdir(parents=True)
    (pred / "SNGS-990.json").write_text(json.dumps(
        state_to_predictions(gt_to_state(labels), "valid", use_jersey=True)))
    (pred / "person_detailed.csv").write_text("stale")
    for jersey in (False, True):
        s = run_gsr(Path(EVAL_PY), Path(TE), tmp_path / "gt", tmp_path / "tr", "valid", "fl",
                    ["SNGS-990"], tmp_path / f"out{jersey}", use_jersey=jersey)
        assert s.combined["hota"] > 99.0
        assert (tmp_path / f"out{jersey}" / "person_detailed.csv").exists()
    assert not (pred / "person_detailed.csv").exists()
