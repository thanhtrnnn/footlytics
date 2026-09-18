"""CLI surface: run, calib-assist, drift-check."""
from pathlib import Path

import pytest
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
FRAME = ROOT / "data/out/frame0.png"
CLIP = ROOT / "data/clips/sample_30s.mp4"
CALIB = ROOT / "calib/brazil_france_30s_frame0.json"
WEIGHTS = ROOT / "weights/yolo_v8x6_finetuned.pt"


def test_help_lists_commands():
    from footlytics.cli import app

    r = CliRunner().invoke(app, ["--help"])
    assert r.exit_code == 0
    for cmd in ("run", "calib-assist", "drift-check"):
        assert cmd in r.output


@pytest.mark.skipif(not CLIP.exists(), reason="clip missing")
def test_calib_assist_writes_candidates(tmp_path):
    from footlytics.cli import app

    r = CliRunner().invoke(app, ["calib-assist", str(CLIP), "--frame", "0", "--out", str(tmp_path / "assist")])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "assist.jpg").exists() and (tmp_path / "assist.json").exists()


@pytest.mark.skipif(not (CLIP.exists() and CALIB.exists() and WEIGHTS.exists()), reason="clip, calib or weights missing")
def test_run_writes_state_quality_and_videos(tmp_path):
    from footlytics.cli import app

    r = CliRunner().invoke(app, ["run", str(CLIP), "--calib", str(CALIB), "--out", str(tmp_path),
                                 "--max-frames", "12", "--device", "mps", "--home", "BRA", "--away", "FRA"])
    assert r.exit_code == 0, r.output
    for f in ("tracks.parquet", "meta.json", "quality.json", "quality.md", "report.json", "radar.mp4"):
        assert (tmp_path / f).exists(), f
