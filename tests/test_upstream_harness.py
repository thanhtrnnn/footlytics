"""Pytest wrappers around the upstream synthetic-truth harnesses in scripts/.

Thresholds are the numbers reported in README.md with headroom, so a regression
in geometry, tracking, team assignment or rendering fails a test instead of only
changing a printed table.
"""
from pathlib import Path
import runpy

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(script: str) -> dict:
    """Execute a harness's top-level definitions without running its main()."""
    return runpy.run_path(str(ROOT / "scripts" / script), run_name="harness")


def test_geometry_harness_asserts_and_numbers(monkeypatch):
    monkeypatch.chdir(ROOT)
    g = runpy.run_path(str(ROOT / "scripts" / "test_geometry.py"), run_name="__main__")
    assert g["err"].max() < 1e-6                      # exact pinhole recovery
    assert g["err_n"].mean() < 0.2                    # README: 0.09 m with 3 px click noise
    assert g["e_tps"].mean() < g["e_plain"].mean()    # TPS earns its keep on a stitched view
    via_feet, via_centre, truth = g["via_feet"], g["via_centre"], g["player_xy"][0]
    assert np.linalg.norm(via_feet - truth) < 0.05
    assert np.linalg.norm(via_centre - truth) > 3.0   # README: 6.7 m if you map the box centre


def test_tracker_appearance_closes_the_duel_gap():
    m = _load("test_tracker.py")
    truth = m["simulate"]()
    sw_motion, coverage, live = m["run"](truth, 0.05, 0.0, 0.0)
    assert coverage > 0.95
    assert live == 23
    sw_app, live_app = m["run_appearance"](truth, 0.20, 0.0, 0.0, emb_noise=0.35)
    assert sw_app < 40                                # README: 10 with appearance
    assert sw_app < sw_motion                         # README: 103 / 164 motion only


def test_teams_distinct_kits_are_separable_and_balanced():
    m = _load("test_teams.py")
    from footlytics.perception.teams import TeamClassifier, kit_descriptor
    from footlytics.state.schema import Team

    fake_crop = m["fake_crop"]
    descs, tids = [], []
    for i in range(22):
        kit = "home" if i < 11 else "away"
        for _ in range(30):
            img, bb = fake_crop(kit, blur=0.3, shadow=0.9)
            descs.append(kit_descriptor(img, bb))
            tids.append(i)
    descs, tids = np.array(descs), np.array(tids)
    clf = TeamClassifier().fit(descs)
    assign = clf.assign_tracks(tids, descs, balanced=True)
    got = np.array([assign[i].team for i in range(22)])
    n_home = int((got == Team.HOME.value).sum())
    assert n_home == 11
    ea = np.array([Team.HOME.value] * 11 + [Team.AWAY.value] * 11)
    eb = np.array([Team.AWAY.value] * 11 + [Team.HOME.value] * 11)
    assert max((got == ea).mean(), (got == eb).mean()) >= 0.95   # README: 100% red vs blue


def test_viz_harness_renders_and_round_trips(monkeypatch):
    monkeypatch.chdir(ROOT)
    g = runpy.run_path(str(ROOT / "scripts" / "test_viz.py"), run_name="__main__")
    # The synthetic 4-4-2 drifts 6 m either way, so validate() rightly flags a >100% length span;
    # any other problem (off-pitch, speeds, unknown teams, split) would be a real regression.
    assert all("span" in p for p in g["probs"]), g["probs"]
    assert (ROOT / "data/out/radar_frame.png").exists()


@pytest.mark.skipif(not (ROOT / "data/raw/soccertrack/117092/117092.txt").exists(),
                    reason="SoccerTrack v2 clip 117092 not downloaded (gated dataset)")
def test_identity_harness_on_real_ground_truth(monkeypatch):
    monkeypatch.chdir(ROOT)
    runpy.run_path(str(ROOT / "scripts" / "test_identity.py"), run_name="__main__")
