"""When the detector has no referee class, officials must be split off before the 2-means team fit,
otherwise a yellow-shirted referee lands in one team and skews the 11-a-side quota."""
import runpy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _crops():
    m = runpy.run_path(str(ROOT / "scripts" / "test_teams.py"), run_name="harness")
    fake_crop, KITS = m["fake_crop"], m["KITS"]
    from footlytics.perception.teams import kit_descriptor

    KITS["home"], KITS["away"] = np.array([200, 40, 40]), np.array([40, 60, 190])
    KITS["ref"] = np.array([235, 220, 40])
    descs, tids = [], []
    for i in range(25):                      # 11 + 11 players, 3 officials
        kit = "home" if i < 11 else ("away" if i < 22 else "ref")
        for _ in range(30):
            img, bb = fake_crop(kit, blur=0.3, shadow=0.9)
            descs.append(kit_descriptor(img, bb)); tids.append(i)
    return np.array(descs), np.array(tids)


def test_split_officials_finds_the_third_kit():
    from footlytics.perception.teams import split_officials

    D, T = _crops()
    officials = split_officials(T, D)
    assert officials == {22, 23, 24}


def test_split_officials_returns_empty_when_only_two_kits():
    from footlytics.perception.teams import split_officials

    D, T = _crops()
    keep = T < 22
    assert split_officials(T[keep], D[keep]) == set()
