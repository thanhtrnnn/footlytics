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


def test_split_officials_weighs_time_on_screen_not_fragments():
    """A third kit seen in over a quarter of all observations is not 3-4 officials, however
    few track ids it spans. Counting ids let 50 of 258 fragments (19%, under the 25% id
    guard) take much of one team on the 3-minute clip and leave a 3 v 7 split."""
    from footlytics.perception.teams import split_officials

    D, T = _crops()
    home_away = T < 22
    ref = np.flatnonzero(T >= 22)
    # 22 players seen 30 times each; the third kit on 5 tracks seen 50 times each
    D_ref = D[ref][np.arange(250) % len(ref)]
    T_ref = 22 + np.arange(250) // 50
    D2 = np.concatenate([D[home_away], D_ref])
    T2 = np.concatenate([T[home_away], T_ref])

    info = {}
    assert split_officials(T2, D2, info=info) == set()
    assert info["id_share"] < 0.25 < info["obs_share"]
    assert "observations" in info["reason"]
    # the id-only guard alone would have accepted it
    assert split_officials(T2, D2, max_obs_share=1.0) == {22, 23, 24, 25, 26}


def test_split_officials_reports_why():
    from footlytics.perception.teams import split_officials

    D, T = _crops()
    info = {}
    split_officials(T, D, info=info)
    assert info["reason"] == "split" and info["tracks"] == 3
