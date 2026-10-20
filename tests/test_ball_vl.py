import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ball_vl", REPO / "scripts" / "ball_vl.py")
ball_vl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ball_vl)

MANIFEST = json.loads((REPO / "bench/vleague/manifest.json").read_text())["matches"]


def _frozen_keys():
    return {f"{m}/{i}" for m, e in MANIFEST.items() for i, f in e["frames"].items() if f["split"] == "frozen"}


def test_dev_windows_never_contain_a_frozen_window(tmp_path):
    wins = ball_vl.dev_windows(tmp_path)
    assert wins and not (set(wins) & _frozen_keys())
    assert all(len(v) == 2 * ball_vl.WINDOW + 1 for v in wins.values())


def test_test_keys_and_training_windows_are_disjoint_and_dev_only():
    keys = ball_vl.test_keys()
    win_keys = {"/".join(k.split("/")[1:3]) for k in keys if k.startswith("win/")}
    assert win_keys and not (win_keys & _frozen_keys())
    assert all(m == "ninh-binh-pvf-cand" for m in {k.split("/")[0] for k in win_keys})
    clip = [int(k.split("/")[1]) for k in keys if k.startswith("clip/")]
    assert min(clip) >= 12000 > 390 * 25      # after the pseudo-labelled head of the clip


def test_eval_scores_recall_and_false_positives(tmp_path):
    cands = {"clip": {
        10: np.array([[100, 100, 10, 10, 0.9, 1], [400, 400, 10, 10, 0.6, 1]], np.float32),   # hit + 1 fp
        11: np.array([[500, 500, 10, 10, 0.9, 1]], np.float32),                                  # miss + 1 fp
        12: np.array([[300, 300, 10, 10, 0.4, 1]], np.float32),                                  # no ball, 1 fp
        13: np.zeros((0, 6), np.float32)}}                                                       # no ball, none
    gt = {"frames": {"clip/000010": {"box": [96, 96, 10, 10]}, "clip/000011": {"box": [10, 10, 10, 10]},
                     "clip/000012": {"box": None}, "clip/000013": {"box": None},
                     "clip/000014": {"box": None, "status": "unsure"}}}
    (tmp_path / "c.pkl").write_bytes(pickle.dumps(cands))
    (tmp_path / "gt.json").write_text(json.dumps(gt))
    out = tmp_path / "out.json"

    class A:
        pass
    a = A()
    a.cands, a.gt, a.thrs, a.out = str(tmp_path / "c.pkl"), str(tmp_path / "gt.json"), [0.5], str(out)
    ball_vl.cmd_eval(a)
    r = json.loads(out.read_text())["all@0.5"]
    assert (r["frames"], r["ball_visible"], r["not_visible"]) == (4, 2, 2)    # unsure frame is not scored
    assert r["recall_any"] == 0.5 and r["recall_top1"] == 0.5
    assert r["fp_per_frame"] == 0.5          # fp: the 0.6 box in frame 10 and the miss in frame 11 (0.4 is below 0.5)
