# Benchmark harness Implementation Plan (plan 1 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One command (`footlytics bench run --suite harness --remote iec4090_2`) ships the current
commit to the lab 4090, scores with the official SoccerNet evaluator, and appends a row to the
committed scoreboard. The harness is then proven: GSR ground truth through our adapters scores
about 100 GS-HOTA, and the published baseline re-scores to 22.26 on test.

**Architecture:** A `footlytics/bench/` package. The host side is a suite runner: tasks feed
adapters, adapters feed the evaluator subprocess, and everything ends in one result row. The M2
side is a remote driver over ssh and tmux. Third-party evaluators run as subprocesses in their
own venv and are never imported. Suites are YAML lists of typed tasks, so plans 2 and 3 add task
kinds without touching the loop.

**Tech Stack:** Python 3.11/3.12, typer, pandas, PyYAML, huggingface_hub, requests;
sn-trackeval (MIT) at a pinned commit; ssh + tmux.

**Spec:** `docs/superpowers/specs/2026-10-11-benchmark-design.md`

**Plan series.** The spec spans four independently testable subsystems, so it is split:
1. **This plan:** host, datasets, GSR adapters, evaluator wrapper, runner, remote driver,
   scoreboard, harness proof (spec build-order step 1).
2. Calibration: full-camera per-frame calibrator (SoccerMaster heads + PnLCalib solver),
   SN-Calibration adapter and evaluator, keyframing, 25 px gate, camera-tracker bridge
   (step 2).
3. Full-pipeline GSR: image-sequence input, quick/nightly/milestone suites, team/ball/tactical
   metrics, noise floor (step 3).
4. V.League mini-GT (step 4, waits for two more matches on the host).

## Global Constraints

- Host: `iec4090_2` (ssh alias for `iec@10.170.100.152`). Everything lives under `~/footlytics-bench/`; tmux sessions are named `footlytics-<run-id>`. Never touch other files, sessions, Docker volumes or images.
- Disk floor: downloads and runs stop below 1.5 GB free.
- Pipeline venv: `--system-site-packages`, must not install a second torch; check `torch.__version__ == "2.7.1+cu128"` and CUDA available (after install and at every run).
- Evaluators: run as subprocesses in `venv-eval`, never imported by footlytics.
- GSR data version >= 1.3 (`Labels-GameState.json` `info.version`).
- Primary score: GS-HOTA with jersey off; the official jersey-on score is reported next to it.
- Test split for milestones only, never for tuning thresholds.
- Every row records git SHA, suite, host, per-clip and aggregate numbers, and host contention (other processes' GPU memory, load average) at start and end.
- A clip that crashes is recorded as failed with its reason and the suite continues; a run with any failed clip or task is `partial`, never compared with complete runs.
- `bench/results/results.jsonl` and `bench/results/SCOREBOARD.md` are committed.
- Tests: `uv run --extra dev pytest`. Line length 100 (ruff).

## Facts verified while planning (2026-10-11)

- **sn-trackeval** `9c25232f6f2b56c9f203f1eb55784ff1e97df683` is MIT, not GPL (the spec said both
  were GPL). sn-calibration `ab38f461bec729fead86b6986839de1bb826f16d` has no licence file. Both
  still run as subprocesses.
- **Evaluator CLI** (ran locally on SNGS-117): `scripts/run_soccernet_gs.py --GT_FOLDER G
  --TRACKERS_FOLDER T --TRACKER_SUB_FOLDER "" --SPLIT_TO_EVAL test --SEQMAP_FILE seqmap.json
  --USE_JERSEY_NUMBERS True|False ...`. GT layout is `G/<split>/<SEQ>/Labels-GameState.json`;
  predictions are `T/SoccerNetGS-<split>/<tracker>/<SEQ>.json` = `{"predictions": [...]}`.
  The seqmap is `[{"name": "SNGS-117", "nframes": 750}]`. Results land in
  `<OUTPUT_FOLDER>/<tracker>/person_detailed.csv`: one row per sequence plus `COMBINED`, with
  `HOTA___AUC`, `DetA___AUC`, `AssA___AUC`, `LocA___AUC` and `IDF1` as fractions. Only
  supercategory `object` is read, the ball is ignored, and the similarity uses
  `bbox_pitch.x/y_bottom_middle` only (but all six keys must exist).
- **Baseline predictions:** sn-gamestate `1c958345067218297d221e45e1a6405f975f83e0` ships
  `examples_predictions/SoccerNetGS-test.zip` (49 test clips, tracker folder `tracklab/`,
  sha256 `b51bea37ebd036ed95bd7cd89eea2d46134b6e3a704c02a7a30eaf6d91a734a5`). SNGS-117 alone
  scores 24.16 jersey-on, which is consistent with 22.26 overall. We use this JSON instead of
  the Zenodo tracker-state `.pklz`, which can only be read by unpickling; Zenodo stays the
  fallback if the re-score misses.
- **GSR zips** (HF `SoccerNet/SN-GSR-2025` revision `ce84ee7d9acb2f9fe999ceed2b38e303ffa5efb9`):
  members are `SNGS-033/Labels-GameState.json` and `SNGS-033/img1/000001.jpg` (DEFLATE). Image
  ids are `f"{split digit}{clip:03d}{frame:06d}"` (train 1, valid 2, test 3, challenge 4),
  frames 1-based. Pitch coords match the Match State: GK `right` at x = +49.7; near-side
  referee at y = +34.5. Team is `left`/`right` (the goal the team defends); jersey is a string
  or null; some GT boxes have `bbox_pitch: null` (in unlabelled frames).
- **SN-Calibration** HF revision `bb74e1a888e33f839bcf8b7c6b87604a770f1da4` (plan 2 uses it;
  the zip is fetched here).
- **Host Python:** `uv pip` ignores system site-packages and would install torch 2.14.1; `pip`
  inside a `uv venv --seed --system-site-packages` venv sees torch 2.7.1+cu128 as satisfied.
  That torch lives in the shared account's `~/.local`, so another user could change it: hence
  the per-run torch check. uv 0.13.0 is installed at `~/footlytics-bench/bin/uv`.

## Review Focus

- **SSH drop mid-run** → the run continues in tmux; `bench run` exits with code 2 and says to reconnect and run `bench resume <run-id>`. Resuming after the row was already appended must not append it twice. (Task 7 tests both.)
- **A clip whose labels are corrupt or below v1.3** → that clip is failed with its reason, the others are scored, and the run is `partial`. (Task 6 test.)
- **The evaluator exits non-zero or writes no CSV** → the task fails with the evaluator's last output lines, not a bare `CalledProcessError`. (Task 3 test.)
- **The disk is near the floor when a download starts** → it refuses before writing a byte and names the free space and the need. (Task 5 test.)
- **Someone upgrades torch in `~/.local` between runs** → the run fails at preflight with "torch changed", before any task runs. (Task 6 test.)

## File structure

```
footlytics/bench/__init__.py         package docstring only
footlytics/bench/paths.py            BenchPaths: every host path, from FOOTLYTICS_BENCH_ROOT
footlytics/bench/host.py             disk / VRAM checks, contention readings, torch info
footlytics/bench/adapters/__init__.py
footlytics/bench/adapters/gsr.py     GSR labels <-> Match State, left/right team mapping
footlytics/bench/evaluators.py       sn-trackeval subprocess + CSV parser
footlytics/bench/results.py          results.jsonl append, SCOREBOARD.md render
footlytics/bench/datasets.py         HF / URL fetch with sha256, label extraction
footlytics/bench/runner.py           suite loop, task kinds gsr_gt_roundtrip / gsr_rescore
footlytics/bench/suites/harness.yaml
footlytics/bench/remote.py           M2 side: ship, setup, start, status, result over ssh
footlytics/bench/host_setup.sh       idempotent host bootstrap
footlytics/bench/cli.py              `footlytics bench ...`
footlytics/cli.py                    + app.add_typer(bench_app, name="bench")
pyproject.toml                       + pyyaml, + package data for suites/*.yaml and host_setup.sh
bench/results/results.jsonl, bench/results/SCOREBOARD.md
tests/bench_fixtures.py              synthetic GSR labels
tests/test_bench_{host,gsr,evaluators,results,datasets,runner,remote}.py
tests/fixtures/bench/person_detailed.csv   real evaluator output (SNGS-117 baseline)
```

---

### Task 1: Paths and host checks

**Files:**
- Create: `footlytics/bench/__init__.py`, `footlytics/bench/paths.py`, `footlytics/bench/host.py`
- Test: `tests/test_bench_host.py`

**Interfaces:**
- Produces: `BenchPaths(root: Path)` with `.from_env()`, `.data`, `.gsr_zip(split)`,
  `.gsr_labels` (GT_FOLDER root), `.baseline_zip`, `.runs`, `.run_dir(run_id)`, `.src`, `.venv`,
  `.venv_eval`, `.eval_python`, `.ext`, `.trackeval`, `.cache`, `.torch_pin`.
  `host.FLOOR_BYTES`, `host.ResourceError`, `host.free_disk_bytes(path) -> int`,
  `host.require_disk(path, need_bytes, floor=FLOOR_BYTES)`,
  `host.parse_gpu(gpu_csv: str, apps_csv: str, own_pids) -> dict`,
  `host.contention(own_pids=None) -> dict`, `host.require_vram(need_mib, readings=None)`,
  `host.torch_info() -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
"""Host checks: disk floor, GPU contention parsing, VRAM gate."""
import pytest

from footlytics.bench import host
from footlytics.bench.paths import BenchPaths


def test_paths_live_under_one_root(tmp_path, monkeypatch):
    monkeypatch.setenv("FOOTLYTICS_BENCH_ROOT", str(tmp_path))
    p = BenchPaths.from_env()
    assert p.root == tmp_path
    for q in (p.gsr_zip("valid"), p.gsr_labels, p.baseline_zip, p.run_dir("r1"), p.eval_python,
              p.trackeval, p.torch_pin):
        assert tmp_path in q.parents


def test_require_disk_refuses_below_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "free_disk_bytes", lambda path: 3_000_000_000)
    host.require_disk(tmp_path, 1_000_000_000)                 # 2 GB left > 1.5 GB floor
    with pytest.raises(host.ResourceError, match="free"):
        host.require_disk(tmp_path, 2_000_000_000)             # 1 GB left < floor


def test_free_disk_walks_up_to_an_existing_parent(tmp_path):
    assert host.free_disk_bytes(tmp_path / "not" / "yet") > 0


GPU = "2687, 24564\n"
APPS = "1661182, 2626\n4242, 900\n"


def test_parse_gpu_separates_other_users():
    r = host.parse_gpu(GPU, APPS, own_pids={4242})
    assert r == {"gpu_used_mib": 2687, "gpu_total_mib": 24564, "gpu_other_mib": 2626,
                 "gpu_procs": 2}


def test_require_vram_names_the_shortfall():
    readings = {"gpu_used_mib": 20000, "gpu_total_mib": 24564, "gpu_other_mib": 20000,
                "gpu_procs": 1}
    host.require_vram(4000, readings)
    with pytest.raises(host.ResourceError, match="VRAM"):
        host.require_vram(8000, readings)


def test_contention_without_nvidia_smi_still_reports_load(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(host.subprocess, "run", boom)
    r = host.contention()
    assert "load1" in r and r["gpu_used_mib"] is None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra dev pytest tests/test_bench_host.py -q`
Expected: FAIL, `ModuleNotFoundError: footlytics.bench`

- [ ] **Step 3: Implement**

`footlytics/bench/__init__.py`:
```python
"""Benchmark harness: our output in SoccerNet formats, scored by the official evaluators."""
```

`footlytics/bench/paths.py`:
```python
"""Where the benchmark keeps things on the host: one root, everything under it.

The host is a shared lab account, so nothing is written outside the root.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = "~/footlytics-bench"


@dataclass(frozen=True)
class BenchPaths:
    root: Path

    @classmethod
    def from_env(cls) -> "BenchPaths":
        return cls(Path(os.getenv("FOOTLYTICS_BENCH_ROOT", DEFAULT_ROOT)).expanduser())

    @property
    def data(self) -> Path:
        return self.root / "data"

    def gsr_zip(self, split: str) -> Path:
        return self.data / "gsr" / f"{split}.zip"

    @property
    def gsr_labels(self) -> Path:
        """The evaluator's GT_FOLDER: `<split>/<SEQ>/Labels-GameState.json` below it."""
        return self.data / "gsr" / "labels"

    @property
    def baseline_zip(self) -> Path:
        return self.data / "baseline" / "SoccerNetGS-test.zip"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    def run_dir(self, run_id: str) -> Path:
        return self.runs / run_id

    @property
    def src(self) -> Path:
        return self.root / "src"

    @property
    def venv(self) -> Path:
        return self.root / "venv"

    @property
    def venv_eval(self) -> Path:
        return self.root / "venv-eval"

    @property
    def eval_python(self) -> Path:
        return self.venv_eval / "bin" / "python"

    @property
    def ext(self) -> Path:
        return self.root / "ext"

    @property
    def trackeval(self) -> Path:
        return self.ext / "sn-trackeval"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def torch_pin(self) -> Path:
        """Torch version recorded by host_setup.sh; every run checks it still holds."""
        return self.venv / ".torch-version"
```

`footlytics/bench/host.py`:
```python
"""Host checks for a shared machine: disk floor, free VRAM, and who else is on the GPU.

Other people run jobs on the same 4090, so seconds-per-minute is noisy and VRAM is not ours
alone. Every run records the contention it ran under, and refuses to start rather than die
mid-clip when the room is not there.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional

FLOOR_BYTES = 1_500_000_000


class ResourceError(RuntimeError):
    """Not enough disk or VRAM to start; the message says how much there is and is needed."""


def free_disk_bytes(path: str | Path) -> int:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def require_disk(path: str | Path, need_bytes: int, floor: int = FLOOR_BYTES) -> None:
    free = free_disk_bytes(path)
    if free - need_bytes < floor:
        raise ResourceError(
            f"{free / 1e9:.1f} GB free at {path}, need {need_bytes / 1e9:.1f} GB and the disk "
            f"must keep {floor / 1e9:.1f} GB free (shared machine)")


def parse_gpu(gpu_csv: str, apps_csv: str, own_pids: Iterable[int] = ()) -> dict:
    used, total = (int(float(v)) for v in gpu_csv.strip().splitlines()[0].split(","))
    own = set(own_pids)
    procs = [tuple(int(float(v)) for v in line.split(",")) for line in apps_csv.strip().splitlines()
             if line.strip()]
    other = sum(mib for pid, mib in procs if pid not in own)
    return {"gpu_used_mib": used, "gpu_total_mib": total, "gpu_other_mib": other,
            "gpu_procs": len(procs)}


def _smi(query: str) -> str:
    return subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True, timeout=20).stdout


def contention(own_pids: Optional[Iterable[int]] = None) -> dict:
    load1, load5, _ = os.getloadavg()
    out = {"load1": round(load1, 2), "load5": round(load5, 2)}
    try:
        gpu = parse_gpu(_smi("--query-gpu=memory.used,memory.total"),
                        _smi("--query-compute-apps=pid,used_memory"),
                        own_pids if own_pids is not None else {os.getpid()})
    except (OSError, subprocess.SubprocessError, ValueError):
        gpu = {"gpu_used_mib": None, "gpu_total_mib": None, "gpu_other_mib": None,
               "gpu_procs": None}
    return {**out, **gpu}


def require_vram(need_mib: int, readings: Optional[dict] = None) -> None:
    r = readings or contention()
    if r["gpu_total_mib"] is None:
        raise ResourceError("no GPU readings (nvidia-smi failed)")
    free = r["gpu_total_mib"] - r["gpu_used_mib"]
    if free < need_mib:
        raise ResourceError(f"{free} MiB VRAM free, need {need_mib} MiB "
                            f"({r['gpu_other_mib']} MiB held by other processes)")


def torch_info() -> dict:
    import torch

    return {"torch": torch.__version__, "cuda": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run --extra dev pytest tests/test_bench_host.py -q` → PASS

- [ ] **Step 5: Commit**

```bash
git add footlytics/bench/__init__.py footlytics/bench/paths.py footlytics/bench/host.py tests/test_bench_host.py
git commit -m "feat(bench): host paths, disk floor, VRAM gate, GPU contention readings"
```

---

### Task 2: GSR adapter

**Files:**
- Create: `footlytics/bench/adapters/__init__.py` (empty), `footlytics/bench/adapters/gsr.py`, `tests/bench_fixtures.py`
- Test: `tests/test_bench_gsr.py`

**Interfaces:**
- Consumes: `footlytics.state.schema.MatchState, MatchMeta, Role, Team`.
- Produces: `SPLIT_DIGIT`, `CATEGORY_ID`, `image_id(split, seq, frame_idx) -> str`,
  `check_version(labels: dict) -> str`, `gt_to_state(labels: dict) -> MatchState`,
  `team_sides(tracks: pd.DataFrame) -> dict[str, str]`,
  `state_to_predictions(state, split, use_jersey=False) -> dict`.
  `tests.bench_fixtures.make_labels(seq="SNGS-990", split="valid", n_frames=3, version="1.3") -> dict`.

- [ ] **Step 1: Write the fixture and failing tests**

`tests/bench_fixtures.py`:
```python
"""Synthetic GSR labels: the shape of Labels-GameState.json v1.3, none of its content."""
from footlytics.bench.adapters.gsr import CATEGORY_ID, image_id

#: (track_id, role, team, jersey, x, y): left team defends x < 0
PEOPLE = [
    (1, "goalkeeper", "left", None, -50.0, 0.0),
    (2, "player", "left", "4", -20.0, -10.0),
    (3, "player", "left", "8", -15.0, 12.0),
    (4, "goalkeeper", "right", None, 49.0, 2.0),
    (5, "player", "right", "9", 10.0, -5.0),
    (6, "player", "right", None, 18.0, 20.0),
    (7, "referee", None, None, 0.0, 30.0),
    (8, "ball", None, None, 1.0, 1.0),
    (9, "other", None, None, 5.0, 36.0),
]


def make_labels(seq="SNGS-990", split="valid", n_frames=3, version="1.3"):
    images, annotations = [], []
    for f in range(n_frames):
        iid = image_id(split, seq, f)
        images.append({"is_labeled": True, "image_id": iid, "file_name": f"{f + 1:06d}.jpg",
                       "height": 1080, "width": 1920, "has_labeled_person": True,
                       "has_labeled_pitch": True, "has_labeled_camera": True})
        for tid, role, team, jersey, x, y in PEOPLE:
            x += 0.1 * f
            annotations.append({
                "id": f"{iid}{tid:02d}", "image_id": iid, "track_id": tid,
                "supercategory": "object", "category_id": CATEGORY_ID[role],
                "attributes": {"role": role, "jersey": jersey, "team": team},
                "bbox_image": {"x": 100 + 10 * tid, "y": 500, "w": 20, "h": 60},
                "bbox_pitch": {"x_bottom_left": x - 0.3, "y_bottom_left": y,
                               "x_bottom_middle": x, "y_bottom_middle": y,
                               "x_bottom_right": x + 0.3, "y_bottom_right": y}})
        annotations.append({"id": f"{iid}99", "image_id": iid, "supercategory": "pitch",
                            "category_id": 5, "lines": {}})
    # one person seen by the camera but not placed on the pitch
    annotations.append({"id": "x", "image_id": images[0]["image_id"], "track_id": 10,
                        "supercategory": "object", "category_id": 1,
                        "attributes": {"role": "player", "jersey": None, "team": "left"},
                        "bbox_image": {"x": 1, "y": 1, "w": 1, "h": 1}, "bbox_pitch": None})
    info = {"version": version, "name": seq, "id": seq.split("-")[-1], "frame_rate": 25,
            "seq_length": n_frames, "im_dir": "img1", "im_ext": ".jpg"}
    return {"info": info, "images": images, "annotations": annotations, "categories": []}
```

`tests/test_bench_gsr.py`:
```python
"""GSR adapter: ground truth -> Match State -> predictions is lossless for what GS-HOTA scores."""
import numpy as np
import pytest

from footlytics.bench.adapters.gsr import (gt_to_state, image_id, state_to_predictions,
                                           team_sides)
from footlytics.state.schema import Role, Team
from tests.bench_fixtures import PEOPLE, make_labels


def test_image_id_is_split_clip_frame_one_based():
    assert image_id("valid", "SNGS-033", 0) == "2033000001"
    assert image_id("test", "SNGS-117", 749) == "3117000750"


def test_gt_to_state_maps_roles_teams_frames():
    s = gt_to_state(make_labels(n_frames=2))
    t = s.tracks
    assert s.meta.match_id == "SNGS-990" and s.meta.source_type == "broadcast"
    assert sorted(t["frame_idx"].unique()) == [0, 1]
    assert len(t) == 2 * len(PEOPLE)                       # the null-pitch box is dropped
    gk = t[t["track_id"] == 1].iloc[0]
    assert gk["role"] == Role.GOALKEEPER.value and gk["team"] == Team.HOME.value
    assert t[t["track_id"] == 4].iloc[0]["team"] == Team.AWAY.value
    assert t[t["track_id"] == 7].iloc[0]["team"] == Team.OFFICIAL.value
    assert t[t["track_id"] == 9].iloc[0]["role"] == Role.OTHER.value
    assert t[t["track_id"] == 2].iloc[0]["jersey"] == 4.0
    assert np.isnan(t[t["track_id"] == 6].iloc[0]["jersey"])


def test_old_label_version_is_refused():
    with pytest.raises(ValueError, match="1.3"):
        gt_to_state(make_labels(version="1.2"))


def test_team_sides_from_goalkeepers_then_mean_x():
    t = gt_to_state(make_labels()).tracks
    assert team_sides(t) == {"home": "left", "away": "right"}
    swapped = t.copy()
    swapped["x"] = -swapped["x"]                           # camera from the other side
    assert team_sides(swapped) == {"home": "right", "away": "left"}
    no_gk = t[t["role"] != Role.GOALKEEPER.value]
    assert team_sides(no_gk) == {"home": "left", "away": "right"}
    one_team = t[t["team"] == Team.AWAY.value]
    assert team_sides(one_team) == {"away": "right"}


def test_roundtrip_predictions_match_ground_truth():
    labels = make_labels(n_frames=2)
    pred = state_to_predictions(gt_to_state(labels), "valid", use_jersey=True)["predictions"]
    gt = {(a["image_id"], a["track_id"]): a for a in labels["annotations"]
          if a["supercategory"] == "object" and a["bbox_pitch"] is not None}
    assert len(pred) == len(gt)
    for p in pred:
        g = gt[(p["image_id"], p["track_id"])]
        assert p["supercategory"] == "object" and p["category_id"] == g["category_id"]
        assert p["attributes"]["role"] == g["attributes"]["role"]
        assert p["attributes"]["team"] == g["attributes"]["team"]
        assert p["attributes"]["jersey"] == (g["attributes"]["jersey"]
                                             if g["attributes"]["role"] == "player" else None)
        for k in ("x_bottom_middle", "y_bottom_middle"):
            assert abs(p["bbox_pitch"][k] - g["bbox_pitch"][k]) < 1e-4
        assert set(p["bbox_pitch"]) == set(g["bbox_pitch"])


def test_jersey_off_writes_null_jerseys():
    pred = state_to_predictions(gt_to_state(make_labels()), "valid")["predictions"]
    assert all(p["attributes"]["jersey"] is None for p in pred)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra dev pytest tests/test_bench_gsr.py -q` → FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `footlytics/bench/adapters/gsr.py`**

```python
"""SoccerNet Game State Reconstruction (GSR) <-> Match State.

GSR pitch coordinates are the Match State's: metres from the centre spot, x along the length
(negative = the goal on the camera's left), y negative towards the far touchline. Positions
pass through unchanged. What needs care is identity:

- frames: GSR images are 1-based (`000001.jpg`), Match State `frame_idx` is 0-based;
- image ids: `f"{split digit}{clip:03d}{frame:06d}"`, e.g. "2033000001";
- teams: GSR names a team by the goal it defends in the clip ("left"/"right"), not home/away.
  Our side is decided from our own positions (`team_sides`), never from ground truth.

GS-HOTA matches on the bottom-middle point only, so predictions carry it in all three
bottom-point slots.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ...state.schema import MatchMeta, MatchState, Role, Team

SPLIT_DIGIT = {"train": 1, "valid": 2, "test": 3, "challenge": 4}
CATEGORY_ID = {"player": 1, "goalkeeper": 2, "referee": 3, "ball": 4, "other": 7}
MIN_VERSION = (1, 3)

_ROLE = {"player": Role.PLAYER.value, "goalkeeper": Role.GOALKEEPER.value,
         "referee": Role.REFEREE.value, "ball": Role.BALL.value}
_TEAM = {"left": Team.HOME.value, "right": Team.AWAY.value}
_PLAYING = (Role.PLAYER.value, Role.GOALKEEPER.value)


def image_id(split: str, seq: str, frame_idx: int) -> str:
    return f"{SPLIT_DIGIT[split]}{int(seq.split('-')[-1]):03d}{frame_idx + 1:06d}"


def check_version(labels: dict) -> str:
    v = str(labels["info"]["version"])
    if tuple(int(p) for p in v.split(".")) < MIN_VERSION:
        raise ValueError(f"GSR labels v{v} for {labels['info'].get('name')}: need >= 1.3 "
                         "(earlier versions have the old bbox_pitch and team labels)")
    return v


def gt_to_state(labels: dict) -> MatchState:
    """Ground truth as a Match State: left team -> HOME, right -> AWAY, officials -> OFFICIAL."""
    info = labels["info"]
    version = check_version(labels)
    fps = float(info.get("frame_rate", 25))
    frame_of = {img["image_id"]: int(Path(img["file_name"]).stem) - 1 for img in labels["images"]}
    rows = []
    for a in labels["annotations"]:
        if a.get("supercategory") != "object" or a.get("bbox_pitch") is None:
            continue
        attrs = a["attributes"]
        role = _ROLE.get(attrs.get("role"), Role.OTHER.value)
        if role == Role.REFEREE.value:
            team = Team.OFFICIAL.value
        else:
            team = _TEAM.get(attrs.get("team"), Team.UNKNOWN.value)
        jersey = attrs.get("jersey")
        f = frame_of[a["image_id"]]
        bi, bp = a.get("bbox_image") or {}, a["bbox_pitch"]
        rows.append(dict(
            frame_idx=f, period=1, timestamp=f / fps, track_id=int(a["track_id"]), role=role,
            team=team, jersey=float(jersey) if jersey not in (None, "") else np.nan,
            bbox_x=bi.get("x"), bbox_y=bi.get("y"), bbox_w=bi.get("w"), bbox_h=bi.get("h"),
            det_conf=1.0, x=bp["x_bottom_middle"], y=bp["y_bottom_middle"],
            z=0.0 if role == Role.BALL.value else np.nan))
    meta = MatchMeta(match_id=info["name"], fps=fps, source_type="broadcast",
                     provenance={"source": "gsr-ground-truth", "gsr_version": version})
    return MatchState(meta, pd.DataFrame(rows))


def team_sides(tracks: pd.DataFrame) -> dict[str, str]:
    """HOME/AWAY -> "left"/"right" for one clip, from our own positions.

    A team defends the goal its keeper stands in front of, so a goalkeeper sighting decides;
    without one, the team's mean x. With both teams the lower x is left; a lone team goes by
    the sign of its x.
    """
    people = tracks[tracks["role"].isin(_PLAYING)
                    & tracks["team"].isin([Team.HOME.value, Team.AWAY.value])]
    xs: dict[str, float] = {}
    for team, g in people.groupby("team", observed=True):
        gk = g[g["role"] == Role.GOALKEEPER.value]
        xs[str(team)] = float((gk if len(gk) else g)["x"].mean())
    if len(xs) == 2:
        home_left = xs[Team.HOME.value] <= xs[Team.AWAY.value]
        return {Team.HOME.value: "left" if home_left else "right",
                Team.AWAY.value: "right" if home_left else "left"}
    return {t: ("left" if x < 0 else "right") for t, x in xs.items()}


def _num(v) -> float | None:
    return float(v) if v is not None and np.isfinite(v) else None


def state_to_predictions(state: MatchState, split: str, use_jersey: bool = False) -> dict:
    """A Match State as one GSR predictions file (`{"predictions": [...]}`).

    Team and jersey are null for anyone not playing, as the evaluator requires; jersey is
    null throughout unless `use_jersey` (we have no shirt-number reader yet).
    """
    seq = state.meta.match_id
    sides = team_sides(state.tracks)
    preds = []
    for r in state.tracks.itertuples(index=False):
        x, y = _num(r.x), _num(r.y)
        if x is None or y is None:
            continue
        role = str(r.role) if str(r.role) in CATEGORY_ID else "other"
        team = sides.get(str(r.team)) if role in _PLAYING else None
        jersey = (str(int(r.jersey)) if use_jersey and role == Role.PLAYER.value
                  and _num(r.jersey) is not None else None)
        box = [_num(v) for v in (r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h)]
        preds.append({
            "id": str(len(preds)), "image_id": image_id(split, seq, int(r.frame_idx)),
            "video_id": seq.split("-")[-1], "track_id": int(r.track_id),
            "supercategory": "object", "category_id": CATEGORY_ID[role],
            "confidence": _num(r.det_conf) if _num(r.det_conf) is not None else 1.0,
            "bbox_image": (dict(zip(("x", "y", "w", "h"), box))
                           if all(v is not None for v in box) else None),
            "bbox_pitch": {f"{a}_bottom_{s}": v for s in ("left", "middle", "right")
                           for a, v in (("x", x), ("y", y))},
            "attributes": {"role": role, "team": team, "jersey": jersey},
        })
    return {"predictions": preds}
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run --extra dev pytest tests/test_bench_gsr.py -q` → PASS

- [ ] **Step 5: Commit**

```bash
git add footlytics/bench/adapters tests/bench_fixtures.py tests/test_bench_gsr.py
git commit -m "feat(bench): GSR labels <-> Match State, left/right from our own positions"
```

---

### Task 3: Evaluator wrapper (sn-trackeval)

**Files:**
- Create: `footlytics/bench/evaluators.py`, `tests/fixtures/bench/person_detailed.csv` (copy of the real SNGS-117 output from the planning run)
- Test: `tests/test_bench_evaluators.py`

**Interfaces:**
- Consumes: `BenchPaths.eval_python`, `BenchPaths.trackeval`.
- Produces: `EvaluatorError`, `GsrScores(clips: dict[str, dict], combined: dict)`,
  `parse_detailed(path) -> GsrScores`,
  `trackeval_command(python, trackeval_dir, gt_root, trackers_root, split, tracker, seqmap, out_dir, use_jersey) -> list[str]`,
  `run_gsr(python, trackeval_dir, gt_root, trackers_root, split, tracker, seqs, out_dir, use_jersey, run=subprocess.run) -> GsrScores`.
  Score keys: `hota, deta, assa, loca, idf1` (percent), `dets, gt_dets` (int).

- [ ] **Step 1: Write the failing tests**

```python
"""sn-trackeval wrapper: command line, CSV parsing, failure reporting."""
import json
import os
import shutil
import subprocess
import sys
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
    assert s.combined["gt_dets"] == 6847


def test_command_switches_jersey_and_pins_split(tmp_path):
    cmd = trackeval_command(Path("/py"), Path("/te"), Path("/gt"), Path("/tr"), "valid", "fl",
                            tmp_path / "seqmap.json", tmp_path / "out", use_jersey=False)
    assert cmd[:2] == ["/py", "/te/scripts/run_soccernet_gs.py"]
    flags = dict(zip(cmd[2::2], cmd[3::2]))
    assert flags["--USE_JERSEY_NUMBERS"] == "False" and flags["--SPLIT_TO_EVAL"] == "valid"
    assert flags["--TRACKER_SUB_FOLDER"] == "" and flags["--TRACKERS_TO_EVAL"] == "fl"


def test_evaluator_failure_carries_its_output(tmp_path):
    def fake(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, stdout="loading\n", stderr="GT file not found")
    with pytest.raises(EvaluatorError, match="GT file not found"):
        run_gsr(Path("/py"), Path("/te"), tmp_path, tmp_path, "valid", "fl", ["SNGS-990"],
                tmp_path / "out", use_jersey=False, run=fake)


def test_evaluator_without_csv_is_an_error(tmp_path):
    def fake(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout="done", stderr="")
    with pytest.raises(EvaluatorError, match="person_detailed.csv"):
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
    for jersey in (False, True):
        s = run_gsr(Path(EVAL_PY), Path(TE), tmp_path / "gt", tmp_path / "tr", "valid", "fl",
                    ["SNGS-990"], tmp_path / f"out{jersey}", use_jersey=jersey)
        assert s.combined["hota"] > 99.0
```

- [ ] **Step 2: Run to verify failure** — `uv run --extra dev pytest tests/test_bench_evaluators.py -q` → FAIL

- [ ] **Step 3: Implement `footlytics/bench/evaluators.py`**

```python
"""Official evaluators, run as subprocesses in their own venv.

We do not reimplement GS-HOTA: our output is written in the SoccerNet format and scored by
sn-trackeval itself (pinned in host_setup.sh). Never import it; it lives in venv-eval.
"""
from __future__ import annotations

import csv
import json
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
                      split: str, tracker: str, seqmap: Path, out_dir: Path,
                      use_jersey: bool) -> list[str]:
    flags = {
        "--GT_FOLDER": gt_root, "--TRACKERS_FOLDER": trackers_root, "--TRACKER_SUB_FOLDER": "",
        "--TRACKERS_TO_EVAL": tracker, "--SPLIT_TO_EVAL": split, "--SEQMAP_FILE": seqmap,
        "--OUTPUT_FOLDER": out_dir, "--USE_JERSEY_NUMBERS": use_jersey,
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
                            seqmap, out_dir, use_jersey)
    p = run(cmd, capture_output=True, text=True, cwd=str(trackeval_dir))
    (out_dir / "evaluator.log").write_text(f"{' '.join(cmd)}\n\n{p.stdout}\n{p.stderr}")
    if p.returncode != 0:
        tail = "\n".join((p.stdout + "\n" + p.stderr).strip().splitlines()[-15:])
        raise EvaluatorError(f"sn-trackeval exited {p.returncode}:\n{tail}")
    found = sorted(out_dir.rglob("person_detailed.csv"))
    if len(found) != 1:
        raise EvaluatorError(f"expected one person_detailed.csv under {out_dir}, found "
                             f"{len(found)}; see {out_dir / 'evaluator.log'}")
    return parse_detailed(found[0])
```

- [ ] **Step 4: Run** — unit tests PASS. Then the integration test against the scratchpad
  checkout of sn-trackeval with a throwaway venv:
  `uv venv /tmp/te-venv && VIRTUAL_ENV=/tmp/te-venv uv pip install numpy scipy`, then
  `FOOTLYTICS_TRACKEVAL=<scratchpad>/ext/sn-trackeval FOOTLYTICS_EVAL_PYTHON=/tmp/te-venv/bin/python uv run --extra dev pytest tests/test_bench_evaluators.py -q`
  → all PASS, roundtrip > 99 on both settings.

- [ ] **Step 5: Commit**

```bash
git add footlytics/bench/evaluators.py tests/test_bench_evaluators.py tests/fixtures/bench/person_detailed.csv
git commit -m "feat(bench): sn-trackeval as a subprocess, GS-HOTA parsed per clip"
```

---

### Task 4: Results and scoreboard

**Files:**
- Create: `footlytics/bench/results.py`
- Test: `tests/test_bench_results.py`

**Interfaces:**
- Produces: `RESULTS_DIR` (repo `bench/results`), `append_result(path, row) -> bool` (False if
  the run id is already there), `load_results(path) -> list[dict]`,
  `headline(task: dict) -> tuple[str, float | None, bool | None]`,
  `render_scoreboard(rows) -> str`, `write_scoreboard(results_dir) -> Path`.
  Row schema (written by Task 6): `run_id, suite, git_sha, host, started_at, finished_at,
  status, env, contention{start,end}, tasks{task_id: {kind, status, seconds, aggregate?, clips?,
  failed?, error?, check?}}`.

- [ ] **Step 1: Write the failing tests**

```python
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


def test_scoreboard_newest_first_and_marks_partial():
    md = render_scoreboard([row("r1", finished="2026-10-12T10:00:00Z"),
                            row("r2", status="partial", hota=50.0, passed=False,
                                finished="2026-10-13T10:00:00Z")])
    assert "## harness" in md
    lines = [l for l in md.splitlines() if l.startswith("| 2026")]
    assert "partial" in lines[0] and "50.00 ✗" in lines[0]
    assert "99.97 ✓" in lines[1]
    assert "abcdef1" in lines[1]
```

- [ ] **Step 2: Run to verify failure** → FAIL

- [ ] **Step 3: Implement `footlytics/bench/results.py`**

```python
"""One row per run in bench/results/results.jsonl; SCOREBOARD.md is regenerated from it.

Both files are committed: the history of the numbers is part of the repository. A `partial`
run (any clip or task failed) is shown but never compared with complete runs.
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "bench" / "results"
HISTORY = 20


def load_results(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def append_result(path: str | Path, row: dict) -> bool:
    p = Path(path)
    if any(r["run_id"] == row["run_id"] for r in load_results(p)):
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return True


def headline(task: dict) -> tuple[str, float | None, bool | None]:
    if task.get("status") == "failed":
        return "error", None, None
    check = task.get("check") or {}
    agg = task.get("aggregate") or {}
    metric = check.get("metric") or next(iter(agg), "")
    return metric, agg.get(metric), check.get("passed")


def _cell(task: dict | None) -> str:
    if task is None:
        return ""
    metric, value, passed = headline(task)
    if value is None:
        return metric
    mark = {True: " ✓", False: " ✗"}.get(passed, "")
    return f"{metric} {value:.2f}{mark}"


def render_scoreboard(rows: list[dict]) -> str:
    out = ["# Benchmark scoreboard", "",
           "Regenerated from `results.jsonl` by `footlytics bench scoreboard`; do not edit.",
           "`partial` runs had a failed clip or task and are not comparable with complete runs.",
           ""]
    for suite in sorted({r["suite"] for r in rows}):
        runs = sorted((r for r in rows if r["suite"] == suite),
                      key=lambda r: r["finished_at"], reverse=True)[:HISTORY]
        task_ids = list(dict.fromkeys(t for r in runs for t in r["tasks"]))
        out += [f"## {suite}", "",
                "| finished (UTC) | commit | status | host | " + " | ".join(task_ids) + " |",
                "|---" * (4 + len(task_ids)) + "|"]
        for r in runs:
            when = r["finished_at"].replace("T", " ")[:16]
            cells = [_cell(r["tasks"].get(t)) for t in task_ids]
            out.append(f"| {when} | {r['git_sha'][:7]} | {r['status']} | {r['host']} | "
                       + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def write_scoreboard(results_dir: str | Path = RESULTS_DIR) -> Path:
    d = Path(results_dir)
    md = d / "SCOREBOARD.md"
    md.write_text(render_scoreboard(load_results(d / "results.jsonl")))
    return md
```

- [ ] **Step 4: Run** → PASS
- [ ] **Step 5: Commit** — `git add footlytics/bench/results.py tests/test_bench_results.py && git commit -m "feat(bench): results.jsonl rows and a generated scoreboard"`

---

### Task 5: Datasets

**Files:**
- Create: `footlytics/bench/datasets.py`
- Test: `tests/test_bench_datasets.py`

**Interfaces:**
- Consumes: `BenchPaths`, `host.require_disk`, `adapters.gsr.check_version`.
- Produces: `GSR_REPO, GSR_REVISION, CALIB_REPO, CALIB_REVISION, BASELINE_URL, BASELINE_SHA256`,
  `sha256_file(path) -> str`, `extract_gsr_labels(zip_path, out_dir) -> list[str]`,
  `clips(paths, split) -> list[str]`,
  `fetch_hf(repo, filename, revision, dest, api=None, download=None) -> Path`,
  `fetch_url(url, dest, sha256, get=requests.get) -> Path`, `FETCHABLE: dict[str, Callable[[BenchPaths], Path]]`,
  `fetch(paths, what) -> Path`.

- [ ] **Step 1: Write the failing tests**

```python
"""Downloads check disk and checksums; label extraction refuses old versions."""
import json
import zipfile
from types import SimpleNamespace

import pytest

from footlytics.bench import datasets, host
from footlytics.bench.paths import BenchPaths
from tests.bench_fixtures import make_labels


def make_zip(path, versions):
    with zipfile.ZipFile(path, "w") as z:
        for i, v in enumerate(versions):
            seq = f"SNGS-{990 + i}"
            z.writestr(f"{seq}/Labels-GameState.json", json.dumps(make_labels(seq=seq, version=v)))
            z.writestr(f"{seq}/img1/000001.jpg", b"\xff\xd8not-really")


def test_extract_labels_only(tmp_path):
    make_zip(tmp_path / "valid.zip", ["1.3", "1.3"])
    seqs = datasets.extract_gsr_labels(tmp_path / "valid.zip", tmp_path / "labels" / "valid")
    assert seqs == ["SNGS-990", "SNGS-991"]
    assert (tmp_path / "labels/valid/SNGS-990/Labels-GameState.json").exists()
    assert not list((tmp_path / "labels").rglob("*.jpg"))
    assert datasets.clips(BenchPaths(tmp_path / "root"), "valid") == []


def test_extract_refuses_old_labels(tmp_path):
    make_zip(tmp_path / "valid.zip", ["1.3", "1.2"])
    with pytest.raises(ValueError, match="1.3"):
        datasets.extract_gsr_labels(tmp_path / "valid.zip", tmp_path / "labels")


def test_fetch_hf_refuses_before_downloading_when_disk_is_short(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "free_disk_bytes", lambda p: 5_000_000_000)
    info = SimpleNamespace(size=11_000_000_000, lfs=SimpleNamespace(sha256="x"))
    api = SimpleNamespace(get_paths_info=lambda *a, **k: [info])
    called = []
    with pytest.raises(host.ResourceError, match="11.0 GB"):
        datasets.fetch_hf("r", "valid.zip", "rev", tmp_path / "valid.zip", api=api,
                          download=lambda **k: called.append(k))
    assert called == []


def test_fetch_hf_deletes_a_corrupt_download(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "free_disk_bytes", lambda p: 50_000_000_000)
    info = SimpleNamespace(size=5, lfs=SimpleNamespace(sha256="0" * 64))
    api = SimpleNamespace(get_paths_info=lambda *a, **k: [info])

    def download(**k):
        p = tmp_path / "valid.zip"
        p.write_bytes(b"hello")
        return str(p)
    with pytest.raises(ValueError, match="sha256"):
        datasets.fetch_hf("r", "valid.zip", "rev", tmp_path / "valid.zip", api=api,
                          download=download)
    assert not (tmp_path / "valid.zip").exists()


def test_fetch_hf_skips_a_verified_file(tmp_path, monkeypatch):
    p = tmp_path / "valid.zip"
    p.write_bytes(b"hello")
    sha = datasets.sha256_file(p)
    (tmp_path / "valid.zip.sha256").write_text(sha)
    info = SimpleNamespace(size=5, lfs=SimpleNamespace(sha256=sha))
    api = SimpleNamespace(get_paths_info=lambda *a, **k: [info])
    assert datasets.fetch_hf("r", "valid.zip", "rev", p, api=api,
                             download=lambda **k: pytest.fail("re-downloaded")) == p
```

- [ ] **Step 2: Run to verify failure** → FAIL

- [ ] **Step 3: Implement `footlytics/bench/datasets.py`**

```python
"""Locate or download the benchmark data on the host, pinned and checksummed.

GSR zips are kept whole and only their label files are extracted (images are read straight
out of the zip later), so the data is never held twice on a shared disk. Every download is
checked against the disk floor before it starts and against its sha256 after it ends.
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Callable

import requests

from . import host
from .adapters.gsr import check_version
from .paths import BenchPaths

GSR_REPO = "SoccerNet/SN-GSR-2025"
GSR_REVISION = "ce84ee7d9acb2f9fe999ceed2b38e303ffa5efb9"
CALIB_REPO = "SoccerNet/SN-Calibration-2023"
CALIB_REVISION = "bb74e1a888e33f839bcf8b7c6b87604a770f1da4"
#: The sn-gamestate baseline's own test predictions, as JSON (the Zenodo copy is a pickle).
BASELINE_URL = ("https://github.com/SoccerNet/sn-gamestate/raw/"
                "1c958345067218297d221e45e1a6405f975f83e0/examples_predictions/"
                "SoccerNetGS-test.zip")
BASELINE_SHA256 = "b51bea37ebd036ed95bd7cd89eea2d46134b6e3a704c02a7a30eaf6d91a734a5"

_LABELS = re.compile(r"^(SNGS-\d+)/Labels-GameState\.json$")


def sha256_file(path: str | Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _verified(dest: Path, sha: str) -> bool:
    marker = dest.with_name(dest.name + ".sha256")
    return dest.exists() and marker.exists() and marker.read_text().strip() == sha


def _verify(dest: Path, sha: str) -> Path:
    got = sha256_file(dest)
    if got != sha:
        dest.unlink()
        raise ValueError(f"sha256 mismatch for {dest.name}: got {got}, expected {sha}; deleted")
    dest.with_name(dest.name + ".sha256").write_text(sha)
    return dest


def fetch_hf(repo: str, filename: str, revision: str, dest: Path, api=None,
             download=None) -> Path:
    from huggingface_hub import HfApi, hf_hub_download

    api = api or HfApi()
    info = api.get_paths_info(repo, [filename], repo_type="dataset", revision=revision)[0]
    if _verified(dest, info.lfs.sha256):
        return dest
    host.require_disk(dest.parent, info.size)
    dest.parent.mkdir(parents=True, exist_ok=True)
    got = Path((download or hf_hub_download)(repo_id=repo, filename=filename,
                                             repo_type="dataset", revision=revision,
                                             local_dir=str(dest.parent)))
    if got != dest:
        got.replace(dest)
    return _verify(dest, info.lfs.sha256)


def fetch_url(url: str, dest: Path, sha256: str, get=requests.get) -> Path:
    if _verified(dest, sha256):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        host.require_disk(dest.parent, int(r.headers.get("Content-Length", 0)))
        with dest.open("wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
    return _verify(dest, sha256)


def extract_gsr_labels(zip_path: str | Path, out_dir: str | Path) -> list[str]:
    """Write `<out_dir>/<SEQ>/Labels-GameState.json` for every clip; images stay zipped."""
    out = Path(out_dir)
    seqs = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            m = _LABELS.match(name)
            if not m:
                continue
            raw = z.read(name)
            check_version(json.loads(raw))
            (out / m.group(1)).mkdir(parents=True, exist_ok=True)
            (out / m.group(1) / "Labels-GameState.json").write_bytes(raw)
            seqs.append(m.group(1))
    return sorted(seqs)


def clips(paths: BenchPaths, split: str) -> list[str]:
    d = paths.gsr_labels / split
    return sorted(p.name for p in d.iterdir() if (p / "Labels-GameState.json").exists()) \
        if d.exists() else []


def _gsr(split: str) -> Callable[[BenchPaths], Path]:
    def go(paths: BenchPaths) -> Path:
        z = fetch_hf(GSR_REPO, f"{split}.zip", GSR_REVISION, paths.gsr_zip(split))
        extract_gsr_labels(z, paths.gsr_labels / split)
        return z
    return go


def _calib(split: str) -> Callable[[BenchPaths], Path]:
    return lambda paths: fetch_hf(CALIB_REPO, f"{split}.zip", CALIB_REVISION,
                                  paths.data / "calibration" / f"{split}.zip")


FETCHABLE: dict[str, Callable[[BenchPaths], Path]] = {
    "gsr-valid": _gsr("valid"),
    "gsr-test": _gsr("test"),
    "calib-valid": _calib("valid"),
    "calib-test": _calib("test"),
    "baseline": lambda paths: fetch_url(BASELINE_URL, paths.baseline_zip, BASELINE_SHA256),
}


def fetch(paths: BenchPaths, what: str) -> Path:
    if what not in FETCHABLE:
        raise KeyError(f"unknown dataset {what!r}; one of {sorted(FETCHABLE)}")
    return FETCHABLE[what](paths)
```

- [ ] **Step 4: Run** → PASS
- [ ] **Step 5: Commit** — `git add footlytics/bench/datasets.py tests/test_bench_datasets.py && git commit -m "feat(bench): pinned, checksummed dataset fetch; GSR labels extracted, images left zipped"`

---

### Task 6: Runner and the harness suite

**Files:**
- Create: `footlytics/bench/runner.py`, `footlytics/bench/suites/harness.yaml`
- Modify: `pyproject.toml` (add `"pyyaml>=6"` to dependencies)
- Test: `tests/test_bench_runner.py`

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: `KINDS`, `kind(name)` decorator, `TaskContext(paths, work, log)`,
  `load_suite(name) -> dict`, `evaluate_check(check, aggregate) -> dict | None`,
  `run_status(tasks) -> str`, `preflight(paths) -> dict`,
  `run_suite(suite, run_id, paths, git_sha, host_label, suite_spec=None) -> dict`
  (writes `status.json` and `result.json` in the run dir), `write_status(run_dir, state, **extra)`,
  `prune_runs(paths, keep=5)`. Task kinds `gsr_gt_roundtrip` and `gsr_rescore`, which return
  `{"clips", "aggregate", "failed"}` with metric keys `hota_jersey_off, deta_jersey_off,
  assa_jersey_off, hota_jersey_on`.

- [ ] **Step 1: Write the failing tests**

```python
"""Suite loop: failure isolation, checks, status, preflight."""
import json

import pytest

from footlytics.bench import runner
from footlytics.bench.evaluators import GsrScores
from footlytics.bench.paths import BenchPaths
from tests.bench_fixtures import make_labels


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "contention", lambda: {"load1": 0.0})
    return BenchPaths(tmp_path)


def spec(*tasks):
    return {"name": "t", "tasks": list(tasks)}


def test_crashing_task_makes_run_partial_not_dead(paths, monkeypatch):
    monkeypatch.setitem(runner.KINDS, "ok", lambda task, ctx: {"aggregate": {"m": 1.0}})
    monkeypatch.setitem(runner.KINDS, "boom", lambda task, ctx: 1 / 0)
    row = runner.run_suite("t", "r1", paths, "sha", "h", suite_spec=spec(
        {"id": "a", "kind": "ok", "check": {"metric": "m", "min": 0.5}},
        {"id": "b", "kind": "boom"}))
    assert row["status"] == "partial"
    assert row["tasks"]["a"]["check"]["passed"] is True
    assert "ZeroDivisionError" in row["tasks"]["b"]["error"]
    assert json.loads((paths.run_dir("r1") / "status.json").read_text())["state"] == "partial"
    assert json.loads((paths.run_dir("r1") / "result.json").read_text())["run_id"] == "r1"


def test_checks_min_and_expected():
    assert runner.evaluate_check({"metric": "m", "min": 99}, {"m": 98.0})["passed"] is False
    c = runner.evaluate_check({"metric": "m", "expected": 22.26, "tolerance": 1.0}, {"m": 22.9})
    assert c["passed"] is True and c["delta"] == pytest.approx(0.64)
    assert runner.evaluate_check({"metric": "m", "min": 1}, {})["passed"] is False


def test_preflight_fails_when_torch_changed(paths, monkeypatch):
    paths.venv.mkdir(parents=True)
    paths.torch_pin.write_text("2.7.1+cu128\n")
    monkeypatch.setattr(runner, "torch_info", lambda: {"torch": "2.14.1", "cuda": True})
    monkeypatch.setitem(runner.KINDS, "ok", lambda task, ctx: pytest.fail("ran a task"))
    row = runner.run_suite("t", "r2", paths, "sha", "h", suite_spec=spec({"id": "a", "kind": "ok"}))
    assert row["status"] == "failed" and "torch changed" in row["error"]


def write_gt(paths, split, seqs, bad=()):
    for s in seqs:
        d = paths.gsr_labels / split / s
        d.mkdir(parents=True)
        labels = make_labels(seq=s, version="1.2" if s in bad else "1.3")
        (d / "Labels-GameState.json").write_text(json.dumps(labels))


def fake_scores(calls):
    def run_gsr(python, te, gt_root, trackers_root, split, tracker, seqs, out_dir, use_jersey):
        calls.append((split, tracker, tuple(seqs), use_jersey))
        for s in seqs:
            assert (trackers_root / f"SoccerNetGS-{split}" / tracker / f"{s}.json").exists()
        v = 99.9 if not use_jersey else 99.8
        return GsrScores({s: {"hota": v, "deta": v, "assa": v} for s in seqs},
                         {"hota": v, "deta": v, "assa": v})
    return run_gsr


def test_gt_roundtrip_isolates_a_bad_clip(paths, monkeypatch):
    write_gt(paths, "valid", ["SNGS-990", "SNGS-991"], bad={"SNGS-991"})
    calls = []
    monkeypatch.setattr(runner, "run_gsr", fake_scores(calls))
    row = runner.run_suite("t", "r3", paths, "sha", "h", suite_spec=spec(
        {"id": "rt", "kind": "gsr_gt_roundtrip", "split": "valid", "clips": "all",
         "check": {"metric": "hota_jersey_off", "min": 99.0}}))
    t = row["tasks"]["rt"]
    assert row["status"] == "partial" and t["status"] == "partial"
    assert "1.3" in t["failed"]["SNGS-991"]
    assert t["aggregate"]["hota_jersey_off"] == 99.9 and t["aggregate"]["hota_jersey_on"] == 99.8
    assert t["clips"]["SNGS-990"]["hota_jersey_off"] == 99.9
    assert {c[3] for c in calls} == {False, True} and all(c[2] == ("SNGS-990",) for c in calls)


def test_rescore_reads_tracker_files_from_the_zip(paths, monkeypatch):
    import zipfile
    write_gt(paths, "test", ["SNGS-990", "SNGS-991"])
    paths.baseline_zip.parent.mkdir(parents=True)
    with zipfile.ZipFile(paths.baseline_zip, "w") as z:
        z.writestr("tracklab/SNGS-990.json", json.dumps({"predictions": []}))
    calls = []
    monkeypatch.setattr(runner, "run_gsr", fake_scores(calls))
    row = runner.run_suite("t", "r4", paths, "sha", "h", suite_spec=spec(
        {"id": "bl", "kind": "gsr_rescore", "split": "test", "clips": "all",
         "predictions": "baseline/SoccerNetGS-test.zip", "tracker": "tracklab"}))
    t = row["tasks"]["bl"]
    assert t["failed"] == {"SNGS-991": "no predictions in baseline/SoccerNetGS-test.zip"}
    assert calls[0][1] == "tracklab"


def test_prune_keeps_the_last_runs_work(paths):
    for i in range(7):
        w = paths.run_dir(f"2026101{i}-000000-t-sha") / "work"
        w.mkdir(parents=True)
        (w.parent / "result.json").write_text("{}")
    runner.prune_runs(paths, keep=5)
    kept = sorted(p.parent.name for p in paths.runs.glob("*/work"))
    assert len(kept) == 5 and kept[0].startswith("20261012")
    assert len(list(paths.runs.glob("*/result.json"))) == 7


def test_harness_suite_is_valid():
    s = runner.load_suite("harness")
    assert {t["kind"] for t in s["tasks"]} <= set(runner.KINDS)
```

- [ ] **Step 2: Run to verify failure** → FAIL

- [ ] **Step 3: Implement**

`footlytics/bench/suites/harness.yaml`:
```yaml
# Proves the harness before any number is trusted (spec: "Proving the harness").
name: harness
tasks:
  - id: gsr-gt-roundtrip-valid
    kind: gsr_gt_roundtrip
    split: valid
    clips: all
    check: {metric: hota_jersey_off, min: 99.0}
  - id: gsr-baseline-test
    kind: gsr_rescore
    split: test
    clips: all
    predictions: baseline/SoccerNetGS-test.zip
    tracker: tracklab
    check: {metric: hota_jersey_on, expected: 22.26, tolerance: 1.0}
```

`footlytics/bench/runner.py`:
```python
"""Suite loop on the host: tasks -> adapters -> evaluators -> one result row.

A suite is a YAML list of typed tasks (`suites/<name>.yaml`); each kind is a function
registered in `KINDS`. A clip that fails is recorded with its reason and the task goes on; a
task that fails is recorded and the suite goes on. Any failure makes the run `partial`.
"""
from __future__ import annotations

import json
import shutil
import time
import traceback
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml

from . import datasets
from .adapters.gsr import gt_to_state, state_to_predictions
from .evaluators import run_gsr
from .host import ResourceError, contention, torch_info
from .paths import BenchPaths

SUITES_DIR = Path(__file__).parent / "suites"
DONE = ("complete", "partial", "failed")
KINDS: dict[str, Callable[[dict, "TaskContext"], dict]] = {}


@dataclass
class TaskContext:
    paths: BenchPaths
    work: Path
    log: Callable[[str], None]


def kind(name: str):
    def register(fn):
        KINDS[name] = fn
        return fn
    return register


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    print(f"[bench {_now()}] {msg}", flush=True)


def load_suite(name: str) -> dict:
    return yaml.safe_load((SUITES_DIR / f"{name}.yaml").read_text())


def write_status(run_dir: Path, state: str, **extra) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / "status.json.tmp"
    tmp.write_text(json.dumps({"state": state, "updated": _now(), **extra}))
    tmp.replace(run_dir / "status.json")


def evaluate_check(check: dict | None, aggregate: dict) -> dict | None:
    if not check:
        return None
    value = aggregate.get(check["metric"])
    out = {**check, "value": value}
    if value is None:
        return {**out, "passed": False}
    if "min" in check:
        return {**out, "passed": value >= check["min"]}
    delta = value - check["expected"]
    return {**out, "delta": round(delta, 3), "passed": abs(delta) <= check["tolerance"]}


def run_status(tasks: dict) -> str:
    states = [t["status"] for t in tasks.values()]
    if states and all(s == "complete" for s in states):
        return "complete"
    if states and all(s == "failed" for s in states):
        return "failed"
    return "partial"


def preflight(paths: BenchPaths) -> dict:
    """Refuse to run on a torch other than the one setup pinned (it lives in a shared ~/.local)."""
    if not paths.torch_pin.exists():
        return {}
    info = torch_info()
    pinned = paths.torch_pin.read_text().strip()
    if info["torch"] != pinned or not info["cuda"]:
        raise ResourceError(f"torch changed: pinned {pinned} with CUDA, found {info['torch']} "
                            f"cuda={info['cuda']}; rerun `footlytics bench setup`")
    return info


def prune_runs(paths: BenchPaths, keep: int = 5) -> None:
    """Keep the bulky work/ of the last `keep` runs; rows and logs stay."""
    works = sorted(paths.runs.glob("*/work"), key=lambda p: p.parent.name)
    for w in works[:-keep] if keep else works:
        shutil.rmtree(w, ignore_errors=True)


def run_suite(suite: str, run_id: str, paths: BenchPaths, git_sha: str, host_label: str,
              suite_spec: dict | None = None) -> dict:
    run_dir = paths.run_dir(run_id)
    write_status(run_dir, "running")
    row = {"run_id": run_id, "suite": suite, "git_sha": git_sha, "host": host_label,
           "started_at": _now(), "contention": {"start": contention()}, "tasks": {}}
    try:
        row["env"] = preflight(paths)
        spec = suite_spec or load_suite(suite)
    except Exception as e:  # noqa: BLE001 - recorded, not raised: the poller must see it
        row.update(status="failed", error=f"{type(e).__name__}: {e}", finished_at=_now())
        return _finish(run_dir, row, paths)
    for task in spec["tasks"]:
        tid = task["id"]
        write_status(run_dir, "running", task=tid)
        _log(f"task {tid} ({task['kind']})")
        t0 = time.time()
        try:
            out = KINDS[task["kind"]](task, TaskContext(paths, run_dir / "work" / tid, _log))
            res = {"kind": task["kind"], **out,
                   "status": "partial" if out.get("failed") else "complete"}
        except Exception as e:  # noqa: BLE001
            res = {"kind": task["kind"], "status": "failed", "error": f"{type(e).__name__}: {e}",
                   "traceback": traceback.format_exc()[-2000:]}
            _log(f"task {tid} failed: {res['error']}")
        res["seconds"] = round(time.time() - t0, 1)
        res["check"] = evaluate_check(task.get("check"), res.get("aggregate") or {})
        row["tasks"][tid] = res
    row["status"] = run_status(row["tasks"])
    row["finished_at"] = _now()
    return _finish(run_dir, row, paths)


def _finish(run_dir: Path, row: dict, paths: BenchPaths) -> dict:
    row["contention"]["end"] = contention()
    (run_dir / "result.json").write_text(json.dumps(row, indent=2, default=str))
    write_status(run_dir, row["status"])
    prune_runs(paths)
    _log(f"run {row['run_id']}: {row['status']}")
    return row


# ---------------------------------------------------------------- task kinds

def _resolve(paths: BenchPaths, split: str, wanted) -> list[str]:
    have = datasets.clips(paths, split)
    if not have:
        raise FileNotFoundError(f"no GSR {split} labels under {paths.gsr_labels / split}; "
                                f"run `footlytics bench fetch gsr-{split}`")
    return have if wanted == "all" else [s for s in wanted]


def _score(ctx: TaskContext, split: str, tracker: str, seqs: list[str]) -> dict:
    """GS-HOTA jersey off (primary) and on (official) for one tracker folder."""
    per: dict[str, dict] = {s: {} for s in seqs}
    agg: dict[str, float] = {}
    for jersey in (False, True):
        tag = "on" if jersey else "off"
        s = run_gsr(ctx.paths.eval_python, ctx.paths.trackeval, ctx.paths.gsr_labels,
                    ctx.work / "trackers", split, tracker, seqs, ctx.work / f"eval-jersey-{tag}",
                    use_jersey=jersey)
        metrics = ("hota", "deta", "assa") if not jersey else ("hota",)
        for seq, m in s.clips.items():
            per.setdefault(seq, {}).update({f"{k}_jersey_{tag}": m[k] for k in metrics})
        agg.update({f"{k}_jersey_{tag}": s.combined[k] for k in metrics})
    return {"clips": per, "aggregate": agg}


@kind("gsr_gt_roundtrip")
def gsr_gt_roundtrip(task: dict, ctx: TaskContext) -> dict:
    """Ground truth -> Match State -> predictions. Scores ~100 or the adapters are wrong."""
    split = task["split"]
    out_dir = ctx.work / "trackers" / f"SoccerNetGS-{split}" / "footlytics"
    out_dir.mkdir(parents=True, exist_ok=True)
    failed, done = {}, []
    for seq in _resolve(ctx.paths, split, task.get("clips", "all")):
        try:
            labels = json.loads((ctx.paths.gsr_labels / split / seq / "Labels-GameState.json")
                                .read_text())
            pred = state_to_predictions(gt_to_state(labels), split, use_jersey=True)
            (out_dir / f"{seq}.json").write_text(json.dumps(pred))
            done.append(seq)
        except Exception as e:  # noqa: BLE001
            failed[seq] = f"{type(e).__name__}: {e}"
    if not done:
        raise RuntimeError(f"no clip converted: {failed}")
    ctx.log(f"converted {len(done)} clips, {len(failed)} failed")
    return {**_score(ctx, split, "footlytics", done), "failed": failed}


@kind("gsr_rescore")
def gsr_rescore(task: dict, ctx: TaskContext) -> dict:
    """Score someone else's published predictions (a zip of `<tracker>/<SEQ>.json`)."""
    split, tracker = task["split"], task.get("tracker", "tracklab")
    out_dir = ctx.work / "trackers" / f"SoccerNetGS-{split}" / tracker
    out_dir.mkdir(parents=True, exist_ok=True)
    failed, done = {}, []
    with zipfile.ZipFile(ctx.paths.data / task["predictions"]) as z:
        names = set(z.namelist())
        for seq in _resolve(ctx.paths, split, task.get("clips", "all")):
            member = f"{tracker}/{seq}.json"
            if member not in names:
                failed[seq] = f"no predictions in {task['predictions']}"
                continue
            (out_dir / f"{seq}.json").write_bytes(z.read(member))
            done.append(seq)
    if not done:
        raise RuntimeError(f"no predictions for any {split} clip in {task['predictions']}")
    return {**_score(ctx, split, tracker, done), "failed": failed}
```

In `pyproject.toml` dependencies add `"pyyaml>=6",`, and so the YAML and shell script ship
with a non-editable install, add:
```toml
[tool.hatch.build.targets.wheel.force-include]
"footlytics/bench/suites" = "footlytics/bench/suites"
"footlytics/bench/host_setup.sh" = "footlytics/bench/host_setup.sh"
```
(Hatch includes non-Python files inside the package by default; keep the force-include only if
`unzip -l` on a built wheel shows them missing.)

- [ ] **Step 4: Run** — `uv run --extra dev pytest tests/test_bench_runner.py -q` → PASS
- [ ] **Step 5: Commit** — `git add footlytics/bench/runner.py footlytics/bench/suites pyproject.toml uv.lock tests/test_bench_runner.py && git commit -m "feat(bench): suite runner with per-clip failure isolation; harness suite"`

---

### Task 7: Remote driver, host bootstrap, CLI

**Files:**
- Create: `footlytics/bench/remote.py`, `footlytics/bench/host_setup.sh`, `footlytics/bench/cli.py`
- Modify: `footlytics/cli.py` (register `bench`), `tests/test_cli.py` (help lists `bench`)
- Test: `tests/test_bench_remote.py`

**Interfaces:**
- Consumes: `runner.run_suite`, `runner.DONE`, `results.append_result`, `results.write_scoreboard`, `datasets.fetch`.
- Produces: `SshDown`, `RemoteError`, `Remote(host, runner=subprocess.run)` with `.root`
  (resolved absolute), `.sh(cmd, input=None, check=True) -> str`, `.ship(sha, archive) -> str`,
  `.setup(sha)`, `.ensure_env(sha, deps_sha) -> bool`, `.start(run_id, suite, sha)`,
  `.start_fetch(sha, what)`, `.status(run_id) -> dict`, `.result(run_id) -> dict`;
  `follow(remote, run_id, results_dir, interval=30, echo=print, sleep=time.sleep) -> dict`;
  `make_run_id(suite, sha, now=None) -> str`. CLI: `footlytics bench run|resume|setup|fetch|scoreboard`
  plus hidden `local`.

- [ ] **Step 1: Write the failing tests**

```python
"""Remote driver: ssh failures are explicit, shipping is idempotent, resume never double-appends."""
import json
import subprocess

import pytest

from footlytics.bench import remote as rm
from footlytics.bench.results import load_results


class FakeSsh:
    """Answers ssh commands from a script of (substring, returncode, stdout)."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def __call__(self, argv, input=None, capture_output=True):
        cmd = argv[-1]
        self.calls.append(cmd)
        for needle, code, out in self.script:
            if needle in cmd:
                return subprocess.CompletedProcess(argv, code, out.encode(), b"err")
        return subprocess.CompletedProcess(argv, 0, b"", b"")


ROOT = [("echo ~/footlytics-bench", 0, "/home/iec/footlytics-bench\n")]


def test_dropped_session_says_how_to_recover():
    r = rm.Remote("iec4090_2", runner=FakeSsh([("echo ~", 255, "")]))
    with pytest.raises(rm.SshDown, match="ssh iec4090_2 true"):
        r.root


def test_ship_skips_a_commit_already_there():
    ssh = FakeSsh(ROOT + [("test -f", 0, "yes\n")])
    rm.Remote("h", runner=ssh).ship("abc", b"tar")
    assert not any("tar -x" in c for c in ssh.calls)


def test_ship_uploads_and_prunes_old_sources():
    ssh = FakeSsh(ROOT + [("test -f", 0, "no\n")])
    d = rm.Remote("h", runner=ssh).ship("abc", b"tar")
    assert d == "/home/iec/footlytics-bench/src/abc"
    assert any("tar -x -C /home/iec/footlytics-bench/src/abc.tmp" in c for c in ssh.calls)
    assert any("tail -n +6" in c for c in ssh.calls)


def test_start_runs_in_a_named_tmux_session():
    ssh = FakeSsh(ROOT)
    rm.Remote("iec4090_2", runner=ssh).start("20261012-100000-harness-abc1234", "harness", "abc")
    cmd = next(c for c in ssh.calls if "tmux new-session" in c)
    assert "-s footlytics-20261012-100000-harness-abc1234" in cmd
    assert "bench local --suite harness" in cmd and "FOOTLYTICS_BENCH_ROOT=" in cmd


def test_status_reports_a_dead_session_with_its_log():
    ssh = FakeSsh(ROOT + [("cat /home/iec/footlytics-bench/runs/r1/status.json", 0,
                           '{"state": "running"}'),
                          ("tmux has-session", 0, "no\n"), ("tail -n 30", 0, "Killed\n")])
    st = rm.Remote("h", runner=ssh).status("r1")
    assert st["state"] == "crashed" and "Killed" in st["log_tail"]


def test_follow_appends_once_even_when_resumed(tmp_path):
    row = {"run_id": "r1", "suite": "harness", "git_sha": "abc", "host": "h",
           "finished_at": "2026-10-12T10:00:00Z", "status": "complete", "tasks": {}}
    ssh = FakeSsh(ROOT + [("status.json", 0, '{"state": "complete"}'),
                          ("result.json", 0, json.dumps(row))])
    r = rm.Remote("h", runner=ssh)
    for _ in range(2):
        rm.follow(r, "r1", tmp_path, interval=0, echo=lambda *a: None, sleep=lambda s: None)
    assert [x["run_id"] for x in load_results(tmp_path / "results.jsonl")] == ["r1"]
    assert (tmp_path / "SCOREBOARD.md").exists()


def test_run_id_is_sortable_and_names_suite_and_commit():
    from datetime import datetime, timezone
    rid = rm.make_run_id("harness", "abcdef123456", datetime(2026, 10, 12, 10, 0, tzinfo=timezone.utc))
    assert rid == "20261012-100000-harness-abcdef1"
```

In `tests/test_cli.py`, add `"bench"` to the tuple in `test_help_lists_commands`.

- [ ] **Step 2: Run to verify failure** → FAIL

- [ ] **Step 3: Implement**

`footlytics/bench/remote.py`:
```python
"""M2 side: ship a commit to the host, start the suite in tmux, poll, pull the row back.

The run lives in a tmux session on the host, so a dropped ssh session (password
ControlMaster until key login is set up) loses only the polling; `bench resume` picks it up.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .paths import DEFAULT_ROOT
from .results import append_result, write_scoreboard
from .runner import DONE


class SshDown(RuntimeError):
    """ssh itself failed (exit 255): the session dropped or auth is gone."""


class RemoteError(RuntimeError):
    """A command on the host failed."""


def make_run_id(suite: str, sha: str, now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"{now:%Y%m%d-%H%M%S}-{suite}-{sha[:7]}"


@dataclass
class Remote:
    host: str
    runner: Callable = subprocess.run
    _root: Optional[str] = field(default=None, repr=False)

    def sh(self, cmd: str, input: Optional[bytes] = None, check: bool = True) -> str:
        p = self.runner(["ssh", "-o", "BatchMode=yes", self.host, cmd], input=input,
                        capture_output=True)
        out, err = p.stdout.decode(), p.stderr.decode()
        if p.returncode == 255:
            raise SshDown(f"ssh to {self.host} failed ({err.strip()[-200:]}). Reconnect with "
                          f"`ssh {self.host} true`, then `footlytics bench resume <run-id> "
                          f"--remote {self.host}`.")
        if check and p.returncode != 0:
            raise RemoteError(f"`{cmd}` exited {p.returncode}: {err.strip()[-800:]}")
        return out

    @property
    def root(self) -> str:
        if self._root is None:
            self._root = self.sh(f"echo {DEFAULT_ROOT}").strip()
        return self._root

    def ship(self, sha: str, archive: bytes) -> str:
        d = f"{self.root}/src/{sha}"
        if self.sh(f"test -f {d}/pyproject.toml && echo yes || echo no").strip() == "yes":
            return d
        self.sh(f"rm -rf {d}.tmp && mkdir -p {d}.tmp && tar -x -C {d}.tmp && mv {d}.tmp {d}",
                input=archive)
        self.sh(f"cd {self.root}/src && ls -1t | tail -n +6 | xargs -r rm -rf")
        return d

    def setup(self, sha: str) -> str:
        d = f"{self.root}/src/{sha}"
        return self.sh(f"bash {d}/footlytics/bench/host_setup.sh {self.root} {d}")

    def ensure_env(self, sha: str, deps_sha: str) -> bool:
        cur = self.sh(f"cat {self.root}/venv/.deps-sha 2>/dev/null || true").strip()
        if cur == deps_sha:
            return False
        self.setup(sha)
        return True

    def _python(self, sha: str, args: str, log: str, session: str) -> None:
        cmd = (f"cd {self.root}/src/{sha} && FOOTLYTICS_BENCH_ROOT={self.root} "
               f"HF_HOME={self.root}/cache/hf {self.root}/venv/bin/python -m footlytics.cli "
               f"{args} > {log} 2>&1")
        self.sh(f"mkdir -p {Path(log).parent} && tmux new-session -d -s {session} "
                f"{shlex.quote(cmd)}")

    def start(self, run_id: str, suite: str, sha: str) -> None:
        self._python(sha, f"bench local --suite {suite} --run-id {run_id} --git-sha {sha} "
                          f"--host {self.host}",
                     f"{self.root}/runs/{run_id}/log.txt", f"footlytics-{run_id}")

    def start_fetch(self, sha: str, what: list[str]) -> str:
        stamp = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
        log = f"{self.root}/runs/fetch-{stamp}.log"
        self._python(sha, "bench fetch " + " ".join(what), log, f"footlytics-fetch-{stamp}")
        return log

    def status(self, run_id: str) -> dict:
        d = f"{self.root}/runs/{run_id}"
        raw = self.sh(f"cat {d}/status.json 2>/dev/null || true").strip()
        st = json.loads(raw) if raw else {"state": "starting"}
        if st["state"] in DONE:
            return st
        alive = self.sh(f"tmux has-session -t footlytics-{run_id} 2>/dev/null "
                        "&& echo yes || echo no").strip() == "yes"
        if not alive:
            return {"state": "crashed",
                    "log_tail": self.sh(f"tail -n 30 {d}/log.txt 2>/dev/null || true")}
        return st

    def result(self, run_id: str) -> dict:
        return json.loads(self.sh(f"cat {self.root}/runs/{run_id}/result.json"))


def follow(remote: Remote, run_id: str, results_dir: Path, interval: float = 30,
           echo: Callable = print, sleep: Callable = time.sleep) -> dict:
    """Poll until the run ends, then append its row and regenerate the scoreboard."""
    last = None
    while True:
        st = remote.status(run_id)
        if st["state"] == "crashed":
            raise RemoteError(f"run {run_id} died without a result:\n{st['log_tail']}")
        shown = (st["state"], st.get("task"))
        if shown != last:
            echo(f"[{run_id}] {st['state']}" + (f": {st['task']}" if st.get("task") else ""))
            last = shown
        if st["state"] in DONE:
            row = remote.result(run_id)
            results_dir = Path(results_dir)
            if append_result(results_dir / "results.jsonl", row):
                echo(f"appended to {results_dir / 'results.jsonl'}")
            write_scoreboard(results_dir)
            return row
        sleep(interval)
```

`footlytics/bench/host_setup.sh`:
```bash
#!/usr/bin/env bash
# Idempotent bootstrap of the benchmark host. Everything goes under $ROOT (shared lab account).
#   usage: host_setup.sh <root> <src dir of the commit being run>
set -euo pipefail
ROOT="$1"; SRC="$2"
TORCH_PIN="2.7.1+cu128"
TRACKEVAL_SHA=9c25232f6f2b56c9f203f1eb55784ff1e97df683
CALIB_SHA=ab38f461bec729fead86b6986839de1bb826f16d
export UV_CACHE_DIR="$ROOT/cache/uv" PIP_CACHE_DIR="$ROOT/cache/pip" HF_HOME="$ROOT/cache/hf"
mkdir -p "$ROOT"/{bin,cache,data,runs,src,ext}

[ -x "$ROOT/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh \
  | env UV_INSTALL_DIR="$ROOT/bin" UV_NO_MODIFY_PATH=1 sh

# Pipeline venv on the system torch. pip, not uv: uv ignores system site-packages and would
# pull a second torch. Code runs from $ROOT/src/<sha> (cwd first on sys.path); the install
# only brings the dependencies.
[ -x "$ROOT/venv/bin/python" ] || "$ROOT/bin/uv" venv -q --seed --system-site-packages \
  --python /usr/bin/python3 "$ROOT/venv"
"$ROOT/venv/bin/pip" install -q "$SRC[dev]"
"$ROOT/venv/bin/python" - "$TORCH_PIN" <<'EOF'
import sys, torch
assert torch.__version__ == sys.argv[1], f"torch {torch.__version__}, expected {sys.argv[1]}"
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, torch.cuda.get_device_name(0))
EOF
echo "$TORCH_PIN" > "$ROOT/venv/.torch-version"

# Evaluator venv: third-party evaluators run here as subprocesses, never imported.
[ -x "$ROOT/venv-eval/bin/python" ] || "$ROOT/bin/uv" venv -q --python /usr/bin/python3 \
  "$ROOT/venv-eval"
VIRTUAL_ENV="$ROOT/venv-eval" "$ROOT/bin/uv" pip install -q numpy scipy tqdm \
  opencv-python-headless matplotlib
clone() {
  [ -d "$ROOT/ext/$1/.git" ] || git clone -q "https://github.com/SoccerNet/$1" "$ROOT/ext/$1"
  git -C "$ROOT/ext/$1" fetch -q origin && git -C "$ROOT/ext/$1" checkout -q "$2"
}
clone sn-trackeval "$TRACKEVAL_SHA"
clone sn-calibration "$CALIB_SHA"

sha256sum "$SRC/pyproject.toml" | cut -d' ' -f1 > "$ROOT/venv/.deps-sha"
echo "host ready: $ROOT"
```

`footlytics/bench/cli.py`:
```python
"""`footlytics bench`: run suites on the benchmark host and keep the scoreboard."""
from __future__ import annotations

import hashlib
import socket
import subprocess
from pathlib import Path
from typing import Optional

import typer

bench_app = typer.Typer(help="Benchmark: our output scored by the official SoccerNet evaluators.")
REPO = Path(__file__).resolve().parents[2]


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=True).stdout


def _clean_sha() -> str:
    dirty = _git("status", "--porcelain", "--untracked-files=no", "--", ".",
                 ":!bench/results").decode().strip()
    if dirty:
        raise typer.BadParameter("uncommitted changes: a row must name the code it scored. "
                                 f"Commit first.\n{dirty}")
    return _git("rev-parse", "HEAD").decode().strip()


def _remote(host: str):
    from .remote import Remote

    return Remote(host)


def _guard(fn):
    """Turn a dropped ssh session into its recovery instructions and exit code 2."""
    from .remote import RemoteError, SshDown

    try:
        return fn()
    except SshDown as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2)
    except RemoteError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)


@bench_app.command("run")
def run(suite: str = typer.Option("harness"),
        remote: Optional[str] = typer.Option(None, help="ssh alias of the host, e.g. iec4090_2"),
        interval: int = typer.Option(30, help="seconds between polls")) -> None:
    """Run a suite (on the host with --remote) and append its row to bench/results."""
    from .remote import follow, make_run_id
    from .results import RESULTS_DIR, append_result, write_scoreboard

    sha = _clean_sha()
    run_id = make_run_id(suite, sha)
    if remote is None:
        from .paths import BenchPaths
        from .runner import run_suite

        row = run_suite(suite, run_id, BenchPaths.from_env(), sha, socket.gethostname())
        append_result(RESULTS_DIR / "results.jsonl", row)
        write_scoreboard(RESULTS_DIR)
        typer.echo(f"{run_id}: {row['status']}")
        return

    def go():
        r = _remote(remote)
        r.ship(sha, _git("archive", "--format=tar", sha))
        deps = hashlib.sha256(_git("show", f"{sha}:pyproject.toml")).hexdigest()
        if r.ensure_env(sha, deps):
            typer.echo("host environment (re)built")
        r.start(run_id, suite, sha)
        typer.echo(f"started {run_id} on {remote}; if ssh drops: "
                   f"`footlytics bench resume {run_id} --remote {remote}`")
        row = follow(r, run_id, RESULTS_DIR, interval, echo=typer.echo)
        typer.echo(f"{run_id}: {row['status']}")
    _guard(go)


@bench_app.command("resume")
def resume(run_id: str, remote: str = typer.Option(...),
           interval: int = typer.Option(30)) -> None:
    """Re-attach to a run after an ssh drop: poll, then pull its row."""
    from .remote import follow
    from .results import RESULTS_DIR

    _guard(lambda: typer.echo(follow(_remote(remote), run_id, RESULTS_DIR, interval,
                                     echo=typer.echo)["status"]))


@bench_app.command("setup")
def setup(remote: str = typer.Option(...)) -> None:
    """(Re)build the host: uv, pipeline venv on the system torch, evaluator venv, evaluators."""
    sha = _clean_sha()

    def go():
        r = _remote(remote)
        r.ship(sha, _git("archive", "--format=tar", sha))
        typer.echo(r.setup(sha))
    _guard(go)


@bench_app.command("fetch")
def fetch(what: list[str] = typer.Argument(..., help="gsr-valid gsr-test calib-valid calib-test baseline"),
          remote: Optional[str] = typer.Option(None, help="start the download on this host in tmux")) -> None:
    """Download pinned datasets (on the host; with --remote, start that in tmux and return)."""
    if remote:
        sha = _clean_sha()

        def go():
            r = _remote(remote)
            r.ship(sha, _git("archive", "--format=tar", sha))
            typer.echo(f"fetching in tmux on {remote}; log: {r.start_fetch(sha, what)}")
        _guard(go)
        return
    from .datasets import fetch as fetch_one
    from .paths import BenchPaths

    paths = BenchPaths.from_env()
    for w in what:
        typer.echo(f"{w}: {fetch_one(paths, w)}")


@bench_app.command("local", hidden=True)
def local(suite: str = typer.Option(...), run_id: str = typer.Option(...),
          git_sha: str = typer.Option(...), host: str = typer.Option(...)) -> None:
    """Host side of `bench run --remote`: run the suite here and leave result.json."""
    from .paths import BenchPaths
    from .runner import run_suite

    row = run_suite(suite, run_id, BenchPaths.from_env(), git_sha, host)
    typer.echo(f"{run_id}: {row['status']}")


@bench_app.command("scoreboard")
def scoreboard() -> None:
    """Regenerate bench/results/SCOREBOARD.md from results.jsonl."""
    from .results import RESULTS_DIR, write_scoreboard

    typer.echo(str(write_scoreboard(RESULTS_DIR)))
```

In `footlytics/cli.py`, after `app = typer.Typer(...)`:
```python
from .bench.cli import bench_app  # noqa: E402

app.add_typer(bench_app, name="bench")
```
and update the module docstring to list `footlytics bench`.

- [ ] **Step 4: Run** — `uv run --extra dev pytest tests/test_bench_remote.py tests/test_cli.py -q` → PASS; then the full suite `uv run --extra dev pytest -q` → no new failures.
- [ ] **Step 5: Commit** — `git add footlytics/bench/{remote.py,host_setup.sh,cli.py} footlytics/cli.py tests/test_bench_remote.py tests/test_cli.py && git commit -m "feat(bench): remote runs in tmux on the host, resumable after ssh drops"`

---

### Task 8: Host bring-up and the harness proof (operational)

No new code unless a step fails, and if one does, fix it with a test in the owning task's file.

- [ ] **Step 1: Setup.** `uv run footlytics bench setup --remote iec4090_2`. Expect it to
  print the `torch 2.7.1+cu128 NVIDIA GeForce RTX 4090` line and `host ready`. Verify with
  `ssh iec4090_2 'ls ~/footlytics-bench; cat ~/footlytics-bench/venv/.torch-version; df -h ~ | tail -1'`.
- [ ] **Step 2: Fetch.** `uv run footlytics bench fetch baseline gsr-valid gsr-test calib-valid --remote iec4090_2`
  (about 21 GB; check that free space stays above the floor). Follow it with
  `ssh iec4090_2 'tail -3 <log>'` until it lists all four. Expect 58 valid clips under
  `data/gsr/labels/valid` and 49 under `labels/test`; if the counts differ, record the real
  ones in the spec.
- [ ] **Step 3: Harness run.** `uv run footlytics bench run --suite harness --remote iec4090_2`.
  Expected: roundtrip `hota_jersey_off` >= 99 ✓, baseline `hota_jersey_on` within 1.0 of
  22.26 ✓. If the roundtrip fails on some clips, look at their per-clip score: a clip near
  50 means the left/right rule swapped the teams. Fix `team_sides` with a regression test
  built from that clip's geometry, never by reading the GT team. If the baseline misses,
  fall back to the Zenodo tracker state and note it in the spec.
- [ ] **Step 4: Record.** Commit `bench/results/results.jsonl` and `SCOREBOARD.md` together
  with any spec corrections (clip counts, licence line, baseline source). Then
  `git commit -m "bench: harness proven on iec4090_2 (GT roundtrip, baseline re-score)"`.
