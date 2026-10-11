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
    def weights(self) -> Path:
        return self.root / "weights" / "soccermaster"

    @property
    def soccermaster_code(self) -> Path:
        """The SoccerMaster checkout (no licence: internal R&D), shipped by `bench setup`."""
        return self.ext / "SoccerMaster"

    @property
    def sn_calibration(self) -> Path:
        return self.ext / "sn-calibration"

    def calib_zip(self, split: str) -> Path:
        return self.data / "calibration" / f"{split}.zip"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def torch_pin(self) -> Path:
        """Torch version recorded by host_setup.sh; every run checks it still holds."""
        return self.venv / ".torch-version"
