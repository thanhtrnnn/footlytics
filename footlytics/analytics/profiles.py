"""Per-club tactical threshold profiles (`--profile club.yaml` and live Workspace tuning).

Allows analysts and Head Coaches to adapt the 5 V1 tactical sequence definitions
(`low_block`, `high_press`, `build_up`, `midfield_loss`, `flank_overload`) to a
club's exact game model without re-running perception or calibration.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..state.schema import MatchState
from .possession import infer_possession
from .sequences import detect_sequences, write_sequence_outputs
from .tactics import normalise_direction

_DETECT_FIELDS: tuple[str, ...] = (
    "sample_every",
    "low_block_line",
    "low_block_compact_m",
    "low_block_min_s",
    "high_press_line",
    "high_press_n",
    "high_press_min_s",
    "flank_y",
    "overload_n",
    "overload_max_defenders",
    "overload_min_visible",
    "overload_min_s",
    "build_up_min_s",
    "build_up_min_centroid_x",
    "still_s",
    "still_m",
    "stopped_mps",
    "loss_window_s",
    "counter_press_s",
    "counter_press_m",
    "retreat_mps",
)


@dataclass(frozen=True)
class TacticalProfile:
    """Club-customizable thresholds passed directly to `detect_sequences`."""

    name: str = "default"
    description: str = "V1 calibrated V.League defaults"

    sample_every: int = 5
    low_block_line: float = -22.5
    low_block_compact_m: float = 14.0
    low_block_min_s: float = 8.0

    high_press_line: float = 7.5
    high_press_n: int = 3
    high_press_min_s: float = 3.0

    flank_y: float | None = None
    overload_n: int = 4
    overload_max_defenders: int = 3
    overload_min_visible: int = 7
    overload_min_s: float = 3.0

    build_up_min_s: float = 4.0
    build_up_min_centroid_x: float = -25.0
    still_s: float = 2.0
    still_m: float = 1.0
    stopped_mps: float = 1.0

    loss_window_s: float = 6.0
    counter_press_s: float = 3.0
    counter_press_m: float = 6.0
    retreat_mps: float = -1.0

    def __post_init__(self) -> None:
        if self.sample_every < 1:
            raise ValueError("sample_every must be >= 1")
        if not (-52.5 <= self.low_block_line <= 0.0):
            raise ValueError(f"low_block_line must be in [-52.5, 0.0], got {self.low_block_line}")
        if self.low_block_compact_m <= 0.0:
            raise ValueError("low_block_compact_m must be > 0")
        if self.low_block_min_s <= 0.0:
            raise ValueError("low_block_min_s must be > 0")
        if not (-20.0 <= self.high_press_line <= 52.5):
            raise ValueError(f"high_press_line must be in [-20.0, 52.5], got {self.high_press_line}")
        if self.high_press_n < 1:
            raise ValueError("high_press_n must be >= 1")
        if self.high_press_min_s <= 0.0:
            raise ValueError("high_press_min_s must be > 0")
        if self.overload_n < 2:
            raise ValueError("overload_n must be >= 2")
        if self.overload_max_defenders < 0:
            raise ValueError("overload_max_defenders must be >= 0")
        if self.overload_min_visible < 2:
            raise ValueError("overload_min_visible must be >= 2")
        if self.overload_min_s <= 0.0:
            raise ValueError("overload_min_s must be > 0")
        if self.build_up_min_s <= 0.0:
            raise ValueError("build_up_min_s must be > 0")
        if self.counter_press_m <= 0.0:
            raise ValueError("counter_press_m must be > 0")
        if self.retreat_mps > 0.0:
            raise ValueError("retreat_mps must be <= 0.0 (negative = towards own goal)")

    def to_detect_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted by `detect_sequences`."""
        return {k: getattr(self, k) for k in _DETECT_FIELDS}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], base: TacticalProfile | None = None) -> TacticalProfile:
        """Build a validated `TacticalProfile` from a partial or full dict.

        With `base`, fields missing from `raw` keep `base`'s values instead of resetting to defaults.
        """
        if not isinstance(raw, dict):
            raise TypeError("profile payload must be a mapping")
        # Allow nested `thresholds:` block in hand-written YAML files
        merged: dict[str, Any] = {}
        if isinstance(raw.get("thresholds"), dict):
            merged.update(raw["thresholds"])
        for k, v in raw.items():
            if k != "thresholds":
                merged[k] = v

        allowed = {f.name: f for f in fields(cls)}
        unknown = sorted(set(merged) - set(allowed))
        if unknown:
            raise ValueError(f"unknown tactical profile field(s): {', '.join(unknown)}")

        coerced: dict[str, Any] = {} if base is None else base.to_dict()
        int_fields = {"sample_every", "high_press_n", "overload_n", "overload_max_defenders", "overload_min_visible"}
        str_fields = {"name", "description"}
        for k, v in merged.items():
            if v is None or v == "":
                if k == "flank_y":
                    coerced[k] = None
                continue
            if k in str_fields:
                coerced[k] = str(v)
            elif k in int_fields:
                coerced[k] = int(v)
            else:
                coerced[k] = float(v)
        return cls(**coerced)


BUILTIN_PROFILES: dict[str, TacticalProfile] = {
    "default": TacticalProfile(
        name="default",
        description="V1 calibrated V.League defaults (pattern detections pending blind analyst validation)",
    ),
    "high_press_aggressive": TacticalProfile(
        name="high_press_aggressive",
        description="High-line pressing clubs (stricter +12m line, 4 pressers, tighter 5m counter-press)",
        high_press_line=12.0,
        high_press_n=4,
        high_press_min_s=2.5,
        counter_press_m=5.0,
        low_block_line=-25.0,
    ),
    "compact_counter": TacticalProfile(
        name="compact_counter",
        description="Mid-low block & counter-attacking profile (-20m line, 12.5m compactness, 6s block)",
        low_block_line=-20.0,
        low_block_compact_m=12.5,
        low_block_min_s=6.0,
        retreat_mps=-0.8,
        build_up_min_s=3.5,
    ),
    "wing_overload": TacticalProfile(
        name="wing_overload",
        description="Wide-channel focus (3+ attackers in flank corridor against <= 2 defenders)",
        overload_n=3,
        overload_max_defenders=2,
        overload_min_s=2.5,
    ),
}


def load_profile(ref: str | Path) -> TacticalProfile:
    """Load a `TacticalProfile` from a built-in preset name or a `.yaml`/`.yml`/`.json` file."""
    key = str(ref).strip()
    if key in BUILTIN_PROFILES:
        return BUILTIN_PROFILES[key]
    p = Path(key)
    if not p.is_file():
        known = ", ".join(sorted(BUILTIN_PROFILES))
        raise ValueError(f"profile {ref!r} not found (use a file path or built-in preset: {known})")
    text = p.read_text(encoding="utf-8")
    data = json.loads(text) if p.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TypeError(f"profile file {p} must contain a YAML/JSON object")
    if "name" not in data:
        data = {"name": p.stem, **data}
    return TacticalProfile.from_dict(data)


def save_profile(profile: TacticalProfile, path: Path | str) -> Path:
    """Write a `TacticalProfile` to `.yaml`/`.yml` or `.json`."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".json":
        out.write_text(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        out.write_text(profile.to_yaml(), encoding="utf-8")
    return out


def backup_file(path: Path) -> Path | None:
    """Copy `path` to `<name>.bak-<UTC timestamp>` beside it, so a rewrite never loses the original."""
    if not path.is_file():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dst = path.with_name(f"{path.name}.bak-{stamp}")
    shutil.copy2(path, dst)
    return dst


def recompute_match_sequences(
    match_dir: Path | str,
    profile: TacticalProfile,
    *,
    write: bool = False,
    state: MatchState | None = None,
) -> pd.DataFrame:
    """Re-run `detect_sequences` on a processed match directory with custom thresholds."""
    mdir = Path(match_dir)
    st = state if state is not None else MatchState.load(mdir)
    norm = normalise_direction(st)
    poss_path = mdir / "possession.csv"
    if poss_path.is_file():
        poss = pd.read_csv(poss_path)
    else:
        poss = infer_possession(norm)
    if write:
        backup_file(mdir / "sequences.csv")
        backup_file(mdir / "turnovers.csv")
        backup_file(mdir / "phase_shares.json")
        backup_file(mdir / "tactical_profile.yaml")
        seqs = write_sequence_outputs(norm, poss, mdir, **profile.to_detect_kwargs())
        save_profile(profile, mdir / "tactical_profile.yaml")
        return seqs
    seqs = detect_sequences(norm, poss, **profile.to_detect_kwargs())
    return seqs
