"""L5 multi-match tactical query index backed by DuckDB.

Discovers match output directories containing `sequences.csv` (and optionally
`meta.json` / `tracks.parquet`), loads them into an in-memory DuckDB database,
and exposes summary aggregations and filtered searches across matches.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import duckdb
import pandas as pd

from .sequences import COLUMNS as SEQUENCE_COLUMNS

_STRING_COLS = ("label", "team", "flank", "response")
_INT_COLS = (
    "period",
    "start_frame",
    "end_frame",
    "n_within_10m",
    "n_attackers",
    "n_defenders",
    "n_pressers",
)
_FLOAT_COLS = (
    "start_s",
    "end_s",
    "duration_s",
    "def_line_x",
    "compactness_m",
    "in_possession_share",
    "ball_dist_m",
    "retreat_mps",
)
SEARCH_COLUMNS = ("match_id", "home_team", "away_team", *SEQUENCE_COLUMNS)
SORTABLE_COLUMNS = frozenset(SEARCH_COLUMNS)


@dataclass(frozen=True)
class MatchEntry:
    """One discovered match output directory."""

    match_id: str
    dir: Path
    sequences_path: Path
    meta_path: Optional[Path]
    tracks_path: Optional[Path]
    home: str
    away: str
    fps: Optional[float]


def _normalize_roots(roots: Sequence[Path | str] | Path | str) -> list[Path]:
    items = [roots] if isinstance(roots, (str, Path)) else list(roots)
    if not items:
        raise ValueError("at least one match or output root path is required")
    return [Path(p) for p in items]


def _read_match_entry(match_dir: Path) -> MatchEntry:
    meta_path = match_dir / "meta.json"
    tracks_path = match_dir / "tracks.parquet"
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
    home_info = meta.get("home") or {}
    away_info = meta.get("away") or {}
    home = home_info.get("name", "") if isinstance(home_info, dict) else str(home_info)
    away = away_info.get("name", "") if isinstance(away_info, dict) else str(away_info)
    fps = float(meta["fps"]) if meta.get("fps") is not None else None
    match_id = str(meta.get("match_id") or match_dir.name)
    return MatchEntry(
        match_id=match_id,
        dir=match_dir,
        sequences_path=match_dir / "sequences.csv",
        meta_path=meta_path if meta_path.is_file() else None,
        tracks_path=tracks_path if tracks_path.is_file() else None,
        home=home,
        away=away,
        fps=fps,
    )


def discover_matches(roots: Sequence[Path | str] | Path | str) -> list[MatchEntry]:
    """Find match directories containing `sequences.csv` under `roots`."""
    dirs: dict[Path, MatchEntry] = {}
    for root in _normalize_roots(roots):
        if not root.exists():
            raise FileNotFoundError(f"path not found: {root}")
        candidates = (
            [root]
            if (root / "sequences.csv").is_file()
            else sorted(p.parent for p in root.rglob("sequences.csv") if p.is_file())
        )
        for d in candidates:
            resolved = d.resolve()
            if resolved not in dirs:
                dirs[resolved] = _read_match_entry(d)
    entries = sorted(dirs.values(), key=lambda e: (e.match_id, str(e.dir)))
    return entries


def _read_sequences_df(entry: MatchEntry) -> pd.DataFrame:
    df = pd.read_csv(entry.sequences_path)
    for col in SEQUENCE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    for col in _STRING_COLS:
        df[col] = df[col].fillna("").astype(str)
    for col in _INT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in _FLOAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df.insert(0, "match_id", entry.match_id)
    df.insert(1, "home_team", entry.home)
    df.insert(2, "away_team", entry.away)
    return df[list(SEARCH_COLUMNS)]


def _empty_sequences_df() -> pd.DataFrame:
    data: dict[str, pd.Series] = {
        "match_id": pd.Series(dtype="object"),
        "home_team": pd.Series(dtype="object"),
        "away_team": pd.Series(dtype="object"),
    }
    for col in SEQUENCE_COLUMNS:
        if col in _STRING_COLS:
            data[col] = pd.Series(dtype="object")
        elif col in _INT_COLS:
            data[col] = pd.Series(dtype="Int64")
        else:
            data[col] = pd.Series(dtype="float64")
    return pd.DataFrame(data, columns=list(SEARCH_COLUMNS))


def build_query_db(roots: Sequence[Path | str] | Path | str) -> duckdb.DuckDBPyConnection:
    """Load discovered matches and sequences into an in-memory DuckDB connection."""
    entries = discover_matches(roots)
    match_rows = [
        {
            "match_id": e.match_id,
            "dir": str(e.dir),
            "home_team": e.home,
            "away_team": e.away,
            "fps": e.fps,
            "has_meta": e.meta_path is not None,
            "has_tracks": e.tracks_path is not None,
        }
        for e in entries
    ]
    matches_df = (
        pd.DataFrame(match_rows)
        if match_rows
        else pd.DataFrame(
            columns=["match_id", "dir", "home_team", "away_team", "fps", "has_meta", "has_tracks"]
        )
    )
    seq_frames = [_read_sequences_df(e) for e in entries]
    sequences_df = (
        pd.concat(seq_frames, ignore_index=True) if seq_frames else _empty_sequences_df()
    )

    con = duckdb.connect(":memory:")
    con.register("_matches_df", matches_df)
    con.register("_sequences_df", sequences_df)
    con.execute("CREATE TABLE matches AS SELECT * FROM _matches_df")
    con.execute("CREATE TABLE sequences AS SELECT * FROM _sequences_df")
    con.unregister("_matches_df")
    con.unregister("_sequences_df")
    return con


def _build_where(
    *,
    match: Optional[str] = None,
    team: Optional[str] = None,
    label: Optional[str] = None,
    period: Optional[int] = None,
    min_duration: Optional[float] = None,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    response: Optional[str] = None,
    flank: Optional[str] = None,
    min_pressers: Optional[int] = None,
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if match is not None:
        clauses.append("match_id = ?")
        params.append(match)
    if team is not None:
        clauses.append("team = ?")
        params.append(team)
    if label is not None:
        clauses.append("label = ?")
        params.append(label)
    if period is not None:
        clauses.append("period = ?")
        params.append(int(period))
    if min_duration is not None:
        clauses.append("duration_s >= ?")
        params.append(float(min_duration))
    if start_s is not None:
        clauses.append("end_s >= ?")
        params.append(float(start_s))
    if end_s is not None:
        clauses.append("start_s <= ?")
        params.append(float(end_s))
    if response is not None:
        clauses.append("response = ?")
        params.append(response)
    if flank is not None:
        clauses.append("flank = ?")
        params.append(flank)
    if min_pressers is not None:
        clauses.append("n_pressers >= ?")
        params.append(int(min_pressers))
    where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return where_sql, params


def summarize_sequences(
    roots: Sequence[Path | str] | Path | str,
    *,
    match: Optional[str] = None,
    team: Optional[str] = None,
    label: Optional[str] = None,
    period: Optional[int] = None,
    min_duration: Optional[float] = None,
    by_match: bool = True,
) -> pd.DataFrame:
    """Aggregate count, total duration, and mean duration per tactical label and team."""
    con = build_query_db(roots)
    try:
        where_sql, params = _build_where(
            match=match, team=team, label=label, period=period, min_duration=min_duration
        )
        group_cols = ["match_id", "team", "label"] if by_match else ["team", "label"]
        group_expr = ", ".join(group_cols)
        sql = (
            f"SELECT {group_expr}, "
            "CAST(COUNT(*) AS BIGINT) AS count, "
            "ROUND(SUM(duration_s), 2) AS total_duration_s, "
            "ROUND(AVG(duration_s), 2) AS mean_duration_s "
            f"FROM sequences{where_sql} "
            f"GROUP BY {group_expr} "
            f"ORDER BY count DESC, total_duration_s DESC, {group_expr}"
        )
        return con.execute(sql, params).df()
    finally:
        con.close()


def search_sequences(
    roots: Sequence[Path | str] | Path | str,
    *,
    match: Optional[str] = None,
    team: Optional[str] = None,
    label: Optional[str] = None,
    period: Optional[int] = None,
    min_duration: Optional[float] = None,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    response: Optional[str] = None,
    flank: Optional[str] = None,
    min_pressers: Optional[int] = None,
    sort_by: str = "start_s",
    descending: bool = False,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Filter and sort tactical sequences across matches (`start_s`/`end_s` overlap)."""
    if sort_by not in SORTABLE_COLUMNS:
        raise ValueError(f"unknown sort_by {sort_by!r}; expected one of {sorted(SORTABLE_COLUMNS)}")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")

    con = build_query_db(roots)
    try:
        where_sql, params = _build_where(
            match=match,
            team=team,
            label=label,
            period=period,
            min_duration=min_duration,
            start_s=start_s,
            end_s=end_s,
            response=response,
            flank=flank,
            min_pressers=min_pressers,
        )
        direction = "DESC" if descending else "ASC"
        order_sql = f" ORDER BY {sort_by} {direction}, match_id ASC, start_s ASC, label ASC"
        limit_sql = ""
        if limit is not None:
            limit_sql = " LIMIT ?"
            params.append(int(limit))
        cols_sql = ", ".join(SEARCH_COLUMNS)
        sql = f"SELECT {cols_sql} FROM sequences{where_sql}{order_sql}{limit_sql}"
        return con.execute(sql, params).df()
    finally:
        con.close()
