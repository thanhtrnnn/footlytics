"""Identity resolution at the tracklet level, with roster priors.

The tracker in `track.py` decides identity frame by frame, which is the wrong
altitude for the question. Identity is a property of a *person over the whole
match*, so it should be decided once per tracklet, using everything the tracklet
knows about itself, and it should be constrained by what we know about football
before we look at a single pixel:

* there are eleven players a side on the pitch;
* the team sheet is known in advance, with shirt numbers;
* one shirt number cannot be in two places at once;
* a player who disappears reappears near where they vanished, moving at a
  plausible human speed.

This is how commercial systems get usable identity. It is not a better neural
network -- Bepro's own pipeline is "human-assisted AI", polished by trained
analysts within 24 hours, and TRACAB's published validity study states plainly
that an operator corrects positions when tracking is lost. Their FIFA EPTS
certification covers positional and velocity accuracy; it does not test player
identification at all. So the goal here is not perfection. It is to shrink the
work left for a human to a few clicks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ..state.schema import MatchState, Role, Team


@dataclass
class Tracklet:
    """A contiguous run of observations believed to be one person."""

    id: int
    track_ids: list[int]
    start: int                       # first frame index
    end: int                         # last frame index
    start_xy: np.ndarray
    end_xy: np.ndarray
    exit_velocity: np.ndarray        # m/s, from the tail of the tracklet
    n_frames: int
    embedding: Optional[np.ndarray] = None
    role: str = Role.PLAYER.value
    team: str = Team.UNKNOWN.value
    jersey: Optional[int] = None
    team_score: float = 0.0          # signed: negative -> team 0, positive -> team 1
    #: number -> summed read confidence, accumulated over the tracklet's life
    jersey_votes: dict[int, float] = field(default_factory=dict)
    #: how many individual reads contributed
    jersey_reads: int = 0

    def best_jersey(self) -> tuple[Optional[int], float, int]:
        """(number, share of evidence, number of reads) for the leading number.

        The share matters more than the raw count. Ten reads split 6/4 between
        two numbers is not evidence of anything; six reads that all agree is.
        """
        if not self.jersey_votes:
            return None, 0.0, 0
        total = sum(self.jersey_votes.values())
        num, w = max(self.jersey_votes.items(), key=lambda kv: kv[1])
        return int(num), (w / total if total > 0 else 0.0), self.jersey_reads

    @property
    def duration(self) -> int:
        return self.end - self.start + 1

    def overlaps(self, other: "Tracklet") -> bool:
        return not (self.end < other.start or other.end < self.start)


def _merge_votes(a: dict[int, float], b: dict[int, float]) -> dict[int, float]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + v
    return out


def add_jersey_reads(tracklets: list[Tracklet], reads: pd.DataFrame) -> list[Tracklet]:
    """Attach per-frame jersey-number reads to the tracklets that own them.

    `reads` needs columns track_id, number, confidence. Reads are sparse by
    nature -- a shirt number is only legible when the player happens to face the
    camera and is not occluded, which on a full-pitch panorama is a small
    minority of frames. That sparsity is fine. The point of a number is not to
    label every frame; it is to label the *person*, once, from whatever handful
    of frames happened to be readable.
    """
    by_track: dict[int, list[Tracklet]] = {}
    for t in tracklets:
        for tid in t.track_ids:
            by_track.setdefault(int(tid), []).append(t)

    for r in reads.itertuples(index=False):
        num = getattr(r, "number", None)
        if num is None or (isinstance(num, float) and np.isnan(num)):
            continue
        for t in by_track.get(int(r.track_id), []):
            n = int(num)
            t.jersey_votes[n] = t.jersey_votes.get(n, 0.0) + float(r.confidence)
            t.jersey_reads += 1
    return tracklets


def merge_by_jersey(
    tracklets: list[Tracklet],
    min_share: float = 0.7,
    min_reads: int = 3,
    max_appearance_dist: float = 0.45,
    use_team: bool = True,
) -> list[Tracklet]:
    """Join tracklets that a shirt number says are the same player.

    This is what motion stitching cannot do. Motion gating is bounded by how far
    a human can travel in the gap, so a player lost for fifteen seconds is gone
    forever. A number does not decay: two fragments twenty minutes apart bearing
    the same confident number are the same person, and merging them recovers an
    identity that no amount of trajectory reasoning could.

    The trap: **a shirt number is not a unique identity.** Both teams field a
    number 10. Merging on the number alone will happily weld two opponents into
    one player, which is worse than leaving them apart -- it averages two
    people's statistics and puts a phantom on the wrong half of the pitch. So
    identity is the pair (team, number), and appearance is checked as well, so
    that a wrong team label cannot by itself cause a bad merge.

    Two tracklets that overlap in time are never merged, whatever the number
    says: one player cannot be in two places, so an overlap means at least one
    of the reads is wrong.
    """
    labelled: dict[tuple, list[Tracklet]] = {}
    rest: list[Tracklet] = []
    for t in tracklets:
        num, share, n = t.best_jersey()
        if num is None or share < min_share or n < min_reads:
            rest.append(t)
            continue
        key = (t.team if use_team else "", num)
        labelled.setdefault(key, []).append(t)

    merged: list[Tracklet] = []
    for key, group in labelled.items():
        # Strongest evidence first, so a confident tracklet anchors the identity.
        group.sort(key=lambda t: -(t.best_jersey()[1] * t.best_jersey()[2]))
        used: list[Tracklet] = []
        for t in group:
            target = None
            for u in used:
                if u.overlaps(t):
                    continue
                if (u.embedding is not None and t.embedding is not None):
                    ea = u.embedding / max(np.linalg.norm(u.embedding), 1e-9)
                    eb = t.embedding / max(np.linalg.norm(t.embedding), 1e-9)
                    if 1.0 - float(ea @ eb) > max_appearance_dist:
                        continue
                target = u
                break
            if target is None:
                used.append(t)
            else:
                target.track_ids += t.track_ids
                target.n_frames += t.n_frames
                target.jersey_votes = _merge_votes(target.jersey_votes, t.jersey_votes)
                target.jersey_reads += t.jersey_reads
                if t.start < target.start:
                    target.start, target.start_xy = t.start, t.start_xy
                if t.end > target.end:
                    target.end, target.end_xy = t.end, t.end_xy
                    target.exit_velocity = t.exit_velocity
        for u in used:
            u.jersey = key[1]
        merged.extend(used)

    return sorted(merged + rest, key=lambda t: t.start)


def build_tracklets(state: MatchState, embeddings: Optional[dict[int, np.ndarray]] = None,
                    tail_frames: int = 10) -> list[Tracklet]:
    """One tracklet per track_id in the state."""
    out: list[Tracklet] = []
    players = state.tracks[state.tracks["role"] != Role.BALL.value]
    fps = state.meta.fps

    for tid, g in players.groupby("track_id", observed=True):
        g = g.sort_values("frame_idx")
        f = g["frame_idx"].to_numpy()
        xy = g[["x", "y"]].to_numpy(float)
        if len(g) < 2:
            vel = np.zeros(2)
        else:
            k = min(tail_frames, len(g) - 1)
            dt = max((f[-1] - f[-1 - k]) / fps, 1e-6)
            vel = (xy[-1] - xy[-1 - k]) / dt
        out.append(Tracklet(
            id=int(tid), track_ids=[int(tid)],
            start=int(f[0]), end=int(f[-1]),
            start_xy=xy[0], end_xy=xy[-1],
            exit_velocity=vel, n_frames=len(g),
            embedding=None if embeddings is None else embeddings.get(int(tid)),
            role=str(g["role"].mode().iloc[0]),
        ))
    return sorted(out, key=lambda t: t.start)


def stitch_tracklets(
    tracklets: list[Tracklet],
    fps: float,
    max_gap_s: float = 2.0,
    max_speed: float = 9.0,
    position_tolerance: float = 2.0,
    appearance_weight: float = 8.0,
    max_appearance_dist: float = 0.45,
    max_passes: int = 6,
    jersey_veto: bool = True,
    jersey_min_share: float = 0.7,
    jersey_min_reads: int = 2,
) -> list[Tracklet]:
    """Join fragments that are plausibly the same player, before and after a gap.

    The gate is physical rather than tuned: over a gap of `g` seconds a player
    cannot have travelled further than `max_speed * g`. `max_speed` is set below
    a true sprint (9 m/s) because a player who vanishes is usually in traffic,
    not sprinting into space -- and a loose gate here is expensive, since a wrong
    merge welds two players into one identity permanently.

    Runs several passes so that chains of fragments (A->B->C) collapse fully.

    `jersey_veto` uses shirt numbers to *refuse* merges rather than to make them.
    This ordering matters more than it looks. Measured on real fragmented
    football, using numbers only afterwards to join leftover pieces left the
    welded-identity count untouched at 19, because the damage had already been
    done during motion stitching and a welded tracklet's number evidence is
    mixed beyond repair. Two fragments carrying confidently different numbers
    cannot be the same player, and saying so *before* the merge is what prevents
    the weld. Call `add_jersey_reads` before this, not after.
    """
    work = list(tracklets)

    for _ in range(max_passes):
        work.sort(key=lambda t: t.start)
        n = len(work)
        cost = np.full((n, n), 1e6)

        for i, a in enumerate(work):
            for j, b in enumerate(work):
                if i == j or b.start <= a.end:
                    continue                     # b must begin after a ends
                gap_frames = b.start - a.end
                gap_s = gap_frames / fps
                if gap_s > max_gap_s:
                    continue
                # Where a was heading, coasting through the gap.
                predicted = a.end_xy + a.exit_velocity * gap_s
                reach = max_speed * gap_s + position_tolerance
                if np.linalg.norm(b.start_xy - a.end_xy) > reach:
                    continue
                if jersey_veto:
                    na, sa, ra = a.best_jersey()
                    nb, sb, rb = b.best_jersey()
                    if (na is not None and nb is not None and na != nb
                            and sa >= jersey_min_share and sb >= jersey_min_share
                            and ra >= jersey_min_reads and rb >= jersey_min_reads):
                        continue          # different players, whatever the geometry says
                d = float(np.linalg.norm(b.start_xy - predicted))
                if a.embedding is not None and b.embedding is not None:
                    ea = a.embedding / max(np.linalg.norm(a.embedding), 1e-9)
                    eb = b.embedding / max(np.linalg.norm(b.embedding), 1e-9)
                    cos_d = 1.0 - float(ea @ eb)
                    if cos_d > max_appearance_dist:
                        continue
                    d += appearance_weight * cos_d
                cost[i, j] = d

        if not np.isfinite(cost).any() or (cost < 1e6).sum() == 0:
            break

        ri, ci = linear_sum_assignment(cost)
        merges = [(int(a), int(b)) for a, b in zip(ri, ci) if cost[a, b] < 1e6]
        if not merges:
            break

        # Apply merges, respecting that each tracklet has one successor at most.
        merged_into: dict[int, int] = {}
        consumed: set[int] = set()
        for a, b in merges:
            if a in consumed or b in consumed:
                continue
            merged_into[b] = a
            consumed.add(b)

        new_work: list[Tracklet] = []
        by_index = {i: t for i, t in enumerate(work)}
        for i, t in by_index.items():
            if i in merged_into:
                continue
            cur = t
            j = next((k for k, v in merged_into.items() if v == i), None)
            while j is not None:
                nxt = by_index[j]
                cur = Tracklet(
                    id=cur.id,
                    track_ids=cur.track_ids + nxt.track_ids,
                    start=cur.start, end=nxt.end,
                    start_xy=cur.start_xy, end_xy=nxt.end_xy,
                    exit_velocity=nxt.exit_velocity,
                    n_frames=cur.n_frames + nxt.n_frames,
                    embedding=(cur.embedding if nxt.embedding is None else
                               (nxt.embedding if cur.embedding is None else
                                (cur.n_frames * cur.embedding + nxt.n_frames * nxt.embedding)
                                / (cur.n_frames + nxt.n_frames))),
                    role=cur.role,
                    jersey_votes=_merge_votes(cur.jersey_votes, nxt.jersey_votes),
                    jersey_reads=cur.jersey_reads + nxt.jersey_reads,
                )
                j = next((k for k, v in merged_into.items() if v == j), None)
            new_work.append(cur)

        if len(new_work) == len(work):
            break
        work = new_work

    return sorted(work, key=lambda t: t.start)


def assign_teams_with_roster(
    tracklets: list[Tracklet],
    team_size: int = 11,
    min_frames: int = 25,
) -> list[Tracklet]:
    """Assign teams so that ~`team_size` players are on each side at any moment.

    The improvement over a single global median split is that this respects
    *time*. A global split balances the whole clip and can still leave a given
    minute at 14 v 8; this fills each side up to its quota as tracklets appear,
    resolving the confident ones first so the ambiguous ones inherit whatever
    slot is actually free.

    `team_score` must already be populated (signed appearance preference:
    negative leans home, positive leans away).
    """
    considered = [t for t in tracklets
                  if t.n_frames >= min_frames and t.role != Role.BALL.value]
    if not considered:
        return tracklets

    # Most confident first: those decide the slots, the doubtful ones take what is left.
    order = sorted(considered, key=lambda t: -abs(t.team_score))
    frames_lo = min(t.start for t in considered)
    frames_hi = max(t.end for t in considered)
    span = frames_hi - frames_lo + 1
    load = {Team.HOME.value: np.zeros(span), Team.AWAY.value: np.zeros(span)}

    for t in order:
        s, e = t.start - frames_lo, t.end - frames_lo + 1
        pref = Team.AWAY.value if t.team_score > 0 else Team.HOME.value
        other = Team.HOME.value if pref == Team.AWAY.value else Team.AWAY.value
        # Would preferring `pref` push that side past its quota for most of the
        # tracklet's life? If so, and the other side has room, put it there.
        over_pref = float((load[pref][s:e] >= team_size).mean())
        over_other = float((load[other][s:e] >= team_size).mean())
        chosen = pref if over_pref <= over_other else other
        t.team = chosen
        load[chosen][s:e] += 1

    for t in tracklets:
        if t.n_frames < min_frames and t.team == Team.UNKNOWN.value:
            t.team = Team.UNKNOWN.value
    return tracklets


def apply_to_state(state: MatchState, tracklets: list[Tracklet]) -> MatchState:
    """Rewrite track_id and team in a MatchState from resolved tracklets."""
    remap: dict[int, int] = {}
    teams: dict[int, str] = {}
    jerseys: dict[int, Optional[int]] = {}
    for t in tracklets:
        for old in t.track_ids:
            remap[old] = t.id
            teams[old] = t.team
            jerseys[old] = t.jersey

    df = state.tracks.copy()
    # Build whole columns rather than assigning into slices. `team`/`role` are
    # Categoricals and `track_id` is int32; partial .loc assignment against
    # either raises under pandas 3, so replace the column outright and let
    # MatchState re-coerce on construction.
    for col in ("team", "role"):
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(object)

    df["team"] = [teams.get(t, cur) for t, cur in zip(df["track_id"], df["team"])]
    if any(j is not None for j in jerseys.values()):
        df["jersey"] = [
            jerseys.get(t) if jerseys.get(t) is not None else cur
            for t, cur in zip(df["track_id"], df["jersey"])
        ]
    df["track_id"] = np.asarray([remap.get(t, t) for t in df["track_id"]], dtype="int32")
    return MatchState(state.meta, df)
