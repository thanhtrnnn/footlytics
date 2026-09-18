"""Tracklet stitching, measured on real football movement.

Takes SoccerTrack v2's hand-annotated ground truth -- 22 players, 6,000 frames,
no fragmentation -- and breaks it deliberately, the way a real tracker does:
tracks are cut at random points with gaps where the player was lost. Then we
stitch and ask how much identity was recovered.

Using real trajectories matters. Players in traffic decelerate, turn and cluster
in ways a random walk does not, and those are exactly the moments a tracker
loses them.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from footlytics.data.soccertrack import SoccerTrackCalibration, load_mot_groundtruth
from footlytics.state.schema import MatchState
from footlytics.perception.identity import build_tracklets, stitch_tracklets

D = "data/raw/soccertrack/117092"
gt = load_mot_groundtruth(f"{D}/117092.txt", SoccerTrackCalibration.load(D, 117092),
                          match_id="ST2")
FPS = gt.meta.fps
print(f"ground truth: {gt.players.track_id.nunique()} players, "
      f"{gt.n_frames} frames @ {FPS:g} fps\n")


def fragment(state, n_cuts_per_player, gap_range_s, seed=0):
    """Cut each player's track into pieces separated by gaps."""
    rng = np.random.default_rng(seed)
    rows, owner, nxt = [], {}, 1
    for tid, g in state.players.groupby("track_id", observed=True):
        g = g.sort_values("frame_idx").reset_index(drop=True)
        cuts = sorted(rng.choice(np.arange(50, len(g) - 50),
                                 size=n_cuts_per_player, replace=False))
        prev = 0
        for c in list(cuts) + [len(g)]:
            gap = int(rng.uniform(*gap_range_s) * FPS)
            seg = g.iloc[prev:c]
            if len(seg) > 5:
                s = seg.copy(); s["track_id"] = nxt
                rows.append(s); owner[nxt] = int(tid); nxt += 1
            prev = min(c + gap, len(g))
    return MatchState(state.meta, pd.concat(rows, ignore_index=True)), owner


def evaluate(tracklets, owner, n_truth):
    """Purity (no welded players) and fragmentation (players per identity)."""
    impure = 0
    per_player = {}
    for t in tracklets:
        owners = {owner[tid] for tid in t.track_ids}
        if len(owners) > 1:
            impure += 1
        for o in owners:
            per_player.setdefault(o, set()).add(t.id)
    frags = [len(v) for v in per_player.values()]
    return {
        "tracklets": len(tracklets),
        "impure": impure,
        "median_frags_per_player": float(np.median(frags)) if frags else 0.0,
        "players_fully_recovered": sum(1 for f in frags if f == 1),
        "n_truth": n_truth,
    }


print(f"{'scenario':<34}{'before':>8}{'after':>8}{'welded':>9}"
      f"{'frags/player':>14}{'perfect':>9}")
print("-" * 82)
for cuts, gaps, label in [
    (2,  (0.2, 0.6), "2 cuts, short gaps (0.2-0.6s)"),
    (5,  (0.2, 0.8), "5 cuts, short gaps"),
    (10, (0.3, 1.2), "10 cuts, medium gaps"),
    (20, (0.3, 1.5), "20 cuts, long gaps"),
    (40, (0.5, 2.0), "40 cuts, very long gaps"),
]:
    frag_state, owner = fragment(gt, cuts, gaps)
    tl = build_tracklets(frag_state)
    before = len(tl)
    stitched = stitch_tracklets(tl, FPS)
    r = evaluate(stitched, owner, 22)
    print(f"{label:<34}{before:>8}{r['tracklets']:>8}{r['impure']:>9}"
          f"{r['median_frags_per_player']:>14.1f}{r['players_fully_recovered']:>7}/22")

print("\n'welded' = tracklets containing frames from more than one real player.")
print("That is the failure that matters: a merged identity is worse than a")
print("split one, because it silently averages two players into one set of stats.")


# --------------------------------------------------------------------------
# Does kit appearance rescue the heavily-fragmented cases?
# --------------------------------------------------------------------------
def with_embeddings(frag_state, owner, dim=32, kit_noise=0.35, seed=3):
    """Give each real player a stable appearance vector, shared within a team.

    Mirrors what a kit descriptor plus a weak ReID signal actually provides:
    teams are easy to tell apart, teammates much less so.
    """
    rng = np.random.default_rng(seed)
    kits = rng.normal(0, 1, (2, dim))
    truth_ids = sorted(set(owner.values()))
    ident = {p: kits[i % 2] + rng.normal(0, 1, dim) * 0.45
             for i, p in enumerate(truth_ids)}
    return {tid: ident[owner[tid]] + rng.normal(0, kit_noise, dim)
            for tid in owner}


print("\n" + "=" * 82)
print("with appearance embeddings (kit signature + weak per-player ReID)")
print("=" * 82)
print(f"{'scenario':<34}{'before':>8}{'after':>8}{'welded':>9}"
      f"{'frags/player':>14}{'perfect':>9}")
print("-" * 82)
for cuts, gaps, label in [
    (10, (0.3, 1.2), "10 cuts, medium gaps"),
    (20, (0.3, 1.5), "20 cuts, long gaps"),
    (40, (0.5, 2.0), "40 cuts, very long gaps"),
]:
    frag_state, owner = fragment(gt, cuts, gaps)
    emb = with_embeddings(frag_state, owner)
    tl = build_tracklets(frag_state, embeddings=emb)
    before = len(tl)
    stitched = stitch_tracklets(tl, FPS)
    r = evaluate(stitched, owner, 22)
    print(f"{label:<34}{before:>8}{r['tracklets']:>8}{r['impure']:>9}"
          f"{r['median_frags_per_player']:>14.1f}{r['players_fully_recovered']:>7}/22")


# --------------------------------------------------------------------------
# Roster-constrained team assignment vs a single global split
# --------------------------------------------------------------------------
from footlytics.perception.identity import assign_teams_with_roster
from footlytics.state.schema import Team

print("\n" + "=" * 82)
print("team assignment: global median split vs time-aware roster quota")
print("=" * 82)

def team_balance(tracklets, n_frames):
    """Per-frame headcount difference between the two sides."""
    lo = np.zeros(n_frames); hi = np.zeros(n_frames)
    for t in tracklets:
        s, e = max(t.start, 0), min(t.end + 1, n_frames)
        if t.team == Team.HOME.value: lo[s:e] += 1
        elif t.team == Team.AWAY.value: hi[s:e] += 1
    active = (lo + hi) > 0
    diff = np.abs(lo - hi)[active]
    exact = ((lo == 11) & (hi == 11))[active]
    return diff, exact, lo, hi

def global_split(tracklets):
    """The previous approach: one median cut over all tracklets by weight."""
    ts = sorted([t for t in tracklets if t.n_frames >= 25], key=lambda t: t.team_score)
    if not ts: return tracklets
    w = np.array([t.n_frames for t in ts], float); cum = np.cumsum(w)
    cut = int(np.searchsorted(cum, cum[-1] / 2))
    for i, t in enumerate(ts):
        t.team = Team.HOME.value if i <= cut else Team.AWAY.value
    return tracklets

for cuts, gaps, score_noise, label in [
    (5,  (0.2, 0.8), 0.15, "lightly fragmented, clear kits"),
    (20, (0.3, 1.5), 0.15, "heavily fragmented, clear kits"),
    (20, (0.3, 1.5), 0.60, "heavily fragmented, ambiguous kits"),
]:
    frag_state, owner = fragment(gt, cuts, gaps)
    emb = with_embeddings(frag_state, owner)
    tl = stitch_tracklets(build_tracklets(frag_state, embeddings=emb), FPS)

    truth_ids = sorted(set(owner.values()))
    true_team = {p: i % 2 for i, p in enumerate(truth_ids)}
    rng = np.random.default_rng(11)
    for t in tl:
        owners = [owner[x] for x in t.track_ids]
        sign = 1.0 if true_team[max(set(owners), key=owners.count)] else -1.0
        t.team_score = float(sign + rng.normal(0, score_noise))

    import copy
    a = global_split(copy.deepcopy(tl))
    b = assign_teams_with_roster(copy.deepcopy(tl), team_size=11)
    print(f"\n  {label}")
    for name, res in (("global median split", a), ("roster quota (time-aware)", b)):
        diff, exact, lo, hi = team_balance(res, gt.n_frames)
        print(f"     {name:<28} median imbalance {np.median(diff):4.1f} "
              f"| worst {diff.max():4.0f} | frames at exactly 11v11 {exact.mean():5.1%}")


# --------------------------------------------------------------------------
# End-to-end: the exact post-processing chain the pipeline now runs
# --------------------------------------------------------------------------
from footlytics.perception.identity import apply_to_state
from footlytics.perception.teams import TeamClassifier

print("\n" + "=" * 82)
print("end-to-end identity chain on real football (fragmented ground truth)")
print("=" * 82)

def histogram_descriptors(owner, dim=24, noise=0.06, seed=5):
    """Kit descriptors shaped like the real thing: non-negative, L1-normalised."""
    rng = np.random.default_rng(seed)
    truth = sorted(set(owner.values()))
    team_of = {p: i % 2 for i, p in enumerate(truth)}
    out = {}
    for tid, p in owner.items():
        base = np.zeros(dim); base[0 if team_of[p] == 0 else dim // 2] = 1.0
        v = np.abs(base + rng.normal(0, noise, dim))
        out[tid] = v / v.sum()
    return out, team_of

frag_state, owner = fragment(gt, 10, (0.3, 1.2), seed=1)
desc_by_track, team_of = histogram_descriptors(owner)

rows, tids = [], []
for tid, g in frag_state.players.groupby("track_id", observed=True):
    for _ in range(len(g)):
        rows.append(desc_by_track[tid]); tids.append(tid)
Dm, Tm = np.array(rows), np.array(tids)

clf = TeamClassifier().fit(Dm)
scores = clf.track_scores(Tm, Dm)
emb = {int(t): Dm[Tm == t].mean(axis=0) for t in np.unique(Tm)}

tl = build_tracklets(frag_state, embeddings=emb)
before = len(tl)
tl = stitch_tracklets(tl, FPS)
for t in tl:
    t.team_score = float(np.mean([scores.get(x, 0.0) for x in t.track_ids]))
tl = assign_teams_with_roster(tl, team_size=11)
final = apply_to_state(frag_state, tl)

print(f"  fragments in           : {before}")
print(f"  tracklets out          : {len(tl)}   (truth: 22)")
r = evaluate(tl, owner, 22)
print(f"  welded identities      : {r['impure']}")
print(f"  players fully recovered: {r['players_fully_recovered']}/22")

acc = np.mean([
    (t.team == Team.HOME.value) == (team_of[max(set(o), key=list(o).count)] == 0)
    for t in tl
    for o in [[owner[x] for x in t.track_ids]]
])
print(f"  team accuracy          : {max(acc, 1 - acc):.1%}")
counts = pd.Series([t.team for t in tl]).value_counts().to_dict()
print(f"  team split             : {counts}")
print(f"\n  MatchState.validate()  : ", end="")
probs = final.validate()
print("clean" if not probs else "; ".join(probs))


# --------------------------------------------------------------------------
# Jersey numbers: recovering identity that motion cannot
# --------------------------------------------------------------------------
from footlytics.perception.identity import add_jersey_reads, merge_by_jersey

print("\n" + "=" * 82)
print("jersey-anchored merging (numbers are read on only a few frames)")
print("=" * 82)

def simulate_jersey_reads(frag_state, owner, p_legible=0.05, p_correct=0.85, seed=9):
    """Sparse, sometimes-wrong shirt-number reads, as real OCR produces.

    A number is legible only when the player faces the camera and is not
    occluded -- on a full-pitch panorama that is a small minority of frames.
    """
    rng = np.random.default_rng(seed)
    truth = sorted(set(owner.values()))
    true_num = {p: (i % 11) + 1 for i, p in enumerate(truth)}
    team_of = {p: i % 2 for i, p in enumerate(truth)}
    rows = []
    for tid, g in frag_state.players.groupby("track_id", observed=True):
        n_reads = rng.binomial(len(g), p_legible)
        for _ in range(n_reads):
            correct = rng.random() < p_correct
            num = true_num[owner[tid]] if correct else int(rng.integers(1, 12))
            rows.append({"track_id": tid, "number": num,
                         "confidence": rng.uniform(0.6, 1.0) if correct
                                       else rng.uniform(0.3, 0.7)})
    return pd.DataFrame(rows), true_num, team_of

print(f"{'scenario':<30}{'method':<22}{'tracklets':>10}{'welded':>8}{'recovered':>11}")
print("-" * 82)
for cuts, gaps, p_leg, label in [
    (20, (0.3, 1.5), 0.05, "20 cuts, 5% legible"),
    (40, (0.5, 2.0), 0.05, "40 cuts, 5% legible"),
    (40, (0.5, 2.0), 0.15, "40 cuts, 15% legible"),
]:
    frag_state, owner = fragment(gt, cuts, gaps, seed=2)
    emb = with_embeddings(frag_state, owner)
    reads, true_num, team_of = simulate_jersey_reads(frag_state, owner, p_legible=p_leg)

    base = stitch_tracklets(build_tracklets(frag_state, embeddings=emb), FPS)
    rb = evaluate(base, owner, 22)

    # numbers attached BEFORE stitching, so they can veto bad merges
    raw = add_jersey_reads(build_tracklets(frag_state, embeddings=emb), reads)
    withj = stitch_tracklets(raw, FPS, jersey_veto=True)
    for t in withj:                       # teams first: (team, number) is the identity
        o = [owner[x] for x in t.track_ids]
        t.team_score = 1.0 if team_of[max(set(o), key=o.count)] else -1.0
    withj = assign_teams_with_roster(withj, team_size=11)
    withj = merge_by_jersey(withj)
    rj = evaluate(withj, owner, 22)

    print(f"{label:<30}{'motion only':<22}{rb['tracklets']:>10}{rb['impure']:>8}"
          f"{rb['players_fully_recovered']:>9}/22")
    print(f"{'':<30}{'+ jersey numbers':<22}{rj['tracklets']:>10}{rj['impure']:>8}"
          f"{rj['players_fully_recovered']:>9}/22")
