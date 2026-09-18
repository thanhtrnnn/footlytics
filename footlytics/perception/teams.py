"""Team assignment from kit appearance.

Two ideas do most of the work here.

1. Describe a player by the colour of their *torso*, not their bounding box.
   A box is mostly grass, shorts and socks; the shirt is the part that differs
   between teams. We crop the upper-middle of the box and drop green pixels.

2. Decide team once per *track*, not per frame. Per-frame kit colour flickers
   with motion blur, shadow and players facing away. Team membership is a
   property of the person and does not change during a half, so we take a
   confidence-weighted vote across the whole track. This converts a noisy
   per-frame signal into a near-deterministic per-player one.

The same descriptor doubles as the appearance embedding for the tracker, which
is why it lives here rather than inside the pipeline.

Measured behaviour (scripts/test_teams.py, synthetic kits with blur, shadow and
sloppy boxes): red-vs-blue gives 100% per-track accuracy; red-vs-red gives 68%;
white-vs-white 64%. Colour alone is not enough when kits clash -- that is what
jersey-number recognition is ultimately for. See `TeamClassifier.diagnose`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..state.schema import Role, Team

# Torso window as a fraction of the detection box: skip the head, stop above
# the shorts. Generous enough to survive sloppy boxes.
TORSO_TOP, TORSO_BOTTOM = 0.15, 0.55
TORSO_LEFT, TORSO_RIGHT = 0.20, 0.80

N_HUE_BINS = 12
N_SAT_BINS = 3
N_VAL_BINS = 3


def _rgb_to_hsv(px: np.ndarray) -> np.ndarray:
    """Vectorised RGB->HSV for an (N, 3) uint8 array. H in [0,1)."""
    a = px.astype(np.float32) / 255.0
    mx, mn = a.max(axis=1), a.min(axis=1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-8
    r, g, b = a[:, 0], a[:, 1], a[:, 2]
    im = a.argmax(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(nz & (im == 0), ((g - b) / d) % 6, h)
        h = np.where(nz & (im == 1), ((b - r) / d) + 2, h)
        h = np.where(nz & (im == 2), ((r - g) / d) + 4, h)
    h = (h / 6.0) % 1.0
    s = np.where(mx > 1e-8, d / np.maximum(mx, 1e-8), 0.0)
    return np.stack([h, s, mx], axis=1)


def kit_descriptor(frame: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
    """A colour histogram of one player's shirt.

    `frame` is RGB (H, W, 3) uint8; `bbox` is (x, y, w, h) in pixels.
    Returns an L1-normalised histogram; all-zeros if the crop is unusable.
    """
    H, W = frame.shape[:2]
    x, y, w, h = map(float, bbox)
    x0 = int(round(x + w * TORSO_LEFT));   x1 = int(round(x + w * TORSO_RIGHT))
    y0 = int(round(y + h * TORSO_TOP));    y1 = int(round(y + h * TORSO_BOTTOM))
    x0, x1 = max(x0, 0), min(x1, W); y0, y1 = max(y0, 0), min(y1, H)
    size = N_HUE_BINS * N_SAT_BINS * N_VAL_BINS
    if x1 - x0 < 2 or y1 - y0 < 2:
        return np.zeros(size, np.float32)

    px = frame[y0:y1, x0:x1].reshape(-1, 3)
    hsv = _rgb_to_hsv(px)
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    # Drop grass (green, reasonably saturated) and near-black shadow. A shirt
    # that is genuinely grass-green is rare; a crop that is mostly pitch is not.
    keep = ~(((hue > 0.20) & (hue < 0.45) & (sat > 0.25)) | (val < 0.12))
    if keep.sum() < 12:
        keep = val >= 0.12                     # fall back rather than return nothing
    if keep.sum() < 4:
        return np.zeros(size, np.float32)

    hb = np.clip((hue[keep] * N_HUE_BINS).astype(int), 0, N_HUE_BINS - 1)
    sb = np.clip((sat[keep] * N_SAT_BINS).astype(int), 0, N_SAT_BINS - 1)
    vb = np.clip((val[keep] * N_VAL_BINS).astype(int), 0, N_VAL_BINS - 1)
    idx = (hb * N_SAT_BINS + sb) * N_VAL_BINS + vb
    hist = np.bincount(idx, minlength=size).astype(np.float32)
    return hist / max(hist.sum(), 1.0)


def _kmeans2(X: np.ndarray, iters: int = 60, restarts: int = 8, seed: int = 0):
    """Minimal 2-means with k-means++ seeding. Two clusters is all we ever need,
    so this avoids a scikit-learn dependency in the hot path."""
    rng = np.random.default_rng(seed)
    best = (np.inf, None, None)
    for _ in range(restarts):
        c0 = X[rng.integers(len(X))]
        d = ((X - c0) ** 2).sum(1)
        p = d / max(d.sum(), 1e-12)
        c1 = X[rng.choice(len(X), p=p)] if d.sum() > 0 else X[rng.integers(len(X))]
        C = np.stack([c0, c1])
        lab = np.zeros(len(X), int)
        for _ in range(iters):
            dist = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            new = dist.argmin(1)
            if (new == lab).all():
                break
            lab = new
            for k in (0, 1):
                if (lab == k).any():
                    C[k] = X[lab == k].mean(0)
        inertia = ((X - C[lab]) ** 2).sum()
        if inertia < best[0]:
            best = (inertia, C.copy(), lab.copy())
    return best[1], best[2], best[0]


@dataclass
class TeamAssignment:
    track_id: int
    team: str
    confidence: float


class TeamClassifier:
    """Clusters kit descriptors into two teams, then votes per track."""

    def __init__(self, seed: int = 0):
        self.centroids: Optional[np.ndarray] = None
        self.seed = seed
        self.balance: float = 0.0

    def fit(self, descriptors: np.ndarray) -> "TeamClassifier":
        X = np.asarray(descriptors, np.float32)
        X = X[X.sum(axis=1) > 0]
        if len(X) < 10:
            raise ValueError(f"only {len(X)} usable kit descriptors -- too few to cluster")
        C, lab, _ = _kmeans2(X, seed=self.seed)
        self.centroids = C
        self.balance = float(1.0 - abs((lab == 0).mean() - 0.5) * 2)
        return self

    def diagnose(self, track_ids: Sequence[int], descriptors: np.ndarray) -> dict:
        """Estimate, without labels, whether this team split can be trusted.

        Two signals, chosen because they are the ones that actually moved with
        accuracy on synthetic kits (scripts/test_teams.py):

        `consistency`
            Mean fraction of a track's frames that agree with that track's own
            majority cluster. Kit colour is a property of a person, so a
            discriminative descriptor gives ~1.0. Clashing kits make individual
            tracks flip between clusters, dropping it to ~0.6.

        `balance`
            How close the split is to 50/50. Eleven-a-side is the strongest
            free prior in football. It catches the degenerate failure where the
            kits are *identical* and k-means splits on noise instead, putting
            nearly everyone in one cluster -- a case `consistency` misses
            completely, because that bad split is perfectly stable.

        What this cannot do
        -------------------
        These detect gross failure, not subtle failure. In testing, white-vs-
        white (64% accurate) and red-vs-orange (100% accurate) scored almost
        identically on both signals. So `ok` is a screen, not a guarantee: it
        catches the matches that are definitely broken. The reliable check is
        still a human glancing once at `cluster_montage()` before the numbers
        go anywhere near a coach.
        """
        track_ids = np.asarray(track_ids)
        X = np.asarray(descriptors, np.float32)
        usable = X.sum(axis=1) > 0
        lab, _ = self._raw_predict(X)

        cons = []
        for tid in np.unique(track_ids):
            m = (track_ids == tid) & usable
            if m.sum() >= 3:
                counts = np.bincount(lab[m], minlength=2)
                cons.append(counts.max() / counts.sum())
        consistency = float(np.mean(cons)) if cons else 0.0
        balance = float(1.0 - abs((lab[usable] == 0).mean() - 0.5) * 2)
        ok = consistency >= 0.85 and balance >= 0.70

        if ok:
            note = "team split looks sound"
        elif balance < 0.70:
            note = (f"lopsided {balance:.0%} split -- the two kits are probably "
                    "near-identical and the clustering has split on noise")
        else:
            note = (f"tracks flip between teams ({consistency:.0%} self-agreement) "
                    "-- likely a kit clash")

        return {
            "consistency": consistency,
            "balance": balance,
            "ok": ok,
            "note": note,
            "summary": (f"consistency {consistency:.2f}, balance {balance:.2f} "
                        f"-- {note}"),
        }

    def cluster_montage(self, frames, boxes, track_ids, descriptors, per_team: int = 12):
        """Sample crops from each cluster so a human can confirm the split in
        one glance. Returns {team_name: [crop, ...]} as RGB arrays."""
        lab, _ = self._raw_predict(np.asarray(descriptors, np.float32))
        out: dict[str, list] = {Team.HOME.value: [], Team.AWAY.value: []}
        seen: dict[int, set] = {0: set(), 1: set()}
        for i, k in enumerate(lab):
            name = Team.HOME.value if k == 0 else Team.AWAY.value
            tid = int(track_ids[i])
            if len(out[name]) >= per_team or tid in seen[k]:
                continue
            x, y, w, h = [int(round(v)) for v in boxes[i]]
            crop = frames[i][max(y, 0):y + h, max(x, 0):x + w]
            if crop.size:
                out[name].append(crop)
                seen[k].add(tid)
        return out

    def _raw_predict(self, descriptors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.centroids is None:
            raise RuntimeError("TeamClassifier.fit() must be called first")
        X = np.asarray(descriptors, np.float32)
        d = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=-1)
        lab = d.argmin(1)
        gap = np.abs(d[:, 0] - d[:, 1]) / np.maximum(d.sum(1), 1e-9)
        return lab, gap

    def assign_tracks(
        self,
        track_ids: Sequence[int],
        descriptors: np.ndarray,
        roles: Optional[Sequence[str]] = None,
        min_votes: int = 5,
        balanced: bool = True,
    ) -> dict[int, TeamAssignment]:
        """One team per track, by confidence-weighted vote over its frames.

        `balanced` applies the strongest free prior in football: eleven a side.
        Plain 2-means optimises cluster compactness and does not care how many
        players end up on each side, so on real footage -- especially with
        similar kits -- it happily returns splits like 17/5. A 17/5 split does
        not merely look wrong on a radar; it silently corrupts every team-level
        number built on top of it, which is exactly why this defaults to True.

        The balancing works on total player-frames rather than track count, so
        short fragmentary tracks cannot outvote the twenty-two real ones.

        It is a real trade-off, not a free win (measured, scripts/test_teams.py):

            kits              balanced=False      balanced=True
            distinct          11v11, 100%         11v11, 100%
            similar           8v14,  86.4%        11v11,  81.8%
            near-identical    18v4,  59.1%        11v11,  72.7%

        With distinct kits it changes nothing. With near-identical kits it is a
        large gain. In between it fixes the count by moving a few players to the
        wrong side -- structurally right, individually slightly worse. That is
        the correct trade for team-level analytics, where an 8v14 split makes
        every aggregate meaningless, but it is worth knowing.

        The prior cannot invent information. When kits genuinely clash it
        produces a confident, balanced, wrong split -- which is why
        `diagnose()` must still be checked and jersey numbers are the real fix.

        Set balanced=False only when the true split genuinely is not even (a red
        card, or a clip that includes substitutes warming up).
        """
        track_ids = np.asarray(track_ids)
        X = np.asarray(descriptors, np.float32)
        usable = X.sum(axis=1) > 0
        lab, gap = self._raw_predict(X)

        out: dict[int, TeamAssignment] = {}
        scored: list[tuple[int, float, float, int]] = []   # tid, score, conf, n_frames

        for tid in np.unique(track_ids):
            m = (track_ids == tid) & usable
            if m.sum() == 0:
                out[int(tid)] = TeamAssignment(int(tid), Team.UNKNOWN.value, 0.0)
                continue
            if roles is not None:
                r = np.asarray(roles)[m]
                if (r == Role.REFEREE.value).mean() > 0.5:
                    out[int(tid)] = TeamAssignment(int(tid), Team.OFFICIAL.value, 1.0)
                    continue
            w = np.array([gap[m][lab[m] == k].sum() for k in (0, 1)])
            if m.sum() < min_votes or w.sum() <= 0:
                out[int(tid)] = TeamAssignment(int(tid), Team.UNKNOWN.value, 0.0)
                continue
            # Signed preference: negative leans to cluster 0, positive to cluster 1.
            score = float((w[1] - w[0]) / w.sum())
            conf = float(w.max() / w.sum())
            scored.append((int(tid), score, conf, int(m.sum())))

        if not scored:
            return out

        if not balanced:
            for tid, score, conf, _ in scored:
                out[tid] = TeamAssignment(
                    tid, Team.AWAY.value if score > 0 else Team.HOME.value, conf)
            return out

        # Split by COUNT, not by player-frames.
        #
        # An earlier version cut at the player-frame-weighted median, reasoning
        # that long tracks should carry more weight. That balances total playing
        # time, which is not the same thing as balancing headcount, and with
        # realistically unequal tracklet lengths it returned something other than
        # 11/11 in 57.7% of cases -- typically 12/10 or 13/9.
        #
        # This is still only the coarse fallback: it has no time information, so
        # it cannot tell whether a given *minute* is balanced. When tracklets
        # with start/end frames are available, use
        # `identity.assign_teams_with_roster`, which fills each side's quota over
        # time and held 100% of frames at exactly 11v11 where this rule managed 0%.
        scored.sort(key=lambda t: t[1])
        cut = len(scored) // 2 - 1
        for i, (tid, score, conf, _) in enumerate(scored):
            team = Team.HOME.value if i <= cut else Team.AWAY.value
            # Confidence is damped for tracks near the boundary: those are the
            # ones the balancing decided, not the appearance signal.
            edge = 1.0 - abs(i - cut) / max(len(scored), 1)
            out[tid] = TeamAssignment(tid, team, float(conf * (1.0 - 0.5 * edge)))
        return out

    def track_scores(self, track_ids: Sequence[int], descriptors: np.ndarray
                     ) -> dict[int, float]:
        """Signed team preference per track: negative leans home, positive away.

        Kept separate from `assign_tracks` so that the *decision* can be made
        elsewhere -- specifically by `identity.assign_teams_with_roster`, which
        knows when each tracklet was on the pitch and can therefore respect the
        eleven-a-side quota minute by minute rather than only in aggregate.
        """
        track_ids = np.asarray(track_ids)
        X = np.asarray(descriptors, np.float32)
        usable = X.sum(axis=1) > 0
        lab, gap = self._raw_predict(X)
        out: dict[int, float] = {}
        for tid in np.unique(track_ids):
            m = (track_ids == tid) & usable
            if m.sum() == 0:
                out[int(tid)] = 0.0
                continue
            w = np.array([gap[m][lab[m] == k].sum() for k in (0, 1)])
            out[int(tid)] = 0.0 if w.sum() <= 0 else float((w[1] - w[0]) / w.sum())
        return out

    def kit_colors(self, descriptors: np.ndarray) -> dict[str, tuple[int, int, int]]:
        """Representative RGB per cluster, for the radar legend."""
        lab, _ = self._raw_predict(descriptors)
        cols = {}
        for k, name in ((0, Team.HOME.value), (1, Team.AWAY.value)):
            c = self.centroids[k]
            top = int(c.argmax())
            hb = top // (N_SAT_BINS * N_VAL_BINS)
            sb = (top // N_VAL_BINS) % N_SAT_BINS
            vb = top % N_VAL_BINS
            h = (hb + 0.5) / N_HUE_BINS
            s = (sb + 0.5) / N_SAT_BINS
            v = (vb + 0.5) / N_VAL_BINS
            i = int(h * 6) % 6
            f = h * 6 - int(h * 6)
            p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
            rgb = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
            cols[name] = tuple(int(255 * c_) for c_ in rgb)
        return cols


# ------------------------------------------------------------- officials split

def split_officials(track_ids: np.ndarray, descriptors: np.ndarray, max_share: float = 0.25,
                    min_separation: float = 0.6, seed: int = 0) -> set[int]:
    """Track ids that wear a third kit (referees), for detectors without a referee class.

    Fits three clusters on per-track mean descriptors. The smallest cluster is "officials"
    only if it is small (<= `max_share` of tracks) and genuinely separate: its centroid must
    sit at least `min_separation` times the two-team distance away from the nearest team
    centroid. Splitting one team's shirts into two sub-clusters fails that test, so with two
    kits the result is empty and the ordinary 2-means fit is untouched.
    """
    from sklearn.cluster import KMeans

    T = np.asarray(track_ids)
    uniq = np.unique(T)
    if len(uniq) < 6:
        return set()
    means = np.stack([descriptors[T == t].mean(axis=0) for t in uniq])
    km = KMeans(n_clusters=3, n_init=10, random_state=seed).fit(means)
    sizes = np.bincount(km.labels_, minlength=3)
    order = np.argsort(sizes)
    small, big1, big2 = order[0], order[1], order[2]
    if sizes[small] > max_share * len(uniq):
        return set()
    c = km.cluster_centers_
    team_gap = np.linalg.norm(c[big1] - c[big2])
    sep = min(np.linalg.norm(c[small] - c[big1]), np.linalg.norm(c[small] - c[big2]))
    if team_gap <= 0 or sep < min_separation * team_gap:
        return set()
    return {int(t) for t in uniq[km.labels_ == small]}
