"""Synthetic 22-player match to measure identity stability under realistic noise."""
import sys; sys.path.insert(0, ".")
import numpy as np
from footlytics.perception.track import PitchTracker, TrackerConfig
from footlytics.state.schema import Role

FPS, N_FRAMES, N_PLAYERS = 25.0, 25 * 60, 22   # one minute


def simulate(seed=0):
    """Smooth, bounded player motion plus a fast ball. Returns (T, N+1, 2) metres."""
    rng = np.random.default_rng(seed)
    pos = np.column_stack([rng.uniform(-45, 45, N_PLAYERS), rng.uniform(-30, 30, N_PLAYERS)])
    vel = rng.normal(0, 1.5, (N_PLAYERS, 2))
    ball = np.array([0.0, 0.0]); bvel = rng.normal(0, 8, 2)
    out = np.zeros((N_FRAMES, N_PLAYERS + 1, 2))
    for t in range(N_FRAMES):
        vel += rng.normal(0, 0.35, (N_PLAYERS, 2))              # acceleration
        sp = np.linalg.norm(vel, axis=1, keepdims=True)
        vel = np.where(sp > 8.0, vel / sp * 8.0, vel)            # cap at 8 m/s
        pos += vel / FPS
        for ax, lim in ((0, 52.0), (1, 33.0)):                   # reflect at the lines
            over = np.abs(pos[:, ax]) > lim
            pos[over, ax] = np.sign(pos[over, ax]) * lim
            vel[over, ax] *= -1
        bvel += rng.normal(0, 3.0, 2)
        bsp = np.linalg.norm(bvel)
        if bsp > 25: bvel = bvel / bsp * 25
        ball += bvel / FPS
        ball = np.clip(ball, [-52, -33], [52, 33])
        out[t, :N_PLAYERS], out[t, N_PLAYERS] = pos, ball
    return out


def run(truth, meas_sigma, dropout, false_pos_rate, seed=1):
    rng = np.random.default_rng(seed)
    trk = PitchTracker(TrackerConfig(fps=FPS))
    switches = 0
    assigned: dict[int, int] = {}       # gt index -> current track id
    covered = 0; total = 0

    for t in range(len(truth)):
        xy, roles, gt_of_det = [], [], []
        for i in range(N_PLAYERS + 1):
            if rng.random() < dropout:
                continue
            xy.append(truth[t, i] + rng.normal(0, meas_sigma, 2))
            roles.append(Role.BALL.value if i == N_PLAYERS else Role.PLAYER.value)
            gt_of_det.append(i)
        for _ in range(rng.poisson(false_pos_rate)):             # spurious detections
            xy.append(rng.uniform([-52, -33], [52, 33]))
            roles.append(Role.PLAYER.value); gt_of_det.append(-1)

        if not xy:
            continue
        matches = trk.update(np.array(xy), roles)
        confirmed = {t_.id for t_ in trk.active()}
        for di, tid in matches:
            gt = gt_of_det[di]
            if gt < 0 or tid not in confirmed:
                continue
            total += 1
            prev = assigned.get(gt)
            if prev is None:
                assigned[gt] = tid
            elif prev != tid:
                switches += 1
                assigned[gt] = tid
            covered += 1
    return switches, covered / max(total, 1), len(trk.active())


# --------------------------------------------------------------------------
# Does appearance actually close the duel gap?
# --------------------------------------------------------------------------
def run_appearance(truth, meas_sigma, dropout, fp_rate, emb_noise, dim=32, seed=1):
    """Same harness, but each object carries a stable appearance vector.

    `emb_noise` models how much the crop's appearance wobbles frame to frame.
    Two teams of 11 share a kit signature, so teammates are much harder to tell
    apart than opponents -- exactly the real situation.
    """
    rng = np.random.default_rng(seed)
    kit = rng.normal(0, 1, (2, dim))                       # home / away kit
    ident = rng.normal(0, 1, (N_PLAYERS + 1, dim)) * 0.45   # per-player ReID signal
    base = np.array([kit[i // 11 if i < N_PLAYERS else 0] for i in range(N_PLAYERS + 1)]) + ident

    trk = PitchTracker(TrackerConfig(fps=FPS))
    switches = 0; assigned = {}; total = 0
    for t in range(len(truth)):
        xy, roles, gt, emb = [], [], [], []
        for i in range(N_PLAYERS + 1):
            if rng.random() < dropout:
                continue
            xy.append(truth[t, i] + rng.normal(0, meas_sigma, 2))
            roles.append(Role.BALL.value if i == N_PLAYERS else Role.PLAYER.value)
            gt.append(i)
            emb.append(base[i] + rng.normal(0, emb_noise, dim))
        for _ in range(rng.poisson(fp_rate)):
            xy.append(rng.uniform([-52, -33], [52, 33]))
            roles.append(Role.PLAYER.value); gt.append(-1)
            emb.append(rng.normal(0, 1, dim))
        if not xy:
            continue
        matches = trk.update(np.array(xy), roles, embeddings=np.array(emb))
        confirmed = {x.id for x in trk.active()}
        for di, tid in matches:
            g = gt[di]
            if g < 0 or tid not in confirmed:
                continue
            total += 1
            if g not in assigned:
                assigned[g] = tid
            elif assigned[g] != tid:
                switches += 1; assigned[g] = tid
    return switches, len(trk.active())



def main():
    truth = simulate()
    print("=" * 74)
    print(f"{N_PLAYERS} players + ball, {N_FRAMES} frames @ {FPS:g} fps "
          f"({N_FRAMES / FPS:.0f}s)  =  {N_FRAMES * (N_PLAYERS + 1):,} detections")
    print("=" * 74)
    print(f"{'measurement noise':>18} {'dropout':>9} {'false pos/f':>12} "
          f"{'ID switches':>12} {'live tracks':>12}")
    for sigma, drop, fp in [
        (0.05, 0.00, 0.0),
        (0.20, 0.00, 0.0),
        (0.35, 0.05, 0.0),
        (0.35, 0.15, 0.5),
        (0.60, 0.25, 1.0),
        (1.00, 0.35, 2.0),
    ]:
        sw, cov, live = run(truth, sigma, drop, fp)
        print(f"{sigma:>15.2f} m {drop:>8.0%} {fp:>12.1f} {sw:>12d} {live:>12d}")

    print("\nA perfect tracker reports 0 switches and 23 live tracks.")
    print("Rows are cumulative-difficulty, not independent: the last row is a")
    print("deliberately abusive detector (1 m position error, a third of players")
    print("missed every frame, 2 phantom players per frame).")



    print("\n" + "=" * 74)
    print("with appearance features (kit signature + weak per-player ReID)")
    print("=" * 74)
    print(f"{'measurement noise':>18} {'dropout':>9} {'false pos/f':>12} "
          f"{'ID switches':>12} {'live tracks':>12}")
    for sigma, drop, fp in [
        (0.05, 0.00, 0.0),
        (0.20, 0.00, 0.0),
        (0.35, 0.05, 0.0),
        (0.35, 0.15, 0.5),
        (0.60, 0.25, 1.0),
        (1.00, 0.35, 2.0),
    ]:
        sw, live = run_appearance(truth, sigma, drop, fp, emb_noise=0.35)
        print(f"{sigma:>15.2f} m {drop:>8.0%} {fp:>12.1f} {sw:>12d} {live:>12d}")


if __name__ == "__main__":
    main()
