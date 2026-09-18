"""Synthetic kit test: red vs blue shirts on grass, with realistic corruption."""
import sys; sys.path.insert(0, ".")
import numpy as np
from footlytics.perception.teams import kit_descriptor, TeamClassifier
from footlytics.state.schema import Team

rng = np.random.default_rng(0)
GRASS = np.array([60, 130, 60])
KITS = {"home": np.array([200, 40, 40]), "away": np.array([40, 60, 190])}


def fake_crop(kit, blur=0.0, shadow=1.0, box_slop=0.0):
    """A 40x22 player on grass: shirt in the torso band, shorts/legs below."""
    h, w = 40, 22
    img = np.tile(GRASS, (h, w, 1)).astype(float)
    img += rng.normal(0, 12, img.shape)                      # grass texture
    t0, t1 = int(h * 0.18), int(h * 0.55)
    img[t0:t1, 3:w - 3] = KITS[kit]
    img[t1:int(h * 0.75), 5:w - 5] = [235, 235, 235]         # white shorts
    img[int(h * .05):t0, 7:w - 7] = [205, 170, 140]          # head
    img = img * shadow + rng.normal(0, blur * 40, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    s = box_slop
    bbox = (0 - s * w, 0 - s * h, w * (1 + 2 * s), h * (1 + 2 * s))
    return img, (0, 0, w, h) if s == 0 else bbox



def main():
    print("=" * 66)
    print("1. clean crops -- can 2-means separate the kits at all?")
    descs, truth = [], []
    for kit in ("home", "away"):
        for _ in range(60):
            img, bb = fake_crop(kit)
            descs.append(kit_descriptor(img, bb)); truth.append(kit)
    descs = np.array(descs)
    clf = TeamClassifier().fit(descs)
    lab, gap = clf._raw_predict(descs)
    truth_bin = np.array([0 if t == "home" else 1 for t in truth])
    acc = max((lab == truth_bin).mean(), (lab != truth_bin).mean())
    print(f"   per-crop cluster purity: {acc:.1%}")
    print(f"   {clf.diagnose(np.arange(len(descs))//60, descs)['summary']}")

    print("\n2. realistic corruption -- blur, shadow, sloppy boxes")
    descs, truth, tids = [], [], []
    for i in range(44):                      # 44 'tracks', 22 per team
        kit = "home" if i < 22 else "away"
        for _ in range(30):                  # 30 frames each
            img, bb = fake_crop(kit, blur=rng.uniform(0, .5),
                                shadow=rng.uniform(.55, 1.15), box_slop=rng.uniform(0, .2))
            descs.append(kit_descriptor(img, bb)); truth.append(kit); tids.append(i)
    descs = np.array(descs); tids = np.array(tids)
    clf = TeamClassifier().fit(descs)
    lab, gap = clf._raw_predict(descs)
    truth_bin = np.array([0 if t == "home" else 1 for t in truth])
    per_frame = max((lab == truth_bin).mean(), (lab != truth_bin).mean())

    assign = clf.assign_tracks(tids, descs)
    got = np.array([assign[i].team for i in range(44)])
    exp_a = np.array([Team.HOME.value] * 22 + [Team.AWAY.value] * 22)
    exp_b = np.array([Team.AWAY.value] * 22 + [Team.HOME.value] * 22)
    per_track = max((got == exp_a).mean(), (got == exp_b).mean())
    print(f"   per-frame accuracy: {per_frame:.1%}")
    print(f"   per-track accuracy: {per_track:.1%}   <- voting is the whole point")
    print(f"   mean track confidence: {np.mean([a.confidence for a in assign.values()]):.2f}")
    print(f"   {clf.diagnose(tids, descs)['summary']}")

    print("\n3. colour clash -- both teams in similar red")
    KITS["away"] = np.array([190, 70, 60])
    d2, t2 = [], []
    for i in range(44):
        kit = "home" if i < 22 else "away"
        for _ in range(30):
            img, bb = fake_crop(kit, blur=.3, shadow=rng.uniform(.6, 1.1))
            d2.append(kit_descriptor(img, bb)); t2.append(i)
    d2 = np.array(d2)
    clf2 = TeamClassifier().fit(d2)
    a2 = clf2.assign_tracks(np.array(t2), d2)
    got2 = np.array([a2[i].team for i in range(44)])
    acc2 = max((got2 == exp_a).mean(), (got2 == exp_b).mean())
    print(f"   {clf2.diagnose(np.array(t2), d2)['summary']}")
    print(f"   per-track accuracy: {acc2:.1%}   (vs {per_track:.1%} with distinct kits)")
    print("   -> diagnose() screens out matches that are definitely broken; it does")
    print("      NOT certify the rest. Confirm with cluster_montage() before shipping.")
    print("\n" + "=" * 66)


    # --------------------------------------------------------------------------
    # The 11-v-11 constraint: does it fix a lopsided split?
    # --------------------------------------------------------------------------
    print("\n" + "=" * 66)
    print("4. lopsided splits -- what the balanced=True prior is for")
    print("=" * 66)

    def lopsided(kit_a, kit_b, blur, label):
        rng2 = np.random.default_rng(7)
        KITS["home"], KITS["away"] = np.array(kit_a), np.array(kit_b)
        d, t = [], []
        for i in range(22):                       # 11 per side, as in real football
            kit = "home" if i < 11 else "away"
            for _ in range(40):
                img, bb = fake_crop(kit, blur=blur, shadow=rng2.uniform(.5, 1.2))
                d.append(kit_descriptor(img, bb)); t.append(i)
        d = np.array(d); t = np.array(t)
        clf = TeamClassifier().fit(d)
        ea = np.array([Team.HOME.value] * 11 + [Team.AWAY.value] * 11)
        eb = np.array([Team.AWAY.value] * 11 + [Team.HOME.value] * 11)

        row = []
        for bal in (False, True):
            a = clf.assign_tracks(t, d, balanced=bal)
            got = np.array([a[i].team for i in range(22)])
            n_home = int((got == Team.HOME.value).sum())
            acc = max((got == ea).mean(), (got == eb).mean())
            row.append((n_home, 22 - n_home, acc))
        print(f"  {label}")
        print(f"     balanced=False -> {row[0][0]:2d} v {row[0][1]:2d}   accuracy {row[0][2]:5.1%}")
        print(f"     balanced=True  -> {row[1][0]:2d} v {row[1][1]:2d}   accuracy {row[1][2]:5.1%}")

    lopsided([200, 40, 40], [40, 60, 190], 0.35, "distinct kits (red vs blue)")
    lopsided([200, 40, 40], [190, 80, 60], 0.35, "similar kits (red vs dark orange)")
    lopsided([230, 230, 230], [205, 205, 210], 0.35, "near-identical (white vs off-white)")
    print("\n  The prior cannot invent information: when kits are genuinely")
    print("  indistinguishable it enforces 11 v 11 but picks the wrong eleven.")
    print("  That is why validate() flags the split AND diagnose() flags the clash.")


if __name__ == "__main__":
    main()
