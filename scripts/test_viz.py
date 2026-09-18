"""Build a synthetic MatchState in a 4-4-2 and render the radar."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from footlytics.state.schema import MatchState, MatchMeta, TeamInfo, Role, Team
from footlytics.analytics.kinematics import add_kinematics, distance_summary

FPS = 25.0
FORMATION = {                     # 4-4-2, as (x, y) offsets from own goal line
    "gk":  [(-48, 0)],
    "def": [(-34, -20), (-34, -7), (-34, 7), (-34, 20)],
    "mid": [(-16, -22), (-16, -8), (-16, 8), (-16, 22)],
    "fwd": [(-4, -9), (-4, 9)],
}

rows, rng = [], np.random.default_rng(3)
N_FRAMES = 200
home = [(p, r) for r, ps in FORMATION.items() for p in ps]
away = [((-x, -y), r) for (x, y), r in home]

for f in range(N_FRAMES):
    t = f / FPS
    drift = 6.0 * np.sin(t * 0.6)                      # the block shuffling across
    for side, squad in (("home", home), ("away", away)):
        for i, ((bx, by), r) in enumerate(squad):
            jitter = rng.normal(0, 0.25, 2)
            x = bx + drift + jitter[0]
            y = by + 2.5 * np.sin(t * 0.9 + i) + jitter[1]
            rows.append(dict(
                frame_idx=f, period=1, timestamp=t,
                track_id=(1 if side == "home" else 100) + i,
                role=Role.GOALKEEPER.value if r == "gk" else Role.PLAYER.value,
                team=Team.HOME.value if side == "home" else Team.AWAY.value,
                jersey=i + 1, bbox_x=0, bbox_y=0, bbox_w=10, bbox_h=30, det_conf=0.9,
                x=x, y=y, z=np.nan))
    rows.append(dict(frame_idx=f, period=1, timestamp=t, track_id=999,
                     role=Role.BALL.value, team=Team.UNKNOWN.value, jersey=np.nan,
                     bbox_x=0, bbox_y=0, bbox_w=4, bbox_h=4, det_conf=0.5,
                     x=12 * np.sin(t * 1.3) + drift, y=14 * np.cos(t * 0.8), z=0.4))

meta = MatchMeta(match_id="SYNTH-001", fps=FPS, source_type="fixed_panoramic",
                 home=TeamInfo("Hà Nội FC", "HAN"), away=TeamInfo("Viettel FC", "VTL"),
                 competition="V.League 1 (synthetic)")
state = MatchState(meta, pd.DataFrame(rows))
state.tracks = add_kinematics(state.tracks, FPS)
print(state)

print("\nvalidate():")
probs = state.validate()
print("  no problems found" if not probs else "\n".join(f"  - {p}" for p in probs))

print("\nsave/load round-trip:")
state.save("data/out/synth")
back = MatchState.load("data/out/synth")
assert len(back.tracks) == len(state.tracks) and back.meta.home.name == "Hà Nội FC"
print(f"  {back}")

print("\nphysical summary (top 5 by distance):")
print(distance_summary(state.players, FPS).head(5).to_string(index=False))

fig, ax = plt.subplots(figsize=(13, 8.5), dpi=110)
fig.patch.set_facecolor("#0d1117")
from footlytics.viz.radar import plot_frame
plot_frame(state, 150, ax=ax, trail_frames=40)
fig.savefig("data/out/radar_frame.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
print("\nwrote data/out/radar_frame.png")
