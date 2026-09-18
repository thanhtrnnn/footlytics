"""Full-match tactical analysis: shape, block, pressing, transitions."""
import sys, time; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from pathlib import Path
from footlytics.data.soccertrack_fullmatch import (load_full_match, load_events,
                                                   load_team_sheet)
from footlytics.analytics import tactics as T
from footlytics.analytics.kinematics import add_kinematics, distance_summary
from footlytics.state.schema import MatchState, Team

D = "data/raw/soccertrack/117092"
OUT = Path("data/out/ST2-117092-FULL"); OUT.mkdir(parents=True, exist_ok=True)

print("loading full match ...")
state, poss = load_full_match(f"{D}/117092_tracker_box_data.xml",
                              f"{D}/117092_player_nodes.csv",
                              match_id="ST2-117092-FULL", verbose=False)
events = load_events(f"{D}/117092_12_class_events.json", f"{D}/117092_player_nodes.csv")
sheet = load_team_sheet(f"{D}/117092_player_nodes.csv")
team_ids = list(dict.fromkeys(sheet["team_id"].dropna().tolist()))
player_team = {int(p): (Team.HOME.value if sheet["team_id"].get(p) == team_ids[0]
                        else Team.AWAY.value)
               for p in sheet.index if pd.notna(p)}
print(f"  {state.n_frames:,} frames, {state.duration_s/60:.0f} min, "
      f"{len(events):,} events")

print("normalising attacking direction ...")
dirs = T.infer_attacking_direction(state)
for k, v in sorted(dirs.items()):
    print(f"   period {k[0]} {k[1]:5s} attacks {'+x' if v > 0 else '-x'}")
state = T.normalise_direction(state)
gk = T.goalkeepers(state)
print(f"   goalkeepers: {gk}")

print("team block ...");     block = T.team_block(state)
print("formation ...");      form = T.formation_over_time(state)
print("ppda ...");           p_ppda = T.ppda(state, events, player_team)
print("press distance ...");  press = T.press_distance(state, poss)
print("transitions ...");    trans = T.transitions(state, poss)
print("physical ...")
state = MatchState(state.meta, add_kinematics(state.tracks, state.meta.fps))
phys = distance_summary(state.players, state.meta.fps)

for name, df in [("block", block), ("formation", form), ("ppda", p_ppda),
                 ("press_distance", press), ("transitions", trans),
                 ("physical", phys), ("events", events), ("possession", poss)]:
    df.to_csv(OUT / f"{name}.csv", index=False)
    df.to_parquet(OUT / f"{name}.parquet", index=False)
state.save(OUT)

print("\n" + "=" * 66)
print(f"{state.meta.home.name}  vs  {state.meta.away.name}")
print("=" * 66)
print("\nPPDA (lower = presses harder):")
print(p_ppda.to_string(index=False))
print("\nTeam block, by period (metres):")
b = block.groupby(["period", "team"]).agg(
    def_line=("def_line_x", "mean"), width=("width_m", "mean"),
    depth=("depth_m", "mean"), compact=("compactness_m", "mean")).round(1)
print(b.to_string())
print("\nFormation, most common per period:")
if len(form):
    print(form.groupby(["period", "team"]).formation.agg(
        lambda s: s.value_counts().index[0]).to_string())
print(f"\nTransitions: {len(trans)} turnovers")
if len(trans):
    print(trans.groupby(["period", "won_by"]).agg(
        n=("frame_idx", "size"), won_at_x=("won_at_x", "mean"),
        gain_5s=("gain_5s_m", "mean")).round(1).to_string())
print("\nPress distance (nearest defender to ball, metres):")
print(press.groupby("pressing_team").agg(
    nearest=("nearest_m", "median"), within5=("n_within_5m", "mean"),
    within10=("n_within_10m", "mean")).round(2).to_string())
print(f"\nPhysical, top 5 by distance:")
print(phys.head(5)[["track_id","team","jersey","minutes","distance_m",
                    "top_speed_ms","sprints"]].round(1).to_string(index=False))
print(f"\nwrote CSV + parquet to {OUT}")
