"""Pitch model and the landmarks we calibrate against.

All coordinates in metres, origin at the centre spot, +x towards the away goal.
Dimensions follow IFAB defaults (105 x 68); everything is derived from
`length`/`width` so a non-standard V.League pitch can be measured and plugged in
rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

# Fixed by the Laws of the Game regardless of overall pitch size.
GOAL_WIDTH = 7.32
PENALTY_AREA_DEPTH = 16.5
PENALTY_AREA_WIDTH = 40.32
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_WIDTH = 18.32
PENALTY_SPOT_DIST = 11.0
CENTRE_CIRCLE_R = 9.15


@dataclass(frozen=True)
class Pitch:
    length: float = 105.0
    width: float = 68.0

    @property
    def half_l(self) -> float:
        return self.length / 2

    @property
    def half_w(self) -> float:
        return self.width / 2

    def landmarks(self) -> dict[str, tuple[float, float]]:
        """Named points a human can unambiguously click on in a video frame.

        Naming: side is L (negative x, home goal) or R (positive x); T is the
        touchline at negative y, B at positive y. Keep these names stable --
        calibration files on disk refer to them.
        """
        L, W = self.half_l, self.half_w
        pa_x = L - PENALTY_AREA_DEPTH
        pa_y = PENALTY_AREA_WIDTH / 2
        ga_x = L - GOAL_AREA_DEPTH
        ga_y = GOAL_AREA_WIDTH / 2
        pen_x = L - PENALTY_SPOT_DIST
        # Where the penalty arc meets the penalty-area line.
        dx = PENALTY_SPOT_DIST - PENALTY_AREA_DEPTH          # 11 - 16.5 = -5.5
        arc_y = sqrt(max(CENTRE_CIRCLE_R**2 - dx**2, 0.0))   # ~7.31

        pts: dict[str, tuple[float, float]] = {
            "corner_LT": (-L, -W), "corner_LB": (-L, W),
            "corner_RT": (L, -W),  "corner_RB": (L, W),
            "halfway_T": (0.0, -W), "halfway_B": (0.0, W),
            "centre_spot": (0.0, 0.0),
            "centre_circle_T": (0.0, -CENTRE_CIRCLE_R),
            "centre_circle_B": (0.0, CENTRE_CIRCLE_R),
        }
        for side, sx in (("L", -1.0), ("R", 1.0)):
            pts |= {
                f"pen_area_{side}T_goalline": (sx * L, -pa_y),
                f"pen_area_{side}B_goalline": (sx * L, pa_y),
                f"pen_area_{side}T_front":    (sx * pa_x, -pa_y),
                f"pen_area_{side}B_front":    (sx * pa_x, pa_y),
                f"goal_area_{side}T_goalline": (sx * L, -ga_y),
                f"goal_area_{side}B_goalline": (sx * L, ga_y),
                f"goal_area_{side}T_front":    (sx * ga_x, -ga_y),
                f"goal_area_{side}B_front":    (sx * ga_x, ga_y),
                f"pen_spot_{side}":  (sx * pen_x, 0.0),
                f"pen_arc_{side}T":  (sx * pa_x, -arc_y),
                f"pen_arc_{side}B":  (sx * pa_x, arc_y),
                f"goalpost_{side}T": (sx * L, -GOAL_WIDTH / 2),
                f"goalpost_{side}B": (sx * L, GOAL_WIDTH / 2),
            }
        return pts

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return abs(x) <= self.half_l + margin and abs(y) <= self.half_w + margin


DEFAULT_PITCH = Pitch()
