"""FIBA basketball court 3D reference model.

World coordinate system:
  - origin at court center (center circle center)
  - X axis along the court length, +X toward the "B" end basket
  - Y axis along the court width
  - Z axis vertical, floor at Z=0
  - units: meters

Source: FIBA Official Basketball Rules, Court and Equipment.
"""

COURT_LENGTH = 28.0
COURT_WIDTH = 15.0
CENTER_CIRCLE_RADIUS = 1.80
FREE_THROW_LINE_DIST = 5.80  # from baseline to free-throw line
LANE_WIDTH = 4.90  # width of the free-throw lane (key)
THREE_POINT_RADIUS = 6.75
THREE_POINT_SIDE_INSET = 0.90  # distance of the straight 3pt segment from the sideline

HOOP_HEIGHT = 3.05
HOOP_DIST_FROM_BASELINE = 1.575

BACKBOARD_WIDTH = 1.80
BACKBOARD_HEIGHT = 1.05
BACKBOARD_BOTTOM_HEIGHT = 2.90
BACKBOARD_DIST_FROM_BASELINE = 1.20

HALF_LENGTH = COURT_LENGTH / 2
HALF_WIDTH = COURT_WIDTH / 2
HALF_LANE = LANE_WIDTH / 2
HALF_BACKBOARD = BACKBOARD_WIDTH / 2
THREE_POINT_SIDE_Y = HALF_WIDTH - THREE_POINT_SIDE_INSET


def _end_points(sign: float) -> dict[str, tuple[float, float, float]]:
    """Named landmarks for one basket end. sign=+1 -> +X end, sign=-1 -> -X end."""
    baseline_x = sign * HALF_LENGTH
    ft_line_x = baseline_x - sign * FREE_THROW_LINE_DIST
    hoop_x = baseline_x - sign * HOOP_DIST_FROM_BASELINE
    backboard_x = baseline_x - sign * BACKBOARD_DIST_FROM_BASELINE
    backboard_top = BACKBOARD_BOTTOM_HEIGHT + BACKBOARD_HEIGHT

    return {
        "corner_pos": (baseline_x, HALF_WIDTH, 0.0),
        "corner_neg": (baseline_x, -HALF_WIDTH, 0.0),
        "lane_corner_baseline_pos": (baseline_x, HALF_LANE, 0.0),
        "lane_corner_baseline_neg": (baseline_x, -HALF_LANE, 0.0),
        "lane_corner_ft_pos": (ft_line_x, HALF_LANE, 0.0),
        "lane_corner_ft_neg": (ft_line_x, -HALF_LANE, 0.0),
        "three_point_baseline_pos": (baseline_x, THREE_POINT_SIDE_Y, 0.0),
        "three_point_baseline_neg": (baseline_x, -THREE_POINT_SIDE_Y, 0.0),
        "hoop_center": (hoop_x, 0.0, HOOP_HEIGHT),
        "backboard_bottom_pos": (backboard_x, HALF_BACKBOARD, BACKBOARD_BOTTOM_HEIGHT),
        "backboard_bottom_neg": (backboard_x, -HALF_BACKBOARD, BACKBOARD_BOTTOM_HEIGHT),
        "backboard_top_pos": (backboard_x, HALF_BACKBOARD, backboard_top),
        "backboard_top_neg": (backboard_x, -HALF_BACKBOARD, backboard_top),
    }


WORLD_POINTS: dict[str, tuple[float, float, float]] = {}
for _sign, _end in [(+1.0, "B"), (-1.0, "A")]:
    for _name, _coords in _end_points(_sign).items():
        WORLD_POINTS[f"{_end}_{_name}"] = _coords

WORLD_POINTS.update(
    {
        "center_court": (0.0, 0.0, 0.0),
        "center_circle_top": (0.0, CENTER_CIRCLE_RADIUS, 0.0),
        "center_circle_bottom": (0.0, -CENTER_CIRCLE_RADIUS, 0.0),
        "halfway_sideline_pos": (0.0, HALF_WIDTH, 0.0),
        "halfway_sideline_neg": (0.0, -HALF_WIDTH, 0.0),
    }
)
