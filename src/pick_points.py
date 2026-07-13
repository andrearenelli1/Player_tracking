"""Interactive tool to pick 2D<->3D point correspondences on a calibration frame.

Usage:
    python src/pick_points.py <camera_name>

e.g. python src/pick_points.py out2

Workflow:
  - The calibration frame for the camera (see CAL_FRAMES in calibration.py) is
    shown in a matplotlib window.
  - Use the toolbar magnifying glass to zoom into the area of interest, then
    toggle zoom off and click on the requested landmark.
  - Press Enter (with no click) or just close and re-run to skip a point that
    is not visible in this view.
  - Points already picked are saved incrementally; re-running resumes and
    lets you overwrite previously picked points.

Output:
  annotations/calibration_points/<camera_name>.json
  {"point_name": [x_px, y_px], ...}
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from calibration import CAL_FRAMES, VIDEOS
from court_model import WORLD_POINTS

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "annotations/calibration_points"


def load_frame(camera_name: str):
    video_names = list(VIDEOS.keys())
    idx = video_names.index(camera_name)
    frame_idx = CAL_FRAMES[idx]

    cap = cv2.VideoCapture(str(VIDEOS[camera_name]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_idx} from {VIDEOS[camera_name]}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def pick_points(image, existing: dict[str, list[float]]) -> dict[str, list[float]]:
    correspondences = dict(existing)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(image)
    for name, (x, y) in correspondences.items():
        ax.plot(x, y, "r+", markersize=12, markeredgewidth=2)
        ax.annotate(name, (x, y), color="red", fontsize=8)

    for name in WORLD_POINTS:
        status = " (already picked, click to overwrite or press Enter to keep)" if name in correspondences else ""
        ax.set_title(f"Zoom with toolbar, toggle zoom off, then click on: {name}{status}", fontsize=10)
        fig.canvas.draw()

        pts = plt.ginput(n=1, timeout=0)
        if not pts:
            continue

        x, y = pts[0]
        correspondences[name] = [x, y]
        ax.plot(x, y, "r+", markersize=12, markeredgewidth=2)
        ax.annotate(name, (x, y), color="red", fontsize=8)

    plt.close(fig)
    return correspondences


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in VIDEOS:
        print(f"usage: python src/pick_points.py <camera_name>  (one of {list(VIDEOS)})")
        sys.exit(1)

    camera_name = sys.argv[1]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{camera_name}.json"

    existing = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        print(f"resuming {len(existing)} previously picked points from {out_path}")

    image = load_frame(camera_name)
    correspondences = pick_points(image, existing)

    out_path.write_text(json.dumps(correspondences, indent=2))
    print(f"saved {len(correspondences)} points to {out_path}")


if __name__ == "__main__":
    main()
