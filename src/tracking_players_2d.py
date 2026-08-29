from ultralytics import YOLO
from collections import defaultdict
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
CALIB_DIR = ROOT / "calibrated_parameters"

#MODEL_PATH = ROOT / "runs/detect/train/weights/best.pt"
MODEL_PATH = "yolo26n.pt"
DISPLAY    = False
INFER_W    = 1920
INFER_H    = 1088
TRAIL_LEN  = 30
CONF       = 0.1
CLASSES    = [0, 32]

CAMERAS = [
    ("cam_0", ROOT / "videos/out2.mp4",  ROOT / "tracking_results/tracking_2d/trajectories/2d_positions0.csv", CALIB_DIR / "cam_2.json"),
    ("cam_1", ROOT / "videos/out4.mp4",  ROOT / "tracking_results/tracking_2d/trajectories/2d_positions1.csv", CALIB_DIR / "cam_4.json"),
    ("cam_2", ROOT / "videos/out13.mp4", ROOT / "tracking_results/tracking_2d/trajectories/2d_positions2.csv", CALIB_DIR / "cam_13.json"),
]

CSV_COLUMNS = ["frame", "cam_id", "class_id", "object_id", "u", "v", "w", "h"]


def load_calibration(calib_path: Path):
    """Reads mtx (K) and dist from a calibrated_parameters/cam_*.json file."""
    with open(calib_path) as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float64)
    dist = np.array(calib["dist"], dtype=np.float64).reshape(-1)
    return mtx, dist


def rectify_center_box(x: float, y: float, w: float, h: float,
                        mtx: np.ndarray, dist: np.ndarray) -> tuple[float, float, float, float]:
    """(x, y, w, h) center-format box in raw/distorted full-resolution pixels ->
    same-format box after undistortion.

    Undistorts the 4 corners (not just the center: undistortion is non-linear,
    so the box shape itself changes) and returns the center-format AABB of the
    undistorted corners. Mirrors evaluate_2d.py's rectify_bb (which does the
    same for the GT boxes, in top-left format).
    """
    x0, y0, x1, y1 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]],
                        dtype=np.float32).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(corners, mtx, dist, P=mtx).reshape(-1, 2)
    x_min, y_min = undistorted.min(axis=0)
    x_max, y_max = undistorted.max(axis=0)
    rw, rh = x_max - x_min, y_max - y_min
    return x_min + rw / 2, y_min + rh / 2, rw, rh


def track_video(model: YOLO, cam_id: str, video_path: Path, csv_path: Path,
                 mtx: np.ndarray, dist: np.ndarray) -> None:
    cap = cv2.VideoCapture(str(video_path))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scale_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  / INFER_W
    scale_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) / INFER_H

    rows  = []
    trail = defaultdict(list)

    for frame_idx in range(total):
        success, frame = cap.read()
        if not success:
            break

        print(f"  {cam_id}  frame {frame_idx + 1} / {total}")

        # Detection runs on the raw (still distorted) frame, just resized for
        # inference; undistortion is applied to the resulting box below,
        # point-wise, instead of pre-warping the whole frame with a manually
        # built remap (see report.tex / README for why: that construction -
        # also used in the course-provided rectified_videos.py - builds the
        # map with cv2.undistortPoints in the wrong direction for cv2.remap,
        # producing up to ~300px of spurious displacement near the frame
        # edges on this calibration).
        small  = cv2.resize(frame, (INFER_W, INFER_H))
        result = model.track(small, persist=True, verbose=False,
                             imgsz=(INFER_H, INFER_W), conf=CONF, classes=CLASSES)[0]

        if result.boxes and result.boxes.is_track:
            boxes     = result.boxes.xywh.cpu().tolist()
            track_ids = result.boxes.id.int().cpu().tolist()
            classes   = result.boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, classes):
                x, y, w, h = box
                # back-project from inference resolution to raw full-res (still distorted) pixels
                rx, ry = x * scale_x, y * scale_y
                rw, rh = w * scale_x, h * scale_y
                ux, uy, uw, uh = rectify_center_box(rx, ry, rw, rh, mtx, dist)
                rows.append([frame_idx, cam_id, cls, track_id, ux, uy, uw, uh])

                if DISPLAY:
                    trail[track_id].append((float(x), float(y)))
                    if len(trail[track_id]) > TRAIL_LEN:
                        trail[track_id].pop(0)

        if DISPLAY:
            annotated = result.plot()
            for track_id, pts in trail.items():
                if len(pts) > 1:
                    points = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(annotated, [points], False, (0, 230, 0), 2)
            cv2.imshow(cam_id, cv2.resize(annotated, (1280, 800)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if DISPLAY:
        cv2.destroyAllWindows()
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(csv_path, index=False)


def main() -> None:
    model = YOLO(MODEL_PATH).to('cuda')
    for cam_id, video_path, csv_path, calib_path in CAMERAS:
        mtx, dist = load_calibration(calib_path)
        track_video(model, cam_id, video_path, csv_path, mtx, dist)


if __name__ == '__main__':
    main()