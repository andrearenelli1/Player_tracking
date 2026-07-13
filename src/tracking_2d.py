from ultralytics import YOLO
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent

MODEL_PATH = ROOT / "runs/detect/train-2/weights/best.pt"
DISPLAY    = False
INFER_W    = 1920
INFER_H    = 1088
TRAIL_LEN  = 30
CONF       = 0.1
CLASSES    = [0]

CAMERAS = [
    ("cam_0", ROOT / "videos/out2.mp4",  ROOT / "tracking_results/tracking_2d/trajectories/2d_positions0.csv"),
    ("cam_1", ROOT / "videos/out4.mp4",  ROOT / "tracking_results/tracking_2d/trajectories/2d_positions1.csv"),
    ("cam_2", ROOT / "videos/out13.mp4", ROOT / "tracking_results/tracking_2d/trajectories/2d_positions2.csv"),
]

CSV_COLUMNS = ["frame", "cam_id", "class_id", "object_id", "u", "v", "w", "h"]


def track_video(model: YOLO, cam_id: str, video_path: Path, csv_path: Path) -> None:
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

        small  = cv2.resize(frame, (INFER_W, INFER_H))
        result = model.track(small, persist=True, verbose=False,
                             imgsz=(INFER_H, INFER_W), conf=CONF, classes=CLASSES)[0]

        if result.boxes and result.boxes.is_track:
            boxes     = result.boxes.xywh.cpu().int()
            track_ids = result.boxes.id.int().cpu().tolist()
            classes   = result.boxes.cls.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, classes):
                x, y, w, h = box
                sw, sh = w.item() * scale_x, h.item() * scale_y
                rows.append([frame_idx, cam_id, cls, track_id,
                              x.item() * scale_x, y.item() * scale_y,
                              sw, sh])

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
    for i, (cam_id, video_path, csv_path) in enumerate(CAMERAS):
        track_video(model, cam_id, video_path, csv_path)


if __name__ == '__main__':
    main()