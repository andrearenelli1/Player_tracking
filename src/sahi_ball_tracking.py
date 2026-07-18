import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

ROOT = Path(__file__).parent.parent

MODEL_PATH  = ROOT / "runs/detect/train-6/weights/best.pt"
DISPLAY     = False
DISPLAY_W   = 3840
DISPLAY_H   = 2176
SLICE_DIM = 1500
CONF        = 0.6
CLASSES     = [0]

CAMERAS = [
    ("cam_0", ROOT / "videos/out2.mp4",  ROOT / "tracking_results/tracking_2d/ball_trajectories/2d_positions0.csv"),
    ("cam_1", ROOT / "videos/out4.mp4",  ROOT / "tracking_results/tracking_2d/ball_trajectories/2d_positions1.csv"),
    ("cam_2", ROOT / "videos/out13.mp4", ROOT / "tracking_results/tracking_2d/ball_trajectories/2d_positions2.csv"),
]

CSV_COLUMNS = ["frame", "cam_id", "class_id", "object_id", "u", "v", "w", "h"]


def track_video(model: AutoDetectionModel, cam_id: str, video_path: Path, csv_path: Path) -> None:
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rows: list[list] = []

    for frame_idx in range(total):
        success, frame = cap.read()
        if not success:
            break

        print(f"  {cam_id}  frame {frame_idx + 1} / {total}")

        result = get_sliced_prediction(
            frame,
            model,
            slice_height=SLICE_DIM,
            slice_width=SLICE_DIM,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.1,
        )

        for pred in result.object_prediction_list:
            cls = pred.category.id
            if cls not in CLASSES:
                continue
            x1, y1, x2, y2 = pred.bbox.to_xyxy()
            u = (x1 + x2) / 2
            v = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            rows.append([frame_idx, cam_id, cls, -1, u, v, w, h])

            if DISPLAY:
                color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        if DISPLAY:
            disp = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            cv2.imshow(cam_id, disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(csv_path, index=False)


def main() -> None:
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(MODEL_PATH),
        confidence_threshold=CONF,
        device="cuda",
    )

    for cam_id, video_path, csv_path in CAMERAS:
        track_video(model, cam_id, video_path, csv_path)


if __name__ == "__main__":
    main()