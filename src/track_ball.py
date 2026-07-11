import cv2 
import numpy as np
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).parent.parent

CAMERAS = [
    ("cam_0", ROOT / "videos/out2.mp4",  ROOT / "tracking_results/tracking_2d/positions/2d_positions0.csv"),
    ("cam_1", ROOT / "videos/out4.mp4",  ROOT / "tracking_results/tracking_2d/positions/2d_positions1.csv"),
    ("cam_2", ROOT / "videos/out13.mp4", ROOT / "tracking_results/tracking_2d/positions/2d_positions2.csv"),
]

CSV_COLUMNS = ["frame", "cam_id", "class_id", "object_id", "u", "v", "w", "h"]
DISPLAY = True
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 600
LOW_THR = (0, 0, 0)
HIGH_THR = (15, 200, 200)


def track_video(cam_id: str, video_path: Path, csv_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_idx in range(total):
        success, frame = cap.read()
        if not success:
            break

        print(f"  {cam_id}  frame {frame_idx + 1} / {total}")

        cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(frame, LOW_THR, HIGH_THR)

        if DISPLAY:
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            pair = np.hstack([frame, mask_bgr])
            cv2.imshow(cam_id, cv2.resize(pair, (IMAGE_WIDTH, IMAGE_HEIGHT)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break   

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    for i, (cam_id, video_path, csv_path) in enumerate(CAMERAS):
        track_video(cam_id, video_path, csv_path)


if __name__ == '__main__':
    main()