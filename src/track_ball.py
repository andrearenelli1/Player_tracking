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
IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 600
HSV_LOW = (0, 120, 50)
HSV_HIGH = (12, 255, 255)
FRAME_DIFF_LOW = (210, 210, 210)
FRAME_DIFF_UP = (245, 245, 245)


def track_video(cam_id: str, video_path: Path, csv_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    new_frame = None

    for frame_idx in range(total):   
        if new_frame is None:
            success, old_frame = cap.read()
        else:
            old_frame = new_frame
        success, new_frame = cap.read()
        if not success:
            break

        print(f"  {cam_id}  frame {frame_idx + 1} / {total}")

        hsv = cv2.cvtColor(new_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
        diff = cv2.inRange((new_frame - old_frame), FRAME_DIFF_LOW, FRAME_DIFF_UP)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)

        if DISPLAY:
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            diff_bgr = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
            pair = np.hstack([diff_bgr, mask_bgr, new_frame])
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