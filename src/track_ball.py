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
HSV_LOW = (0, 60, 40)
HSV_HIGH = (25, 255, 255)
FRAME_DIFF_LOW = (25, 25, 25)
FRAME_DIFF_UP = (255, 255, 255)
MIN_AREA = 5
MAX_AREA = 300


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
        hsv_mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
        diff = cv2.absdiff(new_frame, old_frame)
        diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)
        diff_blur_th = cv2.inRange(diff_blur, FRAME_DIFF_LOW, FRAME_DIFF_UP)
        mask_int = cv2.bitwise_and(diff_blur_th, hsv_mask)

        contours, _ = cv2.findContours(mask_int, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = [cnt for cnt in contours if MIN_AREA < cv2.contourArea(cnt) < MAX_AREA]

        if DISPLAY:
            vis = new_frame.copy()
            cv2.drawContours(vis, candidates, -1, (0, 255, 0), 2)
            mask_bgr = cv2.cvtColor(hsv_mask, cv2.COLOR_GRAY2BGR)
            diff_bgr = cv2.cvtColor(diff_blur_th, cv2.COLOR_GRAY2BGR)
            mask_int_bgr = cv2.cvtColor(mask_int, cv2.COLOR_GRAY2BGR)
            pair = np.hstack([diff_bgr, mask_bgr, mask_int_bgr, vis])
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