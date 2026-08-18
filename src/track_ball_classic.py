import cv2 
import numpy as np
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).parent.parent

# (cam_id, video_path, csv_path, min_area, max_area)
# min/max area are in pixels^2 and are tuned per camera: cam_2 frames the court
# from much further away, so the ball projects onto a much smaller blob there.
CAMERAS = [
    ("cam_0", ROOT / "videos/out2.mp4",  ROOT / "tracking_results/tracking_2d/ball_trajectories/ball_tracking_classic_out2.csv", 100, 8000),
    ("cam_1", ROOT / "videos/out4.mp4",  ROOT / "tracking_results/tracking_2d/ball_trajectories/ball_tracking_classic_out4.csv", 100, 8000),
    ("cam_2", ROOT / "videos/out13.mp4", ROOT / "tracking_results/tracking_2d/ball_trajectories/ball_tracking_classic_out13.csv", 20, 1500),
]

CSV_COLUMNS = ["frame", "cam_id", "class_id", "object_id", "u", "v", "w", "h"]
DISPLAY = True
IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 600
SHADOW_THRESH = 200
VAR_THRESHOLD = 80
OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
MIN_CIRCULARITY = 0.2
MIN_ASPECT_RATIO = 0.2
WHITE_SAT_MAX = 60
WHITE_VAL_MIN = 180
RED_SAT_MIN = 100
MIN_WHITE_FRACTION = 0.05
MIN_RED_FRACTION = 0.1
MAX_CONTOURS = 100


def is_shape_like_ball(cnt) -> bool:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * area / (perimeter ** 2)
    _, _, w, h = cv2.boundingRect(cnt)
    aspect_ratio = min(w, h) / max(w, h)
    return circularity > MIN_CIRCULARITY and aspect_ratio > MIN_ASPECT_RATIO


def has_ball_color_pattern(frame, cnt) -> bool:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask == 255]
    if pixels.size == 0:
        return False
    hue, sat, val = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    white_frac = np.mean((sat < WHITE_SAT_MAX) & (val > WHITE_VAL_MIN))
    red_frac = np.mean(((hue < 10) | (hue > 170)) & (sat > RED_SAT_MIN))
    print(f"    candidate area={pixels.shape[0]:4d}  white_frac={white_frac:.3f}  "
          f"red_frac={red_frac:.3f}  mean_hsv=({hue.mean():.0f},{sat.mean():.0f},{val.mean():.0f})")
    return red_frac > MIN_RED_FRACTION


def track_video(cam_id: str, video_path: Path, csv_path: Path, min_area: float, max_area: float) -> None:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True,
                                                       varThreshold=VAR_THRESHOLD)
    rows: list[list] = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for frame_idx in range(0, total):

        success, frame = cap.read()
        if not success:
            break

        print(f"  {cam_id}  frame {frame_idx + 1} / {total}")

        fg_mask = bg_subtractor.apply(frame, learningRate=-1)
        _, fg_mask_bin = cv2.threshold(fg_mask, SHADOW_THRESH, 255, cv2.THRESH_BINARY)
        fg_mask_closed = cv2.morphologyEx(fg_mask_bin, cv2.MORPH_CLOSE, CLOSE_KERNEL)
        fg_mask_opened = cv2.morphologyEx(fg_mask_closed, cv2.MORPH_OPEN, OPEN_KERNEL)

        contours, _ = cv2.findContours(fg_mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > MAX_CONTOURS:
            print(f"    skipping frame {frame_idx + 1}: {len(contours)} contours (> {MAX_CONTOURS})")
            continue
        candidates = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
        candidates = [cnt for cnt in candidates if is_shape_like_ball(cnt)]
        candidates = [cnt for cnt in candidates if has_ball_color_pattern(frame, cnt)]

        for cnt in candidates:
            x, y, w, h = cv2.boundingRect(cnt)
            u = x + w / 2
            v = y + h / 2
            rows.append([frame_idx, cam_id, 0, -1, u, v, w, h])

        if DISPLAY:
            vis = frame.copy()
            cv2.drawContours(vis, candidates, -1, (0, 255, 0), 2)
            mask_bgr = cv2.cvtColor(fg_mask_opened, cv2.COLOR_GRAY2BGR)
            pair = np.hstack([mask_bgr, vis])
            cv2.imshow(cam_id, cv2.resize(pair, (IMAGE_WIDTH, IMAGE_HEIGHT)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(csv_path, index=False)


def main() -> None:
    for cam_id, video_path, csv_path, min_area, max_area in CAMERAS:
        track_video(cam_id, video_path, csv_path, min_area, max_area)


if __name__ == '__main__':
    main()