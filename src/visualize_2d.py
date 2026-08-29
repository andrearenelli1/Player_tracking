import json
import cv2
import pandas as pd
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
CALIB_DIR = ROOT / "calibrated_parameters"

PL_POS_CSVS = {
    "out2": ROOT / "tracking_results/tracking_2d/trajectories/2d_positions0.csv",
    "out4": ROOT / "tracking_results/tracking_2d/trajectories/2d_positions1.csv",
    "out13": ROOT / "tracking_results/tracking_2d/trajectories/2d_positions2.csv",
}

BL_POS_CSVS = {
    "out2": ROOT / "tracking_results/tracking_2d/ball_trajectories/ball_tracking_wasb_out2.csv",
    "out4": ROOT / "tracking_results/tracking_2d/ball_trajectories/ball_tracking_wasb_out4.csv",
    "out13": ROOT / "tracking_results/tracking_2d/ball_trajectories/ball_tracking_wasb_out13.csv",
}

VIDEOS = {
    "out2": ROOT / "videos/out2.mp4",
    "out4": ROOT / "videos/out4.mp4",
    "out13": ROOT / "videos/out13.mp4",
}
CALIB_FILES = {
    "out2": CALIB_DIR / "cam_2.json",
    "out4": CALIB_DIR / "cam_4.json",
    "out13": CALIB_DIR / "cam_13.json",
}
BALL_CLASS_ID = 0
TRAJ_LENGTH = 10


def load_calibration(calib_path: Path):
    with open(calib_path) as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float64)
    dist = np.array(calib["dist"], dtype=np.float64).reshape(-1)
    return mtx, dist


def build_undistort_map(mtx: np.ndarray, dist: np.ndarray, width: int, height: int):
    """Correct backward map for cv2.remap (cv2.initUndistortRectifyMap), unlike the
    hand-built cv2.undistortPoints-based version used elsewhere in the course
    material -- see report.tex for why that direction is wrong. player_df is already
    rectified point-wise upstream (tracking_players_2d.py), so drawing it on top of
    this correctly-rectified frame lines back up; ball_df (WASB, still raw/distorted)
    is rectified point-wise below before drawing, same reasoning."""
    return cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (width, height), cv2.CV_32FC1)

def rectify_center_bb(u: float, v: float, w: float, h: float,
                       mtx: np.ndarray, dist: np.ndarray) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = u - w / 2, v - h / 2, u + w / 2, v + h / 2
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]],
                        dtype=np.float32).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(corners, mtx, dist, P=mtx).reshape(-1, 2)
    x_min, y_min = undistorted.min(axis=0)
    x_max, y_max = undistorted.max(axis=0)
    rw, rh = x_max - x_min, y_max - y_min
    return x_min + rw / 2, y_min + rh / 2, rw, rh

def compute_corners(u: float, v: float, w: float, h: float) -> dict[str, tuple[int, int]]:
    tl = (int(u - w / 2), int(v - h / 2))
    tr = (int(u + w / 2), int(v - h / 2))
    bl = (int(u - w / 2), int(v + h / 2))
    br = (int(u + w / 2), int(v + h / 2))
    corners = {
        "tl": tl,
        "tr": tr,
        "bl": bl,
        "br": br,
    }
    return corners

def draw_bb(frame: np.ndarray, frame_id: int, bb_df: pd.DataFrame, color: tuple[int, int, int], label: str) -> None:
    bb = bb_df[bb_df["frame"] == frame_id]
    line_thickness = 3

    for _, row in bb.iterrows():
        corners = compute_corners(row["u"], row["v"], row["w"], row["h"])
        cv2.line(frame, corners["tl"], corners["bl"], color, line_thickness)
        cv2.line(frame, corners["bl"], corners["br"], color, line_thickness)
        cv2.line(frame, corners["br"], corners["tr"], color, line_thickness)
        cv2.line(frame, corners["tr"], corners["tl"], color, line_thickness)
        cv2.putText(frame, f"{label} ID={row['object_id']}", corners["tr"], cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        draw_traj(bb_df, frame_id, row, color, frame)

def draw_traj(bb_df: pd.DataFrame, frame_id: int, row: pd.Series, color: tuple[int, int, int], frame: np.ndarray) -> None:
    filtered_df = bb_df[(bb_df["frame"] <= frame_id) &
                        (bb_df["frame"] > max(frame_id - TRAJ_LENGTH, 0)) &
                        (bb_df["object_id"] == row["object_id"])]
    filtered_df = filtered_df.sort_values("frame")
    points = filtered_df[["u", "v"]].to_numpy().astype(np.int32)
    points = points.reshape((-1, 1, 2))
    thickness = 4
    cv2.polylines(frame, points, True, color, thickness)

def disp_video(player_csvs: dict[str, Path], ball_csvs: dict[str, Path]) -> None:
    for (cam_id, vid), (_, player_csv), (_, ball_csv) in zip(VIDEOS.items(), player_csvs.items(), ball_csvs.items()):
        cap = cv2.VideoCapture(vid)
        player_df = pd.read_csv(player_csv)
        ball_df = pd.read_csv(ball_csv)
        if not cap.isOpened():
            print("Error: Could not open the video file.")
            exit()

        calib_path = CALIB_FILES[cam_id]
        mtx, dist = load_calibration(calib_path)
        map_x, map_y = build_undistort_map(mtx, dist, 3840, 2160)

        # ball_df (WASB) is still in raw/distorted pixels, unlike player_df
        # (already rectified point-wise upstream) -- rectify it here to match.
        ball_df[["u", "v", "w", "h"]] = ball_df.apply(
            lambda row: pd.Series(rectify_center_bb(row["u"], row["v"], row["w"], row["h"], mtx, dist)),
            axis=1)

        frame_id = -1
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            frame_id += 1
            rectified = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
            draw_bb(rectified, frame_id, player_df, (0, 255, 0), "PLAYER")
            draw_bb(rectified, frame_id, ball_df, (0, 0, 255), "BALL")
            rectified = cv2.resize(rectified, (960, 540))
            cv2.imshow(cam_id, rectified)

            if cv2.waitKey(25) &0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main() -> None:
    disp_video(PL_POS_CSVS, BL_POS_CSVS)

if __name__ == "__main__":
    main()