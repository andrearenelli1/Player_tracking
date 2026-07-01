import cv2
import pandas as pd
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
POS_CSVS = {
    "out2": ROOT / "tracking_results/tracking_2d/positions/2d_positions0.csv",
    "out4": ROOT / "tracking_results/tracking_2d/positions/2d_positions1.csv",
    "out13": ROOT / "tracking_results/tracking_2d/positions/2d_positions2.csv",
}
VIDEOS = {
    "out2": ROOT / "videos/out2.mp4",
    "out4": ROOT / "videos/out4.mp4",
    "out13": ROOT / "videos/out13.mp4",
}
PERSON_CLASS_ID = 0
BALL_CLASS_ID = 32
TRAJ_LENGTH = 10

def compute_corners(u, v, w, h):
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

def draw_bb(frame, frame_id, bb_df):
    bb = bb_df[bb_df["frame"] == frame_id]
    line_thickness = 3
    red = (0, 0, 255)
    green = (0, 255, 0)

    for _, row in bb.iterrows():
        if row["class_id"] == PERSON_CLASS_ID:
            line_color = green
        elif row["class_id"] == BALL_CLASS_ID:
            line_color = red
        corners = compute_corners(row["u"], row["v"], row["w"], row["h"])
        cv2.line(frame, corners["tl"], corners["bl"], line_color, line_thickness)
        cv2.line(frame, corners["bl"], corners["br"], line_color, line_thickness)
        cv2.line(frame, corners["br"], corners["tr"], line_color, line_thickness)
        cv2.line(frame, corners["tr"], corners["tl"], line_color, line_thickness)
        cv2.putText(frame, f"ID={row["object_id"]}", corners["tr"], cv2.FONT_HERSHEY_SIMPLEX, 1, line_color, 2)
        draw_traj(bb_df, frame_id, row, line_color, frame)

def draw_traj(bb_df, frame_id, row, color, frame):
    filtered_df = bb_df[(bb_df["frame"] <= frame_id) &
                        (bb_df["frame"] > max(frame_id - TRAJ_LENGTH, 0)) &
                        (bb_df["object_id"] == row["object_id"])]
    filtered_df = filtered_df.sort_values("frame")
    points = filtered_df[["u", "v"]].to_numpy().astype(np.int32)
    points = points.reshape((-1, 1, 2))
    thickness = 4
    cv2.polylines(frame, points, True, color, thickness)

def disp_video():
    for (cam_id, vid), (csv_id, csv_file) in zip(VIDEOS.items(), POS_CSVS.items()):
        cap = cv2.VideoCapture(vid)
        bb_df = pd.read_csv(csv_file)
        if not cap.isOpened():
            print("Error: Could not open the video file.")
            exit()

        frame_id = -1
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            frame_id += 1
            draw_bb(frame, frame_id, bb_df)
            frame = cv2.resize(frame, (960, 540))
            cv2.imshow(cam_id, frame)

            if cv2.waitKey(25) &0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        cv2.destroyAllWindows()

def main():
    disp_video()

if __name__ == "__main__":
    main()