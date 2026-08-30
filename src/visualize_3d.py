import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

ROOT = Path(__file__).parent.parent

POS_3D_CSV = ROOT / "tracking_results/tracking_3d/3d_positions.csv"
GLOBAL_ID_MAP_CSV = ROOT / "tracking_results/tracking_3d/global_id_map.csv"
OUT_PNG = ROOT / "tracking_results/tracking_3d/3d_positions.png"
OUT_GIF = ROOT / "tracking_results/tracking_3d/3d_positions.gif"

# Same order/names used in tracking_2d.py and tracking_3d.py: the "frame" field
# of the 3d csv corresponds 1:1 to the frame index of these videos (already synchronized).
VIDEO_PATHS = {
    "out2": ROOT / "videos/out2.mp4",
    "out4": ROOT / "videos/out4.mp4",
    "out13": ROOT / "videos/out13.mp4",
}
TRAJ_2D_CSV = {
    "out2": ROOT / "tracking_results/tracking_2d/trajectories/2d_positions0.csv",
    "out4": ROOT / "tracking_results/tracking_2d/trajectories/2d_positions1.csv",
    "out13": ROOT / "tracking_results/tracking_2d/trajectories/2d_positions2.csv",
}
VIDEO_FRAME_WIDTH = 480  # resize video frames for a lighter animation

FPS = 25            # native frame rate of the tracked video
TRAIL_LEN = 15       # how many trail frames to show behind the current position

# Basketball court 28x15m, origin at center, as per CAL_POINTS in calibration.py.
COURT_HALF_LENGTH = 14.0
COURT_HALF_WIDTH = 8.5
KEY_LENGTH = 8.2   # distance from the baseline center to the free-throw line
KEY_HALF_WIDTH = 2.5


def draw_court(ax):
    outline = [
        (-COURT_HALF_LENGTH, -COURT_HALF_WIDTH), (COURT_HALF_LENGTH, -COURT_HALF_WIDTH),
        (COURT_HALF_LENGTH, COURT_HALF_WIDTH), (-COURT_HALF_LENGTH, COURT_HALF_WIDTH),
        (-COURT_HALF_LENGTH, -COURT_HALF_WIDTH),
    ]
    lines = [
        outline,
        [(0, -COURT_HALF_WIDTH), (0, COURT_HALF_WIDTH)],  # half-court line
        [(-COURT_HALF_LENGTH, -KEY_HALF_WIDTH), (-KEY_LENGTH, -KEY_HALF_WIDTH),
         (-KEY_LENGTH, KEY_HALF_WIDTH), (-COURT_HALF_LENGTH, KEY_HALF_WIDTH)],  # left key
        [(COURT_HALF_LENGTH, -KEY_HALF_WIDTH), (KEY_LENGTH, -KEY_HALF_WIDTH),
         (KEY_LENGTH, KEY_HALF_WIDTH), (COURT_HALF_LENGTH, KEY_HALF_WIDTH)],  # right key
    ]
    for line in lines:
        xs, ys = zip(*line)
        ax.plot(xs, ys, [0] * len(xs), color="gray", linewidth=1, zorder=0)


def plot_trajectories(df, ax):
    cmap = plt.get_cmap("tab20")
    object_ids = sorted(df["object_id"].unique())
    for i, object_id in enumerate(object_ids):
        traj = df[df["object_id"] == object_id].sort_values("frame")
        color = cmap(i % cmap.N)
        ax.plot(traj["X"], traj["Y"], traj["Z"], color=color, linewidth=1.5,
                 label=f"id {object_id}")
        ax.scatter(traj["X"].iloc[-1], traj["Y"].iloc[-1], traj["Z"].iloc[-1],
                    color=color, marker="o", s=25)


def load_bboxes():
    """Reads the 2d bounding boxes per camera, with the object_id already translated to global_id
    (the same id/color used in the 3d plot), so colors match between video and 3d.
    Detections never associated cross-camera (without a global_id) are discarded."""
    id_map = pd.read_csv(GLOBAL_ID_MAP_CSV).set_index(["cam_name", "object_id"])["global_id"]
    bboxes = {}
    for cam_name, csv_path in TRAJ_2D_CSV.items():
        df = pd.read_csv(csv_path)
        df["global_id"] = [id_map.get((cam_name, oid)) for oid in df["object_id"]]
        df = df.dropna(subset=["global_id"]).copy()
        df["global_id"] = df["global_id"].astype(int)
        bboxes[cam_name] = df.set_index("frame")[["global_id", "u", "v", "w", "h"]]
    return bboxes


class VideoFrameSource:
    """Reads a video forward, frame by frame, returning the last read frame
    for a given index (frames in the 3d csv can have gaps)."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.scale = VIDEO_FRAME_WIDTH / w
        self.size = (VIDEO_FRAME_WIDTH, int(h * self.scale))
        self.next_idx = 0
        self.last_frame = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)

    def get(self, frame_idx):
        while self.next_idx <= frame_idx:
            success, frame = self.cap.read()
            if not success:
                break
            frame = cv2.resize(frame, self.size)
            self.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.next_idx += 1
        return self.last_frame

    def release(self):
        self.cap.release()


def animate_trajectories(df, ax, video_sources=None, bboxes=None):
    """Animates the positions frame by frame, with a short trail behind each object.

    If video_sources is passed ({cam_name: (VideoFrameSource, Axes)}), also shows
    the corresponding frame of each camera, synchronized to the animation frame.
    If bboxes is passed ({cam_name: DataFrame indexed by frame, with global_id/u/v/w/h}),
    draws the bounding boxes over the video frames, colored like their respective id in the 3d plot.
    """
    cmap = plt.get_cmap("tab20")
    object_ids = sorted(df["object_id"].unique())
    colors = {oid: cmap(i % cmap.N) for i, oid in enumerate(object_ids)}
    traj_by_id = {oid: df[df["object_id"] == oid].sort_values("frame") for oid in object_ids}

    trails = {oid: ax.plot([], [], [], color=colors[oid], linewidth=1.5)[0] for oid in object_ids}
    points = {oid: ax.plot([], [], [], color=colors[oid], marker="o", markersize=6,
                            linestyle="none", label=f"id {oid}")[0]
              for oid in object_ids}

    frames = sorted(df["frame"].unique())

    video_images = {}
    video_boxes = {}
    if video_sources:
        for cam_name, (source, vax) in video_sources.items():
            video_images[cam_name] = vax.imshow(source.get(frames[0]))
            video_boxes[cam_name] = []

    def update(frame):
        for oid, traj in traj_by_id.items():
            window = traj[(traj["frame"] <= frame) & (traj["frame"] > frame - TRAIL_LEN)]
            trails[oid].set_data_3d(window["X"], window["Y"], window["Z"])
            cur = window[window["frame"] == frame]
            points[oid].set_data_3d(cur["X"], cur["Y"], cur["Z"])
        if video_sources:
            for cam_name, (source, vax) in video_sources.items():
                video_images[cam_name].set_data(source.get(frame))

                for box in video_boxes[cam_name]:
                    box.remove()
                video_boxes[cam_name] = []

                cam_boxes = bboxes.get(cam_name) if bboxes else None
                if cam_boxes is not None and frame in cam_boxes.index:
                    rows = cam_boxes.loc[[frame]]
                    for _, row in rows.iterrows():
                        color = colors.get(row["global_id"], "white")
                        w_box, h_box = row["w"] * source.scale, row["h"] * source.scale
                        x0 = row["u"] * source.scale - w_box / 2
                        y0 = row["v"] * source.scale - h_box / 2
                        rect = Rectangle((x0, y0), w_box, h_box, fill=False,
                                          edgecolor=color, linewidth=1.5)
                        vax.add_patch(rect)
                        video_boxes[cam_name].append(rect)
        ax.set_title(f"Reconstructed 3d trajectories — frame {frame}")
        return [*trails.values(), *points.values(), *video_images.values(),
                *[b for boxes in video_boxes.values() for b in boxes]]

    return FuncAnimation(ax.figure, update, frames=frames, interval=1000 / FPS, blit=False)


def set_equal_aspect(ax, df, z_scale=1.0):
    """Axes with the same meters/unit scale on X and Y (no visual distortion on the court).

    z_scale compresses both the range and the vertical box shown relative to X/Y: player
    heights only span a few meters against the ~30 of court length/width, so leaving it
    at 1 makes the animation a huge, mostly empty cube vertically.
    """
    xs = np.concatenate([df["X"].to_numpy(), [-COURT_HALF_LENGTH, COURT_HALF_LENGTH]])
    ys = np.concatenate([df["Y"].to_numpy(), [-COURT_HALF_WIDTH, COURT_HALF_WIDTH]])
    zs = np.concatenate([df["Z"].to_numpy(), [0.0]])

    x_mid, y_mid, z_mid = xs.mean(), ys.mean(), zs.mean()
    half_range = max(xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()) / 2

    ax.set_xlim(x_mid - half_range, x_mid + half_range)
    ax.set_ylim(y_mid - half_range, y_mid + half_range)
    ax.set_zlim(z_mid - half_range * z_scale, z_mid + half_range * z_scale)
    ax.set_box_aspect((1, 1, z_scale))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--animate", action="store_true",
                         help="show an animation of the movements instead of the static plot")
    parser.add_argument("--save", action="store_true",
                         help="with --animate, save the gif instead of showing it on screen")
    args = parser.parse_args()

    df = pd.read_csv(POS_3D_CSV)

    if args.animate:
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 2.2], hspace=0.15, wspace=0.05)
        ax = fig.add_subplot(gs[1, :], projection="3d")

        video_sources = {}
        for i, (cam_name, video_path) in enumerate(VIDEO_PATHS.items()):
            if not video_path.exists():
                continue
            vax = fig.add_subplot(gs[0, i])
            vax.set_title(cam_name, fontsize=9)
            vax.axis("off")
            video_sources[cam_name] = (VideoFrameSource(video_path), vax)
    else:
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(projection="3d")

    draw_court(ax)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    set_equal_aspect(ax, df, z_scale=0.08 if args.animate else 0.15)
    ax.view_init(elev=25, azim=-60)

    if args.animate:
        bboxes = load_bboxes() if GLOBAL_ID_MAP_CSV.exists() else None
        anim = animate_trajectories(df, ax, video_sources, bboxes)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, ncol=2)
        fig.tight_layout()
        if args.save:
            OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
            anim.save(OUT_GIF, writer=PillowWriter(fps=FPS))
            print(f"Saved {OUT_GIF}")
        else:
            plt.show()
        for source, _ in video_sources.values():
            source.release()
        return

    plot_trajectories(df, ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, ncol=2)
    ax.set_title("Reconstructed 3d trajectories")
    fig.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Saved {OUT_PNG}")

    plt.show()


if __name__ == "__main__":
    main()
