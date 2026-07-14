import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).parent.parent

POS_3D_CSV = ROOT / "tracking_results/tracking_3d/3d_positions.csv"
OUT_PNG = ROOT / "tracking_results/tracking_3d/3d_positions.png"
OUT_GIF = ROOT / "tracking_results/tracking_3d/3d_positions.gif"

FPS = 25            # frame rate nativo del video tracciato
TRAIL_LEN = 15       # quanti frame di scia mostrare dietro alla posizione corrente

# Campo da basket 28x15m, origine al centro, come da CAL_POINTS in calibration.py.
COURT_HALF_LENGTH = 14.0
COURT_HALF_WIDTH = 7.5
KEY_LENGTH = 8.2   # distanza dal centro linea di fondo alla linea del tiro libero
KEY_HALF_WIDTH = 2.4


def draw_court(ax):
    outline = [
        (-COURT_HALF_LENGTH, -COURT_HALF_WIDTH), (COURT_HALF_LENGTH, -COURT_HALF_WIDTH),
        (COURT_HALF_LENGTH, COURT_HALF_WIDTH), (-COURT_HALF_LENGTH, COURT_HALF_WIDTH),
        (-COURT_HALF_LENGTH, -COURT_HALF_WIDTH),
    ]
    lines = [
        outline,
        [(0, -COURT_HALF_WIDTH), (0, COURT_HALF_WIDTH)],  # linea di meta' campo
        [(-COURT_HALF_LENGTH, -KEY_HALF_WIDTH), (-KEY_LENGTH, -KEY_HALF_WIDTH),
         (-KEY_LENGTH, KEY_HALF_WIDTH), (-COURT_HALF_LENGTH, KEY_HALF_WIDTH)],  # area sx
        [(COURT_HALF_LENGTH, -KEY_HALF_WIDTH), (KEY_LENGTH, -KEY_HALF_WIDTH),
         (KEY_LENGTH, KEY_HALF_WIDTH), (COURT_HALF_LENGTH, KEY_HALF_WIDTH)],  # area dx
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


def animate_trajectories(df, ax):
    """Anima le posizioni frame per frame, con una breve scia dietro a ciascun oggetto."""
    cmap = plt.get_cmap("tab20")
    object_ids = sorted(df["object_id"].unique())
    colors = {oid: cmap(i % cmap.N) for i, oid in enumerate(object_ids)}
    traj_by_id = {oid: df[df["object_id"] == oid].sort_values("frame") for oid in object_ids}

    trails = {oid: ax.plot([], [], [], color=colors[oid], linewidth=1.5)[0] for oid in object_ids}
    points = {oid: ax.plot([], [], [], color=colors[oid], marker="o", markersize=6,
                            linestyle="none", label=f"id {oid}")[0]
              for oid in object_ids}

    frames = sorted(df["frame"].unique())

    def update(frame):
        for oid, traj in traj_by_id.items():
            window = traj[(traj["frame"] <= frame) & (traj["frame"] > frame - TRAIL_LEN)]
            trails[oid].set_data_3d(window["X"], window["Y"], window["Z"])
            cur = window[window["frame"] == frame]
            points[oid].set_data_3d(cur["X"], cur["Y"], cur["Z"])
        ax.set_title(f"Traiettorie 3d ricostruite — frame {frame}")
        return [*trails.values(), *points.values()]

    return FuncAnimation(ax.figure, update, frames=frames, interval=1000 / FPS, blit=False)


def set_equal_aspect(ax, df):
    """Assi con la stessa scala metri/unita' su X, Y, Z (niente distorsioni visive)."""
    xs = np.concatenate([df["X"].to_numpy(), [-COURT_HALF_LENGTH, COURT_HALF_LENGTH]])
    ys = np.concatenate([df["Y"].to_numpy(), [-COURT_HALF_WIDTH, COURT_HALF_WIDTH]])
    zs = np.concatenate([df["Z"].to_numpy(), [0.0]])

    x_mid, y_mid, z_mid = xs.mean(), ys.mean(), zs.mean()
    half_range = max(xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()) / 2

    ax.set_xlim(x_mid - half_range, x_mid + half_range)
    ax.set_ylim(y_mid - half_range, y_mid + half_range)
    ax.set_zlim(z_mid - half_range, z_mid + half_range)
    ax.set_box_aspect((1, 1, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--animate", action="store_true",
                         help="mostra un'animazione dei movimenti invece del plot statico")
    parser.add_argument("--save", action="store_true",
                         help="con --animate, salva la gif invece di mostrarla a schermo")
    args = parser.parse_args()

    df = pd.read_csv(POS_3D_CSV)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(projection="3d")

    draw_court(ax)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    set_equal_aspect(ax, df)
    ax.view_init(elev=25, azim=-60)

    if args.animate:
        anim = animate_trajectories(df, ax)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, ncol=2)
        fig.tight_layout()
        if args.save:
            OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
            anim.save(OUT_GIF, writer=PillowWriter(fps=FPS))
            print(f"Salvato {OUT_GIF}")
        else:
            plt.show()
        return

    plot_trajectories(df, ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, ncol=2)
    ax.set_title("Traiettorie 3d ricostruite")
    fig.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Salvato {OUT_PNG}")

    plt.show()


if __name__ == "__main__":
    main()
