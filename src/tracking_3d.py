import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).parent.parent

CALIB_CSV = ROOT / "camera_calibration/camera_calibration.csv"

TRAJ_2D = {
    "out2": ROOT / "tracking_results/tracking_2d/trajectories/2d_positions0.csv",
    "out4": ROOT / "tracking_results/tracking_2d/trajectories/2d_positions1.csv",
    "out13": ROOT / "tracking_results/tracking_2d/trajectories/2d_positions2.csv",
}

CAM_ID_TO_NAME = {"cam_0": "out2", "cam_1": "out4", "cam_2": "out13"}

OUT_CSV = ROOT / "tracking_results/tracking_3d/3d_positions.csv"

# Cross-camera association: a candidate match between two detections is accepted
# only if their triangulated point reprojects within this many pixels in both
# views, and only if the pairing wins the Hungarian assignment in >= MIN_VOTES
# distinct frames (and is a mutual best match) across the whole sequence.
MAX_REPROJ_ERROR_PX = 30.0
MIN_VOTES = 5

# Bounding box di plausibilita' per scartare triangolazioni sbagliate (es. persone
# fuori dal campo, o match cross-camera errati): campo 28x15m (origine al centro,
# vedi CAL_POINTS in calibration.py) + un margine per panchine/arbitri a bordo campo.
COURT_HALF_LENGTH = 14.0
COURT_HALF_WIDTH = 7.5
COURT_MARGIN = 5.0
Z_MIN, Z_MAX = -1.0, 3.0


def in_bounds(X, Y, Z):
    return (-COURT_HALF_LENGTH - COURT_MARGIN <= X <= COURT_HALF_LENGTH + COURT_MARGIN
            and -COURT_HALF_WIDTH - COURT_MARGIN <= Y <= COURT_HALF_WIDTH + COURT_MARGIN
            and Z_MIN <= Z <= Z_MAX)


def load_cameras(path):
    """Legge camera_calibration.csv -> dict camera -> {K, dist, P}."""
    cams = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["camera"]
            K = np.array([[float(row[f"K{i}{j}"]) for j in range(3)] for i in range(3)])
            P = np.array([[float(row[f"P{i}{j}"]) for j in range(4)] for i in range(3)])
            dist = np.array([float(row["k1"]), float(row["k2"]),
                              float(row["p1"]), float(row["p2"])])
            cams[name] = {"K": K, "dist": dist, "P": P}
    return cams


def load_2d_trajectories():
    """Legge le traiettorie 2d di ogni camera, indicizzate per frame (un frame puo' avere piu' oggetti)."""
    tracks = {}
    for cam_name, csv_path in TRAJ_2D.items():
        df = pd.read_csv(csv_path)
        assert (df["cam_id"].map(CAM_ID_TO_NAME) == cam_name).all()
        tracks[cam_name] = df.set_index("frame")[["u", "v", "object_id"]]
    return tracks


def undistort_points(pts, K, dist):
    """pts: (N,2) pixel distorti -> (N,2) pixel undistorted (stessa K)."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 1, 2)
    undist = cv2.undistortPoints(pts, K, dist, P=K)
    return undist.reshape(-1, 2)


def triangulate_point(views):
    """views: lista di (P, (u,v)) da >=2 camere. Ritorna X (3,) via DLT multi-vista."""
    A = []
    for P, (u, v) in views:
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    return Xh[:3] / Xh[3]


def reprojection_error(P_a, pt_a, P_b, pt_b):
    """Errore di riproiezione (somma su entrambe le viste) del punto triangolato da una coppia."""
    X = triangulate_point([(P_a, pt_a), (P_b, pt_b)])
    Xh = np.append(X, 1.0)

    def proj(P):
        p = P @ Xh
        return p[:2] / p[2]

    return np.linalg.norm(proj(P_a) - pt_a) + np.linalg.norm(proj(P_b) - pt_b)


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def vote_pairwise_matches(tracks, cams):
    """Per ogni coppia di camere, conta quante volte ciascuna coppia di object_id
    vince l'assegnamento Hungarian frame per frame con errore di riproiezione accettabile."""
    cam_names = list(tracks.keys())
    votes = {}
    for i in range(len(cam_names)):
        for j in range(i + 1, len(cam_names)):
            cam_a, cam_b = cam_names[i], cam_names[j]
            df_a, df_b = tracks[cam_a], tracks[cam_b]
            P_a, P_b = cams[cam_a]["P"], cams[cam_b]["P"]
            common_frames = df_a.index.intersection(df_b.index).unique()

            pair_votes = defaultdict(int)
            for frame in common_frames:
                rows_a = df_a.loc[[frame]]
                rows_b = df_b.loc[[frame]]
                ids_a = rows_a["object_id"].to_numpy()
                ids_b = rows_b["object_id"].to_numpy()
                pts_a = rows_a[["u", "v"]].to_numpy(dtype=float)
                pts_b = rows_b[["u", "v"]].to_numpy(dtype=float)

                cost = np.empty((len(ids_a), len(ids_b)))
                for m in range(len(ids_a)):
                    for n in range(len(ids_b)):
                        cost[m, n] = reprojection_error(P_a, pts_a[m], P_b, pts_b[n])

                row_idx, col_idx = linear_sum_assignment(cost)
                for m, n in zip(row_idx, col_idx):
                    if cost[m, n] <= MAX_REPROJ_ERROR_PX:
                        pair_votes[(ids_a[m], ids_b[n])] += 1

            votes[(cam_a, cam_b)] = pair_votes
    return votes


def resolve_matches(votes):
    """Tiene solo le coppie mutual-best con abbastanza voti, e le unisce in gruppi cross-camera.

    Le fusioni vengono applicate in ordine di confidenza decrescente (piu' voti prima) e una
    fusione viene scartata se creerebbe un gruppo con due object_id diversi della stessa camera
    (puo' succedere per transitivita': A-B e B-C validi non implicano che A e C siano compatibili).
    """
    candidates = []
    for (cam_a, cam_b), pair_votes in votes.items():
        best_b_for_a = {}
        best_a_for_b = {}
        for (id_a, id_b), count in pair_votes.items():
            if count < MIN_VOTES:
                continue
            if count > best_b_for_a.get(id_a, (0, None))[0]:
                best_b_for_a[id_a] = (count, id_b)
            if count > best_a_for_b.get(id_b, (0, None))[0]:
                best_a_for_b[id_b] = (count, id_a)

        for id_a, (count, id_b) in best_b_for_a.items():
            if best_a_for_b.get(id_b, (0, None))[1] == id_a:
                candidates.append((count, (cam_a, id_a), (cam_b, id_b)))

    candidates.sort(key=lambda c: -c[0])

    uf = UnionFind()
    group_members = {}  # root -> set of (cam, object_id)
    for count, key_a, key_b in candidates:
        root_a, root_b = uf.find(key_a), uf.find(key_b)
        if root_a == root_b:
            continue
        members_a = group_members.get(root_a, {key_a})
        members_b = group_members.get(root_b, {key_b})
        if {cam for cam, _ in members_a} & {cam for cam, _ in members_b}:
            continue  # creerebbe due id della stessa camera nello stesso gruppo
        uf.union(key_a, key_b)
        group_members.pop(root_a, None)
        group_members.pop(root_b, None)
        group_members[uf.find(key_a)] = members_a | members_b
    return uf


def assign_global_ids(tracks, uf):
    """Raggruppa gli (cam, object_id) unificati e assegna un global_id sequenziale a ogni gruppo."""
    groups = defaultdict(list)
    for cam_name, df in tracks.items():
        for object_id in df["object_id"].unique():
            key = (cam_name, object_id)
            if key in uf.parent:
                groups[uf.find(key)].append(key)

    global_id_map = {}
    for global_id, members in enumerate(sorted(groups.values(), key=min)):
        for member in members:
            global_id_map[member] = global_id
    return global_id_map


def main():
    cams = load_cameras(CALIB_CSV)
    tracks = load_2d_trajectories()

    for cam_name, df in tracks.items():
        pts = df[["u", "v"]].to_numpy(dtype=float)
        und = undistort_points(pts, cams[cam_name]["K"], cams[cam_name]["dist"])
        df["u"], df["v"] = und[:, 0], und[:, 1]

    votes = vote_pairwise_matches(tracks, cams)
    uf = resolve_matches(votes)
    global_id_map = assign_global_ids(tracks, uf)

    for cam_name, df in tracks.items():
        df["global_id"] = [global_id_map.get((cam_name, oid)) for oid in df["object_id"]]

    all_frames = sorted(set().union(*(df.index for df in tracks.values())))

    rows = []
    for frame in all_frames:
        detections = defaultdict(list)
        for cam_name, df in tracks.items():
            if frame not in df.index:
                continue
            for _, row in df.loc[[frame]].iterrows():
                gid = row["global_id"]
                if gid is None:
                    continue
                detections[gid].append((cam_name, cams[cam_name]["P"], (row["u"], row["v"])))

        for gid, dets in detections.items():
            if len(dets) < 2:
                continue  # servono almeno 2 viste per triangolare
            views = [(P, uv) for _, P, uv in dets]
            cams_used = [cam_name for cam_name, _, _ in dets]
            X, Y, Z = triangulate_point(views)
            if not in_bounds(X, Y, Z):
                continue  # probabile match cross-camera errato o persona fuori dal campo
            rows.append([frame, gid, X, Y, Z, len(views), "+".join(cams_used)])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows, columns=["frame", "object_id", "X", "Y", "Z", "n_views", "cameras"]
    ).to_csv(OUT_CSV, index=False)
    print(f"Scritte {len(rows)} posizioni 3d in {OUT_CSV}")


if __name__ == "__main__":
    main()
