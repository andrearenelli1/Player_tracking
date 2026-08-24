import csv
from collections import defaultdict
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).parent.parent

CALIB_DIR = ROOT / "calibrated_parameters"
CALIB_FILES = {"out2": "cam_2.json", "out4": "cam_4.json", "out13": "cam_13.json"}


TRAJ_2D = {
    "out2": ROOT / "tracking_results/tracking_2d/trajectories/2d_positions0.csv",
    "out4": ROOT / "tracking_results/tracking_2d/trajectories/2d_positions1.csv",
    "out13": ROOT / "tracking_results/tracking_2d/trajectories/2d_positions2.csv",
}

CAM_ID_TO_NAME = {"cam_0": "out2", "cam_1": "out4", "cam_2": "out13"}

OUT_CSV = ROOT / "tracking_results/tracking_3d/3d_positions.csv"
GLOBAL_ID_MAP_CSV = ROOT / "tracking_results/tracking_3d/global_id_map.csv"

# Cross-camera association: a candidate match between two detections is accepted
# only if their triangulated point reprojects within this many pixels in both
# views, and only if the pairing wins the Hungarian assignment in >= MIN_VOTES
# distinct frames (and is a mutual best match) across the whole sequence.
MAX_REPROJ_ERROR_PX = 30.0
MIN_VOTES = 5

# Plausibility bounding box to discard wrong triangulations (e.g. people
# outside the court, or wrong cross-camera matches): 28x15m court (origin at center,
# see CAL_POINTS in calibration.py) + a margin for benches/referees at the court edge.
COURT_HALF_LENGTH = 14.0
COURT_HALF_WIDTH = 7.5
COURT_MARGIN = 5.0
Z_MIN, Z_MAX = -1.0, 3.0


def in_bounds(X, Y, Z):
    return (-COURT_HALF_LENGTH - COURT_MARGIN <= X <= COURT_HALF_LENGTH + COURT_MARGIN
            and -COURT_HALF_WIDTH - COURT_MARGIN <= Y <= COURT_HALF_WIDTH + COURT_MARGIN
            and Z_MIN <= Z <= Z_MAX)


def load_cameras():
    cams = {}
    for cam_name, fname in CALIB_FILES.items():
        d = json.loads((CALIB_DIR / fname).read_text())
        K = np.array(d["mtx"])
        dist = np.array(d["dist"]).reshape(-1)          # k1,k2,p1,p2,k3
        R, _ = cv2.Rodrigues(np.array(d["rvecs"]).reshape(3, 1))
        t = np.array(d["tvecs"]).reshape(3, 1) / 1000.0  # mm -> m
        P = K @ np.hstack([R, t])
        cams[cam_name] = {"K": K, "dist": dist, "P": P}
    return cams


def load_2d_trajectories():
    """Reads the 2d trajectories for each camera, indexed by frame (a frame can have multiple objects)."""
    tracks = {}
    for cam_name, csv_path in TRAJ_2D.items():
        df = pd.read_csv(csv_path)
        assert (df["cam_id"].map(CAM_ID_TO_NAME) == cam_name).all()
        tracks[cam_name] = df.set_index("frame")[["u", "v", "object_id"]]
    return tracks


def undistort_points(pts, K, dist):
    """pts: (N,2) distorted pixels -> (N,2) undistorted pixels (same K)."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 1, 2)
    undist = cv2.undistortPoints(pts, K, dist, P=K)
    return undist.reshape(-1, 2)


def triangulate_point(views):
    """views: list of (P, (u,v)) from >=2 cameras. Returns X (3,) via multi-view DLT."""
    A = []
    for P, (u, v) in views:
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    return Xh[:3] / Xh[3]


def reprojection_error(P_a, pt_a, P_b, pt_b):
    """Reprojection error (sum over both views) of the point triangulated from a pair."""
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
    """For each pair of cameras, counts how many times each pair of object_ids
    wins the Hungarian assignment frame by frame with acceptable reprojection error."""
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
    """Keeps only mutual-best pairs with enough votes, and merges them into cross-camera groups.

    Merges are applied in decreasing order of confidence (more votes first), and a
    merge is discarded if it would create a group with two different object_ids from the
    same camera (can happen due to transitivity: A-B and B-C being valid doesn't imply A and C are compatible).
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
            continue  # would create two ids from the same camera in the same group
        uf.union(key_a, key_b)
        group_members.pop(root_a, None)
        group_members.pop(root_b, None)
        group_members[uf.find(key_a)] = members_a | members_b
    return uf


def assign_global_ids(tracks, uf):
    """Groups the unified (cam, object_id) pairs and assigns a sequential global_id to each group."""
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
    cams = load_cameras()
    tracks = load_2d_trajectories()

    for cam_name, df in tracks.items():
        pts = df[["u", "v"]].to_numpy(dtype=float)
        und = undistort_points(pts, cams[cam_name]["K"], cams[cam_name]["dist"])
        df["u"], df["v"] = und[:, 0], und[:, 1]

    votes = vote_pairwise_matches(tracks, cams)
    uf = resolve_matches(votes)
    global_id_map = assign_global_ids(tracks, uf)

    pd.DataFrame(
        [(cam_name, object_id, global_id) for (cam_name, object_id), global_id in global_id_map.items()],
        columns=["cam_name", "object_id", "global_id"],
    ).to_csv(GLOBAL_ID_MAP_CSV, index=False)

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
                continue  # need at least 2 views to triangulate
            views = [(P, uv) for _, P, uv in dets]
            cams_used = [cam_name for cam_name, _, _ in dets]
            X, Y, Z = triangulate_point(views)
            if not in_bounds(X, Y, Z):
                continue  # likely a wrong cross-camera match or a person outside the court
            rows.append([frame, gid, X, Y, Z, len(views), "+".join(cams_used)])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows, columns=["frame", "object_id", "X", "Y", "Z", "n_views", "cameras"]
    ).to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(rows)} 3d positions to {OUT_CSV}")


if __name__ == "__main__":
    main()
