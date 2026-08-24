import json
import re
import numpy as np
import pandas as pd

from collections import defaultdict
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from tracking_3d import OUT_CSV, load_cameras, triangulate_point, undistort_points


ROOT = Path(__file__).parent.parent

ANNOTATIONS = ROOT / "annotations/_annotations.coco.json"
EVAL_OUT_CSV = ROOT / "tracking_results/tracking_3d/evaluation_3d.csv"

# Annotations are at 5 fps: the file name "out2_frame_0044.png" is the 44th
# sampled frame (1-indexed) of the native 25 fps video.
FRAME_NAME_RE = re.compile(r"(out\d+)_frame_(\d+)\.png")
SAMPLE_STRIDE = 5

# A triangulated GT point and a reconstructed position are paired in the same
# frame only if their distance is within this threshold (meters); beyond that
# it's considered a miss (no match), not a huge error that skews the average.
MAX_MATCH_DIST_M = 3.0


def load_gt_detections(cams):
    """
    Reads the GT COCO bboxes -> DataFrame [frame, category, cam_name, u, v] (undistorted).
    """
    with open(ANNOTATIONS) as f:
        coco = json.load(f)

    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    img_info = {}
    for img in coco["images"]:
        m = FRAME_NAME_RE.match(img["extra"]["name"])
        cam_name, sample_idx = m.group(1), int(m.group(2))
        img_info[img["id"]] = (cam_name, (sample_idx - 1) * SAMPLE_STRIDE)

    rows = []
    for ann in coco["annotations"]:
        cam_name, frame = img_info[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        rows.append([frame, cat_names[ann["category_id"]], cam_name, x + w / 2, y + h / 2])

    df = pd.DataFrame(rows, columns=["frame", "category", "cam_name", "u", "v"])
    for cam_name, sub in df.groupby("cam_name"):
        und = undistort_points(sub[["u", "v"]].to_numpy(dtype=float),
                                cams[cam_name]["K"], cams[cam_name]["dist"])
        df.loc[sub.index, ["u", "v"]] = und
    return df


def triangulate_gt(gt_df, cams):
    """
    For each (frame, category) with >=2 cameras, triangulates the pseudo-GT 3d position.
    """
    rows = []
    for (frame, category), group in gt_df.groupby(["frame", "category"]):
        if len(group) < 2:
            continue  # need at least 2 views
        views = [(cams[r.cam_name]["P"], (r.u, r.v)) for r in group.itertuples()]
        X, Y, Z = triangulate_point(views)
        rows.append([frame, category, X, Y, Z, len(views)])
    return pd.DataFrame(rows, columns=["frame", "category", "X", "Y", "Z", "n_views"])


def match_and_score(gt_3d, recon_3d):
    """
    Pairs GT and reconstruction frame by frame (Hungarian on 3d distances).
    """
    matches = []
    common_frames = sorted(set(gt_3d["frame"]) & set(recon_3d["frame"]))
    for frame in common_frames:
        gt_rows = gt_3d[gt_3d["frame"] == frame].reset_index(drop=True)
        recon_rows = recon_3d[recon_3d["frame"] == frame].reset_index(drop=True)

        gt_pts = gt_rows[["X", "Y", "Z"]].to_numpy()
        recon_pts = recon_rows[["X", "Y", "Z"]].to_numpy()
        cost = np.linalg.norm(gt_pts[:, None, :] - recon_pts[None, :, :], axis=2)

        gt_idx, recon_idx = linear_sum_assignment(cost)
        for gi, ri in zip(gt_idx, recon_idx):
            dist = cost[gi, ri]
            if dist <= MAX_MATCH_DIST_M:
                matches.append([frame, gt_rows.loc[gi, "category"],
                                 int(recon_rows.loc[ri, "object_id"]), dist])

    return pd.DataFrame(matches, columns=["frame", "category", "object_id", "error_m"])


def main():
    cams = load_cameras()

    gt_df = load_gt_detections(cams)
    gt_3d = triangulate_gt(gt_df, cams)
    recon_3d = pd.read_csv(OUT_CSV)

    matches = match_and_score(gt_3d, recon_3d)

    n_gt = len(gt_3d)
    n_matched = len(matches)
    med = matches["error_m"].mean()
    rmse = np.sqrt((matches["error_m"] ** 2).mean())

    print(f"GT pseudo-3d points (>=2 views): {n_gt}")
    print(f"Matched with reconstruction (threshold {MAX_MATCH_DIST_M} m): {n_matched} "
          f"({100 * n_matched / n_gt:.1f}% coverage)")
    print(f"MED  (Mean Euclidean Distance): {med:.3f} m")
    print(f"RMSE (Root Mean Squared Error): {rmse:.3f} m")
    print()
    print("Average error per category:")
    print(matches.groupby("category")["error_m"].agg(["mean", "count"])
          .sort_values("mean").to_string())

    EVAL_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(EVAL_OUT_CSV, index=False)
    print(f"\nMatch details written to {EVAL_OUT_CSV}")


if __name__ == "__main__":
    main()
