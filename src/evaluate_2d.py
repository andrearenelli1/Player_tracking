import json
import pandas as pd
import numpy as np
import re
import cv2
import torch
from scipy.optimize import linear_sum_assignment
from pathlib import Path
from torchvision.ops import box_iou

STRIDE = 25 // 5
ROOT = Path(__file__).parent.parent
JSON_GT = ROOT / "annotations/_annotations.coco.json"
CALIB_DIR = ROOT / "calibrated_parameters"
CALIB_FILES = {
    "out2": CALIB_DIR / "cam_2.json",
    "out4": CALIB_DIR / "cam_4.json",
    "out13": CALIB_DIR / "cam_13.json",
}
PLAYER_TRACKING_CSVS = {
    "out2": ROOT / "tracking_results/tracking_2d/trajectories/2d_positions0.csv",
    "out4": ROOT / "tracking_results/tracking_2d/trajectories/2d_positions1.csv",
    "out13": ROOT / "tracking_results/tracking_2d/trajectories/2d_positions2.csv",
}
BALL_TRACKING_CSVS_CLASSIC = {
    "out2": ROOT / "tracking_results/tracking_2d/ball_trajectories/ball_tracking_classic_out2.csv",
    "out4": ROOT / "tracking_results/tracking_2d/ball_trajectories/ball_tracking_classic_out4.csv",
    "out13": ROOT / "tracking_results/tracking_2d/ball_trajectories/ball_tracking_classic_out13.csv",
}
BALL_TRACKING_CSVS_WASB = {
    "out2": ROOT / "tracking_results/tracking_2d/ball_trajectories/ball_tracking_wasb_out2.csv",
    "out4": ROOT / "tracking_results/tracking_2d/ball_trajectories/ball_tracking_wasb_out4.csv",
    "out13": ROOT / "tracking_results/tracking_2d/ball_trajectories/ball_tracking_wasb_out13.csv",
}
BALL_TRACKING_CSVS = {
    "classic": BALL_TRACKING_CSVS_CLASSIC,
    "wasb": BALL_TRACKING_CSVS_WASB,
}
# Standard COCO/PASCAL/CLEAR-MOT matching threshold for players; relaxed for
# the ball, which is small/fast and harder to localize precisely.
IOU_THRESHOLD_PLAYERS = 0.5
IOU_THRESHOLD_BALL = 0.3

# PCK-style matching for the ball: a hit is "center within alpha * GT box
# diagonal", instead of IoU. Ball trackers (WASB, and the classic baseline)
# are point/heatmap detectors that draw a roughly fixed-size box around an
# estimated center -- they are not trying to estimate the ball's true extent.
# The GT box, on the other hand, really does span the ball's extent, and that
# extent varies a lot by camera (close-up vs. full-court), which IoU
# conflates with localization quality. See report.tex for the reasoning.
PCK_ALPHA = 0.5


def load_calibration(calib_path: Path):
    with open(calib_path) as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float64)
    dist = np.array(calib["dist"], dtype=np.float64).reshape(-1)
    return mtx, dist


def rectify_bb(bbox: list, mtx: np.ndarray, dist: np.ndarray) -> list:
    """bbox: [x, y, w, h] (top-left corner, raw/distorted pixels) -> rectified [x, y, w, h].

    Undistorts the 4 corners (not just the center, since undistortion is
    non-linear and would otherwise distort the box shape) and returns the
    axis-aligned bounding box of the undistorted corners.
    """
    x, y, w, h = bbox
    corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]],
                        dtype=np.float32).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(corners, mtx, dist, P=mtx).reshape(-1, 2)
    x_min, y_min = undistorted.min(axis=0)
    x_max, y_max = undistorted.max(axis=0)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def load_gt(json_gt: Path) -> dict[str, pd.DataFrame]:
    with open(json_gt, "r") as jsonfile:
        data = json.load(jsonfile)
    cat_id_to_class = {
        cat["id"]: (1 if "ball" in cat["name"].lower() else 0)
        for cat in data["categories"]
    }
    df = pd.DataFrame(data["annotations"])
    df["class_id"] = df["category_id"].map(cat_id_to_class)
    image_frames_df = pd.DataFrame(data["images"])
    image_frames_df["frame"] = image_frames_df["file_name"].apply(
        lambda x: int(re.search(r"frame_(\d+)", x).group(1))
    )
    df = df.merge(image_frames_df, left_on="image_id", right_on="id")
    df = df.sort_values("frame")
    df = df.drop("category_id", axis=1)
    df = df.drop("iscrowd", axis=1)
    df = df.drop("area", axis=1)
    df = df.drop("segmentation", axis=1)
    df = df.drop("date_captured", axis=1)
    df = df.drop("height", axis=1)
    df = df.drop("width", axis=1)
    df = df.drop("license", axis=1)
    df["cam_id"] = df["file_name"].apply(lambda x: x.split("_")[0])
    df = df.drop("file_name", axis=1)
    df = df.drop("extra", axis=1)
    df = df.drop("id_x", axis=1)
    df = df.drop("id_y", axis=1)
    df = df.drop("image_id", axis=1)
    out = {}
    for cam_name in df["cam_id"].unique():
        out[cam_name] = df[df["cam_id"] == cam_name].copy()
        out[cam_name] = out[cam_name].drop("cam_id", axis=1)
        out[cam_name]["frame"] = out[cam_name]["frame"] - 1
        mtx, dist = load_calibration(CALIB_FILES[cam_name])
        out[cam_name]["bbox"] = out[cam_name]["bbox"].apply(lambda box: rectify_bb(box, mtx, dist))
    return out

def rectify_center_bb(box: list, mtx: np.ndarray, dist: np.ndarray) -> list:
    """[u, v, w, h] center-format box in raw/distorted pixels -> same-format box
    after undistortion. Same corner-based approach as rectify_bb above, just
    center-in/center-out (matches tracking_players_2d.py's rectify_center_box)."""
    u, v, w, h = box
    x, y = u - w / 2, v - h / 2
    x_min, y_min, rw, rh = rectify_bb([x, y, w, h], mtx, dist)
    return [x_min + rw / 2, y_min + rh / 2, rw, rh]


def load_track(tracking_csvs: dict[str, Path], calib_files: dict[str, Path] | None = None) -> dict[str, pd.DataFrame]:
    """Reads tracker output CSVs (frame, cam_id, class_id, object_id, u, v, w, h; u,v
    is the box center). Pass `calib_files` only for trackers whose output is still in
    raw/distorted pixels (e.g. the ball trackers) so it gets rectified here to match
    the (already-rectified) GT and player detections -- player CSVs are rectified
    upstream in tracking_players_2d.py, so they must be passed with calib_files=None
    to avoid undistorting them twice."""
    track_df = {}
    for cam_name, csv_path in tracking_csvs.items():
        df = pd.read_csv(csv_path)
        df["bbox"] = df.apply(
            lambda row: [row["u"], row["v"], row["w"], row["h"]], axis=1)
        if calib_files is not None:
            mtx, dist = load_calibration(calib_files[cam_name])
            df["bbox"] = df["bbox"].apply(lambda box: rectify_center_bb(box, mtx, dist))
        df = df.drop("u", axis=1)
        df = df.drop("v", axis=1)
        df = df.drop("w", axis=1)
        df = df.drop("h", axis=1)
        df = df.drop("object_id", axis=1)
        track_df[cam_name] = df.drop("cam_id", axis=1)
    return track_df

def xywh_to_xyxy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bbox"] = df["bbox"].apply(
        lambda box: [box[0], box[1], box[0] + box[2], box[1] + box[3]])
    return df

def uvwh_to_xyxy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bbox"] = df["bbox"].apply(
        lambda box: [box[0] - box[2] / 2, box[1] - box[3] / 2,
                     box[0] + box[2] / 2, box[1] + box[3] / 2])
    return df

def compute_iou_mat(gt_boxes: list, res_boxes: list) -> torch.Tensor | None:
    if len(gt_boxes) == 0 or len(res_boxes) == 0:
        return None
    gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
    res_tensor = torch.tensor(res_boxes, dtype=torch.float32)
    return box_iou(gt_tensor, res_tensor, "xyxy")

def downsample_df(df: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    df_ds = {}
    for cam in df:
        df_ds[cam] = df[cam][df[cam]["frame"] % STRIDE == 0].copy()
        df_ds[cam]["frame_ds"] = (df_ds[cam]["frame"] / STRIDE).astype(int)
    return df_ds



def match_boxes(cost_mat: torch.Tensor | None, n_gt: int, n_res: int, threshold: float) -> tuple[int, int, int, float]:

    if cost_mat is None:
        return 0, n_res, n_gt, 0.0

    rows, cols = linear_sum_assignment(cost_mat.detach().cpu().numpy(), maximize=True)

    tp, iou_sum = 0, 0.0
    matched_gt, matched_res = set(), set()
    for r, c in zip(rows, cols):
        iou_val = cost_mat[r, c].item()
        if iou_val >= threshold:
            tp      += 1
            iou_sum += iou_val
            matched_gt.add(r)
            matched_res.add(c)

    fp = n_res - len(matched_res)
    fn = n_gt  - len(matched_gt)
    return tp, fp, fn, iou_sum


def compute_dist_mat(gt_boxes: list, res_boxes: list) -> np.ndarray | None:
    """Pairwise Euclidean distance between box centers (xyxy in, both lists)."""
    if len(gt_boxes) == 0 or len(res_boxes) == 0:
        return None
    gt_centers = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in gt_boxes])
    res_centers = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in res_boxes])
    return np.linalg.norm(gt_centers[:, None, :] - res_centers[None, :, :], axis=2)


def match_points(dist_mat: np.ndarray | None, gt_boxes: list, n_gt: int, n_res: int,
                  alpha: float) -> tuple[int, int, int, float]:
    """Hungarian assignment minimizing center distance; a pair counts as TP if
    distance <= alpha * (that GT box's own diagonal) -- PCK-style, so the
    threshold adapts to how big the ball actually is in that camera/frame
    instead of penalizing predicted-box shape/scale like IoU does."""
    if dist_mat is None:
        return 0, n_res, n_gt, 0.0

    rows, cols = linear_sum_assignment(dist_mat)  # minimize total distance

    tp, dist_sum = 0, 0.0
    matched_gt, matched_res = set(), set()
    for r, c in zip(rows, cols):
        gx0, gy0, gx1, gy1 = gt_boxes[r]
        diag = np.hypot(gx1 - gx0, gy1 - gy0)
        d = dist_mat[r, c]
        if d <= alpha * diag:
            tp       += 1
            dist_sum += d
            matched_gt.add(r)
            matched_res.add(c)

    fp = n_res - len(matched_res)
    fn = n_gt  - len(matched_gt)
    return tp, fp, fn, dist_sum


def print_metrics_dist(label: str, tp: int, fp: int, fn: int, dist_sum: float) -> None:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) \
                                        if (precision + recall) > 0 else 0.0
    mde       = dist_sum / tp if tp > 0 else 0.0
    print(f"{label}:  TP={tp}  FP={fp}  FN={fn}")
    print(f"  Precision = {precision:.3f}")
    print(f"  Recall    = {recall:.3f}")
    print(f"  F1        = {f1:.3f}")
    print(f"  MDE       = {mde:.1f}px  (mean center distance over matched pairs, hit <= {PCK_ALPHA}*GT diagonal)")


def print_metrics(label: str, tp: int, fp: int, fn: int, iou_sum: float) -> None:
    precision = tp / (tp + fp)              if (tp + fp) > 0           else 0.0
    recall    = tp / (tp + fn)              if (tp + fn) > 0           else 0.0
    f1        = 2 * precision * recall / (precision + recall) \
                                            if (precision + recall) > 0 else 0.0
    motp      = iou_sum / tp               if tp > 0                  else 0.0
    print(f"{label}:  TP={tp}  FP={fp}  FN={fn}")
    print(f"  Precision = {precision:.3f}")
    print(f"  Recall    = {recall:.3f}")
    print(f"  F1        = {f1:.3f}")
    print(f"  MOTP      = {motp:.3f}  (mean IoU over matched pairs)")


def evaluate_players() -> tuple[int, int, int, float]:
    gt_df = load_gt(JSON_GT)
    player_track_df = downsample_df(load_track(PLAYER_TRACKING_CSVS))  # already rectified upstream

    tp_p, fp_p, fn_p, iou_sum_p = 0, 0, 0, 0.0
    per_cam = {}

    for cam in gt_df:
        tp_c, fp_c, fn_c, iou_sum_c = 0, 0, 0, 0.0
        eval_frames = sorted(set(gt_df[cam]["frame"]))
        for frame in eval_frames:
            gt_frame = xywh_to_xyxy(gt_df[cam][gt_df[cam]["frame"] == frame])
            gt_players = gt_frame[gt_frame["class_id"] == 0]["bbox"].tolist()

            player_res_frame = uvwh_to_xyxy(player_track_df[cam][player_track_df[cam]["frame_ds"] == frame])
            res_players = player_res_frame["bbox"].tolist()

            player_mat = compute_iou_mat(gt_players, res_players)
            tp, fp, fn, iou = match_boxes(player_mat, len(gt_players), len(res_players), IOU_THRESHOLD_PLAYERS)
            tp_c += tp; fp_c += fp; fn_c += fn; iou_sum_c += iou

        per_cam[cam] = (tp_c, fp_c, fn_c, iou_sum_c)
        tp_p += tp_c; fp_p += fp_c; fn_p += fn_c; iou_sum_p += iou_sum_c

    print("=== Players (YOLO) ===")
    for cam, (tp_c, fp_c, fn_c, iou_sum_c) in per_cam.items():
        print_metrics(f"  [{cam}]", tp_c, fp_c, fn_c, iou_sum_c)
    print_metrics("  [all cameras]", tp_p, fp_p, fn_p, iou_sum_p)
    return tp_p, fp_p, fn_p, iou_sum_p


def evaluate_ball_tracker(label: str, tracking_csvs: dict[str, Path]) -> tuple[int, int, int, float]:
    gt_df = load_gt(JSON_GT)
    ball_track_df = downsample_df(load_track(tracking_csvs, calib_files=CALIB_FILES))  # raw distorted -> rectify here

    tp_b, fp_b, fn_b, iou_sum_b = 0, 0, 0, 0.0
    per_cam = {}

    for cam in gt_df:
        tp_c, fp_c, fn_c, iou_sum_c = 0, 0, 0, 0.0
        eval_frames = sorted(set(gt_df[cam]["frame"]))
        for frame in eval_frames:
            gt_frame = xywh_to_xyxy(gt_df[cam][gt_df[cam]["frame"] == frame])
            gt_ball = gt_frame[gt_frame["class_id"] == 1]["bbox"].tolist()

            ball_res_frame = uvwh_to_xyxy(ball_track_df[cam][ball_track_df[cam]["frame_ds"] == frame])
            res_ball = ball_res_frame["bbox"].tolist()

            ball_mat = compute_iou_mat(gt_ball, res_ball)
            tp, fp, fn, iou = match_boxes(ball_mat, len(gt_ball), len(res_ball), IOU_THRESHOLD_BALL)
            tp_c += tp; fp_c += fp; fn_c += fn; iou_sum_c += iou

        per_cam[cam] = (tp_c, fp_c, fn_c, iou_sum_c)
        tp_b += tp_c; fp_b += fp_c; fn_b += fn_c; iou_sum_b += iou_sum_c

    print(f"\n=== {label} ===")
    for cam, (tp_c, fp_c, fn_c, iou_sum_c) in per_cam.items():
        print_metrics(f"  [{cam}]", tp_c, fp_c, fn_c, iou_sum_c)
    print_metrics("  [all cameras]", tp_b, fp_b, fn_b, iou_sum_b)
    return tp_b, fp_b, fn_b, iou_sum_b


def evaluate_ball_tracker_dist(label: str, tracking_csvs: dict[str, Path],
                                alpha: float = PCK_ALPHA) -> tuple[int, int, int, float]:
    """Same matching as evaluate_ball_tracker, but PCK-style (center distance
    normalized by GT box size) instead of IoU. Also breaks results down per
    camera, since ball box scale (and WASB tuning) differs a lot camera to
    camera -- an aggregate number alone hides that."""
    gt_df = load_gt(JSON_GT)
    ball_track_df = downsample_df(load_track(tracking_csvs, calib_files=CALIB_FILES))

    tp_b, fp_b, fn_b, dist_sum_b = 0, 0, 0, 0.0
    per_cam = {}

    for cam in gt_df:
        tp_c, fp_c, fn_c, dist_sum_c = 0, 0, 0, 0.0
        eval_frames = sorted(set(gt_df[cam]["frame"]))
        for frame in eval_frames:
            gt_frame = xywh_to_xyxy(gt_df[cam][gt_df[cam]["frame"] == frame])
            gt_ball = gt_frame[gt_frame["class_id"] == 1]["bbox"].tolist()

            ball_res_frame = uvwh_to_xyxy(ball_track_df[cam][ball_track_df[cam]["frame_ds"] == frame])
            res_ball = ball_res_frame["bbox"].tolist()

            dist_mat = compute_dist_mat(gt_ball, res_ball)
            tp, fp, fn, dist_sum = match_points(dist_mat, gt_ball, len(gt_ball), len(res_ball), alpha)
            tp_c += tp; fp_c += fp; fn_c += fn; dist_sum_c += dist_sum

        per_cam[cam] = (tp_c, fp_c, fn_c, dist_sum_c)
        tp_b += tp_c; fp_b += fp_c; fn_b += fn_c; dist_sum_b += dist_sum_c

    print(f"\n=== {label} (PCK-style, center dist <= {alpha}*GT diagonal) ===")
    for cam, (tp_c, fp_c, fn_c, dist_sum_c) in per_cam.items():
        print_metrics_dist(f"  [{cam}]", tp_c, fp_c, fn_c, dist_sum_c)
    print_metrics_dist("  [all cameras]", tp_b, fp_b, fn_b, dist_sum_b)
    return tp_b, fp_b, fn_b, dist_sum_b


def main() -> None:
    evaluate_players()
    print("\n" + "-" * 80)
    evaluate_ball_tracker("Ball: classic tracker", BALL_TRACKING_CSVS["classic"])
    print("\n" + "-" * 80)
    evaluate_ball_tracker("Ball: WASB tracker", BALL_TRACKING_CSVS["wasb"])
    print("\n" + "-" * 80)
    evaluate_ball_tracker_dist("Ball: classic tracker", BALL_TRACKING_CSVS["classic"])
    print("\n" + "-" * 80)
    evaluate_ball_tracker_dist("Ball: WASB tracker", BALL_TRACKING_CSVS["wasb"])


if __name__ == "__main__":
    main()
