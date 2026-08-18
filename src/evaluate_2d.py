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
IOU_THRESHOLD = 1e-5


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

def load_track(tracking_csvs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    track_df = {}
    for cam_name, csv_path in tracking_csvs.items():
        df = pd.read_csv(csv_path)
        df["bbox"] = df.apply(
            lambda row: [row["u"], row["v"], row["w"], row["h"]], axis=1)
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
    player_track_df = downsample_df(load_track(PLAYER_TRACKING_CSVS))

    tp_p, fp_p, fn_p, iou_sum_p = 0, 0, 0, 0.0

    for cam in gt_df:
        eval_frames = sorted(set(gt_df[cam]["frame"]))
        for frame in eval_frames:
            gt_frame = xywh_to_xyxy(gt_df[cam][gt_df[cam]["frame"] == frame])
            gt_players = gt_frame[gt_frame["class_id"] == 0]["bbox"].tolist()

            player_res_frame = uvwh_to_xyxy(player_track_df[cam][player_track_df[cam]["frame_ds"] == frame])
            res_players = player_res_frame["bbox"].tolist()

            player_mat = compute_iou_mat(gt_players, res_players)
            tp, fp, fn, iou = match_boxes(player_mat, len(gt_players), len(res_players), IOU_THRESHOLD)
            tp_p += tp; fp_p += fp; fn_p += fn; iou_sum_p += iou

    print("=== Players (YOLO) ===")
    print_metrics("Players", tp_p, fp_p, fn_p, iou_sum_p)
    return tp_p, fp_p, fn_p, iou_sum_p


def evaluate_ball_tracker(label: str, tracking_csvs: dict[str, Path]) -> tuple[int, int, int, float]:
    gt_df = load_gt(JSON_GT)
    ball_track_df = downsample_df(load_track(tracking_csvs))

    tp_b, fp_b, fn_b, iou_sum_b = 0, 0, 0, 0.0

    for cam in gt_df:
        eval_frames = sorted(set(gt_df[cam]["frame"]))
        for frame in eval_frames:
            gt_frame = xywh_to_xyxy(gt_df[cam][gt_df[cam]["frame"] == frame])
            gt_ball = gt_frame[gt_frame["class_id"] == 1]["bbox"].tolist()

            ball_res_frame = uvwh_to_xyxy(ball_track_df[cam][ball_track_df[cam]["frame_ds"] == frame])
            res_ball = ball_res_frame["bbox"].tolist()

            ball_mat = compute_iou_mat(gt_ball, res_ball)
            tp, fp, fn, iou = match_boxes(ball_mat, len(gt_ball), len(res_ball), IOU_THRESHOLD)
            tp_b += tp; fp_b += fp; fn_b += fn; iou_sum_b += iou

    print(f"\n=== {label} ===")
    print_metrics("Ball", tp_b, fp_b, fn_b, iou_sum_b)
    return tp_b, fp_b, fn_b, iou_sum_b


def main() -> None:
    evaluate_players()
    print("\n" + "-" * 80)
    evaluate_ball_tracker("Ball: classic tracker", BALL_TRACKING_CSVS["classic"])
    print("\n" + "-" * 80)
    evaluate_ball_tracker("Ball: WASB tracker", BALL_TRACKING_CSVS["wasb"])


if __name__ == "__main__":
    main()
