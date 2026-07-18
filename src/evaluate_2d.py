import json
import pandas as pd
import numpy as np
import re
import torch
from scipy.optimize import linear_sum_assignment
from pathlib import Path
from torchvision.ops import box_iou

STRIDE = 25 // 5
ROOT = Path(__file__).parent.parent
JSON_GT = ROOT / "annotations/_annotations.coco.json"
TRACKING_CSVS = {
    "out2": ROOT / "tracking_results/tracking_2d/trajectories/2d_positions0.csv",
    "out4": ROOT / "tracking_results/tracking_2d/trajectories/2d_positions1.csv",
    "out13": ROOT / "tracking_results/tracking_2d/trajectories/2d_positions2.csv",
}
IOU_THRESHOLD = 1e-5


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
        out[cam_name] = df[df["cam_id"] == cam_name]
        out[cam_name] = out[cam_name].drop("cam_id", axis=1)
        out[cam_name]["frame"] = out[cam_name]["frame"] - 1
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
    df["bbox"] = df["bbox"].apply(
        lambda box: [box[0], box[1], box[0] + box[2], box[1] + box[3]])
    return df

def uvwh_to_xyxy(df: pd.DataFrame) -> pd.DataFrame:
    df["bbox"] = df["bbox"].apply(
        lambda box: [box[0] - box[2] / 2, box[1] - box[3] / 2,
                     box[0] + box[2] / 2, box[1] + box[3] / 2])
    return df

def compute_cost_mat(gt_df: pd.DataFrame, res_df: pd.DataFrame) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    gt_df  = xywh_to_xyxy(gt_df)
    res_df = uvwh_to_xyxy(res_df)
    gt_players = gt_df[gt_df["class_id"] == 0]["bbox"].tolist()
    gt_ball = gt_df[gt_df["class_id"] == 1]["bbox"].tolist()
    res_players = res_df[res_df["class_id"] == 1]["bbox"].tolist()
    res_ball = res_df[res_df["class_id"] == 0]["bbox"].tolist()
    gt_pl_tensor = torch.tensor(gt_players, dtype=torch.float32)
    gt_bl_tensor = torch.tensor(gt_ball, dtype=torch.float32)
    res_pl_tensor = torch.tensor(res_players, dtype=torch.float32)
    res_bl_tensor = torch.tensor(res_ball, dtype=torch.float32)
    if len(gt_bl_tensor) != 0 and len(res_bl_tensor) != 0:
        ball_cost_mat = box_iou(gt_bl_tensor, res_bl_tensor, "xyxy")
    else: 
        ball_cost_mat = None
    if len(gt_pl_tensor) != 0 and len(res_pl_tensor) != 0:
        players_cost_mat = box_iou(gt_pl_tensor, res_pl_tensor, "xyxy")
    else: 
        players_cost_mat = None
    return players_cost_mat, ball_cost_mat

def downsample_df(df: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    df_ds = {}
    for cam in df:
        df_ds[cam] = df[cam][df[cam]["frame"] % STRIDE == 0]
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


def main() -> None:
    gt_df    = load_gt(JSON_GT)
    track_df = downsample_df(load_track(TRACKING_CSVS))

    tp_p, fp_p, fn_p, iou_sum_p = 0, 0, 0, 0.0
    tp_b, fp_b, fn_b, iou_sum_b = 0, 0, 0, 0.0

    for cam in gt_df:
        eval_frames = sorted(set(gt_df[cam]["frame"]))
        print(f"Processing {cam}")

        for frame in eval_frames:
            print(f"frame: {frame} in {len(eval_frames)}")
            gt_frame  = gt_df[cam][gt_df[cam]["frame"] == frame]
            res_frame = track_df[cam][track_df[cam]["frame_ds"] == frame]

            n_gt_p  = len(gt_frame[gt_frame["class_id"] == 0])
            n_gt_b  = len(gt_frame[gt_frame["class_id"] == 1])
            n_res_p = len(res_frame[res_frame["class_id"] == 1])
            n_res_b = len(res_frame[res_frame["class_id"] == 0])

            player_mat, ball_mat = compute_cost_mat(gt_frame, res_frame)

            tp, fp, fn, iou = match_boxes(player_mat, n_gt_p, n_res_p, IOU_THRESHOLD)
            tp_p += tp; fp_p += fp; fn_p += fn; iou_sum_p += iou

            tp, fp, fn, iou = match_boxes(ball_mat, n_gt_b, n_res_b, IOU_THRESHOLD)
            tp_b += tp; fp_b += fp; fn_b += fn; iou_sum_b += iou

    print("\n\n\n\n")
    print_metrics("Players", tp_p, fp_p, fn_p, iou_sum_p)
    print()
    print_metrics("Ball",    tp_b, fp_b, fn_b, iou_sum_b)

if __name__ == "__main__":
    main()
