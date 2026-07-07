'''
Steps to implement:
- import the GT data
- import the results of the tracking
- for each frame match the bounding boxes of GT and tracking (min center distance for association)
- compute IOU for each box couple
'''

import json
from pathlib import Path
import pandas as pd
from scipy.optimize import linear_sum_assignment
import numpy as np
import re
import torch
from torchvision.ops import box_iou

STRIDE = 25 // 5
ROOT = Path(__file__).parent.parent
JSON_GT = ROOT / "annotations/_annotations.coco.json"
TRACKING_CSVS = {
    "out2": ROOT / "tracking_results/tracking_2d/positions/2d_positions0.csv",
    "out4": ROOT / "tracking_results/tracking_2d/positions/2d_positions1.csv",
    "out13": ROOT / "tracking_results/tracking_2d/positions/2d_positions2.csv",
}


def load_gt(json_gt):
    with open(json_gt, "r") as jsonfile:
        data = json.load(jsonfile)
    cat_id_to_class = {
        cat["id"]: (32 if "ball" in cat["name"].lower() else 0)
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

def load_track(tracking_csvs):
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

def xywh_to_xyxy(df):
    df["bbox"] = df["bbox"].apply(
        lambda box: [box[0], box[1], box[0] + box[2], box[1] + box[3]])
    return df

def hungarian_matching(gt_df, res_df):
    gt_df = xywh_to_xyxy(gt_df)
    res_df = xywh_to_xyxy(res_df)
    gt_players = gt_df[gt_df["class_id"] == 0]["bbox"].tolist()
    gt_ball = gt_df[gt_df["class_id"] == 32]["bbox"].tolist()
    res_players = res_df[res_df["class_id"] == 0]["bbox"].tolist()
    res_ball = res_df[res_df["class_id"] == 32]["bbox"].tolist()
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

def downsample_df(df):
    df_ds = {}
    for cam in df:
        df_ds[cam] = df[cam][df[cam]["frame"] % STRIDE == 0]
        df_ds[cam]["frame_ds"] = (df_ds[cam]["frame"] / STRIDE).astype(int)
    return df_ds

def main():
    gt_df = load_gt(JSON_GT)
    track_df = load_track(TRACKING_CSVS)
    track_df = downsample_df(track_df)

    for cam in gt_df:
        common_frames = sorted(
            set(gt_df[cam]["frame"]).intersection(track_df[cam]["frame_ds"]))
        for frame in common_frames:
            df1 = gt_df[cam][gt_df[cam]["frame"] == frame]
            df2 = track_df[cam][track_df[cam]["frame_ds"] == frame]
            player_cost_mat, ball_cost_mat = hungarian_matching(df1, df2)
            if player_cost_mat is not None:
                r_p, c_p = linear_sum_assignment(player_cost_mat.numpy(), True)
            if ball_cost_mat is not None:    
                r_b, c_b = linear_sum_assignment(ball_cost_mat.numpy(), True)
            
if __name__ == "__main__":
    main()
