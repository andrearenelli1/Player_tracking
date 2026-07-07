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
    image_frames_df = image_frames_df.rename(columns={"id": "id_2"})
    df = df.merge(image_frames_df, left_on="image_id", right_on="id_2")
    df = df.drop("id", axis=1)
    df = df.drop("category_id", axis=1)
    df = df.drop("iscrowd", axis=1)
    df = df.drop("area", axis=1)
    df = df.drop("segmentation", axis=1)
    df = df.drop("date_captured", axis=1)
    df = df.drop("height", axis=1)
    df = df.drop("width", axis=1)
    df = df.drop("license", axis=1)
    df = df.drop("id_2", axis=1)
    df["cam_id"] = df["file_name"].apply(lambda x: x.split("_")[0])
    df = df.drop("file_name", axis=1)
    df = df.drop("extra", axis=1)
    out = {}
    for cam_name in df["cam_id"].unique():
        out[cam_name] = df[df["cam_id"] == cam_name]
        out[cam_name] = out[cam_name].rename(columns={"image_id": "frame"})
        out[cam_name] = out[cam_name].drop("cam_id", axis=1)
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

def hungarian_matching(gt_df, res_df):
    cost_mat = np.zeros((len(gt_df), len(res_df)))
    
    return

def main():
    gt_df = load_gt(JSON_GT)
    track_df = load_track(TRACKING_CSVS)
    print(gt_df["out2"])
    print(track_df["out2"])


if __name__ == "__main__":
    main()
