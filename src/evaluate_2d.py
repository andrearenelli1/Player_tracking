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
    df = df.drop("id", axis=1)
    df = df.drop("category_id", axis=1)
    df = df.drop("iscrowd", axis=1)
    df = df.drop("area", axis=1)
    df = df.drop("segmentation", axis=1)
    return df

def hungarian_matching(gt_df, res_df):
    cost_mat = np.zeros((len(gt_df), len(res_df)))
    
    return

def IOU_computation():
    return

def main():
    annotation_df = load_gt(JSON_GT)
    track_df = {}
    for cam_name, csv_path in TRACKING_CSVS.items():
        df = pd.read_csv(csv_path)
        track_df[cam_name] = df
    
    for frame in annotation_df["image_id"].unique():
        cam_id = annotation_df[annotation_df["image_id"] == frame]
        #print(cam_id)

    #print(annotation_df)
    #print(track_df["out2"])
    #print(track_df["out4"])
    #print(track_df["out13"])
        


if __name__ == "__main__":
    main()
