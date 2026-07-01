import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).parent.parent

GT_JSON = ROOT / "annotations/_annotations.coco.json"
TRACKING_CSVS = {
    "out2":  ROOT / "tracking_results/tracking_2d/2d_positions0.csv",
    "out4":  ROOT / "tracking_results/tracking_2d/2d_positions1.csv",
    "out13": ROOT / "tracking_results/tracking_2d/2d_positions2.csv",
}
METRICS_JSON = ROOT / "tracking_results/tracking_2d/evaluation/metrics.json"
METRICS_CSV  = ROOT / "tracking_results/tracking_2d/evaluation/metrics_summary.csv"

IOU_THRESHOLD = 0.5
GT_FPS    = 5
VIDEO_FPS = 25
STRIDE    = VIDEO_FPS // GT_FPS

YOLO_PERSON = 0
YOLO_BALL   = 32


# ---------- data loading ----------

def load_gt(json_path):
    with open(json_path) as f:
        coco = json.load(f)
    # Extract id for ball and all ids for players
    ball_cat_ids   = {c["id"] for c in coco["categories"] if c["name"] == "Ball"}
    person_cat_ids = {c["id"] for c in coco["categories"]
                      if c["id"] != 0 and c["name"] != "Ball"}

    # image id → (cam_name, gt_frame_number)
    img_meta = {}
    for img in coco["images"]:
        m = re.match(r"(out\d+)_frame_(\d+)", img["extra"]["name"])
        if m:
            img_meta[img["id"]] = (m.group(1), int(m.group(2)))

    # gt[cam][frame_num][class_id] = list of [x_left, y_top, w, h]
    gt = {}
    for ann in coco["annotations"]:
        if ann["image_id"] not in img_meta:
            continue
        cam, frame_num = img_meta[ann["image_id"]]
        cat_id = ann["category_id"]

        if cat_id in ball_cat_ids:
            cls = YOLO_BALL
        elif cat_id in person_cat_ids:
            cls = YOLO_PERSON
        else:
            continue  # skip root supercategory (id=0)

        gt.setdefault(cam, {}).setdefault(frame_num, {}).setdefault(cls, []).append(ann["bbox"])

    return gt


# ---------- geometry ----------

def coco_to_xyxy(boxes):
    """[x_left, y_top, w, h] → [x1, y1, x2, y2]"""
    b = np.asarray(boxes, dtype=float)
    return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], axis=1)


def center_to_xyxy(boxes):
    """[u_center, v_center, w, h] → [x1, y1, x2, y2]"""
    b = np.asarray(boxes, dtype=float)
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1)


def iou_matrix(a, b):
    """Vectorised N×M IoU. a, b in xyxy format."""
    a = a[:, None, :]   # (N, 1, 4)
    b = b[None, :, :]   # (1, M, 4)
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter  = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union  = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


# ---------- matching ----------

def match(gt_xyxy, pred_xyxy, threshold):
    """
    Hungarian matching on IoU cost.
    Returns (matched [(gt_i, pred_j, iou)], fn_indices, fp_indices).
    """
    n_gt, n_pred = len(gt_xyxy), len(pred_xyxy)
    if n_gt == 0 or n_pred == 0:
        return [], list(range(n_gt)), list(range(n_pred))

    cost = 1 - iou_matrix(gt_xyxy, pred_xyxy)
    rows, cols = linear_sum_assignment(cost)

    matched, matched_gt, matched_pred = [], set(), set()
    for r, c in zip(rows, cols):
        iou_val = 1 - cost[r, c]
        if iou_val >= threshold:
            matched.append((r, c, float(iou_val)))
            matched_gt.add(r)
            matched_pred.add(c)

    fn = [r for r in range(n_gt)   if r not in matched_gt]
    fp = [c for c in range(n_pred) if c not in matched_pred]
    return matched, fn, fp


# ---------- per-camera evaluation ----------

def evaluate_camera(cam_name, csv_path, gt_cam, threshold=IOU_THRESHOLD):
    df = pd.read_csv(csv_path)

    acc = {
        "person": {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0},
        "ball":   {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0},
    }

    for gt_frame_num, gt_classes in gt_cam.items():
        video_frame = (gt_frame_num - 1) * STRIDE
        pred_frame  = df[df["frame"] == video_frame]

        for cls_id, cls_name in [(YOLO_PERSON, "person"), (YOLO_BALL, "ball")]:
            gt_raw  = gt_classes.get(cls_id, [])
            pred_df = pred_frame[pred_frame["class_id"] == cls_id]

            gt_xyxy   = coco_to_xyxy(gt_raw)               if gt_raw         else np.empty((0, 4))
            pred_xyxy = center_to_xyxy(pred_df[["u", "v", "w", "h"]].values) if len(pred_df) else np.empty((0, 4))

            matched, fn, fp = match(gt_xyxy, pred_xyxy, threshold)

            acc[cls_name]["tp"]      += len(matched)
            acc[cls_name]["fn"]      += len(fn)
            acc[cls_name]["fp"]      += len(fp)
            acc[cls_name]["iou_sum"] += sum(m[2] for m in matched)

    metrics = {}
    for cls_name, r in acc.items():
        tp, fp, fn = r["tp"], r["fp"], r["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        mean_iou  = r["iou_sum"] / tp if tp > 0 else 0.0
        metrics[cls_name] = {
            "TP": tp, "FP": fp, "FN": fn,
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "mean_iou":  round(mean_iou, 4),
        }
    return metrics


# ---------- main ----------

def main():
    gt = load_gt(GT_JSON)

    all_metrics = {}
    for cam_name, csv_path in TRACKING_CSVS.items():
        if cam_name not in gt:
            print(f"[skip] no GT for camera {cam_name}")
            continue
        print(f"Evaluating {cam_name} ...")
        metrics = evaluate_camera(cam_name, csv_path, gt[cam_name])
        all_metrics[cam_name] = metrics
        for cls_name, m in metrics.items():
            print(f"  {cls_name:6s}  precision={m['precision']:.3f}  "
                  f"recall={m['recall']:.3f}  mean_iou={m['mean_iou']:.3f}  "
                  f"(TP={m['TP']} FP={m['FP']} FN={m['FN']})")

    with open(METRICS_JSON, "w") as f:
        json.dump(all_metrics, f, indent=2)

    rows = [{"camera": cam, "class": cls, **m}
            for cam, cls_metrics in all_metrics.items()
            for cls, m in cls_metrics.items()]
    pd.DataFrame(rows).to_csv(METRICS_CSV, index=False)

    print(f"\nSaved → {METRICS_JSON}")
    print(f"Saved → {METRICS_CSV}")


if __name__ == "__main__":
    main()
