import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).parent.parent

GT_JSON     = ROOT / "annotations/_annotations.coco.json"
METRICS_CSV = ROOT / "tracking_results/tracking_2d/evaluation/metrics_summary.csv"

TRACKING_CSVS = {
    "out2":  ROOT / "tracking_results/tracking_2d/positions/2d_positions0.csv",
    "out4":  ROOT / "tracking_results/tracking_2d/positions/2d_positions1.csv",
    "out13": ROOT / "tracking_results/tracking_2d/positions/2d_positions2.csv",
}

IOU_THRESHOLD = 0.2
GT_FPS        = 5
VIDEO_FPS     = 25
STRIDE        = VIDEO_FPS // GT_FPS   # GT frame N → video frame (N-1)*5

YOLO_PERSON = 0
YOLO_BALL   = 32


# ---------- data loading ----------

def load_gt(json_path):
    """Load COCO annotations into a flat DataFrame with columns:
       camera, video_frame, class_id, x1, y1, x2, y2
    """
    with open(json_path) as f:
        coco = json.load(f)

    # category id → YOLO class id (skip root supercategory id=0)
    cat_to_class = {}
    for cat in coco["categories"]:
        if cat["name"] == "Ball":
            cat_to_class[cat["id"]] = YOLO_BALL
        elif cat["id"] != 0:
            cat_to_class[cat["id"]] = YOLO_PERSON

    # image id → (camera name, video frame index)
    img_to_frame = {}
    for img in coco["images"]:
        m = re.match(r"(out\d+)_frame_(\d+)", img["extra"]["name"])
        if m:
            cam        = m.group(1)
            gt_frame   = int(m.group(2))
            video_frame = (gt_frame - 1) * STRIDE
            img_to_frame[img["id"]] = (cam, video_frame)

    # build flat list of annotations, converting bbox to xyxy
    rows = []
    for ann in coco["annotations"]:
        cls = cat_to_class.get(ann["category_id"])
        if cls is None or ann["image_id"] not in img_to_frame:
            continue
        cam, video_frame = img_to_frame[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        rows.append({
            "camera":      cam,
            "video_frame": video_frame,
            "class_id":    cls,
            "x1": x,     "y1": y,
            "x2": x + w, "y2": y + h,
        })

    return pd.DataFrame(rows)


# ---------- geometry ----------

def center_to_xyxy(boxes):
    """[u_center, v_center, w, h] → [x1, y1, x2, y2]"""
    b = np.asarray(boxes, dtype=float)
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1)


def pairwise_iou(gt_boxes, pred_boxes):
    """Vectorised N×M IoU matrix. Both inputs in xyxy format."""
    gt   = gt_boxes[:, None, :]    # (N, 1, 4)
    pred = pred_boxes[None, :, :]  # (1, M, 4)

    inter_x1 = np.maximum(gt[..., 0], pred[..., 0])
    inter_y1 = np.maximum(gt[..., 1], pred[..., 1])
    inter_x2 = np.minimum(gt[..., 2], pred[..., 2])
    inter_y2 = np.minimum(gt[..., 3], pred[..., 3])
    inter    = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

    area_gt   = (gt[..., 2]   - gt[..., 0])   * (gt[..., 3]   - gt[..., 1])
    area_pred = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
    union     = area_gt + area_pred - inter

    return np.where(union > 0, inter / union, 0.0)


# ---------- matching ----------

def match_and_count(gt_boxes, pred_boxes, threshold):
    """
    Hungarian matching on IoU. Returns (tp, fp, fn, iou_sum).
    """
    n_gt, n_pred = len(gt_boxes), len(pred_boxes)

    if n_gt == 0 and n_pred == 0:
        return 0, 0, 0, 0.0
    if n_gt == 0:
        return 0, n_pred, 0, 0.0
    if n_pred == 0:
        return 0, 0, n_gt, 0.0

    iou          = pairwise_iou(gt_boxes, pred_boxes)
    row_ind, col_ind = linear_sum_assignment(1 - iou)

    tp, iou_sum  = 0, 0.0
    matched_gt   = set()
    matched_pred = set()
    for r, c in zip(row_ind, col_ind):
        if iou[r, c] >= threshold:
            tp      += 1
            iou_sum += float(iou[r, c])
            matched_gt.add(r)
            matched_pred.add(c)

    fn = n_gt   - len(matched_gt)
    fp = n_pred - len(matched_pred)
    return tp, fp, fn, iou_sum


# ---------- evaluation ----------

def evaluate_camera(cam_name, tracking_csv, gt_df, threshold=IOU_THRESHOLD):
    pred_df = pd.read_csv(tracking_csv)
    gt_cam  = gt_df[gt_df["camera"] == cam_name]

    acc = {cls: {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0}
           for cls in ["person", "ball"]}

    for video_frame in gt_cam["video_frame"].unique():
        gt_frame   = gt_cam[gt_cam["video_frame"] == video_frame]
        pred_frame = pred_df[pred_df["frame"] == video_frame]

        for cls_id, cls_name in [(YOLO_PERSON, "person"), (YOLO_BALL, "ball")]:
            gt_cls   = gt_frame[gt_frame["class_id"] == cls_id]
            pred_cls = pred_frame[pred_frame["class_id"] == cls_id]

            gt_boxes   = gt_cls[["x1", "y1", "x2", "y2"]].values
            pred_boxes = center_to_xyxy(pred_cls[["u", "v", "w", "h"]].values) \
                         if len(pred_cls) else np.empty((0, 4))

            tp, fp, fn, iou_sum = match_and_count(gt_boxes, pred_boxes, threshold)
            acc[cls_name]["tp"]      += tp
            acc[cls_name]["fp"]      += fp
            acc[cls_name]["fn"]      += fn
            acc[cls_name]["iou_sum"] += iou_sum

    metrics = {}
    for cls_name, r in acc.items():
        tp, fp, fn = r["tp"], r["fp"], r["fn"]
        metrics[cls_name] = {
            "TP":        tp,
            "FP":        fp,
            "FN":        fn,
            "precision": round(tp / (tp + fp)      if (tp + fp) > 0 else 0.0, 4),
            "recall":    round(tp / (tp + fn)      if (tp + fn) > 0 else 0.0, 4),
            "mean_iou":  round(r["iou_sum"] / tp   if tp > 0       else 0.0, 4),
        }
    return metrics


# ---------- main ----------

def main():
    gt_df = load_gt(GT_JSON)

    rows = []
    for cam_name, csv_path in TRACKING_CSVS.items():
        if cam_name not in gt_df["camera"].values:
            continue
        metrics = evaluate_camera(cam_name, csv_path, gt_df)
        for cls_name, m in metrics.items():
            rows.append({"camera": cam_name, "class": cls_name, **m})

    pd.DataFrame(rows).to_csv(METRICS_CSV, index=False)


if __name__ == "__main__":
    main()
