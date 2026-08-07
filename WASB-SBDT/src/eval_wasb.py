"""
Quantitative evaluation of WASB (basketball checkpoint) against an
annotated dataset (COCO format exported from Roboflow), using the official
heatmap decoding (connected components, as in detectors/postprocessor.py)
instead of the approximate argmax used in the older infer_video.py
(renamed/rewritten as track_ball.py, which now uses this same decoding).

The annotations are sampled at 5fps while the source videos are at 25fps
(stride = 5): for each annotated image `<video>_frame_NNNN`, the
corresponding frame in the full-temporal-resolution video is the 0-based
index `(NNNN - 1) * 5 + 2` (the +2 offset was verified empirically by
comparing the annotated images against the video frames pixel-by-pixel,
see check_frame_offset.py -- Roboflow/ffmpeg does not sample exactly the
first frame of each block of 5). WASB requires a window of 3 consecutive
frames (at full 25fps) as input, so we reconstruct the triplet from the
original video instead of using only the 309 isolated annotated images.

Metric: matches src/utils/evaluator.py, the repo's own reference
evaluator (TrackNet/TrackNetV2 lineage WASB builds on), so results are
directly comparable to numbers produced by that evaluator. Computed at
several confidence thresholds:
    TP  = ball visible (GT) and detected within tolerance
    FN  = ball visible (GT) but not detected (no blob above threshold)
    FP1 = ball visible (GT) but detected outside tolerance (wrong position)
    FP2 = ball NOT visible (GT) but the model detects something anyway
    TN  = ball NOT visible (GT) and no detection
    Precision = TP / (TP + FP1 + FP2)
    Recall    = TP / (TP + FN)              -- NOTE: FP1 is deliberately
                excluded from the denominator, per the official evaluator:
                a wrong-position detection only penalizes precision, not
                recall (recall measures "did the model attempt a
                detection", not "was it accurate")
    Accuracy  = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
    RMSE      = sqrt(mean squared error) over every frame where GT and a
                prediction both exist (TP and FP1), regardless of whether
                the position was within tolerance

One deliberate deviation from the official evaluator: their
`dist_threshold` (score_threshold aside, see runner/eval.yaml, default 4)
is a FIXED pixel tolerance on THEIR dataset's frame resolution (SAM/
NBA_data broadcast footage) -- an unknown, likely much lower, resolution
than our 3840x2160 frames. Applying "4px" literally here would be
unfairly strict (a ball 40-170px across would rarely land within 4px of
GT even when well localized). We use an adaptive tolerance instead
(`--tol-factor` * GT bbox diagonal, floored by `--tol-min-px`) so it
scales with our frame's actual resolution and the ball's own apparent
size, and document this explicitly rather than silently reusing a number
calibrated for a different dataset.

Usage (inside the container, from /workspace/src):
    python3 eval_wasb.py \
        --annotations /workspace/annotations/_annotations.coco.json \
        --videos-dir /workspace/videos \
        --checkpoint /workspace/pretrained_weights/wasb_basketball_best.pth.tar \
        --config /workspace/src/configs/model/wasb.yaml \
        --output-csv /workspace/eval_results.csv
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_ball import load_model, preprocess_frame  # noqa: E402
from heatmap_decode import get_ranked_detections, apply_mask_rects  # noqa: E402
from tiling import compute_tile_grid, clip_rect_to_tile  # noqa: E402

BALL_CATEGORY_NAME = "Ball"

# Candidate thresholds for the final sweep. Includes both the "noise floor"
# range observed in the earlier test (0.01-0.05) and the official threshold
# (0.5), for a complete picture.
CANDIDATE_THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

FRAME_NAME_RE = re.compile(r"^(?P<video>.+)_frame_(?P<num>\d+)")


def parse_coco_annotations(coco_path):
    """
    Returns a list of dicts, one per annotated image:
        {
          'video': 'out2',
          'frame_num': 2,               # as in the file name (1-indexed)
          'width': 3840, 'height': 2160,
          'gt_xy': (x, y) or None,      # ball bbox center, None if absent
          'gt_diag': float or None,     # bbox diagonal, for adaptive tolerance
        }
    Images whose name doesn't follow the expected pattern
    '<video>_frame_NNNN...' are silently skipped.
    """
    with open(coco_path, "r") as f:
        coco = json.load(f)

    ball_cat_id = None
    for cat in coco["categories"]:
        if cat["name"] == BALL_CATEGORY_NAME:
            ball_cat_id = cat["id"]
            break
    if ball_cat_id is None:
        raise ValueError(f"Category '{BALL_CATEGORY_NAME}' not found in the COCO file")

    # One ball annotation per image_id (the dataset has at most 1 ball/frame)
    ball_ann_by_image = {}
    for ann in coco["annotations"]:
        if ann["category_id"] != ball_cat_id:
            continue
        ball_ann_by_image[ann["image_id"]] = ann

    entries = []
    skipped = 0
    for img in coco["images"]:
        m = FRAME_NAME_RE.match(img["file_name"])
        if not m:
            skipped += 1
            continue

        ann = ball_ann_by_image.get(img["id"])
        if ann is not None:
            x, y, w, h = ann["bbox"]
            gt_xy = (x + w / 2.0, y + h / 2.0)
            gt_diag = float(np.hypot(w, h))
        else:
            gt_xy = None
            gt_diag = None

        entries.append({
            "video": m.group("video"),
            "frame_num": int(m.group("num")),
            "width": img["width"],
            "height": img["height"],
            "gt_xy": gt_xy,
            "gt_diag": gt_diag,
            "file_name": img["file_name"],
        })

    if skipped:
        print(f"[WARNING] {skipped} images skipped: name doesn't match '<video>_frame_NNNN'")

    return entries


def run_inference_for_video(video_path, video_name, targets_by_frame, model, model_cfg, device,
                             mask_rects=None):
    """
    Iterates the video sequentially (no seeking, to avoid decoding
    imprecision), keeps a buffer of `frames_in` frames, and every time the
    buffer's center index matches one of the requested indices runs
    inference. Returns dict {annotated_frame_num: raw_heatmap_np}.

    At the video's edges (first/last frame) the window would fall out of
    range: in that case the target index is "clamped" by duplicating the
    first or last available frame, a reasonable approximation given how
    few edge frames there are per video.
    """
    frames_in = model_cfg["frames_in"]
    inp_width = model_cfg["inp_width"]
    inp_height = model_cfg["inp_height"]
    out_scale_key = model_cfg["out_scales"][0]
    center_idx = frames_in // 2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # The sliding window's center can never coincide with the video's
    # first or last frame (it needs frames_in-1-center_idx frames before
    # it and center_idx frames after it): we clamp the target to this
    # valid range, not to [0, total_frames-1], otherwise edge frames (e.g.
    # the first annotated frame of each video) would silently vanish.
    min_center_idx = frames_in - 1 - center_idx
    max_center_idx = total_frames - 1 - center_idx

    # target video-frame-index (0-based, clamped) -> annotated frame_num
    target_idx_to_frame_num = {}
    for frame_num, video_idx in targets_by_frame.items():
        clamped = min(max(video_idx, min_center_idx), max_center_idx)
        target_idx_to_frame_num[clamped] = frame_num

    max_needed_idx = max(target_idx_to_frame_num.keys()) + (frames_in - 1 - center_idx)

    results = {}
    tensor_buffer = []
    scale_info = None
    frame_idx = 0

    while frame_idx <= max_needed_idx:
        ret, frame_bgr = cap.read()
        if not ret:
            print(f"[WARNING] {video_name}: video ended earlier than expected "
                  f"(frame_idx={frame_idx}, expected up to {max_needed_idx})")
            break

        tensor, scale_info = preprocess_frame(frame_bgr, inp_width, inp_height)
        tensor_buffer.append(tensor)
        if len(tensor_buffer) > frames_in:
            tensor_buffer.pop(0)

        if len(tensor_buffer) == frames_in:
            center_video_idx = frame_idx - (frames_in - 1 - center_idx)
            if center_video_idx in target_idx_to_frame_num:
                input_tensor = torch.cat(tensor_buffer, dim=0).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_tensor)
                heatmaps = torch.sigmoid(output[out_scale_key][0])  # (frames_out, H, W)
                heatmap_np = heatmaps[center_idx].detach().cpu().numpy()

                if mask_rects:
                    scale_x, scale_y = scale_info
                    apply_mask_rects(heatmap_np, mask_rects, scale_x, scale_y)

                frame_num = target_idx_to_frame_num[center_video_idx]
                results[frame_num] = heatmap_np

        frame_idx += 1

    cap.release()
    return results, scale_info


def run_inference_for_video_tiled(video_path, video_name, targets_by_frame, model, model_cfg, device,
                                   tile_boxes, mask_rects=None):
    """
    Same as run_inference_for_video, but runs inference independently on
    each tile in `tile_boxes` (list of (x0,y0,x1,y1) in original-frame
    pixels) instead of the whole frame, to recover effective resolution
    on wide shots where the ball is too small after a single full-frame
    resize (see tiling.py). N times slower per frame (N = number of
    tiles), since each tile needs its own forward pass.

    Returns dict {frame_num: [tile_record, ...]}, where each tile_record
    is {'heatmap', 'x0', 'y0', 'scale_x', 'scale_y'} -- consumed by
    heatmap_decode.get_ranked_detections, which merges tiles back into
    original-image coordinates.
    """
    frames_in = model_cfg["frames_in"]
    inp_width = model_cfg["inp_width"]
    inp_height = model_cfg["inp_height"]
    out_scale_key = model_cfg["out_scales"][0]
    center_idx = frames_in // 2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    min_center_idx = frames_in - 1 - center_idx
    max_center_idx = total_frames - 1 - center_idx

    target_idx_to_frame_num = {}
    for frame_num, video_idx in targets_by_frame.items():
        clamped = min(max(video_idx, min_center_idx), max_center_idx)
        target_idx_to_frame_num[clamped] = frame_num

    max_needed_idx = max(target_idx_to_frame_num.keys()) + (frames_in - 1 - center_idx)

    # --mask-rect regions are given in full-frame coordinates; remap them
    # onto each tile once up front (they're static across frames).
    tile_mask_rects = []
    for tile_box in tile_boxes:
        local_rects = []
        for rect in (mask_rects or []):
            clipped = clip_rect_to_tile(rect, tile_box)
            if clipped is not None:
                local_rects.append(clipped)
        tile_mask_rects.append(local_rects)

    results = {}
    tensor_buffers = [[] for _ in tile_boxes]  # one sliding-window buffer per tile
    tile_scale_info = [None] * len(tile_boxes)
    frame_idx = 0

    while frame_idx <= max_needed_idx:
        ret, frame_bgr = cap.read()
        if not ret:
            print(f"[WARNING] {video_name}: video ended earlier than expected "
                  f"(frame_idx={frame_idx}, expected up to {max_needed_idx})")
            break

        for ti, (x0, y0, x1, y1) in enumerate(tile_boxes):
            crop = frame_bgr[y0:y1, x0:x1]
            tensor, scale_info = preprocess_frame(crop, inp_width, inp_height)
            tensor_buffers[ti].append(tensor)
            if len(tensor_buffers[ti]) > frames_in:
                tensor_buffers[ti].pop(0)
            tile_scale_info[ti] = scale_info

        if len(tensor_buffers[0]) == frames_in:
            center_video_idx = frame_idx - (frames_in - 1 - center_idx)
            if center_video_idx in target_idx_to_frame_num:
                frame_num = target_idx_to_frame_num[center_video_idx]
                tile_records = []
                for ti, (x0, y0, x1, y1) in enumerate(tile_boxes):
                    input_tensor = torch.cat(tensor_buffers[ti], dim=0).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = model(input_tensor)
                    heatmaps = torch.sigmoid(output[out_scale_key][0])
                    heatmap_np = heatmaps[center_idx].detach().cpu().numpy()

                    sx, sy = tile_scale_info[ti]
                    if tile_mask_rects[ti]:
                        apply_mask_rects(heatmap_np, tile_mask_rects[ti], sx, sy)

                    tile_records.append({
                        "heatmap": heatmap_np, "x0": x0, "y0": y0,
                        "scale_x": sx, "scale_y": sy,
                    })
                results[frame_num] = tile_records

        frame_idx += 1

    cap.release()
    return results


def compute_confusion(records, thresh, tol_factor, tol_min_px):
    """
    Computes TP/TN/FP1/FP2/FN (+ accuracy, RMSE, and our own oracle_tp
    diagnostic, see main sweep) over any subset of records (the whole
    dataset, or a single video), at a given threshold. Factored out of
    the sweep loop so it can be reused for the per-video breakdown too.

    Matches the convention in src/utils/evaluator.py (the repo's own
    reference evaluator, inherited from the TrackNet/TrackNetV2 lineage
    WASB builds on) exactly, so results are directly comparable to
    numbers produced by that evaluator:
        FP1 = ball visible (GT) but detected at the WRONG position
        FP2 = ball NOT visible (GT) but the model detects something anyway
        Precision = TP / (TP + FP1 + FP2)
        Recall    = TP / (TP + FN)             -- NOTE: FP1 is excluded
                    from the denominator on purpose (their design: a
                    wrong-position detection only penalizes precision,
                    not recall -- recall measures "did the model attempt
                    a detection", not "was it accurate")
        Accuracy  = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
        RMSE      = sqrt(mean squared error), computed over every frame
                    where GT and a prediction both exist (TP and FP1
                    frames), regardless of whether the position was
                    within tolerance
    """
    tp = fn = fp1 = fp2 = tn = oracle_tp = 0
    n_pos = 0
    squared_errors = []
    for rec in records:
        detections = get_ranked_detections(rec, thresh)
        pred_xy = detections[0][0] if detections else None

        if rec["gt_xy"] is not None:
            n_pos += 1
            tol = max(tol_factor * rec["gt_diag"], tol_min_px)

            if any(
                np.hypot(xy[0] - rec["gt_xy"][0], xy[1] - rec["gt_xy"][1]) <= tol
                for xy, _, _ in detections
            ):
                oracle_tp += 1

            if pred_xy is None:
                fn += 1
            else:
                dist = float(np.hypot(pred_xy[0] - rec["gt_xy"][0], pred_xy[1] - rec["gt_xy"][1]))
                squared_errors.append(dist ** 2)
                if dist <= tol:
                    tp += 1
                else:
                    fp1 += 1
        else:
            if pred_xy is None:
                tn += 1
            else:
                fp2 += 1

    precision = tp / (tp + fp1 + fp2) if (tp + fp1 + fp2) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp1 + fp2 + fn) if (tp + tn + fp1 + fp2 + fn) > 0 else 0.0
    rmse = float(np.sqrt(np.mean(squared_errors))) if squared_errors else float("nan")
    recall_oracle = oracle_tp / n_pos if n_pos > 0 else 0.0

    return {
        "threshold": thresh, "tp": tp, "tn": tn, "fp1": fp1, "fp2": fp2, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
        "rmse": rmse, "recall_oracle": recall_oracle,
    }


def save_overlays(all_records, threshold, images_dir, out_dir, tol_factor, tol_min_px, max_width=1280):
    """
    For each annotated frame, draws on the original image:
      - green box  = ground truth bbox (if the ball is present)
      - red point  = predicted top-blob position (if above threshold)
    and saves it to `out_dir`, with the file name prefixed by the
    category (TP/FN/FP1/FP2/TN) so it's easy to browse them by error type.
    Resized to `max_width` to stay light to inspect.
    """
    os.makedirs(out_dir, exist_ok=True)
    counts = defaultdict(int)

    for rec in all_records:
        img_path = os.path.join(images_dir, rec["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Overlay: could not read {img_path}, skipping")
            continue

        detections = get_ranked_detections(rec, threshold)
        pred_xy, pred_peak, _ = detections[0] if detections else (None, None, None)

        if rec["gt_xy"] is not None:
            tol = max(tol_factor * rec["gt_diag"], tol_min_px)
            gx, gy = rec["gt_xy"]
            half = rec["gt_diag"] / 2.0
            cv2.rectangle(img, (int(gx - half), int(gy - half)), (int(gx + half), int(gy + half)),
                          (0, 200, 0), 3)
            if pred_xy is None:
                category = "FN"
            else:
                dist = float(np.hypot(pred_xy[0] - gx, pred_xy[1] - gy))
                category = "TP" if dist <= tol else "FP1"
        else:
            category = "FP2" if pred_xy is not None else "TN"

        if pred_xy is not None:
            px, py = int(pred_xy[0]), int(pred_xy[1])
            cv2.circle(img, (px, py), 14, (0, 0, 255), 3)
            cv2.putText(img, f"peak={pred_peak:.3f}", (px + 18, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(img, f"{category}  {rec['video']}_frame_{rec['frame_num']:04d}  thr={threshold}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        scale = min(1.0, max_width / img.shape[1])
        if scale < 1.0:
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

        counts[category] += 1
        out_name = f"{category}_{rec['video']}_{rec['frame_num']:04d}.jpg"
        cv2.imwrite(os.path.join(out_dir, out_name), img)

    print(f"[INFO] Overlays saved to {out_dir}: " +
          ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, help="Path to _annotations.coco.json")
    parser.add_argument("--videos-dir", required=True, help="Folder with the source .mp4 videos")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fps-stride", type=int, default=5,
                         help="Ratio video_fps / annotation_fps (default 5: video 25fps, annotations 5fps)")
    parser.add_argument("--frame-offset", type=int, default=2,
                         help="Constant offset (in video frames) between the 5fps sampling and the "
                              "real index in the 25fps video: video_idx = (frame_num-1)*stride + offset. "
                              "Verified empirically by comparing the annotated images against the video "
                              "frames pixel-by-pixel (see check_frame_offset.py); offset=+2 across all 3 "
                              "test videos.")
    parser.add_argument("--tol-factor", type=float, default=0.5,
                         help="Match tolerance = tol-factor * GT bbox diagonal (default 0.5, i.e. the "
                              "predicted center must fall within half the bbox diagonal)")
    parser.add_argument("--tol-min-px", type=float, default=10.0,
                         help="Minimum tolerance in pixels (floor), for very small bboxes")
    parser.add_argument("--output-csv", default=None, help="Path where to save the threshold->metrics table")
    parser.add_argument("--exclude-videos", nargs="*", default=[],
                         help="Video names (e.g. out13) to fully exclude from the evaluation, "
                              "to isolate performance on the other views")
    parser.add_argument("--mask-rect", nargs="*", default=[],
                         help="Regions to zero out in the heatmap before decoding, format "
                              "'video:x0,y0,x1,y1' (pixel coordinates in the original image, "
                              "repeatable). Useful for masking fixed graphical elements overlaid "
                              "on the video (e.g. a logo/clock in a corner, on a static camera) "
                              "that the model mistakes for the ball. Example: "
                              "--mask-rect out13:3750,540,3840,660")
    parser.add_argument("--images-dir", default=None,
                         help="Folder with the annotated images (for the overlays). Default: "
                              "the folder containing --annotations")
    parser.add_argument("--overlay-dir", default=None,
                         help="If set, saves to this folder one image per annotated frame with "
                              "the GT (green) and the top-blob prediction (red), classified as "
                              "TP/FN/FP2/FP1/TN. Useful for visual inspection.")
    parser.add_argument("--overlay-threshold", type=float, default=None,
                         help="Confidence threshold to use for the overlay (default: the best "
                              "F1 threshold found in the sweep)")
    parser.add_argument("--tile-videos", nargs="*", default=[],
                         help="Video names (e.g. out13) to run with tiled inference instead of a "
                              "single full-frame resize: splits each frame into an n x n grid of "
                              "overlapping crops and runs the model on each independently, to "
                              "recover effective resolution when the ball becomes too small after "
                              "the full-frame resize (see tiling.py). N times slower per frame. "
                              "Merged detections are reported in original-image coordinates.")
    parser.add_argument("--tile-n", type=int, default=2,
                         help="Tile grid size (n x n tiles) for videos listed in --tile-videos")
    parser.add_argument("--tile-overlap", type=float, default=0.1,
                         help="Fractional overlap between adjacent tiles (default 0.1 = 10%% wider/"
                              "taller than a plain 1/n split), to avoid splitting the ball across "
                              "two tiles at a grid boundary. Should be comfortably larger than the "
                              "expected ball size relative to the tile size.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.images_dir is None:
        args.images_dir = os.path.dirname(os.path.abspath(args.annotations))

    mask_rects_by_video = defaultdict(list)
    for spec in args.mask_rect:
        video_name, coords = spec.split(":")
        x0, y0, x1, y1 = (float(v) for v in coords.split(","))
        mask_rects_by_video[video_name].append((x0, y0, x1, y1))
    if mask_rects_by_video:
        print(f"[INFO] Active ROI masks: {dict(mask_rects_by_video)}")

    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    model, model_cfg = load_model(args.config, args.checkpoint, device)
    print(f"[INFO] frames_in={model_cfg['frames_in']}, "
          f"input size={model_cfg['inp_width']}x{model_cfg['inp_height']}")

    entries = parse_coco_annotations(args.annotations)

    if args.exclude_videos:
        excluded = set(args.exclude_videos)
        before = len(entries)
        entries = [e for e in entries if e["video"] not in excluded]
        print(f"[INFO] Excluded videos {sorted(excluded)}: {before - len(entries)} frames removed")

    n_pos = sum(1 for e in entries if e["gt_xy"] is not None)
    n_neg = len(entries) - n_pos
    print(f"[INFO] {len(entries)} annotated frames total ({n_pos} with ball, {n_neg} without)")

    by_video = defaultdict(list)
    for e in entries:
        by_video[e["video"]].append(e)

    # For each annotated frame: either a single full-frame heatmap
    # ('heatmap'/'scale_x'/'scale_y') or a list of per-tile heatmaps
    # ('tiles', see run_inference_for_video_tiled) + GT info. Both shapes
    # are handled transparently downstream by get_ranked_detections.
    all_records = []
    tile_videos = set(args.tile_videos)

    for video_name, video_entries in sorted(by_video.items()):
        video_path = os.path.join(args.videos_dir, f"{video_name}.mp4")
        if not os.path.exists(video_path):
            print(f"[WARNING] Video not found, skipping: {video_path}")
            continue

        targets_by_frame = {
            e["frame_num"]: (e["frame_num"] - 1) * args.fps_stride + args.frame_offset
            for e in video_entries
        }

        if video_name in tile_videos:
            frame_w, frame_h = video_entries[0]["width"], video_entries[0]["height"]
            tile_boxes = compute_tile_grid(frame_w, frame_h, n=args.tile_n, overlap_frac=args.tile_overlap)
            print(f"[INFO] {video_name}: {len(targets_by_frame)} frames to evaluate (TILED, "
                  f"{args.tile_n}x{args.tile_n}={len(tile_boxes)} tiles), from {video_path}")

            tiles_by_frame = run_inference_for_video_tiled(
                video_path, video_name, targets_by_frame, model, model_cfg, device,
                tile_boxes, mask_rects=mask_rects_by_video.get(video_name),
            )

            for e in video_entries:
                tile_records = tiles_by_frame.get(e["frame_num"])
                if tile_records is None:
                    continue  # video ended before reaching this frame
                all_records.append({
                    "video": video_name,
                    "frame_num": e["frame_num"],
                    "gt_xy": e["gt_xy"],
                    "gt_diag": e["gt_diag"],
                    "tiles": tile_records,
                    "file_name": e["file_name"],
                })
        else:
            print(f"[INFO] {video_name}: {len(targets_by_frame)} frames to evaluate, "
                  f"from {video_path}")

            heatmaps_by_frame, scale_info = run_inference_for_video(
                video_path, video_name, targets_by_frame, model, model_cfg, device,
                mask_rects=mask_rects_by_video.get(video_name),
            )
            scale_x, scale_y = scale_info

            for e in video_entries:
                hm = heatmaps_by_frame.get(e["frame_num"])
                if hm is None:
                    continue  # video ended before reaching this frame
                all_records.append({
                    "video": video_name,
                    "frame_num": e["frame_num"],
                    "gt_xy": e["gt_xy"],
                    "gt_diag": e["gt_diag"],
                    "heatmap": hm,
                    "scale_x": scale_x,
                    "scale_y": scale_y,
                    "file_name": e["file_name"],
                })

    print(f"[INFO] Inference completed on {len(all_records)}/{len(entries)} annotated frames")

    # ------------------------------------------------------------------
    # Threshold sweep: for each one, re-decode the blobs (connected
    # components depend on the threshold) and compute the TP/TN/FP1/FP2/FN
    # metric.
    # ------------------------------------------------------------------
    # Besides the "official" metric (only the highest-peak blob counts as
    # the prediction), we also compute an "oracle" recall: is the ball
    # present AMONG the detected blobs (at any rank), even if it's not the
    # one with the highest peak? If the oracle is much higher than the
    # normal recall, it means the model often *does* see the ball but gets
    # distracted by other objects (hoop, reflections, court lines) with
    # equal or higher confidence -- a ranking/ambiguity problem, not a
    # model sensitivity problem. Useful to figure out whether a tracker
    # (temporal consistency) is needed instead of a per-frame argmax.
    n_pos = sum(1 for r in all_records if r["gt_xy"] is not None)

    print("\n[RESULTS] Threshold -> Precision / Recall / F1 / Accuracy / RMSE "
          "(matches src/utils/evaluator.py's convention)")
    print(f"{'thresh':>7} | {'TP':>4} {'TN':>4} {'FP1':>4} {'FP2':>4} {'FN':>4} | "
          f"{'precision':>9} {'recall':>7} {'f1':>6} {'accuracy':>8} {'rmse':>7} | {'recall_oracle':>13}")

    rows = []
    for thresh in CANDIDATE_THRESHOLDS:
        row = compute_confusion(all_records, thresh, args.tol_factor, args.tol_min_px)
        rows.append(row)
        print(f"{thresh:7.3f} | {row['tp']:4d} {row['tn']:4d} {row['fp1']:4d} {row['fp2']:4d} {row['fn']:4d} | "
              f"{row['precision']:9.3f} {row['recall']:7.3f} {row['f1']:6.3f} {row['accuracy']:8.3f} "
              f"{row['rmse']:7.1f} | {row['recall_oracle']:13.3f}")

    best = max(rows, key=lambda r: r["f1"])
    print(f"\n[BEST FOR F1] threshold={best['threshold']} "
          f"precision={best['precision']:.3f} recall={best['recall']:.3f} f1={best['f1']:.3f} "
          f"accuracy={best['accuracy']:.3f} rmse={best['rmse']:.1f}")

    # Per-video breakdown at the best threshold: if one video (e.g. a
    # wide-angle shot with a much smaller ball) is dragging down the
    # aggregate average, it shows up here right away.
    print(f"\n[PER-VIDEO BREAKDOWN] at threshold={best['threshold']}")
    print(f"{'video':>8} | {'TP':>4} {'TN':>4} {'FP1':>4} {'FP2':>4} {'FN':>4} | "
          f"{'precision':>9} {'recall':>7} {'f1':>6} {'accuracy':>8} {'rmse':>7} | {'recall_oracle':>13}")
    for video_name in sorted(by_video.keys()):
        video_records = [r for r in all_records if r["video"] == video_name]
        if not video_records:
            continue
        row = compute_confusion(video_records, best["threshold"], args.tol_factor, args.tol_min_px)
        print(f"{video_name:>8} | {row['tp']:4d} {row['tn']:4d} {row['fp1']:4d} {row['fp2']:4d} {row['fn']:4d} | "
              f"{row['precision']:9.3f} {row['recall']:7.3f} {row['f1']:6.3f} {row['accuracy']:8.3f} "
              f"{row['rmse']:7.1f} | {row['recall_oracle']:13.3f}")

    if args.output_csv:
        import csv
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] Table saved to: {args.output_csv}")

    if args.overlay_dir:
        overlay_thresh = args.overlay_threshold if args.overlay_threshold is not None else best["threshold"]
        print(f"\n[INFO] Generating overlays at threshold={overlay_thresh} in {args.overlay_dir}")
        save_overlays(all_records, overlay_thresh, args.images_dir, args.overlay_dir,
                      args.tol_factor, args.tol_min_px)


if __name__ == "__main__":
    main()
