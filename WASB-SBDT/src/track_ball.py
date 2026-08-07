"""
2D ball tracking with the WASB (HRNet) model: runs inference over an
entire video and produces the two deliverables for the project's 2D
tracking step:
  - a per-frame CSV of ball detections (--csv-output), one row per
    detected frame (frames with no ball above threshold are omitted, not
    written with empty fields), columns `frame,cam_id,class_id,object_id,
    u,v,w,h` -- the same schema written by tracking_players_2d.py and
    track_ball_classic.py in the main project, so eval_2d.py can read all
    three tracking CSVs uniformly. `u,v` is the predicted ball center,
    `w,h` is a SYNTHETIC box size derived from the detected blob's own
    pixel extent (the model predicts a point, not a real box; see
    heatmap_decode.get_ranked_detections). `class_id` is always 0 and
    `object_id` always -1 (no per-instance tracking, matching
    track_ball_classic.py's convention -- there's only ever one ball).
  - an overlay video (--output), for visual sanity-checking

For performance evaluation against ground truth (precision/recall/F1/
accuracy/RMSE on the annotated frames), see eval_wasb.py instead -- this
script does NOT need annotations, it just runs the tracker.

Usage:
    python3 track_ball.py \
        --video /path/to/video.mp4 \
        --checkpoint /workspace/pretrained_weights/wasb_basketball_best.pth.tar \
        --config /workspace/src/configs/model/wasb.yaml \
        --cam-id cam_0 \
        --csv-output /path/to/detections.csv \
        --output /path/to/output_video.mp4 \
        --conf-thresh 0.02

Must be run INSIDE the docker container, from /workspace/src, so that the
project's relative imports (models, etc.) work correctly.
"""

import argparse
import csv
import sys
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from attrdict import AttrDict

# Import the model from the WASB-SBDT repo
# Assumes the script is launched from /workspace/src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_model  # noqa: E402
from heatmap_decode import get_ranked_detections, apply_mask_rects  # noqa: E402
from tiling import compute_tile_grid, clip_rect_to_tile  # noqa: E402


# ---------------------------------------------------------------------------
# Preprocessing: same normalization used in dataloaders/__init__.py
# ---------------------------------------------------------------------------
PREPROCESS = T.Compose([
    T.ToTensor(),  # HWC uint8 [0,255] -> CHW float [0,1]
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(config_path, checkpoint_path, device):
    """Builds the model from the config and loads the pretrained weights."""
    with open(config_path, "r") as f:
        model_cfg = yaml.safe_load(f)

    # HRNet's code (hrnet.py) accesses the config both with dict notation
    # (cfg['MODEL']['EXTRA']) and attribute notation (cfg.MODEL.EXTRA) in
    # different places. AttrDict supports both, which is why we use it
    # here instead of a plain Python dict.
    model_cfg_attr = AttrDict(model_cfg)
    cfg = {"model": model_cfg_attr}
    model = build_model(cfg)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Some checkpoints store the state_dict directly, others wrap it under
    # a key ('state_dict' or 'model'). Handle both cases.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Strip any 'module.' prefix (typical if the model was trained with
    # nn.DataParallel / DistributedDataParallel)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[new_key] = v

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        print(f"[WARNING] Missing keys in the checkpoint: {missing}")
    if unexpected:
        print(f"[WARNING] Unexpected keys in the checkpoint: {unexpected}")

    model.to(device)
    model.eval()

    return model, model_cfg


def preprocess_frame(frame_bgr, inp_width, inp_height):
    """
    Converts a BGR (OpenCV) frame into a tensor ready for the network.
    Also returns the scale factor to map the predicted coordinates back
    to the original resolution.
    """
    orig_h, orig_w = frame_bgr.shape[:2]

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (inp_width, inp_height))

    tensor = PREPROCESS(frame_resized)  # (3, inp_height, inp_width)

    scale_x = orig_w / inp_width
    scale_y = orig_h / inp_height

    return tensor, (scale_x, scale_y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--checkpoint", required=True, help="Path to the pretrained weights (.pth.tar)")
    parser.add_argument("--config", required=True, help="Path to the model's yaml config file (wasb.yaml)")
    parser.add_argument("--output", default=None, help="Path to the output video with overlay (optional)")
    parser.add_argument("--csv-output", default=None,
                         help="Path to a CSV with one row per DETECTED frame (no row when no ball is "
                              "found above threshold): frame,cam_id,class_id,object_id,u,v,w,h -- same "
                              "schema as tracking_players_2d.py/track_ball_classic.py, so eval_2d.py can "
                              "read all tracking CSVs uniformly. u,v is the predicted ball center, w,h is "
                              "a SYNTHETIC box size derived from the detected blob's own pixel extent "
                              "(the model predicts a point, not a real box; see "
                              "heatmap_decode.get_ranked_detections). Requires --cam-id.")
    parser.add_argument("--cam-id", default=None,
                         help="Camera id to stamp in the CSV's cam_id column (e.g. cam_0/cam_1/cam_2, "
                              "matching the CAMERAS convention in tracking_players_2d.py / "
                              "track_ball_classic.py: cam_0=out2, cam_1=out4, cam_2=out13). Required if "
                              "--csv-output is set.")
    parser.add_argument("--conf-thresh", type=float, default=0.02,
                         help="Confidence threshold on the heatmap [0,1] used to binarize it before "
                              "blob detection (see decode_blob_concomp in heatmap_decode.py). Default "
                              "0.02 is the value found to maximize F1 against ground truth in "
                              "eval_wasb.py on close-up shots (out2/out4) -- NOT the model's official "
                              "0.5, which is tuned for NBA broadcast footage and yields near-zero "
                              "recall on this domain.")
    parser.add_argument("--mask-rect", nargs="*", default=[],
                         help="Regions to zero out in the heatmap before decoding, format "
                              "'x0,y0,x1,y1' (pixel coordinates in the original image, repeatable). "
                              "Useful for masking fixed graphical elements overlaid on a static "
                              "camera (e.g. a logo/clock in a corner) that the model mistakes for "
                              "the ball -- see eval_wasb.py's diagnosis of the out13 video.")
    parser.add_argument("--tile-n", type=int, default=1,
                         help="Split each frame into an n x n grid of overlapping tiles and run "
                              "inference on each independently, merging detections back into "
                              "original-image coordinates (see tiling.py). n=1 (default) is a "
                              "plain single full-frame pass, identical to the old behavior. Use "
                              "e.g. --tile-n 2 to recover effective resolution on wide shots where "
                              "the ball becomes too small after a single full-frame resize (see "
                              "eval_wasb.py's diagnosis of the out13 video). n^2 times slower per frame.")
    parser.add_argument("--tile-overlap", type=float, default=0.1,
                         help="Fractional overlap between adjacent tiles when --tile-n > 1 "
                              "(default 0.1 = 10%% wider/taller than a plain 1/n split), to avoid "
                              "splitting the ball across two tiles at a grid boundary.")
    parser.add_argument("--debug-dir", default=None, help="Folder where to save the highest/lowest confidence frames (visual inspection)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.csv_output and not args.cam_id:
        parser.error("--cam-id is required when --csv-output is set (stamped into the CSV's cam_id "
                      "column, shared schema with the player/classic-ball tracking CSVs)")

    mask_rects = [tuple(float(v) for v in spec.split(",")) for spec in args.mask_rect]
    if mask_rects:
        print(f"[INFO] Active ROI masks: {mask_rects}")

    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    model, model_cfg = load_model(args.config, args.checkpoint, device)

    frames_in = model_cfg["frames_in"]
    inp_width = model_cfg["inp_width"]
    inp_height = model_cfg["inp_height"]
    out_scale_key = model_cfg["out_scales"][0]
    center_idx = frames_in // 2

    print(f"[INFO] frames_in={frames_in}, input size={inp_width}x{inp_height}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {orig_w}x{orig_h}, {fps:.1f} fps, {total_frames} frames")

    # n=1 gives back a single tile covering the whole frame, making the
    # tiled and non-tiled code paths below identical -- no special-casing
    # needed for the default (non-tiled) behavior.
    tile_boxes = compute_tile_grid(orig_w, orig_h, n=args.tile_n, overlap_frac=args.tile_overlap)
    if args.tile_n > 1:
        print(f"[INFO] Tiled inference: {args.tile_n}x{args.tile_n} = {len(tile_boxes)} tiles")

    # --mask-rect regions are given in full-frame coordinates; remap them
    # onto each tile once up front (they're static across frames).
    tile_mask_rects = []
    for tile_box in tile_boxes:
        local_rects = []
        for rect in mask_rects:
            clipped = clip_rect_to_tile(rect, tile_box)
            if clipped is not None:
                local_rects.append(clipped)
        tile_mask_rects.append(local_rects)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (orig_w, orig_h))

    # Circular buffer of the raw frames (BGR, original resolution), to
    # build the sliding window and to draw/save the display frame. One
    # preprocessed-tensor buffer per tile.
    raw_buffer = []
    tensor_buffers = [[] for _ in tile_boxes]
    tile_scale_info = [None] * len(tile_boxes)

    frame_idx = 0
    # list of (frame_idx, (x,y) or None, (w,h) or None, conf) -- one entry
    # per frame that had a full 3-frame window (i.e. every frame except
    # the very first/last of the video, where the sliding window has no
    # center yet/anymore).
    detections = []
    raw_frames_cache = []  # center frames corresponding to each entry of 'detections' (only if --debug-dir)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        raw_buffer.append(frame_bgr)
        if len(raw_buffer) > frames_in:
            raw_buffer.pop(0)

        for ti, (x0, y0, x1, y1) in enumerate(tile_boxes):
            crop = frame_bgr[y0:y1, x0:x1]
            tensor, scale_info = preprocess_frame(crop, inp_width, inp_height)
            tensor_buffers[ti].append(tensor)
            if len(tensor_buffers[ti]) > frames_in:
                tensor_buffers[ti].pop(0)
            tile_scale_info[ti] = scale_info

        if len(tensor_buffers[0]) == frames_in:
            tile_records = []
            max_conf = 0.0
            for ti, (x0, y0, x1, y1) in enumerate(tile_boxes):
                # Concatenate the tile's 3 frames on the channel axis: (3*frames_in, H, W)
                input_tensor = torch.cat(tensor_buffers[ti], dim=0).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)  # dict {scale: (1, frames_out, H, W)}

                heatmaps = torch.sigmoid(output[out_scale_key][0])  # raw logits need an explicit sigmoid
                heatmap_np = heatmaps[center_idx].detach().cpu().numpy()

                sx, sy = tile_scale_info[ti]
                if tile_mask_rects[ti]:
                    apply_mask_rects(heatmap_np, tile_mask_rects[ti], sx, sy)

                max_conf = max(max_conf, float(heatmap_np.max()))
                tile_records.append({"heatmap": heatmap_np, "x0": x0, "y0": y0, "scale_x": sx, "scale_y": sy})

            # Same blob-based decoding used for the validated metrics in
            # eval_wasb.py (connected components on the binarized heatmap,
            # ranked by peak value, merged across tiles into original-image
            # coordinates), so what gets drawn/exported here matches what
            # was measured against ground truth. "conf" is the raw peak
            # across all tiles (before thresholding, but after masking),
            # kept regardless of whether a blob survives the threshold,
            # for the sweep printed below.
            record = {"tiles": tile_records}
            dets = get_ranked_detections(record, args.conf_thresh)
            if dets:
                pos_for_display, _, wh_for_display = dets[0]
            else:
                pos_for_display, wh_for_display = None, None
            conf = max_conf

            # The frame this prediction refers to is the buffer's center
            # frame, i.e. frame_idx - (frames_in - 1 - center_idx)
            target_frame_idx = frame_idx - (frames_in - 1 - center_idx)
            detections.append((target_frame_idx, pos_for_display, wh_for_display, conf))

            if args.debug_dir:
                # Save a resized version (not the original 4K frame) to
                # avoid running out of RAM on long videos.
                debug_frame = raw_buffer[center_idx]
                debug_w = 960
                debug_scale = debug_w / debug_frame.shape[1]
                debug_h = int(debug_frame.shape[0] * debug_scale)
                debug_frame_small = cv2.resize(debug_frame, (debug_w, debug_h))
                raw_frames_cache.append((debug_frame_small, debug_scale))

            if writer is not None:
                display_frame = raw_buffer[center_idx].copy()
                if pos_for_display is not None:
                    x, y = int(pos_for_display[0]), int(pos_for_display[1])
                    cv2.circle(display_frame, (x, y), 8, (0, 0, 255), 2)
                    cv2.putText(display_frame, f"{conf:.3f}", (x + 12, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(display_frame, f"no ball (max={conf:.3f})", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                writer.write(display_frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"[INFO] Processed {frame_idx}/{total_frames} frames")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[INFO] Output video saved to: {args.output}")

    if args.csv_output:
        with open(args.csv_output, "w", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(["frame", "cam_id", "class_id", "object_id", "u", "v", "w", "h"])
            n_written = 0
            for f_idx, pos, wh, conf in detections:
                if pos is None:
                    continue
                x, y = pos
                w, h = wh
                writer_csv.writerow([f_idx, args.cam_id, 0, -1, f"{x:.2f}", f"{y:.2f}", f"{w:.2f}", f"{h:.2f}"])
                n_written += 1
        n_missing_edges = total_frames - len(detections)
        print(f"[INFO] Detections CSV saved to: {args.csv_output} ({n_written}/{len(detections)} "
              f"processed frames had a ball above threshold; {n_missing_edges} frame(s) at the very "
              f"start/end of the video have no row at all -- the sliding window needs a frame before "
              f"and after to have a center)")

    # Summary recall statistics at the threshold chosen on the command line
    n_total = len(detections)
    n_detected = sum(1 for _, pos, wh, conf in detections if pos is not None)
    recall = n_detected / n_total if n_total > 0 else 0.0
    print(f"[RESULT] Threshold={args.conf_thresh}: Frames processed: {n_total}, "
          f"ball detected: {n_detected} ({recall*100:.1f}%)")

    # Sweep of candidate thresholds, to calibrate the best value without
    # having to rerun the whole script every time. Note: this is just the
    # fraction of frames whose raw peak confidence clears each threshold,
    # not a true recall against ground truth (that requires annotations,
    # see eval_wasb.py) -- useful as a quick, unsupervised sanity check.
    print("\n[THRESHOLD SWEEP] Fraction of frames above each confidence threshold:")
    candidate_thresholds = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    all_confs = [conf for _, _, _, conf in detections]
    for t in candidate_thresholds:
        n_above = sum(1 for c in all_confs if c >= t)
        r = n_above / n_total if n_total > 0 else 0.0
        print(f"  threshold={t:<6} -> {r*100:5.1f}%  ({n_above}/{n_total} frames)")

    if all_confs:
        print(f"\n[CONFIDENCE STATISTICS] min={min(all_confs):.4f} "
              f"max={max(all_confs):.4f} mean={np.mean(all_confs):.4f} "
              f"median={np.median(all_confs):.4f}")

    # Save to disk the highest- and lowest-confidence frames, for manual
    # visual inspection: they tell us whether the confidence signal is
    # reliable (peaks = ball clearly visible and correctly localized) or
    # whether even the peaks are placed at random.
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

        # Sort by descending confidence
        sorted_detections = sorted(
            zip(detections, raw_frames_cache), key=lambda d: d[0][3], reverse=True
        )

        top_n = 10
        print(f"\n[DEBUG] Saving the top-{top_n} frames by confidence to {args.debug_dir}")
        for rank, ((f_idx, pos, wh, conf), (frame_img, dscale)) in enumerate(sorted_detections[:top_n]):
            img = frame_img.copy()
            if pos is not None:
                x, y = int(pos[0] * dscale), int(pos[1] * dscale)
                cv2.circle(img, (x, y), 10, (0, 255, 0), 3)
            cv2.putText(img, f"frame={f_idx} conf={conf:.4f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            out_path = os.path.join(args.debug_dir, f"top_{rank:02d}_frame{f_idx}_conf{conf:.4f}.jpg")
            cv2.imwrite(out_path, img)

        bottom_n = 5
        print(f"[DEBUG] Also saving {bottom_n} low-confidence frames (background noise) for comparison")
        for rank, ((f_idx, pos, wh, conf), (frame_img, dscale)) in enumerate(sorted_detections[-bottom_n:]):
            img = frame_img.copy()
            cv2.putText(img, f"frame={f_idx} conf={conf:.4f} (low)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            out_path = os.path.join(args.debug_dir, f"low_{rank:02d}_frame{f_idx}_conf{conf:.4f}.jpg")
            cv2.imwrite(out_path, img)


if __name__ == "__main__":
    main()
