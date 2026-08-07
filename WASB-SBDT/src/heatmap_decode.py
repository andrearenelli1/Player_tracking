"""
Shared heatmap decoding utilities, used by both track_ball.py and
eval_wasb.py so the two scripts stay consistent (the visual overlay/CSV
in track_ball.py should match what eval_wasb.py's metrics were computed
against).
"""

import cv2
import numpy as np


def decode_blob_concomp(heatmap_np, score_threshold, use_hm_weight=True):
    """
    Based on detectors/postprocessor.py:_detect_blob_concomp, with one
    change: the original ranks blobs by the SUM of the region's values,
    here we rank by PEAK value instead. At the low thresholds required for
    an uncalibrated domain, the sum wrongly favors large-but-weak regions
    (diffuse background noise) over a small-but-very-confident region
    where the ball really is, just because they cover more pixels. The
    peak is invariant to the region's area.

    Binarizes the heatmap above the threshold, finds the connected
    components and for each one computes the centroid (weighted by the
    heatmap values if use_hm_weight=True) and pixel extent (bounding box
    of the blob's pixels -- used downstream as a synthetic detection
    size, since the model itself only predicts a point). Returns the
    list of (xy, peak, (w, h)) sorted by descending peak -- the first
    element is the "best" blob, consistent with the single-ball-per-frame
    assumption. xy, w, h are all in heatmap-pixel space here; scaling to
    original-image coordinates happens in get_ranked_detections.
    """
    if float(heatmap_np.max()) <= score_threshold:
        return []

    _, hm_th = cv2.threshold(heatmap_np, score_threshold, 1, cv2.THRESH_BINARY)
    n_labels, labels = cv2.connectedComponents(hm_th.astype(np.uint8))

    blobs = []
    for m in range(1, n_labels):
        ys, xs = np.where(labels == m)
        ws = heatmap_np[ys, xs]
        if use_hm_weight:
            x = float(np.sum(xs * ws) / np.sum(ws))
            y = float(np.sum(ys * ws) / np.sum(ws))
        else:
            x = float(np.mean(xs))
            y = float(np.mean(ys))
        peak = float(ws.max())
        w = float(xs.max() - xs.min() + 1)
        h = float(ys.max() - ys.min() + 1)
        blobs.append(((x, y), peak, (w, h)))

    blobs.sort(key=lambda b: b[1], reverse=True)
    return blobs


def get_ranked_detections(record, score_threshold, use_hm_weight=True):
    """
    Unified detection interface: works whether `record` holds a single
    full-frame heatmap (keys 'heatmap', 'scale_x', 'scale_y') or a list of
    per-tile heatmaps (key 'tiles', each a dict with 'heatmap', 'x0',
    'y0', 'scale_x', 'scale_y' -- see tiling.py). Always returns a list of
    ((x, y), peak, (w, h)) in ORIGINAL IMAGE PIXEL coordinates, merged
    across tiles if any, sorted by descending peak. Callers should just
    take detections[0] as "the" prediction, consistent with the
    single-ball-per-frame assumption, without needing to know whether
    tiling was used. (w, h) is a synthetic detection size derived from
    the blob's own pixel extent, centered on (x, y) -- not a real
    bounding box (the model predicts a point), but a defensible size
    estimate for downstream code that expects a box (e.g. a CSV export
    matching a player-detector's format).
    """
    detections = []

    tiles = record.get("tiles")
    if tiles:
        for tile in tiles:
            blobs = decode_blob_concomp(tile["heatmap"], score_threshold, use_hm_weight)
            x0, sx, sy = tile["x0"], tile["scale_x"], tile["scale_y"]
            y0 = tile["y0"]
            for (bx, by), peak, (bw, bh) in blobs:
                detections.append(((x0 + bx * sx, y0 + by * sy), peak, (bw * sx, bh * sy)))
    else:
        blobs = decode_blob_concomp(record["heatmap"], score_threshold, use_hm_weight)
        sx, sy = record["scale_x"], record["scale_y"]
        for (bx, by), peak, (bw, bh) in blobs:
            detections.append(((bx * sx, by * sy), peak, (bw * sx, bh * sy)))

    detections.sort(key=lambda d: d[1], reverse=True)
    return detections


def apply_mask_rects(heatmap_np, rects_orig, scale_x, scale_y):
    """
    Zeroes out in-place the regions of `heatmap_np` (network space, e.g.
    288x512) corresponding to the rectangles `rects_orig` (x0,y0,x1,y1)
    expressed in original-image pixels. Used to mask fixed graphical
    elements overlaid on the video (e.g. a logo/clock in a corner) that a
    static camera always shows in the same spot and that the model
    mistakes for the ball.
    """
    hm_h, hm_w = heatmap_np.shape
    for x0, y0, x1, y1 in rects_orig:
        hx0 = max(0, int(x0 / scale_x))
        hx1 = min(hm_w, int(np.ceil(x1 / scale_x)))
        hy0 = max(0, int(y0 / scale_y))
        hy1 = min(hm_h, int(np.ceil(y1 / scale_y)))
        heatmap_np[hy0:hy1, hx0:hx1] = 0.0
