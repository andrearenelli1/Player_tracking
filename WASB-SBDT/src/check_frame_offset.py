"""
Finds the constant offset between the frame number in the annotated
images' file names (`<video>_frame_NNNN...`, sampled at 5fps by Roboflow)
and the real frame index in the 25fps source video.

For a sample of annotated frames (`anchors` below), compares the
annotated image pixel-by-pixel against a window of candidate video frames
around the naive estimate `(NNNN-1)*5`, and reports which offset
minimizes the mean absolute difference. Sequential video reading (no
repeated `cap.set`/seeking, which is very slow on compressed 4K video).

Used to calibrate eval_wasb.py's `--frame-offset` parameter.
Standalone (host) script, not meant to run inside the Docker container --
just needs opencv/numpy. Uses paths relative to cwd, so run it from the
MAIN PROJECT repo root (where annotations/ and videos/ live, not
WASB-SBDT's own root -- session 5 deduped both out of WASB-SBDT/):
    python3 WASB-SBDT/src/check_frame_offset.py
"""
import cv2
import numpy as np
import json

with open('annotations/_annotations.coco.json') as f:
    coco = json.load(f)

by_name = {img['file_name']: img for img in coco['images']}


def find_anno(video, frame_num):
    prefix = f'{video}_frame_{frame_num:04d}_'
    for fn, img in by_name.items():
        if fn.startswith(prefix):
            return img
    return None


WINDOW = 10
anchors = {
    'out2': [2, 10, 30, 50, 70, 90, 100],
    'out4': [2, 10, 30, 50, 70, 90, 105],
    'out13': [2, 10, 30, 50, 70, 90, 105],
}

for video, frame_nums in anchors.items():
    # idx -> list of (frame_num, anno_img)
    idx_to_targets = {}
    max_idx = 0
    for fn in frame_nums:
        img_meta = find_anno(video, fn)
        if img_meta is None:
            print(f'{video} frame_num={fn}: not found in coco')
            continue
        anno_img = cv2.imread('annotations/' + img_meta['file_name'])
        base = (fn - 1) * 5
        for idx in range(max(0, base - WINDOW), base + WINDOW + 1):
            idx_to_targets.setdefault(idx, []).append((fn, base, anno_img))
        max_idx = max(max_idx, base + WINDOW)

    diffs = {}  # frame_num -> list of (idx, diff)
    cap = cv2.VideoCapture(f'videos/{video}.mp4')
    idx = 0
    while idx <= max_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in idx_to_targets:
            for fn, base, anno_img in idx_to_targets[idx]:
                d = float(np.mean(np.abs(frame.astype(np.int32) - anno_img.astype(np.int32))))
                diffs.setdefault(fn, []).append((idx, d))
        idx += 1
    cap.release()

    for fn in frame_nums:
        if fn not in diffs:
            continue
        base = (fn - 1) * 5
        best_idx, best_d = min(diffs[fn], key=lambda t: t[1])
        print(f'{video} frame_num={fn:4d} base={base:4d} best_idx={best_idx:4d} '
              f'offset={best_idx-base:+d} diff={best_d:.3f}')
