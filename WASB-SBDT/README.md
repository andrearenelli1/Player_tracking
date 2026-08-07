# Ball tracking (2D) — WASB/HRNet fork

2D basketball tracking for the project's 2D-tracking step: runs the
**WASB** (HRNet) model on a fixed camera view and produces, per view, a
per-frame CSV of ball detections plus a performance evaluation against
annotated ground truth. Player tracking (YOLO) is handled elsewhere; this
repo only covers the ball.

Forked from [nttcom/WASB-SBDT](https://github.com/nttcom/WASB-SBDT)
(BMVC2023, see citation below) and trimmed down to just the WASB/HRNet
basketball checkpoint — the original repo's training framework, other
model architectures, and other-sport checkpoints were removed since this
fork only does zero-shot inference with one pretrained model. See
`claude_context.md` for the full history of how this was built and
validated (bugs found, tiling for wide shots, evaluation methodology).

This repo lives as a subfolder of the main project repo. The Docker
container is always launched from the **main repo root**, not from
`WASB-SBDT/` — that's what lets the tracker reach the shared `videos/`
and write into `tracking_results/` without leaving the mount.

## 1. Get the Docker image

There's no registry — "getting" the image means building it locally from
this repo's `Dockerfile`, plus downloading the one checkpoint it needs.

```bash
cd WASB-SBDT
docker build -t wasb-sbdt .                   # builds the image (needs nvidia-container-toolkit + GPU at run time, not at build time)
cd src && sh setup_scripts/setup_weights.sh   # downloads pretrained_weights/wasb_basketball_best.pth.tar (~6MB)
cd ../..                                      # back to the main repo root
```

Rebuild (`docker build -t wasb-sbdt .` again) any time `Dockerfile`
changes — an existing image does **not** pick up `Dockerfile` edits on
its own (bit us once in session 5: a stale image was missing `attrdict`
even though the `Dockerfile` had it).

## 2. Start the container

**First time**, or whenever you want a clean container:

```bash
cd <main repo root>
docker run -it --gpus all -v $(pwd):/workspace -w /workspace/WASB-SBDT/src wasb-sbdt
```

**Resuming later**: prefer restarting the *same* container over running a
new one, to keep anything installed by hand inside it (`docker run` always
starts from the image fresh; `docker start` reuses the existing
container's filesystem):

```bash
docker ps -a                        # find the container name/id (e.g. wasb-tracking)
docker start -ai wasb-tracking
```

Sanity-check the mount before trusting any output — `docker inspect
<container> --format '{{json .Mounts}}'` should show the main repo root
as the source, `/workspace` as the destination. A stale/wrong mount is
the single most common way this setup silently produces nothing new (see
`claude_context.md`).

All commands below run **inside the container**, from
`/workspace/WASB-SBDT/src` (the default workdir set by `-w` above).

## 3. Running it — the different cases

### Case A — full pipeline: track all 3 videos + evaluate against ground truth
The normal case; both scripts already have the right settings baked in
(tiling + mask for `out13`, defaults for `out2`/`out4`).

```bash
sh run_tracking.sh   # -> tracking_results/tracking_2d/ball_trajectories/*.csv
                      #    tracking_results/tracking_2d/evaluation/*_result.mp4
sh run_eval.sh        # -> tracking_results/tracking_2d/evaluation/eval_results.csv
```

### Case B — track a single video by hand
Use this to re-run just one view, or with different settings than the
scripts' defaults.

```bash
python3 track_ball.py \
    --video /workspace/videos/out4.mp4 \
    --checkpoint /workspace/WASB-SBDT/pretrained_weights/wasb_basketball_best.pth.tar \
    --config /workspace/WASB-SBDT/src/configs/model/wasb.yaml \
    --cam-id cam_1 \
    --csv-output /workspace/tracking_results/tracking_2d/ball_trajectories/out4_detections.csv \
    --output /workspace/tracking_results/tracking_2d/evaluation/out4_result.mp4
```

Camera-id convention (`--cam-id`, matching `tracking_players_2d.py`/
`track_ball_classic.py` in the main project):

| video | `--cam-id` | needs tiling? |
|---|---|---|
| `out2.mp4` | `cam_0` | no |
| `out4.mp4` | `cam_1` | no |
| `out13.mp4` | `cam_2` | **yes** — full-court shot, ball too small otherwise |

`out13` needs `--tile-n 3 --mask-rect 3700,500,3840,680` added to the
command above (see `claude_context.md` for why); `out2`/`out4` work with
plain defaults.

### Case C — evaluate only, against ground truth
If CSVs already exist and you just want fresh metrics (or want to sweep
different tolerances/thresholds), run `eval_wasb.py` directly — it
re-runs inference itself, it does not read the CSVs from Case A/B:

```bash
python3 eval_wasb.py \
    --annotations /workspace/annotations/_annotations.coco.json \
    --videos-dir /workspace/videos \
    --checkpoint /workspace/WASB-SBDT/pretrained_weights/wasb_basketball_best.pth.tar \
    --config /workspace/WASB-SBDT/src/configs/model/wasb.yaml \
    --tile-videos out13 --tile-n 3 --mask-rect out13:3700,500,3840,680 \
    --output-csv /workspace/tracking_results/tracking_2d/evaluation/eval_results.csv
```

Add `--overlay-dir /workspace/tracking_results/tracking_2d/evaluation/overlays`
to also save per-frame images with GT (green) vs prediction (red),
labeled TP/FN/FP2/FP1/TN, for visual inspection. Add `--exclude-videos
out13` to score only the close-up views.

### Case D — inspect confidence on a video without producing final output
Quick sanity check / threshold calibration on a video, no CSV or overlay
video written:

```bash
python3 track_ball.py \
    --video /workspace/videos/out2.mp4 \
    --checkpoint /workspace/WASB-SBDT/pretrained_weights/wasb_basketball_best.pth.tar \
    --config /workspace/WASB-SBDT/src/configs/model/wasb.yaml \
    --cam-id cam_0 \
    --debug-dir /workspace/tracking_results/tracking_2d/evaluation/debug_out2
```

Prints a threshold sweep + confidence stats, and saves the
highest/lowest-confidence frames to `--debug-dir` for a manual look.

## Scripts reference

- **`src/track_ball.py`** — runs the tracker on a full video (Cases B, D)
- **`src/eval_wasb.py`** — evaluates precision/recall/F1/accuracy/RMSE
  against annotated ground truth (Case C), matching the metric convention
  used by this model family (TrackNet/TrackNetV2/WASB papers)
- **`src/run_tracking.sh`**, **`src/run_eval.sh`** — wrappers for Case A
- **`src/check_frame_offset.py`** — one-off utility to verify the mapping
  between annotated-frame numbers and real video-frame indices, if
  annotations get re-exported at a different frame rate
- **`src/heatmap_decode.py`**, **`src/tiling.py`** — shared decoding/
  tiling logic used by `track_ball.py` and `eval_wasb.py`

## Citation

Model and pretrained weights from:
```
@inproceedings{tarashima2023wasb,
	title={Widely Applicable Strong Baseline for Sports Ball Detection and Tracking},
	author={Tarashima, Shuhei and Haq, Muhammad Abdul and Wang, Yushan and Tagawa, Norio},
	booktitle={BMVC},
	year={2023}
}
```
