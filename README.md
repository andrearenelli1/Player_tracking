# Player Tracking — basketball, multi-camera (2D + 3D)

Tracks players and the ball across 3 fixed camera views (`out2`, `out4`,
`out13`), each processed independently in 2D, then fused into a single 3D
reconstruction via triangulation. Player detection uses YOLO, ball
detection has two interchangeable options (a classic CV baseline and a
pretrained deep model, WASB/HRNet, run in Docker — see below).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Player tracking (`tracking_players_2d.py`) hardcodes `.to('cuda')` — needs
a CUDA-capable GPU + a matching torch build. Ball tracking's WASB option
runs in its own Docker container (separate environment, see the dedicated
section below); its classic-CV option needs nothing extra beyond the venv.

The 3 source videos (`out2.mp4`, `out4.mp4`, `out13.mp4`) are not in this
repo (`.gitignore`'d — too large). Download them from this Drive folder
and place them in `videos/`:
<https://drive.google.com/drive/folders/1BMnV2TBXa9lhLrVAurMdJCnnp1KS1A5N>

## Repo layout

- `videos/` — the 3 source `.mp4`s (`out2`, `out4`, `out13`), 25fps, 4K
- `annotations/` — ground truth (Roboflow COCO export, 309 annotated
  frames at 5fps) + the annotated images themselves
- `calibrated_parameters/` — per-camera intrinsics (`cam_2.json`,
  `cam_4.json`, `cam_13.json` — **named after the video**, e.g.
  `cam_2.json` is `out2`'s calibration, not to be confused with the
  `cam_0`/`cam_1`/`cam_2` tracking ids below — see the table further down)
- `src/` — pipeline scripts (this README's Pipeline section)
- `WASB-SBDT/` — the WASB/HRNet ball detector (own Docker setup, see below)
- `tracking_results/tracking_2d/trajectories/` — player 2D positions
- `tracking_results/tracking_2d/ball_trajectories/` — ball 2D positions
- `tracking_results/tracking_2d/evaluation/` — 2D visual/metric outputs
- `tracking_results/tracking_3d/` — 3D reconstruction + evaluation

Camera-id convention used throughout `src/` and `WASB-SBDT/` (three
different names for the same 3 views — worth keeping straight):

| video | calibration file | tracking `cam_id` |
|---|---|---|
| `out2.mp4` | `calibrated_parameters/cam_2.json` | `cam_0` |
| `out4.mp4` | `calibrated_parameters/cam_4.json` | `cam_1` |
| `out13.mp4` | `calibrated_parameters/cam_13.json` | `cam_2` |

## Pipeline

Run from the repo root, inside the venv, in this order:

**1. Player 2D tracking (YOLO)**
```bash
python3 src/tracking_players_2d.py
```
→ `tracking_results/tracking_2d/trajectories/2d_positions{0,1,2}.csv`
(one file per camera, columns `frame,cam_id,class_id,object_id,u,v,w,h`).

**2. Ball 2D tracking** — pick one:
- **Classic CV baseline** (background subtraction, no GPU, weaker recall):
  ```bash
  python3 src/track_ball_classic.py
  ```
  → `tracking_results/tracking_2d/ball_trajectories/ball_tracking_classic_out{2,4,13}.csv`,
  same schema as step 1.
- **WASB/HRNet** (pretrained deep model, better recall/precision,
  needs Docker + GPU — see the dedicated section below):
  ```bash
  sh WASB-SBDT/src/run_tracking.sh   # from inside the container
  ```
  → `tracking_results/tracking_2d/ball_trajectories/ball_tracking_wasb_out{2,4,13}.csv`
  + overlay videos in `tracking_results/tracking_2d/evaluation/`.

  The evaluation and visualization scripts now intentionally use the
  explicit classic/WASB file names above; there is no separate evaluation
  script for WASB anymore.

**3. 2D evaluation** (players + classic-ball + WASB-ball, IoU-matched against ground truth):
```bash
python3 src/evaluate_2d.py
```
Prints precision/recall/F1/MOTP to stdout in this fixed order: players (YOLO), ball with the classic tracker, then ball with WASB — **does not save a file**.

**4. 3D tracking** (triangulates matched detections across cameras):
```bash
python3 src/tracking_3d.py
```
→ `tracking_results/tracking_3d/3d_positions.csv` + `global_id_map.csv`.
`load_cameras()` reads `calibrated_parameters/cam_{2,4,13}.json`
(`mtx`/`dist`/`rvecs`/`tvecs`) and builds each camera's projection matrix
from them; note `tvecs` are in millimeters and get converted to meters
to match the court's coordinate system (see `COURT_HALF_LENGTH`/`WIDTH`).

**5. 3D evaluation**:
```bash
python3 src/evaluate_3d.py
```
→ `tracking_results/tracking_3d/evaluation_3d.csv`.

**6. Visualization**:
```bash
python3 src/visualize_2d.py            # live overlay of player+ball boxes per video (press q to quit)
python3 src/visualize_3d.py            # static 3D plot -> tracking_results/tracking_3d/3d_positions.png
python3 src/visualize_3d.py --animate --save   # animated -> .../3d_positions.gif
```

## Ball tracking with WASB/HRNet (Docker)

`WASB-SBDT/` is a trimmed fork of
[nttcom/WASB-SBDT](https://github.com/nttcom/WASB-SBDT) (BMVC2023),
kept to just its basketball checkpoint for zero-shot ball detection —
see `WASB-SBDT/claude_context.md` for the full history (why it was
picked, threshold calibration, tiling for wide shots).

### Get the image

No registry — build it locally, and fetch the one checkpoint it needs:

```bash
cd WASB-SBDT
docker build -t wasb-sbdt .
cd src && sh setup_scripts/setup_weights.sh   # downloads pretrained_weights/wasb_basketball_best.pth.tar (~6MB)
cd ../..
```
Rebuild (`docker build -t wasb-sbdt .` again) any time `WASB-SBDT/Dockerfile`
changes — an existing image does not pick up `Dockerfile` edits on its own.

### Start the container

The container mounts the **main repo root** (not `WASB-SBDT/`'s own root),
so it can reach the shared `videos/` and write into `tracking_results/`:

```bash
docker run -it --gpus all -v $(pwd):/workspace -w /workspace/WASB-SBDT/src wasb-sbdt   # first time / clean container
docker start -ai <container_name>                                                      # resume later, keeps anything installed by hand
```
Verify the mount matches before trusting any output:
`docker inspect <container> --format '{{json .Mounts}}'`.

All commands below run **inside the container**, from
`/workspace/WASB-SBDT/src`.

### Running it — the supported cases

**Case A — full WASB tracking run for all 3 videos**
```bash
sh run_tracking.sh
```
This writes the per-camera WASB CSVs to
`tracking_results/tracking_2d/ball_trajectories/ball_tracking_wasb_out{2,4,13}.csv`
and the overlay videos in `tracking_results/tracking_2d/evaluation/`.

**Case B — track a single video by hand**
```bash
python3 track_ball.py \
    --video /workspace/videos/out4.mp4 \
    --checkpoint /workspace/WASB-SBDT/pretrained_weights/wasb_basketball_best.pth.tar \
    --config /workspace/WASB-SBDT/src/configs/model/wasb.yaml \
    --cam-id cam_1 \
    --csv-output /workspace/tracking_results/tracking_2d/ball_trajectories/ball_tracking_wasb_out4.csv \
    --output /workspace/tracking_results/tracking_2d/evaluation/out4_result.mp4
```
`out13` needs `--tile-n 3 --mask-rect 3700,500,3840,680` added (full-court
shot, ball otherwise too small after the mandatory resize — see
`claude_context.md`); `out2`/`out4` work with plain defaults.

**Case C — inspect confidence on a video without producing final output**
(threshold calibration / quick sanity check):
```bash
python3 track_ball.py \
    --video /workspace/videos/out2.mp4 \
    --checkpoint /workspace/WASB-SBDT/pretrained_weights/wasb_basketball_best.pth.tar \
    --config /workspace/WASB-SBDT/src/configs/model/wasb.yaml \
    --cam-id cam_0 \
    --debug-dir /workspace/tracking_results/tracking_2d/evaluation/debug_out2
```

After any tracker run, use the single evaluation entry point in the main repo:
`python3 src/evaluate_2d.py`.

## Citation

WASB model and pretrained weights from:
```
@inproceedings{tarashima2023wasb,
	title={Widely Applicable Strong Baseline for Sports Ball Detection and Tracking},
	author={Tarashima, Shuhei and Haq, Muhammad Abdul and Wang, Yushan and Tagawa, Norio},
	booktitle={BMVC},
	year={2023}
}
```
