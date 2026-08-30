#!/bin/bash
# Runs track_ball.py on all 3 project videos with the validated settings
# (default conf-thresh=0.02 baked into track_ball.py;
# out13 needs tiling + a mask for the corner-graphic confuser, out2/out4
# work with plain defaults). Produces one CSV (schema shared with
# tracking_players_2d.py/track_ball_classic.py, see track_ball.py's
# docstring) in tracking_results/tracking_2d/ball_trajectories/, and one
# overlay video in tracking_results/tracking_2d/evaluation/, per input
# video.
#
# Since WASB-SBDT got imported into the main project repo (session 5),
# the container must be launched with the MAIN REPO ROOT mounted at
# /workspace (not WASB-SBDT's own root) so this script can reach the
# source videos and write into tracking_results/ without leaving the
# mount:
#   cd <main repo root>
#   docker run -it --gpus all -v $(pwd):/workspace -w /workspace/WASB-SBDT/src wasb-sbdt
#
# Run from inside the container, from /workspace/WASB-SBDT/src:
#   sh run_tracking.sh
set -e

CHECKPOINT=/workspace/WASB-SBDT/pretrained_weights/wasb_basketball_best.pth.tar
CONFIG=/workspace/WASB-SBDT/src/configs/model/wasb.yaml
VIDEOS_DIR=/workspace/videos
BALL_DIR=/workspace/tracking_results/tracking_2d/ball_trajectories
EVAL_DIR=/workspace/tracking_results/tracking_2d/evaluation

echo "=== out2 (cam_0) ==="
python3 track_ball.py \
  --video ${VIDEOS_DIR}/out2.mp4 \
  --checkpoint ${CHECKPOINT} \
  --config ${CONFIG} \
  --cam-id cam_0 \
  --csv-output ${BALL_DIR}/ball_tracking_wasb_out2.csv \
  --output ${EVAL_DIR}/out2_result.mp4

echo "=== out4 (cam_1) ==="
python3 track_ball.py \
  --video ${VIDEOS_DIR}/out4.mp4 \
  --checkpoint ${CHECKPOINT} \
  --config ${CONFIG} \
  --cam-id cam_1 \
  --csv-output ${BALL_DIR}/ball_tracking_wasb_out4.csv \
  --output ${EVAL_DIR}/out4_result.mp4

echo "=== out13 (cam_2, tiled 3x3 + mask on the corner-graphic confuser) ==="
python3 track_ball.py \
  --video ${VIDEOS_DIR}/out13.mp4 \
  --checkpoint ${CHECKPOINT} \
  --config ${CONFIG} \
  --cam-id cam_2 \
  --tile-n 3 \
  --mask-rect 3700,500,3840,680 \
  --csv-output ${BALL_DIR}/ball_tracking_wasb_out13.csv \
  --output ${EVAL_DIR}/out13_result.mp4

echo "=== done: CSVs written to ${BALL_DIR}, overlay videos to ${EVAL_DIR} ==="
