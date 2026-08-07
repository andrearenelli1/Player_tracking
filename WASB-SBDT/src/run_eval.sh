#!/bin/bash
# Runs eval_wasb.py against ground truth with the final settings
# validated in claude_context.md (tiling 3x3 + mask-rect for out13's
# corner-graphic confuser). Produces the threshold->metrics CSV.
#
# Since WASB-SBDT got imported into the main project repo (session 5),
# the container must be launched with the MAIN REPO ROOT mounted at
# /workspace (not WASB-SBDT's own root):
#   cd <main repo root>
#   docker run -it --gpus all -v $(pwd):/workspace -w /workspace/WASB-SBDT/src wasb-sbdt
#
# Run from inside the container, from /workspace/WASB-SBDT/src:
#   sh run_eval.sh
set -e

python3 eval_wasb.py \
  --annotations /workspace/annotations/_annotations.coco.json \
  --videos-dir /workspace/videos \
  --checkpoint /workspace/WASB-SBDT/pretrained_weights/wasb_basketball_best.pth.tar \
  --config /workspace/WASB-SBDT/src/configs/model/wasb.yaml \
  --tile-videos out13 \
  --tile-n 3 \
  --mask-rect out13:3700,500,3840,680 \
  --output-csv /workspace/tracking_results/tracking_2d/evaluation/eval_results.csv
