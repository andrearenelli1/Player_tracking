#!/bin/bash
# Downloads only the checkpoint this fork actually uses (WASB/HRNet,
# basketball). The original repo's script downloaded 35 checkpoints
# across 7 models x 5 sports; trimmed down since this repo only runs
# HRNet/WASB now (see src/models/__init__.py).
SCRIPT_DIR=$(cd $(dirname $0); pwd)
BASE_DIR=$SCRIPT_DIR/../..

mkdir -p ${BASE_DIR}/pretrained_weights

wget https://drive.google.com/uc?id=1nfECuSyJvPUmz3njZCdFERSQQbERt8FU -O ${BASE_DIR}/pretrained_weights/wasb_basketball_best.pth.tar
