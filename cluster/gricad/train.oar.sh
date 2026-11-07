#!/bin/bash
# OAR job: train a YOLOv8 experiment on one A100 GPU of the GRICAD bigfoot cluster.
#
# Submit from the repository root (cloned under /bettik), passing the experiment config:
#   oarsub -S "./cluster/gricad/train.oar.sh configs/experiments/03_oversampled_14classes.yaml"
# Override resources on the command line if needed, e.g. -l /nodes=1/gpu=1,walltime=04:00:00
#
# VENV: Python virtualenv with the packages of requirements.txt
#       (default: /bettik/PROJECTS/pr-material-acceleration/$USER/yolo_venv).

#OAR -n yolo-cadot
#OAR -l /nodes=1/gpu=1,walltime=08:00:00
#OAR -p gpumodel='A100'
#OAR --project pr-material-acceleration
#OAR --stdout yolo.%jobid%.out
#OAR --stderr yolo.%jobid%.err

set -euo pipefail

CONFIG="${1:?usage: oarsub -S \"./cluster/gricad/train.oar.sh <experiment-config.yaml>\"}"
VENV="${VENV:-/bettik/PROJECTS/pr-material-acceleration/${USER}/yolo_venv}"

# CUDA 12.1 toolkit, compatible with the PyTorch cu12x wheels.
source /applis/environments/cuda_env.sh bigfoot 12.1
source "${VENV}/bin/activate"

echo "Job ${OAR_JOB_ID:-local} on $(hostname) - $(date)"
echo "Config: ${CONFIG}"
set -x
python -m src.yolo.train "${CONFIG}"
