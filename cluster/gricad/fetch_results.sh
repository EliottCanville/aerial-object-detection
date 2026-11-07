#!/bin/bash
# Copy a finished training run from the cluster into experiments/<run_name>/.
# Metrics, plots and args are copied as-is; only best.pt is kept from the checkpoints.
#
# Usage: ./cluster/gricad/fetch_results.sh <run_name>     (e.g. 03_oversampled_14classes)
# Env:   GRICAD_HOST (default bigfoot.ciment), REMOTE_DIR (repository path on the cluster)

set -euo pipefail

RUN="${1:?usage: $0 <run_name>}"
HOST="${GRICAD_HOST:-bigfoot.ciment}"
REMOTE_DIR="${REMOTE_DIR:-/bettik/PROJECTS/pr-material-acceleration/\$USER/aerial-object-detection}"
DEST="experiments/${RUN}"

mkdir -p "${DEST}/weights"
rsync -av --exclude 'weights/' --exclude 'train_batch*.jpg' "${HOST}:${REMOTE_DIR}/runs/${RUN}/" "${DEST}/"
rsync -av "${HOST}:${REMOTE_DIR}/runs/${RUN}/weights/best.pt" "${DEST}/weights/"

echo "Run ${RUN} copied to ${DEST}"
