#!/bin/bash
# Poll the status and the latest log lines of an OAR job from a local machine.
#
# Usage: ./cluster/gricad/monitor_job.sh <job_id> [interval_seconds]
# Env:   GRICAD_HOST (default bigfoot.ciment), REMOTE_DIR (repository path on the cluster)

set -euo pipefail

JOB_ID="${1:?usage: $0 <job_id> [interval_seconds]}"
INTERVAL="${2:-60}"
HOST="${GRICAD_HOST:-bigfoot.ciment}"
REMOTE_DIR="${REMOTE_DIR:-/bettik/PROJECTS/pr-material-acceleration/\$USER/aerial-object-detection}"

while true; do
    clear
    date
    echo "=== Job ${JOB_ID} ==="
    ssh "${HOST}" "oarstat -j ${JOB_ID}"
    echo
    echo "=== Last log lines ==="
    ssh "${HOST}" "tail -20 ${REMOTE_DIR}/yolo.${JOB_ID}.out 2>/dev/null" || echo "No output yet"
    echo
    echo "Refreshing in ${INTERVAL}s (Ctrl+C to stop)"
    sleep "${INTERVAL}"
done
