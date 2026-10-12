#!/usr/bin/env bash
# Runs after M2: overfit sanity check (M3a), then the 1-hour pilot that starts the full run (M3b).
set -euo pipefail
while kill -0 "$(cat /content/jobs/pipeline.pid)" 2>/dev/null; do sleep 30; done
[[ -f /content/ckpt/stage2.pt ]] || { echo "stage 2 checkpoint missing"; exit 1; }
echo "=== M3a overfit"
bash lam/jobs/m3_overfit.sh
echo "=== M3b pilot (first hour of the full run)"
bash lam/jobs/m4_train.sh 1
echo "=== M3 DONE"
