#!/usr/bin/env bash
# M3 pilot + M4 full joint training (§IV-F stages 3-5) in resumable chunks.
# The first chunk is the pilot: inspect its loss balance and samples before continuing.
#   bash lam/jobs/m4_train.sh HOURS [WEIGHTS_JSON]
set -euo pipefail
# Training is held here while the state migrates to another GPU (scripts/migrate_vm.sh).
if [[ -f /content/HOLD_TRAINING ]]; then echo "=== training held (HOLD_TRAINING)"; exit 0; fi
HOURS="${1:-5}"
WEIGHTS="${2:-}"
STEPS="${LAM_TOTAL_STEPS:-11000}"   # ~24 GPU-hours at ~2 s per sample x 4 accumulation
RESUME=""
[[ -f /content/ckpt/latest.pt ]] && RESUME="--resume /content/ckpt/latest.pt"
python -m lam.train_joint --steps "$STEPS" --hours "$HOURS" --accum 4 --sample-every 750 $RESUME \
  ${WEIGHTS:+--weights "$WEIGHTS"}
echo "=== CHUNK DONE"
