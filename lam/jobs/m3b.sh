#!/usr/bin/env bash
# After the character-token redesign: overfit check first (fast feedback), then Stage 2 on the
# new encoder, then the 1-hour pilot that starts the full run.
set -euo pipefail
rm -rf /content/ckpt_overfit
echo "=== M3a overfit"
bash lam/jobs/m3_overfit.sh
echo "=== M2 stage 2 (new encoder)"
python -m lam.train_stage2 --steps 3000
echo "=== M3b pilot"
bash lam/jobs/m4_train.sh 1
echo "=== M3 DONE"
