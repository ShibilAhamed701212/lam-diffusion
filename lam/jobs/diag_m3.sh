#!/usr/bin/env bash
# Diagnose the failed M3a overfit (see lam/tests/diag_step0.py):
#   1. step-0 images of the overfit specs: fresh encoder / action off / no region mask / old stage2.pt
#   2. overfit again with a FRESH encoder (no old checkpoint) and 5x lower learning rates
set -euo pipefail
echo "=== step-0 diagnosis"
python -m lam.tests.diag_step0
echo "=== overfit: fresh encoder, lr 2e-5"
python -m lam.train_joint --overfit 16 --steps 300 --accum 1 --hours 0.6 --sample-every 100 \
  --ckpt-dir /content/ckpt_overfit_fresh --init none --lr 2e-5
echo "=== DIAG DONE"
