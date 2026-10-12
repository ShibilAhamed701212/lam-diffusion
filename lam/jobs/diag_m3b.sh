#!/usr/bin/env bash
# Test the shared-softmax action stream + full-sequence character tokens (see conditioning.py):
# smoke test, step-0 images, a 600-step overfit with a fresh encoder, then the trained
# checkpoint rendered with and without quality tags.
set -euo pipefail
echo "=== smoke"
python -m lam.tests.smoke
echo "=== step-0 images"
python -m lam.tests.diag_step0 --old "" --out /content/diag_step0_v3.jpg
echo "=== overfit v3: fresh encoder, lr 5e-5, 600 steps"
python -m lam.train_joint --overfit 16 --steps 600 --accum 1 --hours 0.9 --sample-every 200 \
  --ckpt-dir /content/ckpt_overfit_v3 --init none --lr 5e-5
echo "=== render trained checkpoint"
python -m lam.tests.diag_step0 --ckpt /content/ckpt_overfit_v3/latest.pt --out /content/diag_ckpt_v3.jpg
echo "=== DIAG V3 DONE"
