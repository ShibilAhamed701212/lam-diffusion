#!/usr/bin/env bash
# M3 sanity check: train on 16 triplets for 300 steps; samples of those same triplets must
# start reproducing them (proves gradients reach the action stream and the losses are wired).
set -euo pipefail
python -m lam.train_joint --overfit 16 --steps 300 --accum 1 --hours 0.6 --sample-every 100 \
  --ckpt-dir /content/ckpt_overfit --init /content/ckpt/stage2.pt
echo "=== OVERFIT DONE"
