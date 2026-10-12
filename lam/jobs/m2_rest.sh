#!/usr/bin/env bash
# Resume M1 from precompute, then M2 (LAM Encoder pre-training).
set -euo pipefail
echo "=== precompute"
python -m lam.data.precompute
echo "=== report"
python -m lam.data.report --n 30
echo "=== pose library"
python -m lam.pose_library
echo "=== stage 2"
python -m lam.train_stage2 --steps 3000
echo "=== M1+M2 DONE"
