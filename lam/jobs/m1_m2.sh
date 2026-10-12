#!/usr/bin/env bash
# M1 (dataset) + M2 (LAM Encoder pre-training) on the VM, in order. Every stage is resumable.
set -euo pipefail
ANN=/content/data/annotations.jsonl

echo "=== annotate: 5 processes"
for i in 0 1 2 3 4; do
  python -m lam.data.annotate --workers 1 --shard $i --num-shards 5 > /content/jobs/annotate_$i.log 2>&1 &
done
wait
cat $ANN.part* >> $ANN && rm -f $ANN.part*
echo "=== annotations: $(wc -l < $ANN)"
echo "=== precompute"
python -m lam.data.precompute
echo "=== report"
python -m lam.data.report --n 30
echo "=== stage 2"
python -m lam.train_stage2 --steps 3000
echo "=== M1+M2 DONE"
