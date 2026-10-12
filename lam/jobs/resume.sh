#!/usr/bin/env bash
# Rebuild the VM state from lam-studio/checkpoints/vm_state/essentials.tgz, then continue M3.
set -euo pipefail
mkdir -p /content/data /content/ckpt
cd /content/data && tar -xzf /content/essentials.tgz && mv -f stage2.pt stage2.json /content/ckpt/ && cd /content/lam_studio
# Only the images that were annotated are needed: no re-selection, no re-annotation.
python - <<'PY'
import json
ids = {json.loads(l)["id"] for l in open("/content/data/annotations.jsonl")}
with open("/content/data/selection_annotated.jsonl", "w") as out:
    for l in open("/content/data/selection.jsonl"):
        if json.loads(l)["id"] in ids:
            out.write(l)
print("images to fetch:", len(ids))
PY
echo "=== fetch"
python -m lam.data.fetch --selection /content/data/selection_annotated.jsonl --workers 40
echo "=== precompute"
python -m lam.data.precompute
python -m lam.pose_library
echo "=== continue M3"
bash lam/jobs/m3b.sh
