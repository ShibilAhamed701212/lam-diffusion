#!/usr/bin/env bash
# Prep work on a fresh VM, independent of the LAM conditioning design:
#   dataset rebuild from essentials.tgz -> Drive, then Algorithm 1's G_θ and §IV-F stage 1
#   (anime VAE), each saved to Drive as soon as it finishes.
# Needs /content/essentials.tgz (scripts/colab_upload_big.sh). Drive saves are best-effort: if Drive
# isn't mounted yet the work stays on the VM and can be saved later with drive_data.sh.
set -euo pipefail
mkdir -p /content/data /content/ckpt
cd /content/data && tar -xzf /content/essentials.tgz
mv -f stage2.pt stage2.json /content/ckpt/ 2>/dev/null || true
cd /content/lam_studio
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
echo "=== save dataset to Drive"
bash lam/jobs/drive_data.sh save || echo "WARN: dataset not saved to Drive (mount it, then: bash lam/jobs/drive_data.sh save)"
echo "=== Algorithm 1: G_theta"
python -m lam.train_inverse --out /content/ckpt/inverse.pt
bash lam/jobs/drive_data.sh ckpt /content/ckpt/inverse.pt /content/ckpt/inverse.json || echo "WARN: G_theta not saved to Drive"
echo "=== stage 1: anime VAE"
python -m lam.train_vae --out /content/ckpt/vae_anime
bash lam/jobs/drive_data.sh ckpt /content/ckpt/vae_anime || echo "WARN: anime VAE not saved to Drive"
echo "=== PREP DONE"
