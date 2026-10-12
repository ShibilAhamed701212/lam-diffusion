#!/usr/bin/env bash
# Phase A (own L4 session; Drive mounted): restore the dataset from Drive, re-fetch the raw images
# (only the VAE needs them), then §IV-F stage 1 (anime VAE), stage 2a (action-text) and stage 2
# (anime triplets, v3 design). Each result goes to Drive as it finishes; rerunning skips done stages.
set -euo pipefail
C=/content/ckpt; DR=/content/drive/MyDrive/lam_studio/ckpt; D=lam/jobs/drive_data.sh
mkdir -p $C
for f in vae_anime stage2a.pt stage2a.json stage2_from2a.pt stage2_from2a.json inverse.pt; do
  [[ -e $C/$f || ! -e $DR/$f ]] || cp -r $DR/$f $C/
done
[[ -f /content/data/index.jsonl ]] || bash $D restore
if [[ ! -f $C/vae_anime/stage1.json ]]; then
  [[ $(ls /content/data/images 2>/dev/null | wc -l) -ge 28000 ]] || python -m lam.data.fetch \
    --selection /content/data/selection_annotated.jsonl --workers 40
  echo "=== stage 1: anime VAE"
  python -m lam.train_vae --out $C/vae_anime && bash $D ckpt $C/vae_anime
fi
if [[ ! -f $C/stage2a.pt ]]; then
  echo "=== stage 2a: action-text pre-training"
  python -m lam.train_stage2a --out $C/stage2a.pt && bash $D ckpt $C/stage2a.pt $C/stage2a.json
fi
if [[ ! -f $C/stage2_from2a.pt ]]; then
  echo "=== stage 2: anime triplets (from 2a)"
  python -m lam.train_stage2 --init $C/stage2a.pt --out $C/stage2_from2a.pt --steps 3000 \
    && bash $D ckpt $C/stage2_from2a.pt $C/stage2_from2a.json
fi
echo "=== PHASE A DONE"
