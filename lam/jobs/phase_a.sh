#!/usr/bin/env bash
# Phase A: §IV-F stage 1 (anime VAE), stage 2a (action-text) and stage 2 (anime triplets), resumable:
# rerunning skips finished stages. Needs the dataset in /content/data (build it, or download the
# precomputed set from Kaggle) plus the raw images for stage 1. If a backup mount exists at
# /content/drive, results are also copied there and restored from it on a fresh VM (optional).
set -euo pipefail
C=/content/ckpt; DR=/content/drive/MyDrive/lam_studio/ckpt; D=lam/jobs/drive_data.sh
HAVE_BACKUP=0; [[ -d /content/drive/MyDrive ]] && HAVE_BACKUP=1
backup() { [[ $HAVE_BACKUP == 1 ]] && bash $D ckpt "$@" || true; }
mkdir -p $C
if [[ $HAVE_BACKUP == 1 ]]; then
  for f in vae_anime stage2a.pt stage2a.json stage2_from2a.pt stage2_from2a.json inverse.pt; do
    [[ -e $C/$f || ! -e $DR/$f ]] || cp -r $DR/$f $C/
  done
  [[ -f /content/data/index.jsonl ]] || bash $D restore
fi
[[ -f /content/data/index.jsonl ]] || { echo "no dataset in /content/data (see docs/DATASET.md)"; exit 1; }
if [[ ! -f $C/vae_anime/stage1.json ]]; then
  [[ $(ls /content/data/images 2>/dev/null | wc -l) -ge 28000 ]] || python -m lam.data.fetch \
    --selection /content/data/selection_annotated.jsonl --workers 40
  echo "=== stage 1: anime VAE"
  python -m lam.train_vae --out $C/vae_anime && backup $C/vae_anime
fi
if [[ ! -f $C/stage2a.pt ]]; then
  echo "=== stage 2a: action-text pre-training"
  python -m lam.train_stage2a --out $C/stage2a.pt && backup $C/stage2a.pt $C/stage2a.json
fi
if [[ ! -f $C/stage2_from2a.pt ]]; then
  echo "=== stage 2: anime triplets (from 2a)"
  python -m lam.train_stage2 --init $C/stage2a.pt --out $C/stage2_from2a.pt --steps 3000 \
    && backup $C/stage2_from2a.pt $C/stage2_from2a.json
fi
echo "=== PHASE A DONE"
