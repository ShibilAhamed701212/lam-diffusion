#!/usr/bin/env bash
# Paper-completion stages that don't need the full U-Net (cheap; fine on an L4):
#   §IV-F stage 1   anime-optimized VAE (decoder fine-tune, latent space unchanged)
#   §IV-F stage 2a  action-text pre-training (HumanML3D) + Temporal Attention order supervision
#   §IV-F stage 2   anime action-pose-text triplets, continuing from 2a
#   Algorithm 1     G_θ: VAE latent -> action descriptor
# Needs /content/data/{images,samples,index.jsonl}. Each stage is skipped if its output exists.
set -euo pipefail
C=/content/ckpt
[[ -f $C/vae_anime/stage1.json ]] || { echo "=== stage 1: anime VAE"; python -m lam.train_vae --out $C/vae_anime; }
[[ -f $C/stage2a.pt ]] || { echo "=== stage 2a: action-text pre-training"; python -m lam.train_stage2a --out $C/stage2a.pt; }
[[ -f $C/stage2_from2a.pt ]] || { echo "=== stage 2: anime triplets (from 2a)"; python -m lam.train_stage2 --init $C/stage2a.pt --out $C/stage2_from2a.pt --steps 3000; }
[[ -f $C/inverse.pt ]] || { echo "=== Algorithm 1: G_theta"; python -m lam.train_inverse --out $C/inverse.pt; }
echo "=== PAPER STAGES DONE"
