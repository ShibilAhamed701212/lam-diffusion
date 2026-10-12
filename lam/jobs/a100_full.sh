#!/usr/bin/env bash
# §IV-F stages 3-5 as the paper specifies: the whole U-Net is trained (not LoRA), jointly with
# the LAM Encoder, plain L_diff (no min-SNR) + the other Eq. 3 terms, curriculum, CFG dropout.
# Needs an A100-class GPU and /content/ckpt/stage2_from2a.pt (paper_stages.sh).
#   bash lam/jobs/a100_full.sh overfit        16-triplet overfit check (gate before the real run)
#   bash lam/jobs/a100_full.sh HOURS          one resumable chunk of the full run (first = pilot)
set -euo pipefail
INIT="${LAM_INIT:-/content/ckpt/stage2_from2a.pt}"
COMMON=(--unet-mode full --min-snr 0 --unet-lr "${LAM_UNET_LR:-1e-5}")
if [[ "${1:-}" == "overfit" ]]; then
  python -m lam.train_joint "${COMMON[@]}" --overfit 16 --steps 300 --accum 1 --hours 0.6 --sample-every 100 \
    --ckpt-dir /content/ckpt_overfit_full --init "$INIT"
  echo "=== OVERFIT (FULL U-NET) DONE"
  exit 0
fi
HOURS="${1:-5}"
STEPS="${LAM_TOTAL_STEPS:-11000}"
RESUME=""
[[ -f /content/ckpt_full/latest.pt ]] && RESUME="--resume /content/ckpt_full/latest.pt"
python -m lam.train_joint "${COMMON[@]}" --steps "$STEPS" --hours "$HOURS" --batch "${LAM_BATCH:-1}" \
  --accum "${LAM_ACCUM:-4}" --sample-every 750 --ckpt-dir /content/ckpt_full --init "$INIT" $RESUME
echo "=== CHUNK DONE"
