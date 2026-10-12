#!/usr/bin/env bash
# Phase B (A100 — only when the user says go; Drive mounted): full-U-Net overfit gate, then the
# full §IV-F stage 3-5 run, backing up to Drive every 30 min so a new VM resumes where it stopped.
#   bash lam/jobs/phase_b.sh            gate only (inspect Drive lam_studio/overfit_full_samples)
#   LAM_GO=1 bash lam/jobs/phase_b.sh   gate (if not done) + full run / resume
set -euo pipefail
C=/content/ckpt; DR=/content/drive/MyDrive/lam_studio/ckpt
mkdir -p $C /content/ckpt_full
[[ -f /content/data/index.jsonl ]] || bash lam/jobs/drive_data.sh restore
[[ -f $C/stage2_from2a.pt ]] || cp $DR/stage2_from2a.pt $C/
if [[ ! -f $DR/overfit_full.done ]]; then
  bash lam/jobs/a100_full.sh overfit
  cp -r /content/ckpt_overfit_full/samples $DR/overfit_full_samples && touch $DR/overfit_full.done
fi
[[ "${LAM_GO:-0}" == 1 ]] || { echo "=== GATE DONE: check overfit_full_samples, then run with LAM_GO=1"; exit 0; }

backup() {  # newest LAM weights + U-Net + sample grids -> Drive (old copy removed first: Drive space)
  local w u
  w=$(ls -t /content/ckpt_full/weights_step*.pt 2>/dev/null | head -1); u=$(ls -td /content/ckpt_full/unet_step* 2>/dev/null | head -1)
  [[ -n "$w" && -n "$u" ]] || return 0
  rm -rf $DR/unet_latest $DR/weights_latest.pt
  cp -r "$u" $DR/unet_latest && cp "$w" $DR/weights_latest.pt
  cp -r /content/ckpt_full/samples $DR/full_samples 2>/dev/null || true
  echo "[backup] $(basename "$w") -> Drive"
}
( while sleep 1800; do backup || true; done ) &
BK=$!
RESUME=()
if [[ -f /content/ckpt_full/latest.pt ]]; then RESUME=(--resume /content/ckpt_full/latest.pt)
elif [[ -f $DR/weights_latest.pt ]]; then RESUME=(--resume $DR/weights_latest.pt --unet-path $DR/unet_latest); fi
python -m lam.train_joint --unet-mode full --min-snr 0 --unet-lr "${LAM_UNET_LR:-1e-5}" \
  --steps "${LAM_TOTAL_STEPS:-11000}" --hours 999 --batch "${LAM_BATCH:-1}" --accum "${LAM_ACCUM:-4}" --sample-every 750 \
  --ckpt-dir /content/ckpt_full --init $C/stage2_from2a.pt "${RESUME[@]}"
kill $BK 2>/dev/null || true
backup
echo "=== PHASE B DONE"
