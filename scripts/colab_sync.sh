#!/usr/bin/env bash
# Pull the newest training weights and sample grids from the VM to lam-studio/checkpoints/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="${LAM_SESSION:-lamstudio}"
DEST="$ROOT/checkpoints"
mkdir -p "$DEST"
latest="$(echo "import glob; f=sorted(glob.glob('/content/ckpt/weights_step*.pt')); print('LATEST=' + (f[-1] if f else ''))" \
  | colab exec -s "$SESSION" --timeout 60 | grep -o 'LATEST=.*' | cut -d= -f2)"
if [[ -n "$latest" && ! -f "$DEST/$(basename "$latest")" ]]; then
  colab download -s "$SESSION" "$latest" "$DEST/$(basename "$latest")"
  ls -t "$DEST"/weights_step*.pt | tail -n +3 | xargs rm -f   # keep two locally
fi
"$ROOT/scripts/colab_job.sh" pull /content/ckpt/samples "$DEST/samples" >/dev/null 2>&1 || true
ls -la "$DEST" | tail -4
