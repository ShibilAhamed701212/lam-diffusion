#!/usr/bin/env bash
# Move the prepared training state from the current VM to a new one on a different GPU, through
# this Mac (Colab VMs can't talk to each other and the upload endpoint caps file size).
#   scripts/migrate_vm.sh A100
# Moves: precomputed samples + index + pose library + /content/ckpt (stage 2, latest training state).
set -euo pipefail

GPU="${1:-A100}"
SESSION="${LAM_SESSION:-lamstudio}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$ROOT/checkpoints/migrate"
PART=400m
mkdir -p "$STAGE"
rexec() { perl -e 'alarm shift; exec @ARGV' "${2:-900}" colab exec -s "$SESSION" --timeout "${2:-900}" <<<"$1"; }

echo "[migrate] packing on the old VM"
rexec "
import subprocess
cmd = ('cd /content && tar -cf - data/samples data/index.jsonl data/pose_library.json data/annotations.jsonl ckpt '
       '| split -b $PART - /content/mig_ && ls /content/mig_* | wc -l')
print('parts', subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip())
" 1800

parts="$(rexec "import glob; print(' '.join(sorted(glob.glob('/content/mig_*'))))" 120 | tail -1)"
echo "[migrate] downloading $(wc -w <<<"$parts") parts"
for p in $parts; do
  f="$STAGE/$(basename "$p")"
  [[ -f "$f" ]] || perl -e 'alarm 1800; exec @ARGV' colab download -s "$SESSION" "$p" "$f" >/dev/null
done
( cd "$STAGE" && cat mig_* | md5 -q > checksum.txt )

echo "[migrate] stopping the old VM and starting $GPU"
kill "$(cat "$ROOT/.keepalive.pid")" 2>/dev/null || true
rm -f "$ROOT/.keepalive.pid"
colab stop -s "$SESSION" || true
LAM_GPU="$GPU" "$ROOT/scripts/colab_session.sh"
perl -e 'alarm 900; exec @ARGV' colab exec -s "$SESSION" -f "$ROOT/scripts/vm_setup.py" --timeout 800 | tail -1

echo "[migrate] uploading (40 MB chunks; the upload endpoint rejects large files)"
rexec "import os; os.makedirs('/content/up', exist_ok=True)" 60
CHUNKS="$STAGE/chunks"
mkdir -p "$CHUNKS"
cat "$STAGE"/mig_* | split -b 40m - "$CHUNKS/c_"
for c in "$CHUNKS"/c_*; do
  perl -e 'alarm 600; exec @ARGV' colab upload -s "$SESSION" "$c" "/content/up/$(basename "$c")" >/dev/null
done
rexec "
import subprocess, hashlib
r = subprocess.run('cd /content && cat up/c_* > mig.tar && md5sum mig.tar && tar -xf mig.tar && rm -rf up mig.tar',
                   shell=True, capture_output=True, text=True)
print(r.stdout.split()[0] if r.stdout else r.stderr[-500:])
" 1800 | tail -1 > "$STAGE/remote_checksum.txt"
if [[ "$(cat "$STAGE/checksum.txt")" == "$(cat "$STAGE/remote_checksum.txt")" ]]; then
  echo "[migrate] verified ($(cat "$STAGE/checksum.txt")); state restored on $GPU"
else
  echo "[migrate] CHECKSUM MISMATCH: local $(cat "$STAGE/checksum.txt") remote $(cat "$STAGE/remote_checksum.txt")" >&2
  exit 1
fi
