#!/usr/bin/env bash
# Upload a large file to a Colab VM: the upload endpoint rejects big files, so split into
# 40 MB parts, upload each, reassemble on the VM and verify the sha256.
#   LAM_SESSION=name scripts/colab_upload_big.sh LOCAL_FILE REMOTE_PATH
set -euo pipefail
SESSION="${LAM_SESSION:?set LAM_SESSION}"
src="$1"; dest="$2"
tmp="$(mktemp -d)"
# A colab call can hang when it collides with a keep-alive ping; retry with a deadline instead.
retry() { for i in 1 2 3 4; do perl -e 'alarm shift; exec @ARGV' "$@" && return 0; echo "[upload] retry $i: ${*:2:3}"; sleep 5; done; return 1; }
trap 'rm -rf "$tmp"' EXIT
sum="$(shasum -a 256 "$src" | cut -d' ' -f1)"
split -b 40m "$src" "$tmp/part_"
parts=("$tmp"/part_*)
echo "[upload] ${#parts[@]} parts"
# (code in a file, not stdin: a retried command would find stdin already consumed)
echo "import os, shutil; shutil.rmtree('/content/_up', ignore_errors=True); os.makedirs('/content/_up'); print('ready')" > "$tmp/mk.py"
retry 90 colab exec -s "$SESSION" --timeout 60 -f "$tmp/mk.py"
for p in "${parts[@]}"; do
  retry 300 colab upload -s "$SESSION" "$p" "/content/_up/$(basename "$p")" >/dev/null && echo "[upload] $(basename "$p") ok"
done
perl -e 'alarm 900; exec @ARGV' colab exec -s "$SESSION" --timeout 600 <<PY
import glob, hashlib, os
parts = sorted(glob.glob('/content/_up/part_*'))
os.makedirs(os.path.dirname('$dest') or '.', exist_ok=True)
h = hashlib.sha256()
with open('$dest', 'wb') as out:
    for p in parts:
        b = open(p, 'rb').read(); out.write(b); h.update(b); os.remove(p)
print('sha256 ok' if h.hexdigest() == '$sum' else 'SHA256 MISMATCH', len(parts), 'parts')
PY
