#!/usr/bin/env bash
# Keep the precomputed dataset and small checkpoints on Google Drive so any VM can resume fast.
#   bash lam/jobs/drive_data.sh save      /content/data -> Drive (one uncompressed tar: npz/jpg are already compressed)
#   bash lam/jobs/drive_data.sh restore   Drive -> /content/data (~15 min instead of a ~1.5 h rebuild)
#   bash lam/jobs/drive_data.sh ckpt PATH...  copy checkpoint files/folders to Drive
# Needs Drive mounted at /content/drive (colab drivemount).
set -euo pipefail
D=/content/drive/MyDrive/lam_studio
[[ -d /content/drive/MyDrive ]] || { echo "Drive is not mounted at /content/drive"; exit 1; }
mkdir -p "$D/data" "$D/ckpt"
case "${1:-}" in
  save)
    cd /content/data
    tar -cf /content/_data.tar samples index.jsonl annotations.jsonl $(ls pose_library.json selection*.jsonl 2>/dev/null)
    cp /content/_data.tar "$D/data/data.tar.part" && mv "$D/data/data.tar.part" "$D/data/data.tar"
    rm -f /content/_data.tar
    ls -la "$D/data" ;;
  restore)
    mkdir -p /content/data
    cp "$D/data/data.tar" /content/_data.tar
    tar -xf /content/_data.tar -C /content/data && rm -f /content/_data.tar
    echo "restored: $(ls /content/data/samples | wc -l) files in samples" ;;
  ckpt)
    shift
    for p in "$@"; do
      cp -r "$p" "$D/ckpt/" && echo "saved $(basename "$p") to Drive"
    done ;;
  *) sed -n 2,6p "$0"; exit 1 ;;
esac
