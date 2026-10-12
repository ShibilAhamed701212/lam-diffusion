#!/usr/bin/env bash
# Stop the keep-alive and release the Colab VM so it stops using compute units.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="${LAM_SESSION:-lamstudio}"
# One keep-alive per Colab session: killing or replacing one must never stop another session's
# pings (Colab reclaims an un-pinged VM ~20 min later). "lamstudio" keeps the original names.
if [[ "$SESSION" == "lamstudio" ]]; then
  PIDFILE="$ROOT/.keepalive.pid"; KALOG="$ROOT/.keepalive.log"
else
  PIDFILE="$ROOT/.keepalive.$SESSION.pid"; KALOG="$ROOT/.keepalive.$SESSION.log"
fi
if [[ -f "$PIDFILE" ]]; then kill "$(cat "$PIDFILE")" 2>/dev/null || true; rm -f "$PIDFILE"; fi
colab stop -s "$SESSION"
