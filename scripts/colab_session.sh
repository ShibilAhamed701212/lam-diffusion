#!/usr/bin/env bash
# Ensure the Colab GPU session exists and a single keep-alive is pinging it.
#   LAM_GPU=L4|T4|G4|A100 (default L4)
set -euo pipefail

SESSION="${LAM_SESSION:-lamstudio}"
GPU="${LAM_GPU:-L4}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# One keep-alive per Colab session: killing or replacing one must never stop another session's
# pings (Colab reclaims an un-pinged VM ~20 min later). "lamstudio" keeps the original names.
if [[ "$SESSION" == "lamstudio" ]]; then
  PIDFILE="$ROOT/.keepalive.pid"; KALOG="$ROOT/.keepalive.log"
else
  PIDFILE="$ROOT/.keepalive.$SESSION.pid"; KALOG="$ROOT/.keepalive.$SESSION.log"
fi

# Capture first: `colab sessions | grep -q` under pipefail fails (SIGPIPE) when the match is on line 1.
SESSIONS="$(colab sessions 2>/dev/null || true)"
if ! grep -q "\[$SESSION\]" <<<"$SESSIONS"; then
  echo "[lam] creating Colab session '$SESSION' on $GPU"
  colab new -s "$SESSION" --gpu "$GPU"
fi

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "[lam] keep-alive already running (pid $(cat "$PIDFILE"))"
else
  LAM_SESSION="$SESSION" nohup "$ROOT/scripts/colab_keepalive.sh" >> "$KALOG" 2>&1 &
  echo $! > "$PIDFILE"
  echo "[lam] keep-alive started (pid $(cat "$PIDFILE"))"
fi
