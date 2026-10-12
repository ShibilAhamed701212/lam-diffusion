#!/usr/bin/env bash
# Colab reclaims a runtime ~15-20 min after the last *cell execution*; servers and training
# jobs running as background processes don't count. Run a trivial cell every few minutes.
# Started by colab_up.sh / colab_job.sh, stopped by colab_down.sh.
# Only exits after the session has been confirmed gone several checks in a row, so a single
# network blip can't stop the pings (which would then let Colab reclaim the VM).
set -uo pipefail

SESSION="${LAM_SESSION:-lamstudio}"
INTERVAL="${LAM_KEEPALIVE_SECONDS:-180}"
GONE_CONFIRMATIONS=3

# Keep the Mac from idle-sleeping while we ping; a sleeping Mac stops the pings.
caffeinate -i -w $$ &

# macOS has no `timeout`; perl's alarm kills a ping whose websocket hangs, so one stuck ping
# can't block every later one (which would let Colab reclaim the VM as idle).
with_deadline() { perl -e 'alarm shift; exec @ARGV' "$@"; }

gone=0
while true; do
  if echo "None" | with_deadline 90 colab exec -s "$SESSION" --timeout 60 >/dev/null 2>&1; then
    gone=0
  elif grep -q "\[$SESSION\]" <<<"$(with_deadline 60 colab sessions 2>/dev/null || true)"; then
    gone=0
    echo "[keepalive] $(date '+%H:%M:%S') ping failed but session listed; retrying"
  else
    gone=$((gone + 1))
    echo "[keepalive] $(date '+%H:%M:%S') session not found ($gone/$GONE_CONFIRMATIONS)"
    if (( gone >= GONE_CONFIRMATIONS )); then
      echo "[keepalive] session '$SESSION' is gone, exiting"
      exit 0
    fi
    sleep 20
    continue
  fi
  sleep "$INTERVAL"
done
