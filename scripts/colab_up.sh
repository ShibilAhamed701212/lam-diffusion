#!/usr/bin/env bash
# Start (or redeploy) the LAM Studio GPU server on Colab and point the web app at it.
#   LAM_GPU=L4|T4|G4|A100   (default L4)    LAM_PLANNER=llm|rules   (default llm)
set -euo pipefail

SESSION="${LAM_SESSION:-lamstudio}"
GPU="${LAM_GPU:-L4}"
PLANNER="${LAM_PLANNER:-llm}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT/web/.env.local"

TOKEN=""
if [[ -f "$ENV_FILE" ]]; then
  TOKEN="$(grep -E '^LAM_TOKEN=' "$ENV_FILE" | cut -d= -f2- || true)"
fi
[[ -n "$TOKEN" ]] || TOKEN="$(openssl rand -hex 24)"

# One keep-alive per Colab session: killing or replacing one must never stop another session's
# pings (Colab reclaims an un-pinged VM ~20 min later). "lamstudio" keeps the original names.
if [[ "$SESSION" == "lamstudio" ]]; then
  PIDFILE="$ROOT/.keepalive.pid"; KALOG="$ROOT/.keepalive.log"
else
  PIDFILE="$ROOT/.keepalive.$SESSION.pid"; KALOG="$ROOT/.keepalive.$SESSION.log"
fi
if [[ -f "$PIDFILE" ]]; then kill "$(cat "$PIDFILE")" 2>/dev/null || true; rm -f "$PIDFILE"; fi

# Capture first: `colab sessions | grep -q` under pipefail fails (SIGPIPE) when the match is on line 1.
SESSIONS="$(colab sessions 2>/dev/null || true)"
if grep -q "\[$SESSION\]" <<<"$SESSIONS"; then
  echo "[lam] session '$SESSION' exists, restarting kernel to redeploy"
  colab restart-kernel -s "$SESSION"
else
  echo "[lam] creating Colab session '$SESSION' on $GPU"
  colab new -s "$SESSION" --gpu "$GPU"
fi

TGZ="$(mktemp -t lam_server).tgz"
tar -czf "$TGZ" -C "$ROOT/server" --exclude __pycache__ .
colab upload -s "$SESSION" "$TGZ" /content/lam_server.tgz
rm -f "$TGZ"

echo "[lam] installing deps and starting server (a few minutes on first run)"
OUT="$(colab exec -s "$SESSION" -f "$ROOT/scripts/colab_bootstrap.py" --timeout 900 \
  --env "LAM_TOKEN=$TOKEN" --env "LAM_PLANNER=$PLANNER")"
URL="$(grep -o 'LAM_API_URL=https://[^[:space:]]*' <<<"$OUT" | cut -d= -f2- || true)"
if [[ -z "$URL" ]]; then
  echo "$OUT" | tail -30
  echo "[lam] tunnel did not come up; see 'colab log -s $SESSION'" >&2
  exit 1
fi

mkdir -p "$ROOT/web"
umask 077
printf 'LAM_API_URL=%s\nLAM_TOKEN=%s\n' "$URL" "$TOKEN" > "$ENV_FILE"
echo "[lam] API at $URL (written to web/.env.local)"

LAM_SESSION="$SESSION" nohup "$ROOT/scripts/colab_keepalive.sh" > "$KALOG" 2>&1 &
echo $! > "$PIDFILE"
echo "[lam] keep-alive running (pid $(cat "$PIDFILE")); stop everything with scripts/colab_down.sh"

echo "[lam] waiting for models to load (first run downloads ~15 GB)"
for _ in $(seq 1 120); do
  STATUS="$(curl -s --max-time 10 "$URL/health" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["status"], d.get("error") or "")' 2>/dev/null || echo "starting")"
  case "$STATUS" in
    ready*) echo "[lam] ready"; exit 0 ;;
    error*) echo "[lam] model load failed: $STATUS" >&2; exit 1 ;;
  esac
  sleep 15
done
echo "[lam] still loading after 30 min; check 'colab log -s $SESSION'" >&2
exit 1
