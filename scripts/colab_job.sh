#!/usr/bin/env bash
# Run long jobs on the Colab VM as background processes (not tied to `colab exec` timeouts).
#   colab_job.sh sync                         upload lam/ and server/ to /content/lam_studio
#   colab_job.sh start NAME cmd args...       sync, then run `cmd args...` in the background
#   colab_job.sh log NAME [LINES]             tail the job log
#   colab_job.sh status [NAME]                running / exited (with exit code) for one or all jobs
#   colab_job.sh pull REMOTE_PATH LOCAL_PATH  download a file or directory (tarred) from the VM
set -euo pipefail

SESSION="${LAM_SESSION:-lamstudio}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE=/content/lam_studio
JOBS=/content/jobs

rexec() { colab exec -s "$SESSION" --timeout "${2:-120}" <<<"$1"; }

sync_code() {
  local tgz
  tgz="$(mktemp -t lam_code).tgz"
  tar -czf "$tgz" -C "$ROOT" --exclude __pycache__ --exclude '*.pyc' lam server
  colab upload -s "$SESSION" "$tgz" /content/lam_code.tgz >/dev/null
  rm -f "$tgz"
  rexec "import tarfile, os; os.makedirs('$REMOTE', exist_ok=True); tarfile.open('/content/lam_code.tgz').extractall('$REMOTE', filter='data'); print('code synced')"
}

cmd="${1:-}"; shift || true
case "$cmd" in
  sync) sync_code ;;
  start)
    name="$1"; shift
    sync_code
    args_json="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@")"
    rexec "
import json, os, subprocess
os.makedirs('$JOBS', exist_ok=True)
env = dict(os.environ, PYTHONPATH='$REMOTE', PYTHONUNBUFFERED='1')
log = open('$JOBS/$name.log', 'a')
# nice: keep the Jupyter kernel responsive (Colab judges liveness through it)
p = subprocess.Popen(['nice', '-n', '10'] + json.loads(r'''$args_json'''), cwd='$REMOTE', stdout=log, stderr=subprocess.STDOUT,
                     env=env, start_new_session=True)
open('$JOBS/$name.pid', 'w').write(str(p.pid))
print('started $name pid', p.pid)
" ;;
  log) rexec "import subprocess; print(subprocess.run(['tail', '-n', '${2:-40}', '$JOBS/$1.log'], capture_output=True, text=True).stdout)" ;;
  status)
    rexec "
import glob, os
for pidf in sorted(glob.glob('$JOBS/${1:-*}.pid')):
    name = os.path.basename(pidf)[:-4]; pid = int(open(pidf).read())
    try:
        os.kill(pid, 0); state = 'running'
        st = open(f'/proc/{pid}/stat').read().split()
        if st[2] == 'Z':
            state = 'exited (zombie)'
    except ProcessLookupError:
        state = 'exited'
    print(f'{name}: {state} (pid {pid})')
" ;;
  pull)
    remote="$1"; local_path="$2"
    rexec "import os, tarfile
p = '$remote'
if os.path.isdir(p):
    with tarfile.open('/content/_pull.tgz', 'w:gz') as t: t.add(p, arcname='.')
    print('tarred')
else:
    print('file')" 600 | tail -1 | grep -q tarred && is_dir=1 || is_dir=0
    mkdir -p "$(dirname "$local_path")"
    if [[ $is_dir == 1 ]]; then
      colab download -s "$SESSION" /content/_pull.tgz "$local_path.tgz"
      mkdir -p "$local_path" && tar -xzf "$local_path.tgz" -C "$local_path" && rm -f "$local_path.tgz"
    else
      colab download -s "$SESSION" "$remote" "$local_path"
    fi ;;
  *) sed -n 2,8p "$0"; exit 1 ;;
esac
