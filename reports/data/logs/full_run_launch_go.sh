. ~/venv/bin/activate
cd ~/lam_studio
export PYTHONPATH=$HOME/lam_studio PYTHONUNBUFFERED=1 LAM_GO=1 LAM_BATCH=4 LAM_ACCUM=1
exec bash lam/jobs/phase_b.sh
