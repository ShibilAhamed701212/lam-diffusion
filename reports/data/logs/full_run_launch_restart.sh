pkill -f "bash lam/jobs/phase_b"
pkill -f "lam.train_joint"
sleep 8
rm -rf /content/ckpt_full/* 2>/dev/null
cat > ~/go.sh <<'EOF'
. ~/venv/bin/activate
cd ~/lam_studio
export PYTHONPATH=$HOME/lam_studio PYTHONUNBUFFERED=1 LAM_GO=1 LAM_BATCH=4 LAM_ACCUM=1
exec bash lam/jobs/phase_b.sh
EOF
nohup bash ~/go.sh > ~/phaseB.log 2>&1 < /dev/null &
echo relaunched
