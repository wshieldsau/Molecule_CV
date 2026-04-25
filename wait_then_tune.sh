#!/bin/bash
# Wait for another Claude's GLM run (PID passed as $1) to finish,
# then run custom_cnn and admet tuning in fresh Python processes.
# Does NOT launch GLM itself.
#
# Usage: nohup bash wait_then_tune.sh <GLM_PID> > wait_then_tune.log 2>&1 &

set -u
GLM_PID="${1:?need GLM PID as arg}"
cd /workspace/Molecule_CV
source /workspace/.venv/bin/activate

log() { echo "[$(date -Iseconds)] $*"; }

log "waiting for GLM PID=$GLM_PID to exit before starting tuning"
while kill -0 "$GLM_PID" 2>/dev/null; do
    sleep 300
done
log "GLM PID=$GLM_PID has exited; starting custom_cnn"

nice -n 10 python -u tune.py --model custom_cnn >> tune_v2.log 2>&1
log "custom_cnn exited with code $?"

log "starting admet"
nice -n 10 python -u tune.py --model admet >> tune_v2.log 2>&1
log "admet exited with code $?"

log "all tuning done"
