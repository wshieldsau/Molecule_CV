#!/bin/bash
# Run custom_cnn, then admet in separate Python processes (so TF state
# gets wiped between models). Then hand off to GLM.
#
# Usage: nohup bash chain_runs.sh > chain.log 2>&1 &

set -u
cd /workspace/Molecule_CV
source /workspace/.venv/bin/activate

log() { echo "[$(date -Iseconds)] $*"; }

log "starting custom_cnn"
nice -n 10 python -u tune.py --model custom_cnn >> tune_v2.log 2>&1
CC_EXIT=$?
log "custom_cnn exited with code $CC_EXIT"

log "starting admet"
nice -n 10 python -u tune.py --model admet >> tune_v2.log 2>&1
AD_EXIT=$?
log "admet exited with code $AD_EXIT"

log "tuning chain complete. Handing off to GLM."
cd /workspace/GLM
nice -n 10 Rscript R/run_pipeline.R > pipeline_run.log 2>&1
log "GLM pipeline exited with code $?"
