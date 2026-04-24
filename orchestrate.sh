#!/bin/bash
# Wait for tune.py (PID passed as $1) to exit, then launch the GLM
# pipeline. Runs detached; no Claude-in-the-loop token cost.
#
# Usage: nohup bash orchestrate.sh <TUNE_PID> > orchestrate.log 2>&1 &

set -u
TUNE_PID="${1:?need tune.py PID as arg}"
GLM_ROOT=/workspace/GLM
GLM_LOG=$GLM_ROOT/pipeline_run.log
TUNE_LOG=/workspace/Molecule_CV/tune_v2.log
START_TS=$(date -Iseconds)

echo "[$START_TS] orchestrator starting, waiting for tune.py PID=$TUNE_PID"

while kill -0 "$TUNE_PID" 2>/dev/null; do
    sleep 300
done

END_TS=$(date -Iseconds)
echo "[$END_TS] tune.py PID=$TUNE_PID has exited"

# Sanity check: did tune.py finish cleanly?
if tail -5 "$TUNE_LOG" 2>/dev/null | grep -q "^Done\\.$"; then
    echo "tune.py finished cleanly (saw 'Done.' in log)"
else
    echo "WARNING: tune.py log does not end with 'Done.' — check tune_v2.log"
fi

# Kick off GLM pipeline regardless, since user wants it to run next.
echo "[$(date -Iseconds)] launching GLM pipeline"
cd "$GLM_ROOT" || { echo "FATAL: cannot cd to $GLM_ROOT"; exit 1; }
nice -n 10 Rscript R/run_pipeline.R > "$GLM_LOG" 2>&1
GLM_EXIT=$?
echo "[$(date -Iseconds)] GLM pipeline exited with code $GLM_EXIT"
