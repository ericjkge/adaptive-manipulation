#!/bin/bash
# Full pick-and-place with expert waypoints every 25 steps: official LeWM + CEM, 300 samples.
# Iterations {1,3,10,30,50,100} with replanning after 5 blocks, plus 10 x 300 replanning after every block.
# 100 clean pick-and-place episodes per config. Configs with an existing result file are skipped (resumable).
set -u
cd "$(dirname "$0")/.."   # official_lewm/
export STABLEWM_HOME=${STABLEWM_HOME:-/workspace/swm} MUJOCO_GL=egl
EPS=$STABLEWM_HOME/datasets/ogbench/cube_clean_eval.h5   # from waypoints/make_expert_episodes.py
OUT=waypoints/results/full_task
mkdir -p $OUT
run() {  # iters samples replan
  f=$OUT/it$1_ns$2_rh$3.json
  [ -f $f ] && { echo "skip $f"; return; }
  echo "=== it=$1 ns=$2 rh=$3 $(date +%H:%M:%S)"
  .venv/bin/python -u waypoints/eval_waypoints.py --source clean --episodes-file $EPS --episodes 100 \
    --iters $1 --samples $2 --replan $3 --out $f 2>&1 | grep -vE 'warn|Warn|INFO' | grep -E '^success|Traceback|Error|^t=.*finished'
}
for it in 1 3 10 30 50 100; do run $it 300 5; done
run 10 300 1
echo SWEEP_DONE $(date +%H:%M:%S)
