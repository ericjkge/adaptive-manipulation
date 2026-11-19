#!/bin/bash
# Per-phase success vs CEM iterations: 25-step segments (reach 0, grasp 25, place 50) x iterations, 300 samples.
cd "$(dirname "$0")/.."   # official_lewm/
export STABLEWM_HOME=${STABLEWM_HOME:-/workspace/swm} MUJOCO_GL=egl
EPS=$STABLEWM_HOME/datasets/ogbench/cube_clean_eval.h5   # from waypoints/make_expert_episodes.py
OUT=waypoints/results/segments; mkdir -p $OUT
for s in 25 50 0; do for it in 1 3 10 30 100; do
  f=$OUT/seg${s}_it$it.json; [ -f $f ] && continue
  echo "=== seg $s it $it $(date +%H:%M:%S)"
  .venv/bin/python -u waypoints/eval_waypoints.py --source segment --seg-start $s --episodes-file $EPS --episodes 100 \
    --iters $it --samples 300 --replan 5 --out $f 2>&1 | grep -E '^success|Traceback|Error'
done; done
echo SEG_DONE
