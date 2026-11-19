#!/bin/bash
# Official LeWM cube eval (paper setting 10 it x 300 samples, replan after 5) on seeds 42,0..4, then phase split.
export STABLEWM_HOME=${STABLEWM_HOME:-/workspace/swm} MUJOCO_GL=egl
cd "$(dirname "$0")/.."   # official_lewm/
H5=$(find $STABLEWM_HOME/datasets/ogbench -name 'cube_single_expert.h5' | head -1); echo "h5 $H5"
R=phase_split/results; mkdir -p $R
ARGS=""
for s in 42 0 1 2 3 4; do
  .venv/bin/python eval_official.py --config-name=cube.yaml policy=cube/lewm solver.device=cuda solver.n_steps=10 \
    solver.num_samples=300 solver.topk=30 plan_config.receding_horizon=5 eval.num_eval=50 seed=$s \
    +out=$PWD/$R/seed$s.json > $R/seed$s.log 2>&1 && ARGS="$ARGS $R/seed$s.json:$s"
  echo "seed $s: $(grep -o '"success_rate": [0-9.]*' $R/seed$s.json)"
done
.venv/bin/python phase_split/phase_split.py $H5 $ARGS > $R/phase.log 2>&1
cat $R/phase.log
