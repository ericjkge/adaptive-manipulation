#!/usr/bin/env bash
# Reduced fixed-budget sweep for LeWM + CEM on OGBench-Cube (no gate). Throttled for overnight runs.
# Usage: ./run_cube_sweep.sh [device]      Smoke test: SMOKE=1 ./run_cube_sweep.sh
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
DEVICE="${1:-mps}"
NUM_EVAL="${NUM_EVAL:-50}"
export SWEEP_THROTTLE="${SWEEP_THROTTLE:-1}"
COOLDOWN="${COOLDOWN:-60}"
OUT=results/cube_sweep
COMMON=(--config-name=cube.yaml policy=cube/lewm solver.device="$DEVICE")

# paper default for Cube first (10 iterations, 300 samples, replan after 5): validates the setup (~74% in paper)
CONFIGS=("10 300 5")
for it in 1 3 10 30; do for ns in 100 300; do CONFIGS+=("$it $ns 5"); done; done
CONFIGS+=("10 300 1" "10 300 2")     # does frequent replanning hurt on Cube too?
if [ "${SMOKE:-0}" = 1 ]; then CONFIGS=("3 100 5"); NUM_EVAL=4; OUT=results/cube_smoke; fi
mkdir -p "$OUT/logs"

STARTED=0
for c in "${CONFIGS[@]}"; do
  read -r it ns rh <<< "$c"
  name="it${it}_ns${ns}_rh${rh}"
  [ -f "$OUT/$name.json" ] && { echo "skip $name"; continue; }
  [ "$STARTED" = 1 ] && sleep "$COOLDOWN"; STARTED=1
  echo "=== $name  (topk=$(( ns / 10 )))  $(date)"
  python sweep_eval.py "${COMMON[@]}" solver.n_steps="$it" solver.num_samples="$ns" solver.topk=$(( ns / 10 )) \
    plan_config.receding_horizon="$rh" eval.num_eval="$NUM_EVAL" \
    +sweep_out="$PWD/$OUT/$name.json" > "$OUT/logs/$name.log" 2>&1 \
    && python -c "import json;r=json.load(open('$OUT/$name.json'));print(f\"  success={r['success_rate']:.1f}%\")" \
    || echo "  FAILED, see $OUT/logs/$name.log"
done
echo "=== cube success sweep done $(date)"
[ "${SMOKE:-0}" = 1 ] && exit 0

# unthrottled timing benchmark (per-plan cost depends only on iterations x samples)
TOUT=results/cube_timing; mkdir -p "$TOUT/logs"
for it in 1 3 10 30; do for ns in 100 300; do
  name="it${it}_ns${ns}"
  [ -f "$TOUT/$name.json" ] && continue
  sleep "$COOLDOWN"; echo "=== timing $name  $(date)"
  SWEEP_THROTTLE=0 python sweep_eval.py "${COMMON[@]}" solver.n_steps="$it" solver.num_samples="$ns" solver.topk=$(( ns / 10 )) \
    plan_config.receding_horizon=1 eval.num_eval=10 +sweep_out="$PWD/$TOUT/$name.json" > "$TOUT/logs/$name.log" 2>&1 \
    && python -c "import json;r=json.load(open('$TOUT/$name.json'));print(f\"  s/plan={r['seconds_per_plan']:.4f}\")" \
    || echo "  FAILED, see $TOUT/logs/$name.log"
done; done
echo "=== cube all done $(date)"
