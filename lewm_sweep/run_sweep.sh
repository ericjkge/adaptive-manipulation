#!/usr/bin/env bash
# Fixed-budget sweep for LeWM + CEM on Push-T (no gate).
# Usage: ./run_sweep.sh [device]   device = mps (default) | cuda | cpu
# Smoke test first:  SMOKE=1 ./run_sweep.sh
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
DEVICE="${1:-mps}"
# Thermal throttling: idle THROTTLE x each planning call, and COOLDOWN seconds between configs.
export SWEEP_THROTTLE="${SWEEP_THROTTLE:-1}"
COOLDOWN="${COOLDOWN:-60}"
NUM_EVAL="${NUM_EVAL:-50}"
OUT=results/pusht_sweep
mkdir -p "$OUT/logs"

ITERS=(1 3 10 30)
SAMPLES=(30 100 300)
RECEDING=(1 2 5)
if [ "${SMOKE:-0}" = 1 ]; then ITERS=(3); SAMPLES=(100); RECEDING=(5); NUM_EVAL=4; OUT=results/smoke; mkdir -p "$OUT/logs"; fi

CONFIGS=()
[ "${SMOKE:-0}" = 1 ] || CONFIGS+=("30 300 5")   # paper default first: validates the setup
for rh in "${RECEDING[@]}"; do for it in "${ITERS[@]}"; do for ns in "${SAMPLES[@]}"; do CONFIGS+=("$it $ns $rh"); done; done; done

for c in "${CONFIGS[@]}"; do
  read -r it ns rh <<< "$c"
  name="it${it}_ns${ns}_rh${rh}"
  [ -f "$OUT/$name.json" ] && { echo "skip $name"; continue; }
  topk=$(( ns / 10 ))   # paper ratio: 30 elites of 300
  [ "${STARTED:-0}" = 1 ] && sleep "$COOLDOWN"; STARTED=1
  echo "=== $name  (topk=$topk)  $(date)"
  python sweep_eval.py --config-name=pusht.yaml policy=pusht/lewm \
    solver.device="$DEVICE" solver.n_steps="$it" solver.num_samples="$ns" solver.topk="$topk" \
    plan_config.receding_horizon="$rh" eval.num_eval="$NUM_EVAL" \
    +sweep_out="$PWD/$OUT/$name.json" > "$OUT/logs/$name.log" 2>&1 \
    && python -c "import json;r=json.load(open('$OUT/$name.json'));print(f\"  success={r['success_rate']:.1f}%  s/plan={r['seconds_per_plan']:.3f}\")" \
    || echo "  FAILED, see $OUT/logs/$name.log"
done
echo "=== success sweep done $(date)"

# Timing benchmark: unthrottled, so seconds-per-plan is accurate. Per-plan cost depends only on
# iterations x samples; replanning every block over 10 episodes gives many timed calls.
[ "${SMOKE:-0}" = 1 ] && exit 0
TOUT=results/pusht_timing; mkdir -p "$TOUT/logs"
for it in "${ITERS[@]}"; do for ns in "${SAMPLES[@]}"; do
  name="it${it}_ns${ns}"
  [ -f "$TOUT/$name.json" ] && { echo "skip timing $name"; continue; }
  sleep "$COOLDOWN"
  echo "=== timing $name  $(date)"
  SWEEP_THROTTLE=0 python sweep_eval.py --config-name=pusht.yaml policy=pusht/lewm \
    solver.device="$DEVICE" solver.n_steps="$it" solver.num_samples="$ns" solver.topk=$(( ns / 10 )) \
    plan_config.receding_horizon=1 eval.num_eval=10 \
    +sweep_out="$PWD/$TOUT/$name.json" > "$TOUT/logs/$name.log" 2>&1 \
    && python -c "import json;r=json.load(open('$TOUT/$name.json'));print(f\"  s/plan={r['seconds_per_plan']:.4f}\")" \
    || echo "  FAILED, see $TOUT/logs/$name.log"
done; done
echo "=== all done $(date)"
