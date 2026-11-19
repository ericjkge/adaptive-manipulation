#!/bin/bash
# Pod setup: python env (the LeWM sweep lockfile + ogbench), official le-wm code, OGBench visual-cube-single data.
set -euo pipefail
cd /workspace
log() { echo "[$(date +%H:%M:%S)] $*"; }
log "apt"; apt-get update -qq && apt-get install -y -qq libegl1 libgl1 libglfw3 > /dev/null
log "uv"; pip install -q uv
[ -d adaptive-manipulation ] || git clone -q https://github.com/ericjkge/adaptive-manipulation.git
[ -d le-wm ] || git clone -q --depth 1 https://github.com/lucas-maes/le-wm.git
log "data (background)"
mkdir -p ogbench_data
( cd ogbench_data && for f in visual-cube-single-play-v0.npz visual-cube-single-play-v0-val.npz; do
    [ -f $f ] || curl -s -o $f https://rail.eecs.berkeley.edu/datasets/ogbench/$f; done; echo DATA_DONE ) > data.log 2>&1 &
log "python env"
uv venv -q --python=3.10 /workspace/venv
VIRTUAL_ENV=/workspace/venv uv pip install -q -r adaptive-manipulation/official_lewm/requirements.lock.txt
VIRTUAL_ENV=/workspace/venv uv pip install -q ogbench
/workspace/venv/bin/python -c "import torch, ogbench, stable_pretraining, mujoco; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
wait
log "ENV_DONE"
