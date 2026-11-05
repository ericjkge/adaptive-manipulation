# LeWM fixed-budget planning sweep

Sweeps CEM iterations, samples and replanning frequency for pretrained LeWM on Push-T and OGBench-Cube, using the
authors' eval protocol. Every result in `results/` came from these scripts on an Apple M3 Max (PyTorch MPS).

## Setup (Python 3.10, [uv](https://docs.astral.sh/uv/))

```bash
cd lewm_sweep
uv venv --python=3.10
source .venv/bin/activate
uv pip install -r requirements.lock.txt
```

The lockfile pins the versions that reproduce the paper. The important pins:

- `stable-worldmodel` at GitHub commit `abdced4`, which matches the LeWM code. No PyPI release does.
- `stable-pretraining==0.1.6`.
- `transformers` 4.57.6 (below 5), because the checkpoints use 4.x layer names.
- `hdf5plugin`, needed to read the datasets.

## Checkpoints and data

Everything lives under `~/.stable_worldmodel/` (override with `STABLEWM_HOME`).

```bash
hf download quentinll/lewm-pusht --local-dir ~/.stable_worldmodel/checkpoints/pusht/lewm
hf download quentinll/lewm-cube  --local-dir ~/.stable_worldmodel/checkpoints/cube/lewm
hf download quentinll/lewm-pusht --repo-type dataset --local-dir ~/dl_pusht
hf download quentinll/lewm-cube  --repo-type dataset --local-dir ~/dl_cube
```

Unpack so the files land at `~/.stable_worldmodel/datasets/pusht_expert_train.h5` (46 GB) and
`~/.stable_worldmodel/datasets/ogbench/cube_single_expert.h5` (102 GB). Push-T ships as a `.h5.zst`
(`zstd -d`); Cube as a `.tar.zst` (`tar --zstd -xf`).

Optional: shrink each dataset to the frames the eval reads (0.84 GB and 4.3 GB), then delete the original.

```bash
python slim_dataset.py build  BIG.h5 SLIM.h5
python slim_dataset.py verify BIG.h5 SLIM.h5
```

The slim files only support the seeds and episode counts listed in `slim_dataset.py`.

## Run

```bash
SMOKE=1 ./run_sweep.sh mps
./run_sweep.sh mps
./run_cube_sweep.sh mps
```

Use `cuda` instead of `mps` on an NVIDIA machine. On a Mac the sweep is throttled to keep it cool
(`SWEEP_THROTTLE=1`, `COOLDOWN=60`); set `SWEEP_THROTTLE=0` to disable. Configs that already have a result file
are skipped, so an interrupted sweep resumes.

## Outputs

- `results/<task>_sweep/itI_nsN_rhR.json`: one per config, with success rate and per-episode successes.
  Throttled runs have `throttle: 1`; their timing fields are inflated, so use the timing benchmark instead.
- `results/<task>_timing/itI_nsN.json`: unthrottled seconds per plan.
- `results/pusht_summary.csv`, `results/cube_summary.csv`: success joined with accurate timing.
- `results/diagnostics/`: the warm-start-off check for replanning after every block.

`config/eval/` is copied from [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) (MIT, see `LICENSE-le-wm`).
`sweep_eval.py` is that repo's `eval.py` with device selection, timing, throttling and JSON output added.
