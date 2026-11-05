# LeWM Push-T sweep: setup notes

## Sources (all official)
- Code: github.com/lucas-maes/le-wm at commit HEAD of 2026-05-22 ("fix data config")
- Planning library: github.com/galilai-group/stable-worldmodel at commit `abdced4` (2026-05-20),
  the last commit before the le-wm commit. Neither PyPI 0.0.6 nor 0.1.x matches le-wm's eval.py.
- Checkpoint: huggingface.co/quentinll/lewm-pusht (`weights.pt` + `config.json`), loaded unmodified
- Data: huggingface.co/datasets/quentinll/lewm-pusht (`pusht_expert_train.h5`, 46 GB decompressed)

## Version pins and why
- `stable-worldmodel @ abdced4`: has both `swm.data.HDF5Dataset` and `swm.wm.utils.load_pretrained`, which eval.py needs.
- `stable-pretraining==0.1.6`: the release current on 2026-05-20.
- `transformers<5` (4.57.6): the checkpoint's ViT weights use the 4.x parameter names; every 5.x release
  renamed them. With 4.x the checkpoint loads with strict key matching and no conversion.
- `hdf5plugin`: from stable-worldmodel's `format` extra, which the README install line omits.
- `datasets>=3`: otherwise uv resolves an ancient datasets 1.1.1 that breaks on modern pyarrow.
- Full lockfile: `requirements.lock.txt`.

## File layout under ~/.stable_worldmodel
- `checkpoints/pusht/lewm/{weights.pt,config.json}`
- `datasets/pusht_expert_train.h5`

## Eval protocol (unchanged from the paper / config/eval/pusht.yaml)
50 episodes sampled with seed 42 from the expert dataset, goal = state 25 steps ahead,
budget 50 env steps, horizon 5 blocks, frameskip 5. Same 50 start/goal pairs for every config.
Defaults: CEM 300 samples, 30 iterations, 30 elites, replan after all 5 blocks.

## Changes in sweep_eval.py vs eval.py
- device from `solver.device` (mps on Mac) instead of hardcoded cuda; MUJOCO_GL=egl only on Linux
- times every solver call; per-plan time excludes the first call (GPU warmup)
- writes one JSON per config with per-episode successes

## Sweep grid (run_sweep.sh)
iterations {1,3,10,30} x samples {30,100,300} (elites = 10% of samples) x receding horizon {1,2,5} blocks.
Paper default (30, 300, 5) runs first as a reproduction check.

## Thermal throttling and timing (added 2026-11-04)
- Configs after the first 9 run with SWEEP_THROTTLE=1: after each CEM round the GPU idles as long as the
  round took, plus a 60 s rest between configs. Each JSON has a `throttle` field.
- Throttling does not change success rates, but it inflates measured time per plan by about 22%
  (GPU clocks drop during idle). So use `results/pusht_timing/` (unthrottled, per iterations x samples)
  for planning cost, not the throttled sweep JSONs.
- Run-to-run noise: rerunning one config unthrottled gave 42% vs 44%, with 7 of 50 episodes flipping.
  The Mac GPU is not bit-for-bit deterministic. Treat differences of a few points as noise.

## Replanning after every block underperforms (verified not a bug)
- Replanning after 1 block scores 24-44% across iterations/samples, vs 94% for the paper default (replan after 5).
- Diagnostic: 10 iterations x 300 samples, replan after 1 block, warm start OFF scored 20% vs 24% with it ON
  (within noise). The warm start is the only code path that differs from the default, so this is not a bug.
- LeWM's cost scores only the final imagined state. Episode videos show the planner stitching first blocks of
  different plans and knocking the T out of place. Result in `results/diagnostics/`.

## Slim dataset (disk: 46 GB -> 0.84 GB)
- The eval takes its first observation AND its goal image from the dataset's stored images, so the images
  can't all be dropped. `slim_dataset.py` keeps every low-dimensional column in full, plus images only for
  frames the eval reads: start..start+25 of the episodes sampled for seed 42 x {4,10,50,100,200,500}
  and seeds 0-4 x {50,100,200}. Row indexing is unchanged, so no config changes are needed.
- Verified: all columns identical, all kept frames identical, the same episode selection for all 21 settings,
  and identical init/goal/video inputs through the library's own loader. A 4-episode eval on the slim file
  reproduced the original smoke test.
- Limits: other seeds, other episode counts, or a goal offset other than 25 select frames that aren't kept
  (they would read as black images). For those, re-download (13 GB, ~6 min) and rebuild with a bigger POOL.
- For Cube: download, decompress, `python slim_dataset.py build/verify`, then delete the big file
  (peak disk use roughly 150-200 GB for about an hour).

## OGBench-Cube setup
- Checkpoint: huggingface.co/quentinll/lewm-cube -> `checkpoints/cube/lewm/` (strict load OK).
- Data: huggingface.co/datasets/quentinll/lewm-cube (46 GB archive -> 102 GB h5) -> slim 4.3 GB at
  `datasets/ogbench/cube_single_expert.h5`. Verified byte-identical columns (the action column has 50,000 NaNs,
  one final block per episode, so the check compares raw bytes), identical kept frames, identical eval inputs.
- Rendering on macOS: MUJOCO_GL=glfw works headless (`cgl` is rejected by dm_control). The authors' data was
  rendered on Linux/EGL, so small rendering differences could lower success vs the paper's ~74%.
- Sweep: `run_cube_sweep.sh` (paper default 10 iterations x 300 samples first; iterations {1,3,10,30} x samples
  {100,300} replanning after 5 blocks; plus 10 x 300 replanning after 1 and 2 blocks; then a timing benchmark).
  Starts automatically after the Push-T run once `datasets/ogbench/.cube_ready` exists.

## Cube results (2026-11-04, all 10 configs + timing, no failures)
- Paper setting 10 x 300, replan after 5: 72% (paper ~74%), so the Mac setup is validated on Cube too.
- Cube levels off early: 3 x 300 already reaches 72%; 30 x 300 gets 70%. 1 iteration still gets 42-46%.
- Replanning frequency does not hurt on Cube (74% / 72% / 72% for replan after 1 / 2 / 5 at 10 x 300),
  unlike Push-T. The Push-T replanning effect is task-specific.
- Summary: `results/cube_summary.csv`. Charts: `dashboard/index.html` at the repo root.
