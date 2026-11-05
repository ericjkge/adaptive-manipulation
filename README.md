# Adaptive Planning for Real-Time Robot Manipulation

MIT 6.4212 final project (Fall 2026). We study when a robot should spend more compute on planning. A learned
gate will choose the planning budget at each step for a world-model planner, under a time cost.

## Current setup

- **World model:** [LeWorldModel (LeWM)](https://arxiv.org/abs/2603.19312), a 15M-parameter JEPA trained from pixels.
- **Planner:** the cross-entropy method (CEM) in LeWM's latent space, with model-predictive control. Each plan is
  5 action blocks of 5 steps; the cost is the distance between the predicted final latent and the goal image's latent.
- **Tasks:** Push-T and OGBench-Cube, using the authors' pretrained checkpoints and eval protocol.
- **Knobs:** CEM iterations, CEM samples, and how many blocks run before replanning.

## Results so far: fixed-budget sweep (no gate)

Open [`dashboard/index.html`](dashboard/index.html) in a browser for interactive charts. All numbers are
50 episodes per config, so each rate is uncertain by about ±7 points.

| Task | Setting | Success |
|---|---|---|
| Push-T | Paper setting: 30 iterations × 300 samples, replan after 5 blocks | 94% (paper ~96%) |
| Push-T | 10 × 300, replan after 5 blocks | 90%, at 38% of the planning time |
| Push-T | Best of 12 budgets when replanning after every block | 44% |
| Cube | Paper setting: 10 × 300, replan after 5 blocks | 72% (paper ~74%) |
| Cube | 3 × 300, replan after 5 blocks | 72%, at 39% of the planning time |

Main findings:

1. **On Push-T, replanning frequency matters most.** Committing to the full plan beats replanning every 1 or 2
   blocks at every budget above 1 iteration. A diagnostic with warm-starting turned off confirmed this is not a bug.
2. **Thinking only pays off when the robot commits.** With committed plans, Push-T success rises from 0–12% at
   1 iteration to 66–94% at 30. With frequent replanning it stays flat.
3. **Iterations and samples need balancing.** 30 × 30 (66%) is worse than 10 × 100 (84%).
4. **Episodes vary a lot in difficulty.** Cheap budgets solve a subset of what expensive ones solve. Picking the
   cheapest budget that solves each episode would use about 17× less compute than the default (an optimistic
   upper bound). This is the headroom an adaptive gate can exploit.
5. **Cube behaves differently.** It levels off after about 3 iterations, and replanning after every block does not
   hurt (74% vs 72%). On Cube the gate's opportunity is mainly saving compute.

## Repository layout

- [`lewm_sweep/`](lewm_sweep/) — sweep code, configs and every result file. Setup and reproduction steps are in
  [`lewm_sweep/README.md`](lewm_sweep/README.md); details and caveats in [`lewm_sweep/SETUP_NOTES.md`](lewm_sweep/SETUP_NOTES.md).
- [`dashboard/`](dashboard/) — standalone results page. Rebuild it with `python dashboard/build_dashboard.py`.

## Next steps

- Add a time cost and disturbances so that thinking costs task time, not just compute.
- Log the planner's cost per step to test whether difficulty varies within an episode.
- Train the gate (supervised from the sweep's labels first, then PPO), and compare against fixed budgets and a
  heuristic that stops CEM when its cost plateaus.
