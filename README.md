# Hierarchical World-Model Planning for Cube Pick-and-Place

MIT 6.4212 final project (Fall 2026). Can a hierarchical planner solve long-horizon (200-step) pick-and-place from
pixels on OGBench `visual-cube-single`?

**The hierarchy:**
- **High level:** a waypoint generator proposes a latent subgoal 5 or 10 steps ahead.
- **Low level:** CEM in [LeWM](https://arxiv.org/abs/2603.19312) latent space picks the action chunk whose predicted
  outcome is closest to the waypoint. A GCBC policy optionally proposes candidate chunks.
- **Execution:** run one chunk, then replan.

**Two open problems:**
1. **Learned waypoints are weak:** flow waypoints get 8% vs. 23% with oracle waypoints.
2. **LeWM adds little over the policy:** 23% vs. 21.6% for the policy alone. Its errors concentrate in the contact
   phases.

## Manipulation phases

We split every pick-and-place into phases by whether the cube is resting on the table or lifted (more than 1 cm
above its rest height), at the start and at the end of a window:

| Phase | Cube at start → end | What happens |
|---|---|---|
| Approach | resting | the arm moves to the cube |
| Grasp | resting | the gripper closes on the cube |
| Lift | resting → lifted | the cube is picked up |
| Carry | lifted → lifted | the cube moves through the air |
| Place | lifted → resting | the cube is set down |
| Release | resting | the gripper opens and lets go of the cube |
| Retreat | resting | the arm moves away from the cube |

Note: approach, grasp, release and retreat all leave the cube resting, so cube height alone cannot tell them apart.
Separating them needs gripper state (open/closed, in contact).

## Models

All three are trained by us on OGBench `visual-cube-single-play`: 1,000 play trajectories (about 1M frames) at 64×64 px.
Actions are 5-d.

| | LeWM (world model) | GCBC policy | Flow waypoint model |
|---|---|---|---|
| **Input** | last 3 frames (5 steps apart) + a 25-d action block per frame (5 actions × 5 dims) | current frame + goal frame, stacked channel-wise | last 3 LeWM latents (t, t−5, t−10) + goal latent |
| **Output** | 192-d latent of the frame 5 steps later; rolled out autoregressively for planning | chunk of 5 actions | 2 waypoint latents, 5 and 10 steps ahead |
| **Architecture** | ViT-tiny encoder (patch 8) + 6-layer autoregressive predictor, about 15M params | IMPALA-small CNN + MLP (512×3), OGBench's baseline | 4-layer AdaLN transformer, width 512, on frozen LeWM latents |
| **Loss** | MSE on the next latent + 0.09 × SIGReg | MSE on z-scored actions; goals sampled from the same trajectory's future | rectified flow matching; sampled with 16 Euler steps from noise |
| **Training** | official recipe: AdamW, lr 5e-5, batch 128, 88 epochs (about 9 h, L40S) | Adam, lr 3e-4, batch 256, 500k steps, random-shift augmentation | AdamW, lr 3e-4 → 3e-5, batch 1024, 100k steps, EMA |

## Results

**Hierarchy PoC** ([`own_lewm/`](own_lewm/)):
- **Eval:** the 5 official tasks, a 200-step budget, 50 episodes per task (20 where marked †). A block is 5 actions.

| Controller | Waypoints | Plan | Success |
|---|---|---|---|
| Policy alone | — | — | 21.6% |
| LeWM + CEM | — | 5 blocks | 0% |
| LeWM + CEM | oracle, 10 steps ahead | 2 blocks | 0% |
| LeWM + CEM + policy | oracle, 10 steps ahead | 2 blocks | 12.8% |
| LeWM + CEM † | oracle, 5 steps ahead | 1 block | 1% |
| **LeWM + CEM + policy †** | **oracle, 5 steps ahead** | **1 block** | **23%** |
| LeWM + CEM + policy † | flow, 5 steps ahead | 1 block | 8% |

## Next steps

1. **Better waypoints:** close the gap from 8% (flow) to 23% (oracle). For example, measure how far flow waypoints
   are from the expert path in true state, or propose waypoints from policy rollouts.
2. **A better low level:** make LeWM add value over the policy at contact. Options are better contact dynamics, a
   cube-aware cost, or tighter use of the policy prior. Benchmark in the oracle, 5-steps-ahead, 1-block setting
   against the policy alone (21.6%).

## Repository

```
official_lewm/     authors' checkpoint (quentinll/lewm-cube) + official eval
  waypoints/       expert waypoints on full tasks
  phase_split/     success by manipulation phase
own_lewm/          hierarchy PoC: train_*.py, eval_flat.py, eval_hierarchical.py, diag_cost_ranking.py, results/
```

**Setup:** Python 3.10, `uv pip install -r official_lewm/requirements.lock.txt`, plus `ogbench` for `own_lewm/`.

**Checkpoints:** on the public HF repo [`ericjkge/ogb-cube`](https://huggingface.co/ericjkge/ogb-cube).
