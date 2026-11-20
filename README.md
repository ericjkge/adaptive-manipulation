# Hierarchical World-Model Planning for Cube Pick-and-Place

MIT 6.4212 final project (Fall 2026). Can a hierarchical planner solve long-horizon (200-step) pick-and-place from
pixels on OGBench `visual-cube-single`?

**The hierarchy:**
- **High level:** a waypoint generator proposes a latent subgoal 5 or 10 steps ahead.
- **Low level:** CEM in [LeWM](https://arxiv.org/abs/2603.19312) latent space picks the action chunk whose predicted
  outcome is closest to the waypoint. A GCBC policy optionally proposes candidate chunks.
- **Execution:** run one chunk, then replan.

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

**Where closed-loop episodes fail** ([`own_lewm/failure_phases.py`](own_lewm/failure_phases.py)):
- **Setup:** the 23% setting (LeWM + CEM + policy, oracle waypoints 5 steps ahead) against the policy alone, on
  the same seeds: 5 tasks × 20 episodes. Each episode is labelled with the furthest phase the robot reached, from
  the true sim state.

| Furthest phase reached | LeWM + CEM + policy | Policy alone |
|---|---|---|
| Approach (never touched the cube) | 9 | 0 |
| Grasp (touched it, never lifted it) | 17 | 0 |
| Lift (never held it up for a full block) | 1 | 0 |
| Carry (never brought it over the target) | 44 | 72 |
| Place (over the target, not within 4 cm) | 3 | 8 |
| **Success** | **26** | **20** |

- **Carry is the main failure for both.** All 44 of LeWM + CEM + policy's carry failures dropped the cube. The policy
  dropped it in 44 episodes and was still holding it at the 200-step limit in the other 28.
- **LeWM + CEM + policy also fails before the lift** (26 episodes; the policy never does). The arm gets within about
  3.5 cm of the cube but never grasps it.
- **Waypoint tracking cannot recover.** On failed episodes it ends at a median 84% of the expert path, and 74% of
  replans make no progress. After a drop it keeps aiming at the late, placing waypoints, because progress along the
  expert path only moves forward.

## Next steps

1. **Better waypoints:** the low level reaches its waypoints well, including through contact. One block
   from expert states, LeWM + CEM + policy reaches 96% of grasp, lift and place waypoints (vs. 79/37/100% for the
   policy). Closed-loop failures come from the waypoints instead: tracking a fixed expert path stalls and only moves
   forward, so after a miss or a drop it keeps aiming at waypoints that no longer fit the scene. The 23% "oracle"
   is therefore not a ceiling.
   - **Recovery:** waypoints conditioned on the current state should re-plan the pick from wherever the cube is.
   - **Quality:** close the gap from 8% (flow) and beyond the expert-path oracle.
2. **LeWM refinement:** revisit once waypoints recover. Check whether the carry drops persist with good
   waypoints. Approach is the one phase where LeWM + CEM + policy is weaker than the policy alone (58% vs. 64%).

## Repository

```
official_lewm/     authors' checkpoint (quentinll/lewm-cube) + official eval
  waypoints/       expert waypoints on full tasks
  phase_split/     success by manipulation phase
own_lewm/          hierarchy PoC: train_*.py, eval_flat.py, eval_hierarchical.py, diag_cost_ranking.py,
                   failure_phases.py, results/
```

**Setup:** Python 3.10, `uv pip install -r official_lewm/requirements.lock.txt`, plus `ogbench` for `own_lewm/`.

**Checkpoints:** on the public HF repo [`ericjkge/ogb-cube`](https://huggingface.co/ericjkge/ogb-cube).
