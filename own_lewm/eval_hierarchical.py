"""Oracle-waypoint test on the official OGBench tasks (default: visual-cube-single-v0, 5 tasks, 200-step budget).

For every episode, OGBench's scripted expert (CubePlanOracle, noise 0) is first run from the same reset to record an
expert trajectory of pixel frames. The controller is then run from the identical reset. At each replan it tracks its
progress along the expert path (nearest expert latent, searched forward from the last match) and targets the expert
frame `--lookahead` env steps further on (default 10 = 2 blocks). LeWM is the official 88-epoch PoC model.

Controllers (--controller):
  cem     : plain CEM, horizon 2 blocks, all candidates Gaussian around the CEM mean (init 0, std 1).
  cem_mix : same, but in every iteration the first `--n-policy` candidates take block 0 from N(policy chunk, 0.3^2)
            (the 5-action GCBC chunk, z-scored); their block 1 and all other blocks are CEM-Gaussian. Candidate 0 is the
            exact policy chunk. Elites are the top-k of all candidates; the policy proposals stay anchored to the policy.
Waypoint source (--waypoints): 'oracle' (above) or 'flow': the learned latent waypoint generator (train_waypoint_flow.py),
conditioned on the latents at the last `history` replans (one block apart; padded with the first) and the goal latent;
CEM targets its waypoint `--flow-index` (0: 5 env steps ahead, 1: 10 steps ahead). One sample per replan.
Both: terminal cost ||z_pred(H) - z_waypoint||^2; execute block 0 of the elite mean (--execute mean) or of the
lowest-cost final-iteration candidate (--execute best); replan every block.
"""
import os, sys, json, time, argparse
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import torch

sys.path.insert(0, os.environ.get('LEWM_REPO', '/workspace/le-wm'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ogbench  # noqa: E402
from ogbench.manipspace import lie  # noqa: E402
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle  # noqa: E402
from common import to_model_pixels  # noqa: E402
from train_policy import GCBCPolicy  # noqa: E402
from train_waypoint_flow import WaypointFlow  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--env', default='visual-cube-single-v0')
ap.add_argument('--controller', choices=['cem', 'cem_mix'], required=True)
ap.add_argument('--lewm', required=True)
ap.add_argument('--gcbc', default=None)
ap.add_argument('--episodes', type=int, default=50, help='per task')
ap.add_argument('--samples', type=int, default=300)
ap.add_argument('--n-policy', type=int, default=200)
ap.add_argument('--policy-std', type=float, default=0.3)
ap.add_argument('--iters', type=int, default=30)
ap.add_argument('--topk', type=int, default=30)
ap.add_argument('--horizon', type=int, default=2, help='blocks')
ap.add_argument('--lookahead', type=int, default=10, help='env steps ahead of current expert progress')
ap.add_argument('--search', type=int, default=20, help='expert frames searched forward for progress')
ap.add_argument('--waypoints', choices=['oracle', 'flow'], default='oracle')
ap.add_argument('--flow', default=None, help='waypoint flow checkpoint')
ap.add_argument('--flow-index', type=int, default=0)
ap.add_argument('--flow-steps', type=int, default=16)
ap.add_argument('--execute', choices=['best', 'mean'], default='mean', help='plan executed after the last iteration')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', required=True)
a = ap.parse_args()
dev = 'cuda'
torch.manual_seed(a.seed); np.random.seed(a.seed)

env = ogbench.make_env_and_datasets(a.env.replace('-v0', '-play-v0'), env_only=True)
U = env.unwrapped
n_tasks = U.num_tasks

ck = torch.load(a.lewm, map_location=dev, weights_only=False)
wm = ck['model'].to(dev).eval()
amean, astd = ck['action_mean'].to(dev), ck['action_std'].to(dev)
FS, HIST = ck['frameskip'], ck['history']
A = len(amean)
BD = FS * A
# valid action range [-1, 1] in z-scored units, tiled over one block: candidates are clipped to it before scoring,
# so LeWM evaluates exactly the actions the env will execute
ZLO = ((-1 - amean) / astd).repeat(FS)
ZHI = ((1 - amean) / astd).repeat(FS)

pol = None
if a.controller == 'cem_mix':
    g = torch.load(a.gcbc, map_location=dev, weights_only=False)
    assert g['chunk'] == FS, 'policy must output exactly one block'
    pol = GCBCPolicy(g['action_dim'], g['chunk']).to(dev).eval()
    pol.load_state_dict(g['state_dict'])

flow = None
if a.waypoints == 'flow':
    fc = torch.load(a.flow, map_location=dev, weights_only=False)
    flow = WaypointFlow(**fc['cfg']).to(dev).eval()
    flow.load_state_dict(fc['state_dict'])
    assert a.horizon == a.flow_index + 1, 'CEM horizon must end at the targeted waypoint'


@torch.no_grad()
def encode(frames):  # list/array of (H, W, 3) uint8 -> (N, D)
    x = torch.from_numpy(np.ascontiguousarray(np.stack(frames))).to(dev)
    out = []
    for i in range(0, len(x), 256):
        out.append(wm.encode({'pixels': to_model_pixels(x[i:i + 256][:, None])})['emb'][:, 0])
    return torch.cat(out)


@torch.no_grad()
def rollout_cost(z0, zt, cand):  # cand (S, H, BD) z-scored -> (S,)
    S = cand.shape[0]
    emb = z0.expand(S, 1, -1)
    ae = wm.action_encoder(cand)
    for k in range(cand.shape[1]):
        lo = max(0, emb.shape[1] - HIST)
        nxt = wm.predict(emb[:, lo:], ae[:, lo:k + 1])[:, -1:]
        emb = torch.cat([emb, nxt], 1)
    return (emb[:, -1] - zt).pow(2).sum(-1)


@torch.no_grad()
def plan(z0, zt, prior=None):  # prior: (BD,) z-scored policy block or None
    mean = torch.zeros(a.horizon, BD, device=dev)
    std = torch.ones_like(mean)
    for _ in range(a.iters):
        cand = mean + std * torch.randn(a.samples, a.horizon, BD, device=dev)
        cand[0] = mean
        if prior is not None:
            n = a.n_policy
            cand[:n, 0] = prior + a.policy_std * torch.randn(n, BD, device=dev)
            cand[0, 0] = prior
        cand = torch.maximum(torch.minimum(cand, ZHI), ZLO)
        cost = rollout_cost(z0, zt, cand)
        elite = cand[cost.topk(a.topk, largest=False).indices]
        mean, std = elite.mean(0), elite.std(0)
    # 'best': lowest-cost candidate of the final iteration (an evaluated plan); 'mean': refitted elite mean
    out = cand[cost.argmin()] if a.execute == 'best' else mean
    return (out[0].view(FS, A) * astd + amean).clamp(-1, 1).cpu().numpy()


def expert_rollout(ob, info):
    """Run the scripted expert from the current state; returns frames (incl. the start) and whether it succeeded."""
    oracle = CubePlanOracle(env=env, noise=0.0)
    mid = U._cube_target_mocap_ids[0]
    info = dict(info)
    info['privileged/target_block'] = 0
    info['privileged/target_block_pos'] = U._data.mocap_pos[mid].copy()
    info['privileged/target_block_yaw'] = np.array([lie.SO3(wxyz=U._data.mocap_quat[mid]).compute_yaw_radians()])
    oracle.reset(ob, info)
    frames, success = [ob], False
    for _ in range(200):
        ob, _, term, trunc, info = env.step(oracle.select_action(ob, info))
        frames.append(ob)
        success = success or bool(info.get('success', False))
        if oracle.done or term or trunc:
            break
    return frames, success


res, t_plan, n_plan, expert_ok, expert_len = {}, 0.0, 0, [], []
t0 = time.time()
for task in range(1, n_tasks + 1):
    succ = []
    for ep in range(a.episodes):
        seed = a.seed * 10000 + task * 1000 + ep
        np.random.seed(seed)  # oracle keyframe jitter
        ob, info = env.reset(options=dict(task_id=task, render_goal=False), seed=seed)
        if flow is None:
            frames, ok = expert_rollout(ob, info)
            expert_ok.append(ok); expert_len.append(len(frames) - 1)
            E = encode(frames)
            ob, info = env.reset(options=dict(task_id=task, render_goal=False), seed=seed)
        goal = info['goal']
        zg = encode([goal])
        done, success, queue, k, hist = False, False, [], 0, []
        if flow is not None:
            expert_ok.append(float('nan')); expert_len.append(float('nan'))
        while not done:
            if not queue:
                tq = time.time()
                z0 = encode([ob])
                if flow is None:
                    hi = min(len(E), k + a.search + 1)
                    k = k + int((E[k:hi] - z0).pow(2).sum(-1).argmin())
                    zt = E[min(k + a.lookahead, len(E) - 1)]
                else:
                    hist = (hist + [z0[0]])[-flow.cfg['history']:]
                    hh = torch.stack([hist[0]] * (flow.cfg['history'] - len(hist)) + hist)[None]
                    zt = flow.sample(hh, zg, steps=a.flow_steps)[0, a.flow_index]
                prior = None
                if pol is not None:
                    o = torch.from_numpy(np.ascontiguousarray(ob)).to(dev)[None]
                    gl = torch.from_numpy(np.ascontiguousarray(goal)).to(dev)[None]
                    prior = ((pol.act(o, gl)[0] - amean) / astd).reshape(BD)
                queue = list(plan(z0, zt, prior))
                t_plan += time.time() - tq; n_plan += 1
            ob, _, term, trunc, info = env.step(queue.pop(0))
            success = success or bool(info.get('success', False))
            done = term or trunc
        succ.append(success)
    res[f'task{task}'] = float(np.mean(succ) * 100)
    print(f'{a.controller} task {task}: {res[f"task{task}"]:.0f}%  expert {np.mean(expert_ok[-a.episodes:])*100:.0f}% '
          f'len {np.mean(expert_len[-a.episodes:]):.0f}  ({time.time()-t0:.0f}s)', flush=True)

res['mean'] = float(np.mean([res[f'task{t}'] for t in range(1, n_tasks + 1)]))
res.update(controller=a.controller, env=a.env, episodes_per_task=a.episodes, seconds_per_query=t_plan / max(1, n_plan),
           expert_success=float(np.nanmean(expert_ok) * 100) if flow is None else None, expert_len=float(np.nanmean(expert_len)) if flow is None else None, args=vars(a))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, 'w'), indent=1)
print(f'MEAN {res["mean"]:.1f}%  expert {res["expert_success"]}  EVAL_DONE', flush=True)
