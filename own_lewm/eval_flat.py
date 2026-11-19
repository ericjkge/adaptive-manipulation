"""Evaluate on the official OGBench tasks (default: visual-cube-single-v0, 5 tasks, 200-step budget, env success).

Controllers (--controller):
  gcbc      : GCBC policy alone; predicts an action chunk (5 or 25 actions), executes `--exec` actions of it, then re-queries.
  cem       : LeWM + CEM, the official LeWM planning setup: horizon 5 blocks (25 actions), 300 samples, 30 iterations,
              top 30, terminal cost ||z_pred(H) - z_goal||^2, receding horizon `--replan` blocks.
  cem_prior : as cem, but the initial CEM mean is the GCBC chunk (all blocks it covers; the rest start at 0 = mean action).
Writes a JSON with per-task success rates and timing.
"""
import os, sys, json, time, argparse
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import torch

sys.path.insert(0, os.environ.get('LEWM_REPO', '/workspace/le-wm'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ogbench  # noqa: E402
from common import to_model_pixels  # noqa: E402
from train_policy import GCBCPolicy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--env', default='visual-cube-single-v0')
ap.add_argument('--controller', choices=['gcbc', 'cem', 'cem_prior'], required=True)
ap.add_argument('--gcbc', default=None)
ap.add_argument('--lewm', default=None)
ap.add_argument('--episodes', type=int, default=50, help='per task')
ap.add_argument('--exec', type=int, default=5, help='gcbc: actions executed per query (1..chunk)')
ap.add_argument('--samples', type=int, default=300)
ap.add_argument('--iters', type=int, default=30)
ap.add_argument('--topk', type=int, default=30)
ap.add_argument('--horizon', type=int, default=5)
ap.add_argument('--replan', type=int, default=5, help='blocks executed per plan')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', required=True)
a = ap.parse_args()
dev = 'cuda'
torch.manual_seed(a.seed); np.random.seed(a.seed)

env = ogbench.make_env_and_datasets(a.env.replace('-v0', '-play-v0') if 'play' not in a.env else a.env, env_only=True)
n_tasks = env.unwrapped.num_tasks

pol = None
if a.gcbc:
    ck = torch.load(a.gcbc, map_location=dev, weights_only=False)
    pol = GCBCPolicy(ck['action_dim'], ck['chunk']).to(dev).eval()
    pol.load_state_dict(ck['state_dict'])

if a.controller.startswith('cem'):
    ck = torch.load(a.lewm, map_location=dev, weights_only=False)
    wm = ck['model'].to(dev).eval()
    amean, astd = ck['action_mean'].to(dev), ck['action_std'].to(dev)
    FS, HIST = ck['frameskip'], ck['history']
    A = len(amean)
    # valid action range [-1, 1] in z-scored units, tiled over one block (candidates are clipped before scoring)
    ZLO, ZHI = ((-1 - amean) / astd).repeat(FS), ((1 - amean) / astd).repeat(FS)

    @torch.no_grad()
    def encode(u8):  # (H, W, 3) uint8 -> (1, D)
        x = to_model_pixels(torch.from_numpy(np.ascontiguousarray(u8)).to(dev)[None, None])
        return wm.encode({'pixels': x})['emb'][:, 0]

    @torch.no_grad()
    def rollout_cost(z0, zg, cand):  # cand (S, H, FS*A) z-scored; returns (S,)
        S = cand.shape[0]
        emb = z0.expand(S, 1, -1)
        ae = wm.action_encoder(cand)
        for k in range(cand.shape[1]):
            lo = max(0, emb.shape[1] - HIST)
            nxt = wm.predict(emb[:, lo:], ae[:, lo:k + 1])[:, -1:]
            emb = torch.cat([emb, nxt], 1)
        return (emb[:, -1] - zg).pow(2).sum(-1)

    @torch.no_grad()
    def cem_plan(obs, goal, init_block0=None):
        z0, zg = encode(obs), encode(goal)
        mean = torch.zeros(a.horizon, FS * A, device=dev)
        if init_block0 is not None:  # (n_blocks, FS*A) z-scored prior plan
            n = min(len(init_block0), a.horizon)
            mean[:n] = init_block0[:n]
        std = torch.ones_like(mean)
        for _ in range(a.iters):
            cand = mean + std * torch.randn(a.samples, *mean.shape, device=dev)
            cand[0] = mean
            cand = torch.maximum(torch.minimum(cand, ZHI), ZLO)
            cost = rollout_cost(z0, zg, cand)
            elite = cand[cost.topk(a.topk, largest=False).indices]
            mean, std = elite.mean(0), elite.std(0)
        acts = mean[:a.replan].reshape(a.replan * FS, A) * astd + amean  # raw scale
        return acts.clamp(-1, 1).cpu().numpy()


def policy_chunk(obs, goal):
    o = torch.from_numpy(np.ascontiguousarray(obs)).to(dev)[None]
    g = torch.from_numpy(np.ascontiguousarray(goal)).to(dev)[None]
    return pol.act(o, g)[0]  # (5, A) raw scale


res, t_plan, n_plan = {}, 0.0, 0
t0 = time.time()
for task in range(1, n_tasks + 1):
    succ = []
    for ep in range(a.episodes):
        obs, info = env.reset(options=dict(task_id=task, render_goal=False), seed=a.seed * 10000 + task * 1000 + ep)
        goal = info['goal']
        done, success, queue = False, False, []
        while not done:
            if not queue:
                tq = time.time()
                if a.controller == 'gcbc':
                    queue = list(policy_chunk(obs, goal)[:a.exec].cpu().numpy())
                else:
                    init = None
                    if a.controller == 'cem_prior':
                        init = ((policy_chunk(obs, goal) - amean) / astd).reshape(-1, FS * A)
                    queue = list(cem_plan(obs, goal, init))
                t_plan += time.time() - tq; n_plan += 1
            obs, _, terminated, truncated, info = env.step(queue.pop(0))
            success = success or bool(info.get('success', False))
            done = terminated or truncated
        succ.append(success)
    res[f'task{task}'] = float(np.mean(succ) * 100)
    print(f'{a.controller} task {task}: {res[f"task{task}"]:.0f}%  ({time.time()-t0:.0f}s)', flush=True)

res['mean'] = float(np.mean([res[f'task{t}'] for t in range(1, n_tasks + 1)]))
res.update(controller=a.controller, env=a.env, episodes_per_task=a.episodes, seconds_per_query=t_plan / max(1, n_plan),
           args=vars(a))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, 'w'), indent=1)
print(f'MEAN {res["mean"]:.1f}%  EVAL_DONE', flush=True)
