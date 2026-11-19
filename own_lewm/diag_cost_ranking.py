"""Does LeWM's planning cost rank actions correctly? (diagnostic for the oracle-waypoint, 1-block cem_mix setup)

Runs the same controller as eval_hierarchical.py (--controller cem_mix --horizon 1 --lookahead 5 --execute best) on the
official tasks. Every `--probe-every` replans it probes the current state: it takes a set of candidate blocks --
the plan CEM picked, the exact policy chunk, and `--n-rand` candidates from CEM's final population spread over its
cost quantiles -- and executes each one for real, by resetting to the same seed and replaying the executed action
prefix (MuJoCo is deterministic; checked once per episode). For every candidate it records:
  pred      : LeWM-predicted cost ||z_pred - z_wp||^2 (what CEM optimises)
  actual    : actual latent cost ||enc(real next frame) - z_wp||^2
  model_err : ||z_pred - enc(real next frame)||^2
  cube_d    : true cube-position distance to the expert's cube position at the waypoint frame
  arm_d     : true end-effector distance to the expert's end-effector position at the waypoint frame
Summary: per-probe Spearman rank correlations (pred vs actual, pred vs cube_d, actual vs cube_d, actual vs arm_d),
how often CEM's pick beats the policy chunk (on pred / actual / cube_d), and model error for each candidate kind.
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

ap = argparse.ArgumentParser()
ap.add_argument('--env', default='visual-cube-single-v0')
ap.add_argument('--lewm', required=True)
ap.add_argument('--gcbc', required=True)
ap.add_argument('--episodes', type=int, default=10, help='per task')
ap.add_argument('--samples', type=int, default=300)
ap.add_argument('--n-policy', type=int, default=200)
ap.add_argument('--policy-std', type=float, default=0.3)
ap.add_argument('--iters', type=int, default=30)
ap.add_argument('--topk', type=int, default=30)
ap.add_argument('--lookahead', type=int, default=5)
ap.add_argument('--search', type=int, default=20)
ap.add_argument('--probe-every', type=int, default=4, help='probe every k-th replan')
ap.add_argument('--n-rand', type=int, default=8, help='extra CEM-population candidates per probe')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', required=True)
a = ap.parse_args()
dev = 'cuda'
torch.manual_seed(a.seed); np.random.seed(a.seed)

env = ogbench.make_env_and_datasets(a.env.replace('-v0', '-play-v0'), env_only=True)
U = env.unwrapped
ck = torch.load(a.lewm, map_location=dev, weights_only=False)
wm = ck['model'].to(dev).eval()
amean, astd = ck['action_mean'].to(dev), ck['action_std'].to(dev)
FS, HIST = ck['frameskip'], ck['history']
A = len(amean); BD = FS * A
ZLO = ((-1 - amean) / astd).repeat(FS); ZHI = ((1 - amean) / astd).repeat(FS)
g = torch.load(a.gcbc, map_location=dev, weights_only=False)
assert g['chunk'] == FS
pol = GCBCPolicy(g['action_dim'], g['chunk']).to(dev).eval(); pol.load_state_dict(g['state_dict'])


def key(info, suffix):
    ks = [k for k in info if k.endswith(suffix)]
    assert ks, f'no info key ending in {suffix}: {sorted(info)}'
    return np.asarray(info[ks[0]], dtype=np.float64)[:3]


cube_pos = lambda info: key(info, 'block_0_pos')
arm_pos = lambda info: key(info, 'effector_pos')


@torch.no_grad()
def encode(frames):
    x = torch.from_numpy(np.ascontiguousarray(np.stack(frames))).to(dev)
    return torch.cat([wm.encode({'pixels': to_model_pixels(x[i:i + 256][:, None])})['emb'][:, 0] for i in range(0, len(x), 256)])


@torch.no_grad()
def predict(z0, cand):  # cand (S, BD) -> (S, D) predicted latent after one block
    return wm.predict(z0.expand(len(cand), 1, -1), wm.action_encoder(cand[:, None]))[:, -1]


@torch.no_grad()
def plan(z0, zt, prior):
    mean = torch.zeros(BD, device=dev); std = torch.ones_like(mean)
    for _ in range(a.iters):
        cand = mean + std * torch.randn(a.samples, BD, device=dev); cand[0] = mean
        cand[:a.n_policy] = prior + a.policy_std * torch.randn(a.n_policy, BD, device=dev); cand[0] = prior
        cand = torch.maximum(torch.minimum(cand, ZHI), ZLO)
        cost = (predict(z0, cand) - zt).pow(2).sum(-1)
        elite = cand[cost.topk(a.topk, largest=False).indices]
        mean, std = elite.mean(0), elite.std(0)
    return cand, cost


to_env = lambda z: (z.view(FS, A) * astd + amean).clamp(-1, 1).cpu().numpy()


def expert_rollout(ob, info):
    oracle = CubePlanOracle(env=env, noise=0.0)
    mid = U._cube_target_mocap_ids[0]
    info = dict(info)
    info['privileged/target_block'] = 0
    info['privileged/target_block_pos'] = U._data.mocap_pos[mid].copy()
    info['privileged/target_block_yaw'] = np.array([lie.SO3(wxyz=U._data.mocap_quat[mid]).compute_yaw_radians()])
    oracle.reset(ob, info)
    frames, cubes, arms, success = [ob], [cube_pos(info)], [arm_pos(info)], False
    for _ in range(200):
        ob, _, term, trunc, info = env.step(oracle.select_action(ob, info))
        frames.append(ob); cubes.append(cube_pos(info)); arms.append(arm_pos(info))
        success = success or bool(info.get('success', False))
        if oracle.done or term or trunc:
            break
    return frames, np.stack(cubes), np.stack(arms), success


def replay(task, seed, prefix, block):
    """Reset to the episode seed, replay the executed prefix, execute `block`; return final frame, cube, arm."""
    np.random.seed(seed)
    ob, info = env.reset(options=dict(task_id=task, render_goal=False), seed=seed)
    for act in list(prefix) + list(block):
        ob, _, _, _, info = env.step(act)
    return ob, cube_pos(info), arm_pos(info)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


probes, ep_success, replay_err = [], [], []
t0 = time.time()
for task in range(1, U.num_tasks + 1):
    for ep in range(a.episodes):
        seed = a.seed * 10000 + task * 1000 + ep
        np.random.seed(seed)
        ob, info = env.reset(options=dict(task_id=task, render_goal=False), seed=seed)
        frames, E_cube, E_arm, _ = expert_rollout(ob, info)
        E = encode(frames)
        np.random.seed(seed)
        ob, info = env.reset(options=dict(task_id=task, render_goal=False), seed=seed)
        goal = info['goal']
        gl = torch.from_numpy(np.ascontiguousarray(goal)).to(dev)[None]
        done, success, k, executed, n_replan, checked = False, False, 0, [], 0, False
        while not done:
            z0 = encode([ob])
            hi = min(len(E), k + a.search + 1)
            k = k + int((E[k:hi] - z0).pow(2).sum(-1).argmin())
            wi = min(k + a.lookahead, len(E) - 1)
            zt = E[wi]
            prior = ((pol.act(torch.from_numpy(np.ascontiguousarray(ob)).to(dev)[None], gl)[0] - amean) / astd).reshape(BD)
            prior = torch.maximum(torch.minimum(prior, ZHI), ZLO)
            cand, cost = plan(z0, zt, prior)
            best = cand[cost.argmin()]
            if n_replan % a.probe_every == 0:
                order = cost.argsort()
                qs = order[np.linspace(0, len(order) - 1, a.n_rand).round().astype(int)]
                C = torch.cat([best[None], prior[None], cand[qs]])
                kinds = ['cem_pick', 'policy'] + ['pop'] * a.n_rand
                zp = predict(z0, C)
                real = [replay(task, seed, executed, to_env(c)) for c in C]
                if not checked:  # determinism check: replaying the executed prefix reproduces the current frame
                    o_chk, _, _ = replay(task, seed, executed, [])
                    replay_err.append(float(np.abs(o_chk.astype(int) - ob.astype(int)).max())); checked = True
                za = encode([r[0] for r in real])
                probes.append(dict(
                    task=task, ep=ep, step=len(executed), kinds=kinds,
                    pred=(zp - zt).pow(2).sum(-1).tolist(), actual=(za - zt).pow(2).sum(-1).tolist(),
                    model_err=(zp - za).pow(2).sum(-1).tolist(),
                    cube_d=[float(np.linalg.norm(r[1] - E_cube[wi])) for r in real],
                    arm_d=[float(np.linalg.norm(r[2] - E_arm[wi])) for r in real]))
            n_replan += 1
            for act in to_env(best):
                ob, _, term, trunc, info = env.step(act)
                executed.append(act)
                success = success or bool(info.get('success', False))
                done = term or trunc
                if done:
                    break
        ep_success.append(success)
    print(f'task {task}: success {np.mean(ep_success[-a.episodes:])*100:.0f}%  probes {len(probes)}  '
          f'({time.time()-t0:.0f}s)', flush=True)

# ---------------------------------------------------------------- summary
def per_probe(f):
    v = [f(p) for p in probes]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)), float(np.median(v))

S = dict(
    n_probes=len(probes), success=float(np.mean(ep_success) * 100), replay_max_pixel_err=max(replay_err),
    spearman_pred_vs_actual=per_probe(lambda p: spearman(p['pred'], p['actual'])),
    spearman_pred_vs_cube=per_probe(lambda p: spearman(p['pred'], p['cube_d'])),
    spearman_actual_vs_cube=per_probe(lambda p: spearman(p['actual'], p['cube_d'])),
    spearman_actual_vs_arm=per_probe(lambda p: spearman(p['actual'], p['arm_d'])),
    pick_beats_policy=dict((m, float(np.mean([p[m][0] < p[m][1] for p in probes]))) for m in ['pred', 'actual', 'cube_d', 'arm_d']),
)
for kind in ['cem_pick', 'policy', 'pop']:
    S[kind] = {m: float(np.mean([v for p in probes for v, kk in zip(p[m], p['kinds']) if kk == kind]))
               for m in ['pred', 'actual', 'model_err', 'cube_d', 'arm_d']}
print(json.dumps(S, indent=1))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(dict(summary=S, args=vars(a), probes=probes), open(a.out, 'w'))
print('DIAG_DONE', flush=True)
