"""Per-phase quality of the low level: one 5-step hop at a time, starting from expert states.

For each episode of the official tasks, OGBench's scripted expert (noise 0) is run from the reset. Then, for every
block start t along the expert path, the env is restored to the expert state at t (reset with the same seed and
replay the expert's actions; MuJoCo is deterministic, checked on every hop) and each controller executes ONE block
(5 actions) aimed at the expert frame t+5 (the oracle waypoint of the 1-block benchmark):
  policy     : the GCBC chunk alone (conditioned on the task goal, as in the benchmark)
  cem_policy : CEM + policy proposals, 1-block horizon, execute the best candidate (the 23% benchmark controller)
  cem        : plain CEM, same settings without the policy
Every hop is labelled with one of 7 phases from the expert trajectory (cube height + gripper contact):
  approach, grasp, lift, carry, place, release, retreat
and scored against the expert's own state at t+5: cube error and end-effector error (m). A hop "succeeds" when the
cube is within --cube-tol and the end-effector within --ee-tol of the expert's.
Usage: python phase_hops.py --lewm lewm.pt --gcbc policy_chunk5.pt --episodes 10 --out results/phase_hops.json
"""
import os, sys, json, time, argparse
from collections import defaultdict
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
ap.add_argument('--cube-tol', type=float, default=0.01)
ap.add_argument('--ee-tol', type=float, default=0.02)
ap.add_argument('--lift', type=float, default=0.01, help='cube counts as lifted this far above its rest height')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', required=True)
a = ap.parse_args()
dev = 'cuda'
torch.manual_seed(a.seed)

env = ogbench.make_env_and_datasets(a.env.replace('-v0', '-play-v0'), env_only=True)
U = env.unwrapped
ck = torch.load(a.lewm, map_location=dev, weights_only=False)
wm = ck['model'].to(dev).eval()
amean, astd = ck['action_mean'].to(dev), ck['action_std'].to(dev)
FS, HIST = ck['frameskip'], ck['history']
A = len(amean); BD = FS * A
ZLO = ((-1 - amean) / astd).repeat(FS); ZHI = ((1 - amean) / astd).repeat(FS)
g = torch.load(a.gcbc, map_location=dev, weights_only=False)
assert g['chunk'] == FS, 'policy must output exactly one block'
pol = GCBCPolicy(g['action_dim'], g['chunk']).to(dev).eval(); pol.load_state_dict(g['state_dict'])


def key(info, suffix, n=None):
    ks = [k for k in info if k.endswith(suffix)]
    assert ks, f'no info key ending in {suffix}: {sorted(info)}'
    v = np.asarray(info[ks[0]], dtype=np.float64).reshape(-1)
    return v[:n] if n else v


cube = lambda info: key(info, 'block_0_pos', 3)
ee = lambda info: key(info, 'effector_pos', 3)
contact = lambda info: float(key(info, 'gripper_contact')[0]) > 0.5


@torch.no_grad()
def encode(frames):
    x = torch.from_numpy(np.ascontiguousarray(np.stack(frames))).to(dev)
    return torch.cat([wm.encode({'pixels': to_model_pixels(x[i:i + 256][:, None])})['emb'][:, 0]
                      for i in range(0, len(x), 256)])


@torch.no_grad()
def plan(z0, zt, prior):  # same as eval_hierarchical.py with --horizon 1 --execute best
    mean = torch.zeros(1, BD, device=dev); std = torch.ones_like(mean)
    for _ in range(a.iters):
        cand = mean + std * torch.randn(a.samples, 1, BD, device=dev); cand[0] = mean
        if prior is not None:
            cand[:a.n_policy, 0] = prior + a.policy_std * torch.randn(a.n_policy, BD, device=dev); cand[0, 0] = prior
        cand = torch.maximum(torch.minimum(cand, ZHI), ZLO)
        emb = wm.predict(z0.expand(a.samples, 1, -1), wm.action_encoder(cand))[:, -1]
        cost = (emb - zt).pow(2).sum(-1)
        elite = cand[cost.topk(a.topk, largest=False).indices]
        mean, std = elite.mean(0), elite.std(0)
    return to_env(cand[cost.argmin()][0])


to_env = lambda zblock: (zblock.view(FS, A) * astd + amean).clamp(-1, 1).cpu().numpy()


def reset(task, seed):
    np.random.seed(seed)
    return env.reset(options=dict(task_id=task, render_goal=False), seed=seed)


def expert(task, seed):
    ob, info = reset(task, seed)
    oracle = CubePlanOracle(env=env, noise=0.0)
    mid = U._cube_target_mocap_ids[0]
    oi = dict(info)
    oi['privileged/target_block'] = 0
    oi['privileged/target_block_pos'] = U._data.mocap_pos[mid].copy()
    oi['privileged/target_block_yaw'] = np.array([lie.SO3(wxyz=U._data.mocap_quat[mid]).compute_yaw_radians()])
    oracle.reset(ob, oi)
    obs, infos, acts, ok, cur = [ob], [info], [], False, oi
    for _ in range(200):
        act = oracle.select_action(ob, cur)
        ob, _, term, trunc, cur = env.step(act)
        obs.append(ob); infos.append(cur); acts.append(act)
        ok = ok or bool(cur.get('success', False))
        if oracle.done or trunc:  # not `term`: the env terminates on success, before place/release/retreat
            break
    return obs, infos, np.array(acts), ok, info['goal']


def restore(task, seed, acts, t):
    ob, info = reset(task, seed)
    for k in range(t):
        ob, _, _, _, info = env.step(acts[k])
    return ob, info


def label(infos, t, z_rest):
    up = lambda i: cube(infos[i])[2] > z_rest + a.lift
    lifted = [i for i in range(len(infos)) if up(i)]
    if not lifted:
        return None
    u0, u1 = up(t), up(t + FS)
    if u0 and u1: return 'carry'
    if not u0 and u1: return 'lift'
    if u0 and not u1: return 'place'
    touch = any(contact(infos[i]) for i in range(t, t + FS + 1))
    if t + FS <= lifted[0]:
        return 'grasp' if touch else 'approach'
    if t >= lifted[-1]:
        return 'release' if touch else 'retreat'
    return None  # rests in the middle of the episode (e.g. a re-grasp): not one of the 7 phases


hops, skipped, t0 = [], [], time.time()
for task in range(1, U.num_tasks + 1):
    for ep in range(a.episodes):
        seed = a.seed * 10000 + task * 1000 + ep
        obs, infos, acts, ok, goal = expert(task, seed)
        if not ok:
            continue
        z_rest = cube(infos[0])[2]
        E = encode(obs)
        gl = torch.from_numpy(np.ascontiguousarray(goal)).to(dev)[None]
        for t in range(0, len(acts) - FS + 1, FS):
            ph = label(infos, t, z_rest)
            if ph is None:
                continue
            ob, info = restore(task, seed, acts, t)
            drift = max(np.abs(cube(info) - cube(infos[t])).max(), np.abs(ee(info) - ee(infos[t])).max())
            if drift > 1e-4:  # replay did not reproduce the expert state: skip and count
                skipped.append(dict(task=task, ep=ep, t=t, drift=float(drift), pix=float(np.abs(ob.astype(int) - obs[t]).max())))
                continue
            o = torch.from_numpy(np.ascontiguousarray(ob)).to(dev)[None]
            pchunk = pol.act(o, gl)[0]
            prior = ((pchunk - amean) / astd).reshape(BD)
            blocks = {'policy': pchunk.clamp(-1, 1).cpu().numpy(),
                      'cem_policy': plan(E[t:t + 1], E[t + FS], prior),
                      'cem': plan(E[t:t + 1], E[t + FS], None)}
            rec = dict(task=task, ep=ep, t=t, phase=ph)
            for name, blk in blocks.items():
                ob2, info2 = restore(task, seed, acts, t)
                for act in blk:
                    ob2, _, _, _, info2 = env.step(act)
                ce = float(np.linalg.norm(cube(info2) - cube(infos[t + FS])))
                ee_ = float(np.linalg.norm(ee(info2) - ee(infos[t + FS])))
                rec[name] = dict(cube_err=ce, ee_err=ee_, ok=bool(ce <= a.cube_tol and ee_ <= a.ee_tol))
            hops.append(rec)
    print(f'task {task}: {len(hops)} hops so far ({time.time()-t0:.0f}s)', flush=True)

PH = ['approach', 'grasp', 'lift', 'carry', 'place', 'release', 'retreat']
C = ['policy', 'cem_policy', 'cem']
summary = {}
for ph in PH + ['all']:
    hs = [h for h in hops if ph == 'all' or h['phase'] == ph]
    if not hs:
        continue
    summary[ph] = dict(n=len(hs), **{c: dict(success=100 * float(np.mean([h[c]['ok'] for h in hs])),
                                            cube_err_cm=100 * float(np.median([h[c]['cube_err'] for h in hs])),
                                            ee_err_cm=100 * float(np.median([h[c]['ee_err'] for h in hs]))) for c in C})
print(f'{"phase":9s} {"n":>4s}  ' + '  '.join(f'{c:>24s}' for c in C))
print(f'{"":9s} {"":4s}  ' + '  '.join(f'{"succ%  cube_cm  ee_cm":>24s}' for c in C))
for ph, s in summary.items():
    print(f'{ph:9s} {s["n"]:4d}  ' + '  '.join(f'{s[c]["success"]:6.0f}  {s[c]["cube_err_cm"]:7.2f}  {s[c]["ee_err_cm"]:5.2f}'.rjust(24) for c in C))
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
print(f'skipped {len(skipped)} hops whose replay drifted (max {max([x["drift"] for x in skipped], default=0):.2g} m)')
json.dump(dict(summary=summary, args=vars(a), hops=hops, skipped=skipped), open(a.out, 'w'))
print('PHASE_HOPS_DONE', flush=True)
