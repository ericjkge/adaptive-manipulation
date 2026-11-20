"""Where do closed-loop episodes fail? CEM + policy (oracle waypoints) vs. the policy alone, on the same seeds.

Runs the two benchmark controllers on the official tasks (200-step budget, env success), with identical seeds:
  cem_policy : eval_hierarchical.py --controller cem_mix --horizon 1 --lookahead 5 --execute best (the 23% setting)
  policy     : eval_flat.py --controller gcbc --exec 5 (the policy alone)
Each episode is labelled with the furthest phase the robot itself reached (true sim state):
  approach : never touched the cube
  grasp    : touched it, cube never lifted (> --lift above its rest height)
  lift     : lifted, but never held up for a full block (5 consecutive steps)
  carry    : held up for a block or more, but never brought within --near (xy) of the target
  place    : brought over the target, but never within the env's 4 cm success radius
  success  : the env's success test (the episode terminates there, so release/retreat are never reached)
Every step's cube/arm position, gripper contact and (cem_policy) tracked expert-progress index are saved, so
waypoint-tracking stalls and drift can be checked afterwards without re-running.
Usage: python failure_phases.py --lewm lewm.pt --gcbc policy_chunk5.pt --episodes 20 --out results/failure_phases.json
"""
import os, sys, json, time, argparse
from collections import Counter
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
ap.add_argument('--episodes', type=int, default=20, help='per task')
ap.add_argument('--samples', type=int, default=300)
ap.add_argument('--n-policy', type=int, default=200)
ap.add_argument('--policy-std', type=float, default=0.3)
ap.add_argument('--iters', type=int, default=30)
ap.add_argument('--topk', type=int, default=30)
ap.add_argument('--lookahead', type=int, default=5)
ap.add_argument('--search', type=int, default=20)
ap.add_argument('--lift', type=float, default=0.01)
ap.add_argument('--near', type=float, default=0.04, help='xy distance to the target that counts as "over the target"')
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
FS = ck['frameskip']; A = len(amean); BD = FS * A
ZLO = ((-1 - amean) / astd).repeat(FS); ZHI = ((1 - amean) / astd).repeat(FS)
g = torch.load(a.gcbc, map_location=dev, weights_only=False)
assert g['chunk'] == FS
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
def plan(z0, zt, prior):  # eval_hierarchical.py with --horizon 1 --execute best
    mean = torch.zeros(1, BD, device=dev); std = torch.ones_like(mean)
    for _ in range(a.iters):
        cand = mean + std * torch.randn(a.samples, 1, BD, device=dev); cand[0] = mean
        cand[:a.n_policy, 0] = prior + a.policy_std * torch.randn(a.n_policy, BD, device=dev); cand[0, 0] = prior
        cand = torch.maximum(torch.minimum(cand, ZHI), ZLO)
        cost = (wm.predict(z0.expand(a.samples, 1, -1), wm.action_encoder(cand))[:, -1] - zt).pow(2).sum(-1)
        elite = cand[cost.topk(a.topk, largest=False).indices]
        mean, std = elite.mean(0), elite.std(0)
    return (cand[cost.argmin()][0].view(FS, A) * astd + amean).clamp(-1, 1).cpu().numpy()


def reset(task, seed):
    np.random.seed(seed)
    return env.reset(options=dict(task_id=task, render_goal=False), seed=seed)


def expert(ob, info):  # as in eval_hierarchical.py: stops at success/termination
    oracle = CubePlanOracle(env=env, noise=0.0)
    mid = U._cube_target_mocap_ids[0]
    oi = dict(info)
    oi['privileged/target_block'] = 0
    oi['privileged/target_block_pos'] = U._data.mocap_pos[mid].copy()
    oi['privileged/target_block_yaw'] = np.array([lie.SO3(wxyz=U._data.mocap_quat[mid]).compute_yaw_radians()])
    oracle.reset(ob, oi)
    frames, cubes, ees, cur = [ob], [cube(info)], [ee(info)], oi
    for _ in range(200):
        ob, _, term, trunc, cur = env.step(oracle.select_action(ob, cur))
        frames.append(ob); cubes.append(cube(cur)); ees.append(ee(cur))
        if oracle.done or term or trunc:
            break
    return frames, np.array(cubes), np.array(ees)


def furthest_phase(cubes, contacts, target, z_rest, success):
    if success:
        return 'success'
    if not any(contacts):
        return 'approach'
    up = cubes[:, 2] > z_rest + a.lift
    if not up.any():
        return 'grasp'
    run, best = 0, 0
    for u in up:
        run = run + 1 if u else 0
        best = max(best, run)
    if best < FS:
        return 'lift'
    if not (up & (np.linalg.norm(cubes[:, :2] - target[:2], axis=1) <= a.near)).any():
        return 'carry'
    return 'place'


def run_episode(task, seed, controller):
    ob, info = reset(task, seed)
    target = U._data.mocap_pos[U._cube_target_mocap_ids[0]].copy()
    z_rest = cube(info)[2]
    goal = info['goal']
    gl = torch.from_numpy(np.ascontiguousarray(goal)).to(dev)[None]
    E = None
    if controller == 'cem_policy':
        frames, ex_cube, ex_ee = expert(ob, info)
        E = encode(frames)
        ob, info = reset(task, seed)
    log = dict(cube=[cube(info).tolist()], ee=[ee(info).tolist()], contact=[contact(info)], progress=[])
    done, success, queue, k = False, False, [], 0
    while not done:
        if not queue:
            o = torch.from_numpy(np.ascontiguousarray(ob)).to(dev)[None]
            chunk = pol.act(o, gl)[0]
            if controller == 'policy':
                queue = list(chunk.clamp(-1, 1).cpu().numpy())
            else:
                z0 = encode([ob])
                hi = min(len(E), k + a.search + 1)
                k = k + int((E[k:hi] - z0).pow(2).sum(-1).argmin())
                log['progress'].append(k)
                queue = list(plan(z0, E[min(k + a.lookahead, len(E) - 1)], ((chunk - amean) / astd).reshape(BD)))
        ob, _, term, trunc, info = env.step(queue.pop(0))
        log['cube'].append(cube(info).tolist()); log['ee'].append(ee(info).tolist()); log['contact'].append(contact(info))
        success = success or bool(info.get('success', False))
        done = term or trunc
    ph = furthest_phase(np.array(log['cube']), log['contact'], target, z_rest, success)
    rec = dict(task=task, seed=seed, controller=controller, success=success, phase=ph, steps=len(log['cube']) - 1,
               target=target.tolist(), z_rest=z_rest, **log)
    if E is not None:
        rec.update(expert_cube=ex_cube.tolist(), expert_ee=ex_ee.tolist())
    return rec


eps, t0 = [], time.time()
for task in range(1, U.num_tasks + 1):
    for ep in range(a.episodes):
        seed = a.seed * 10000 + task * 1000 + ep
        for c in ('cem_policy', 'policy'):
            eps.append(run_episode(task, seed, c))
    s = {c: np.mean([e['success'] for e in eps if e['controller'] == c and e['task'] == task]) * 100 for c in ('cem_policy', 'policy')}
    print(f'task {task}: cem_policy {s["cem_policy"]:.0f}%  policy {s["policy"]:.0f}%  ({time.time()-t0:.0f}s)', flush=True)

PH = ['approach', 'grasp', 'lift', 'carry', 'place', 'success']
summary = {c: dict(Counter(e['phase'] for e in eps if e['controller'] == c)) for c in ('cem_policy', 'policy')}
print(f'{"ended in":10s} {"cem_policy":>11s} {"policy":>8s}')
for ph in PH:
    print(f'{ph:10s} {summary["cem_policy"].get(ph, 0):11d} {summary["policy"].get(ph, 0):8d}')
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(dict(summary=summary, args=vars(a), episodes=eps), open(a.out, 'w'))
print('FAILURE_PHASES_DONE', flush=True)
