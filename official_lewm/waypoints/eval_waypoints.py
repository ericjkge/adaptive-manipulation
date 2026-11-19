"""Waypoint (subgoal) evaluation of LeWM + CEM on OGBench-Cube.

Each task is an episode with a sequence of goal waypoints. The planner is given one waypoint goal image at a time.
It moves to the next waypoint when the current one is reached or after `seg_budget` env steps, whichever is first;
on each switch the policy's queued plan and warm start are flushed, so it replans toward the new goal.

Sources (`--source`):
  clean : episodes from gen_clean_episodes.py (start at reset, end at the first completed placement).
          Waypoints every `spacing` steps plus the final step. Intermediate waypoints need the cube within 4 cm
          AND the end-effector within `ee_tol` of the expert's; the final waypoint needs only the cube within 4 cm
          of the task target (the env's own success test).
  lewm  : the original 25-step protocol (seed-42 starts from the LeWM dataset, one waypoint 25 steps ahead,
          cube-only success). Sanity check: should match the original eval (~72% at 10 x 300).

Writes one JSON per config with per-episode results, plus a per-plan log (.npz).
Example:
  python subgoal_eval.py --source clean --episodes-file clean.h5 --iters 10 --samples 300 --out results/x.json
"""
import os, sys, json, time, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path
import numpy as np, h5py, hdf5plugin, torch
import stable_pretraining as spt
import stable_worldmodel as swm
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

BIG = os.path.join(os.environ.get('STABLEWM_HOME', '/workspace/swm'), 'datasets/ogbench/cube_single_expert.h5')
CUBE_TOL = 0.04


def img_transform():
    return transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
                               transforms.Normalize(**spt.data.dataset_stats.ImageNet), transforms.Resize(size=224)])


# ---------------------------------------------------------------- tasks
def tasks_clean(path, n, spacing, min_gap):
    f = h5py.File(path, 'r')
    out = []
    for e in range(min(n, len(f['ep_len']))):
        s, L = int(f['ep_offset'][e]), int(f['ep_len'][e])
        idx = [i for i in range(spacing, L - 1, spacing) if (L - 1) - i >= min_gap] + [L - 1]
        wps = []
        for j, i in enumerate(idx):
            final = j == len(idx) - 1
            wps.append(dict(step=i, pixels=f['pixels'][s + i],
                            cube=f['target_pos'][s + L - 1] if final else f['block_pos'][s + i],
                            quat=f['block_quat'][s + i], ee=None if final else f['effector_pos'][s + i]))
        out.append(dict(qpos=f['qpos'][s], qvel=f['qvel'][s], pixels=f['pixels'][s], waypoints=wps, length=L))
    return out


def tasks_segment(path, n, start, offset=25, ee=False):
    """One 25-step task per clean episode, from step `start` to `start+offset` (capped at the last step).
    Cube-only success by default, like the LeWM protocol; `ee=True` also requires the end-effector."""
    f = h5py.File(path, 'r')
    out = []
    for e in range(min(n, len(f['ep_len']))):
        s, L = int(f['ep_offset'][e]), int(f['ep_len'][e])
        a_, g_ = s + start, s + min(start + offset, L - 1)
        out.append(dict(qpos=f['qpos'][a_], qvel=f['qvel'][a_], pixels=f['pixels'][a_], length=g_ - a_,
                        waypoints=[dict(step=g_ - s, pixels=f['pixels'][g_], cube=f['block_pos'][g_],
                                        quat=f['block_quat'][g_], ee=f['effector_pos'][g_] if ee else None)]))
    return out


def tasks_lewm(n, seed=42, offset=25, start_range=None):
    f = h5py.File(BIG, 'r')
    ep_len = f['ep_len'][:]; ep = f['ep_idx'][:]; step = f['step_idx'][:]
    g = np.random.default_rng(seed)
    if start_range is None:  # the original eval.py sampling
        valid = np.nonzero(step <= (ep_len - offset - 1)[ep])[0]
        pick = np.sort(valid[g.choice(len(valid) - 1, size=n, replace=False)])
    else:  # n random episodes, start step uniform in [lo, hi]
        lo, hi = start_range
        eps = g.choice(len(ep_len), size=n, replace=False)
        pick = np.sort(f['ep_offset'][:][eps] + g.integers(lo, hi + 1, size=n))
    out = []
    for r in pick:
        g_ = r + offset
        out.append(dict(qpos=f['qpos'][r], qvel=f['qvel'][r], pixels=f['pixels'][r], length=offset,
                        waypoints=[dict(step=offset, pixels=f['pixels'][g_], cube=f['privileged_block_0_pos'][g_],
                                        quat=f['privileged_block_0_quat'][g_], ee=None)]))
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['clean', 'lewm', 'segment'], required=True)
    ap.add_argument('--seg-start', type=int, default=0, help='segment source: start step within each clean episode')
    ap.add_argument('--seg-ee', action='store_true', help='segment source: also require the end-effector')
    ap.add_argument('--dump-frames', type=int, default=0, help='save camera frames of the first N episodes')
    ap.add_argument('--lewm-start-range', type=int, nargs=2, default=None,
                    help='lewm source: sample start steps uniformly in [lo, hi] instead of the eval.py sampling')
    ap.add_argument('--episodes-file', default=None)
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--iters', type=int, default=10)
    ap.add_argument('--samples', type=int, default=300)
    ap.add_argument('--replan', type=int, default=5, help='blocks executed before replanning')
    ap.add_argument('--spacing', type=int, default=25)
    ap.add_argument('--min-gap', type=int, default=10)
    ap.add_argument('--seg-budget', type=int, default=50)
    ap.add_argument('--ee-tol', type=float, default=0.05)
    ap.add_argument('--checkpoint', default='cube/lewm')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    if a.source == 'clean':
        tasks = tasks_clean(a.episodes_file, a.episodes, a.spacing, a.min_gap)
    elif a.source == 'segment':
        tasks = tasks_segment(a.episodes_file, a.episodes, a.seg_start, ee=a.seg_ee)
    else:
        tasks = tasks_lewm(a.episodes, start_range=tuple(a.lewm_start_range) if a.lewm_start_range else None)
    n = len(tasks)
    K = np.array([len(t['waypoints']) for t in tasks])
    total_budget = int(K.max() * a.seg_budget)
    print(f'{n} tasks, waypoints per task: {dict(zip(*np.unique(K, return_counts=True)))}', flush=True)

    world = swm.World('swm/OGBCube-v0', num_envs=n, image_shape=(224, 224), max_episode_steps=total_budget + 10,
                      env_type='single', ob_type='states', multiview=False, width=224, height=224,
                      visualize_info=False, terminate_at_goal=False)

    # action normaliser fitted on the LeWM training data's action column (as in eval.py)
    act = h5py.File(BIG, 'r')['action'][:]
    scaler = preprocessing.StandardScaler().fit(act[~np.isnan(act).any(axis=1)])

    model = swm.wm.utils.load_pretrained(a.checkpoint).to(a.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    solver = swm.solver.CEMSolver(model=model, batch_size=1, num_samples=a.samples, var_scale=1.0,
                                  n_steps=a.iters, topk=max(1, a.samples // 10), device=a.device, seed=a.seed)
    plan_log = dict(t=[], env=[], wp=[], cost=[], seconds=[])
    cur = dict(t=0)
    _orig = solver.solve

    def timed_solve(info_dict, *args, **kw):
        envs_in_call = len(next(iter(info_dict.values())))
        torch.cuda.synchronize()
        t0 = time.time()
        out = _orig(info_dict, *args, **kw)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / envs_in_call
        ids = cur['replan_ids']
        costs = [float(np.asarray(c).reshape(-1)[0]) for c in out.get('costs', [])] or [np.nan] * len(ids)
        for j, i in enumerate(ids):
            plan_log['t'].append(cur['t']); plan_log['env'].append(i); plan_log['wp'].append(int(wp_idx[i]))
            plan_log['cost'].append(costs[j] if j < len(costs) else np.nan); plan_log['seconds'].append(dt)
        return out
    solver.solve = timed_solve
    cfg = swm.PlanConfig(horizon=5, receding_horizon=a.replan, action_block=5)
    tf = img_transform()
    policy = swm.policy.WorldModelPolicy(solver=solver, config=cfg, process={'action': scaler},
                                         transform={'pixels': tf, 'goal': tf})
    world.set_policy(policy)

    # ---- reset and load task start states
    world.reset(seed=a.seed)
    wp_idx = np.zeros(n, int)
    for i, t in enumerate(tasks):
        env = world.envs.envs[i].unwrapped
        env.set_state(qpos=t['qpos'], qvel=t['qvel'])
        w = t['waypoints'][0]
        env.set_target_pos(cube_id=0, target_pos=w['cube'], target_quat=w['quat'])
    pix = world.infos['pixels']
    world.infos['pixels'] = np.stack([t['pixels'] for t in tasks])[:, None].astype(pix.dtype)
    goal = lambda: np.stack([tasks[i]['waypoints'][wp_idx[i]]['pixels'] for i in range(n)])[:, None]
    world.infos['goal'] = goal()

    finished = np.zeros(n, bool)
    reached = [[False] * k for k in K]
    min_cube = [[np.inf] * k for k in K]  # closest approach to each waypoint (m)
    min_ee = [[np.inf] * k for k in K]
    seg_steps = np.zeros(n, int)
    steps_used = np.zeros(n, int)
    flush = np.zeros(n, bool)
    frames = []
    t0 = time.time()
    for t in range(total_budget):
        cur['t'] = t
        world.infos['terminated'] = finished.copy()[:, None]  # env pool keeps (n_envs, 1) buffers
        world.infos['_needs_flush'] = flush.copy()
        buf = policy._action_buffer
        cur['replan_ids'] = [i for i in range(n) if not finished[i] and (flush[i] or buf is None or len(buf[i]) == 0)]
        actions = policy.get_action(world.infos)
        flush[:] = False
        _, _, _, _, world.infos = world.envs.step(actions, mask=~finished)
        cube = world.infos['privileged/block_0_pos'][:, 0]
        ee = world.infos['proprio/effector_pos'][:, 0]
        for i in np.flatnonzero(~finished):
            steps_used[i] += 1; seg_steps[i] += 1
            w = tasks[i]['waypoints'][wp_idx[i]]
            dc = np.linalg.norm(cube[i] - w['cube'])
            ok = dc <= CUBE_TOL
            min_cube[i][wp_idx[i]] = min(min_cube[i][wp_idx[i]], float(dc))
            if w['ee'] is not None:
                de = np.linalg.norm(ee[i] - w['ee'])
                min_ee[i][wp_idx[i]] = min(min_ee[i][wp_idx[i]], float(de))
                ok = ok and de <= a.ee_tol
            last = wp_idx[i] == K[i] - 1
            if ok:
                reached[i][wp_idx[i]] = True
            if ok or seg_steps[i] >= a.seg_budget:
                if last:
                    finished[i] = True
                else:
                    wp_idx[i] += 1; seg_steps[i] = 0; flush[i] = True
                    nw = tasks[i]['waypoints'][wp_idx[i]]
                    world.envs.envs[i].unwrapped.set_target_pos(cube_id=0, target_pos=nw['cube'], target_quat=nw['quat'])
        world.infos['goal'] = goal()
        if a.dump_frames:
            frames.append(np.asarray(world.infos['pixels'][:a.dump_frames, 0]).copy())
        if finished.all():
            break
        if t % 25 == 0:
            print(f't={t} finished {finished.sum()}/{n} elapsed {time.time()-t0:.0f}s', flush=True)

    # ---- results
    plans = np.array(plan_log['env'])
    secs = np.array(plan_log['seconds'])
    warm = secs[1:] if len(secs) > 1 else secs  # first call includes GPU warmup
    final = np.array([r[-1] for r in reached])
    rec = dict(source=a.source, iterations=a.iters, samples=a.samples, replan_after_blocks=a.replan,
               seg_budget=a.seg_budget, spacing=a.spacing, ee_tol=a.ee_tol, n_episodes=n,
               success_rate=float(final.mean() * 100),
               waypoint_rate=float(np.mean([np.mean(r) for r in reached]) * 100),
               seconds_per_plan=float(warm.mean()) if len(warm) else None,
               plans_per_episode=float(len(plans) / n),
               planning_seconds_per_episode=float(warm.mean() * len(plans) / n) if len(warm) else None,
               steps_per_episode=float(steps_used.mean()),
               wall_seconds=time.time() - t0,
               episodes=[dict(success=bool(final[i]), reached=reached[i], steps=int(steps_used[i]),
                              min_cube_dist=[None if np.isinf(x) else x for x in min_cube[i]],
                              min_ee_dist=[None if np.isinf(x) else x for x in min_ee[i]],
                              plans=int((plans == i).sum()), length=int(tasks[i]['length'])) for i in range(n)])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    np.savez(Path(a.out).with_suffix('.plans.npz'), **{k: np.array(v) for k, v in plan_log.items()})
    if a.dump_frames:
        np.savez_compressed(Path(a.out).with_suffix('.frames.npz'), frames=np.stack(frames),
                            goals=np.stack([[w['pixels'] for w in tasks[i]['waypoints']][:3] for i in range(a.dump_frames)]))
    print(f"success {rec['success_rate']:.1f}%  waypoints {rec['waypoint_rate']:.1f}%  "
          f"plan s/ep {rec['planning_seconds_per_episode']}  steps/ep {rec['steps_per_episode']:.0f}", flush=True)


if __name__ == '__main__':
    main()
