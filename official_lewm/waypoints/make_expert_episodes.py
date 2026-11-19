"""Generate clean single pick-and-place OGBench-Cube episodes for waypoint evaluation.

Uses the same expert as the LeWM data (stable-worldmodel's OGBench ExpertPolicy with the plan oracle,
which reproduces the LeWM data timing: first success at steps 61-65, vs 62-65 in the LeWM file;
data_collection mode). Each episode starts at reset and ends at the expert's first completed
placement: the first step where the cube is on target (env success), the gripper has released and
the arm has lifted clear. Episodes that don't get there within MAX_STEPS are discarded.

Output HDF5 (concatenated rows + ep_offset / ep_len, like the LeWM files):
  pixels (N,224,224,3) uint8, qpos, qvel, block_pos, block_quat, target_pos, effector_pos,
  gripper_opening, action (action taken AFTER observing that row; zeros on the last row), success.
Usage: python gen_clean_episodes.py OUT.h5 [--episodes 120] [--num-envs 16] [--seed 1000]
"""
import os, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, h5py
import stable_worldmodel as swm
from stable_worldmodel.envs.ogbench import ExpertPolicy

MAX_STEPS = 200
RELEASE_OPEN = 0.1   # gripper_opening below this = released
CLEAR_Z = 0.06       # end-effector this far above the cube = lifted clear
LIFT_Z = 0.05        # cube must have been lifted above this height (a real pick) before the episode can end
KEYS = {  # output name -> info key
    'block_pos': 'privileged/block_0_pos', 'block_quat': 'privileged/block_0_quat',
    'target_pos': 'privileged/target_block_pos', 'effector_pos': 'proprio/effector_pos',
    'gripper_opening': 'proprio/gripper_opening',
}


def snapshot(world, i):
    I = world.infos
    env = world.envs.envs[i].unwrapped
    row = {k: np.asarray(I[v][i, 0]).copy() for k, v in KEYS.items()}
    row['pixels'] = np.asarray(I['pixels'][i, 0]).copy()
    row['qpos'] = env._data.qpos.copy()
    row['qvel'] = env._data.qvel.copy()
    row['success'] = bool(np.asarray(I['success'][i]).reshape(-1)[0])
    return row


def make_world(n):
    return swm.World('swm/OGBCube-v0', num_envs=n, image_shape=(224, 224), max_episode_steps=MAX_STEPS + 50,
                     env_type='single', ob_type='states', multiview=False, width=224, height=224,
                     visualize_info=False, terminate_at_goal=False, mode='data_collection')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--episodes', type=int, default=120)
    ap.add_argument('--num-envs', type=int, default=16); ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--oracle', default='plan_oracle', choices=['plan_oracle', 'markov_oracle'])
    a = ap.parse_args()

    pol = ExpertPolicy(policy_type=a.oracle, seed=a.seed)

    episodes, discarded, batch = [], 0, 0
    while len(episodes) < a.episodes:
        # Fresh World per batch: re-resetting one World applies random visual variation (floor colours) and can
        # produce garbage frames, which made every batch after the first unusable.
        world = make_world(a.num_envs)
        world.set_policy(pol)
        world.reset(seed=a.seed + batch * a.num_envs)
        batch += 1
        n = a.num_envs
        rows = [[snapshot(world, i)] for i in range(n)]
        acts = [[] for _ in range(n)]
        tgt0 = [rows[i][0]['target_pos'].copy() for i in range(n)]
        done = np.zeros(n, bool); ok = np.zeros(n, bool); lifted = np.zeros(n, bool)
        # skip tasks that start already solved (cube within 4 cm of the target)
        done |= np.array([np.linalg.norm(rows[i][0]['block_pos'] - tgt0[i]) <= 0.04 for i in range(n)])
        for t in range(MAX_STEPS):
            act = pol.get_action(world.infos)
            _, _, _, _, world.infos = world.envs.step(act)
            for i in range(n):
                if done[i]:
                    continue
                r = snapshot(world, i)
                acts[i].append(act[i].copy()); rows[i].append(r)
                # end of the first pick-and-place: cube on target, gripper released, arm lifted clear
                released = float(np.ravel(r['gripper_opening'])[0]) < RELEASE_OPEN
                clear = r['effector_pos'][2] >= r['block_pos'][2] + CLEAR_Z
                lifted[i] |= r['block_pos'][2] > LIFT_Z
                if lifted[i] and r['success'] and released and clear and np.linalg.norm(r['target_pos'] - tgt0[i]) < 1e-6:
                    done[i] = True; ok[i] = True
            if done.all():
                break
        for i in range(n):
            if ok[i] and len(episodes) < a.episodes:
                acts[i].append(np.zeros_like(acts[i][0]))
                ep = {k: np.stack([r[k] for r in rows[i]]) for k in rows[i][0]}
                ep['action'] = np.stack(acts[i]).astype(np.float32)
                ep['seed'] = a.seed + (batch - 1) * a.num_envs + i
                episodes.append(ep)
            elif not ok[i]:
                discarded += 1
        world.close() if hasattr(world, 'close') else None
        lens = [len(e['qpos']) for e in episodes]
        print(f'batch {batch}: kept {len(episodes)}, discarded {discarded}, len median {np.median(lens):.0f} '
              f'min {min(lens)} max {max(lens)}', flush=True)

    lens = np.array([len(e['qpos']) for e in episodes])
    with h5py.File(a.out, 'w') as f:
        f['ep_len'] = lens
        f['ep_offset'] = np.concatenate([[0], np.cumsum(lens)[:-1]])
        f['seed'] = np.array([e['seed'] for e in episodes])
        for k in episodes[0]:
            if k == 'seed':
                continue
            f.create_dataset(k, data=np.concatenate([e[k] for e in episodes]),
                             chunks=(1, 224, 224, 3) if k == 'pixels' else None,
                             compression='gzip' if k == 'pixels' else None)
    print('wrote', a.out, 'episodes', len(lens), 'steps', lens.sum())


if __name__ == '__main__':
    main()
