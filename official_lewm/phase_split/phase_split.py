"""Split official LeWM cube eval success by manipulation phase of each start->goal window.

For every (seed, num_eval) eval JSON given, recomputes the exact rows eval.py sampled (eval_rows below),
reads the low-dimensional columns over the 25-step window [start, start+goal_offset], and labels the window:
  free        : cube does not move (< MOVE m); arm-only reaching / retreat
  grasp_lift  : cube rests on the table at the start and is lifted at the goal
  carry       : cube lifted at both start and goal
  place       : cube lifted at the start and resting at the goal
  other       : cube moves but rests at both ends (e.g. a full pick-and-place inside the window, or a push)
Usage: python phase_split.py DATASET.h5 RESULT.json:SEED [RESULT.json:SEED ...]
"""
import sys, json, os
from collections import defaultdict
import numpy as np, h5py, hdf5plugin  # noqa: F401

GOAL_OFFSET, MOVE, LIFT = 25, 0.01, 0.01


def eval_rows(f, seed, num_eval, goal_offset):
    """Global row indices eval.py would pick (copy of its sampling logic)."""
    col = "episode_idx" if "episode_idx" in f else "ep_idx"
    ep_col = f[col][:]
    step_idx = f["step_idx"][:]
    ep_indices = np.unique(ep_col)
    # get_episodes_length: max step_idx + 1 per episode (computed vectorised, same result)
    max_step = np.full(ep_indices.max() + 1, -1, dtype=np.int64)
    np.maximum.at(max_step, ep_col, step_idx)
    episode_len = max_step[ep_indices] + 1
    max_start = episode_len - goal_offset - 1
    lut = np.zeros(ep_indices.max() + 1, dtype=np.int64); lut[ep_indices] = max_start
    valid_indices = np.nonzero(step_idx <= lut[ep_col])[0]
    g = np.random.default_rng(seed)
    pick = g.choice(len(valid_indices) - 1, size=num_eval, replace=False)
    return np.sort(valid_indices[pick])



f = h5py.File(sys.argv[1], 'r')
keys = list(f.keys())
print('columns:', keys)
bk = [k for k in keys if k.endswith('block_0_pos') and not k.startswith('goal')]
assert bk, 'no block_0_pos column'
block = f[bk[0]]
print('cube column:', bk[0])
z_rest = float(np.median(block[::97, 2]))  # cube sits on the table most of the time
print(f'cube rest height {z_rest:.4f}')

rows_out, stats = [], defaultdict(lambda: [0, 0])
for arg in sys.argv[2:]:
    path, seed = arg.split(':')
    res = json.load(open(path))
    succ = res['episode_successes']
    rows = eval_rows(f, int(seed), len(succ), GOAL_OFFSET)
    for r, s in zip(rows, succ):
        w = block[r:r + GOAL_OFFSET + 1]
        moved = float(np.linalg.norm(w - w[0], axis=1).max())
        up0, up1 = w[0, 2] > z_rest + LIFT, w[-1, 2] > z_rest + LIFT
        if moved < MOVE:
            ph = 'free'
        elif not up0 and up1:
            ph = 'grasp_lift'
        elif up0 and up1:
            ph = 'carry'
        elif up0 and not up1:
            ph = 'place'
        else:
            ph = 'other'
        stats[ph][0] += int(s); stats[ph][1] += 1
        rows_out.append(dict(seed=int(seed), row=int(r), success=int(s), phase=ph, cube_moved=moved))

print(f'\n{"phase":<12}{"n":>5}{"success":>10}')
for ph in ['free', 'grasp_lift', 'carry', 'place', 'other']:
    k, n = stats[ph]
    print(f'{ph:<12}{n:>5}{(100 * k / n if n else float("nan")):>9.0f}%')
tot_k = sum(v[0] for v in stats.values()); tot_n = sum(v[1] for v in stats.values())
con = [v for p, v in stats.items() if p != 'free']
print(f'{"contact":<12}{sum(v[1] for v in con):>5}{100 * sum(v[0] for v in con) / max(1, sum(v[1] for v in con)):>9.0f}%')
print(f'{"all":<12}{tot_n:>5}{100 * tot_k / tot_n:>9.0f}%')
json.dump(rows_out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'windows.json'), 'w'), indent=1)
print('PHASE_DONE')
