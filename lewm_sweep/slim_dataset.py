"""Build a slim copy of a LeWM eval dataset: all low-dimensional columns in full, but camera
images ONLY for the frames the eval reads (start..start+goal_offset of each sampled episode).

The eval takes its first observation and goal image from the dataset's stored images, so those
frames must be kept. Images are stored as a full-size chunked array (1 frame per chunk); frames
that were never written take no disk space, so row indexing is identical to the original.

Episode sampling replicates eval.py exactly (same valid-start rule, same RNG calls).

Usage:
  python slim_dataset.py build  SRC.h5 DST.h5 [--goal-offset 25]
  python slim_dataset.py verify SRC.h5 DST.h5 [--goal-offset 25]
"""
import argparse, sys
from pathlib import Path
import numpy as np, h5py, hdf5plugin

# (seed, num_eval) pairs whose frames are kept. Sweep uses (42, 50); timing (42, 10); smoke (42, 4).
POOL = [(42, n) for n in (4, 10, 50, 100, 200, 500)] + [(s, n) for s in range(5) for n in (50, 100, 200)]


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


def needed_frames(f, goal_offset):
    rows = set()
    for seed, n in POOL:
        for r in eval_rows(f, seed, n, goal_offset):
            rows.update(range(int(r), int(r) + goal_offset + 1))
    return np.array(sorted(rows))


def build(src, dst, goal_offset):
    with h5py.File(src, "r") as f, h5py.File(dst, "w") as g:
        for k in f.keys():
            if k == "pixels":
                continue
            g.create_dataset(k, data=f[k][:], dtype=f[k].dtype, chunks=f[k].chunks)  # dtype keeps text columns
        px = f["pixels"]
        frames = needed_frames(f, goal_offset)
        print(f"keeping {len(frames)} of {px.shape[0]} frames ({100*len(frames)/px.shape[0]:.2f}%)")
        out = g.create_dataset("pixels", shape=px.shape, dtype=px.dtype,
                               chunks=(1,) + px.shape[1:], fillvalue=0,
                               **hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE))
        # copy contiguous runs to keep reads efficient
        runs = np.split(frames, np.nonzero(np.diff(frames) != 1)[0] + 1)
        for i, run in enumerate(runs):
            out[run[0]:run[-1] + 1] = px[run[0]:run[-1] + 1]
            if i % 500 == 0:
                print(f"  {i}/{len(runs)} runs", flush=True)
    print("wrote", dst, f"{Path(dst).stat().st_size/1e9:.2f} GB")


def verify(src, dst, goal_offset):
    """Every column must match exactly; pixels must match on every kept frame."""
    with h5py.File(src, "r") as f, h5py.File(dst, "r") as g:
        assert set(f.keys()) == set(g.keys()), (set(f.keys()), set(g.keys()))
        for k in f.keys():
            if k != "pixels":
                a, b = f[k][:], g[k][:]
                same = (a.tobytes() == b.tobytes()) if a.dtype != object else np.array_equal(a, b)  # bytes: NaN-safe
                assert same and a.dtype == b.dtype and a.shape == b.shape, k
        frames = needed_frames(f, goal_offset)
        for seed, n in POOL:  # recompute selection from the SLIM file: must equal the original's
            assert np.array_equal(eval_rows(f, seed, n, goal_offset), eval_rows(g, seed, n, goal_offset)), (seed, n)
        runs = np.split(frames, np.nonzero(np.diff(frames) != 1)[0] + 1)
        for run in runs:
            a, b = run[0], run[-1] + 1
            assert np.array_equal(f["pixels"][a:b], g["pixels"][a:b]), (a, b)
    print(f"VERIFIED: all columns identical; {len(frames)} kept frames identical; "
          f"episode selection identical for {len(POOL)} (seed, num_eval) settings")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "verify"]); ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--goal-offset", type=int, default=25)
    a = ap.parse_args()
    (build if a.cmd == "build" else verify)(a.src, a.dst, a.goal_offset)
