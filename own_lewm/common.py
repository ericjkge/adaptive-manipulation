"""Shared data utilities for OGBench visual-cube experiments (LeWM pretraining, GCBC policy).

The OGBench .npz files store, per trajectory, observations s0..sT and actions a0..aT, where the last action of each
trajectory has no next state (terminals[t] == 1 marks the last state). We keep the raw layout and precompute, for every
row, the index of its trajectory's last row (`ep_end`), so windows can be checked to stay inside one trajectory.
Everything is loaded onto the GPU once (64x64 uint8 frames: ~12 GB for 1M frames), so there is no dataloader.
"""
import os
import numpy as np
import torch

DATA_DIR = os.environ.get('OGB_DATA', '/workspace/ogbench_data')
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_split(name, split, device):
    """name e.g. 'visual-cube-single-play-v0'; split 'train' or 'val'."""
    f = os.path.join(DATA_DIR, f'{name}.npz' if split == 'train' else f'{name}-val.npz')
    z = np.load(f)
    obs = z['observations']  # (N, 64, 64, 3) uint8
    act = z['actions'].astype(np.float32)  # (N, A)
    term = z['terminals'].astype(bool)  # (N,) True at the last state of each trajectory
    assert obs.dtype == np.uint8 and obs.shape[1:] == (64, 64, 3), obs.shape
    ends = np.flatnonzero(term)
    starts = np.concatenate([[0], ends[:-1] + 1])
    ep_end = np.repeat(ends, ends - starts + 1)
    assert len(ep_end) == len(obs)
    return dict(
        obs=torch.from_numpy(obs).to(device),
        act=torch.from_numpy(act).to(device),
        ep_end=torch.from_numpy(ep_end).long().to(device),
        n_traj=len(ends), traj_len=float(np.mean(ends - starts + 1)),
    )


def action_stats(act, ep_end):
    """Mean/std over valid actions (the last row of each trajectory has no valid action)."""
    idx = torch.arange(len(act), device=act.device)
    valid = act[idx != ep_end]
    return valid.mean(0), valid.std(0)


def to_model_pixels(x):
    """uint8 (..., 64, 64, 3) -> float (..., 3, 64, 64), ImageNet-normalised (as in LeWM's preprocessor)."""
    sh = x.shape[:-3]
    x = x.reshape(-1, *x.shape[-3:]).permute(0, 3, 1, 2).float() / 255.0
    x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
    return x.reshape(*sh, *x.shape[1:])


def random_shift(x, pad=3):
    """DrQ-style random shift on uint8 (B, H, W, C) images with edge padding (OGBench's augmentation, padding 3)."""
    B, H, W, C = x.shape
    xp = torch.nn.functional.pad(x.permute(0, 3, 1, 2).float(), (pad, pad, pad, pad), mode='replicate')
    dx = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    dy = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    rows = (dy[:, None] + torch.arange(H, device=x.device)[None])  # (B, H)
    cols = (dx[:, None] + torch.arange(W, device=x.device)[None])  # (B, W)
    out = xp[torch.arange(B, device=x.device)[:, None, None, None], torch.arange(C, device=x.device)[None, :, None, None],
             rows[:, None, :, None], cols[:, None, None, :]]
    return out.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)
