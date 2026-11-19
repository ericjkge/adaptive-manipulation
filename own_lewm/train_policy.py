"""Goal-conditioned behaviour cloning on OGBench visual datasets, predicting action chunks (default 5 actions = one LeWM block).

Mirrors OGBench's GCBC baseline (impls/agents/gcbc.py with --agent.encoder=impala_small --agent.p_aug=0.5):
early fusion (current frame and goal frame stacked channel-wise) -> IMPALA-small CNN -> MLP(512,512,512); goals are
uniformly sampled future states of the same trajectory (actor_p_trajgoal=1, no geometric sampling); random-shift
augmentation (pad 3) with probability 0.5 per batch; Adam lr 3e-4, batch 256, 500k steps.
Differences: PyTorch; the head outputs a chunk of `--chunk` z-scored actions (MSE, i.e. a constant-std Gaussian) instead of one.

"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import load_split, action_stats, random_shift

CHUNK = 5


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(F.relu(x))))


class ImpalaSmall(nn.Module):
    """IMPALA encoder with stack sizes (16, 32, 32), one residual block per stack, MLP(512) head."""
    def __init__(self, in_ch, out=512):
        super().__init__()
        layers, c = [], in_ch
        for n in (16, 32, 32):
            layers += [nn.Conv2d(c, n, 3, padding=1), nn.MaxPool2d(3, 2, padding=1), ResBlock(n)]
            c = n
        self.convs = nn.Sequential(*layers)
        self.fc = nn.Linear(32 * 8 * 8, out)

    def forward(self, x):
        return F.gelu(self.fc(F.relu(self.convs(x)).flatten(1)))


class GCBCPolicy(nn.Module):
    def __init__(self, action_dim, chunk=CHUNK):
        super().__init__()
        self.enc = ImpalaSmall(6)
        self.mlp = nn.Sequential(nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(),
                                 nn.Linear(512, 512), nn.GELU(), nn.Linear(512, chunk * action_dim))
        self.action_dim, self.chunk = action_dim, chunk
        self.register_buffer('amean', torch.zeros(action_dim))
        self.register_buffer('astd', torch.ones(action_dim))

    def forward(self, obs_u8, goal_u8):
        """obs/goal: uint8 (B, 64, 64, 3). Returns z-scored action chunk (B, chunk, action_dim)."""
        x = torch.cat([obs_u8, goal_u8], -1).permute(0, 3, 1, 2).float() / 255.0
        return self.mlp(self.enc(x)).view(-1, self.chunk, self.action_dim)

    @torch.no_grad()
    def act(self, obs_u8, goal_u8):
        """Raw-scale action chunk, clipped to [-1, 1] (the env's action bounds)."""
        return (self(obs_u8, goal_u8) * self.astd + self.amean).clamp(-1, 1)


def sample(d, starts, B, dev, chunk=CHUNK):
    t = starts[torch.randint(len(starts), (B,), device=dev)]
    end = d['ep_end'][t]
    # goal uniformly in (t, end] of the same trajectory
    g = t + 1 + (torch.rand(B, device=dev) * (end - t).float()).long().clamp(max=(end - t - 1))
    acts = d['act'][t[:, None] + torch.arange(chunk, device=dev)[None]]
    return d['obs'][t], d['obs'][g], acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='visual-cube-single-play-v0')
    ap.add_argument('--out', default='/workspace/runs/gcbc_cube_single')
    ap.add_argument('--steps', type=int, default=500_000)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--p-aug', type=float, default=0.5)
    ap.add_argument('--ckpt-every', type=int, default=100_000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--chunk', type=int, default=CHUNK)
    a = ap.parse_args()
    C = a.chunk
    dev = 'cuda'
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, 'log.txt'), 'a')
    def P(*s):
        m = ' '.join(str(x) for x in s); print(m, flush=True); log.write(m + '\n'); log.flush()

    tr, va = load_split(a.dataset, 'train', dev), load_split(a.dataset, 'val', dev)
    A = tr['act'].shape[1]
    amean, astd = action_stats(tr['act'], tr['ep_end'])
    idx = lambda d: torch.arange(len(d['obs']), device=dev)
    # starts whose C actions are all valid (the last row of a trajectory has no valid action)
    tr_starts = idx(tr)[idx(tr) + C - 1 < tr['ep_end']]
    va_starts = idx(va)[idx(va) + C - 1 < va['ep_end']]

    pol = GCBCPolicy(A, C).to(dev)
    pol.amean.copy_(amean); pol.astd.copy_(astd)
    opt = torch.optim.Adam(pol.parameters(), lr=a.lr)
    P(f'GCBC params {sum(p.numel() for p in pol.parameters())/1e6:.2f}M, train starts {len(tr_starts)}')

    @torch.no_grad()
    def val():
        pol.eval(); torch.manual_seed(1); ls = []
        for _ in range(20):
            o, g, ac = sample(va, va_starts, 512, dev, C)
            ls.append(F.mse_loss(pol(o, g), (ac - amean) / astd).item())
        pol.train(); return float(np.mean(ls))

    t0 = time.time()
    for step in range(1, a.steps + 1):
        o, g, ac = sample(tr, tr_starts, a.batch, dev, C)
        if torch.rand(()) < a.p_aug:  # same shift for obs and goal of a sample, as in OGBench
            both = random_shift(torch.cat([o, g], -1))
            o, g = both[..., :3], both[..., 3:]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = F.mse_loss(pol(o, g).float(), (ac - amean) / astd)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 2000 == 0:
            P(f'step {step} loss {loss.item():.4f} {(time.time()-t0)/60:.1f} min')
        if step % a.ckpt_every == 0 or step == a.steps:
            torch.save({'state_dict': pol.state_dict(), 'action_dim': A, 'chunk': C, 'step': step, 'args': vars(a)},
                       os.path.join(a.out, f'gcbc_{step}.pt'))
            P(f'== step {step} val mse {val():.4f}')
    P('TRAIN_DONE')


if __name__ == '__main__':
    main()
