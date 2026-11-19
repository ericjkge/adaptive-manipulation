"""Latent waypoint generator: conditional flow matching in frozen LeWM latent space (PoC, single-cube play data).

Step 1 (cached): encode every frame of the train/val datasets with the frozen LeWM encoder -> 192-d latents.
Step 2: train a small AdaLN transformer that generates a short latent path of `--waypoints` points, FS (=5) env steps
apart (default: t+5 and t+10), conditioned on the last `--history` latents (t, t-5, t-10; clamped at the trajectory
start) and a goal latent sampled uniformly from the future of the same trajectory.
Flow matching (rectified flow): x_t = (1-t) x0 + t x1, x0 ~ N(0, I), target velocity x1 - x0; sampling integrates
16 Euler steps from noise. Latents are z-scored per dimension with dataset statistics (stored in the checkpoint).
"""
import os, sys, json, time, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.environ.get('LEWM_REPO', '/workspace/le-wm'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def time_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, device=t.device) / max(half - 1, 1))
    ang = t[:, None] * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], -1)


class AdaLNBlock(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.mod = nn.Linear(d, 4 * d)
        nn.init.zeros_(self.mod.weight); nn.init.zeros_(self.mod.bias)
        self.n1, self.n2 = nn.LayerNorm(d, elementwise_affine=False), nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, c):
        s1, b1, s2, b2 = self.mod(F.silu(c)).chunk(4, -1)
        h = self.n1(x) * (1 + s1[:, None]) + b1[:, None]
        x = x + self.attn(h, h, h, need_weights=False)[0]
        h = self.n2(x) * (1 + s2[:, None]) + b2[:, None]
        return x + self.mlp(h)


class WaypointFlow(nn.Module):
    """Velocity field over a (B, W, D) latent path, conditioned on (B, H, D) history and (B, D) goal latents."""
    def __init__(self, D=192, waypoints=2, history=3, d=512, depth=4, heads=8, ff=2048, tdim=64):
        super().__init__()
        self.cfg = dict(D=D, waypoints=waypoints, history=history, d=d, depth=depth, heads=heads, ff=ff, tdim=tdim)
        self.inp = nn.Linear(D, d)
        self.pos = nn.Parameter(torch.randn(waypoints, d) * 0.02)
        self.hist = nn.Linear(history * D, d)
        self.goal = nn.Linear(D, d)
        self.time = nn.Sequential(nn.Linear(tdim, d), nn.SiLU(), nn.Linear(d, d))
        self.fuse = nn.Sequential(nn.Linear(3 * d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([AdaLNBlock(d, heads, ff) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, D))
        self.tdim = tdim
        self.register_buffer('zmean', torch.zeros(D))
        self.register_buffer('zstd', torch.ones(D))

    def forward(self, xt, hist, goal, t):  # all normalised latents
        c = self.fuse(torch.cat([self.hist(hist.flatten(1)), self.goal(goal), self.time(time_embedding(t, self.tdim))], -1))
        x = self.inp(xt) + self.pos[None]
        for b in self.blocks:
            x = b(x, c)
        return self.out(x)

    @torch.no_grad()
    def sample(self, hist, goal, steps=16):
        """hist (B, H, D), goal (B, D) raw latents -> (B, W, D) raw latent path."""
        n = lambda z: (z - self.zmean) / self.zstd
        h, g = n(hist), n(goal)
        x = torch.randn(len(g), self.cfg['waypoints'], self.cfg['D'], device=g.device)
        for i in range(steps):
            t = torch.full((len(g),), i / steps, device=g.device)
            x = x + self(x, h, g, t) / steps
        return x * self.zstd + self.zmean


def main():
    from common import load_split, to_model_pixels
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='visual-cube-single-play-v0')
    ap.add_argument('--lewm', required=True)
    ap.add_argument('--out', default='/workspace/runs/flow_cube_single')
    ap.add_argument('--waypoints', type=int, default=2)
    ap.add_argument('--history', type=int, default=3)
    ap.add_argument('--steps', type=int, default=100_000)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--final-lr', type=float, default=3e-5)
    ap.add_argument('--warmup', type=int, default=2000)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--ema', type=float, default=0.9999)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    dev = 'cuda'
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, 'log.txt'), 'a')
    def P(*s):
        m = ' '.join(str(x) for x in s); print(m, flush=True); log.write(m + '\n'); log.flush()

    ck = torch.load(a.lewm, map_location=dev, weights_only=False)
    wm = ck['model'].to(dev).eval()
    FS = ck['frameskip']

    # ---------------------------------------------------------------- latent cache
    def latents(split):
        path = os.path.join(a.out, f'latents_{split}.pt')
        if os.path.exists(path):
            return torch.load(path, map_location=dev)
        d = load_split(a.dataset, split, dev)
        z = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for i in range(0, len(d['obs']), 2048):
                z.append(wm.encode({'pixels': to_model_pixels(d['obs'][i:i + 2048][:, None])})['emb'][:, 0].float())
        idx = torch.arange(len(d['obs']), device=dev)
        is_start = torch.ones_like(idx, dtype=torch.bool)
        is_start[1:] = d['ep_end'][:-1] == idx[:-1]
        ep_start = torch.cummax(torch.where(is_start, idx, torch.zeros_like(idx)), 0).values
        out = dict(z=torch.cat(z), ep_end=d['ep_end'], ep_start=ep_start)
        del d; torch.cuda.empty_cache()
        torch.save(out, path)
        return out
    t0 = time.time()
    tr, va = latents('train'), latents('val')
    P(f'latents: train {tuple(tr["z"].shape)} val {tuple(va["z"].shape)} ({time.time()-t0:.0f}s)')
    zmean, zstd = tr['z'].mean(0), tr['z'].std(0)

    W, H = a.waypoints, a.history
    def starts(d):
        idx = torch.arange(len(d['z']), device=dev)
        return idx[idx + FS * W <= d['ep_end']]
    tr_s, va_s = starts(tr), starts(va)

    def batch(d, s_all, B, gen=None):
        t = s_all[torch.randint(len(s_all), (B,), device=dev, generator=gen)]
        hidx = (t[:, None] - FS * torch.arange(H - 1, -1, -1, device=dev)[None]).maximum(d['ep_start'][t][:, None])
        widx = t[:, None] + FS * torch.arange(1, W + 1, device=dev)[None]
        end = d['ep_end'][t]
        u = torch.rand(B, device=dev, generator=gen)
        g = t + 1 + (u * (end - t).float()).long().clamp(max=(end - t - 1))
        n = lambda z: (z - zmean) / zstd
        return n(d['z'][hidx]), n(d['z'][g]), n(d['z'][widx])

    model = WaypointFlow(D=tr['z'].shape[1], waypoints=W, history=H).to(dev)
    model.zmean.copy_(zmean); model.zstd.copy_(zstd)
    ema = WaypointFlow(**model.cfg).to(dev); ema.load_state_dict(model.state_dict()); ema.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    P(f'flow params {sum(p.numel() for p in model.parameters())/1e6:.1f}M, train starts {len(tr_s)}')

    @torch.no_grad()
    def evaluate():
        """Val: flow-matching loss, and latent error of sampled waypoints vs real (and vs copying the current latent)."""
        g = torch.Generator(device=dev).manual_seed(1)
        h, gl, x1 = batch(va, va_s, 4096, g)
        x0 = torch.randn(x1.shape, device=dev, generator=g)
        t = torch.rand(len(x1), device=dev, generator=g)
        fm = F.mse_loss(ema(x0 + t[:, None, None] * (x1 - x0), h, gl, t), x1 - x0).item()
        raw = lambda z: z * zstd + zmean
        samp = ema.sample(raw(h), raw(gl))
        err = (samp - raw(x1)).pow(2).sum(-1).median(0).values.tolist()
        copy = (raw(h[:, -1:]) - raw(x1)).pow(2).sum(-1).median(0).values.tolist()
        return fm, err, copy

    t0 = time.time()
    for step in range(1, a.steps + 1):
        lr = a.lr * step / a.warmup if step <= a.warmup else \
            a.final_lr + 0.5 * (a.lr - a.final_lr) * (1 + math.cos(math.pi * (step - a.warmup) / (a.steps - a.warmup)))
        for gp in opt.param_groups: gp['lr'] = lr
        h, gl, x1 = batch(tr, tr_s, a.batch)
        x0 = torch.randn_like(x1)
        t = torch.rand(len(x1), device=dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            v = model(x0 + t[:, None, None] * (x1 - x0), h, gl, t)
        loss = F.mse_loss(v.float(), x1 - x0)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - a.ema)
        if step % 1000 == 0:
            P(f'step {step} loss {loss.item():.4f} lr {lr:.2e} {(time.time()-t0)/60:.1f} min')
        if step % 25_000 == 0 or step == a.steps:
            fm, err, copy = evaluate()
            P(f'== step {step} val fm {fm:.4f}  sampled-waypoint err {[round(e, 2) for e in err]}  '
              f'copy-current err {[round(e, 2) for e in copy]}')
            torch.save({'state_dict': ema.state_dict(), 'cfg': ema.cfg, 'frameskip': FS, 'step': step, 'args': vars(a)},
                       os.path.join(a.out, f'flow_{step}.pt'))
    P('TRAIN_DONE')


if __name__ == '__main__':
    main()
