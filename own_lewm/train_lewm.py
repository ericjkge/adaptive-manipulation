"""Pretrain LeWM on an OGBench visual dataset (default: visual-cube-single-play-v0, 64x64).

Model and loss follow the official LeWM recipe (le-wm/config/train/lewm.yaml): ViT-tiny encoder trained from scratch,
CLS token -> MLP projector (BatchNorm) -> 192-d latent; 6-layer AR predictor conditioned on action embeddings;
loss = MSE(next latent) + 0.09 * SIGReg; AdamW lr 5e-5, wd 1e-3, batch 128, bf16, grad clip 1.0; history 3, 1 prediction,
frameskip 5 (5 env actions concatenated per step). Deviations: 64x64 input with patch size 8 (OGBench native resolution);
OGBench's own train/val files instead of a random 90/10 split; a plain PyTorch loop with all data on the GPU.

Each training sample is 4 frames at t, t+5, t+10, t+15 inside one trajectory, and the 25-d action blocks starting at
those frames (z-scored per action dimension; the last block is not used by the loss).
Checkpoints (every `--ckpt-every` epochs and at the end) store the whole model plus action stats; each one also logs a
rollout diagnostic: latent error after k=1..5 predicted blocks with the dataset's actions vs random actions.
"""
import os, sys, json, time, math, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.environ.get('LEWM_REPO', '/workspace/le-wm'))
from jepa import JEPA  # noqa: E402
from module import ARPredictor, Embedder, MLP, SIGReg  # noqa: E402
from stable_pretraining.backbone.utils import vit_hf  # noqa: E402
from common import load_split, action_stats, to_model_pixels  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', default='visual-cube-single-play-v0')
ap.add_argument('--out', default='/workspace/runs/lewm_cube_single')
ap.add_argument('--epochs', type=int, default=100)
ap.add_argument('--max-hours', type=float, default=11.0, help='cap: shrink the epoch count to fit this wall-clock budget')
ap.add_argument('--batch', type=int, default=128)
ap.add_argument('--lr', type=float, default=5e-5)
ap.add_argument('--wd', type=float, default=1e-3)
ap.add_argument('--warmup-epochs', type=float, default=1.0)
ap.add_argument('--patch', type=int, default=8)
ap.add_argument('--ckpt-every', type=int, default=10)
ap.add_argument('--seed', type=int, default=3072)
ap.add_argument('--smoke-steps', type=int, default=0, help='debug: override steps per epoch')
a = ap.parse_args()

FS, HIST, NPRED, D = 5, 3, 1, 192
T = HIST + NPRED  # frames per sample
dev = 'cuda'
torch.manual_seed(a.seed); np.random.seed(a.seed)
os.makedirs(a.out, exist_ok=True)
log = open(os.path.join(a.out, 'log.txt'), 'a')
def P(*s):
    msg = ' '.join(str(x) for x in s); print(msg, flush=True); log.write(msg + '\n'); log.flush()

# ---------------------------------------------------------------- data
tr = load_split(a.dataset, 'train', dev)
va = load_split(a.dataset, 'val', dev)
A = tr['act'].shape[1]
amean, astd = action_stats(tr['act'], tr['ep_end'])
P(f'train frames {len(tr["obs"])} trajs {tr["n_traj"]} (len {tr["traj_len"]:.0f}); val frames {len(va["obs"])}; action dim {A}')

def valid_starts(d, span_frames, span_actions):
    """Rows t such that frames t..t+span_frames and actions t..t+span_actions-1 stay inside one trajectory."""
    idx = torch.arange(len(d['obs']), device=dev)
    return idx[(idx + span_frames <= d['ep_end']) & (idx + span_actions - 1 < d['ep_end'])]

# training windows: frames t + 5i (i<4); the loss uses action blocks 0..2 (actions t..t+14)
tr_starts = valid_starts(tr, FS * (T - 1), FS * HIST)
va_starts = valid_starts(va, FS * (T - 1), FS * HIST)
# diagnostic windows: 6 frames (start + 5 predicted blocks) with all 25 actions valid
va_diag = valid_starts(va, FS * 5, FS * 5)
g = torch.Generator(device='cpu').manual_seed(0)
va_diag = va_diag[torch.randperm(len(va_diag), generator=g)[:512].to(dev)]
P(f'train windows {len(tr_starts)}, val windows {len(va_starts)}')

def act_blocks(d, starts, nblocks):
    """(B, nblocks, FS*A) z-scored action blocks starting at each start row."""
    off = torch.arange(nblocks * FS, device=dev)
    a_ = d['act'][starts[:, None] + off[None]]  # (B, nblocks*FS, A)
    a_ = (a_ - amean) / astd
    return a_.reshape(len(starts), nblocks, FS * A)

def batch(d, starts):
    fr = starts[:, None] + FS * torch.arange(T, device=dev)[None]
    # only the first HIST action blocks are used by the loss (and only those are guaranteed in-bounds)
    return {'pixels': to_model_pixels(d['obs'][fr]), 'action': act_blocks(d, starts, HIST)}

# ---------------------------------------------------------------- model (official LeWM config, 64 px / patch 8)
encoder = vit_hf(size='tiny', patch_size=a.patch, image_size=64, pretrained=False, use_mask_token=False)
model = JEPA(
    encoder=encoder,
    predictor=ARPredictor(num_frames=HIST, input_dim=D, hidden_dim=D, output_dim=D, depth=6, heads=16, mlp_dim=2048,
                          dim_head=64, dropout=0.1, emb_dropout=0.0),
    action_encoder=Embedder(input_dim=FS * A, emb_dim=D),
    projector=MLP(input_dim=D, output_dim=D, hidden_dim=2048, norm_fn=nn.BatchNorm1d),
    pred_proj=MLP(input_dim=D, output_dim=D, hidden_dim=2048, norm_fn=nn.BatchNorm1d),
).to(dev)
sigreg = SIGReg(knots=17, num_proj=1024).to(dev)
P(f'params {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

def loss_fn(b):
    out = model.encode(b)
    emb, act_emb = out['emb'], out['act_emb']
    pred = model.predict(emb[:, :HIST], act_emb[:, :HIST])
    pred_loss = (pred - emb[:, NPRED:]).pow(2).mean()
    sig = sigreg(emb.transpose(0, 1))
    return pred_loss + 0.09 * sig, pred_loss, sig

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)

# ---------------------------------------------------------------- diagnostic
@torch.no_grad()
def rollout_diag():
    """Latent error ||z_pred(k) - z(real frame t+5k)||^2 for k=1..5 with dataset actions vs random (z-scored) actions."""
    model.eval()
    res = {}
    for i in range(0, len(va_diag), 128):
        s = va_diag[i:i + 128]
        fr = s[:, None] + FS * torch.arange(6, device=dev)[None]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = model.encode({'pixels': to_model_pixels(va['obs'][fr])})['emb'].float()  # (B, 6, D)
            for name, acts in [('data', act_blocks(va, s, 5)), ('random', torch.randn(len(s), 5, FS * A, device=dev))]:
                emb = z[:, :1]
                ae = model.action_encoder(acts)
                for k in range(5):
                    lo = max(0, emb.shape[1] - HIST)
                    nxt = model.predict(emb[:, lo:], ae[:, lo:k + 1])[:, -1:]
                    emb = torch.cat([emb, nxt.float()], 1)
                err = (emb[:, 1:] - z[:, 1:]).pow(2).sum(-1)  # (B, 5)
                res.setdefault(name, []).append(err)
    model.train()
    out = {n: torch.cat(v).median(0).values.tolist() for n, v in res.items()}
    out['copy_start'] = (z[:, 1:] - z[:, :1]).pow(2).sum(-1).median(0).values.tolist()
    return out

@torch.no_grad()
def val_loss():
    model.eval(); tot = []
    for i in range(0, min(len(va_starts), 128 * 40), 128):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            tot.append(loss_fn(batch(va, va_starts[i:i + 128]))[1].float().item())
    model.train(); return float(np.mean(tot))

def save(tag, epoch):
    path = os.path.join(a.out, f'lewm_{tag}.pt')
    torch.save({'model': model, 'action_mean': amean.cpu(), 'action_std': astd.cpu(), 'epoch': epoch,
                'args': vars(a), 'frameskip': FS, 'history': HIST, 'img_size': 64}, path)
    return path

# ---------------------------------------------------------------- train
steps_per_epoch = a.smoke_steps or len(tr_starts) // a.batch
model.train()
# measure speed to fit the wall-clock budget
t0 = time.time(); step = 0
def train_step(lr):
    for gp in opt.param_groups: gp['lr'] = lr
    s = tr_starts[torch.randint(len(tr_starts), (a.batch,), device=dev)]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, pl, sg = loss_fn(batch(tr, s))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss, pl, sg

for _ in range(30):  # warmup kernels (tiny lr)
    train_step(1e-8)
torch.cuda.synchronize(); t0 = time.time()
for _ in range(100):
    train_step(1e-8)
torch.cuda.synchronize()
sec_per_step = (time.time() - t0) / 100
budget_epochs = int(a.max_hours * 3600 * 0.93 / (sec_per_step * steps_per_epoch))
epochs = max(1, min(a.epochs, budget_epochs))
total = epochs * steps_per_epoch
warm = int(a.warmup_epochs * steps_per_epoch)
P(f'{sec_per_step*1000:.0f} ms/step, {steps_per_epoch} steps/epoch ({sec_per_step*steps_per_epoch/60:.1f} min) '
  f'-> {epochs} epochs ({total} steps, ~{total*sec_per_step/3600:.1f} h)')
json.dump({'epochs': epochs, 'steps_per_epoch': steps_per_epoch, 'sec_per_step': sec_per_step},
          open(os.path.join(a.out, 'plan.json'), 'w'))
P('diag init', json.dumps({k: [round(x, 2) for x in v] for k, v in rollout_diag().items()}))

t0 = time.time()
for step in range(total):
    lr = a.lr * (step + 1) / warm if step < warm else 0.5 * a.lr * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
    loss, pl, sg = train_step(lr)
    if step % 500 == 0:
        P(f'step {step}/{total} ep {step/steps_per_epoch:.2f} loss {loss.item():.4f} pred {pl.item():.4f} '
          f'sigreg {sg.item():.3f} lr {lr:.2e} {(time.time()-t0)/3600:.2f}h')
    if not math.isfinite(loss.item()):
        P('NaN loss, stopping'); save('nan', step / steps_per_epoch); sys.exit(1)
    if (step + 1) % steps_per_epoch == 0:
        ep = (step + 1) // steps_per_epoch
        if ep % a.ckpt_every == 0 or ep == epochs:
            p = save(f'ep{ep}', ep)
            d = rollout_diag()
            P(f'== epoch {ep}: val pred {val_loss():.4f}  diag', json.dumps({k: [round(x, 2) for x in v] for k, v in d.items()}), 'saved', p)
save('final', epochs)
P('TRAIN_DONE')
