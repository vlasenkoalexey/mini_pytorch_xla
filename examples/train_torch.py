"""Train a real torch.nn transformer on the TPU through mini-pytorch-xla.

Unlike train_mini.py (which used a hand-rolled Tensor/autograd), here the model is
plain `torch.nn`, the loss is `F.cross_entropy`, and the optimizer is the real
`torch.optim.AdamW`. PyTorch owns the program and autograd; mini-pytorch-xla only
provides the device + op lowering via __torch_dispatch__ — exactly like PyTorch/XLA.
Move a module onto the TPU with `to_xla_(module)`.

(Uses relu instead of erf-gelu and dropout=0 to stay within the implemented op set.)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mini_pytorch_xla import backend as xb


def to_xla_(module: nn.Module) -> nn.Module:
    """In-place: replace every parameter with an XLATensor leaf on the TPU."""
    for m in module.modules():
        for name, p in list(m.named_parameters(recurse=False)):
            xp = xb.to_xla(p.detach())
            xp.requires_grad_(True)
            setattr(m, name, nn.Parameter(xp, requires_grad=True))
    return module


# ---- model (plain PyTorch) -------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.h, self.dk = h, d // h
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.f1, self.f2 = nn.Linear(d, ff), nn.Linear(ff, d)

    def attn(self, x, mask):
        B, L, D = x.shape
        sp = lambda t: t.view(B, L, self.h, self.dk).transpose(1, 2)
        q, k, v = sp(self.q(x)), sp(self.k(x)), sp(self.v(x))
        s = (q @ k.transpose(-2, -1)) * (self.dk ** -0.5) + mask
        a = F.softmax(s, dim=-1)
        return self.o((a @ v).transpose(1, 2).reshape(B, L, D))

    def forward(self, x, mask):
        x = x + self.attn(self.n1(x), mask)
        x = x + self.f2(F.relu(self.f1(self.n2(x))))
        return x


class LLM(nn.Module):
    def __init__(self, vocab, block, d, layers, h, ff):
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block, d)
        self.blocks = nn.ModuleList([Block(d, h, ff) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        mask = torch.triu(torch.full((block, block), -1e9), diagonal=1).view(1, 1, block, block)
        self.register_buffer("mask", mask)

    def forward(self, idx):                     # idx: LongTensor [B, L] (kept on host)
        B, L = idx.shape
        pos = torch.arange(L)
        x = self.tok(idx) + self.pos(pos)        # embedding lowers via one-hot @ table
        for blk in self.blocks:
            x = blk(x, self._mask_xla)
        return self.head(self.lnf(x))


def train_step(model, opt, x, y, V):
    logits = model(x)
    logp = F.log_softmax(logits.reshape(-1, V), dim=-1)
    onehot = xb.to_xla(F.one_hot(y.reshape(-1), V).float())
    loss = -(logp * onehot).sum(dim=-1).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss


def collect_profile(model, opt, make_batch, V, logdir, prof_steps):
    """Capture an on-device op profile of the backend and write an xprof xplane.pb.

    Uses the project's own profiler (mini_pytorch_xla.profiler) — no torch_xla / jax.
    Each StableHLO op is timed across the PJRT device-completion barrier; since eager
    ops run sequentially+synchronously, the timeline is faithful. Written as an XSpace
    (xplane.pb) that xprof / TensorBoard's Trace Viewer reads.
    """
    from mini_pytorch_xla import profiler

    for _ in range(3):                           # warm up: compile all op programs
        x, y = make_batch()
        train_step(model, opt, x, y, V)

    print(f"profiling {prof_steps} steps on the TPU -> {logdir}")
    with profiler.OpProfile() as prof:
        for _ in range(prof_steps):
            x, y = make_batch()
            loss = train_step(model, opt, x, y, V)
            xb.to_cpu(loss.detach())             # force device completion in-window
    print(prof.report())
    path = prof.write_xspace(logdir)
    print(f"\nxplane written: {path}")
    print(f"inspect with:   xprof -l {logdir} -p 8791")


@torch.no_grad()
def generate(model, V, L, stoi, itos, prompt, max_new_tokens, temperature=0.8, top_k=40):
    """nanoGPT-style sampling. Each step's forward runs on the TPU (fixed block_size
    window -> cached executable); softmax/top-k/multinomial sampling is on the host."""
    out = [stoi[c] for c in prompt if c in stoi] or [0]
    for _ in range(max_new_tokens):
        window = out[-L:]
        if len(window) < L:
            window = [0] * (L - len(window)) + window          # left-pad to block_size
        idx = torch.tensor([window], dtype=torch.long)          # [1, L]
        logits = xb.to_cpu(model(idx).detach())[0, -1] / temperature   # forward on TPU
        if top_k:
            v, _ = torch.topk(logits, min(top_k, V))
            logits[logits < v[-1]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        out.append(int(torch.multinomial(probs, 1)))
    return "".join(itos[i] for i in out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", type=str, default=None, help="logdir to write a TPU xprof trace")
    ap.add_argument("--profile_steps", type=int, default=5)
    ap.add_argument("--generate", type=int, default=0, help="generate N chars after training")
    ap.add_argument("--prompt", type=str, default="\n")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--block_size", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log_every", type=int, default=20)
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)

    path = "data/shakespeare.txt"
    if not Path(path).exists():
        Path("data").mkdir(exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        Path(path).write_text(requests.get(url).text, encoding="utf-8")
    text = Path(path).read_text(encoding="utf-8")
    chars = sorted(set(text)); V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    B, L = args.batch_size, args.block_size

    model = LLM(V, L, args.d_model, args.layers, args.heads, args.d_ff)
    to_xla_(model)                                    # params -> TPU
    model._mask_xla = xb.to_xla(model.mask.float())   # mask constant on TPU
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, foreach=False, capturable=False)

    nparam = sum(p.numel() for p in model.parameters())
    print(f"real torch.nn transformer on mini-pytorch-xla | TPU | {nparam/1e6:.2f}M params | vocab {V}")
    print("PyTorch owns autograd+optimizer; every aten op lowered to StableHLO on the TPU.\n")

    def batch():
        ix = torch.randint(0, len(data) - L - 1, (B,))
        x = torch.stack([data[i:i + L] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + L] for i in ix])
        return x, y

    if args.profile:
        collect_profile(model, opt, batch, V, args.profile, args.profile_steps)
        return

    for step in range(args.steps):
        x, y = batch()                                # ids stay on host (embedding gather)
        loss = train_step(model, opt, x, y, V)        # fwd+bwd+AdamW, all on TPU
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"Step {step:5d} | Loss: {xb.to_cpu(loss.detach()).item():.4f}", flush=True)

    print("\nTrained a real torch.nn model on the TPU via __torch_dispatch__.")

    if args.generate:
        itos = chars
        print(f"\n--- generating {args.generate} chars on the TPU "
              f"(prompt={args.prompt!r}, temp={args.temperature}, top_k={args.top_k}) ---")
        print(generate(model, V, L, stoi, itos, args.prompt,
                       args.generate, args.temperature, args.top_k))


if __name__ == "__main__":
    main()
