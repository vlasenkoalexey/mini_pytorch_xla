"""Char-level Shakespeare transformer trained on the TPU via mini-pytorch-xla.

Same architecture as train_xla.py (pre-LN transformer, AdamW), but every op is
lowered to StableHLO and executed eagerly on the TPU through our pure-ctypes PJRT
client — no torch, no torch_xla, no jax. Token/target ids are one-hot'd on the
host so embedding and loss are matmuls/reductions.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nn
import tensor as T
from nn import Module, Linear, Embedding, LayerNorm
from tensor import from_numpy


# ---- data ------------------------------------------------------------------- #
def get_text():
    path = "data/shakespeare.txt"
    if Path(path).exists():
        return open(path, encoding="utf-8").read()
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    txt = requests.get(url).text
    Path("data").mkdir(exist_ok=True)
    open(path, "w", encoding="utf-8").write(txt)
    return txt


# ---- model (mirrors train_xla.py) ------------------------------------------- #
class MultiHeadAttention(Module):
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.h, self.dk = num_heads, d_model // num_heads
        self.q = Linear(d_model, d_model, bias=False)
        self.k = Linear(d_model, d_model, bias=False)
        self.v = Linear(d_model, d_model, bias=False)
        self.o = Linear(d_model, d_model, bias=False)
        self.scale = self.dk ** -0.5

    def _split(self, x, B, L):
        x = T.reshape(x, (B, L, self.h, self.dk))
        return T.transpose(x, [0, 2, 1, 3])          # [B,H,L,dk]

    def forward(self, x, mask):
        B, L, D = x.shape
        q = self._split(self.q(x), B, L)
        k = self._split(self.k(x), B, L)
        v = self._split(self.v(x), B, L)
        scores = T.bmm(q, T.transpose_last2(k)) * self.scale   # [B,H,L,L]
        scores = scores + mask                                 # broadcast [1,1,L,L]
        attn = nn.softmax(scores, axis=-1)
        out = T.bmm(attn, v)                                   # [B,H,L,dk]
        out = T.reshape(T.transpose(out, [0, 2, 1, 3]), (B, L, D))
        return self.o(out)


class MLP(Module):
    def __init__(self, d_model, d_ff):
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class Block(Module):
    def __init__(self, d_model, num_heads, d_ff):
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.mlp = MLP(d_model, d_ff)
        self.n1 = LayerNorm(d_model)
        self.n2 = LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + self.attn(self.n1(x), mask)
        x = x + self.mlp(self.n2(x))
        return x


class LLM(Module):
    def __init__(self, vocab, block_size, d_model, num_layers, num_heads, d_ff):
        self.block_size = block_size
        self.tok = Embedding(vocab, d_model)
        self.pos = Embedding(block_size, d_model)
        self.blocks = [Block(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.lnf = LayerNorm(d_model)
        self.head = Linear(d_model, vocab, bias=False)
        # constant causal mask [1,1,L,L]: 0 on/below diag, -1e9 above
        m = np.triu(np.full((block_size, block_size), -1e9, np.float32), k=1)
        self._mask = from_numpy(m.reshape(1, 1, block_size, block_size))
        # constant position one-hot [L, block_size] (identity since L==block_size)
        self._pos_oh = from_numpy(np.eye(block_size, dtype=np.float32))

    def forward(self, tok_onehot):                 # tok_onehot: [B,L,V]
        B, L, V = tok_onehot.shape
        x = self.tok(tok_onehot) + self.pos(self._pos_oh)   # [B,L,D] + [L,D]
        for blk in self.blocks:
            x = blk(x, self._mask)
        x = self.lnf(x)
        return self.head(x)                          # [B,L,V]


def onehot(ids, V):
    out = np.zeros((*ids.shape, V), np.float32)
    np.put_along_axis(out, ids[..., None], 1.0, axis=-1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--block_size", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log_every", type=int, default=20)
    args = ap.parse_args()
    np.random.seed(0)

    text = get_text()
    chars = sorted(set(text))
    V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    B, L = args.batch_size, args.block_size

    def batch():
        ix = np.random.randint(0, len(data) - L - 1, size=B)
        x = np.stack([data[i:i + L] for i in ix])
        y = np.stack([data[i + 1:i + 1 + L] for i in ix])
        return onehot(x, V), onehot(y, V)

    model = LLM(V, L, args.d_model, args.layers, args.heads, args.d_ff)
    params = model.parameters()
    opt = nn.AdamW(params, lr=args.lr)
    nparam = sum(int(np.prod(p.shape)) for p in params)
    print(f"mini-pytorch-xla | TPU | {nparam/1e6:.2f}M params | vocab {V} | "
          f"{args.layers}L d{args.d_model} h{args.heads} block {L} batch {B}")
    print("Step 0 compiles each op's StableHLO (one-time); steps then reuse cached executables.\n")

    import time
    for step in range(args.steps):
        xb, yb = batch()
        x = from_numpy(xb)
        y = from_numpy(yb)
        logits = model(x)
        loss = nn.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"Step {step:5d} | Loss: {loss.item():.4f}", flush=True)

    print("\nTraining complete on TPU via mini-pytorch-xla.")


if __name__ == "__main__":
    main()
