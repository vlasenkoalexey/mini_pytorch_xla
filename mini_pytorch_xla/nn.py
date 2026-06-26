"""A tiny torch.nn-like API on top of the StableHLO autograd Tensor.

Deliberately mirrors the shapes/wiring of the train_xla.py transformer so the
model code reads the same. Token/target ids are one-hot'd on the host, so the
embedding and the loss are plain matmuls/reductions — no gather/scatter needed.
"""

from __future__ import annotations

import numpy as np
from . import tensor as T
from .tensor import Tensor, from_numpy, no_grad


def parameter(np_array) -> Tensor:
    return from_numpy(np.asarray(np_array, dtype=np.float32), requires_grad=True)


class Module:
    def parameters(self):
        seen, out = set(), []

        def visit(o):
            if isinstance(o, Tensor):
                if o.requires_grad and id(o) not in seen:
                    seen.add(id(o)); out.append(o)
            elif isinstance(o, Module):
                for v in o.__dict__.values():
                    visit(v)
            elif isinstance(o, (list, tuple)):
                for v in o:
                    visit(v)
        visit(self)
        return out

    def __call__(self, *a, **k):
        return self.forward(*a, **k)


# ----------------------------------------------------------------------------- #
# layers
# ----------------------------------------------------------------------------- #
class Linear(Module):
    """y = x @ W (+ b). Weight stored [in, out] (so forward is a clean matmul)."""

    def __init__(self, d_in, d_out, bias=True, scale=None):
        s = scale if scale is not None else (1.0 / np.sqrt(d_in))
        self.W = parameter(np.random.randn(d_in, d_out) * s)
        self.b = parameter(np.zeros(d_out)) if bias else None

    def forward(self, x):
        y = T.linear(x, self.W)
        return y + self.b if self.b is not None else y


class Embedding(Module):
    """Lookup as one-hot @ table. `idx_onehot` is [..., num_embeddings] (host one-hot)."""

    def __init__(self, num_embeddings, dim, scale=0.02):
        self.W = parameter(np.random.randn(num_embeddings, dim) * scale)

    def forward(self, idx_onehot):
        return T.linear(idx_onehot, self.W)


class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        self.g = parameter(np.ones(dim))
        self.b = parameter(np.zeros(dim))
        self.eps = eps

    def forward(self, x):
        ax = [x.ndim - 1]
        n = x.shape[-1]
        mean = T.reduce_sum(x, ax, keepdim=True) / float(n)
        xc = x - mean
        var = T.reduce_sum(xc * xc, ax, keepdim=True) / float(n)
        xn = xc * T.rsqrt(var + self.eps)
        return xn * self.g + self.b


# ----------------------------------------------------------------------------- #
# functional
# ----------------------------------------------------------------------------- #
def gelu(x):
    # tanh approximation (exact-erf gelu isn't a StableHLO primitive)
    x3 = x * x * x
    inner = (x + x3 * 0.044715) * 0.7978845608028654
    return x * 0.5 * (T.tanh(inner) + 1.0)


def softmax(x, axis=-1):
    ax = axis if axis >= 0 else x.ndim + axis
    m = T.reduce_max(x, [ax], keepdim=True)          # detached (shift only)
    e = T.exp(x - m)
    return e / T.reduce_sum(e, [ax], keepdim=True)


def cross_entropy(logits, target_onehot):
    """logits, target_onehot both [..., V]; mean NLL over all leading positions."""
    V = logits.shape[-1]
    n = int(np.prod(logits.shape[:-1]))
    lg = T.reshape(logits, (n, V))
    tg = T.reshape(target_onehot, (n, V))
    m = T.reduce_max(lg, [1], keepdim=True)          # detached
    lse = T.log(T.reduce_sum(T.exp(lg - m), [1], keepdim=True)) + m
    logprob = lg - lse                                # [n, V]
    return T.reduce_sum(tg * logprob, [0, 1]) * (-1.0 / n)


# ----------------------------------------------------------------------------- #
# optimizer
# ----------------------------------------------------------------------------- #
class AdamW:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr, self.b1, self.b2 = lr, betas[0], betas[1]
        self.eps, self.wd = eps, weight_decay
        self.t = 0
        self.m = [T.zeros(p.shape) for p in self.params]
        self.v = [T.zeros(p.shape) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        with no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad
                self.m[i] = self.m[i] * self.b1 + g * (1.0 - self.b1)
                self.v[i] = self.v[i] * self.b2 + (g * g) * (1.0 - self.b2)
                mhat = self.m[i] * (1.0 / bc1)
                vhat = self.v[i] * (1.0 / bc2)
                update = mhat / (T.sqrt(vhat) + self.eps)
                if self.wd:
                    update = update + p * self.wd
                new_p = p - update * self.lr
                p.buf = new_p.buf           # in-place param update (keep the leaf object)
