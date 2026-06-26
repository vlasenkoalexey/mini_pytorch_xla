"""Hand-rolled eager autograd Tensor — the original *no-PyTorch* frontend.

This predates the __torch_dispatch__ backend: it depends on no torch at all. It is a
tiny reverse-mode autograd engine where each op lowers to StableHLO on the TPU. It
**reuses the shared layers** of mini-pytorch-xla — `ops.py` for the buffer-level
StableHLO lowerings and `pjrt.py` for the libtpu runtime — so this file is purely the
autograd layer (tape + backward formulas), with no duplicated MLIR.

Forward op records a `_backward` closure + its differentiable parents; `backward()`
seeds the scalar output with 1, walks the tape in reverse, and each closure
accumulates parent grads using the same ops (so backward runs on the TPU too). A
`no_grad` flag held during the walk stops second-order taping.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # for mini_pytorch_xla
import numpy as np
from mini_pytorch_xla import ops

_RECORD = True


class no_grad:
    def __enter__(self):
        global _RECORD
        self._prev = _RECORD
        _RECORD = False

    def __exit__(self, *a):
        global _RECORD
        _RECORD = self._prev


class Tensor:
    def __init__(self, buf, requires_grad=False, _prev=(), _backward=None):
        self.buf = buf
        self.requires_grad = requires_grad
        self.grad = None
        self._prev = list(_prev)
        self._backward = _backward or (lambda: None)

    @property
    def shape(self): return self.buf.shape

    @property
    def dtype(self): return self.buf.dtype

    @property
    def ndim(self): return len(self.buf.shape)

    def numpy(self): return self.buf.to_numpy()

    def item(self): return float(self.numpy().reshape(-1)[0])

    def __add__(s, o): return add(s, _as(o))
    def __radd__(s, o): return add(_as(o), s)
    def __sub__(s, o): return sub(s, _as(o))
    def __rsub__(s, o): return sub(_as(o), s)
    def __mul__(s, o): return mul(s, _as(o))
    def __rmul__(s, o): return mul(_as(o), s)
    def __truediv__(s, o): return div(s, _as(o))
    def __rtruediv__(s, o): return div(_as(o), s)
    def __neg__(s): return neg(s)
    def __matmul__(s, o): return matmul(s, o)

    def backward(self):
        topo, seen = [], set()

        def build(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._prev:
                build(p)
            topo.append(t)

        build(self)
        self.grad = ones(self.shape)            # seed dL/dL = 1
        with no_grad():
            for t in reversed(topo):
                t._backward()


# --- construction ----------------------------------------------------------- #
def from_numpy(arr, requires_grad=False) -> Tensor:
    arr = np.asarray(arr)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    if arr.dtype == np.int64:
        arr = arr.astype(np.int32)
    return Tensor(ops.from_np(arr), requires_grad=requires_grad)


def _as(o) -> Tensor:
    return o if isinstance(o, Tensor) else from_numpy(np.asarray(o, dtype=np.float32))


def ones(shape) -> Tensor:
    return Tensor(ops.const(1.0, tuple(int(s) for s in shape)))


def zeros(shape) -> Tensor:
    return Tensor(ops.const(0.0, tuple(int(s) for s in shape)))


# --- autograd plumbing ------------------------------------------------------- #
def _record(out: Tensor, parents, backward) -> Tensor:
    if _RECORD and any(p.requires_grad for p in parents):
        out.requires_grad = True
        out._prev = [p for p in parents if p.requires_grad]
        out._backward = backward
    return out


def _acc(p: Tensor, g: Tensor):
    if not p.requires_grad:
        return
    g = _unbroadcast(g, p.shape)
    p.grad = g if p.grad is None else add(p.grad, g)


def _unbroadcast(g: Tensor, shape) -> Tensor:
    shape = tuple(int(s) for s in shape)
    if g.shape == shape:
        return g
    while g.ndim > len(shape):
        g = reduce_sum(g, [0], keepdim=False)
    axes = [i for i, (gd, td) in enumerate(zip(g.shape, shape)) if td == 1 and gd != 1]
    if axes:
        g = reduce_sum(g, axes, keepdim=True)
    if g.shape != shape:
        g = reshape(g, shape)
    return g


# --- elementwise binary (forward via ops.binary; backward = chain rule) ------ #
def add(a, b):
    out = Tensor(ops.binary("add", a.buf, b.buf))
    return _record(out, [a, b], lambda: (_acc(a, out.grad), _acc(b, out.grad)))


def sub(a, b):
    out = Tensor(ops.binary("subtract", a.buf, b.buf))
    return _record(out, [a, b], lambda: (_acc(a, out.grad), _acc(b, neg(out.grad))))


def mul(a, b):
    out = Tensor(ops.binary("multiply", a.buf, b.buf))
    return _record(out, [a, b], lambda: (_acc(a, mul(out.grad, b)), _acc(b, mul(out.grad, a))))


def div(a, b):
    out = Tensor(ops.binary("divide", a.buf, b.buf))

    def bw():
        _acc(a, div(out.grad, b))
        _acc(b, neg(div(mul(out.grad, a), mul(b, b))))
    return _record(out, [a, b], bw)


# --- elementwise unary ------------------------------------------------------- #
def neg(a):
    out = Tensor(ops.unary("negate", a.buf))
    return _record(out, [a], lambda: _acc(a, neg(out.grad)))


def exp(a):
    out = Tensor(ops.unary("exponential", a.buf))
    return _record(out, [a], lambda: _acc(a, mul(out.grad, out)))


def log(a):
    out = Tensor(ops.unary("log", a.buf))
    return _record(out, [a], lambda: _acc(a, div(out.grad, a)))


def sqrt(a):
    out = Tensor(ops.unary("sqrt", a.buf))
    return _record(out, [a], lambda: _acc(a, div(out.grad, mul(from_numpy(np.float32(2.0)), out))))


def rsqrt(a):
    out = Tensor(ops.unary("rsqrt", a.buf))          # out = a^-0.5 ; d/da = -0.5*out/a
    return _record(out, [a], lambda: _acc(a, mul(out.grad, mul(from_numpy(np.float32(-0.5)), div(out, a)))))


def tanh(a):
    out = Tensor(ops.unary("tanh", a.buf))           # d/da = 1 - out^2
    return _record(out, [a], lambda: _acc(a, mul(out.grad, sub(from_numpy(np.float32(1.0)), mul(out, out)))))


# --- reductions -------------------------------------------------------------- #
def reduce_sum(a, axes, keepdim=False):
    axes = [int(x) for x in axes]
    out = Tensor(ops.reduce_sum(a.buf, axes))
    if keepdim:
        kd = tuple(1 if i in axes else s for i, s in enumerate(a.shape))
        out = reshape(out, kd)

    def bw():
        kd = tuple(1 if i in axes else s for i, s in enumerate(a.shape))
        _acc(a, broadcast_to(reshape(out.grad, kd), a.shape))
    return _record(out, [a], bw)


def reduce_max(a, axes, keepdim=False):
    axes = [int(x) for x in axes]
    out = Tensor(ops.reduce_max(a.buf, axes))        # detached (shift only)
    if keepdim:
        kd = tuple(1 if i in axes else s for i, s in enumerate(a.shape))
        out = reshape(out, kd)
    return out


# --- shape ------------------------------------------------------------------- #
def broadcast_to(a, shape):
    out = Tensor(ops.broadcast_to(a.buf, tuple(int(s) for s in shape)))
    return _record(out, [a], lambda: _acc(a, out.grad))


def reshape(a, shape):
    shape = tuple(int(s) for s in shape)
    if a.shape == shape:
        return a
    out = Tensor(ops.reshape(a.buf, shape))
    old = a.shape
    return _record(out, [a], lambda: _acc(a, reshape(out.grad, old)))


def transpose(a, perm):
    perm = [int(p) for p in perm]
    out = Tensor(ops.transpose(a.buf, perm))
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return _record(out, [a], lambda: _acc(a, transpose(out.grad, inv)))


def transpose_last2(a):
    perm = list(range(a.ndim))
    perm[-1], perm[-2] = perm[-2], perm[-1]
    return transpose(a, perm)


# --- matmul ------------------------------------------------------------------ #
def mm(a, b):
    out = Tensor(ops.mm(a.buf, b.buf))

    def bw():
        _acc(a, mm(out.grad, transpose(b, [1, 0])))
        _acc(b, mm(transpose(a, [1, 0]), out.grad))
    return _record(out, [a, b], bw)


def bmm(a, b):
    out = Tensor(ops.bmm(a.buf, b.buf))

    def bw():
        _acc(a, bmm(out.grad, transpose_last2(b)))
        _acc(b, bmm(transpose_last2(a), out.grad))
    return _record(out, [a, b], bw)


def matmul(a, b):
    return mm(a, b) if a.ndim == 2 and b.ndim == 2 else bmm(a, b)


def linear(x, W):
    """x[...,in] @ W[in,out] -> [...,out], folding leading dims for a 2-D matmul."""
    in_dim = W.shape[0]
    lead = x.shape[:-1]
    x2 = reshape(x, (int(np.prod(lead)) if lead else 1, in_dim))
    y2 = mm(x2, W)
    return reshape(y2, tuple(lead) + (W.shape[1],))
