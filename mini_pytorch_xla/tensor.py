"""Eager autograd Tensor whose every primitive lowers to StableHLO on the TPU.

Design mirrors PyTorch/XLA's *idea* — intercept high-level ops, lower each to an
XLA program, run on device — but here the "interception" is just method calls on
this Tensor, and lowering is one StableHLO module per op (hlo.run, cached).

Autograd is textbook reverse-mode: forward ops record a `_backward` closure and
their differentiable parents; `Tensor.backward()` seeds the scalar output with 1,
walks the tape in reverse, and each closure accumulates parent grads using the
SAME primitives — so the backward pass also executes on the TPU. A `no_grad`
flag (held during the backward walk) stops those accumulation ops from taping
themselves (no second-order graph).
"""

from __future__ import annotations

import numpy as np
from . import pjrt
from .hlo import ttype, run

_RECORD = True


class no_grad:
    def __enter__(self):
        global _RECORD
        self._prev = _RECORD
        _RECORD = False

    def __exit__(self, *a):
        global _RECORD
        _RECORD = self._prev


def _f(v) -> str:
    return f"{float(v):.8e}"


def _arr(xs) -> str:
    return "[" + ", ".join(str(int(x)) for x in xs) + "]"


class Tensor:
    def __init__(self, buf, requires_grad=False, _prev=(), _backward=None):
        self.buf = buf
        self.requires_grad = requires_grad
        self.grad: Tensor | None = None
        self._prev = list(_prev)
        self._backward = _backward or (lambda: None)

    @property
    def shape(self):
        return self.buf.shape

    @property
    def dtype(self):
        return self.buf.dtype

    @property
    def ndim(self):
        return len(self.buf.shape)

    def numpy(self):
        return self.buf.to_numpy()

    def item(self):
        return float(self.numpy().reshape(-1)[0])

    # operator sugar
    def __add__(s, o): return add(s, _as(o, s))
    def __radd__(s, o): return add(_as(o, s), s)
    def __sub__(s, o): return sub(s, _as(o, s))
    def __rsub__(s, o): return sub(_as(o, s), s)
    def __mul__(s, o): return mul(s, _as(o, s))
    def __rmul__(s, o): return mul(_as(o, s), s)
    def __truediv__(s, o): return div(s, _as(o, s))
    def __rtruediv__(s, o): return div(_as(o, s), s)
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
        self.grad = ones(self.shape)
        with no_grad():
            for t in reversed(topo):
                t._backward()


def from_numpy(arr, requires_grad=False) -> Tensor:
    arr = np.asarray(arr)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    if arr.dtype == np.int64:
        arr = arr.astype(np.int32)
    return Tensor(pjrt.client().from_host(np.ascontiguousarray(arr)),
                  requires_grad=requires_grad)


def _as(o, like: Tensor) -> Tensor:
    if isinstance(o, Tensor):
        return o
    return from_numpy(np.asarray(o, dtype=np.float32))  # python scalar -> 0-d


def ones(shape) -> Tensor:
    return from_numpy(np.ones(shape, dtype=np.float32))


def zeros(shape) -> Tensor:
    return from_numpy(np.zeros(shape, dtype=np.float32))


# ----------------------------------------------------------------------------- #
# raw executor + autograd plumbing
# ----------------------------------------------------------------------------- #
def _apply(module: str, inputs: list[Tensor]) -> Tensor:
    (out_buf,) = run(module, [t.buf for t in inputs], n_out=1)
    return Tensor(out_buf)


def _record(out: Tensor, parents, backward):
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


# ----------------------------------------------------------------------------- #
# broadcasting
# ----------------------------------------------------------------------------- #
def _bshape(sa, sb):
    ra, rb = len(sa), len(sb)
    r = max(ra, rb)
    sa = (1,) * (r - ra) + tuple(sa)
    sb = (1,) * (r - rb) + tuple(sb)
    out = []
    for x, y in zip(sa, sb):
        if x == y or x == 1 or y == 1:
            out.append(max(x, y))
        else:
            raise ValueError(f"cannot broadcast {sa} vs {sb}")
    return tuple(out)


def _bdims(in_shape, out_shape):
    r = len(out_shape) - len(in_shape)
    return list(range(r, len(out_shape)))


def broadcast_to(a: Tensor, shape) -> Tensor:
    shape = tuple(int(s) for s in shape)
    if a.shape == shape:
        return a
    dims = _bdims(a.shape, shape)
    Ta, To = ttype(a.shape, a.dtype), ttype(shape, a.dtype)
    m = (f"module {{\n  func.func public @main(%a: {Ta}) -> {To} {{\n"
         f"    %r = stablehlo.broadcast_in_dim %a, dims = {_arr(dims)} : ({Ta}) -> {To}\n"
         f"    return %r : {To}\n  }}\n}}")
    out = _apply(m, [a])
    return _record(out, [a], lambda: _acc(a, out.grad))


def _unbroadcast(g: Tensor, shape) -> Tensor:
    shape = tuple(int(s) for s in shape)
    if g.shape == shape:
        return g
    # sum away leading extra dims
    while g.ndim > len(shape):
        g = reduce_sum(g, [0], keepdim=False)
    # sum dims that were 1 in target
    axes = [i for i, (gd, td) in enumerate(zip(g.shape, shape)) if td == 1 and gd != 1]
    if axes:
        g = reduce_sum(g, axes, keepdim=True)
    if g.shape != shape:
        g = reshape(g, shape)
    return g


# ----------------------------------------------------------------------------- #
# elementwise binary
# ----------------------------------------------------------------------------- #
def _binary(opname, a, b, out_shape):
    Ta, Tb, To = ttype(a.shape, a.dtype), ttype(b.shape, a.dtype), ttype(out_shape, a.dtype)
    lines = ["module {", f"  func.func public @main(%a: {Ta}, %b: {Tb}) -> {To} {{"]
    av = "%a"
    if a.shape != out_shape:
        lines.append(f"    %ab = stablehlo.broadcast_in_dim %a, dims = {_arr(_bdims(a.shape, out_shape))} : ({Ta}) -> {To}")
        av = "%ab"
    bv = "%b"
    if b.shape != out_shape:
        lines.append(f"    %bb = stablehlo.broadcast_in_dim %b, dims = {_arr(_bdims(b.shape, out_shape))} : ({Tb}) -> {To}")
        bv = "%bb"
    lines.append(f"    %r = stablehlo.{opname} {av}, {bv} : {To}")
    lines += [f"    return %r : {To}", "  }", "}"]
    return _apply("\n".join(lines), [a, b])


def add(a, b):
    out = _binary("add", a, b, _bshape(a.shape, b.shape))
    return _record(out, [a, b], lambda: (_acc(a, out.grad), _acc(b, out.grad)))


def sub(a, b):
    out = _binary("subtract", a, b, _bshape(a.shape, b.shape))
    return _record(out, [a, b], lambda: (_acc(a, out.grad), _acc(b, neg(out.grad))))


def mul(a, b):
    out = _binary("multiply", a, b, _bshape(a.shape, b.shape))
    return _record(out, [a, b], lambda: (_acc(a, mul(out.grad, b)), _acc(b, mul(out.grad, a))))


def div(a, b):
    out = _binary("divide", a, b, _bshape(a.shape, b.shape))

    def bw():
        _acc(a, div(out.grad, b))
        _acc(b, neg(div(mul(out.grad, a), mul(b, b))))
    return _record(out, [a, b], bw)


# ----------------------------------------------------------------------------- #
# elementwise unary
# ----------------------------------------------------------------------------- #
def _unary(opname, a) -> Tensor:
    T = ttype(a.shape, a.dtype)
    m = (f"module {{\n  func.func public @main(%a: {T}) -> {T} {{\n"
         f"    %r = stablehlo.{opname} %a : {T}\n    return %r : {T}\n  }}\n}}")
    return _apply(m, [a])


def neg(a):
    out = _unary("negate", a)
    return _record(out, [a], lambda: _acc(a, neg(out.grad)))


def exp(a):
    out = _unary("exponential", a)
    return _record(out, [a], lambda: _acc(a, mul(out.grad, out)))


def log(a):
    out = _unary("log", a)
    return _record(out, [a], lambda: _acc(a, div(out.grad, a)))


def sqrt(a):
    out = _unary("sqrt", a)
    return _record(out, [a], lambda: _acc(a, div(out.grad, mul(from_numpy(np.float32(2.0)), out))))


def rsqrt(a):
    out = _unary("rsqrt", a)  # out = a^-0.5 ; d/da = -0.5 * out / a
    return _record(out, [a], lambda: _acc(a, mul(out.grad, mul(from_numpy(np.float32(-0.5)), div(out, a)))))


def tanh(a):
    out = _unary("tanh", a)  # d/da = 1 - out^2
    return _record(out, [a], lambda: _acc(a, mul(out.grad, sub(from_numpy(np.float32(1.0)), mul(out, out)))))


# ----------------------------------------------------------------------------- #
# reductions
# ----------------------------------------------------------------------------- #
def _reduce(opname, init, a, axes):
    axes = sorted(int(x) for x in axes)
    out_shape = tuple(s for i, s in enumerate(a.shape) if i not in axes)
    Tin, Tsc = ttype(a.shape, a.dtype), ttype((), a.dtype)
    Tout = ttype(out_shape, a.dtype)
    m = (
        "module {\n"
        f"  func.func public @main(%a: {Tin}) -> {Tout} {{\n"
        f"    %init = stablehlo.constant dense<{_f(init)}> : {Tsc}\n"
        f'    %r = "stablehlo.reduce"(%a, %init) ({{\n'
        f"    ^bb0(%x: {Tsc}, %y: {Tsc}):\n"
        f"      %s = stablehlo.{opname} %x, %y : {Tsc}\n"
        f"      stablehlo.return %s : {Tsc}\n"
        f"    }}) {{dimensions = array<i64: {', '.join(str(x) for x in axes)}>}} : ({Tin}, {Tsc}) -> {Tout}\n"
        f"    return %r : {Tout}\n  }}\n}}"
    )
    return _apply(m, [a]), out_shape


def reduce_sum(a, axes, keepdim=False):
    out, out_shape = _reduce("add", 0.0, a, axes)
    if keepdim:
        kd = tuple(1 if i in [int(x) for x in axes] else s for i, s in enumerate(a.shape))
        out = reshape(out, kd)

    def bw():
        g = out.grad
        kd = tuple(1 if i in [int(x) for x in axes] else s for i, s in enumerate(a.shape))
        _acc(a, broadcast_to(reshape(g, kd), a.shape))
    return _record(out, [a], bw)


def reduce_max(a, axes, keepdim=False):
    # used only for numerical stability (shift), treated as a constant (no grad)
    out, out_shape = _reduce("maximum", -3.0e38, a, axes)
    if keepdim:
        kd = tuple(1 if i in [int(x) for x in axes] else s for i, s in enumerate(a.shape))
        out = reshape(out, kd)
    return out  # detached on purpose


# ----------------------------------------------------------------------------- #
# shape ops
# ----------------------------------------------------------------------------- #
def reshape(a, shape):
    shape = tuple(int(s) for s in shape)
    if a.shape == shape:
        return a
    Ta, To = ttype(a.shape, a.dtype), ttype(shape, a.dtype)
    m = (f"module {{\n  func.func public @main(%a: {Ta}) -> {To} {{\n"
         f"    %r = stablehlo.reshape %a : ({Ta}) -> {To}\n    return %r : {To}\n  }}\n}}")
    out = _apply(m, [a])
    old = a.shape
    return _record(out, [a], lambda: _acc(a, reshape(out.grad, old)))


def transpose(a, perm):
    perm = [int(p) for p in perm]
    out_shape = tuple(a.shape[p] for p in perm)
    Ta, To = ttype(a.shape, a.dtype), ttype(out_shape, a.dtype)
    m = (f'module {{\n  func.func public @main(%a: {Ta}) -> {To} {{\n'
         f'    %r = stablehlo.transpose %a, dims = {_arr(perm)} : ({Ta}) -> {To}\n'
         f"    return %r : {To}\n  }}\n}}")
    out = _apply(m, [a])
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return _record(out, [a], lambda: _acc(a, transpose(out.grad, inv)))


def transpose_last2(a):
    perm = list(range(a.ndim))
    perm[-1], perm[-2] = perm[-2], perm[-1]
    return transpose(a, perm)


# ----------------------------------------------------------------------------- #
# matmul (dot_general)
# ----------------------------------------------------------------------------- #
def _dot_general(a, b, lbatch, rbatch, lcon, rcon, out_shape):
    Ta, Tb, To = ttype(a.shape, a.dtype), ttype(b.shape, a.dtype), ttype(out_shape, a.dtype)
    dn = (f"#stablehlo.dot<lhs_batching_dimensions = {_arr(lbatch)}, "
          f"rhs_batching_dimensions = {_arr(rbatch)}, "
          f"lhs_contracting_dimensions = {_arr(lcon)}, "
          f"rhs_contracting_dimensions = {_arr(rcon)}>")
    m = (f'module {{\n  func.func public @main(%a: {Ta}, %b: {Tb}) -> {To} {{\n'
         f'    %r = "stablehlo.dot_general"(%a, %b) {{dot_dimension_numbers = {dn}, '
         f'precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]}} : ({Ta}, {Tb}) -> {To}\n'
         f"    return %r : {To}\n  }}\n}}")
    return _apply(m, [a, b])


def mm(a, b):
    """2-D matmul [m,k] @ [k,n] -> [m,n]."""
    m_, k = a.shape
    k2, n = b.shape
    assert k == k2, (a.shape, b.shape)
    out = _dot_general(a, b, [], [], [1], [0], (m_, n))

    def bw():
        _acc(a, mm(out.grad, transpose(b, [1, 0])))
        _acc(b, mm(transpose(a, [1, 0]), out.grad))
    return _record(out, [a, b], bw)


def bmm(a, b):
    """Batched matmul over all leading dims: [...,m,k] @ [...,k,n] -> [...,m,n]."""
    r = a.ndim
    assert b.ndim == r and r >= 3
    batch = list(range(r - 2))
    m_, k = a.shape[-2], a.shape[-1]
    n = b.shape[-1]
    out_shape = tuple(a.shape[:-2]) + (m_, n)
    out = _dot_general(a, b, batch, batch, [r - 1], [r - 2], out_shape)

    def bw():
        _acc(a, bmm(out.grad, transpose_last2(b)))
        _acc(b, bmm(transpose_last2(a), out.grad))
    return _record(out, [a, b], bw)


def matmul(a, b):
    if a.ndim == 2 and b.ndim == 2:
        return mm(a, b)
    return bmm(a, b)


def linear(x, W):
    """x[...,in] @ W[in,out] -> [...,out], folding leading dims for a 2-D matmul."""
    in_dim = W.shape[0]
    lead = x.shape[:-1]
    x2 = reshape(x, (int(np.prod(lead)) if lead else 1, in_dim))
    y2 = mm(x2, W)
    return reshape(y2, tuple(lead) + (W.shape[1],))
