"""A real PyTorch backend: a `torch.Tensor` subclass that lowers aten ops to the TPU.

This is the honest PyTorch/XLA model. `XLATensor` is a wrapper subclass holding a
PJRT device buffer; `__torch_dispatch__` intercepts every aten op and lowers it to
StableHLO (ops.py) on the TPU. PyTorch's autograd runs *above* __torch_dispatch__,
so we implement only forward ops — backward is PyTorch's, and its backward formulas
re-dispatch through here, so the backward pass also runs on the TPU. Composite ops
we don't implement are expanded by PyTorch's core-aten decompositions into the
primitives we do. (Real torch_xla does the same interception in C++ via a dispatch
key; __torch_dispatch__ is the Python-level equivalent.)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils._pytree import tree_map

from . import pjrt, ops

aten = torch.ops.aten

_NP_TO_TORCH = {np.dtype("float32"): torch.float32, np.dtype("int32"): torch.int32,
                np.dtype("int64"): torch.int64, np.dtype("bool"): torch.bool}
_TORCH_TO_NP = {torch.float32: np.float32, torch.float64: np.float32,
                torch.int64: np.int64, torch.int32: np.int32, torch.bool: np.bool_}


def _torch_dtype(npdt):
    return _NP_TO_TORCH[np.dtype(npdt)]


class XLATensor(torch.Tensor):
    @staticmethod
    def __new__(cls, buf, requires_grad=False):
        return torch.Tensor._make_wrapper_subclass(
            cls, tuple(buf.shape), dtype=_torch_dtype(buf.dtype),
            device="cpu", requires_grad=requires_grad)

    def __init__(self, buf, requires_grad=False):
        self._buf = buf

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return _dispatch(func, args, kwargs or {})

    def __repr__(self):
        return f"XLATensor(shape={tuple(self.shape)}, dtype={self.dtype}, device=tpu)"


# ---- host <-> device API ---------------------------------------------------- #
def to_xla(t: torch.Tensor) -> XLATensor:
    arr = t.detach().contiguous().cpu().numpy()
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    return XLATensor(pjrt.client().from_host(arr), requires_grad=t.requires_grad)


def to_cpu(x):
    if isinstance(x, XLATensor):
        return torch.from_numpy(x._buf.to_numpy())
    return x


def to_xla_(module: "torch.nn.Module"):
    """In-place: replace every parameter of `module` with an XLATensor leaf on the TPU.
    The analogue of `model.to(xla_device)` in PyTorch/XLA."""
    for m in module.modules():
        for name, p in list(m.named_parameters(recurse=False)):
            xp = to_xla(p.detach())
            xp.requires_grad_(True)
            setattr(m, name, torch.nn.Parameter(xp, requires_grad=True))
    return module


def _wrap(buf) -> XLATensor:
    return XLATensor(buf)


def _buf(x, ref_dtype=np.float32):
    """Coerce an op argument to a device Buffer."""
    if isinstance(x, XLATensor):
        return x._buf
    if isinstance(x, torch.Tensor):
        return pjrt.client().from_host(x.detach().contiguous().cpu().numpy())
    if isinstance(x, (int, float, bool)):
        return ops.const(x, (), ref_dtype)
    raise TypeError(f"cannot move {type(x)} to device")


# ---- dispatch registry ------------------------------------------------------ #
HANDLERS: dict = {}


def impl(*ops_overloads):
    def deco(fn):
        for o in ops_overloads:
            HANDLERS[o] = fn
        return fn
    return deco


try:
    from torch._decomp import core_aten_decompositions, get_decompositions
    _DECOMP = dict(core_aten_decompositions())
    # Pull in forward-composite decompositions so softmax / layernorm / log_softmax
    # expand to the primitives we implement (their backwards are already in core).
    _DECOMP.update(get_decompositions([
        aten._softmax.default, aten._log_softmax.default,
        aten.native_layer_norm.default,
    ]))
except Exception:
    _DECOMP = {}


def _dispatch(func, args, kwargs):
    h = HANDLERS.get(func) or HANDLERS.get(getattr(func, "overloadpacket", None))
    if h is not None:
        return h(*args, **kwargs)
    if func in _DECOMP:
        return _DECOMP[func](*args, **kwargs)
    raise NotImplementedError(f"mini-xla: no lowering for {func}  (args dtypes "
                              f"{[getattr(a,'dtype',type(a).__name__) for a in args]})")


# =========================================================================== #
# primitive lowerings
# =========================================================================== #
def _kd(buf, reduced, dims):
    """Reshape a reduced buffer back to keepdim shape (1s at reduced dims)."""
    shape = tuple(1 if i in dims else s for i, s in enumerate(buf.shape))
    return ops.reshape(reduced, shape)


@impl(aten.var_mean.correction)
def _var_mean(self, dim=None, *, correction=1, keepdim=False):
    b = _buf(self)
    dims = [d % len(b.shape) for d in (dim if dim is not None else range(len(b.shape)))]
    n = 1
    for d in dims:
        n *= b.shape[d]
    mean_kd = _kd(b, ops.reduce_sum(b, dims), dims)
    mean_kd = ops.binary("divide", mean_kd, ops.const(n, (), np.float32))
    xc = ops.binary("subtract", b, mean_kd)
    var_kd = _kd(b, ops.reduce_sum(ops.binary("multiply", xc, xc), dims), dims)
    var_kd = ops.binary("divide", var_kd, ops.const(max(n - correction, 1), (), np.float32))
    if keepdim:
        return _wrap(var_kd), _wrap(mean_kd)
    sq = tuple(s for i, s in enumerate(b.shape) if i not in dims)
    return _wrap(ops.reshape(var_kd, sq)), _wrap(ops.reshape(mean_kd, sq))


def _norm_shape(size, numel=None):
    size = list(size)
    if -1 in size:
        known = 1
        for s in size:
            if s != -1:
                known *= s
        size[size.index(-1)] = numel // known
    return tuple(size)


@impl(aten.mm.default)
def _mm(a, b):
    return _wrap(ops.mm(_buf(a), _buf(b)))


@impl(aten.bmm.default)
def _bmm(a, b):
    return _wrap(ops.bmm(_buf(a), _buf(b)))


@impl(aten.add.Tensor, aten.add.Scalar)
def _add(a, b, *, alpha=1):
    bb = _buf(b)
    if alpha != 1:
        bb = ops.binary("multiply", bb, ops.const(alpha, (), np.float32))
    return _wrap(ops.binary("add", _buf(a), bb))


@impl(aten.sub.Tensor, aten.sub.Scalar)
def _sub(a, b, *, alpha=1):
    bb = _buf(b)
    if alpha != 1:
        bb = ops.binary("multiply", bb, ops.const(alpha, (), np.float32))
    return _wrap(ops.binary("subtract", _buf(a), bb))


@impl(aten.mul.Tensor, aten.mul.Scalar)
def _mul(a, b):
    return _wrap(ops.binary("multiply", _buf(a), _buf(b)))


@impl(aten.div.Tensor, aten.div.Scalar)
def _div(a, b):
    return _wrap(ops.binary("divide", _buf(a), _buf(b)))


@impl(aten.neg.default)
def _neg(a):
    return _wrap(ops.unary("negate", _buf(a)))


@impl(aten.exp.default)
def _exp(a):
    return _wrap(ops.unary("exponential", _buf(a)))


@impl(aten.log.default)
def _log(a):
    return _wrap(ops.unary("log", _buf(a)))


@impl(aten.sqrt.default)
def _sqrt(a):
    return _wrap(ops.unary("sqrt", _buf(a)))


@impl(aten.rsqrt.default)
def _rsqrt(a):
    return _wrap(ops.unary("rsqrt", _buf(a)))


@impl(aten.tanh.default)
def _tanh(a):
    return _wrap(ops.unary("tanh", _buf(a)))


@impl(aten.reciprocal.default)
def _recip(a):
    return _wrap(ops.binary("divide", ops.const(1.0, (), np.float32), _buf(a)))


@impl(aten.pow.Tensor_Scalar)
def _pow(a, e):
    b = _buf(a)
    if float(e) == 2.0:
        return _wrap(ops.binary("multiply", b, b))
    if float(e) == 3.0:
        return _wrap(ops.binary("multiply", ops.binary("multiply", b, b), b))
    # general: exp(e * log(a))  (a>0 assumed)
    return _wrap(ops.unary("exponential",
                           ops.binary("multiply", ops.const(float(e), (), np.float32),
                                      ops.unary("log", b))))


@impl(aten.sum.default)
def _sum_all(a, *, dtype=None):
    b = _buf(a)
    return _wrap(ops.reduce_sum(b, list(range(len(b.shape)))))


@impl(aten.sum.dim_IntList)
def _sum_dim(a, dim, keepdim=False, *, dtype=None):
    b = _buf(a)
    dims = [d % len(b.shape) for d in (dim if dim is not None else range(len(b.shape)))]
    out = ops.reduce_sum(b, dims)
    if keepdim:
        kd = tuple(1 if i in dims else s for i, s in enumerate(b.shape))
        out = ops.reshape(out, kd)
    return _wrap(out)


@impl(aten.amax.default)
def _amax(a, dim, keepdim=False):
    b = _buf(a)
    dims = [d % len(b.shape) for d in dim]
    out = ops.reduce_max(b, dims)
    if keepdim:
        kd = tuple(1 if i in dims else s for i, s in enumerate(b.shape))
        out = ops.reshape(out, kd)
    return _wrap(out)


@impl(aten.expand.default)
def _expand(a, size, *, implicit=False):
    b = _buf(a)
    size = list(size)
    size = [b.shape[i] if s == -1 else s for i, s in enumerate(size)]
    return _wrap(ops.broadcast_to(b, tuple(size)))


@impl(aten.view.default, aten._unsafe_view.default, aten.reshape.default)
def _view(a, size):
    b = _buf(a)
    numel = int(np.prod(b.shape)) if b.shape else 1
    return _wrap(ops.reshape(b, _norm_shape(size, numel)))


@impl(aten.t.default)
def _t(a):
    b = _buf(a)
    return _wrap(ops.transpose(b, [1, 0]) if len(b.shape) == 2 else b)


@impl(aten.transpose.int)
def _transpose(a, d0, d1):
    b = _buf(a)
    perm = list(range(len(b.shape)))
    perm[d0], perm[d1] = perm[d1] % len(b.shape), perm[d0] % len(b.shape)
    return _wrap(ops.transpose(b, perm))


@impl(aten.permute.default)
def _permute(a, dims):
    return _wrap(ops.transpose(_buf(a), [d % len(_buf(a).shape) for d in dims]))


@impl(aten.unsqueeze.default)
def _unsqueeze(a, dim):
    b = _buf(a)
    dim = dim % (len(b.shape) + 1)
    return _wrap(ops.reshape(b, b.shape[:dim] + (1,) + b.shape[dim:]))


@impl(aten.squeeze.dim, aten.squeeze.dims)
def _squeeze(a, dim):
    b = _buf(a)
    dims = [dim % len(b.shape)] if isinstance(dim, int) else [d % len(b.shape) for d in dim]
    shape = tuple(s for i, s in enumerate(b.shape) if not (i in dims and s == 1))
    return _wrap(ops.reshape(b, shape))


@impl(aten.where.self)
def _where(cond, a, b):
    return _wrap(ops.select(_buf(cond), _buf(a), _buf(b)))


@impl(aten.ge.Scalar, aten.ge.Tensor)
def _ge(a, b):
    return _wrap(ops.compare("GE", _buf(a), _buf(b)))


@impl(aten.le.Scalar, aten.le.Tensor)
def _le(a, b):
    return _wrap(ops.compare("LE", _buf(a), _buf(b)))


@impl(aten.gt.Scalar, aten.gt.Tensor)
def _gt(a, b):
    return _wrap(ops.compare("GT", _buf(a), _buf(b)))


@impl(aten.lt.Scalar, aten.lt.Tensor)
def _lt(a, b):
    return _wrap(ops.compare("LT", _buf(a), _buf(b)))


@impl(aten.ones_like.default)
def _ones_like(a, **kw):
    b = _buf(a)
    return _wrap(ops.const(1.0, b.shape, b.dtype))


@impl(aten.zeros_like.default)
def _zeros_like(a, **kw):
    b = _buf(a)
    return _wrap(ops.const(0.0, b.shape, b.dtype))


@impl(aten.detach.default, aten.alias.default, aten.clone.default, aten.lift_fresh.default)
def _identity(a, **kw):
    return _wrap(_buf(a))


@impl(aten._to_copy.default, aten.to.dtype)
def _to_copy(a, *args, **kw):
    return _wrap(_buf(a))   # dtype/device coercion is a no-op (everything is f32 on TPU)


# ---- fused / nn ops --------------------------------------------------------- #
@impl(aten.addmm.default)
def _addmm(bias, m1, m2, *, beta=1, alpha=1):
    r = ops.mm(_buf(m1), _buf(m2))
    if alpha != 1:
        r = ops.binary("multiply", r, ops.const(alpha, (), np.float32))
    bb = _buf(bias)
    if beta != 1:
        bb = ops.binary("multiply", bb, ops.const(beta, (), np.float32))
    return _wrap(ops.binary("add", r, bb))      # bias [out] broadcasts to [N, out]


@impl(aten.maximum.default)
def _maximum(a, b):
    return _wrap(ops.binary("maximum", _buf(a), _buf(b)))


@impl(aten.clamp_min.default)
def _clamp_min(a, mn):
    return _wrap(ops.binary("maximum", _buf(a), ops.const(float(mn), (), np.float32)))


@impl(aten.relu.default)
def _relu(a):
    return _wrap(ops.binary("maximum", _buf(a), ops.const(0.0, (), np.float32)))


@impl(aten.threshold_backward.default)
def _threshold_backward(grad, self, threshold):
    g = _buf(grad)
    mask = ops.compare("GT", _buf(self), ops.const(float(threshold), (), np.float32))
    return _wrap(ops.select(mask, g, ops.const(0.0, g.shape, np.float32)))


@impl(aten.mean.default)
def _mean_all(a, *, dtype=None):
    b = _buf(a)
    n = int(np.prod(b.shape)) or 1
    s = ops.reduce_sum(b, list(range(len(b.shape))))
    return _wrap(ops.binary("divide", s, ops.const(n, (), np.float32)))


@impl(aten.mean.dim)
def _mean_dim(a, dim, keepdim=False, *, dtype=None):
    b = _buf(a)
    dims = [d % len(b.shape) for d in dim]
    cnt = 1
    for d in dims:
        cnt *= b.shape[d]
    out = ops.binary("divide", ops.reduce_sum(b, dims), ops.const(cnt, (), np.float32))
    if keepdim:
        kd = tuple(1 if i in dims else s for i, s in enumerate(b.shape))
        out = ops.reshape(out, kd)
    return _wrap(out)


@impl(aten.mse_loss.default)
def _mse_loss(inp, tgt, reduction=1):
    d = ops.binary("subtract", _buf(inp), _buf(tgt))
    sq = ops.binary("multiply", d, d)
    if reduction == 0:           # none
        return _wrap(sq)
    s = ops.reduce_sum(sq, list(range(len(sq.shape))))
    if reduction == 2:           # sum
        return _wrap(s)
    n = int(np.prod(sq.shape)) or 1   # mean
    return _wrap(ops.binary("divide", s, ops.const(n, (), np.float32)))


@impl(aten.mse_loss_backward.default)
def _mse_loss_backward(grad, inp, tgt, reduction=1):
    d = ops.binary("subtract", _buf(inp), _buf(tgt))
    g = ops.binary("multiply", d, ops.const(2.0, (), np.float32))
    if reduction == 1:
        n = int(np.prod(d.shape)) or 1
        g = ops.binary("divide", g, ops.const(n, (), np.float32))
    return _wrap(ops.binary("multiply", g, _buf(grad)))


# ---- embedding (one-hot @ table, both directions; avoids gather/scatter) ---- #
def _idx_np(indices):
    t = indices._buf.to_numpy() if isinstance(indices, XLATensor) else indices.detach().cpu().numpy()
    return t.astype(np.int64)


@impl(aten.embedding.default)
def _embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    return _wrap(ops.gather_rows(_buf(weight), _idx_np(indices)))


@impl(aten.embedding_dense_backward.default)
def _embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq):
    g = _buf(grad)
    idx = _idx_np(indices).reshape(-1)
    oh = np.zeros((idx.size, num_weights), np.float32)
    oh[np.arange(idx.size), idx] = 1.0
    g2 = ops.reshape(g, (idx.size, g.shape[-1]))         # [N, D]
    return _wrap(ops.mm(ops.transpose(ops.from_np(oh), [1, 0]), g2))   # [V, D]


# ---- in-place ops (so a real torch.optim.AdamW step runs on device) --------- #
@impl(aten.add_.Tensor, aten.add_.Scalar)
def _iadd(self, other, *, alpha=1):
    o = _buf(other)
    if alpha != 1:
        o = ops.binary("multiply", o, ops.const(alpha, (), np.float32))
    self._buf = ops.binary("add", _buf(self), o)
    return self


@impl(aten.sub_.Tensor)
def _isub(self, other, *, alpha=1):
    o = _buf(other)
    if alpha != 1:
        o = ops.binary("multiply", o, ops.const(alpha, (), np.float32))
    self._buf = ops.binary("subtract", _buf(self), o)
    return self


@impl(aten.mul_.Tensor, aten.mul_.Scalar)
def _imul(self, other):
    self._buf = ops.binary("multiply", _buf(self), _buf(other))
    return self


@impl(aten.addcmul_.default)
def _iaddcmul(self, t1, t2, *, value=1):
    p = ops.binary("multiply", _buf(t1), _buf(t2))
    if value != 1:
        p = ops.binary("multiply", p, ops.const(value, (), np.float32))
    self._buf = ops.binary("add", _buf(self), p)
    return self


@impl(aten.addcdiv_.default)
def _iaddcdiv(self, t1, t2, *, value=1):
    q = ops.binary("divide", _buf(t1), _buf(t2))
    if value != 1:
        q = ops.binary("multiply", q, ops.const(value, (), np.float32))
    self._buf = ops.binary("add", _buf(self), q)
    return self


@impl(aten.lerp_.Scalar)
def _ilerp(self, end, weight):
    diff = ops.binary("subtract", _buf(end), _buf(self))
    self._buf = ops.binary("add", _buf(self),
                           ops.binary("multiply", diff, ops.const(float(weight), (), np.float32)))
    return self


@impl(aten.copy_.default)
def _icopy(self, src, non_blocking=False):
    self._buf = _buf(src)
    return self


@impl(aten.zero_.default)
def _izero(self):
    b = _buf(self)
    self._buf = ops.const(0.0, b.shape, b.dtype)
    return self
