"""Buffer-level StableHLO op lowerings (no autograd here).

Each function takes/returns `pjrt.Buffer`s and emits one StableHLO module that runs
on the TPU. These are the primitives the aten dispatcher (backend.py) lowers to.
Autograd is PyTorch's job now — this layer is pure forward compute, the analogue of
PyTorch/XLA's lowering_context node lowerings.
"""

from __future__ import annotations

import numpy as np
from . import pjrt
from .hlo import ttype, run


def _arr(xs):
    return "[" + ", ".join(str(int(x)) for x in xs) + "]"


def _f(v):
    return f"{float(v):.8e}"


def _run1(module, inputs):
    (out,) = run(module, inputs, n_out=1)
    return out


def const(value, shape, dtype=np.float32) -> pjrt.Buffer:
    return pjrt.client().from_host(np.full(shape, value, dtype=dtype))


def from_np(arr) -> pjrt.Buffer:
    return pjrt.client().from_host(np.ascontiguousarray(arr))


# ---- broadcasting ----------------------------------------------------------- #
def bshape(sa, sb):
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


def broadcast_to(a: pjrt.Buffer, shape) -> pjrt.Buffer:
    shape = tuple(int(s) for s in shape)
    if a.shape == shape:
        return a
    Ta, To = ttype(a.shape, a.dtype), ttype(shape, a.dtype)
    m = (f"module {{\n  func.func public @main(%a: {Ta}) -> {To} {{\n"
         f"    %r = stablehlo.broadcast_in_dim %a, dims = {_arr(_bdims(a.shape, shape))} : ({Ta}) -> {To}\n"
         f"    return %r : {To}\n  }}\n}}")
    return _run1(m, [a])


# ---- elementwise ------------------------------------------------------------ #
def binary(opname, a: pjrt.Buffer, b: pjrt.Buffer) -> pjrt.Buffer:
    out = bshape(a.shape, b.shape)
    Ta, Tb, To = ttype(a.shape, a.dtype), ttype(b.shape, a.dtype), ttype(out, a.dtype)
    lines = ["module {", f"  func.func public @main(%a: {Ta}, %b: {Tb}) -> {To} {{"]
    av = "%a"
    if a.shape != out:
        lines.append(f"    %ab = stablehlo.broadcast_in_dim %a, dims = {_arr(_bdims(a.shape, out))} : ({Ta}) -> {To}")
        av = "%ab"
    bv = "%b"
    if b.shape != out:
        lines.append(f"    %bb = stablehlo.broadcast_in_dim %b, dims = {_arr(_bdims(b.shape, out))} : ({Tb}) -> {To}")
        bv = "%bb"
    lines += [f"    %r = stablehlo.{opname} {av}, {bv} : {To}", f"    return %r : {To}", "  }", "}"]
    return _run1("\n".join(lines), [a, b])


def unary(opname, a: pjrt.Buffer) -> pjrt.Buffer:
    T = ttype(a.shape, a.dtype)
    m = (f"module {{\n  func.func public @main(%a: {T}) -> {T} {{\n"
         f"    %r = stablehlo.{opname} %a : {T}\n    return %r : {T}\n  }}\n}}")
    return _run1(m, [a])


def erf(a: pjrt.Buffer) -> pjrt.Buffer:
    # chlo.erf legalizes to StableHLO during XLA compile -> real erf on the TPU
    # (this is what lets gelu's decomposition run on-device).
    T = ttype(a.shape, a.dtype)
    m = (f'module {{\n  func.func public @main(%a: {T}) -> {T} {{\n'
         f'    %r = "chlo.erf"(%a) : ({T}) -> {T}\n    return %r : {T}\n  }}\n}}')
    return _run1(m, [a])


def compare(direction, a: pjrt.Buffer, b: pjrt.Buffer) -> pjrt.Buffer:
    """Elementwise compare -> bool buffer. direction in {GE,GT,LE,LT,EQ,NE}."""
    out = bshape(a.shape, b.shape)
    Ta, Tb = ttype(a.shape, a.dtype), ttype(b.shape, a.dtype)
    To = ttype(out, a.dtype)
    Tbool = ttype(out, np.bool_)
    lines = ["module {", f"  func.func public @main(%a: {Ta}, %b: {Tb}) -> {Tbool} {{"]
    av, bv = "%a", "%b"
    if a.shape != out:
        lines.append(f"    %ab = stablehlo.broadcast_in_dim %a, dims = {_arr(_bdims(a.shape, out))} : ({Ta}) -> {To}"); av = "%ab"
    if b.shape != out:
        lines.append(f"    %bb = stablehlo.broadcast_in_dim %b, dims = {_arr(_bdims(b.shape, out))} : ({Tb}) -> {To}"); bv = "%bb"
    lines += [f"    %r = stablehlo.compare {direction}, {av}, {bv} : ({To}, {To}) -> {Tbool}",
              f"    return %r : {Tbool}", "  }", "}"]
    return _run1("\n".join(lines), [a, b])


def select(pred: pjrt.Buffer, a: pjrt.Buffer, b: pjrt.Buffer) -> pjrt.Buffer:
    T = ttype(a.shape, a.dtype)
    Tb = ttype(pred.shape, np.bool_)
    m = (f"module {{\n  func.func public @main(%p: {Tb}, %a: {T}, %b: {T}) -> {T} {{\n"
         f"    %r = stablehlo.select %p, %a, %b : ({Tb}, {T}, {T}) -> {T}\n"
         f"    return %r : {T}\n  }}\n}}")
    return _run1(m, [pred, a, b])


# ---- shape ------------------------------------------------------------------ #
def reshape(a: pjrt.Buffer, shape) -> pjrt.Buffer:
    shape = tuple(int(s) for s in shape)
    if a.shape == shape:
        return a
    Ta, To = ttype(a.shape, a.dtype), ttype(shape, a.dtype)
    m = (f"module {{\n  func.func public @main(%a: {Ta}) -> {To} {{\n"
         f"    %r = stablehlo.reshape %a : ({Ta}) -> {To}\n    return %r : {To}\n  }}\n}}")
    return _run1(m, [a])


def transpose(a: pjrt.Buffer, perm) -> pjrt.Buffer:
    perm = [int(p) for p in perm]
    out = tuple(a.shape[p] for p in perm)
    Ta, To = ttype(a.shape, a.dtype), ttype(out, a.dtype)
    m = (f"module {{\n  func.func public @main(%a: {Ta}) -> {To} {{\n"
         f"    %r = stablehlo.transpose %a, dims = {_arr(perm)} : ({Ta}) -> {To}\n"
         f"    return %r : {To}\n  }}\n}}")
    return _run1(m, [a])


def slice_dim(a: pjrt.Buffer, dim, start, limit, stride=1) -> pjrt.Buffer:
    starts = [0] * a.ndim if hasattr(a, "ndim") else [0] * len(a.shape)
    rank = len(a.shape)
    starts = [0] * rank; limits = list(a.shape); strides = [1] * rank
    starts[dim] = start; limits[dim] = limit; strides[dim] = stride
    out = tuple((limits[i] - starts[i] + strides[i] - 1) // strides[i] for i in range(rank))
    Ta, To = ttype(a.shape, a.dtype), ttype(out, a.dtype)
    m = (f"module {{\n  func.func public @main(%a: {Ta}) -> {To} {{\n"
         f"    %r = stablehlo.slice %a [{', '.join(f'{starts[i]}:{limits[i]}:{strides[i]}' for i in range(rank))}] : ({Ta}) -> {To}\n"
         f"    return %r : {To}\n  }}\n}}")
    return _run1(m, [a])


# ---- matmul ----------------------------------------------------------------- #
def _dot_general(a, b, lb, rb, lc, rc, out_shape) -> pjrt.Buffer:
    Ta, Tb, To = ttype(a.shape, a.dtype), ttype(b.shape, a.dtype), ttype(out_shape, a.dtype)
    dn = (f"#stablehlo.dot<lhs_batching_dimensions = {_arr(lb)}, "
          f"rhs_batching_dimensions = {_arr(rb)}, "
          f"lhs_contracting_dimensions = {_arr(lc)}, "
          f"rhs_contracting_dimensions = {_arr(rc)}>")
    m = (f'module {{\n  func.func public @main(%a: {Ta}, %b: {Tb}) -> {To} {{\n'
         f'    %r = "stablehlo.dot_general"(%a, %b) {{dot_dimension_numbers = {dn}, '
         f'precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]}} : ({Ta}, {Tb}) -> {To}\n'
         f"    return %r : {To}\n  }}\n}}")
    return _run1(m, [a, b])


def mm(a, b):
    return _dot_general(a, b, [], [], [1], [0], (a.shape[0], b.shape[1]))


def bmm(a, b):
    r = len(a.shape)
    batch = list(range(r - 2))
    return _dot_general(a, b, batch, batch, [r - 1], [r - 2],
                        tuple(a.shape[:-2]) + (a.shape[-2], b.shape[-1]))


# ---- reductions ------------------------------------------------------------- #
def reduce(opname, init, a: pjrt.Buffer, axes) -> pjrt.Buffer:
    axes = sorted(int(x) for x in axes)
    out_shape = tuple(s for i, s in enumerate(a.shape) if i not in axes)
    Tin, Tsc, Tout = ttype(a.shape, a.dtype), ttype((), a.dtype), ttype(out_shape, a.dtype)
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
    return _run1(m, [a])


def reduce_sum(a, axes):
    return reduce("add", 0.0, a, axes)


def reduce_max(a, axes):
    return reduce("maximum", -3.0e38, a, axes)


def gather_rows(table: pjrt.Buffer, idx: np.ndarray) -> pjrt.Buffer:
    """Embedding-style row gather: table[V,D], idx int array [...] -> [...,D].
    Done as one-hot @ table so we stay within implemented primitives."""
    V, D = table.shape
    flat = idx.reshape(-1).astype(np.int64)
    oh = np.zeros((flat.size, V), np.float32)
    oh[np.arange(flat.size), flat] = 1.0
    out = mm(from_np(oh), table)                 # [N, D]
    return reshape(out, tuple(idx.shape) + (D,))
