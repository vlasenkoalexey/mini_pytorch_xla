"""StableHLO emission + an eager compile/execute cache.

Each primitive op in tensor.py renders a tiny StableHLO module whose `@main`
takes the op's input buffers as arguments and returns its result(s). We compile
that module once (keyed by its text, which encodes op + shapes + dtypes) and
re-run the cached TPU executable on every later call — so a training step that
repeats the same shapes pays compilation only on iteration 0.

This is "eager op lowering": one HLO module per PyTorch-level op, no graph fusion.
"""

from __future__ import annotations

import numpy as np
from . import pjrt

_NP_TO_MLIR = {
    np.dtype("float32"): "f32",
    np.dtype("int64"): "i64",
    np.dtype("int32"): "i32",
    np.dtype("bool"): "i1",
}


def mlir_dtype(dt) -> str:
    return _NP_TO_MLIR[np.dtype(dt)]


def ttype(shape, dt) -> str:
    """MLIR tensor type, e.g. (2,3),f32 -> 'tensor<2x3xf32>'; () -> 'tensor<f32>'."""
    d = mlir_dtype(dt)
    if len(shape) == 0:
        return f"tensor<{d}>"
    return "tensor<" + "x".join(str(int(s)) for s in shape) + f"x{d}>"


_EXE_CACHE: dict[str, pjrt.Executable] = {}

# optional profiling hook: a callable(op_kind: str, seconds: float, compiled: bool)
_TIMER = None


def set_timer(fn):
    global _TIMER
    _TIMER = fn


def _op_kind(text: str) -> str:
    import re
    hits = re.findall(r"stablehlo\.(\w+)", text)
    # last stablehlo op is the result-producing one (broadcasts precede it)
    return hits[-1] if hits else "?"


def run(module_text: str, inputs: list["pjrt.Buffer"], n_out: int = 1):
    """Compile (cached) and execute a StableHLO module on the TPU."""
    exe = _EXE_CACHE.get(module_text)
    compiled = exe is None
    if compiled:
        exe = pjrt.client().compile(module_text, num_outputs=n_out)
        _EXE_CACHE[module_text] = exe
    if _TIMER is None:
        return exe.execute(inputs)
    import time
    t0 = time.perf_counter()
    out = exe.execute(inputs)          # pjrt awaits device completion -> real exec time
    _TIMER(_op_kind(module_text), time.perf_counter() - t0, compiled)
    return out
