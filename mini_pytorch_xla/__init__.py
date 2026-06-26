"""mini-pytorch-xla: a standalone eager TPU/XLA backend in pure Python.

Pipeline mirrors PyTorch/XLA's idea — intercept high-level ops, lower each to an
XLA program, run on the TPU — with zero dependency on torch_xla or jax:

    tensor.py  high-level op  ->  hlo.py  StableHLO module  ->  pjrt.py  libtpu/TPU

`sync()` exists for API parity with train_xla.py; execution is already eager
(every op runs on device immediately), so it is a no-op.
"""
from . import pjrt, hlo, tensor, nn
from .tensor import Tensor, from_numpy, no_grad

__all__ = ["pjrt", "hlo", "tensor", "nn", "Tensor", "from_numpy", "no_grad", "sync", "device"]


def device() -> str:
    return "tpu:0"


def sync():
    """No-op: this backend is eager (each op already executed on the TPU)."""
    return None
