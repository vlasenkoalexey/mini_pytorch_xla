"""mini-pytorch-xla: a real PyTorch TPU/XLA backend in pure Python.

PyTorch owns the program, autograd, and optimizer. This package only provides the
device and the op lowering — a `torch.Tensor` subclass (`XLATensor`) whose
`__torch_dispatch__` intercepts each aten op and lowers it to StableHLO that runs
eagerly on the TPU via libtpu's PJRT C API (ctypes). No torch_xla, no jax.

    aten op  ──__torch_dispatch__──►  ops.py (StableHLO)  ──►  pjrt.py (libtpu/TPU)

This is the Python-level analogue of how PyTorch/XLA intercepts ops with a C++
dispatch key. Usage:

    import torch, torch.nn as nn
    from mini_pytorch_xla import backend as xb
    model = nn.Linear(4, 4)
    xb.to_xla_(model) if hasattr(xb, "to_xla_") else None   # move params to TPU
    y = model(xb.to_xla(torch.randn(8, 4)))                 # runs on the TPU
"""
from . import pjrt, hlo, ops, backend, profiler
from .backend import XLATensor, to_xla, to_cpu, to_xla_

__all__ = ["pjrt", "hlo", "ops", "backend", "profiler",
           "XLATensor", "to_xla", "to_cpu", "to_xla_"]
