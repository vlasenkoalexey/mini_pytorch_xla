"""Numeric checks for the buffer-level StableHLO ops (forward only) vs numpy.

Autograd is no longer ours (PyTorch's now — see test_backend.py); this just checks
that each StableHLO lowering computes the right thing on the TPU.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from mini_pytorch_xla import ops

rng = np.random.default_rng(0)


def close(buf, ref, tol=3e-2, msg=""):
    got = buf.to_numpy()
    assert got.shape == ref.shape, f"{msg}: shape {got.shape} vs {ref.shape}"
    assert np.allclose(got, ref, atol=tol, rtol=tol), f"{msg}: max {np.abs(got-ref).max()}"


a = rng.standard_normal((3, 4)).astype(np.float32)
b = rng.standard_normal((3, 4)).astype(np.float32)
ba, bb = ops.from_np(a), ops.from_np(b)
close(ops.binary("add", ba, bb), a + b, msg="add")
close(ops.binary("multiply", ba, bb), a * b, msg="mul")
close(ops.binary("subtract", ba, bb), a - b, msg="sub")
close(ops.binary("divide", ba, bb), a / b, msg="div")
close(ops.unary("exponential", ba), np.exp(a), msg="exp")
close(ops.unary("tanh", ba), np.tanh(a), msg="tanh")

# broadcast
c = rng.standard_normal((4,)).astype(np.float32)
close(ops.binary("add", ba, ops.from_np(c)), a + c, msg="bcast")

# matmul (TPU MXU bf16 -> loose tol)
m1 = rng.standard_normal((4, 5)).astype(np.float32)
m2 = rng.standard_normal((5, 6)).astype(np.float32)
close(ops.mm(ops.from_np(m1), ops.from_np(m2)), m1 @ m2, msg="mm")

b1 = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
b2 = rng.standard_normal((2, 3, 5, 6)).astype(np.float32)
close(ops.bmm(ops.from_np(b1), ops.from_np(b2)), b1 @ b2, msg="bmm")

# reductions / shape
close(ops.reduce_sum(ba, [1]), a.sum(1), msg="sum")
close(ops.transpose(ops.from_np(b1), [0, 1, 3, 2]), b1.transpose(0, 1, 3, 2), msg="transpose")
close(ops.reshape(ops.from_np(b1), (2, 3, 20)), b1.reshape(2, 3, 20), msg="reshape")

# gather (embedding)
table = rng.standard_normal((7, 3)).astype(np.float32)
idx = np.array([[0, 3], [6, 1]], dtype=np.int64)
close(ops.gather_rows(ops.from_np(table), idx), table[idx], msg="gather_rows")

print("ops forward numerics: ALL PASS")
