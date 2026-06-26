"""Numeric + gradient checks for the StableHLO-lowered primitives, vs numpy."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import mini_pytorch_xla.tensor as T

rng = np.random.default_rng(0)


def close(a, b, tol=1e-3, msg=""):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"{msg}: shape {a.shape} vs {b.shape}"
    assert np.allclose(a, b, atol=tol, rtol=tol), f"{msg}: max diff {np.abs(a-b).max()}"


# ---- forward correctness ---------------------------------------------------- #
a = rng.standard_normal((2, 3)).astype(np.float32)
b = rng.standard_normal((2, 3)).astype(np.float32)
ta, tb = T.from_numpy(a), T.from_numpy(b)
close((ta + tb).numpy(), a + b, msg="add")
close((ta * tb).numpy(), a * b, msg="mul")
close((ta - tb).numpy(), a - b, msg="sub")
close((ta / tb).numpy(), a / b, msg="div")
close(T.exp(ta).numpy(), np.exp(a), msg="exp")
close(T.tanh(ta).numpy(), np.tanh(a), msg="tanh")

# broadcasting
c = rng.standard_normal((3,)).astype(np.float32)
close((ta + T.from_numpy(c)).numpy(), a + c, msg="bcast add")

# matmul
m1 = rng.standard_normal((4, 5)).astype(np.float32)
m2 = rng.standard_normal((5, 6)).astype(np.float32)
# matmul runs on the TPU MXU in bf16 at precision DEFAULT (like real XLA) -> looser tol
close(T.mm(T.from_numpy(m1), T.from_numpy(m2)).numpy(), m1 @ m2, tol=3e-2, msg="mm")

# batched matmul
b1 = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
b2 = rng.standard_normal((2, 3, 5, 6)).astype(np.float32)
close(T.bmm(T.from_numpy(b1), T.from_numpy(b2)).numpy(), b1 @ b2, tol=3e-2, msg="bmm")

# reduce
close(T.reduce_sum(ta, [1]).numpy(), a.sum(1), msg="sum")
close(T.reduce_sum(ta, [1], keepdim=True).numpy(), a.sum(1, keepdims=True), msg="sum kd")

# transpose
close(T.transpose(T.from_numpy(b1), [0, 1, 3, 2]).numpy(), b1.transpose(0, 1, 3, 2), msg="transpose")
print("forward: OK")


# ---- gradient check via finite differences ---------------------------------- #
def grad_check(f_t, f_np, *shapes, tol=2e-2):
    inps = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    ts = [T.from_numpy(x, requires_grad=True) for x in inps]
    out = f_t(*ts)
    loss = T.reduce_sum(out, list(range(out.ndim)))  # scalar
    loss.backward()
    for i, t in enumerate(ts):
        ana = t.grad.numpy()
        num = np.zeros_like(inps[i])
        eps = 1e-3
        flat = inps[i].reshape(-1)
        for j in range(flat.size):
            d = np.zeros_like(flat); d[j] = eps
            p = [x.copy() for x in inps]; p[i] = (flat + d).reshape(inps[i].shape)
            m = [x.copy() for x in inps]; m[i] = (flat - d).reshape(inps[i].shape)
            num.reshape(-1)[j] = (f_np(*p).sum() - f_np(*m).sum()) / (2 * eps)
        close(ana, num, tol=tol, msg=f"grad arg{i} of {f_t.__name__}")


grad_check(lambda x, y: x * y, lambda x, y: x * y, (3, 4), (3, 4))
grad_check(lambda x, y: x / y, lambda x, y: x / y, (3, 4), (3, 4))
grad_check(lambda x, y: T.mm(x, y), lambda x, y: x @ y, (3, 4), (4, 5))
grad_check(lambda x: T.tanh(x), lambda x: np.tanh(x), (3, 4))
grad_check(lambda x: T.exp(x), lambda x: np.exp(x), (3, 4))
grad_check(lambda x: T.reduce_sum(x, [0]), lambda x: x.sum(0), (3, 4))
grad_check(lambda x, y: x + y, lambda x, y: x + y, (3, 4), (4,))  # broadcast grad
print("grad-check: OK")
print("\nALL TESTS PASS")
