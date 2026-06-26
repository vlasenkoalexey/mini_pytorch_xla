"""Prove real torch.autograd runs through __torch_dispatch__ onto the TPU."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from mini_pytorch_xla import backend as xb


def close(a, b, tol=2e-2, msg=""):
    assert torch.allclose(a, b, atol=tol, rtol=tol), f"{msg}: max {(a-b).abs().max()}"


torch.manual_seed(0)

# ---- autograd through dispatch: d/dA sum(A@B) = ones@B^T, d/dB = A^T@ones ---- #
A = torch.randn(3, 4)
B = torch.randn(4, 5)
a = xb.to_xla(A); a.requires_grad_(True)
b = xb.to_xla(B); b.requires_grad_(True)
loss = (a @ b).sum()
loss.backward()

Ar = A.clone().requires_grad_(True)
Br = B.clone().requires_grad_(True)
(Ar @ Br).sum().backward()

print("loss   :", xb.to_cpu(loss.detach()).item(), "vs", (Ar @ Br).sum().item())
close(xb.to_cpu(a.grad), Ar.grad, msg="dL/dA")
close(xb.to_cpu(b.grad), Br.grad, msg="dL/dB")
print("grad A/B through __torch_dispatch__ on TPU: MATCH cpu autograd")

# ---- a tiny MLP (real nn.Linear + activation) forward + backward ------------ #
import torch.nn as nn
import torch.nn.functional as F

lin1 = nn.Linear(4, 8)
lin2 = nn.Linear(8, 2)


def to_xla_module(m):
    for n, p in list(m.named_parameters(recurse=False)):
        xp = xb.to_xla(p.detach()); xp.requires_grad_(True)
        setattr(m, n, torch.nn.Parameter(xp, requires_grad=True))
    return m


# reference on cpu
x = torch.randn(6, 4)
ref = F.mse_loss(lin2(F.relu(lin1(x))), torch.zeros(6, 2))
ref.backward()
ref_g = lin1.weight.grad.clone()

# same weights on tpu
lin1x = nn.Linear(4, 8); lin2x = nn.Linear(8, 2)
lin1x.load_state_dict(lin1.state_dict()); lin2x.load_state_dict(lin2.state_dict())
to_xla_module(lin1x); to_xla_module(lin2x)
xx = xb.to_xla(x)
out = lin2x(F.relu(lin1x(xx)))
loss2 = F.mse_loss(out, xb.to_xla(torch.zeros(6, 2)))
loss2.backward()

print("\nMLP loss:", xb.to_cpu(loss2.detach()).item(), "vs cpu", ref.item())
close(xb.to_cpu(lin1x.weight.grad), ref_g, tol=3e-2, msg="MLP dL/dW1")
print("MLP (nn.Linear+relu+mse) forward+backward on TPU: MATCH cpu autograd")
print("\nALL BACKEND TESTS PASS")
