# mini-pytorch-xla

A **real, minimal PyTorch TPU/XLA backend in pure Python**. PyTorch owns the
`Tensor`, autograd, and optimizer; this project only provides the **device and the
op lowering** — a `torch.Tensor` subclass whose `__torch_dispatch__` intercepts
every **aten** op and lowers it to **StableHLO** that runs eagerly on the TPU via
**libtpu's PJRT C API (ctypes)**. No `torch_xla`, no `jax`.

```
  a real torch.nn model  →  aten op  ──__torch_dispatch__──►  StableHLO  ──►  TPU
       (PyTorch autograd + torch.optim drive it)        (ops.py)        (pjrt.py / libtpu)
```

This is the Python-level analogue of how PyTorch/XLA works: torch_xla intercepts
ops with a C++ dispatch key and lowers them to XLA; here `__torch_dispatch__` is the
interception point and we emit StableHLO. Because autograd runs *above*
`__torch_dispatch__`, we implement **only forward ops** — the backward pass is
PyTorch's, and its backward formulas re-dispatch through us, so backward also runs
on the TPU. Composite ops we don't implement are expanded by PyTorch's **core-aten
decompositions** into the primitives we do.

## It runs unmodified `torch.nn`

```python
import torch, torch.nn as nn, torch.nn.functional as F
from mini_pytorch_xla import backend as xb

model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))
xb.to_xla_(model)                                  # params -> TPU  (like model.to(xla))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)

x = xb.to_xla(torch.randn(8, 4))
loss = F.mse_loss(model(x), xb.to_xla(torch.zeros(8, 2)))
loss.backward()        # PyTorch autograd -> aten backward ops -> StableHLO on TPU
opt.step()             # real torch.optim, in-place ops lowered to TPU
```

A full char-level transformer (real `nn.Embedding`/`nn.LayerNorm`/`nn.Linear`/
`F.softmax`, `torch.optim.AdamW`) trains on a single TPU:

```
$ python examples/train_torch.py --steps 200
real torch.nn transformer on mini-pytorch-xla | TPU | 0.29M params | vocab 65
PyTorch owns autograd+optimizer; every aten op lowered to StableHLO on the TPU.
Step     0 | Loss: 4.3403
Step    40 | Loss: 3.3720
Step   100 | Loss: 2.8965
Step   199 | Loss: 2.5965
```

## How it works

- **`pjrt.py`** — a ctypes client over `libtpu.so`'s PJRT C API: model the whole
  `PJRT_Api` function table, `from_host`/`compile`/`execute`/`to_numpy`, a
  hand-serialized `CompileOptionsProto`, and a forced row-major read-back layout.
- **`hlo.py`** — StableHLO type strings + a compile/execute cache (keyed by program
  text, so a repeated op compiles once).
- **`ops.py`** — buffer-level StableHLO lowerings (`add`, `dot_general`, `reduce`,
  `transpose`, `broadcast_in_dim`, `select`, one-hot `gather_rows`, …). No autograd.
- **`backend.py`** — `XLATensor(torch.Tensor)` wrapper subclass + `__torch_dispatch__`
  router + an aten→ops registry, plus `to_xla` / `to_cpu` / `to_xla_`. Forward
  primitives + a few fused ops (`addmm`, `embedding`) + in-place ops (so real
  `torch.optim.AdamW` runs); everything else goes through core-aten decompositions.
- **`profiler.py`** — device identity, HBM stats, on-device op timing, throughput.

## What maps to real PyTorch/XLA

| PyTorch/XLA (C++) | mini-pytorch-xla (Python) |
|---|---|
| dispatch-key interception of aten ops | `__torch_dispatch__` on a tensor subclass |
| lowering_context node → HLO | `ops.py` aten handler → StableHLO |
| `torch_xla/csrc/runtime` PjRt client | `pjrt.py` (ctypes over libtpu) |
| `model.to(xm.xla_device())` | `xb.to_xla_(model)` |
| autograd & `torch.optim` (PyTorch's) | autograd & `torch.optim` (PyTorch's — unchanged) |

The one deliberate simplification: **eager, no graph fusion** (one StableHLO program
per aten op, cached). Real PyTorch/XLA traces a whole step into one HLO graph and
lets XLA fuse it; we keep every op's lowering visible and self-contained.

## Is the backend "registered"? (the pure-Python limit)

Two ways to add a backend to PyTorch:

1. **A registered device** (a `torch.device('tpu')` via the PrivateUse1 key, like
   torch_xla / torch_npu) — so `cpu_tensor.to('tpu')` and `tensor.device.type=='tpu'`
   work. This requires a **C++ `DeviceGuardImpl` + allocator** registered for the
   device. We tried it (`rename_privateuse1_backend("tpu")` + the subclass on the
   `tpu` device): tensors then report `device='tpu:0'` and forward ops work — but the
   **autograd backward pass crashes** (`PyTorch is not linked with support for tpu
   devices`), because the engine instantiates a C++ DeviceGuard for the tensor's
   device and PrivateUse1 has none. That C++ shim is the one piece a pure-Python
   project cannot provide.

2. **A tensor-subclass + `__torch_dispatch__`** (what this does) — the *pure-Python*
   registration mechanism, the same one functorch / quantization / tracing modes use.
   It intercepts every aten op (forward and autograd-emitted backward) and works with
   full autograd + `torch.optim`. The tensor reports `device='cpu'`, so entry is
   `to_xla(t)` / `to_xla_(model)` rather than `t.to('tpu')`.

So: the **op interception is genuinely registered** with PyTorch's dispatcher; the
**device name** is the only thing missing, and adding it needs ~50–100 lines of C++.
Everything else — Tensor, autograd, optimizer — is real PyTorch.

## Run it

```bash
pip install -r requirements.txt          # numpy + libtpu (a TPU VM); torch (CPU build)
python tests/probe_add.py                # PJRT foundation: a+b on the TPU
python tests/test_ops.py                 # StableHLO op numerics vs numpy
python tests/test_backend.py             # real torch.autograd through dispatch == cpu autograd
python examples/train_torch.py --steps 200
python examples/profile_proof.py         # device identity + HBM + op profile + host/device accounting
```

`profile_proof.py` shows that one full `torch.nn` + autograd + AdamW step is **N TPU
executes with 0 mid-step host readbacks** — i.e. the entire forward, backward, and
optimizer ran on the TPU, not on the host.

## Scope / limitations (intentional)

- **Eager**, one StableHLO program per aten op; per-op dispatch + device sync.
- f32 + bf16-matmul (TPU MXU default); single device; no collectives.
- relu (not erf-gelu) and dropout=0 in the example, to stay within the op set.
- embedding is one-hot @ table (avoids gather/scatter); token ids stay on host.
- ~40 aten primitives implemented; composites handled by core-aten decompositions.
- A real builder (MLIR Python bindings) would be safer than text emission, but isn't
  available without pulling in jaxlib/torch_xla or heavy MLIR wheels — so we emit
  StableHLO text and keep the project dependency-free.
