# mini-pytorch-xla

A **minimal, standalone, pure-Python eager TPU/XLA backend** — enough to train a
char-level Shakespeare transformer (the `train_xla.py` model) on a real TPU, with
**no dependency on `torch_xla`, `jax`, or even `torch`**. It talks straight to
`libtpu.so` through the **PJRT C API via ctypes**, lowering each high-level op to
a tiny **StableHLO** module that the TPU compiler turns into an executable.

It's a teaching-grade reimplementation of *how PyTorch/XLA works*, distilled to
its essence: intercept a high-level op → lower to an XLA program → run on device.

```
  tensor.py  (a high-level op, e.g. matmul)
      │   emit one StableHLO module for this op
      ▼
  hlo.py     (StableHLO text  +  compile/execute cache)
      │   PJRT_Client_Compile / _Execute
      ▼
  pjrt.py    (ctypes bindings over libtpu's PJRT C API)  ──►  TPU
```

## What maps to what in real PyTorch/XLA

| PyTorch/XLA (the real thing) | mini-pytorch-xla |
|---|---|
| `XLATensor` / lazy IR graph (`csrc/ir`) | `tensor.Tensor` — but **eager**: one op = one program |
| Lowering each IR node to HLO (`csrc/lowering_context`) | `tensor._binary`/`_dot_general`/… emit StableHLO text |
| `torch_xla/csrc/runtime` PjRt computation client | `pjrt.PjrtClient` (ctypes over `libtpu.so`) |
| `mark_step()` / `torch_xla.sync()` flush of the graph | nothing to flush — already eager (`sync()` is a no-op) |
| StableHLO export (`torch_xla.stablehlo`) | the StableHLO we hand to `PJRT_Client_Compile` |
| HLO/MXU runs matmul in bf16 at `precision DEFAULT` | same — our matmul tolerance reflects bf16 |

The deliberate simplification is **eager, no graph fusion**: real PyTorch/XLA
*traces* a whole step into one HLO graph and lets XLA fuse/schedule it; we compile
one StableHLO module **per op** and execute it immediately (caching the compiled
executable by program text, so step *N>0* reuses step 0's compilations). That's
slower but makes the lowering for every op visible and self-contained.

## How it works, end to end

1. **PJRT over ctypes (`pjrt.py`).** Model the entire `PJRT_Api` function table
   (pjrt_c_api.h v0.72) as a ctypes `Structure`, `GetPjrtApi()` from `libtpu.so`,
   `PJRT_Plugin_Initialize` → `PJRT_Client_Create`. Then: `from_host` (numpy →
   `BufferFromHostBuffer`), `compile` (StableHLO text → `Client_Compile` with a
   hand-serialized 6-byte `CompileOptionsProto`), `execute`
   (`LoadedExecutable_Execute`), `to_numpy` (`Buffer_ToHostBuffer`). Read-back
   forces a **row-major host layout** so that ops XLA implements as a pure layout
   change (e.g. a bare transpose) materialize correctly on the host.

2. **StableHLO per op (`hlo.py` + `tensor.py`).** Each primitive renders a
   `func.func public @main(...)` whose body is the op (`stablehlo.add`,
   `dot_general`, `reduce`, `transpose`, `broadcast_in_dim`, …). Broadcasting is
   emitted explicitly inside the module. Programs are cached by text.

3. **Reverse-mode autograd (`tensor.py`).** Forward ops record a `_backward`
   closure + their differentiable parents; `Tensor.backward()` seeds the scalar
   loss with 1 and walks the tape in reverse. Backward closures reuse the *same*
   primitives (so the backward pass also runs on the TPU); a `no_grad` flag during
   the walk prevents second-order taping.

4. **nn-like API (`nn.py`).** `Linear`, `Embedding` (one-hot @ table), `LayerNorm`,
   `gelu` (tanh approx), `softmax`, `cross_entropy`, `AdamW`. Token/target ids are
   one-hot'd **on the host**, which removes gather/scatter and keeps the primitive
   set to ~16 ops.

## Run it

```bash
# env with libtpu present (any TPU VM); set MINI_XLA_LIBTPU if autodetect fails
python tests/probe_add.py        # foundation: a+b on the TPU via raw PJRT
python tests/test_ops.py         # numeric + finite-diff gradient checks vs numpy
python examples/train_mini.py --steps 500
```

Typical output (TPU v6e, single chip):

```
mini-pytorch-xla | TPU | 0.29M params | vocab 65 | 2L d128 h4 block 64 batch 32
Step     0 | Loss: 4.8047
Step    10 | Loss: 3.4317
...
```

Loss starts at ~ln(65)=4.17-ish (random) and falls steadily — the transformer is
learning Shakespeare character statistics, trained entirely through StableHLO
lowered onto the TPU by this project.

## Files

- `mini_pytorch_xla/pjrt.py` — ctypes PJRT client over `libtpu.so`
- `mini_pytorch_xla/hlo.py` — StableHLO type strings + compile/execute cache
- `mini_pytorch_xla/tensor.py` — eager autograd Tensor; every primitive → StableHLO
- `mini_pytorch_xla/nn.py` — Linear/Embedding/LayerNorm/softmax/cross_entropy/AdamW
- `examples/train_mini.py` — the Shakespeare transformer (mirrors `train_xla.py`)
- `tests/` — `probe_add.py` (PJRT foundation), `test_ops.py` (numeric + grad checks)

## Scope / limitations (intentional)

- **Eager only**, no fusion; per-op dispatch + device sync makes steps ~0.5s.
- f32 + bf16-matmul (TPU MXU default); one device; no `reshard`/collectives.
- one-hot embedding/loss instead of gather/scatter; tanh-gelu instead of erf-gelu.
- ~16 primitives — exactly what this transformer + AdamW need, nothing more.
