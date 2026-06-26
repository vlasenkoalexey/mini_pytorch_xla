# no_pytorch — the original frontend (no PyTorch at all)

This is the **original** version of mini-pytorch-xla, before the `__torch_dispatch__`
backend. It depends on **no torch** — it's a tiny, self-contained reverse-mode autograd
engine where each op lowers to StableHLO and runs eagerly on the TPU.

It **reuses the shared layers** of the parent project rather than duplicating them:

```
no_pytorch/tensor.py  (autograd: tape + backward formulas)
no_pytorch/nn.py      (Linear / Embedding / LayerNorm / softmax / cross_entropy / AdamW)
        │  reuses ↓
mini_pytorch_xla/ops.py   (buffer-level StableHLO lowerings)
mini_pytorch_xla/pjrt.py  (libtpu PJRT runtime, ctypes)
```

So `tensor.py` here is *purely* the autograd layer (Tensor class, reverse-mode
`backward()`, and per-op backward closures); all StableHLO emission and TPU execution
come from the parent package's `ops.py` / `pjrt.py`.

## How it differs from the main (torch) version

| | `no_pytorch/` (this) | main project (`backend.py`) |
|---|---|---|
| dependency | numpy + libtpu only (**no torch**) | torch + libtpu |
| Tensor / autograd | hand-rolled here | PyTorch's |
| Module / optimizer | hand-rolled (`nn.py`) | `torch.nn` / `torch.optim` |
| op interception | explicit `Tensor` method calls | `__torch_dispatch__` on aten |
| StableHLO + TPU runtime | **shared** (`ops.py`, `pjrt.py`) | **shared** (`ops.py`, `pjrt.py`) |

The main version is the honest "PyTorch backend" (PyTorch owns Tensor/autograd/optim);
this one is the standalone "tiny autograd on the TPU" that proves the StableHLO→libtpu
lowering works without any framework.

## Run

```bash
# from the project root (so `mini_pytorch_xla` is importable)
python no_pytorch/train_mini.py --steps 500
```

Trains a char-level Shakespeare transformer (modeled after Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT)) on a single TPU — loss drops from
~4.8 toward ~2.2, entirely through the hand-rolled autograd lowered to StableHLO.
