"""Prove the real-PyTorch backend runs on the TPU, and collect a profile.

Evidence (all from the live runtime; PyTorch owns autograd/optimizer):
  1. DEVICE IDENTITY  - PJRT reports platform "tpu" / kind "TPU v6e"
  2. HBM RESIDENCY    - device memory-in-use grows as params upload to TPU HBM
  3. THROUGHPUT       - measured matmul GFLOP/s on the TPU vs a numpy/CPU baseline
  4. OP PROFILE       - per-StableHLO-op on-device timing over one training step
  5. ACCOUNTING       - host<->device traffic: 0 mid-step readbacks => full fwd+bwd
                        +optimizer ran on the TPU
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from mini_pytorch_xla import pjrt, profiler, backend as xb
import train_torch as tt


def mb(x):
    return "n/a" if x is None else f"{x/1e6:.1f} MB"


def main():
    c = pjrt.client()
    torch.manual_seed(0)

    print("=" * 70 + "\n1. DEVICE IDENTITY (PJRT / libtpu)\n" + "=" * 70)
    plat, kind = c.platform_name(), c.device_kind()
    print(f"   platform={plat}  kind={kind}  devices={c.num_devices()}")
    base = c.memory_stats()
    print(f"   HBM in use: {mb(base['bytes_in_use'])} (limit {mb(base['bytes_limit'])})")

    print("\n" + "=" * 70 + "\n2. HBM RESIDENCY (real torch.nn params on TPU)\n" + "=" * 70)
    V, L = 65, 64
    model = tt.LLM(V, L, d=128, layers=2, h=4, ff=256)
    xb.to_xla_(model)
    model._mask_xla = xb.to_xla(model.mask.float())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False, capturable=False)
    after = c.memory_stats()
    npar = sum(p.numel() for p in model.parameters())
    print(f"   built {npar/1e6:.2f}M-param torch.nn transformer")
    print(f"   HBM: {mb(base['bytes_in_use'])} -> {mb(after['bytes_in_use'])} "
          f"(+{mb(after['bytes_in_use']-base['bytes_in_use'])} resident on TPU)")

    print("\n" + "=" * 70 + "\n3. THROUGHPUT (TPU vs numpy/CPU)\n" + "=" * 70)
    for n in (1024, 2048):
        g = profiler.matmul_throughput(n=n, iters=30)
        x = np.random.randn(n, n).astype(np.float32)
        t0 = time.perf_counter()
        for _ in range(5):
            x @ x
        cpu = 2.0 * n ** 3 * 5 / (time.perf_counter() - t0) / 1e9
        print(f"   {n}x{n}: TPU {g:8.1f} GFLOP/s  |  numpy/CPU {cpu:8.1f} GFLOP/s")

    data = torch.randint(0, V, (5000,))

    def batch():
        ix = torch.randint(0, len(data) - L - 1, (16,))
        x = torch.stack([data[i:i + L] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + L] for i in ix])
        return x, y

    def step():
        x, y = batch()
        logits = model(x)
        logp = F.log_softmax(logits.reshape(-1, V), dim=-1)
        loss = -(logp * xb.to_xla(F.one_hot(y.reshape(-1), V).float())).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        return loss

    step()  # warm up (compile all op programs)

    print("\n" + "=" * 70 + "\n4. OP PROFILE + ACCOUNTING (one full step: fwd+bwd+AdamW)\n" + "=" * 70)
    c.counters.reset()
    with profiler.OpProfile() as prof:
        loss = step()
        rb_mid = c.counters.readbacks
        _ = xb.to_cpu(loss.detach()).item()
    print(prof.report())
    print(f"\n   TPU executes (on-device compute) : {c.counters.executes}")
    print(f"   readbacks DURING compute         : {rb_mid}   (0 => no host fallback)")
    print(f"   readbacks total                  : {c.counters.readbacks}   (only loss for logging)")
    ok = "tpu" in plat.lower() and after["bytes_in_use"] > base["bytes_in_use"] and rb_mid == 0
    print("\nVERDICT:", "PROVEN — real torch.nn+autograd+AdamW executed entirely on the TPU"
          if ok else "inconclusive")


if __name__ == "__main__":
    main()
