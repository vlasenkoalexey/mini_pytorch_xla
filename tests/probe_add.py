"""Foundation probe: drive libtpu's PJRT directly to add two arrays on the TPU."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from mini_pytorch_xla import pjrt

HLO = """
module @m {
  func.func public @main(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = stablehlo.add %a, %b : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
"""

c = pjrt.client()
print("client up; device acquired")
a = np.arange(6, dtype=np.float32).reshape(2, 3)
b = np.ones((2, 3), dtype=np.float32) * 10
ba, bb = c.from_host(a), c.from_host(b)
print("host->TPU ok:", ba.shape, ba.dtype)
exe = c.compile(HLO, num_outputs=1)
print("compiled StableHLO -> TPU executable")
(out,) = exe.execute([ba, bb])
res = out.to_numpy()
print("result shape", res.shape, "dtype", res.dtype)
print(res)
expected = a + b
assert np.allclose(res, expected), (res, expected)
print("\nPASS: a+b on TPU matches numpy")
