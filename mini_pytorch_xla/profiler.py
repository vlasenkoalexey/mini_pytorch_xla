"""TPU profiling for the standalone backend.

Two things live here:

1. `OpProfile` / `matmul_throughput` — the *reachable* profile: per-StableHLO-op
   on-device timing and measured GFLOP/s. This is the real, working proof that
   ops execute on the TPU (timed across the PJRT device-completion barrier).

2. `Trace` — a best-effort XSpace capture via the PJRT profiler extension. It is
   ABI-correct (walk `PJRT_Api.extension_start` -> Profiler extension type=1 ->
   `PLUGIN_Profiler_Api`, then create/start/stop/collect_data — exactly what
   xla::profiler::PluginTracer does, see plugin_tracer.cc), BUT it returns an
   EMPTY buffer when driven standalone. Reason (confirmed against torch_xla):
   the plugin profiler is only a *contributor* to `tsl::profiler::ProfilerSession`
   (torch_xla/csrc/runtime/profiler.cpp + RegisterProfilerForPlugin). That C++
   session is what arms the TPU hardware tracer and merges host+device XSpaces,
   and it is NOT exposed in the PJRT C API — so a pure-ctypes collector cannot
   populate the trace. We keep `Trace` to document the mechanism; prefer
   `OpProfile` for actual proof.
"""

from __future__ import annotations

import ctypes
import re
from collections import Counter

from . import pjrt

c_sz = ctypes.c_size_t
c_vp = ctypes.c_void_p
c_i64 = ctypes.c_int64

_EXT_PROFILER = 1  # PJRT_Extension_Type_Profiler


class _ExtBase(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("type", ctypes.c_int), ("next", c_vp)]


class _ProfilerExt(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("type", ctypes.c_int), ("next", c_vp),
                ("profiler_api", c_vp), ("traceme_context_id", c_i64)]


class _ProfilerApi(ctypes.Structure):
    # PLUGIN_Profiler_Api: function-pointer table (profiler_c_api.h)
    _fields_ = [("struct_size", c_sz), ("priv", c_vp),
                ("error_destroy", c_vp), ("error_message", c_vp), ("error_get_code", c_vp),
                ("create", c_vp), ("destroy", c_vp), ("start", c_vp),
                ("stop", c_vp), ("collect_data", c_vp)]


class _Create_Args(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("options", c_vp),
                ("options_size", c_sz), ("profiler", c_vp)]


class _Profiler_Args(ctypes.Structure):  # Start / Stop / Destroy
    _fields_ = [("struct_size", c_sz), ("profiler", c_vp)]


class _CollectData_Args(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("profiler", c_vp),
                ("buffer", ctypes.POINTER(ctypes.c_uint8)), ("buffer_size_in_bytes", c_sz)]


class _Err_Msg_Args(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("priv", c_vp), ("error", c_vp),
                ("message", ctypes.c_char_p), ("message_size", c_sz)]


class _Err_Destroy_Args(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("priv", c_vp), ("error", c_vp)]


_FN = ctypes.CFUNCTYPE(c_vp, c_vp)


def _hdr(s):
    s.struct_size = ctypes.sizeof(type(s))
    return s


def _find_profiler_api(client) -> _ProfilerApi:
    ext = client._api.contents.extension_start
    while ext:
        base = ctypes.cast(ext, ctypes.POINTER(_ExtBase)).contents
        if base.type == _EXT_PROFILER:
            pe = ctypes.cast(ext, ctypes.POINTER(_ProfilerExt)).contents
            return ctypes.cast(pe.profiler_api, ctypes.POINTER(_ProfilerApi)).contents
        ext = base.next
    raise RuntimeError("PJRT Profiler extension not present in this plugin")


class Trace:
    """Context manager: capture a TPU XSpace trace of the enclosed work."""

    def __init__(self, out_path: str = "trace.xspace.pb"):
        self.out_path = out_path
        self._api = _find_profiler_api(pjrt.client())
        self._profiler = None
        self.data: bytes = b""

    def _call(self, fn_addr, args):
        err = _FN(fn_addr)(ctypes.byref(args))
        if err:
            m = _hdr(_Err_Msg_Args()); m.error = err
            _FN(self._api.error_message)(ctypes.byref(m))
            msg = m.message.decode() if m.message else "(no message)"
            d = _hdr(_Err_Destroy_Args()); d.error = err
            _FN(self._api.error_destroy)(ctypes.byref(d))
            raise RuntimeError(f"profiler: {msg}")

    def __enter__(self):
        # serialized tsl.ProfileOptions: host_tracer_level=2 (f2), device_tracer_level=1 (f3)
        self._opts = (ctypes.c_uint8 * 4)(0x10, 2, 0x18, 1)
        a = _hdr(_Create_Args())
        a.options = ctypes.cast(self._opts, c_vp)
        a.options_size = 4
        self._call(self._api.create, a)
        self._profiler = a.profiler
        s = _hdr(_Profiler_Args()); s.profiler = self._profiler
        self._call(self._api.start, s)
        return self

    def __exit__(self, *exc):
        s = _hdr(_Profiler_Args()); s.profiler = self._profiler
        self._call(self._api.stop, s)
        q = _hdr(_CollectData_Args()); q.profiler = self._profiler; q.buffer = None
        self._call(self._api.collect_data, q)
        n = int(q.buffer_size_in_bytes)
        buf = (ctypes.c_uint8 * n)()
        c = _hdr(_CollectData_Args()); c.profiler = self._profiler
        c.buffer = buf; c.buffer_size_in_bytes = n
        self._call(self._api.collect_data, c)
        self.data = bytes(buf)
        with open(self.out_path, "wb") as f:
            f.write(self.data)
        d = _hdr(_Profiler_Args()); d.profiler = self._profiler
        self._call(self._api.destroy, d)
        return False


class OpProfile:
    """On-device operator timing profile. Each timed op's wall-time includes the
    PJRT await of device completion, so it reflects real TPU execution time."""

    def __init__(self):
        from collections import defaultdict
        self.calls = defaultdict(int)
        self.secs = defaultdict(float)
        self.compiles = defaultdict(int)

    def __enter__(self):
        from . import hlo
        hlo.set_timer(self._record)
        return self

    def __exit__(self, *exc):
        from . import hlo
        hlo.set_timer(None)
        return False

    def _record(self, kind, seconds, compiled):
        self.calls[kind] += 1
        self.secs[kind] += seconds
        if compiled:
            self.compiles[kind] += 1

    def report(self) -> str:
        rows = sorted(self.secs.items(), key=lambda kv: -kv[1])
        total = sum(self.secs.values()) or 1e-9
        out = [f"  {'op (StableHLO)':<20}{'calls':>8}{'total ms':>12}{'%time':>8}{'µs/call':>10}"]
        for kind, s in rows:
            n = self.calls[kind]
            out.append(f"  {kind:<20}{n:>8}{s*1e3:>12.1f}{100*s/total:>7.1f}%{s/n*1e6:>10.1f}")
        out.append(f"  {'TOTAL':<20}{sum(self.calls.values()):>8}{total*1e3:>12.1f}{'100.0':>7}%")
        return "\n".join(out)


def matmul_throughput(n: int = 2048, iters: int = 50) -> float:
    """Measured GFLOP/s of an n×n matmul on the TPU (proof of real compute)."""
    import time
    import numpy as np
    from . import ops
    a = ops.from_np(np.random.randn(n, n).astype(np.float32))
    b = ops.from_np(np.random.randn(n, n).astype(np.float32))
    ops.mm(a, b).to_numpy()                  # warm/compile
    t0 = time.perf_counter()
    for _ in range(iters):
        c = ops.mm(a, b)
    c.to_numpy()                             # ensure the last execute completed
    dt = time.perf_counter() - t0
    flops = 2.0 * n ** 3 * iters
    return flops / dt / 1e9


def summarize_xspace(data: bytes) -> dict:
    """Dependency-free scan of a serialized XSpace: device planes + op-name events."""
    strings = [s.decode("ascii", "ignore") for s in re.findall(rb"[ -~]{3,}", data)]
    planes = sorted({s for s in strings if s.startswith("/device:") or s.startswith("/host:")})
    tpu_planes = [s for s in planes if "TPU" in s]
    op_kinds = ("fusion", "convolution", "dot", "transpose", "reduce", "add",
                "multiply", "subtract", "divide", "broadcast", "reshape", "copy",
                "exponential", "tanh", "log", "rsqrt", "select", "convert", "while")
    ops = Counter()
    for s in strings:
        low = s.lower()
        for k in op_kinds:
            if k in low:
                ops[k] += 1
    return {
        "bytes": len(data),
        "device_planes": planes,
        "tpu_planes": tpu_planes,
        "has_tensorcore": any("TensorCore" in s or "Tensor Core" in s for s in strings),
        "op_event_hits": dict(ops.most_common()),
        "total_strings": len(strings),
    }
