"""Minimal PJRT (Pretty-much Just another RunTime) client over libtpu, via ctypes.

This is the *runtime* layer of mini-pytorch-xla — the equivalent of PyTorch/XLA's
`torch_xla/csrc/runtime` PjRt computation client, but in pure Python with zero
dependency on torch_xla or jax. It talks straight to `libtpu.so`'s PJRT C API
(`GetPjrtApi`), the same plugin those frameworks load.

What it gives the rest of the project:
  - PjrtClient.from_host(np_array)          -> Buffer        (host -> TPU HBM)
  - PjrtClient.compile(stablehlo_text)      -> Executable    (StableHLO -> TPU)
  - Executable.execute([Buffer, ...])       -> [Buffer, ...] (run on TPU)
  - Buffer.to_numpy()                       -> np.ndarray     (TPU HBM -> host)

Every PJRT entry point has the same C signature: `PJRT_Error* fn(FOO_Args*)`,
where the Args struct begins with `{size_t struct_size; void* extension_start; ...}`
and a NULL return means success. We model the whole `PJRT_Api` function table as a
ctypes Structure (exact field order from pjrt_c_api.h, v0.72) so we can call any
entry by name.
"""

from __future__ import annotations

import ctypes
import os
import numpy as np

# ----------------------------------------------------------------------------- #
# The PJRT_Api function table (pjrt_c_api.h order). Every entry is a function
# pointer `PJRT_Error* (*)(Args*)`. Order is load-bearing — it mirrors the header.
# ----------------------------------------------------------------------------- #
_FN_NAMES = [
    "PJRT_Error_Destroy", "PJRT_Error_Message", "PJRT_Error_GetCode",
    "PJRT_Plugin_Initialize", "PJRT_Plugin_Attributes",
    "PJRT_Event_Destroy", "PJRT_Event_IsReady", "PJRT_Event_Error",
    "PJRT_Event_Await", "PJRT_Event_OnReady",
    "PJRT_Client_Create", "PJRT_Client_Destroy", "PJRT_Client_PlatformName",
    "PJRT_Client_ProcessIndex", "PJRT_Client_PlatformVersion", "PJRT_Client_Devices",
    "PJRT_Client_AddressableDevices", "PJRT_Client_LookupDevice",
    "PJRT_Client_LookupAddressableDevice", "PJRT_Client_AddressableMemories",
    "PJRT_Client_Compile", "PJRT_Client_DefaultDeviceAssignment",
    "PJRT_Client_BufferFromHostBuffer",
    "PJRT_DeviceDescription_Id", "PJRT_DeviceDescription_ProcessIndex",
    "PJRT_DeviceDescription_Attributes", "PJRT_DeviceDescription_Kind",
    "PJRT_DeviceDescription_DebugString", "PJRT_DeviceDescription_ToString",
    "PJRT_Device_GetDescription", "PJRT_Device_IsAddressable",
    "PJRT_Device_LocalHardwareId", "PJRT_Device_AddressableMemories",
    "PJRT_Device_DefaultMemory", "PJRT_Device_MemoryStats",
    "PJRT_Memory_Id", "PJRT_Memory_Kind", "PJRT_Memory_DebugString",
    "PJRT_Memory_ToString", "PJRT_Memory_AddressableByDevices",
    "PJRT_Executable_Destroy", "PJRT_Executable_Name", "PJRT_Executable_NumReplicas",
    "PJRT_Executable_NumPartitions", "PJRT_Executable_NumOutputs",
    "PJRT_Executable_SizeOfGeneratedCodeInBytes", "PJRT_Executable_GetCostAnalysis",
    "PJRT_Executable_OutputMemoryKinds", "PJRT_Executable_OptimizedProgram",
    "PJRT_Executable_Serialize",
    "PJRT_LoadedExecutable_Destroy", "PJRT_LoadedExecutable_GetExecutable",
    "PJRT_LoadedExecutable_AddressableDevices", "PJRT_LoadedExecutable_Delete",
    "PJRT_LoadedExecutable_IsDeleted", "PJRT_LoadedExecutable_Execute",
    "PJRT_Executable_DeserializeAndLoad", "PJRT_LoadedExecutable_Fingerprint",
    "PJRT_Buffer_Destroy", "PJRT_Buffer_ElementType", "PJRT_Buffer_Dimensions",
    "PJRT_Buffer_UnpaddedDimensions", "PJRT_Buffer_DynamicDimensionIndices",
    "PJRT_Buffer_GetMemoryLayout", "PJRT_Buffer_OnDeviceSizeInBytes",
    "PJRT_Buffer_Device", "PJRT_Buffer_Memory", "PJRT_Buffer_Delete",
    "PJRT_Buffer_IsDeleted", "PJRT_Buffer_CopyToDevice", "PJRT_Buffer_ToHostBuffer",
    "PJRT_Buffer_IsOnCpu", "PJRT_Buffer_ReadyEvent", "PJRT_Buffer_UnsafePointer",
    "PJRT_Buffer_IncreaseExternalReferenceCount",
    "PJRT_Buffer_DecreaseExternalReferenceCount",
    "PJRT_Buffer_OpaqueDeviceMemoryDataPointer",
    "PJRT_CopyToDeviceStream_Destroy", "PJRT_CopyToDeviceStream_AddChunk",
    "PJRT_CopyToDeviceStream_TotalBytes", "PJRT_CopyToDeviceStream_GranuleSize",
    "PJRT_CopyToDeviceStream_CurrentBytes",
    "PJRT_TopologyDescription_Create", "PJRT_TopologyDescription_Destroy",
    "PJRT_TopologyDescription_PlatformName", "PJRT_TopologyDescription_PlatformVersion",
    "PJRT_TopologyDescription_GetDeviceDescriptions",
    "PJRT_TopologyDescription_Serialize", "PJRT_TopologyDescription_Attributes",
    "PJRT_Compile", "PJRT_Executable_OutputElementTypes",
    "PJRT_Executable_OutputDimensions", "PJRT_Buffer_CopyToMemory",
    "PJRT_Client_CreateViewOfDeviceBuffer", "PJRT_Executable_Fingerprint",
    "PJRT_Client_TopologyDescription", "PJRT_Executable_GetCompiledMemoryStats",
    "PJRT_Memory_Kind_Id", "PJRT_ExecuteContext_Create", "PJRT_ExecuteContext_Destroy",
    "PJRT_Buffer_CopyRawToHost",
    "PJRT_AsyncHostToDeviceTransferManager_Destroy",
    "PJRT_AsyncHostToDeviceTransferManager_TransferData",
    "PJRT_Client_CreateBuffersForAsyncHostToDevice",
    "PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer",
    "PJRT_AsyncHostToDeviceTransferManager_Device",
    "PJRT_AsyncHostToDeviceTransferManager_BufferCount",
    "PJRT_AsyncHostToDeviceTransferManager_BufferSize",
    "PJRT_AsyncHostToDeviceTransferManager_SetBufferError",
    "PJRT_AsyncHostToDeviceTransferManager_AddMetadata",
    "PJRT_Client_DmaMap", "PJRT_Client_DmaUnmap",
    "PJRT_Client_CreateUninitializedBuffer",
]

c_sz = ctypes.c_size_t
c_vp = ctypes.c_void_p
c_i64 = ctypes.c_int64
c_int = ctypes.c_int


class _ApiVersion(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("extension_start", c_vp),
                ("major", c_int), ("minor", c_int)]


class PJRT_Api(ctypes.Structure):
    _fields_ = ([("struct_size", c_sz), ("extension_start", c_vp),
                 ("pjrt_api_version", _ApiVersion)]
                + [(n, c_vp) for n in _FN_NAMES])


# Every PJRT entry: PJRT_Error* fn(void* args)
_PJRT_FN = ctypes.CFUNCTYPE(c_vp, c_vp)


def _args_header(struct_instance):
    """All *_Args structs start with {size_t struct_size; void* extension_start;}."""
    struct_instance.struct_size = ctypes.sizeof(type(struct_instance))
    struct_instance.extension_start = None
    return struct_instance


# ---- PJRT_Buffer_Type enum (subset we use) ---------------------------------- #
PRED, S8, S16, S32, S64, U8, U16, U32, U64, F16, F32, F64, BF16 = range(1, 14)

_NP_TO_PJRT = {
    np.dtype("bool"): PRED, np.dtype("int8"): S8, np.dtype("int16"): S16,
    np.dtype("int32"): S32, np.dtype("int64"): S64, np.dtype("uint8"): U8,
    np.dtype("float16"): F16, np.dtype("float32"): F32, np.dtype("float64"): F64,
}
_PJRT_TO_NP = {v: k for k, v in _NP_TO_PJRT.items()}

# kImmutableUntilTransferCompletes — runtime may keep `data` until transfer done.
_HBS_UNTIL_DONE = 1


# ----------------------------------------------------------------------------- #
# Minimal CompileOptionsProto (hand-serialized; no protobuf dependency).
#   CompileOptionsProto.executable_build_options (field 3, message)
#     ExecutableBuildOptionsProto.num_replicas  (field 4, varint) = 1
#     ExecutableBuildOptionsProto.num_partitions(field 5, varint) = 1
# ----------------------------------------------------------------------------- #
def _compile_options_bytes() -> bytes:
    ebo = bytes([(4 << 3) | 0, 1, (5 << 3) | 0, 1])      # num_replicas=1, num_partitions=1
    return bytes([(3 << 3) | 2, len(ebo)]) + ebo          # field 3, length-delimited


def _find_libtpu() -> str:
    if os.environ.get("MINI_XLA_LIBTPU"):
        return os.environ["MINI_XLA_LIBTPU"]
    # The pip `libtpu` package ships libtpu.so next to its __init__.
    try:
        import libtpu
        cand = os.path.join(os.path.dirname(libtpu.__file__), "libtpu.so")
        if os.path.exists(cand):
            return cand
    except Exception:
        pass
    # Fall back to scanning site-packages.
    import glob
    import site
    roots = list(site.getsitepackages()) if hasattr(site, "getsitepackages") else []
    roots += [p for p in os.environ.get("PYTHONPATH", "").split(":") if p]
    for r in roots:
        hits = glob.glob(os.path.join(r, "libtpu", "libtpu.so"))
        if hits:
            return hits[0]
    raise RuntimeError("libtpu.so not found; `pip install libtpu` or set "
                       "MINI_XLA_LIBTPU=/path/to/libtpu.so")


class PjrtError(RuntimeError):
    pass


class Buffer:
    """A handle to a tensor living in TPU HBM (a PJRT_Buffer*)."""

    def __init__(self, client: "PjrtClient", handle, shape, np_dtype):
        self._client = client
        self._h = handle
        self.shape = tuple(shape)
        self.dtype = np.dtype(np_dtype)
        self._deleted = False

    def to_numpy(self) -> np.ndarray:
        return self._client._buffer_to_host(self)

    def delete(self):
        if not self._deleted and self._h:
            self._client._call("PJRT_Buffer_Destroy",
                               _Buffer_Destroy_Args, dict(buffer=self._h))
            self._deleted = True

    def __del__(self):
        try:
            self.delete()
        except Exception:
            pass


class Executable:
    """A compiled StableHLO program loaded on the TPU (a PJRT_LoadedExecutable*)."""

    def __init__(self, client: "PjrtClient", handle, num_outputs):
        self._client = client
        self._h = handle
        self.num_outputs = num_outputs

    def execute(self, inputs: list[Buffer]) -> list[Buffer]:
        return self._client._execute(self, inputs)


# ---- ctypes Args structs (exact field order from pjrt_c_api.h) --------------- #
class _Client_Create_Args(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("extension_start", c_vp),
                ("create_options", c_vp), ("num_options", c_sz),
                ("kv_get_callback", c_vp), ("kv_get_user_arg", c_vp),
                ("kv_put_callback", c_vp), ("kv_put_user_arg", c_vp),
                ("client", c_vp),
                ("kv_try_get_callback", c_vp), ("kv_try_get_user_arg", c_vp)]


class _Plugin_Initialize_Args(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("extension_start", c_vp)]


class _AddressableDevices_Args(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("extension_start", c_vp),
                ("client", c_vp), ("addressable_devices", c_vp),
                ("num_addressable_devices", c_sz)]


class _Program(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("extension_start", c_vp),
                ("code", c_vp), ("code_size", c_sz),
                ("format", c_vp), ("format_size", c_sz)]


class _Compile_Args(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("extension_start", c_vp),
                ("client", c_vp), ("program", c_vp),
                ("compile_options", c_vp), ("compile_options_size", c_sz),
                ("executable", c_vp)]


class _BufferFromHost_Args(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("extension_start", c_vp),
                ("client", c_vp), ("data", c_vp), ("type", c_int),
                ("dims", ctypes.POINTER(c_i64)), ("num_dims", c_sz),
                ("byte_strides", ctypes.POINTER(c_i64)), ("num_byte_strides", c_sz),
                ("host_buffer_semantics", c_int), ("device", c_vp), ("memory", c_vp),
                ("device_layout", c_vp), ("done_with_host_buffer", c_vp),
                ("buffer", c_vp)]


class _ExecuteOptions(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("extension_start", c_vp),
                ("send_callbacks", c_vp), ("recv_callbacks", c_vp),
                ("num_send_ops", c_sz), ("num_recv_ops", c_sz),
                ("launch_id", c_int), ("non_donatable_input_indices", c_vp),
                ("num_non_donatable_input_indices", c_sz), ("context", c_vp)]


class _Execute_Args(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("extension_start", c_vp),
                ("executable", c_vp), ("options", c_vp),
                ("argument_lists", c_vp), ("num_devices", c_sz), ("num_args", c_sz),
                ("output_lists", c_vp), ("device_complete_events", c_vp),
                ("execute_device", c_vp)]


class _ToHost_Args(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("extension_start", c_vp),
                ("src", c_vp), ("host_layout", c_vp),
                ("dst", c_vp), ("dst_size", c_sz), ("event", c_vp)]


class _Buffer_Destroy_Args(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("extension_start", c_vp), ("buffer", c_vp)]


class _Buffer_Dimensions_Args(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("extension_start", c_vp), ("buffer", c_vp),
                ("dims", ctypes.POINTER(c_i64)), ("num_dims", c_sz)]


class _PlatformName_Args(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("extension_start", c_vp), ("client", c_vp),
                ("platform_name", ctypes.c_char_p), ("platform_name_size", c_sz)]


class _GetDescription_Args(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("extension_start", c_vp), ("device", c_vp),
                ("device_description", c_vp)]


class _Kind_Args(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("extension_start", c_vp),
                ("device_description", c_vp),
                ("device_kind", ctypes.c_char_p), ("device_kind_size", c_sz)]


class _MemoryStats_Args(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("extension_start", c_vp), ("device", c_vp),
                ("bytes_in_use", c_i64),
                ("peak_bytes_in_use", c_i64), ("peak_bytes_in_use_is_set", ctypes.c_bool),
                ("num_allocs", c_i64), ("num_allocs_is_set", ctypes.c_bool),
                ("largest_alloc_size", c_i64), ("largest_alloc_size_is_set", ctypes.c_bool),
                ("bytes_limit", c_i64), ("bytes_limit_is_set", ctypes.c_bool),
                ("bytes_reserved", c_i64), ("bytes_reserved_is_set", ctypes.c_bool),
                ("peak_bytes_reserved", c_i64), ("peak_bytes_reserved_is_set", ctypes.c_bool),
                ("bytes_reservable_limit", c_i64), ("bytes_reservable_limit_is_set", ctypes.c_bool),
                ("largest_free_block_bytes", c_i64), ("largest_free_block_bytes_is_set", ctypes.c_bool),
                ("pool_bytes", c_i64), ("pool_bytes_is_set", ctypes.c_bool),
                ("peak_pool_bytes", c_i64), ("peak_pool_bytes_is_set", ctypes.c_bool)]


class _Buffer_ElementType_Args(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("extension_start", c_vp), ("buffer", c_vp),
                ("type", c_int)]


class _Event_Args(ctypes.Structure):  # Destroy / Await / Error share this shape
    _fields_ = [("struct_size", c_sz), ("extension_start", c_vp), ("event", c_vp)]


class _MemoryLayout(ctypes.Structure):
    # PJRT_Buffer_MemoryLayout with the Tiled union member inlined (type=Tiled=0).
    _fields_ = [("struct_size", c_sz), ("extension_start", c_vp),
                ("t_struct_size", c_sz), ("t_extension_start", c_vp),
                ("minor_to_major", ctypes.POINTER(c_i64)), ("minor_to_major_size", c_sz),
                ("tile_dims", c_vp), ("tile_dim_sizes", c_vp), ("num_tiles", c_sz),
                ("type", c_int)]


def _rowmajor_layout(rank: int):
    """Standard row-major (descending minor_to_major) tiled layout, no tiling."""
    lay = _MemoryLayout()
    lay.struct_size = ctypes.sizeof(_MemoryLayout)
    lay.extension_start = None
    lay.t_struct_size = 7 * 8  # sizeof(PJRT_Buffer_MemoryLayout_Tiled)
    lay.t_extension_start = None
    m2m = (c_i64 * rank)(*range(rank - 1, -1, -1))
    lay.minor_to_major = m2m
    lay.minor_to_major_size = rank
    lay.tile_dims = None
    lay.tile_dim_sizes = None
    lay.num_tiles = 0
    lay.type = 0  # Tiled
    return lay, m2m  # return m2m to keep it alive


class _Error_Message_Args(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("extension_start", c_vp),
                ("error", c_vp), ("message", ctypes.c_char_p), ("message_size", c_sz)]


class _Error_Destroy_Args(ctypes.Structure):
    _fields_ = [("struct_size", c_sz), ("extension_start", c_vp), ("error", c_vp)]


class _Counters:
    """Accounting of host<->device traffic vs on-device executes (proof tooling)."""
    def __init__(self):
        self.executes = 0      # TPU program executions (compute on device)
        self.uploads = 0       # host -> TPU (BufferFromHostBuffer)
        self.readbacks = 0     # TPU -> host (Buffer_ToHostBuffer)

    def reset(self):
        self.executes = self.uploads = self.readbacks = 0


class PjrtClient:
    def __init__(self, libtpu_path: str | None = None):
        self.counters = _Counters()
        path = libtpu_path or _find_libtpu()
        self._lib = ctypes.CDLL(path)
        self._lib.GetPjrtApi.restype = ctypes.POINTER(PJRT_Api)
        self._api = self._lib.GetPjrtApi()
        if not self._api:
            raise PjrtError("GetPjrtApi returned NULL")
        self._fns: dict[str, object] = {}
        # Plugin_Initialize must run before Client_Create for the TPU plugin.
        self._raw_call("PJRT_Plugin_Initialize", _args_header(_Plugin_Initialize_Args()))
        cc = _args_header(_Client_Create_Args())
        self._raw_call("PJRT_Client_Create", cc)
        self._client = cc.client
        self._device = self._first_device()

    # -- low-level dispatch ---------------------------------------------------- #
    def _fn(self, name):
        f = self._fns.get(name)
        if f is None:
            addr = getattr(self._api.contents, name)
            if not addr:
                raise PjrtError(f"{name} is NULL in this PJRT plugin")
            f = _PJRT_FN(addr)
            self._fns[name] = f
        return f

    def _check_error(self, err_ptr):
        if not err_ptr:
            return
        m = _args_header(_Error_Message_Args())
        m.error = err_ptr
        self._fn("PJRT_Error_Message")(ctypes.byref(m))
        msg = m.message.decode() if m.message else "(no message)"
        d = _args_header(_Error_Destroy_Args())
        d.error = err_ptr
        self._fn("PJRT_Error_Destroy")(ctypes.byref(d))
        raise PjrtError(msg)

    def _raw_call(self, name, args_struct):
        err = self._fn(name)(ctypes.byref(args_struct))
        self._check_error(err)
        return args_struct

    def _call(self, name, args_cls, fields: dict):
        a = _args_header(args_cls())
        for k, v in fields.items():
            setattr(a, k, v)
        return self._raw_call(name, a)

    def _await(self, event_ptr):
        if not event_ptr:
            return
        ev = _args_header(_Event_Args())
        ev.event = event_ptr
        self._raw_call("PJRT_Event_Await", ev)
        ev2 = _args_header(_Event_Args())
        ev2.event = event_ptr
        self._fn("PJRT_Event_Destroy")(ctypes.byref(ev2))

    # -- device ---------------------------------------------------------------- #
    def _first_device(self):
        a = _args_header(_AddressableDevices_Args())
        a.client = self._client
        self._raw_call("PJRT_Client_AddressableDevices", a)
        if a.num_addressable_devices < 1:
            raise PjrtError("no addressable TPU devices")
        dev_array = ctypes.cast(a.addressable_devices, ctypes.POINTER(c_vp))
        return dev_array[0]

    # -- device identity + memory (proof it's a TPU) --------------------------- #
    def platform_name(self) -> str:
        a = self._call("PJRT_Client_PlatformName", _PlatformName_Args, dict(client=self._client))
        return a.platform_name.decode() if a.platform_name else "?"

    def device_kind(self) -> str:
        g = self._call("PJRT_Device_GetDescription", _GetDescription_Args, dict(device=self._device))
        k = self._call("PJRT_DeviceDescription_Kind", _Kind_Args,
                       dict(device_description=g.device_description))
        return k.device_kind.decode() if k.device_kind else "?"

    def num_devices(self) -> int:
        a = _args_header(_AddressableDevices_Args())
        a.client = self._client
        self._raw_call("PJRT_Client_AddressableDevices", a)
        return int(a.num_addressable_devices)

    def memory_stats(self) -> dict:
        a = self._call("PJRT_Device_MemoryStats", _MemoryStats_Args, dict(device=self._device))
        return {
            "bytes_in_use": int(a.bytes_in_use),
            "peak_bytes_in_use": int(a.peak_bytes_in_use) if a.peak_bytes_in_use_is_set else None,
            "bytes_limit": int(a.bytes_limit) if a.bytes_limit_is_set else None,
            "num_allocs": int(a.num_allocs) if a.num_allocs_is_set else None,
        }

    # -- host <-> device ------------------------------------------------------- #
    def from_host(self, arr: np.ndarray) -> Buffer:
        self.counters.uploads += 1
        arr = np.ascontiguousarray(arr)
        if arr.dtype not in _NP_TO_PJRT:
            raise PjrtError(f"unsupported dtype {arr.dtype}")
        dims = (c_i64 * arr.ndim)(*arr.shape)
        a = _args_header(_BufferFromHost_Args())
        a.client = self._client
        a.data = arr.ctypes.data
        a.type = _NP_TO_PJRT[arr.dtype]
        a.dims = dims if arr.ndim else None
        a.num_dims = arr.ndim
        a.byte_strides = None
        a.num_byte_strides = 0
        a.host_buffer_semantics = _HBS_UNTIL_DONE
        a.device = self._device
        a.memory = None
        a.device_layout = None
        self._raw_call("PJRT_Client_BufferFromHostBuffer", a)
        self._await(a.done_with_host_buffer)   # safe to free `arr` after this
        return Buffer(self, a.buffer, arr.shape, arr.dtype)

    def _buffer_to_host(self, buf: Buffer) -> np.ndarray:
        self.counters.readbacks += 1
        out = np.empty(buf.shape, dtype=buf.dtype)
        a = _args_header(_ToHost_Args())
        a.src = buf._h
        # Force a standard row-major host layout: XLA may keep a transpose as a
        # pure layout change on device, so without this the bytes come back
        # un-permuted. Requesting descending minor_to_major makes PJRT materialize.
        layout, _m2m = _rowmajor_layout(len(buf.shape)) if buf.shape else (None, None)
        a.host_layout = ctypes.cast(ctypes.byref(layout), c_vp) if layout else None
        a.dst = out.ctypes.data
        a.dst_size = out.nbytes
        self._raw_call("PJRT_Buffer_ToHostBuffer", a)
        self._await(a.event)
        return out

    # -- compile + execute ----------------------------------------------------- #
    def compile(self, stablehlo_text: str, num_outputs: int = 1) -> Executable:
        code = stablehlo_text.encode("utf-8")
        fmt = b"mlir"
        prog = _args_header(_Program())
        prog.code = ctypes.cast(ctypes.c_char_p(code), c_vp)
        prog.code_size = len(code)
        prog.format = ctypes.cast(ctypes.c_char_p(fmt), c_vp)
        prog.format_size = len(fmt)
        opts = _compile_options_bytes()
        a = _args_header(_Compile_Args())
        a.client = self._client
        a.program = ctypes.cast(ctypes.byref(prog), c_vp)
        a.compile_options = ctypes.cast(ctypes.c_char_p(opts), c_vp)
        a.compile_options_size = len(opts)
        self._raw_call("PJRT_Client_Compile", a)
        return Executable(self, a.executable, num_outputs)

    def _execute(self, exe: Executable, inputs: list[Buffer]) -> list[Buffer]:
        self.counters.executes += 1
        n_in = len(inputs)
        n_out = exe.num_outputs
        in_arr = (c_vp * n_in)(*[b._h for b in inputs])
        in_lists = (ctypes.POINTER(c_vp) * 1)(ctypes.cast(in_arr, ctypes.POINTER(c_vp)))
        out_arr = (c_vp * n_out)()
        out_lists = (ctypes.POINTER(c_vp) * 1)(ctypes.cast(out_arr, ctypes.POINTER(c_vp)))
        complete = (c_vp * 1)()

        opts = _args_header(_ExecuteOptions())
        a = _args_header(_Execute_Args())
        a.executable = exe._h
        a.options = ctypes.cast(ctypes.byref(opts), c_vp)
        a.argument_lists = ctypes.cast(in_lists, c_vp)
        a.num_devices = 1
        a.num_args = n_in
        a.output_lists = ctypes.cast(out_lists, c_vp)
        a.device_complete_events = ctypes.cast(complete, c_vp)
        a.execute_device = None
        self._raw_call("PJRT_LoadedExecutable_Execute", a)
        self._await(complete[0])
        # Outputs are self-describing: query shape + dtype back from the device.
        return [self._wrap_output(out_arr[i]) for i in range(n_out)]

    def _wrap_output(self, handle) -> Buffer:
        d = _args_header(_Buffer_Dimensions_Args())
        d.buffer = handle
        self._raw_call("PJRT_Buffer_Dimensions", d)
        shape = tuple(d.dims[i] for i in range(d.num_dims))
        e = _args_header(_Buffer_ElementType_Args())
        e.buffer = handle
        self._raw_call("PJRT_Buffer_ElementType", e)
        np_dtype = _PJRT_TO_NP.get(e.type, np.dtype("float32"))
        return Buffer(self, handle, shape, np_dtype)


_GLOBAL: PjrtClient | None = None


def client() -> PjrtClient:
    """Process-wide singleton (one TPU client; the chips are a shared resource)."""
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = PjrtClient()
    return _GLOBAL
