#!/usr/bin/env python3
# probe_pynv_device_input.py

import sys
import traceback

import numpy as np
import torch

try:
    import PyNvVideoCodec as nvc
except Exception as e:
    print("FAIL: Could not import PyNvVideoCodec:", e)
    sys.exit(1)

W, H = 64, 64
CODEC = "h264"   # or "hevc"; keep it simple for the probe
FMT   = "NV12"   # exact match with docs: 2 planes (Y; interleaved UV)

# ---------- Helpers for NV12 device-input (CAI) ----------
class AppCAI:
    """CUDA Array Interface plane pointing to an existing torch CUDA tensor."""
    def __init__(self, t: torch.Tensor):
        assert t.is_cuda and t.dtype == torch.uint8 and t.is_contiguous()
        itemsize = t.element_size()  # 1 for uint8
        strides_bytes = tuple(int(s * itemsize) for s in t.stride())
        self.__cuda_array_interface__ = {
            "shape": tuple(int(x) for x in t.shape),   # e.g., (H,W,1) or (H/2,W/2,2)
            "strides": strides_bytes,                  # BYTES
            "typestr": "|u1",
            "data": (int(t.data_ptr()), False),
            "version": 3,
        }
        self._keep = t  # keep tensor alive

class AppFrameNV12:
    """Matches the Programming Guide: .cuda() returns a list of planes."""
    def __init__(self, y_hw1: torch.Tensor, uv_h2w2x2: torch.Tensor):
        self._planes = [AppCAI(y_hw1), AppCAI(uv_h2w2x2)]
    def cuda(self):
        return self._planes

def try_device_input_nv12() -> bool:
    """Return True if device-input encode succeeds (or at least doesn’t throw)."""
    # 1) Create encoder (device path)
    enc = nvc.CreateEncoder(
        width=W,
        height=H,
        fmt=FMT,
        usecpuinputbuffer=False,    # <- device path
        codec=CODEC,
        gop="1",                    # I-frames only to avoid buffering confusion
        rc="vbr",
        # preset="p4",
        profile="main",
        constqp="28",
        fps="30",
        bf="0", b_adapt="0",
    )

    # 2) Create NV12 planes on CUDA with torch
    # Y plane: (H, W, 1)
    y = torch.zeros((H, W, 1), dtype=torch.uint8, device="cuda")
    # make a visible pattern
    y[..., 0] = torch.arange(W, device="cuda", dtype=torch.uint8).unsqueeze(0) % 256

    # UV plane (interleaved): (H//2, W//2, 2)
    uv = torch.full((H//2, W//2, 2), 128, dtype=torch.uint8, device="cuda")

    # 3) Wrap in AppFrameNV12 (CAI planes)
    frame = AppFrameNV12(y.contiguous(), uv.contiguous())

    # 4) Encode one frame (device input). Might return empty bytes due to GOP settings,
    #    but it should NOT throw if device input is supported.
    try:
        bitstream = enc.Encode(frame)
        # flush as well
        tail = enc.EndEncode()
    except Exception:
        print("DEVICE_INPUT: EXCEPTION during Encode()")
        traceback.print_exc(limit=1)
        return False

    print("DEVICE_INPUT: Encode() returned without throwing.")
    return True

def try_cpu_input_nv12() -> bool:
    """Return True if CPU-input encode works (control case)."""
    enc = nvc.CreateEncoder(
        width=W,
        height=H,
        fmt=FMT,
        usecpuinputbuffer=True,     # <- CPU path
        codec=CODEC,
        gop="1",
        rc="vbr",
        # preset="p1",
        profile="main",
        constqp="28",
        fps="30",
        bf="0", b_adapt="0",
    )

    # Build a CPU NV12 buffer: Y (H*W) then interleaved UV (H/2*W/2*2)
    y = np.zeros((H, W), dtype=np.uint8)
    y[:] = np.arange(W, dtype=np.uint8).reshape(1, W)
    uv = np.full((H//2, W//2, 2), 128, dtype=np.uint8)

    # NV12 linear buffer
    buf = np.concatenate([y.flatten(), uv.flatten()]).astype(np.uint8)

    try:
        bitstream = enc.Encode(buf)
        tail = enc.EndEncode()
    except Exception:
        print("CPU_INPUT: EXCEPTION during Encode()")
        traceback.print_exc(limit=1)
        return False

    print("CPU_INPUT: Encode() returned without throwing.")
    return True

def main():
    print("PyNvVideoCodec version:", getattr(nvc, "__version__", "unknown"))
    print("Testing device-input (CUDA Array Interface) with NV12 ...")
    ok_dev = try_device_input_nv12()
    print("Testing CPU-input fallback with NV12 ...")
    ok_cpu = try_cpu_input_nv12()

    print("\nRESULTS:")
    print("  Device-input (CAI) :", "PASS" if ok_dev else "FAIL")
    print("  CPU-input          :", "PASS" if ok_cpu else "FAIL")

    if ok_cpu and not ok_dev:
        print("\nConclusion: Your PyNvVideoCodec build very likely lacks the device-input (CUDA Array Interface) path,")
        print("or expects a different overload. CPU path works; device path fails.")
        sys.exit(2)
    elif ok_dev:
        print("\nConclusion: Device-input path is supported. If your training loop still fails, it’s likely a packing/format mismatch or stream/context issue.")
        sys.exit(0)
    else:
        print("\nConclusion: Both paths failed; installation or runtime environment is likely broken.")
        sys.exit(3)

if __name__ == "__main__":
    main()
