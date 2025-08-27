#!/usr/bin/env python3
import argparse, io, json, os, math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

# --- DCVC + helpers (adjust import roots to your repo layout) ---
from src.models.image_model import DMCI
from src.utils.common import get_state_dict, create_folder
from src.utils.stream_helper import SPSHelper, write_sps, write_ip
from TeTriRF.lib.dcvc_wrapper import (
    pack_planes_to_rgb, unpack_rgb_to_planes, _normalize_planes, DCVC_ALIGN
)
from src.models.dcvc_codec import DCVCImageCodec

"""
Usage:
python compress_triplane_ckpt_dcvc.py --src_ckpt logs/out_triplane/flame_steak_image/fine_last_0.tar --dst_dir logs/out_triplane/flame_steak_image/dcvc_qp0 --qp 0 --save_json
"""

# ===============================================================
# CLI
# ===============================================================
def parse_args():
    p = argparse.ArgumentParser("TriPlane → DCVC compress → save corrupted TriPlane (+density)")
    p.add_argument("--src_ckpt", required=True,
                   help="Path to original TriPlane checkpoint: fine_last_<frameid>.tar")
    p.add_argument("--dst_dir", required=True,
                   help="Directory to write corrupted TriPlane checkpoint + streams")
    p.add_argument("--frame_id", type=int, default=None,
                   help="(Optional) frame id for naming only; inferred from src_ckpt if omitted")
    p.add_argument("--dcvc_weights", default='/home/tungichen_umass_edu/DCVC/checkpoints/cvpr2025_image.pth.tar',
                   help="Path to DCVC image model weights (e.g., cvpr2025_image.pth.tar)")
    p.add_argument("--qp", type=int, required=True, help="QP (0..63)")
    p.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    # feature planes packing/quantization (must match training)
    p.add_argument("--packing_mode", default="flatten", choices=["flatten","mosaic"])
    p.add_argument("--quant_mode", default="global", choices=["global","per_channel"])
    p.add_argument("--global_range", type=float, nargs=2, default=[-20.0, 20.0])
    p.add_argument("--force_zero_thres", type=float, default=0.12)
    p.add_argument("--save_json", action="store_true", help="Emit stats json next to outputs")
    return p.parse_args()

# ===============================================================
# DCVC setup
# ===============================================================
@torch.no_grad()
def dcvc_setup(weights_path: str, device: str, force_zero_thres: float = 0.12) -> DCVCImageCodec:
    return DCVCImageCodec(weight_path=weights_path, device=device)

# ===============================================================
# Feature-planes (xy/xz/yz) compression (unchanged)
# ===============================================================
@torch.no_grad()
def compress_one_plane_with_dcvc(
    codec: DCVCImageCodec,           # <— wrapper
    plane: torch.Tensor,             # (1,C,H,W), fp32
    qp: int,
    packing_mode: str,
    quant_mode: str,
    global_range: Tuple[float,float],
    device: str,
    out_bin_path: str
) -> Dict:
    assert plane.dim() == 4 and plane.size(0) == 1
    plane = plane.to(device=device, dtype=torch.float32)

    # 1) normalize → [0,1], pack → RGB-like, pad
    x01, c_min, scale = _normalize_planes(plane, mode=quant_mode, global_range=global_range)
    y_pad, orig_size = pack_planes_to_rgb(x01, mode=packing_mode)        # [1,3,H2,W2] in [0,1]
    H2p, W2p = y_pad.shape[-2], y_pad.shape[-1]

    # 2) encode with your wrapper (wrapper does RGB->YCbCr internally)
    enc = codec.compress(y_pad, qp=qp)                                   # returns dict with 'bit_stream' and 'shape'

    # 3) write SPS + IP (to match official logging)
    out_dir = os.path.dirname(out_bin_path)
    create_folder(out_dir, True)
    import io
    buff = io.BytesIO()
    Hp, Wp = enc['shape'][-2], enc['shape'][-1]
    sps = {'sps_id': -1, 'height': Hp, 'width': Wp, 'ec_part': 0, 'use_ada_i': 0}
    sps_helper = SPSHelper()
    sps_id, sps_new = sps_helper.get_sps_id(sps); sps['sps_id'] = sps_id
    sps_bytes   = write_sps(buff, sps) if sps_new else 0
    stream_bytes= write_ip(buff, is_i_frame=True, sps_id=sps_id, qp=qp, bit_stream=enc['bit_stream'])
    with open(out_bin_path, "wb") as f:
        f.write(buff.getbuffer())
    total_bits = int((sps_bytes + stream_bytes) * 8)

    # 4) decode via wrapper for exact “official” behavior
    rec_rgb_pad = codec.decompress(enc)                                  # [1,3,Hp,Wp], RGB
    rec_rgb     = rec_rgb_pad[..., :H2p, :W2p]                           # crop padding

    # 5) unpack + de-normalize
    rec01 = unpack_rgb_to_planes(rec_rgb.to(torch.float32), x01.shape[1], orig_size, mode=packing_mode)
    recon = (rec01 * scale + c_min).to(torch.float32)

    # 6) PSNR in raw domain
    if quant_mode == "global":
        peak = float(global_range[1] - global_range[0])
        mse = F.mse_loss(recon, plane)
        psnr = float(10.0 * torch.log10((peak ** 2) / (mse + 1e-12)))
    else:
        mse = F.mse_loss(recon, plane)
        psnr = float(10.0 * torch.log10(1.0 / (mse + 1e-12)))

    bpp = float(total_bits) / float(Hp * Wp)
    return dict(recon=recon, bits=total_bits, bpp=bpp, psnr=psnr)

@torch.no_grad()
def compress_density_with_dcvc(
    codec: DCVCImageCodec,           # <— wrapper
    density: torch.Tensor,           # [1,1,Dy,Dx,Dz], fp32
    qp: int,
    device: str,
    out_bin_path: str,
    *,
    act_shift: torch.Tensor | float | None = None,
    voxel_size_ratio: float | int = 1.0,
) -> Dict:

    assert density.dim() == 5 and density.size(0) == 1 and density.size(1) == 1
    density = density.to(device=device, dtype=torch.float32).contiguous()
    _, _, Dy, Dx, Dz = density.shape

    # foreground masking (same as your packer; if no act_shift present, skip)
    if act_shift is not None:
        if not torch.is_tensor(act_shift):
            act_shift = torch.tensor(float(act_shift), device=density.device, dtype=density.dtype)
        d_for_mask = density + act_shift.view(1,1,1,1,1)
        alpha = 1.0 - (torch.exp(d_for_mask) + 1.0).pow(float(-voxel_size_ratio))
        alpha = F.max_pool3d(alpha, kernel_size=3, stride=1, padding=1)
        mask_bg = alpha < 1e-4
        d_ref = density.clone()
        d_ref[mask_bg] = -5.0
    else:
        d_ref = density

    # fixed global map to [0,1] (optionally keep the rounding if you want parity with PNG path;
    # removing the rounding generally gives you a bit more PSNR)
    def _to01(d):
        return (d.clamp(-5.0, 30.0) + 5.0) / 35.0
    def _from01(t01):
        return t01 * 35.0 - 5.0

    d01 = _to01(d_ref)                              # [1,1,Dy,Dx,Dz]
    d01_chw = d01.view(1, Dy, Dx, Dz)               # [1,C,H,W] with C=Dy, H=Dx, W=Dz

    # tiling (row-wise)
    def _compute_tiling_canvas(C, H, W):
        tiles_w = int(math.ceil(math.sqrt(C)))
        tiles_h = int(math.ceil(C / tiles_w))
        return tiles_h * H, tiles_w * W, tiles_h, tiles_w

    def _tile_1xCHW(feat):
        _, C, H, W = feat.shape
        Hc, Wc, tiles_h, tiles_w = _compute_tiling_canvas(C, H, W)
        canvas = feat.new_zeros(Hc, Wc)
        filled = 0
        for r in range(tiles_h):
            y = 0
            for c in range(tiles_w):
                if filled >= C: break
                canvas[r*H:(r+1)*H, y:y+W] = feat[0, filled]
                y += W; filled += 1
        return canvas, (Hc, Wc)

    mono, (Hc, Wc) = _tile_1xCHW(d01_chw)          # [Hc, Wc]

    # Make 3ch (repeat) → pad to DCVC stride
    y = mono.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)   # [1,3,Hc,Wc]
    pad_h = (DCVC_ALIGN - Hc % DCVC_ALIGN) % DCVC_ALIGN
    pad_w = (DCVC_ALIGN - Wc % DCVC_ALIGN) % DCVC_ALIGN
    y_pad = F.pad(y, (0, pad_w, 0, pad_h), mode="replicate")  # [1,3,Hp,Wp]
    Hp, Wp = y_pad.shape[-2], y_pad.shape[-1]

    # encode via wrapper
    enc = codec.compress(y_pad, qp=qp)             # wrapper converts to YCbCr internally

    # write SPS + IP
    out_dir = os.path.dirname(out_bin_path)
    create_folder(out_dir, True)
    buff = io.BytesIO()
    Hp, Wp = enc['shape'][-2], enc['shape'][-1]
    sps = {'sps_id': -1, 'height': Hp, 'width': Wp, 'ec_part': 0, 'use_ada_i': 0}
    sps_helper = SPSHelper()
    sps_id, sps_new = sps_helper.get_sps_id(sps); sps['sps_id'] = sps_id
    sps_bytes   = write_sps(buff, sps) if sps_new else 0
    stream_bytes= write_ip(buff, is_i_frame=True, sps_id=sps_id, qp=qp, bit_stream=enc['bit_stream'])
    with open(out_bin_path, "wb") as f:
        f.write(buff.getbuffer())
    total_bits = int((sps_bytes + stream_bytes) * 8)
    bpp = float(total_bits) / float(Hp * Wp)

    # decode via wrapper
    rec_pad = codec.decompress(enc)                # [1,3,Hp,Wp] in RGB
    rec     = rec_pad[..., :Hc, :Wc]           # crop
    mono_rec    = rec[:, 0].squeeze(0).to(torch.float32)  # [Hc, Wc]

    # untile back to [1,Dy,Dx,Dz]
    def _untile_to_1xCHW(canvas, C, H, W):
        Hc, Wc = canvas.shape[-2], canvas.shape[-1]
        tiles_h = Hc // H; tiles_w = Wc // W
        out = canvas.new_zeros(1, C, H, W)
        filled = 0
        for r in range(tiles_h):
            y = 0
            for c in range(tiles_w):
                if filled >= C: break
                out[0, filled] = canvas[r*H:(r+1)*H, y:y+W]
                y += W; filled += 1
        return out

    d01_rec_chw = _untile_to_1xCHW(mono_rec, Dy, Dx, Dz)
    d_rec       = _from01(d01_rec_chw).view(1,1,Dy,Dx,Dz)

    # PSNR vs the masked reference (this is what we encoded)
    mse  = F.mse_loss(d_rec, d_ref)
    psnr = float(10.0 * torch.log10((35.0 ** 2) / (mse + 1e-12)))

    return dict(recon=d_rec, bits=total_bits, bpp=bpp, psnr=psnr,
                canvas_hw=(Hc, Wc), padded_hw=(Hp, Wp))



# ===============================================================
# TriPlane checkpoint I/O
# ===============================================================
def load_triplane_ckpt(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict']
    meta = {'model_kwargs': ckpt.get('model_kwargs', None)}
    return state, meta

def save_triplane_ckpt(dst_path: str, state: Dict, model_kwargs: Dict):
    ckpt = {'model_state_dict': state, 'model_kwargs': model_kwargs}
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    torch.save(ckpt, dst_path)
    print(f"[save] Corrupted TriPlane saved → {dst_path}")

# ===============================================================
# main
# ===============================================================
def main():
    args = parse_args()
    device = args.device if args.device == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.dst_dir, exist_ok=True)

    # Infer frame id from filename if not given
    frame_id = args.frame_id
    if frame_id is None:
        base = os.path.basename(args.src_ckpt)
        try:
            frame_id = int(base.split('_')[-1].split('.')[0])  # fine_last_<fid>.tar
        except Exception:
            frame_id = 0
    print(f"[info] Frame id = {frame_id}")

    # Load TriPlane tensors
    print(f"[load] Loading TriPlane checkpoint → {args.src_ckpt}")
    state, meta = load_triplane_ckpt(args.src_ckpt, device=device)

    # Get planes (handle (C,H,W) vs (1,C,H,W))
    def _ensure_4d(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0) if t.dim() == 3 else t

    xy = _ensure_4d(state['k0.xy_plane']).to(torch.float32)
    xz = _ensure_4d(state['k0.xz_plane']).to(torch.float32)
    yz = _ensure_4d(state['k0.yz_plane']).to(torch.float32)

    # Density grid: [1,1,Dy,Dx,Dz]
    density = state['density.grid'].to(torch.float32)
    ##################
    # density_copy = density.clone()
    ##################
    assert density.dim() == 5 and density.size(0) == 1 and density.size(1) == 1, \
        f"Unexpected density shape: {tuple(density.shape)}"

    # Setup DCVC image model
    codec = dcvc_setup(args.dcvc_weights, device=device, force_zero_thres=args.force_zero_thres)

    # Folder for bitstreams
    stream_dir = os.path.join(args.dst_dir, "streams")
    create_folder(stream_dir, True)


    # ========== Compress feature planes ==========
    planes_res = {}
    for ax, plane in [('xy', xy), ('xz', xz), ('yz', yz)]:
        out_bin = os.path.join(stream_dir, f"frame{frame_id:04d}_{ax}_qp{args.qp}.bin")
        print(f"[dcvc] Compressing plane {ax} → {out_bin}")
        res = compress_one_plane_with_dcvc(
            codec=codec, plane=plane, qp=args.qp, packing_mode=args.packing_mode,
            quant_mode=args.quant_mode, global_range=tuple(args.global_range),
            device=device, out_bin_path=out_bin
        )
        planes_res[ax] = res

    # ========== Compress density grid ==========
    act_shift = state.get('act_shift', None)
    voxel_size_ratio = float(meta['model_kwargs'].get('voxel_size_ratio', 1.0))
    dens_bin = os.path.join(stream_dir, f"frame{frame_id:04d}_density_qp{args.qp}.bin")
    print(f"[dcvc] Compressing density → {dens_bin}")
    dens_res = compress_density_with_dcvc(
        codec=codec, density=density, qp=args.qp, device=device, out_bin_path=dens_bin,
        act_shift=act_shift, voxel_size_ratio=voxel_size_ratio
    )

    # Build a new state dict with reconstructed tensors (preserve original dtypes)
    new_state = state.copy()
    new_state['k0.xy_plane'] = planes_res['xy']['recon'].to(state['k0.xy_plane'].dtype)
    new_state['k0.xz_plane'] = planes_res['xz']['recon'].to(state['k0.xz_plane'].dtype)
    new_state['k0.yz_plane'] = planes_res['yz']['recon'].to(state['k0.yz_plane'].dtype)
    new_state['density.grid'] = dens_res['recon'].to(state['density.grid'].dtype)
    ####################
    # new_state['density.grid'] = density_copy
    ####################

    # Save checkpoint
    dst_ckpt = os.path.join(args.dst_dir, f"fine_last_{frame_id}.tar")
    print(f"[save] Saving TriPlane checkpoint → {dst_ckpt}")
    save_triplane_ckpt(dst_ckpt, new_state, meta['model_kwargs'])

    # Stats
    total_bits = int(
        planes_res['xy']['bits'] +
        planes_res['xz']['bits'] +
        planes_res['yz']['bits'] +
        dens_res['bits']
    )
    stats = {
        "frame_id": frame_id,
        "qp": args.qp,
        "packing_mode": args.packing_mode,
        "quant_mode": args.quant_mode,
        "global_range": args.global_range,
        "planes": {
            ax: dict(bits=int(planes_res[ax]['bits']),
                     bpp=float(planes_res[ax]['bpp']),
                     psnr=float(planes_res[ax]['psnr']))
            for ax in ('xy','xz','yz')
        },
        "density": {
            "bits": int(dens_res['bits']),
            "bpp": float(dens_res['bpp']),
            "psnr": float(dens_res['psnr']),
            "canvas_hw": dens_res['canvas_hw'],
            "padded_hw": dens_res['padded_hw'],
        },
        "total_bits": total_bits,
        "dst_ckpt": dst_ckpt,
        "streams_dir": stream_dir,
    }
    print(json.dumps(stats, indent=2))
    if args.save_json:
        with open(os.path.join(args.dst_dir, f"compress_stats_f{frame_id:04d}_q{args.qp}.json"), "w") as f:
            json.dump(stats, f, indent=2)

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
