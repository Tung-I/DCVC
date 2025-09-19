import argparse, io, json, os
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from src.utils.common import create_folder
from src.utils.stream_helper import SPSHelper, write_sps, write_ip
from src.models.model_utils import (
    DCVC_ALIGN,
    normalize_planes,
    pack_planes_to_rgb, unpack_rgb_to_planes,
    pack_density_to_rgb, unpack_density_from_rgb,
    dens_from01,
)
from src.models.dcvc_codec import DCVCImageCodec

"""
Usage:
python eval_dcvc_ckpt_compress.py --src_ckpt logs/out_triplane/flame_steak_image/fine_last_0.tar --dst_dir logs/out_triplane/flame_steak_image/dcvc_qp0 --qp 0 --save_json
python eval_dcvc_ckpt_compress.py \
    --src_ckpt logs/dynerf_sear_steak/dcvc_qp60_gradbpp/fine_last_0.tar \
    --packing_mode flatten --qp 60 --save_json
python eval_dcvc_ckpt_compress.py \
    --src_ckpt logs/nerf_synthetic/dcvc_qp24_flatten_flatten/fine_last_0.tar \
    --plane_packing_mode flatten --grid_packing_mode flatten --qp 24 --save_json
"""

# ===============================================================
# CLI
# ===============================================================
def parse_args():
    p = argparse.ArgumentParser("TriPlane → DCVC compress → save corrupted TriPlane (+density)")
    p.add_argument("--src_ckpt", required=True,
                   help="Path to original TriPlane checkpoint: fine_last_<frameid>.tar")
    p.add_argument("--frame_id", type=int, default=None,
                   help="(Optional) frame id for naming only; inferred from src_ckpt if omitted")
    p.add_argument("--dcvc_weights",
                   default="/home/tungichen_umass_edu/DCVC/checkpoints/cvpr2025_image.pth.tar",
                   help="Path to DCVC image model weights")
    p.add_argument("--qp", type=int, required=True, help="QP (0..63)")
    p.add_argument("--device", default="cuda", choices=["cuda","cpu"])

    # NEW: separate packing modes to mirror training wrapper
    p.add_argument("--plane_packing_mode", choices=["flatten", "mosaic", "flat4"], default="flatten",
                   help="Packing mode for feature planes (xy/xz/yz)")
    p.add_argument("--grid_packing_mode", choices=["flatten", "mosaic", "flat4"], default="flatten",
                   help="Packing mode for density grid")

    # Quantisation (exactly like training wrapper)
    p.add_argument("--quant_mode", choices=["global"], default="global",
                   help="Feature quant mode (training wrapper currently uses global)")
    p.add_argument("--global_range", type=float, nargs=2, default=[-20.0, 20.0])

    p.add_argument("--save_json", action="store_true", help="Emit stats json next to outputs")
    return p.parse_args()


# ===============================================================
# DCVC setup
# ===============================================================
@torch.no_grad()
def dcvc_setup(weights_path: str, device: str) -> DCVCImageCodec:
    codec = DCVCImageCodec(weight_path=weights_path, device=device)
    return codec


# ===============================================================
# Feature-planes (xy/xz/yz)
# ===============================================================
@torch.no_grad()
def compress_one_plane_with_dcvc(
    codec: DCVCImageCodec,
    plane: torch.Tensor,                 # [1,C,H,W] float32
    qp: int,
    plane_packing_mode: str,
    quant_mode: str,
    global_range: Tuple[float, float],
    device: str,
    out_bin_path: str,
) -> Dict:
    assert plane.dim() == 4 and plane.size(0) == 1
    x = plane.to(device=device, dtype=torch.float32)

    # 1) Quantize to [0,1]
    x01, c_min, scale = normalize_planes(x, mode=quant_mode, global_range=global_range)

    # 2) Pack (+pad) with *plane* mode (matches training wrapper)
    y_pad, orig_hw = pack_planes_to_rgb(x01, align=DCVC_ALIGN, mode=plane_packing_mode)  # [1,3,Hp,Wp]
    Hp, Wp = y_pad.shape[-2], y_pad.shape[-1]
    y_half = y_pad.to(device).to(torch.float16).contiguous(memory_format=torch.channels_last)

    # 3) DCVC encode
    enc = codec.compress(y_half, qp=qp)

    # 4) Bit accounting via SPS/IP container (same as before)
    out_dir = os.path.dirname(out_bin_path)
    create_folder(out_dir, True)
    buff = io.BytesIO()
    Hp_codec, Wp_codec = enc.get("shape", y_half.shape)[-2], enc.get("shape", y_half.shape)[-1]
    sps = {"sps_id": -1, "height": Hp_codec, "width": Wp_codec, "ec_part": 0, "use_ada_i": 0}
    sps_helper = SPSHelper()
    sps_id, sps_new = sps_helper.get_sps_id(sps); sps["sps_id"] = sps_id
    sps_bytes    = write_sps(buff, sps) if sps_new else 0
    stream_bytes = write_ip(buff, is_i_frame=True, sps_id=sps_id, qp=qp, bit_stream=enc["bit_stream"])
    with open(out_bin_path, "wb") as f:
        f.write(buff.getbuffer())
    total_bits = int((sps_bytes + stream_bytes) * 8)

    # 5) DCVC decode → clamp to padded (Hp,Wp)
    rec_pad = codec.decompress(enc)               # [1,3,Hp_codec,Wp_codec]
    x_hat   = rec_pad[..., :Hp, :Wp].to(torch.float32)

    # 6) Unpack with same mode, then de-normalize
    rec01 = unpack_rgb_to_planes(x_hat, x01.shape[1], orig_hw, mode=plane_packing_mode)
    recon = (rec01 * scale + c_min).to(torch.float32)

    # 7) Metrics
    bpp  = float(total_bits) / float(Hp * Wp)     # over padded canvas
    peak = float(global_range[1] - global_range[0])
    mse  = F.mse_loss(recon, x)
    psnr = float(10.0 * torch.log10((peak ** 2) / (mse + 1e-12)))

    return dict(recon=recon, bits=total_bits, bpp=bpp, psnr=psnr, orig_hw=tuple(orig_hw), padded_hw=(Hp, Wp))


# ===============================================================
# Density grid (1×1×Dy×Dx×Dz)
# ===============================================================
@torch.no_grad()
def compress_density_with_dcvc(
    codec: DCVCImageCodec,
    density_1x1: torch.Tensor,          # [1,1,Dy,Dx,Dz]
    qp: int,
    grid_packing_mode: str,
    device: str,
    out_bin_path: str,
) -> Dict:
    assert density_1x1.dim() == 5 and density_1x1.shape[:2] == (1,1)
    d_ref = density_1x1.to(device=device, dtype=torch.float32).contiguous()

    # 1) Pack (+pad) with *grid* mode (handles dens_to01 internally)
    y_pad, orig_hw = pack_density_to_rgb(d_ref, align=DCVC_ALIGN, mode=grid_packing_mode)  # [1,3,Hp,Wp], (H2,W2)
    Hp, Wp = y_pad.shape[-2], y_pad.shape[-1]
    y_half = y_pad.to(torch.float16).contiguous(memory_format=torch.channels_last)

    # 2) DCVC encode
    enc = codec.compress(y_half, qp=qp)

    # 3) Bit accounting (SPS/IP)
    out_dir = os.path.dirname(out_bin_path)
    create_folder(out_dir, True)
    buff = io.BytesIO()
    Hp_codec, Wp_codec = enc.get("shape", y_half.shape)[-2], enc.get("shape", y_half.shape)[-1]
    sps = {"sps_id": -1, "height": Hp_codec, "width": Wp_codec, "ec_part": 0, "use_ada_i": 0}
    sps_helper = SPSHelper()
    sps_id, sps_new = sps_helper.get_sps_id(sps); sps["sps_id"] = sps_id
    sps_bytes    = write_sps(buff, sps) if sps_new else 0
    stream_bytes = write_ip(buff, is_i_frame=True, sps_id=sps_id, qp=qp, bit_stream=enc["bit_stream"])
    with open(out_bin_path, "wb") as f:
        f.write(buff.getbuffer())
    total_bits = int((sps_bytes + stream_bytes) * 8)

    # 4) DCVC decode → clamp to padded → crop to pre-pad
    rec_pad = codec.decompress(enc)               # [1,3,Hp_codec,Wp_codec]
    x_hat   = rec_pad[..., :Hp, :Wp].to(torch.float32)
    H2, W2  = orig_hw
    x_hat_c = x_hat[..., :H2, :W2]

    # 5) Invert grid pack -> [1,1,Dy,Dx,Dz] in [0,1] -> raw density
    Dy, Dx, Dz = d_ref.shape[-3:]
    d01 = unpack_density_from_rgb(x_hat_c, Dy, Dx, Dz, orig_size=orig_hw, mode=grid_packing_mode)
    d_rec = dens_from01(d01)

    # 6) Metrics
    bpp  = float(total_bits) / float(Hp * Wp)
    mse  = F.mse_loss(d_rec, d_ref)
    psnr = float(10.0 * torch.log10((35.0 ** 2) / (mse + 1e-12)))

    return dict(recon=d_rec, bits=total_bits, bpp=bpp, psnr=psnr,
                canvas_hw=tuple(orig_hw), padded_hw=(Hp, Wp))


# ===============================================================
# TriPlane checkpoint I/O
# ===============================================================
def load_triplane_ckpt(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    meta  = {"model_kwargs": ckpt.get("model_kwargs", None)}
    return state, meta

def save_triplane_ckpt(dst_path: str, state: Dict, model_kwargs: Dict):
    ckpt = {"model_state_dict": state, "model_kwargs": model_kwargs}
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    torch.save(ckpt, dst_path)
    print(f"[save] Corrupted TriPlane saved → {dst_path}")


# ===============================================================
# main
# ===============================================================
def main():
    args   = parse_args()
    device = args.device if args.device == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")

    # Destination folders
    dst_dir   = os.path.join(os.path.dirname(args.src_ckpt), f"dcvc_qp{args.qp}")
    stream_dir= os.path.join(dst_dir, "streams")
    create_folder(dst_dir, True)
    create_folder(stream_dir, True)

    # Frame id
    frame_id = args.frame_id
    if frame_id is None:
        base = os.path.basename(args.src_ckpt)
        try:
            frame_id = int(base.split("_")[-1].split(".")[0])  # fine_last_<fid>.tar
        except Exception:
            frame_id = 0
    print(f"[info] Frame id = {frame_id}")

    # Load TriPlane tensors
    print(f"[load] {args.src_ckpt}")
    state, meta = load_triplane_ckpt(args.src_ckpt, device=device)

    def _ensure_4d(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0) if t.dim() == 3 else t

    xy = _ensure_4d(state["k0.xy_plane"]).to(torch.float32)
    xz = _ensure_4d(state["k0.xz_plane"]).to(torch.float32)
    yz = _ensure_4d(state["k0.yz_plane"]).to(torch.float32)
    density = state["density.grid"].to(torch.float32)   # [1,1,Dy,Dx,Dz]

    # DCVC model
    codec = dcvc_setup(args.dcvc_weights, device=device)

    # ========== Compress feature planes (xy/xz/yz) ==========
    planes_res = {}
    for ax, plane in [("xy", xy), ("xz", xz), ("yz", yz)]:
        out_bin = os.path.join(stream_dir, f"frame{frame_id:04d}_{ax}_qp{args.qp}.bin")
        print(f"[dcvc] Compressing plane {ax} → {out_bin}")
        res = compress_one_plane_with_dcvc(
            codec=codec,
            plane=plane,
            qp=args.qp,
            plane_packing_mode=args.plane_packing_mode,
            quant_mode=args.quant_mode,
            global_range=tuple(args.global_range),
            device=device,
            out_bin_path=out_bin,
        )
        planes_res[ax] = res

    # ========== Compress density grid ==========
    dens_bin = os.path.join(stream_dir, f"frame{frame_id:04d}_density_qp{args.qp}.bin")
    print(f"[dcvc] Compressing density → {dens_bin}")
    dens_res = compress_density_with_dcvc(
        codec=codec,
        density_1x1=density,
        qp=args.qp,
        grid_packing_mode=args.grid_packing_mode,
        device=device,
        out_bin_path=dens_bin,
    )

    # Rebuild checkpoint with reconstructed tensors (preserve dtypes)
    new_state = state.copy()
    new_state["k0.xy_plane"]  = planes_res["xy"]["recon"].to(state["k0.xy_plane"].dtype)
    new_state["k0.xz_plane"]  = planes_res["xz"]["recon"].to(state["k0.xz_plane"].dtype)
    new_state["k0.yz_plane"]  = planes_res["yz"]["recon"].to(state["k0.yz_plane"].dtype)
    new_state["density.grid"] = dens_res["recon"].to(state["density.grid"].dtype)

    dst_ckpt = os.path.join(dst_dir, f"fine_last_{frame_id}.tar")
    print(f"[save] Saving TriPlane checkpoint → {dst_ckpt}")
    save_triplane_ckpt(dst_ckpt, new_state, meta.get("model_kwargs", {}))

    # Stats
    total_bits = int(
        planes_res["xy"]["bits"] +
        planes_res["xz"]["bits"] +
        planes_res["yz"]["bits"] +
        dens_res["bits"]
    )
    stats = {
        "frame_id": frame_id,
        "qp": args.qp,
        "plane_packing_mode": args.plane_packing_mode,
        "grid_packing_mode": args.grid_packing_mode,
        "quant_mode": args.quant_mode,
        "global_range": args.global_range,
        "planes": {
            ax: {
                "bits": int(planes_res[ax]["bits"]),
                "bpp":  float(planes_res[ax]["bpp"]),
                "psnr": float(planes_res[ax]["psnr"]),
                "orig_hw": planes_res[ax]["orig_hw"],
                "padded_hw": planes_res[ax]["padded_hw"],
            } for ax in ("xy", "xz", "yz")
        },
        "density": {
            "bits": int(dens_res["bits"]),
            "bpp": float(dens_res["bpp"]),
            "psnr": float(dens_res["psnr"]),
            "canvas_hw": dens_res["canvas_hw"],
            "padded_hw": dens_res["padded_hw"],
        },
        "total_bits": total_bits,
        "dst_ckpt": dst_ckpt,
        "streams_dir": stream_dir,
    }
    print(json.dumps(stats, indent=2))
    if args.save_json:
        with open(os.path.join(dst_dir, f"compress_stats_f{frame_id:04d}_q{args.qp}.json"), "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
