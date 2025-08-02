# lib/plane_codec_dcvc.py  (new file)
import torch
from einops import rearrange
from src.models.video_model import DMC
from src.models.image_model import DMCI
from src.utils.common import get_state_dict
import numpy as np
import torch.nn.functional as F
import math
from src.models.dcvc_codec_forward import dcvc_image_codec_forward
DCVC_ALIGN = 32

def pack_planes_to_rgb(x, align=DCVC_ALIGN):
    """
    x : [T, C, H, W]   C∈{4,12}
    ↦   y_pad : [T, 3, H2_pad, W2_pad]   with H2_pad, W2_pad  %  align == 0
        orig  : (H2_orig, W2_orig)
    """
    T, C, H, W = x.shape

    # --- build the 2×2 mosaic canvas ----------------------------------
    if C == 4:                                      # single plane
        mono = F.pixel_shuffle(x, 2)                # [T,1,2H,2W]
        y = mono.repeat(1, 3, 1, 1)                 # broadcast
    elif C == 12:                                   # concat xy/xz/yz
        x = x.view(T, 3, 4, H, W)                   # split groups
        tiles = [F.pixel_shuffle(x[:, g], 2) for g in range(3)]  # B,G,R?
        y = torch.cat(tiles[::-1], dim=1)           # BGR order
    else:
        raise ValueError(f"pack: C must be 4 or 12 (got {C})")

    # --- pad to multiples of <align> ----------------------------------
    _, _, h2, w2 = y.shape
    pad_h = (align - h2 % align) % align
    pad_w = (align - w2 % align) % align
    y_pad = F.pad(y, (0, pad_w, 0, pad_h), mode='replicate')          # right & bottom

    return y_pad, (h2, w2)


def unpack_rgb_to_planes(y_pad, C, orig_size):
    """
    y_pad : [T,3,H2_pad,W2_pad]   (as produced by pack_planes_to_rgb)
    C     : 4 or 12
    orig_size : (H2_orig, W2_orig)
    ↦  x  : [T,C,H,W]
    """
    H2, W2 = orig_size
    y = y_pad[..., :H2, :W2]                        # crop padding

    if C == 4:
        mono = y[:, :1]                             # any channel
        x = F.pixel_unshuffle(mono, 2)              # [T,4,H,W]
        return x
    elif C == 12:
        b, g, r = y.split(1, dim=1)
        blocks = [F.pixel_unshuffle(ch, 2) for ch in (r, g, b)]
        return torch.cat(blocks, dim=1)             # [T,12,H,W]
    else:
        raise ValueError(f"unpack: C must be 4 or 12 (got {C})")

def _normalize_planes(seq, mode="per_channel", global_range=(-20.0, 20.0), eps=1e-6):
    """
    Normalize tri-plane tensor to [0,1].
    seq : [T,C,H,W] (float32/float16)
    Returns:
      seq_n    : normalized to [0,1]
      c_min    : broadcastable min used
      scale    : broadcastable (max-min)
    """
    if mode == "per_channel":
        # per-channel min/max over T,H,W
        c_min = seq.amin(dim=(0, 2, 3), keepdim=True)             # [1,C,1,1]
        c_max = seq.amax(dim=(0, 2, 3), keepdim=True)
    elif mode == "global":
        lo, hi = global_range
        c_min = torch.as_tensor(lo, dtype=seq.dtype, device=seq.device).view(1, 1, 1, 1)
        c_max = torch.as_tensor(hi, dtype=seq.dtype, device=seq.device).view(1, 1, 1, 1)
    else:
        raise ValueError(f"Unknown quant_mode: {mode}")

    scale = (c_max - c_min).clamp_(eps)
    seq_n = ((seq - c_min) / scale).clamp_(0, 1)
    return seq_n, c_min, scale

class DCVCPlaneCodec(torch.nn.Module):
    """
    Differentiable encoder/decoder for a sequence of tri-plane feature maps.
    Expects input  [T, C, H, W]   (float32 in arbitrary range)
    Returns recon  [T, C, H, W]   and  bpp  (scalar)
    """
    def __init__(self, device='cuda', force_zero_thres=0.12, qp=0):
        super().__init__()

        self.i_frame_net = DMCI()
        i_state_dict = get_state_dict("checkpoints/cvpr2025_image.pth.tar")
        self.i_frame_net.load_state_dict(i_state_dict)
        self.i_frame_net.to(device)
        self.i_frame_net.eval()
        self.i_frame_net.update(force_zero_thres)
        self.i_frame_net.half()

        self.p_frame_net = DMC()
        p_state_dict = get_state_dict("checkpoints/cvpr2025_video.pth.tar")
        self.p_frame_net.load_state_dict(p_state_dict)
        self.p_frame_net.to(device)
        self.p_frame_net.eval()
        self.p_frame_net.update(force_zero_thres)
        self.p_frame_net.half()

        for p in self.i_frame_net.parameters():   
            p.requires_grad_(False)
        for p in self.p_frame_net.parameters():
            p.requires_grad_(False)

        # qp_i = []
        # for i in np.linspace(0, DMC.get_qp_num() - 1, num=4):
        #     qp_i.append(int(i+0.5)) # [0, 21, 42, 63]

        self.qp = qp 
        print(f"Using quantization parameter {self.qp} for DCVC codec")
        self.pack_fn   = pack_planes_to_rgb
        self.unpack_fn = unpack_rgb_to_planes



    def forward(self, seq):
        with torch.no_grad(): 
            device   = next(self.i_frame_net.parameters()).device
            dtype_in = seq.dtype
            T, C, H, W = seq.shape

            # 1. normalise
            c_min  = seq.amin(dim=(0, 2, 3), keepdim=True)
            c_max  = seq.amax(dim=(0, 2, 3), keepdim=True)
            scale  = (c_max - c_min).clamp_(1e-6)
            seq_n  = ((seq - c_min) / scale).clamp_(0, 1)

            # 2. pack & pad
            y_pad, orig_size = pack_planes_to_rgb(seq_n)      # H2_pad, W2_pad
            H2_pad, W2_pad = y_pad.shape[-2:]
            y_pad = y_pad.to(device=device, dtype=torch.float16)

            # 3. DCVC frame loop
            self.p_frame_net.clear_dpb()
            self.p_frame_net.set_curr_poc(0)

            recon_rgb, bpp_list = [], []
            for t in range(T):
                frame_in = y_pad[t:t+1]                       # [1,3,H2p,W2p]

                if t == 0:
                    out = self.i_frame_net(frame_in, self.qp)
                else:
                    out = self.p_frame_net(frame_in, self.qp)

                # --- crop decoded frame to encoder's pad window -------------
                x_hat_crop = out["x_hat"][..., :H2_pad, :W2_pad]

                if t == 0:                                    # DPB reference
                    self.p_frame_net.add_ref_frame(None, x_hat_crop)

                recon_rgb.append(x_hat_crop.to(torch.float32))
                bpp_list.append(out["bpp"])

            recon_rgb = torch.cat(recon_rgb, dim=0)           # [T,3,H2p,W2p]
            avg_bpp   = torch.stack(bpp_list).mean()

            # 4. unpack  →  feature planes
            recon_n = unpack_rgb_to_planes(recon_rgb, C, orig_size)  # [T,C,H,W]

            # 5. denormalise
            recon_seq = (recon_n * scale + c_min).to(dtype=dtype_in)
        return recon_seq, avg_bpp
    

class DCVCImageCodec(torch.nn.Module):
    def __init__(self, device='cuda', force_zero_thres=0.12, qp=0):
        super().__init__()

        self.i_frame_net = DMCI()
        i_state_dict = get_state_dict("checkpoints/cvpr2025_image.pth.tar")
        self.i_frame_net.load_state_dict(i_state_dict)
        self.i_frame_net.to(device)
        self.i_frame_net.eval()
        self.i_frame_net.update(force_zero_thres)
        # self.i_frame_net.half()

        for p in self.i_frame_net.parameters():   
            p.requires_grad_(False)

        self.pack_fn   = pack_planes_to_rgb
        self.unpack_fn = unpack_rgb_to_planes
        self.qp = qp
        self.device = device

    def forward(self, seq):

        device   = next(self.i_frame_net.parameters()).device
        dtype_in = seq.dtype
        T, C, H, W = seq.shape

        # 1. normalise
        c_min  = seq.amin(dim=(0, 2, 3), keepdim=True)
        c_max  = seq.amax(dim=(0, 2, 3), keepdim=True)
        scale  = (c_max - c_min).clamp_(1e-6)
        seq_n  = ((seq - c_min) / scale).clamp_(0, 1)

        # 2. pack & pad
        y_pad, orig_size = pack_planes_to_rgb(seq_n)      # H2_pad, W2_pad
        H2_pad, W2_pad = y_pad.shape[-2:]
        y_pad = y_pad.to(device=device, dtype=torch.float16)


        result = dcvc_image_codec_forward(
            y_pad, self.qp, self.i_frame_net, device=self.device
        )
        recon_rgb = result['x_hat']
        bpp = result['bpp']

        recon_n = unpack_rgb_to_planes(recon_rgb, C, orig_size)  # [T,C,H,W]

        # 5. denormalise
        recon = (recon_n * scale + c_min).to(dtype=dtype_in)

        return recon, bpp

    # def infer(self)
