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
from src.models.dcvc_codec import DCVCImageCodecWrapper
from TeTriRF.lib.unet import SmallUNet, create_mlp, BoundedProjector
from TeTriRF.lib import utils


DCVC_ALIGN = 32

class DCVCImageCodecInfer(torch.nn.Module):
    def __init__(
            self, 
            device='cuda', 
            qp=None, 
            quant_mode = "global", 
            in_channels: int = 12,
            global_range = (-20.0, 20.0), 
            packing_mode = "flatten",
        ):
        super().__init__()

        # self.codec_wrapper = DCVCImageCodecWrapper("/home/tungichen_umass_edu/DCVC/checkpoints/cvpr2025_image.pth.tar")
        self.i_frame_net = DMCI()
        i_state_dict = get_state_dict("checkpoints/cvpr2025_image.pth.tar")
        self.i_frame_net.load_state_dict(i_state_dict)
        self.i_frame_net.to(device)
        self.i_frame_net.eval()
        self.i_frame_net.update(0.12)
        
        self.pack_fn   = pack_planes_to_rgb
        self.unpack_fn = unpack_rgb_to_planes
        self.packing_mode = packing_mode
        self.qp = qp
        self.device = device
        self.quant_mode = quant_mode
        self.global_range = global_range
        self.in_channels = in_channels
        self.align = DCVC_ALIGN

    def forward(self, frame: torch.Tensor):
        """
        frame: [C,H,W] (float). Returns (recon_frame, bpp).
        """
        x = frame
        dtype_in = x.dtype
        if x.dim() == 3:
            x = x.unsqueeze(0)   # [1,C,H,W]
        assert x.shape[1] == self.in_channels, f"expected C={self.in_channels}, got {x.shape[1]}"

    
        # ------------------- Fallback: pack/unpack path -----------------
        # pack to 3ch canvas (mosaic/flatten), pad, DCVC, unpack
        x01, c_min, scale = _normalize_planes(x, mode=self.quant_mode, global_range=self.global_range)
        y_pad, orig_size = self.pack_fn(x01, mode=self.packing_mode)
        H2p, W2p = y_pad.shape[-2:]
        # y_pad = y_pad.to(device=self.device, dtype=torch.float16)
        y_pad = y_pad.to(device=self.device, dtype=torch.float32)

        # enc_result = self.codec_wrapper.compress(y_pad, self.qp)
        # bits = self.codec_wrapper.measure_size(enc_result, self.qp)
        # dec_result  = self.codec_wrapper.decompress(enc_result)

        result = dcvc_image_codec_forward(
                y_pad, self.qp, self.i_frame_net, device=self.device
            )
        dec_result = result['x_hat'][..., :H2p, :W2p].to(torch.float32)
        x_hat = dec_result
        bits = 0

        # print(f"dec_result: {dec_result.size()}") # torch.Size([1, 3, 1632, 2176])
   

        # result = dcvc_image_codec_forward(
        #     y_pad, self.qp, self.i_frame_net, device=self.device
        # )
        # x_hat = result['x_hat'][..., :H2p, :W2p].to(torch.float32)
        # bpp = result['bpp']


        diff = F.mse_loss(x_hat, y_pad).to(torch.float32)
        plane_psnr =  utils.mse2psnr(diff.detach())

        rec01 = self.unpack_fn(x_hat, x01.shape[1], orig_size, mode=self.packing_mode)
        recon = (rec01 * scale + c_min).to(torch.float32)

        return recon, bits, plane_psnr


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

        self.qp = qp 
        print(f"Using quantization parameter {self.qp} for DCVC codec")
        self.pack_fn   = pack_planes_to_rgb
        self.unpack_fn = unpack_rgb_to_planes



    def forward(self, seq: torch.Tensor):
        """
        seq: [T,C,H,W]  (float, arbitrary range). Returns (recon, avg_bpp).
        """
        with torch.no_grad():
            device   = next(self.i_frame_net.parameters()).device
            dtype_in = seq.dtype
            T, C, H, W = seq.shape

            # 1) normalize to [0,1]
            seq_n, c_min, scale = _normalize_planes(
                seq, mode=self.quant_mode, global_range=self.global_range
            )

            # 2) pack & pad to DCVC canvas
            y_pad, orig_size = self.pack_fn(seq_n)      # [T,3,H2p,W2p]
            H2_pad, W2_pad = y_pad.shape[-2:]
            y_pad = y_pad.to(device=device, dtype=torch.float16)

            # 3) DCVC temporal loop
            self.p_frame_net.clear_dpb()
            self.p_frame_net.set_curr_poc(0)

            recon_rgb, bpp_list = [], []
            for t in range(T):
                frame_in = y_pad[t:t+1]                       # [1,3,H2p,W2p]
                if t == 0:
                    out = self.i_frame_net(frame_in, self.qp)
                else:
                    out = self.p_frame_net(frame_in, self.qp)

                x_hat_crop = out["x_hat"][..., :H2_pad, :W2_pad]  # crop to padded size

                if t == 0:
                    self.p_frame_net.add_ref_frame(None, x_hat_crop)

                recon_rgb.append(x_hat_crop.to(torch.float32))
                bpp_list.append(out["bpp"])

            recon_rgb = torch.cat(recon_rgb, dim=0)           # [T,3,H2p,W2p]
            avg_bpp   = torch.stack(bpp_list).mean()

            # 4) unpack back to planes & de-normalize
            recon_n = self.unpack_fn(recon_rgb, C, orig_size) # [T,C,H,W]
            recon_seq = (recon_n * scale + c_min).to(dtype=dtype_in)

        return recon_seq, avg_bpp
    

class DCVCImageCodec(torch.nn.Module):
    def __init__(
            self, 
            device='cuda', 
            force_zero_thres=0.12, 
            qp=None, 
            quant_mode = "global", 
            sandwich: bool = False,
            in_channels: int = 12,
            unet_pre_base: int = 32,             # UNet width
            unet_post_base: int = 32,
            global_range = (-20.0, 20.0), 
            packing_mode = "flatten",
            mlp_layers = 2,
            freeze_dcvc_enc=True, 
            freeze_dcvc_dec=True,
            eps=1e-3, 
            convert_ycbcr=True
        ):
        super().__init__()

        self.i_frame_net = DMCI()
        i_state_dict = get_state_dict("checkpoints/cvpr2025_image.pth.tar")
        self.i_frame_net.load_state_dict(i_state_dict)
        self.i_frame_net.to(device)
        self.i_frame_net.eval()
        self.i_frame_net.update(force_zero_thres)
        # self.i_frame_net.half()

        if freeze_dcvc_enc:
            freeze_iframenet_enc(self.i_frame_net)
        if freeze_dcvc_dec:
            freeze_iframenet_dec(self.i_frame_net)
        self.freeze_dcvc_enc = freeze_dcvc_enc
        self.freeze_dcvc_dec = freeze_dcvc_dec

        self.sandwich = sandwich
        if self.sandwich:
            # encoder: C -> 3, same spatial size
            self.pre_processor = SmallUNet(in_ch=in_channels, out_ch=3, base=unet_pre_base)
            self.mlp_pre   = create_mlp(in_channels, 3, mlp_layers)
            self.bound_to_01    = BoundedProjector(3, eps=eps)
            # decoder: 3 -> C, same spatial size
            self.post_processor = SmallUNet(in_ch=3, out_ch=in_channels, base=unet_post_base)
            self.mlp_post   = create_mlp(3, in_channels, mlp_layers)
            self.convert_ycbcr = convert_ycbcr
        else:
            self.pack_fn   = pack_planes_to_rgb
            self.unpack_fn = unpack_rgb_to_planes
            self.packing_mode = packing_mode

        self.qp = qp
        self.device = device
        self.quant_mode = quant_mode
        self.global_range = global_range
        self.in_channels = in_channels
        self.align = DCVC_ALIGN

    def forward(self, frame: torch.Tensor):
        """
        frame: [C,H,W] (float). Returns (recon_frame, bpp).
        """
        x = frame
        dtype_in = x.dtype
        if x.dim() == 3:
            x = x.unsqueeze(0)   # [1,C,H,W]
        assert x.shape[1] == self.in_channels, f"expected C={self.in_channels}, got {x.shape[1]}"

        # # Normalize *before* encoder so UNets see a bounded domain
        # x01, c_min, scale = _normalize_planes(x, mode=self.quant_mode, global_range=self.global_range)

        if self.sandwich:
            # ------------------- Learnable encoder --> 3ch ------------------
            y3 = self.mlp_pre(x) + self.pre_processor(x)                       # [1,3,H,W] # value in range [0,1]

            # y01, c_min, scale = _normalize_planes(y3, mode=self.quant_mode, global_range=self.global_range)
            # Use differentialbe clamping to [0,1] instead of normalization
            y01 = self.bound_to_01(y3) 

            # Pad to codec alignment
            y_pad, orig_hw = _pad_to_align(y01, align=self.align)     # [1,3,H2p,W2p]
            # y_pad = y_pad.to(device=self.device, dtype=torch.float16)
            y_pad = y_pad.to(device=self.device, dtype=torch.float32)

            # ------------------- DCVC I-frame (frozen) ----------------------
        
            out = dcvc_image_codec_forward(
                y_pad, self.qp, self.i_frame_net, device=self.device, convert_ycbcr=self.convert_ycbcr
            )
            x_hat = out["x_hat"]
            bpp   = out["bpp"]
                # x_hat_cropped = _crop_from_align(x_hat, orig_hw).to(torch.float32)  # [1,3,H,W]

            # rec01 = (x_hat01 * scale + c_min).to(dtype=dtype_in)
            # print(f"x_hat shape: {x_hat.shape}, y_pad shape: {y_pad.shape}")
            diff = F.mse_loss(x_hat, y_pad.to(torch.float32))
            plane_psnr =  utils.mse2psnr(diff.detach())

            # ------------------- Learnable decoder -> C ---------------------
            x_hat_cropped = _crop_from_align(x_hat, orig_hw).to(torch.float32)
            recon = self.mlp_post(x_hat_cropped) + self.post_processor(x_hat_cropped)    
            # print(f"recon shape: {recon.shape}")               # [1,C,H,W]

        else:
            # ------------------- Fallback: pack/unpack path -----------------
            # pack to 3ch canvas (mosaic/flatten), pad, DCVC, unpack
            x01, c_min, scale = _normalize_planes(x, mode=self.quant_mode, global_range=self.global_range)
            y_pad, orig_size = self.pack_fn(x01, mode=self.packing_mode)
            H2p, W2p = y_pad.shape[-2:]
            # y_pad = y_pad.to(device=self.device, dtype=torch.float16)
            y_pad = y_pad.to(device=self.device, dtype=torch.float32)


            result = dcvc_image_codec_forward(
                y_pad, self.qp, self.i_frame_net, device=self.device
            )
            x_hat = result['x_hat'][..., :H2p, :W2p].to(torch.float32)
            bpp = result['bpp']


            diff = F.mse_loss(x_hat, y_pad).to(torch.float32)
            plane_psnr =  utils.mse2psnr(diff.detach())

            rec01 = self.unpack_fn(x_hat, x01.shape[1], orig_size, mode=self.packing_mode)
            recon = (rec01 * scale + c_min).to(torch.float32)

        return recon, bpp, plane_psnr


    # def infer(self)



def pack_planes_to_rgb(x, align=DCVC_ALIGN, mode: str = "flatten"):
    """
    x : [T, C, H, W], C ∈ {4, 12}
    -> y_pad : [T, 3, H2_pad, W2_pad]   (H2_pad, W2_pad are multiples of `align`)
       orig  : (H2_orig, W2_orig)
    modes:
      - "mosaic"  (default): original behavior
      - "flatten": for C=12, tile channels to a mono 3×4 grid [T,1,3H,4W], then repeat to RGB
                   for C=4, this reduces to the same 2×2 mono as mosaic.
    """
    T, C, H, W = x.shape
    if mode not in ("mosaic", "flatten"):
        raise ValueError(f"pack: unknown mode '{mode}'")

    if mode == "mosaic":
        if C == 4:
            mono = F.pixel_shuffle(x, 2)              # [T,1,2H,2W]
            y = mono.repeat(1, 3, 1, 1)               # [T,3,2H,2W]
        elif C == 12:
            # 3 groups of 4 channels each → pixel-shuffle each group → concat as RGB
            xg = x.view(T, 3, 4, H, W)
            tiles = [F.pixel_shuffle(xg[:, g], 2) for g in range(3)]  # each [T,1,2H,2W]
            y = torch.cat(tiles[::-1], dim=1)         # [T,3,2H,2W]  (B,G,R)→RGB-ish
        else:
            raise ValueError(f"pack: C must be 4 or 12 (got {C})")

    else:  # mode == "flatten"
        if C == 12:
            # Arrange 12 channels as a 3×4 tile grid (mono), then duplicate to 3 channels
            # [T,12,H,W] -> [T,1,3H,4W]
            mono = rearrange(x, 'T (r c) H W -> T 1 (r H) (c W)', r=3, c=4)
            y = mono.repeat(1, 3, 1, 1)               # [T,3,3H,4W]
        elif C == 4:
            # With 4 channels we can only form a 2×2 grid; same as pixel_shuffle→mono
            mono = F.pixel_shuffle(x, 2)              # [T,1,2H,2W]
            y = mono.repeat(1, 3, 1, 1)               # [T,3,2H,2W]
        else:
            raise ValueError(f"pack(flatten): C must be 4 or 12 (got {C})")

    # pad to multiples of `align`
    _, _, h2, w2 = y.shape
    pad_h = (align - h2 % align) % align
    pad_w = (align - w2 % align) % align
    y_pad = F.pad(y, (0, pad_w, 0, pad_h), mode='replicate')

    return y_pad, (h2, w2)   # keep (H2_orig, W2_orig) so we can crop on unpack


def unpack_rgb_to_planes(y_pad, C, orig_size, mode: str = "flatten"):
    """
    y_pad : [T,3,H2_pad,W2_pad]  (from pack_planes_to_rgb)
    C     : 4 or 12
    orig_size : (H2_orig, W2_orig)  (returned by pack)
    -> x : [T,C,H,W]
    """
    if mode not in ("mosaic", "flatten"):
        raise ValueError(f"unpack: unknown mode '{mode}'")

    H2, W2 = orig_size
    y = y_pad[..., :H2, :W2]                     # remove padding

    if mode == "mosaic":
        if C == 4:
            mono = y[:, :1]                      # any channel
            x = F.pixel_unshuffle(mono, 2)       # [T,4,H,W]
            return x
        elif C == 12:
            # invert the mosaic (split RGB, unshuffle, concat)
            b, g, r = y.split(1, dim=1)
            blocks = [F.pixel_unshuffle(ch, 2) for ch in (r, g, b)]
            return torch.cat(blocks, dim=1)      # [T,12,H,W]
        else:
            raise ValueError(f"unpack(mosaic): C must be 4 or 12 (got {C})")

    else:  # mode == "flatten"
        if C == 12:
            # y was mono repeated 3×; take first channel and invert 3×4 tiling
            mono = y[:, :1]                      # [T,1,3H,4W]
            # infer H,W from H2,W2 and the 3×4 tiling
            if H2 % 3 != 0 or W2 % 4 != 0:
                raise ValueError(f"unpack(flatten): orig_size {(H2,W2)} not divisible by (3,4)")
            H = H2 // 3
            W = W2 // 4
            x = rearrange(mono, 'T 1 (r H) (c W) -> T (r c) H W', r=3, c=4, H=H, W=W)  # [T,12,H,W]
            return x
        elif C == 4:
            # flatten with C=4 was 2×2 mono; invert as usual
            mono = y[:, :1]                      # [T,1,2H,2W]
            x = F.pixel_unshuffle(mono, 2)       # [T,4,H,W]
            return x
        else:
            raise ValueError(f"unpack(flatten): C must be 4 or 12 (got {C})")

def _normalize_planes(
        seq, 
        mode="global", 
        global_range=(-20.0, 20.0), 
        eps=1e-6):
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


# ------------------------------------------------------------------

def _pad_to_align(x, align=DCVC_ALIGN, mode="replicate"):
    """
    x : [B,3,H,W]  -> pad bottom/right so H,W are multiples of `align`.
    Returns y_pad, (H_orig, W_orig)
    """
    B, C, H, W = x.shape
    pad_h = (align - H % align) % align
    pad_w = (align - W % align) % align
    y = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return y, (H, W)


def _crop_from_align(x, orig_size):
    """
    x : [B,3,H_pad,W_pad] ; orig_size=(H_orig, W_orig)
    """
    H, W = orig_size
    return x[..., :H, :W]


# --- tiny togglers ------------------------------------------------------------
def _toggle_module(mod, trainable: bool):
    if mod is None: 
        return
    for p in mod.parameters():
        p.requires_grad_(trainable)
    # frozen side in eval(); trainable side in train()
    mod.train(trainable)

def _toggle_param(param: torch.nn.Parameter | None, trainable: bool):
    if param is None:
        return
    param.requires_grad_(trainable)

# --- what belongs to each "side" ---------------------------------------------
def _encoder_modules(dmci):
    # modules used to produce latents (y, z) & encoder quant scales
    return dict(
        mods = [
            dmci.enc,          # image -> y
            dmci.hyper_enc,    # y -> z
        ],
        params = [
            dmci.q_scale_enc,  # encoder quant scales
        ],
    )

def _decoder_modules(dmci):
    # modules used to decode y/z and reconstruct x_hat & decoder quant scales
    return dict(
        mods = [
            dmci.hyper_dec,                    # z_hat -> params
            dmci.y_prior_fusion,
            dmci.y_spatial_prior_reduction,
            dmci.y_spatial_prior_adaptor_1,
            dmci.y_spatial_prior_adaptor_2,
            dmci.y_spatial_prior_adaptor_3,
            dmci.y_spatial_prior,              # context / spatial prior
            dmci.dec,                          # y_hat -> x_hat
        ],
        params = [
            dmci.q_scale_dec,                  # decoder quant scales
        ],
    )

# # Optional: entropy/bit estimator pieces (often learnable); treat as "decoder"
# def _optional_entropy_modules(dmci):
#     mods = []
#     for name in ("bit_estimator_z", "gaussian_encoder", "entropy_coder"):
#         if hasattr(dmci, name):
#             m = getattr(dmci, name)
#             # some "coders" don’t have params; toggling is harmless
#             if isinstance(m, torch.nn.Module):
#                 mods.append(m)
#     return mods

# --- public helpers you can call ---------------------------------------------
def freeze_iframenet_enc(dmci: "DMCI"):
    """Freeze DCVC encoder side, enable decoder side for finetuning."""
    enc = _encoder_modules(dmci)
    dec = _decoder_modules(dmci)

    # freeze encoder
    for m in enc["mods"]:   _toggle_module(m, False)
    for p in enc["params"]: _toggle_param(p, False)

    # enable decoder
    for m in dec["mods"]:   _toggle_module(m, True)
    for p in dec["params"]: _toggle_param(p, True)

    # # (optional) also enable entropy model bits with decoder
    # for m in _optional_entropy_modules(dmci):
    #     _toggle_module(m, True)

def freeze_iframenet_dec(dmci: "DMCI"):
    """Freeze DCVC decoder side, enable encoder side for finetuning."""
    enc = _encoder_modules(dmci)
    dec = _decoder_modules(dmci)

    # enable encoder
    for m in enc["mods"]:   _toggle_module(m, True)
    for p in enc["params"]: _toggle_param(p, True)

    # freeze decoder
    for m in dec["mods"]:   _toggle_module(m, False)
    for p in dec["params"]: _toggle_param(p, False)

    # # (optional) entropy model with decoder → freeze it too
    # for m in _optional_entropy_modules(dmci):
    #     _toggle_module(m, False)

# --- collect trainable codec params (to add to your optimizer) ---------------
def collect_trainable_iframe_params(dmci: "DMCI"):
    params = []
    # for m in [
    #     dmci.enc, dmci.hyper_enc, dmci.hyper_dec, dmci.y_prior_fusion,
    #     dmci.y_spatial_prior_reduction, dmci.y_spatial_prior_adaptor_1,
    #     dmci.y_spatial_prior_adaptor_2, dmci.y_spatial_prior_adaptor_3,
    #     dmci.y_spatial_prior, dmci.dec,
    # ] + _optional_entropy_modules(dmci):
    for m in [
        dmci.enc, dmci.hyper_enc, dmci.hyper_dec, dmci.y_prior_fusion,
        dmci.y_spatial_prior_reduction, dmci.y_spatial_prior_adaptor_1,
        dmci.y_spatial_prior_adaptor_2, dmci.y_spatial_prior_adaptor_3,
        dmci.y_spatial_prior, dmci.dec,
    ]:
        if m is None: 
            continue
        for p in m.parameters():
            if p.requires_grad: 
                params.append(p)
    # add the per-QP scales if they’re trainable
    if dmci.q_scale_enc.requires_grad: params.append(dmci.q_scale_enc)
    if dmci.q_scale_dec.requires_grad: params.append(dmci.q_scale_dec)
    return params


# Function that returns the optimizable parameters of self.pre_processor, self.mlp_pre, self.post_processor, self.mlp_post

def collect_trainable_sandwich_params(model: "DCVCImageCodec"):
    params = []
    for m in [model.pre_processor, model.mlp_pre, model.post_processor, model.mlp_post]:
        if m is None:
            continue
        for p in m.parameters():
            if p.requires_grad:
                params.append(p)
    return params