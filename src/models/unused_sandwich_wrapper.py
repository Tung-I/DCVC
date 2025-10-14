class DCVCImageCodecWrapper(torch.nn.Module):
    """
    Differentiable wrapper over DCVC image codec.
    Uses separate packing modes for feature planes vs density grid.
    """
    def __init__(self, cfg_dcvc, device,
                 weight_path="/home/tungichen_umass_edu/DCVC/checkpoints/cvpr2025_image.pth.tar"):
        super().__init__()
        self.device = device
        self.weight_path = weight_path

        self.codec_wrapper = DCVCImageCodec(
            weight_path=self.weight_path,
            require_grad=False
        )

        # === NEW: two modes ===
        self.plane_mode = cfg_dcvc.plane_packing_mode  # "flatten" | "mosaic" | "flat4"
        self.grid_mode  = cfg_dcvc.grid_packing_mode   # "flatten" | "mosaic" | "flat4"

        self.qp          = cfg_dcvc.dcvc_qp
        self.quant_mode  = cfg_dcvc.quant_mode
        self.global_range= cfg_dcvc.global_range
        self.align       = DCVC_ALIGN
        self.in_channels = cfg_dcvc.in_channels
        self.cfg_dcvc    = cfg_dcvc
        self.use_amp     = cfg_dcvc.use_amp
        self.amp_dtype   = torch.float16  # keep TriPlane in fp32, run DCVC in fp16

        # ----- optional differentiable bpp estimator -----
        self.use_gradbpp_est = getattr(cfg_dcvc, "gradbpp", False)
        self.bpp_estimator = None
        if self.use_gradbpp_est:
            self.bpp_estimator = DMCI()
            self.bpp_estimator.load_state_dict(get_state_dict(self.weight_path))
            self.bpp_estimator.update(0.12)
            self.bpp_estimator = self.bpp_estimator.to(self.device)
            for p in self.bpp_estimator.parameters():
                p.requires_grad_(False)
            self.bpp_estimator.eval()
            self.bpp_estimator.half()

    # ---------------------- differentiable BPP estimator ----------------------

    def compute_bpp(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        image_tensor: [B,3,H,W] in [0,1]
        Returns differentiable scalar (bits per padded pixel).
        """
        assert self.bpp_estimator is not None, "BPP estimator not initialized"
        x_in = rgb2ycbcr(image_tensor.to(torch.float16))
        bpp = self.bpp_estimator.estimate_bpp(x_in, qp=self.qp)
        return bpp

    def estimate_bpp_only(self, planes_1xCHW: torch.Tensor) -> torch.Tensor:
        """
        planes_1xCHW: [1,C,H,W] raw-domain (same as forward()).
        Uses the *plane* packer/mode.
        """
        assert self.use_gradbpp_est and self.bpp_estimator is not None, "BPP estimator not enabled"
        # 1) normalize to [0,1]
        x01, _, _ = normalize_planes(planes_1xCHW, mode=self.quant_mode, global_range=self.global_range)
        # 2) pack (+ pad inside) with plane mode
        canv_pad, _ = pack_planes_to_rgb(x01, align=self.align, mode=self.plane_mode)
        # 3) differentiable bpp
        return self.compute_bpp(canv_pad)

    def estimate_bpp_density_only(self, density_1x1: torch.Tensor) -> torch.Tensor:
        """
        density_1x1: [1,1,Dy,Dx,Dz] raw-domain.
        Uses the *grid* packer/mode.
        """
        assert self.use_gradbpp_est and self.bpp_estimator is not None, "BPP estimator not enabled"
        # pack_density_to_rgb does mapping to [0,1] and padding internally
        canv_pad, _ = pack_density_to_rgb(density_1x1, align=self.align, mode=self.grid_mode)
        return self.compute_bpp(canv_pad)

    # --------------------------- tri-plane feature path ---------------------------

    def forward(self, frame: torch.Tensor):
        """
        Args:
            frame: [1, C, H, W] float on any device.
        Returns:
            recon [1,C,H,W] float32 on input device
            bpp   scalar tensor (bits per padded pixel)
            psnr  scalar tensor (raw-domain, global peak)
        """
        x = frame
        assert x.shape[1] == self.in_channels, f"expected C={self.in_channels}, got {x.shape[1]}"

        # Quantize feature planes to [0,1]
        x01, c_min, scale = normalize_planes(x, mode=self.quant_mode, global_range=self.global_range)

        # Pack (+pad) with the *plane* mode
        y_pad, orig_size = pack_planes_to_rgb(x01, align=self.align, mode=self.plane_mode)
        H2p, W2p = y_pad.shape[-2:]
        y_pad = y_pad.to(device=self.device)

        # Optimize layout
        try:
            y_pad = y_pad.contiguous(memory_format=torch.channels_last)
        except Exception:
            pass

        # Run DCVC coding
        y_half = y_pad.to(torch.float16)
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            enc_result = self.codec_wrapper.compress(y_half, self.qp)
            dec_result = self.codec_wrapper.decompress(enc_result)
            x_hat_half = dec_result[..., :H2p, :W2p]  # remove any internal padding

            bits = self.codec_wrapper.measure_size(enc_result, self.qp)
            bpp = torch.tensor(float(bits) / float(H2p * W2p), device=y_pad.device, dtype=torch.float32)

        x_hat32 = x_hat_half.to(torch.float32)

        # Unpack with the *same plane mode*
        rec01 = unpack_rgb_to_planes(x_hat32, x01.shape[1], orig_size, mode=self.plane_mode)

        # Rescale to original range
        recon = (rec01 * scale + c_min).to(torch.float32)

        if self.quant_mode == "global":
            peak = float(self.global_range[1] - self.global_range[0])  # e.g., 40.0
            mse_raw = F.mse_loss(recon, x.to(recon.dtype))
            plane_psnr = tetrirf_utils.mse2psnr_with_peak(mse_raw, peak=peak)
        else:
            raise NotImplementedError("mse2psnr_with_peak only implemented for global mode")

        return recon, bpp, plane_psnr

    # ------------------------------ density path ---------------------------------

    def forward_density(self, density_1x1: torch.Tensor):
        """
        Args:
            density_1x1: [1,1,Dy,Dx,Dz] float32 on any device.
        Returns:
            d_rec [1,1,Dy,Dx,Dz] float32 on input device
            bpp   scalar tensor (bits per padded pixel)
            psnr  scalar tensor (peak=35 for [-5,30] mapping)
        """
        assert density_1x1.dim() == 5 and density_1x1.shape[0] == 1 and density_1x1.shape[1] == 1

        # Pack (+pad) with the *grid* mode (handles dens_to01 + reshape internally)
        y_pad, orig_hw = pack_density_to_rgb(density_1x1, align=self.align, mode=self.grid_mode)  # [1,3,Hp,Wp], (H2,W2)
        Hp, Wp = y_pad.shape[-2:]
        y_pad = y_pad.to(self.device)

        # AMP + codec forward
        y_half = y_pad.to(torch.float16).contiguous(memory_format=torch.channels_last)
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            enc_result = self.codec_wrapper.compress(y_half, self.qp)
            dec_result = self.codec_wrapper.decompress(enc_result)
            x_hat_half = dec_result[..., :Hp, :Wp]  # keep padded shape (Hp,Wp)

            bits = self.codec_wrapper.measure_size(enc_result, self.qp)
            bpp = torch.tensor(float(bits) / float(Hp * Wp), device=y_pad.device, dtype=torch.float32)

        x_hat = x_hat_half.to(torch.float32)

        # Crop back to pre-pad size and invert grid packer
        H2, W2 = orig_hw
        x_hat_cropped = x_hat[..., :H2, :W2]
        # unpack_density_from_rgb returns density in [0,1] as [1,1,Dy,Dx,Dz]
        d01 = unpack_density_from_rgb(x_hat_cropped, *density_1x1.shape[-3:], orig_size=orig_hw, mode=self.grid_mode)
        d_rec = dens_from01(d01)  # map to raw [-5,30]

        # PSNR in raw density domain (peak = 35)
        mse = F.mse_loss(d_rec, density_1x1.to(d_rec.dtype))
        psnr = 10.0 * torch.log10((35.0 ** 2) / (mse + 1e-12))

        # If enabled, replace bpp with differentiable estimate from packed canvas
        if self.use_gradbpp_est and self.bpp_estimator is not None:
            bpp = self.compute_bpp(image_tensor=y_pad).to(y_pad.device, dtype=torch.float32)

        return d_rec, bpp, psnr


class DCVCSandwichImageCodecWrapper(torch.nn.Module):
    '''
    Autograd: 
        - Calling the codec in .eval() mode does not disable gradients; 
        - it only toggles internal behaviors like dropout/batchnorm. 
        - Avoid torch.no_grad() around the codec if you want gradients.
    '''
    def __init__(
            self, cfg_dcvc, device, 
            weight_path="/home/tungichen_umass_edu/DCVC/checkpoints/cvpr2025_image.pth.tar"):
        super().__init__()
        self.device = device
        self.weight_path = weight_path

        self.codec_wrapper = DCVCImageCodec(
            weight_path=self.weight_path,
            require_grad=False
            )

        self.qp = cfg_dcvc.dcvc_qp
        self.quant_mode = cfg_dcvc.quant_mode
        self.global_range = cfg_dcvc.global_range
        self.align = DCVC_ALIGN
        self.in_channels = cfg_dcvc.in_channels
        self.cfg_dcvc = cfg_dcvc
        self.use_amp = cfg_dcvc.use_amp
        self.amp_dtype = torch.float16  # keep TriPlane in fp32, only DCVC in fp16

        self.use_gradbpp_est = getattr(cfg_dcvc, "gradbpp", False)
        self.bpp_estimator = None
        if self.use_gradbpp_est:
            self.bpp_estimator = DMCI()  
            self.bpp_estimator.load_state_dict(get_state_dict(self.weight_path))
            self.bpp_estimator.update(0.12)
            self.bpp_estimator = self.bpp_estimator.to(self.device)
            # Freeze params so grads flow to input only
            for p in self.bpp_estimator.parameters():
                p.requires_grad_(False)
            self.bpp_estimator.eval()
            # self.bpp_estimator.half()

        self.use_sandwich = bool(getattr(cfg_dcvc, "use_sandwich", False))
        self.use_linearpack_per_axis = bool(getattr(cfg_dcvc, "use_linearpack_per_axis", False))
        if self.use_sandwich and self.use_linearpack_per_axis:
            in_ch = int(cfg_dcvc.in_channels)
            mid_dw = bool(getattr(cfg_dcvc, "lp_mid_dw", False))
            eps_pre = float(getattr(cfg_dcvc, "eps_pre", 1e-3))
            eps_post = float(getattr(cfg_dcvc, "eps_post", 0.0))  # not used by LinearPack (it clamps in post)

            # one LinearPack per plane
            self.lp_xy = LinearPack(in_ch=in_ch, mid_dw=mid_dw, eps=eps_pre).to(self.device)
            self.lp_xz = LinearPack(in_ch=in_ch, mid_dw=mid_dw, eps=eps_pre).to(self.device)
            self.lp_yz = LinearPack(in_ch=in_ch, mid_dw=mid_dw, eps=eps_pre).to(self.device)

            # expose a simple axis->module map
            self._lp = {"xy": self.lp_xy, "xz": self.lp_xz, "yz": self.lp_yz}

            # mark packing mode for logs
            self.packing_mode = "linearpack_per_axis"

            # axis-aware pack/unpack closures
            def _pack_axis(axis: str, x01: torch.Tensor, align=DCVC_ALIGN):
                x01_dev = x01.to(self.device)
                y = self._lp[axis].forward_pre(x01_dev)          # [1,3,H,W] in [0,1]
                H, W = y.shape[-2:]
                pad_h = (align - H % align) % align
                pad_w = (align - W % align) % align
                y_pad = F.pad(y, (0, pad_w, 0, pad_h), mode="replicate") if (pad_h or pad_w) else y
                return y_pad, (H, W)

            def _unpack_axis(axis: str, y_hat32: torch.Tensor, C: int, orig_size):
                H, W = orig_size
                y = y_hat32[..., :H, :W].to(self.device)
                x01_rec = self._lp[axis].forward_post(y)         # clamp(0,1) inside
                return x01_rec

            self.pack_axis = _pack_axis         # [axis, x01] -> (y_pad, (H,W))
            self.unpack_axis = _unpack_axis     # [axis, canvas, C, (H,W)] -> x01_rec
        elif self.use_sandwich:
            # ---- build sandwich modules ----
            in_ch          = int(cfg_dcvc.in_channels)
            unet_pre_base  = int(getattr(cfg_dcvc, "unet_pre_base", 32))
            unet_post_base = int(getattr(cfg_dcvc, "unet_post_base", 32))
            mlp_layers     = int(getattr(cfg_dcvc, "mlp_layers", 1))
            eps_pre        = float(getattr(cfg_dcvc, "eps", 1e-3))
            eps_post       = float(getattr(cfg_dcvc, "eps_post", 0.0))   # often 0.0 (optional bound on output)

            # modules live on the same device as the codec wrapper
            self.pre_unet   = SmallUNet(in_ch=in_ch, out_ch=3,  base=unet_pre_base).to(self.device)
            self.pre_mlp    = create_mlp(in_ch, 3, mlp_layers).to(self.device)
            self.bound_pre  = BoundedProjector(3, eps=eps_pre).to(self.device)

            self.post_unet  = SmallUNet(in_ch=3,  out_ch=in_ch, base=unet_post_base).to(self.device)
            self.post_mlp   = create_mlp(3, in_ch, mlp_layers).to(self.device)
            self.bound_post = BoundedProjector(in_ch, eps=eps_post).to(self.device) if eps_post > 0 else None

            # ---- swap pack/unpack with closures that keep your old call signature ----
            # old signature: pack_fn(x01, align=DCVC_ALIGN, mode=...)
            def _pack_sandwich(x01, align=DCVC_ALIGN, mode=None):
                # x01 may be on arbitrary device; modules are on self.device
                x01_dev = x01.to(self.device)
                y_pad, orig = sandwich_planes_to_rgb(
                    x01_dev,
                    pre_unet=self.pre_unet, pre_mlp=self.pre_mlp, bound_pre=self.bound_pre,
                    align=align
                )
                return y_pad, orig

            # old signature: unpack_fn(y_hat32, C, orig_size, mode=...)
            def _unpack_sandwich(y_hat32, C, orig_size, mode=None):
                y_hat_dev = y_hat32.to(self.device)
                rec01 = sandwich_rgb_to_planes(
                    y_hat_dev, orig_size,
                    post_unet=self.post_unet, post_mlp=self.post_mlp, post_bound=self.bound_post
                )
                return rec01
            self.packing_mode = "sandwich"
            self.pack_fn   = _pack_sandwich
            self.unpack_fn = _unpack_sandwich
        else:
            self.packing_mode = cfg_dcvc.packing_mode
            self.pack_fn   = pack_planes_to_rgb
            self.unpack_fn = unpack_rgb_to_planes

    def compute_bpp(
        self,
        image_tensor: torch.Tensor   # (B,3,H,W) in [0,1]
    ) -> torch.Tensor:
        """
        Returns differentiable bpp (scalar tensor) w.r.t. image_tensor.
        """
        assert self.bpp_estimator is not None, "BPP estimator not initialized"
        T, C, H, W = image_tensor.shape
        x_in = rgb2ycbcr(image_tensor)
        bpp = self.bpp_estimator.estimate_bpp(x_in, qp=self.qp)
        return bpp
    
    def estimate_bpp_only(self, planes_1xCHW: torch.Tensor) -> torch.Tensor:
        """
        planes_1xCHW: [1,C,H,W] float on any device, raw-domain (same input you pass to .forward())
        Returns differentiable bpp scalar (bits per padded pixel).
        """
        assert self.use_gradbpp_est and self.bpp_estimator is not None, "BPP estimator not enabled"
        dev = planes_1xCHW.device
        # 1) normalize to [0,1]
        x01, _, _ = normalize_planes(planes_1xCHW, mode=self.quant_mode, global_range=self.global_range)  # [1,C,H,W] on dev
        # 2) pack + pad to align (what codec sees)
        canv_pad, _ = self.pack_fn(x01, align=self.align, mode=self.packing_mode)  # [1,3,Hp,Wp]
        # 3) DMCI estimator (works on [0,1] RGB-like)
        return self.compute_bpp(canv_pad)  # differentiable scalar tensor

    def estimate_bpp_density_only(self, density_1x1: torch.Tensor) -> torch.Tensor:
        """
        density_1x1: [1,1,Dy,Dx,Dz] float on any device.
        Returns differentiable bpp scalar for density.
        """
        assert self.use_gradbpp_est and self.bpp_estimator is not None, "BPP estimator not enabled"
        dev = density_1x1.device
        # map to [0,1]
        d01 = dens_to01(density_1x1)                             # [1,1,Dy,Dx,Dz]
        Dy, Dx, Dz = d01.shape[2:]
        chw = d01.view(1, Dy, Dx, Dz)                            # [1,C,H,W] with C=Dy
        mono, (Hc, Wc) = tile_1xCHW(chw)                         # [Hc,Wc]
        canv = mono.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)    # [1,3,Hc,Wc]
        # pad to align
        pad_h = (self.align - Hc % self.align) % self.align
        pad_w = (self.align - Wc % self.align) % self.align
        canv_pad = F.pad(canv, (0, pad_w, 0, pad_h), mode="replicate")  # [1,3,Hp,Wp]
        # entropy estimator on [0,1]
        return self.compute_bpp(canv_pad)                        # differentiable scalar tensor

    def forward(self, frame: torch.Tensor):
        """
        Args:
            frame: [1, C, H, W] float on any device.
        Returns:
            recon [1,C,H,W] float32 on input device
            bpp   scalar tensor (bits per padded pixel)
            psnr  scalar tensor (raw-domain, global peak)
        """
        x = frame
        assert x.shape[1] == self.in_channels, f"expected C={self.in_channels}, got {x.shape[1]}"

        # Quantize feature planes to 0-1
        x01, c_min, scale = normalize_planes(
            x, mode=self.quant_mode, global_range=self.global_range)

        # Pack the quantized feature planes to 3 channels (and padding)
        y_pad, orig_size = self.pack_fn(x01, mode=self.packing_mode)
        H2p, W2p = y_pad.shape[-2:]
        y_pad = y_pad.to(device=self.device)

        # Optimize memory layout
        try:
            y_pad = y_pad.contiguous(memory_format=torch.channels_last)
        except Exception:
            pass

        # Run DCVC coding
        y_half = y_pad.to(torch.float16)
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):

                enc_result = self.codec_wrapper.compress(y_half, self.qp)
                bits = self.codec_wrapper.measure_size(enc_result, self.qp)
                dec_result  = self.codec_wrapper.decompress(enc_result)
                x_hat_half = dec_result[..., :H2p, :W2p]

                # mimic training bpp for consistency (bits / padded pixels)
                bits = self.codec_wrapper.measure_size(enc_result, self.qp)
                bpp = torch.tensor(float(bits) / float(H2p * W2p), device=y_pad.device, dtype=torch.float32)

        # Exit AMP, cast to fp32 for numerics and to match TriPlane later
        x_hat32 = x_hat_half.to(torch.float32)

        # Unpack the reconstructed feature planes and crop to ori size
        rec01 = self.unpack_fn(x_hat32, x01.shape[1], orig_size, mode=self.packing_mode)

        # Rescale to original range
        recon = (rec01 * scale + c_min).to(torch.float32) 

        if self.quant_mode == "global":
            peak = float(self.global_range[1] - self.global_range[0])  # = 40.0
            mse_raw = F.mse_loss(recon, x.to(recon.dtype))
            plane_psnr = tetrirf_utils.mse2psnr_with_peak(mse_raw, peak=peak)
        else:
            raise NotImplementedError("mse2psnr_with_peak only implemented for global mode")

        return recon, bpp, plane_psnr

    def forward_with_canvases(self, frame: torch.Tensor):
        """
        Same as forward(), but also returns:
        - y_pad : [1,3,Hp,Wp] pre-processor output after pad (in [0,1])
        - y_codec: [1,3,Hp,Wp] codec's decoded canvas (float32), same pad size
        - orig_size: (H2_orig, W2_orig) before pad
        """
        x = frame
        assert x.shape[1] == self.in_channels

        # 1) normalize planes
        x01, c_min, scale = normalize_planes(x, mode=self.quant_mode, global_range=self.global_range)

        # 2) learned (or classic) pack + pad
        y_pad, orig_size = self.pack_fn(x01, mode=self.packing_mode)     # [1,3,Hp,Wp]
        Hp, Wp = y_pad.shape[-2:]
        y_pad_dev = y_pad.to(device=self.device).contiguous(memory_format=torch.channels_last)

        # 3) codec fwd (AMP)
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            y_half = y_pad_dev.to(torch.float16)
            enc_result = self.codec_wrapper.compress(y_half, self.qp)
            dec_result = self.codec_wrapper.decompress(enc_result)
            y_codec_half = dec_result[..., :Hp, :Wp]
            bits = self.codec_wrapper.measure_size(enc_result, self.qp)
            bpp_detached = float(bits) / float(Hp * Wp)

        y_codec = y_codec_half.to(torch.float32)       # [1,3,Hp,Wp]

        # 4) unpack to planes (for recon/psnr)
        rec01 = self.unpack_fn(y_codec, x01.shape[1], orig_size, mode=self.packing_mode)
        recon = (rec01 * scale + c_min).to(torch.float32)

        if self.quant_mode == "global":
            peak = float(self.global_range[1] - self.global_range[0])
            mse_raw = F.mse_loss(recon, x.to(recon.dtype))
            plane_psnr = tetrirf_utils.mse2psnr_with_peak(mse_raw, peak=peak)
        else:
            raise NotImplementedError

        # bpp: allow differentiable variant if enabled (on y_pad)
        if self.use_gradbpp_est and self.bpp_estimator is not None:
            bpp = self.compute_bpp(image_tensor=y_pad.to(self.device, dtype=torch.float32))
        else:
            bpp = torch.tensor(bpp_detached, device=self.device, dtype=torch.float32)

        return recon, bpp, plane_psnr, y_pad.to(torch.float32), y_codec, orig_size

    def forward_with_canvases_axis(self, axis: str, frame: torch.Tensor):
        """
        Like forward(), but for a specific plane kind (xy/xz/yz),
        returning recon/bpp/psnr and both canvases (y_pad, y_codec) + orig_size.
        """
        assert self.use_sandwich and self.use_linearpack_per_axis, "axis-forward requires LinearPack per axis"
        x = frame
        assert x.shape[1] == self.in_channels

        # normalize to [0,1]
        x01, c_min, scale = normalize_planes(x, mode=self.quant_mode, global_range=self.global_range)

        # pack via axis module
        y_pad, orig_size = self.pack_axis(axis, x01, align=self.align)  # [1,3,Hp,Wp], (H,W)
        Hp, Wp = y_pad.shape[-2:]
        y_pad_dev = y_pad.to(self.device).contiguous(memory_format=torch.channels_last)

        # codec pass
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            y_half = y_pad_dev.to(torch.float16)
            enc_result = self.codec_wrapper.compress(y_half, self.qp)
            dec_result = self.codec_wrapper.decompress(enc_result)
            y_codec_half = dec_result[..., :Hp, :Wp]
            bits = self.codec_wrapper.measure_size(enc_result, self.qp)
            bpp_detached = float(bits) / float(Hp * Wp)

        y_codec = y_codec_half.to(torch.float32)

        # unpack via axis module
        rec01 = self.unpack_axis(axis, y_codec, x01.shape[1], orig_size)
        recon = (rec01 * scale + c_min).to(torch.float32)

        # psnr in raw domain
        if self.quant_mode == "global":
            peak = float(self.global_range[1] - self.global_range[0])
            mse_raw = F.mse_loss(recon, x.to(recon.dtype))
            plane_psnr = tetrirf_utils.mse2psnr_with_peak(mse_raw, peak=peak)
        else:
            raise NotImplementedError

        # bpp: differentiable estimator (on y_pad) if enabled
        if self.use_gradbpp_est and self.bpp_estimator is not None:
            bpp = self.compute_bpp(y_pad.to(self.device, dtype=torch.float32))
        else:
            bpp = torch.tensor(bpp_detached, device=self.device, dtype=torch.float32)

        return recon, bpp, plane_psnr, y_pad.to(torch.float32), y_codec, orig_size


    def forward_density(self, density_1x1: torch.Tensor):
        """
        Args:
            density_1x1: [1,1,Dy,Dx,Dz] float32 on any device
        Returns:
            d_rec [1,1,Dy,Dx,Dz] float32 on input device
            bpp   scalar tensor (bits per padded pixel)
            psnr  scalar tensor (peak=35 for [-5,30] mapping)
        """
        assert density_1x1.dim() == 5 and density_1x1.shape[0] == 1 and density_1x1.shape[1] == 1
        _, _, Dy, Dx, Dz = density_1x1.shape

        # Map to [0,1], pack as mono canvas, then 3ch by repetition
        d01 = dens_to01(density_1x1)                # [1,1,Dy,Dx,Dz]
        d01_chw = d01.view(1, Dy, Dx, Dz)                 # [1,C,H,W] with C=Dy,H=Dx,W=Dz
        mono, (Hc, Wc) = tile_1xCHW(d01_chw)        # [Hc,Wc]
        y = mono.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)# [1,3,Hc,Wc]

        # Align to DCVC stride
        pad_h = (self.align - Hc % self.align) % self.align
        pad_w = (self.align - Wc % self.align) % self.align
        y_pad = F.pad(y, (0, pad_w, 0, pad_h), mode="replicate").to(self.device)

        # AMP + codec forward
        y_half = y_pad.to(torch.float16).contiguous(memory_format=torch.channels_last)
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            enc_result = self.codec_wrapper.compress(y_half, self.qp)
            dec_result = self.codec_wrapper.decompress(enc_result)
            x_hat_half = dec_result[..., :Hc, :Wc]

            # mimic training bpp for consistency (bits / padded pixels)
            Hp, Wp = y_pad.shape[-2:]
            bits = self.codec_wrapper.measure_size(enc_result, self.qp)
            bpp = torch.tensor(float(bits) / float(Hp * Wp), device=y_pad.device, dtype=torch.float32)

        x_hat = x_hat_half.to(torch.float32)

        # Take one channel back to mono canvas (any of the three; they should match)
        mono_rec = x_hat[:, 0].squeeze(0)                 # [Hc,Wc] fp32

        # Untile → [1,C,H,W] → [1,1,Dy,Dx,Dz]
        d01_rec_chw = untile_to_1xCHW(mono_rec, Dy, Dx, Dz)   # [1,Dy,Dx,Dz]
        d_rec = dens_from01(d01_rec_chw).view(1, 1, Dy, Dx, Dz)

        # PSNR in raw density domain (peak = 35)
        mse = F.mse_loss(d_rec, density_1x1)
        psnr = 10.0 * torch.log10((35.0 ** 2) / (mse + 1e-12))

        # ---- NEW: switch bpp source ----
        if self.use_gradbpp_est and self.bpp_estimator is not None:
            # Use packed/padded canvas (what the codec sees) for an apples-to-apples bpp
            # y_pad is (B=1, 3, Hp, Wp) in [0,1].
            bpp_est = self.compute_bpp(image_tensor=y_pad)
            bpp = bpp_est.to(y_pad.device, dtype=torch.float32)

        return d_rec, bpp, psnr
    
    def sandwich_parameters(self):
        if self.use_sandwich and self.use_linearpack_per_axis:
            S = []
            for nm in ("lp_xy", "lp_xz", "lp_yz"):
                if hasattr(self, nm):
                    S += list(getattr(self, nm).parameters())
            return S
        # else: keep old path
        S = []
        for name in ("pre_unet","pre_mlp","bound_pre","post_unet","post_mlp","bound_post"):
            if hasattr(self, name):
                S += list(getattr(self, name).parameters())
        return S

    def core_parameters(self):
        if self.use_sandwich and self.use_linearpack_per_axis:
            S_PREFIX = ("lp_xy","lp_xz","lp_yz")
        else:
            S_PREFIX = ("pre_unet","pre_mlp","bound_pre","post_unet","post_mlp","bound_post")
        for n, p in self.named_parameters():
            if n.startswith(S_PREFIX):
                continue
            yield p