import torch
import torch.nn.functional as F
from typing import Dict, Optional, Literal, List, Tuple

from src.models.video_model import DMC
from src.models.image_model import DMCI
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb


def dcvc_image_codec_forward(
    image_tensor: torch.Tensor,
    qp: int,
    model: DMCI,
    device: str = 'cuda',
    convert_ycbcr: bool = True,
):
    """
    image_tensor: (B,3,H,W) in [0,1].
    space="ycbcr": convert RGB->YCbCr for DMCI; back to RGB after.
    space="identity": feed as-is to DMCI and return as-is.
    """
    in_dtype = image_tensor.dtype
    image_tensor = image_tensor.to(device)
    T, C, H, W = image_tensor.shape

    if convert_ycbcr:
        inp = rgb2ycbcr(image_tensor)
        res = model.forward(inp, qp)
        x_hat = ycbcr2rgb(res["x_hat"]).clamp_(0, 1)
    else:
        res = model.forward(image_tensor, qp)
        x_hat = res["x_hat"].clamp_(0, 1)
    return {
        "x_hat": x_hat.to(in_dtype), 
        "bpp": res["bpp"], 
        "is_video": False}

# def dcvc_image_codec_forward(
#         image_tensor: torch.Tensor,
#         qp: int,
#         model: DMCI,
#         device: str = 'cuda'
# ) -> Dict:
#     """
#     DCVC Image Compression forward pass for training.

#     Args:
#         image_tensor: RGB image tensor of shape (B, 3, H, W) or video tensor (B, T, 3, H, W) in range [0, 1]
#         qp: Quantization parameter (0-63), lower = higher quality
#         model: Pre-initialized DMCI model
#         device: Device to run model on

#     Returns:
#         dict: Contains x_hat, bpp, and shape information
#     """
#     image_tensor = image_tensor.to(device)

#     # Handle video tensor by processing frame by frame
#     if image_tensor.dim() == 5:
#         # Video tensor: process frame by frame
#         B, T, C, H, W = image_tensor.shape
#         reconstructed_frames = []
#         total_bpp = 0

#         for t in range(T):
#             frame = image_tensor[:, t]  # (B, 3, H, W)

#             # Convert to YCbCr and forward pass
#             frame_ycbcr = rgb2ycbcr(frame)
#             result = model.forward(frame_ycbcr, qp)

#             # Convert back to RGB
#             x_hat_ycbcr = result["x_hat"]
#             x_hat_rgb = ycbcr2rgb(x_hat_ycbcr)
#             x_hat_rgb = torch.clamp(x_hat_rgb, 0, 1)

#             reconstructed_frames.append(x_hat_rgb)
#             total_bpp += result["bpp"]

#         # Stack frames back into video tensor
#         video_tensor = torch.stack(reconstructed_frames, dim=1)  # (B, T, C, H, W)

#         return {
#             'x_hat': video_tensor,
#             'bpp': total_bpp / T,  # Average bpp per frame
#             'bpp_total': total_bpp,
#             'shape': (B, T, C, H, W),
#             'is_video': True,
#             'z_bpp': result.get("z_bpp", 0),  # From last frame
#             'y_bpp': result.get("y_bpp", 0),  # From last frame
#         }
#     else:
#         # Image tensor: single forward pass
#         b, c, h, w = image_tensor.shape

#         # Convert to YCbCr and forward pass
#         image_ycbcr = rgb2ycbcr(image_tensor)
#         result = model.forward(image_ycbcr, qp)

#         # Convert back to RGB
#         x_hat_ycbcr = result["x_hat"]
#         x_hat_rgb = ycbcr2rgb(x_hat_ycbcr)
#         x_hat_rgb = torch.clamp(x_hat_rgb, 0, 1)

#         return {
#             'x_hat': x_hat_rgb,
#             'bpp': result["bpp"],
#             'shape': (b, c, h, w),
#             'is_video': False,
#             'z_bpp': result.get("z_bpp", 0),
#             'y_bpp': result.get("y_bpp", 0),
#         }


def dcvc_video_codec_forward(
        video_tensor: torch.Tensor,
        qp: int,
        i_frame_net: DMCI,
        p_frame_net: DMC,
        reset_interval: int = 32,
        intra_period: int = -1,
        device: str = 'cuda',
        half_precision: bool = False,
) -> Dict:
    """
    DCVC Video Compression forward pass for training.

    Args:
        video_tensor: Input video tensor [B, T, C, H, W] in range [0, 1]
        qp: Quantization parameter
        i_frame_net: Pre-initialized I-frame model (DMCI)
        p_frame_net: Pre-initialized P-frame model (DMC)
        reset_interval: Reset interval for adaptive features
        intra_period: Intra frame period (-1 for no periodic I-frames)
        device: Device to run on
        half_precision: Whether to use half precision

    Returns:
        Dictionary with x_hat, bpp, and frame information
    """
    # video_tensor shape: [B, T, C, H, W], B should be 1 for consistency
    assert video_tensor.shape[0] == 1, "Batch size must be 1"

    video_tensor = video_tensor.squeeze(0)  # [T, C, H, W]
    T, C, H, W = video_tensor.shape

    # Get padding size
    padding_r, padding_b = DMCI.get_padding_size(H, W, 16)

    # Determine if we should use two entropy coders
    use_two_entropy_coders = H * W > 1280 * 720
    i_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)
    p_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)

    # Initialize
    p_frame_net.set_curr_poc(0)
    p_frame_net.clear_dpb()
    reconstructed_frames = []
    frame_types = []
    total_bpp = 0
    total_z_bpp = 0
    total_y_bpp = 0
    index_map = [0, 1, 0, 2, 0, 2, 0, 2]

    last_qp = qp
    for frame_idx in range(T):
        # Convert frame to YCbCr
        x = video_tensor[frame_idx:frame_idx + 1].to(device)
        if half_precision:
            x = x.to(torch.float16)
        x = rgb2ycbcr(x)

        # Pad if necessary
        x_padded = F.pad(x, (0, padding_r, 0, padding_b), mode='replicate')

        # Determine if this should be an I-frame
        is_i_frame = frame_idx == 0 or (intra_period > 0 and frame_idx % intra_period == 0)

        if is_i_frame:
            # I-frame forward pass
            result = i_frame_net.forward(x_padded, qp)
            x_hat = result['x_hat']

            # Update DPB
            p_frame_net.clear_dpb()
            p_frame_net.add_ref_frame(None, x_hat)

            curr_qp = qp
            frame_types.append(0)  # I-frame type
        else:
            # P-frame forward pass
            fa_idx = index_map[frame_idx % 8]
            if reset_interval > 0 and frame_idx % reset_interval == 1:
                p_frame_net.prepare_feature_adaptor_i(last_qp)

            curr_qp = p_frame_net.shift_qp(qp, fa_idx)
            result = p_frame_net.forward(x_padded, curr_qp)
            x_hat = result['x_hat']
            last_qp = curr_qp
            frame_types.append(1)  # P-frame type

        # Crop to original size
        x_hat = x_hat[:, :, :H, :W]

        # Convert back to RGB
        x_hat_rgb = ycbcr2rgb(x_hat)
        x_hat_rgb = torch.clamp(x_hat_rgb, 0, 1)
        reconstructed_frames.append(x_hat_rgb)

        # Accumulate bpp
        total_bpp += result['bpp']
        total_z_bpp += result.get('z_bpp', 0)
        total_y_bpp += result.get('y_bpp', 0)

    # Stack frames: [T, 1, C, H, W] -> [1, T, C, H, W]
    reconstructed = torch.cat(reconstructed_frames, dim=0)
    reconstructed = reconstructed.unsqueeze(0)

    # Count I and P frames
    i_frame_num = sum(1 for ft in frame_types if ft == 0)
    p_frame_num = sum(1 for ft in frame_types if ft == 1)

    return {
        'x_hat': reconstructed,
        'bpp': total_bpp / T,  # Average bpp per frame
        'bpp_total': total_bpp,
        'z_bpp': total_z_bpp / T,
        'y_bpp': total_y_bpp / T,
        'frame_types': frame_types,
        'i_frame_num': i_frame_num,
        'p_frame_num': p_frame_num,
        'height': H,
        'width': W,
        'padding_r': padding_r,
        'padding_b': padding_b,
        'use_two_entropy_coders': use_two_entropy_coders
    }


def dcvc_sample_codec_forward(
        video_tensor: torch.Tensor,
        qp: int,
        mode: Literal["image", "video", "sample_image", "sample_video"],
        sample_frequency: int,
        i_frame_net: DMCI,
        p_frame_net: Optional[DMC] = None,
        device: str = 'cuda',
        reset_interval: int = 32,
        intra_period: int = -1,
        half_precision: bool = False,
        zero_fill: bool = True,
        forward_fill: bool = True,
) -> Dict:
    """
    DCVC Sample Codec forward pass for training with sampling support.

    Args:
        video_tensor: Input video tensor [B, T, C, H, W] in range [0, 1]
        qp: Quantization parameter
        mode: Compression mode
        sample_frequency: Frame sampling frequency (for sample modes)
        i_frame_net: Pre-initialized I-frame model (DMCI)
        p_frame_net: Pre-initialized P-frame model (DMC) - required for video modes
        device: Device to run on
        reset_interval: Reset interval for adaptive features
        intra_period: Intra frame period
        half_precision: Whether to use half precision
        zero_fill: Whether to return zero-filled video
        forward_fill: Whether to return forward-filled video

    Returns:
        Dictionary with reconstructed videos and bpp information
    """
    B, T, C, H, W = video_tensor.shape

    if mode == "image":
        # Standard image sequence compression
        result = dcvc_image_codec_forward(video_tensor, qp, i_frame_net, device)
        result['video'] = result['x_hat']
        if zero_fill:
            result['zero_filled'] = result['x_hat']
        if forward_fill:
            result['forward_filled'] = result['x_hat']
        return result

    elif mode == "video":
        # Standard video compression
        if p_frame_net is None:
            raise ValueError("p_frame_net is required for video mode")

        result = dcvc_video_codec_forward(
            video_tensor, qp, i_frame_net, p_frame_net,
            reset_interval, intra_period, device, half_precision
        )
        result['video'] = result['x_hat']
        if zero_fill:
            result['zero_filled'] = result['x_hat']
        if forward_fill:
            result['forward_filled'] = result['x_hat']
        return result

    elif mode == "sample_image":
        # Sample frames and compress individually
        sampled_indices = list(range(0, T, sample_frequency))
        sampled_frames = []
        total_bpp = 0
        total_z_bpp = 0
        total_y_bpp = 0

        for idx in sampled_indices:
            frame = video_tensor[:, idx:idx + 1]  # [B, 1, C, H, W]
            result = dcvc_image_codec_forward(frame, qp, i_frame_net, device)
            sampled_frames.append(result['x_hat'][:, 0])  # Remove time dimension
            total_bpp += result['bpp']
            total_z_bpp += result.get('z_bpp', 0)
            total_y_bpp += result.get('y_bpp', 0)

        # Create sampled video
        sampled_video = torch.stack(sampled_frames, dim=1)  # [B, T_sampled, C, H, W]

        result_dict = {
            'video': sampled_video,
            'x_hat': sampled_video,
            'sampled_indices': sampled_indices,
            'sample_frequency': sample_frequency,
            'bpp': total_bpp / len(sampled_indices),  # Average bpp per sampled frame
            'bpp_total': total_bpp,
            'z_bpp': total_z_bpp / len(sampled_indices),
            'y_bpp': total_y_bpp / len(sampled_indices),
            'num_sampled_frames': len(sampled_indices),
            'shape': (B, T, C, H, W),
            'mode': 'sample_image'
        }

    else:  # sample_video
        if p_frame_net is None:
            raise ValueError("p_frame_net is required for sample_video mode")

        # Extract sampled frames and compress as shorter video
        sampled_indices = list(range(0, T, sample_frequency))
        sampled_frames = []

        for idx in sampled_indices:
            sampled_frames.append(video_tensor[:, idx])

        sampled_video = torch.stack(sampled_frames, dim=1)  # [B, T_sampled, C, H, W]

        # Adjust reset interval for sampled video
        adjusted_reset_interval = reset_interval // sample_frequency

        result = dcvc_video_codec_forward(
            sampled_video, qp, i_frame_net, p_frame_net,
            adjusted_reset_interval, intra_period, device, half_precision
        )

        result_dict = {
            'video': result['x_hat'],
            'x_hat': result['x_hat'],
            'sampled_indices': sampled_indices,
            'sample_frequency': sample_frequency,
            'bpp': result['bpp'],  # Already averaged per frame
            'bpp_total': result['bpp_total'],
            'z_bpp': result.get('z_bpp', 0),
            'y_bpp': result.get('y_bpp', 0),
            'num_sampled_frames': len(sampled_indices),
            'shape': (B, T, C, H, W),
            'mode': 'sample_video',
            'frame_types': result.get('frame_types', []),
            'i_frame_num': result.get('i_frame_num', 0),
            'p_frame_num': result.get('p_frame_num', 0),
        }

    # Create zero-filled video if requested
    if zero_fill:
        zero_filled = torch.zeros((B, T, C, H, W), device=result_dict['video'].device)
        for i, idx in enumerate(sampled_indices):
            zero_filled[:, idx] = result_dict['video'][:, i]
        result_dict['zero_filled'] = zero_filled

    # Create forward-filled video if requested
    if forward_fill:
        forward_filled = torch.zeros((B, T, C, H, W), device=result_dict['video'].device)
        for i, idx in enumerate(sampled_indices):
            # Fill from current sample to next sample (or end)
            end_idx = sampled_indices[i + 1] if i + 1 < len(sampled_indices) else T
            forward_filled[:, idx:end_idx] = result_dict['video'][:, i:i + 1]
        result_dict['forward_filled'] = forward_filled

    return result_dict


# Example usage for training
if __name__ == "__main__":
    # Test parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, T, C, H, W = 1, 16, 3, 128, 128
    qp = 32
    sample_freq = 4

    # Initialize models once
    from dcvc.utils.common import get_state_dict

    # Load I-frame model
    i_frame_net = DMCI()
    i_frame_net.load_state_dict(get_state_dict("./checkpoint/cvpr2025_image.pth.tar"))
    i_frame_net.update(0.12)
    i_frame_net = i_frame_net.to(device)
    i_frame_net.train()

    # Load P-frame model
    p_frame_net = DMC()
    p_frame_net.load_state_dict(get_state_dict("./checkpoint/cvpr2025_video.pth.tar"))
    p_frame_net.update(None)
    p_frame_net = p_frame_net.to(device)
    p_frame_net.train()

    # Create synthetic video
    video = torch.rand(B, T, C, H, W).to(device)

    print("Testing Image Codec Forward:")
    result = dcvc_image_codec_forward(video, qp, i_frame_net, device)
    print(f"  BPP: {result['bpp']:.4f}")
    print(f"  Output shape: {result['x_hat'].shape}")

    print("\nTesting Video Codec Forward:")
    result = dcvc_video_codec_forward(video, qp, i_frame_net, p_frame_net, device=device)
    print(f"  BPP: {result['bpp']:.4f}")
    print(f"  I-frames: {result['i_frame_num']}, P-frames: {result['p_frame_num']}")

    print("\nTesting Sample Image Codec Forward:")
    result = dcvc_sample_codec_forward(
        video, qp, mode='sample_image',
        sample_frequency=sample_freq,
        i_frame_net=i_frame_net,
        device=device
    )
    print(f"  BPP: {result['bpp']:.4f}")
    print(f"  Sampled frames: {result['num_sampled_frames']}")
    print(f"  Zero-filled shape: {result.get('zero_filled', video).shape}")

    print("\nTesting Sample Video Codec Forward:")
    result = dcvc_sample_codec_forward(
        video, qp, mode='sample_video',
        sample_frequency=sample_freq,
        i_frame_net=i_frame_net,
        p_frame_net=p_frame_net,
        device=device
    )
    print(f"  BPP: {result['bpp']:.4f}")
    print(f"  Sampled frames: {result['num_sampled_frames']}")
    print(f"  Forward-filled shape: {result.get('forward_filled', video).shape}")