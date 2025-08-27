import torch
import io
import os
import matplotlib.pyplot as plt
import torchvision

from typing import Dict, Optional, Literal, List, Tuple
from fractions import Fraction

# Import required modules from official DCVC
from src.models.video_model import DMC
from src.models.image_model import DMCI
from src.layers.cuda_inference import replicate_pad
from src.utils.common import get_state_dict
from src.utils.stream_helper import SPSHelper, NalType, write_sps, read_header, \
    read_sps_remaining, read_ip_remaining, write_ip
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb, yuv_444_to_420

class DCVCImageCodec:
    """DCVC Image Compression Codec for tensor inputs."""

    def __init__(self, weight_path="./checkpoint/cvpr2025_image.pth.tar", device='cuda'):
        """
        Initialize the DCVC codec.

        Args:
            weight_path (str): Path to model weights
            device (str): Device to run model on ('cuda' or 'cpu')
        """
        self.device = device

        # Load model
        self.model = DMCI()
        self.model.load_state_dict(get_state_dict(weight_path))
        self.model.update(0.12)
        # self.model(None)
        self.model = self.model.to(device)
        self.model.eval()


        print(f"Loaded DCVC image model from: {weight_path}")

    def measure_size(self, encoded_dict, qp, sps=None, is_i_frame=True):
        """
        Measure size of encoded data in bits using DCVC's method.

        Args:
            encoded_dict: Dictionary containing 'bit_stream' key
            qp: Quantization parameter
            sps: SPS dictionary (ignored, always set to None internally)
            is_i_frame: Whether this is an I-frame (default True for image codec)

        Returns:
            int: Size in bits
        """
        output_buff = io.BytesIO()

        # Ignore sps in size measurement
        sps = None

        # Get bit stream from encoded dict
        bit_stream = encoded_dict['bit_stream']

        # Write I/P frame data
        stream_bytes = write_ip(output_buff, is_i_frame=0 if is_i_frame else 1,
                                sps_id=-1, qp=qp, bit_stream=bit_stream)

        # Calculate total bits
        bits = stream_bytes * 8
        output_buff.close()

        return bits

    def compress(self, image_tensor, qp=32):
        """
        Compress an image tensor or video tensor frame by frame.

        Args:
            image_tensor (torch.Tensor): RGB image tensor of shape (B, 3, H, W) or video tensor (B, T, 3, H, W) in range [0, 1]
            qp (int): Quantization parameter (0-63), lower = higher quality

        Returns:
            dict: Compressed data containing bit_stream(s), sps, qp, shape, and size information
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)

            # Detect if input is video (5D) or image (4D)
            if image_tensor.dim() == 5:
                # Video tensor: process frame by frame
                B, T, C, H, W = image_tensor.shape
                compressed_frames = []
                encoded_dicts = []
                total_bits = 0

                # Create SPS once for all frames (kept for compatibility)
                sps = {
                    'sps_id': -1,
                    'height': H,
                    'width': W,
                    'ec_part': 0,
                    'use_ada_i': 0,
                }

                for t in range(T):
                    frame = image_tensor[:, t]  # (B, 3, H, W)

                    # Convert and compress
                    frame_ycbcr = rgb2ycbcr(frame)
                    encoded = self.model.compress(frame_ycbcr, qp)
                    compressed_frames.append(encoded['bit_stream'])
                    encoded_dicts.append(encoded)

                    # Measure size (sps ignored in measurement)
                    frame_bits = self.measure_size(encoded, qp)
                    total_bits += frame_bits

                return {
                    'bit_streams': compressed_frames,  # List of bit streams
                    'encoded_dicts': encoded_dicts,  # List of full encoded dicts
                    'sps': sps,
                    'qp': qp,
                    'shape': (B, T, C, H, W),
                    'is_video': True,
                    'total_bits': total_bits,
                    'bits_per_frame': total_bits / T
                }
            else:
                # Image tensor: original behavior
                b, c, h, w = image_tensor.shape
                # print(f"Compressing image of shape: {image_tensor.shape}")
                # print(f"qp: {qp}")
                # raise Exception

                # Convert and compress
                image_ycbcr = rgb2ycbcr(image_tensor)
                encoded = self.model.compress(image_ycbcr, qp)

                # Measure size
                total_bits = self.measure_size(encoded, qp)

                return {
                    'bit_stream': encoded['bit_stream'],
                    'encoded_dict': encoded,
                    'qp': qp,
                    'shape': (b, c, h, w),
                    'is_video': False,
                    'total_bits': total_bits
                }

    def decompress(self, compressed_data):
        """
        Decompress image or video from compressed data.

        Args:
            compressed_data (dict): Output from compress()

        Returns:
            torch.Tensor: Reconstructed RGB tensor in range [0, 1]
        """
        with torch.no_grad():
            if compressed_data.get('is_video', False):
                # Video decompression: process frame by frame
                B, T, C, H, W = compressed_data['shape']
                reconstructed_frames = []

                for bit_stream in compressed_data['bit_streams']:
                    # Create minimal SPS for decompression (required by DCVC)
                    sps = {
                        'sps_id': 0,
                        'height': H,
                        'width': W,
                        'ec_part': 0,
                        'use_ada_i': 0,
                    }
                    decoded = self.model.decompress(
                        bit_stream,
                        sps,
                        compressed_data['qp']
                    )
                    output_ycbcr = decoded["x_hat"]
                    output_rgb = ycbcr2rgb(output_ycbcr)
                    reconstructed_frames.append(output_rgb)

                # Stack frames back into video tensor
                video_tensor = torch.stack(reconstructed_frames, dim=1)  # (B, T, C, H, W)
                return torch.clamp(video_tensor, 0, 1)
            else:
                # Image decompression: original behavior
                b, c, h, w = compressed_data['shape']

                # Create minimal SPS for decompression (required by DCVC)
                sps = {
                    'sps_id': 0,
                    'height': h,
                    'width': w,
                    'ec_part': 0,
                    'use_ada_i': 0,
                }
                decoded = self.model.decompress(
                    compressed_data['bit_stream'],
                    sps,
                    compressed_data['qp']
                )
                output_ycbcr = decoded["x_hat"]
                output_rgb = ycbcr2rgb(output_ycbcr)
                return torch.clamp(output_rgb, 0, 1)


class DCVCVideoCodec:
    """Wrapper for official DCVC implementation."""

    def __init__(self, i_frame_weight_path: str,
                 p_frame_weight_path: str,
                 reset_interval: int = 32,
                 intra_period: int = -1,  # Add intra_period parameter
                 device: str = 'cuda',
                 half_precision: bool = False,
                 ):
        self.device = torch.device(device)
        self.half_precision = half_precision
        self.intra_period = intra_period  # Store intra period
        self.reset_interval = reset_interval
        self.i_frame_net = None
        self.p_frame_net = None

        # Check if model files exist
        if not os.path.exists(i_frame_weight_path):
            raise FileNotFoundError(f"I-frame model not found: {i_frame_weight_path}")
        if not os.path.exists(p_frame_weight_path):
            raise FileNotFoundError(f"P-frame model not found: {p_frame_weight_path}")

        # Initialize I-frame model
        print(f"Loading I-frame model from {i_frame_weight_path}")
        self.i_frame_net = DMCI()
        i_state_dict = get_state_dict(i_frame_weight_path)
        self.i_frame_net.load_state_dict(i_state_dict)
        self.i_frame_net = self.i_frame_net.to(self.device)
        self.i_frame_net.eval()

        # Update model (from official code)
        self.i_frame_net.update(None)  # force_zero_thres=None

        # Convert to half precision after update
        if self.half_precision:
            self.i_frame_net = self.i_frame_net.half()

        print(f"I-frame model loaded successfully, type: {type(self.i_frame_net)}")

        # Initialize P-frame model
        print(f"Loading P-frame model from {p_frame_weight_path}")
        self.p_frame_net = DMC()
        p_state_dict = get_state_dict(p_frame_weight_path)
        self.p_frame_net.load_state_dict(p_state_dict)
        self.p_frame_net = self.p_frame_net.to(self.device)
        self.p_frame_net.eval()

        # Update model (from official code)
        self.p_frame_net.update(None)  # force_zero_thres=None

        # Convert to half precision after update
        if self.half_precision:
            self.p_frame_net = self.p_frame_net.half()

        print(f"P-frame model loaded successfully, type: {type(self.p_frame_net)}")

    def compress(self, video_tensor: torch.Tensor, qp: int) -> Dict:
        """Compress video tensor using official DCVC."""
        # video_tensor shape: [B, T, C, H, W], B should be 1
        assert video_tensor.shape[0] == 1, "Batch size must be 1"

        video_tensor = video_tensor.squeeze(0)  # [T, C, H, W]
        T, C, H, W = video_tensor.shape

        # Get padding size
        padding_r, padding_b = DMCI.get_padding_size(H, W, 16)

        # Determine if we should use two entropy coders
        use_two_entropy_coders = H * W > 1280 * 720
        self.i_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)
        self.p_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)

        # Initialize
        self.p_frame_net.set_curr_poc(0)
        self.p_frame_net.clear_dpb()  # Clear DPB at start to free memory
        sps_helper = SPSHelper()
        output_buff = io.BytesIO()
        total_bits = 0
        compressed_frames = []
        frame_types = []  # Track frame types (0 for I-frame, 1 for P-frame)
        index_map = [0, 1, 0, 2, 0, 2, 0, 2]

        with torch.no_grad():
            last_qp = qp
            for frame_idx in range(T):
                # Clear cache periodically
                if frame_idx > 0 and frame_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Convert frame to YCbCr if needed
                x = video_tensor[frame_idx:frame_idx + 1].to(self.device)
                if self.half_precision:
                    x = x.to(torch.float16)
                # if x.max() > 1.0:  # Assume RGB in [0, 255]
                #     x = x / 255.0
                x = rgb2ycbcr(x)

                # Pad if necessary
                x_padded = replicate_pad(x, padding_b, padding_r)

                # Determine if this should be an I-frame
                is_i_frame = frame_idx == 0 or (self.intra_period > 0 and frame_idx % self.intra_period == 0)

                if is_i_frame:
                    # I-frame
                    sps = {
                        'sps_id': -1,
                        'height': H,
                        'width': W,
                        'ec_part': 1 if use_two_entropy_coders else 0,
                        'use_ada_i': 0,
                    }
                    encoded = self.i_frame_net.compress(x_padded, qp)

                    # Check if compress returns x_hat directly
                    if 'x_hat' in encoded:
                        x_hat = encoded['x_hat']
                    else:
                        # Decode to get x_hat for DPB
                        x_hat = self.i_frame_net.decompress(encoded['bit_stream'], sps, qp)['x_hat']

                    self.p_frame_net.clear_dpb()
                    self.p_frame_net.add_ref_frame(None, x_hat)

                    # Clean up x_hat after adding to DPB
                    del x_hat

                    curr_qp = qp
                    frame_types.append(0)  # I-frame type
                else:
                    # P-frame
                    fa_idx = index_map[frame_idx % 8]
                    if self.reset_interval > 0 and frame_idx % self.reset_interval == 1:
                        use_ada_i = 1
                        self.p_frame_net.prepare_feature_adaptor_i(last_qp)
                    else:
                        use_ada_i = 0

                    curr_qp = self.p_frame_net.shift_qp(qp, fa_idx)
                    sps = {
                        'sps_id': -1,
                        'height': H,
                        'width': W,
                        'ec_part': 1 if use_two_entropy_coders else 0,
                        'use_ada_i': use_ada_i,
                    }
                    encoded = self.p_frame_net.compress(x_padded, curr_qp)
                    last_qp = curr_qp
                    frame_types.append(1)  # P-frame type

                # Clean up padded frame
                del x_padded, x

                # Write to stream
                sps_id, sps_new = sps_helper.get_sps_id(sps)
                sps['sps_id'] = sps_id
                sps_bytes = 0
                if sps_new:
                    sps_bytes = write_sps(output_buff, sps)
                stream_bytes = write_ip(output_buff, is_i_frame, sps_id, curr_qp, encoded['bit_stream'])

                frame_bits = (stream_bytes + sps_bytes) * 8
                total_bits += frame_bits

                compressed_frames.append({
                    'bits': frame_bits,
                    'sps': sps.copy(),
                    'qp': curr_qp,
                    'is_i_frame': is_i_frame,
                    'frame_type': 0 if is_i_frame else 1  # Match official frame type encoding
                })

                # Clean up encoded dict
                del encoded

        # Get compressed bitstream
        compressed_data = output_buff.getvalue()
        output_buff.close()

        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Count I and P frames
        i_frame_num = sum(1 for ft in frame_types if ft == 0)
        p_frame_num = sum(1 for ft in frame_types if ft == 1)

        return {
            'compressed_data': compressed_data,
            'total_bits': total_bits,
            'compressed_frames': compressed_frames,
            'frame_types': frame_types,  # List of frame types (0=I, 1=P)
            'i_frame_num': i_frame_num,
            'p_frame_num': p_frame_num,
            'height': H,
            'width': W,
            'padding_r': padding_r,
            'padding_b': padding_b,
            'use_two_entropy_coders': use_two_entropy_coders
        }

    def decompress(self, compressed_dict: Dict) -> torch.Tensor:
        """Decompress using official DCVC."""
        compressed_data = compressed_dict['compressed_data']
        H = compressed_dict['height']
        W = compressed_dict['width']
        use_two_entropy_coders = compressed_dict['use_two_entropy_coders']

        self.i_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)
        self.p_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)

        # Initialize
        self.p_frame_net.set_curr_poc(0)
        self.p_frame_net.clear_dpb()
        sps_helper = SPSHelper()
        input_buff = io.BytesIO(compressed_data)
        reconstructed_frames = []

        with torch.no_grad():
            while True:
                try:
                    header = read_header(input_buff)
                except:
                    break  # End of stream

                # Handle SPS
                while header['nal_type'] == NalType.NAL_SPS:
                    sps = read_sps_remaining(input_buff, header['sps_id'])
                    sps_helper.add_sps_by_id(sps)
                    try:
                        header = read_header(input_buff)
                    except:
                        break
                    continue

                if header['nal_type'] not in [NalType.NAL_I, NalType.NAL_P]:
                    break

                sps_id = header['sps_id']
                sps = sps_helper.get_sps_by_id(sps_id)
                qp, bit_stream = read_ip_remaining(input_buff)

                # Decode frame
                if header['nal_type'] == NalType.NAL_I:
                    decoded = self.i_frame_net.decompress(bit_stream, sps, qp)
                    self.p_frame_net.clear_dpb()
                    self.p_frame_net.add_ref_frame(None, decoded['x_hat'])
                else:  # NAL_P
                    if sps['use_ada_i']:
                        self.p_frame_net.reset_ref_feature()
                    decoded = self.p_frame_net.decompress(bit_stream, sps, qp)

                # Crop to original size
                x_hat = decoded['x_hat'][:, :, :H, :W]

                # Convert back to RGB
                x_hat_rgb = ycbcr2rgb(x_hat)
                x_hat_rgb = torch.clamp(x_hat_rgb, 0, 1)

                reconstructed_frames.append(x_hat_rgb)

        input_buff.close()

        # Stack frames: [T, 1, C, H, W] -> [1, T, C, H, W]
        reconstructed = torch.cat(reconstructed_frames, dim=0)
        reconstructed = reconstructed.unsqueeze(0)

        return reconstructed


class DCVCSampleCodec:
    """DCVC Codec with sampling support for efficient video compression."""

    def __init__(
            self,
            mode: Literal["image", "video", "sample_image", "sample_video"] = "sample_image",
            sample_frequency: int = 8,
            i_frame_weight_path: str = "./checkpoint/cvpr2025_image.pth.tar",
            p_frame_weight_path: Optional[str] = "./checkpoint/cvpr2025_video.pth.tar",
            device: str = 'cuda',
            **kwargs
    ):
        """
        Initialize the DCVC Sample Codec.

        Args:
            mode: Compression mode
            sample_frequency: Frame sampling frequency (for sample modes)
            i_frame_weight_path: Path to I-frame/image model weights
            p_frame_weight_path: Path to P-frame model weights (for video modes)
            device: Device to run on
            **kwargs: Additional arguments passed to underlying codecs
        """
        self.mode = mode
        self.sample_frequency = sample_frequency
        self.device = device

        # Initialize appropriate codec based on mode
        if mode in ["image", "sample_image"]:
            self.codec = DCVCImageCodec(weight_path=i_frame_weight_path, device=device)
        else:  # video or sample_video
            # Extract all known parameters from kwargs to avoid duplicate argument errors
            reset_interval = kwargs.pop('reset_interval', 32)
            intra_period = kwargs.pop('intra_period', -1)
            half_precision = kwargs.pop('half_precision', False)

            if mode == "sample_video":
                # Adjust reset interval for sampled video
                reset_interval = reset_interval // sample_frequency

            self.codec = DCVCVideoCodec(
                i_frame_weight_path=i_frame_weight_path,
                p_frame_weight_path=p_frame_weight_path,
                device=device,
                reset_interval=reset_interval,
                intra_period=intra_period,
                half_precision=half_precision,
            )

    def compress(self, video_tensor: torch.Tensor, qp: int = 32) -> Dict:
        """
        Compress video tensor based on mode.

        Args:
            video_tensor: Input video tensor [B, T, C, H, W] in range [0, 1]
            qp: Quantization parameter

        Returns:
            Dictionary with compressed data and metadata
        """
        B, T, C, H, W = video_tensor.shape

        if self.mode == "image":
            # Standard image sequence compression
            return self.codec.compress(video_tensor, qp)

        elif self.mode == "video":
            # Standard video compression
            return self.codec.compress(video_tensor, qp)

        elif self.mode == "sample_image":
            # Sample frames and compress individually
            sampled_indices = list(range(0, T, self.sample_frequency))
            compressed_frames = []
            total_bits = 0

            for idx in sampled_indices:
                frame = video_tensor[:, idx:idx + 1]  # [B, 1, C, H, W]
                compressed = self.codec.compress(frame, qp)
                compressed_frames.append(compressed)
                total_bits += compressed['total_bits']

            return {
                'compressed_frames': compressed_frames,
                'sampled_indices': sampled_indices,
                'sample_frequency': self.sample_frequency,
                'total_bits': total_bits,  # Only counts compressed frames
                'num_sampled_frames': len(sampled_indices),
                'shape': (B, T, C, H, W),
                'qp': qp,
                'mode': 'sample_image'
            }

        else:  # sample_video
            # Extract sampled frames and compress as shorter video
            sampled_indices = list(range(0, T, self.sample_frequency))
            sampled_frames = []

            for idx in sampled_indices:
                sampled_frames.append(video_tensor[:, idx])

            sampled_video = torch.stack(sampled_frames, dim=1)  # [B, T_sampled, C, H, W]
            compressed = self.codec.compress(sampled_video, qp)

            return {
                'compressed_data': compressed,
                'sampled_indices': sampled_indices,
                'sample_frequency': self.sample_frequency,
                'total_bits': compressed['total_bits'],  # Bits from compressed sampled video
                'num_sampled_frames': len(sampled_indices),
                'shape': (B, T, C, H, W),
                'mode': 'sample_video'
            }

    def decompress(self, compressed_data: Dict, zero_fill: bool = True, forward_fill: bool = True) -> Dict[
        str, torch.Tensor]:
        """
        Decompress data and return reconstructed videos.

        Args:
            compressed_data: Compressed data from compress()
            zero_fill: Whether to return zero-filled video
            forward_fill: Whether to return forward-filled video

        Returns:
            Dictionary with:
                - 'video': Reconstructed/sampled video
                - 'zero_filled': Full video with zeros between samples (if enabled and applicable)
                - 'forward_filled': Full video with forward-filled samples (if enabled and applicable)
        """
        mode = compressed_data.get('mode', self.mode)
        result = {}

        if mode in ["image", "video"]:
            # Standard decompression
            reconstructed = self.codec.decompress(compressed_data)
            result['video'] = reconstructed
            if zero_fill:
                result['zero_filled'] = reconstructed
            if forward_fill:
                result['forward_filled'] = reconstructed
            return result

        # For sampling modes, we need the shape
        B, T, C, H, W = compressed_data['shape']

        if mode == "sample_image":
            # Decompress individual frames
            sampled_indices = compressed_data['sampled_indices']
            sampled_frames = []

            for compressed_frame in compressed_data['compressed_frames']:
                frame = self.codec.decompress(compressed_frame)  # [B, 1, C, H, W]
                sampled_frames.append(frame[:, 0])  # Remove time dimension

            # Create sampled video
            sampled_video = torch.stack(sampled_frames, dim=1)  # [B, T_sampled, C, H, W]

        else:  # sample_video
            # Decompress short video
            sampled_video = self.codec.decompress(compressed_data['compressed_data'])
            sampled_indices = compressed_data['sampled_indices']

        result['video'] = sampled_video

        # Create zero-filled video if requested
        if zero_fill:
            zero_filled = torch.zeros((B, T, C, H, W), device=sampled_video.device)
            for i, idx in enumerate(sampled_indices):
                zero_filled[:, idx] = sampled_video[:, i]
            result['zero_filled'] = zero_filled

        # Create forward-filled video if requested
        if forward_fill:
            forward_filled = torch.zeros((B, T, C, H, W), device=sampled_video.device)
            for i, idx in enumerate(sampled_indices):
                # Fill from current sample to next sample (or end)
                end_idx = sampled_indices[i + 1] if i + 1 < len(sampled_indices) else T
                forward_filled[:, idx:end_idx] = sampled_video[:, i:i + 1]
            result['forward_filled'] = forward_filled

        return result


# Visualization unit test
if __name__ == "__main__":
    # Test parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, T, C, H, W = 1, 16, 3, 128, 128
    qp = 0
    sample_freq = 4

    # Create synthetic video with distinct frames
    video = torch.zeros(B, T, C, H, W)
    for t in range(T):
        # Create a moving gradient pattern
        x_offset = t * 8  # Horizontal movement
        y_offset = t * 4  # Vertical movement

        # Create coordinate grids
        y_coords = torch.arange(H).float().unsqueeze(1).expand(H, W)
        x_coords = torch.arange(W).float().unsqueeze(0).expand(H, W)

        # Red channel: diagonal gradient that moves
        video[0, t, 0] = ((x_coords + x_offset) % W) / W

        # Green channel: circular pattern that expands
        center_y, center_x = H // 2, W // 2
        dist = torch.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
        video[0, t, 1] = 0.5 + 0.5 * torch.sin(dist / 10 - t * 0.5)

        # Blue channel: vertical bars that move
        video[0, t, 2] = 0.5 + 0.5 * torch.sin((x_coords + t * 10) * 0.1)

        # Add frame number as text-like pattern in corner
        if t < 10:
            video[0, t, :, 5:15, 5:15] = t / T

    video = video.clamp(0, 1).to(device)

    # Test all four modes
    modes = ["image", "video", "sample_image", "sample_video"]
    fig, axes = plt.subplots(len(modes), 6, figsize=(24, 16))

    # Define positions to visualize
    sampled_pos = 4  # This will be sampled (4 % 4 == 0)
    unsampled_pos = 6  # This won't be sampled (6 % 4 != 0)

    for mode_idx, mode in enumerate(modes):
        print(f"\nTesting mode: {mode}")

        # Initialize codec
        codec = DCVCSampleCodec(
            mode=mode,
            sample_frequency=sample_freq,
            device=device
        )

        # Compress
        compressed = codec.compress(video, qp)
        print(f"Compressed size: {compressed.get('total_bits', 0)} bits")
        if mode in ['sample_image', 'sample_video']:
            print(f"Sampled frames: {compressed.get('num_sampled_frames', 0)}")
            print(f"Sampled indices: {compressed.get('sampled_indices', [])}")

        # Decompress
        decompressed = codec.decompress(compressed, zero_fill=True, forward_fill=True)
        sampled = decompressed['video']
        zero_filled = decompressed.get('zero_filled', sampled)
        forward_filled = decompressed.get('forward_filled', sampled)

        # Column 0: Original at sampled position (t=4)
        axes[mode_idx, 0].imshow(video[0, sampled_pos].permute(1, 2, 0).cpu().clamp(0, 1))
        axes[mode_idx, 0].set_title(f"{mode}: Original (t={sampled_pos})")
        axes[mode_idx, 0].axis('off')

        # Column 1: Original at unsampled position (t=6)
        axes[mode_idx, 1].imshow(video[0, unsampled_pos].permute(1, 2, 0).cpu().clamp(0, 1))
        axes[mode_idx, 1].set_title(f"Original (t={unsampled_pos})")
        axes[mode_idx, 1].axis('off')

        # Column 2: Sampled/Reconstructed video info
        if mode in ["sample_image", "sample_video"]:
            # For sampling modes, show the compressed sampled frame at index 1 (corresponds to t=4)
            if sampled.shape[1] > 1:
                axes[mode_idx, 2].imshow(sampled[0, 1].permute(1, 2, 0).cpu().clamp(0, 1))
                axes[mode_idx, 2].set_title(f"Sampled (idx=1, t=4)")
            else:
                axes[mode_idx, 2].text(0.5, 0.5, 'No sample at idx 1', ha='center', va='center')
        else:
            # For non-sampling modes, show reconstructed
            axes[mode_idx, 2].imshow(sampled[0, sampled_pos].permute(1, 2, 0).cpu().clamp(0, 1))
            axes[mode_idx, 2].set_title(f"Reconstructed (t={sampled_pos})")
        axes[mode_idx, 2].axis('off')

        # Column 3: Zero-filled at sampled position (t=4)
        axes[mode_idx, 3].imshow(zero_filled[0, sampled_pos].permute(1, 2, 0).cpu().clamp(0, 1))
        axes[mode_idx, 3].set_title(f"Zero-filled (t={sampled_pos})")
        axes[mode_idx, 3].axis('off')

        # Column 4: Zero-filled at unsampled position (t=6)
        axes[mode_idx, 4].imshow(zero_filled[0, unsampled_pos].permute(1, 2, 0).cpu().clamp(0, 1))
        axes[mode_idx, 4].set_title(f"Zero-filled (t={unsampled_pos})")
        axes[mode_idx, 4].axis('off')

        # Column 5: Forward-filled at unsampled position (t=6)
        axes[mode_idx, 5].imshow(forward_filled[0, unsampled_pos].permute(1, 2, 0).cpu().clamp(0, 1))
        axes[mode_idx, 5].set_title(f"Forward-filled (t={unsampled_pos})")
        axes[mode_idx, 5].axis('off')

    plt.tight_layout()
    plt.savefig('dcvc_sample_codec_test.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nVisualization saved to 'dcvc_sample_codec_test.png'")