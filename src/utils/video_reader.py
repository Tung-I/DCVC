# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import cv2
import numpy as np
from PIL import Image


class PNGReader():
    def __init__(self, src_path, width, height, start_num=1):
        self.eof = False
        self.src_path = src_path
        self.width = width
        self.height = height
        self.current_frame_index = start_num

        # detect zero-padding from filenames
        pngs = os.listdir(self.src_path)
        if any(name.startswith("im1.") for name in pngs):
            self.padding = 1
        elif any(name.startswith("im00001.") for name in pngs):
            self.padding = 5
        else:
            raise ValueError("unknown image naming convention; please specify")

    def read_one_frame(self):
        if self.eof:
            return None

        fname = f"im{str(self.current_frame_index).zfill(self.padding)}.png"
        png_path = os.path.join(self.src_path, fname)
        if not os.path.exists(png_path):
            self.eof = True
            return None

        # --- load as uint16 or uint8, unchanged ---
        raw = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise IOError(f"could not read {png_path}")

        # raw shape: H×W (mono) or H×W×C
        # 1) if mono, add channel dim
        if raw.ndim == 2:
            raw = raw[:, :, None]

        # 2) if only 1 channel, replicate to 3
        if raw.shape[2] == 1:
            raw = np.repeat(raw, 3, axis=2)
        # if >3 channels, just keep first 3:
        elif raw.shape[2] > 3:
            raise ValueError(f"more than 3 channels not supported: {raw.shape[2]}")

        # 3) now raw is H×W×3, dtype uint8 or uint16
        h, w, _ = raw.shape
        assert h == self.height and w == self.width

        # 4) normalize to float32 [0,1]
        raw = raw.astype(np.float32)
        raw /= 65535.0


        # 5) transpose to C×H×W
        frame = raw.transpose(2, 0, 1)

        self.current_frame_index += 1
        return frame

    def close(self):
        self.current_frame_index = 1

# class PNGReader():
#     def __init__(self, src_path, width, height, start_num=1):
#         self.eof = False
#         self.src_path = src_path
#         self.width = width
#         self.height = height
#         pngs = os.listdir(self.src_path)
#         if 'im1.png' in pngs:
#             self.padding = 1
#         elif 'im00001.png' in pngs:
#             self.padding = 5
#         else:
#             raise ValueError('unknown image naming convention; please specify')
#         self.current_frame_index = start_num

#     def read_one_frame(self):
#         # rgb: 3xhxw uint8 numpy array
#         if self.eof:
#             return None

#         png_path = os.path.join(self.src_path,
#                                 f"im{str(self.current_frame_index).zfill(self.padding)}.png"
#                                 )
#         if not os.path.exists(png_path):
#             self.eof = True
#             return None

#         rgb = Image.open(png_path).convert('RGB')
#         rgb = np.asarray(rgb).astype(np.uint8).transpose(2, 0, 1)
#         _, height, width = rgb.shape
#         assert height == self.height
#         assert width == self.width

#         self.current_frame_index += 1
#         return rgb

#     def close(self):
#         self.current_frame_index = 1


class YUV420Reader():
    def __init__(self, src_path, width, height, skip_frame=0):
        self.eof = False
        if not src_path.endswith('.yuv'):
            src_path = src_path + '.yuv'
        self.src_path = src_path

        self.y_size = width * height
        self.y_width = width
        self.y_height = height
        self.uv_size = width * height // 2
        self.uv_width = width // 2
        self.uv_height = height // 2
        # pylint: disable=R1732
        self.file = open(src_path, "rb")
        # pylint: enable=R1732
        skipped_frame = 0
        while not self.eof and skipped_frame < skip_frame:
            y = self.file.read(self.y_size)
            uv = self.file.read(self.uv_size)
            if not y or not uv:
                self.eof = True
            skipped_frame += 1

    def read_one_frame(self):
        # y: 1xhxw uint8 numpy array
        # uv: 2x(h/2)x(w/2) uint8 numpy array
        if self.eof:
            return None, None
        y = self.file.read(self.y_size)
        uv = self.file.read(self.uv_size)
        if not y or not uv:
            self.eof = True
            return None, None
        y = np.frombuffer(y, dtype=np.uint8).copy().reshape(1, self.y_height, self.y_width)
        uv = np.frombuffer(uv, dtype=np.uint8).copy().reshape(2, self.uv_height, self.uv_width)

        return y, uv

    def close(self):
        self.file.close()
