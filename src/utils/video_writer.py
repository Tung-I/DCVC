# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import cv2
import numpy as np
from PIL import Image

class PNGWriter():
    def __init__(self, dst_path, width, height):
        self.dst_path = dst_path
        self.width = width
        self.height = height
        self.padding = 5
        self.current_frame_index = 1
        os.makedirs(dst_path, exist_ok=True)

    def write_one_frame(self, frame, qp=None):
        """
        frame: 3×H×W numpy (float32 in [0,1], or uint16 already)
        We will extract one channel (they’re all identical), convert to uint16, and save.
        """
        # bring to H×W×C
        arr = frame.transpose(1, 2, 0)

        # # if float in [0,1], convert to uint16 range [0,65535]
        # if arr.dtype in (np.float32, np.float64):
        #     arr = np.clip(arr, 0.0, 1.0)
        #     arr = (arr * 65535.0).round().astype(np.uint16)
        # elif arr.dtype == np.uint8:
        #     # maybe somebody fell back to 8-bit → upscale?
        #     arr = (arr.astype(np.uint16) * 257)  # 255→65535
        # otherwise assume uint16 already

        # now arr is H×W×C with dtype uint16
        # pick one channel (all three are the same)
        gray = arr[..., 0]  # H×W uint16
        if qp is not None:
            fname = f"im{str(self.current_frame_index).zfill(self.padding)}_qp{qp}.png"
        else:
            fname = f"im{str(self.current_frame_index).zfill(self.padding)}.png"
        out_path = os.path.join(self.dst_path, fname)

        # cv2.imwrite will respect uint16
        cv2.imwrite(out_path, gray)

        self.current_frame_index += 1

    def close(self):
        self.current_frame_index = 1

# class PNGWriter():
#     def __init__(self, dst_path, width, height):
#         self.dst_path = dst_path
#         self.width = width
#         self.height = height
#         self.padding = 5
#         self.current_frame_index = 1
#         os.makedirs(dst_path, exist_ok=True)

#     def write_one_frame(self, rgb):
#         # rgb: 3xhxw uint8 numpy array
#         rgb = rgb.transpose(1, 2, 0)

#         png_path = os.path.join(self.dst_path,
#                                 f"im{str(self.current_frame_index).zfill(self.padding)}.png"
#                                 )
#         Image.fromarray(rgb).save(png_path)

#         self.current_frame_index += 1

#     def close(self):
#         self.current_frame_index = 1


class YUV420Writer():
    def __init__(self, dst_path, width, height):
        if not dst_path.endswith('.yuv'):
            dst_path = dst_path + '/out.yuv'
        self.dst_path = dst_path
        self.width = width
        self.height = height

        # pylint: disable=R1732
        self.file = open(dst_path, "wb")
        # pylint: enable=R1732

    def write_one_frame(self, y, uv):
        # y: 1xhxw uint8 numpy array
        # uv: 2x(h/2)x(w/2) uint8 numpy array
        self.file.write(y.tobytes())
        self.file.write(uv.tobytes())

    def close(self):
        self.file.close()
