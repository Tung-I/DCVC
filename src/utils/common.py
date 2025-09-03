# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
from unittest.mock import patch
import os, uuid, shutil, time, errno, getpass, pathlib, subprocess, sys

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import numpy as np


def pick_local_base() -> str:
    return os.path.abspath("./.local_tmp")

def cleanup_stale_locks(root):
    p = pathlib.Path(root)
    if p.exists():
        for lock in p.rglob("lock"):
            try: lock.unlink()
            except Exception: pass

def safe_import_eff_distloss(max_retries=4):
    """
    Try importing the package that triggers JIT build; on locking errors:
    - clean locks
    - switch to a fresh unique TORCH_EXTENSIONS_DIR
    - retry with backoff
    """
    for attempt in range(1, max_retries + 1):
        try:
            import torch_efficient_distloss  # noqa: F401
            return
        except OSError as e:
            msg = str(e)
            if "Stale file handle" in msg or getattr(e, "errno", None) in (116, errno.EIO, errno.ESTALE):
                # rotate to a brand-new cache dir and try again
                setup_unique_torch_extensions_dir(run_id=uuid.uuid4().hex[:8])
                time.sleep(0.3 * attempt)  # small backoff
                continue
            raise
        except Exception:
            # Any other import failure—bubble up after 1 retry with new cache
            if attempt == 1:
                setup_unique_torch_extensions_dir(run_id=uuid.uuid4().hex[:8])
                continue
            raise

def setup_unique_torch_extensions_dir(run_id=None, ref_cache=None):
    """
    - Creates a unique TORCH_EXTENSIONS_DIR to avoid lock contention across concurrent jobs.
    - Optionally copies a prebuilt reference cache tree (if provided) to avoid recompile.
    """
    base = pick_local_base()
    run_id = run_id or os.environ.get("RUN_ID") or uuid.uuid4().hex[:8]
    ext_dir = os.path.join(base, f"torch_ext_{run_id}")
    os.makedirs(ext_dir, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = ext_dir
    # optional: keep TORCH_HOME here too
    os.environ.setdefault("TORCH_HOME", os.path.join(ext_dir, "torch_home"))
    if ref_cache and os.path.isdir(ref_cache):
        try:
            # Copy the whole tree once; cheap if on same FS (reflink) and prevents rebuilds.
            shutil.copytree(ref_cache, ext_dir, dirs_exist_ok=True)
        except Exception:
            pass
    # defensively remove any stale locks inside both the new and legacy locations
    cleanup_stale_locks(ext_dir)
    cleanup_stale_locks(os.path.expanduser("~/.cache/torch_extensions"))
    return ext_dir



def str2bool(v):
    return str(v).lower() in ("yes", "y", "true", "t", "1")


def set_torch_env():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(seed=0)
    try:
        # require pytorch >= 2.2.0
        torch.utils.deterministic.fill_uninitialized_memory = False
    except Exception:  # pylint: disable=W0718
        pass


def create_folder(path, print_if_create=False):
    if not os.path.exists(path):
        os.makedirs(path)
        if print_if_create:
            print(f"created folder: {path}")


def get_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=True)
    if "state_dict" in ckpt:
        ckpt = ckpt['state_dict']
    if "net" in ckpt:
        ckpt = ckpt["net"]
    consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
    return ckpt


@patch('json.encoder.c_make_encoder', None)
def dump_json(obj, fid, float_digits=-1, **kwargs):
    of = json.encoder._make_iterencode  # pylint: disable=W0212

    def inner(*args, **kwargs):
        args = list(args)
        # fifth argument is float formater which we will replace
        args[4] = lambda o: format(o, '.%df' % float_digits)
        return of(*args, **kwargs)

    with patch('json.encoder._make_iterencode', wraps=inner):
        json.dump(obj, fid, **kwargs)


def generate_log_json(frame_num, frame_pixel_num, test_time, frame_types, bits, psnrs, ssims,
                      verbose=False, avg_encoding_time=None, avg_decoding_time=None):
    include_yuv = len(psnrs[0]) > 1
    assert not include_yuv or (len(psnrs[0]) == 4 and len(ssims[0]) == 4)
    i_bits = 0
    i_psnr = 0
    i_psnr_y = 0
    i_psnr_u = 0
    i_psnr_v = 0
    i_ssim = 0
    i_ssim_y = 0
    i_ssim_u = 0
    i_ssim_v = 0
    p_bits = 0
    p_psnr = 0
    p_psnr_y = 0
    p_psnr_u = 0
    p_psnr_v = 0
    p_ssim = 0
    p_ssim_y = 0
    p_ssim_u = 0
    p_ssim_v = 0
    i_num = 0
    p_num = 0
    for idx in range(frame_num):
        if frame_types[idx] == 0:
            i_bits += bits[idx]
            i_psnr += psnrs[idx][0]
            i_ssim += ssims[idx][0]
            i_num += 1
            if include_yuv:
                i_psnr_y += psnrs[idx][1]
                i_psnr_u += psnrs[idx][2]
                i_psnr_v += psnrs[idx][3]
                i_ssim_y += ssims[idx][1]
                i_ssim_u += ssims[idx][2]
                i_ssim_v += ssims[idx][3]
        else:
            p_bits += bits[idx]
            p_psnr += psnrs[idx][0]
            p_ssim += ssims[idx][0]
            p_num += 1
            if include_yuv:
                p_psnr_y += psnrs[idx][1]
                p_psnr_u += psnrs[idx][2]
                p_psnr_v += psnrs[idx][3]
                p_ssim_y += ssims[idx][1]
                p_ssim_u += ssims[idx][2]
                p_ssim_v += ssims[idx][3]

    log_result = {}
    log_result['frame_pixel_num'] = frame_pixel_num
    log_result['i_frame_num'] = i_num
    log_result['p_frame_num'] = p_num
    log_result['ave_i_frame_bpp'] = i_bits / i_num / frame_pixel_num
    log_result['ave_i_frame_psnr'] = i_psnr / i_num
    log_result['ave_i_frame_msssim'] = i_ssim / i_num
    if include_yuv:
        log_result['ave_i_frame_psnr_y'] = i_psnr_y / i_num
        log_result['ave_i_frame_psnr_u'] = i_psnr_u / i_num
        log_result['ave_i_frame_psnr_v'] = i_psnr_v / i_num
        log_result['ave_i_frame_msssim_y'] = i_ssim_y / i_num
        log_result['ave_i_frame_msssim_u'] = i_ssim_u / i_num
        log_result['ave_i_frame_msssim_v'] = i_ssim_v / i_num
    if verbose:
        log_result['frame_bpp'] = list(np.array(bits) / frame_pixel_num)
        log_result['frame_psnr'] = [v[0] for v in psnrs]
        log_result['frame_msssim'] = [v[0] for v in ssims]
        log_result['frame_type'] = frame_types
        if include_yuv:
            log_result['frame_psnr_y'] = [v[1] for v in psnrs]
            log_result['frame_psnr_u'] = [v[2] for v in psnrs]
            log_result['frame_psnr_v'] = [v[3] for v in psnrs]
            log_result['frame_msssim_y'] = [v[1] for v in ssims]
            log_result['frame_msssim_u'] = [v[2] for v in ssims]
            log_result['frame_msssim_v'] = [v[3] for v in ssims]
    log_result['test_time'] = test_time
    if p_num > 0:
        total_p_pixel_num = p_num * frame_pixel_num
        log_result['ave_p_frame_bpp'] = p_bits / total_p_pixel_num
        log_result['ave_p_frame_psnr'] = p_psnr / p_num
        log_result['ave_p_frame_msssim'] = p_ssim / p_num
        if include_yuv:
            log_result['ave_p_frame_psnr_y'] = p_psnr_y / p_num
            log_result['ave_p_frame_psnr_u'] = p_psnr_u / p_num
            log_result['ave_p_frame_psnr_v'] = p_psnr_v / p_num
            log_result['ave_p_frame_msssim_y'] = p_ssim_y / p_num
            log_result['ave_p_frame_msssim_u'] = p_ssim_u / p_num
            log_result['ave_p_frame_msssim_v'] = p_ssim_v / p_num
    else:
        log_result['ave_p_frame_bpp'] = 0
        log_result['ave_p_frame_psnr'] = 0
        log_result['ave_p_frame_msssim'] = 0
        if include_yuv:
            log_result['ave_p_frame_psnr_y'] = 0
            log_result['ave_p_frame_psnr_u'] = 0
            log_result['ave_p_frame_psnr_v'] = 0
            log_result['ave_p_frame_msssim_y'] = 0
            log_result['ave_p_frame_msssim_u'] = 0
            log_result['ave_p_frame_msssim_v'] = 0
    log_result['ave_all_frame_bpp'] = (i_bits + p_bits) / (frame_num * frame_pixel_num)
    log_result['ave_all_frame_psnr'] = (i_psnr + p_psnr) / frame_num
    log_result['ave_all_frame_msssim'] = (i_ssim + p_ssim) / frame_num
    if avg_encoding_time is not None and avg_decoding_time is not None:
        log_result['avg_frame_encoding_time'] = avg_encoding_time
        log_result['avg_frame_decoding_time'] = avg_decoding_time
    if include_yuv:
        log_result['ave_all_frame_psnr_y'] = (i_psnr_y + p_psnr_y) / frame_num
        log_result['ave_all_frame_psnr_u'] = (i_psnr_u + p_psnr_u) / frame_num
        log_result['ave_all_frame_psnr_v'] = (i_psnr_v + p_psnr_v) / frame_num
        log_result['ave_all_frame_msssim_y'] = (i_ssim_y + p_ssim_y) / frame_num
        log_result['ave_all_frame_msssim_u'] = (i_ssim_u + p_ssim_u) / frame_num
        log_result['ave_all_frame_msssim_v'] = (i_ssim_v + p_ssim_v) / frame_num

    return log_result
