from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_ssim
from .lpips_clipiqa_musiq import calculate_lpips_outer_function,calculate_clipiqa_outer_function,calculate_musiq_outer_function

calculate_lpips_outer_function()
calculate_clipiqa_outer_function()
calculate_musiq_outer_function()
from .infrared_BGR_fusion import *
__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
