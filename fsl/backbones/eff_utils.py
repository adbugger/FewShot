"""Code functions taken from https://github.com/facebookresearch/pycls/blob/master/pycls/core/net.py"""
# INCOMPLETE FILE
from argparse import Namespace
import math

import torch
import torch.nn as nn


def drop_connect(data, drop_ratio):
    """Drop connect (adapted from DARTS)."""
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([data.shape[0], 1, 1, 1], dtype=data.dtype, device=data.device)
    mask.bernoulli_(keep_ratio)
    data.div_(keep_ratio)
    data.mul_(mask)
    return data


def init_weights(layer, cfg):
    """Performs ResNet-style weight initialization."""
    if isinstance(layer, nn.Conv2d):
        # Note that there is no bias due to BN
        fan_out = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
        layer.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
    elif isinstance(layer, nn.BatchNorm2d):
        zero_init_gamma = (
            hasattr(layer, "final_bn") and layer.final_bn and cfg.BN.ZERO_INIT_FINAL_GAMMA
        )
        layer.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        layer.weight.data.normal_(mean=0.0, std=0.01)
        layer.bias.data.zero_()


def make_cfg():
    cfg = Namespace()
    cfg.BN = Namespace(EPS=1e-5, MOM=0.1, ZERO_INIT_FINAL_GAMMA=False)
    cfg.EN = Namespace(
        DROPOUT_RATIO=0.0, DC_RATIO=0.0,
        STEM_W=32, DEPTHS=[], WIDTHS=[], EXP_RATIOS=[], SE_R=0.25,
        STRIDES=[], KERNELS=[], )
