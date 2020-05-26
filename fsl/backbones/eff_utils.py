# Code functions taken from https://github.com/facebookresearch/pycls/blob/master/pycls/core/net.py

from argparse import Namespace
import math

import torch
nn = torch.nn

def drop_connect(x, drop_ratio):
    """Drop connect (adapted from DARTS)."""
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x

def init_weights(m, cfg):
    """Performs ResNet-style weight initialization."""
    if isinstance(m, nn.Conv2d):
        # Note that there is no bias due to BN
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
    elif isinstance(m, nn.BatchNorm2d):
        zero_init_gamma = (
            hasattr(m, "final_bn") and m.final_bn and cfg.BN.ZERO_INIT_FINAL_GAMMA
        )
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()

def make_cfg(options):
    cfg = Namespace()
    cfg.BN = Namespace(EPS=1e-5, MOM=0.1, ZERO_INIT_FINAL_GAMMA=False)
    cfg.EN = Namespace(
        DROPOUT_RATIO=0.0, DC_RATIO=0.0,
        STEM_W=32, DEPTHS=[], WIDTHS=[], EXP_RATIOS=[], SE_R=0.25, )