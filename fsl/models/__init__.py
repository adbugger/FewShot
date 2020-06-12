import sys

import torch
import torch.nn as nn

from .SimCLR import SimCLRModel
from .MoCo import MoCoModel
from .SelfLabel import SelfLabelModel

__all__ = ['get_model', 'get_old_state', 'SimCLRModel', 'MoCoModel', 'SelfLabelModel']

# Each model must be initialized with model(options)
def get_model(options):
    model = (getattr(sys.modules[__name__], options.model)(options))
    if options.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device=options.cuda_device)
    if options.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[options.local_rank],
            output_device=options.local_rank
        )
    else:
        model = nn.DataParallel(model)
    return model

def get_old_state(options):
    ckpt = torch.load(options.load_from.name)

    if 'option' in ckpt:
        old_opts = ckpt['option']
    elif 'options' in ckpt:
        old_opts = ckpt['options']
    else:
        raise Exception("no options saved")
    
    old_opts.cuda_device = options.cuda_device
    model = get_model(old_opts)
    model.load_state_dict(ckpt['model_state_dict'])
    return model, old_opts