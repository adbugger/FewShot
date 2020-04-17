import sys
import os

import torch
nn = torch.nn

import backbones
import heads

__all__ = ['get_model', 'get_old_state']

class SimCLRModel(nn.Module):
    def __init__(self, options):
        super(SimCLRModel, self).__init__()

        self.backbone = getattr(backbones, options.backbone)(options)
        self.head = getattr(heads, options.head)(options)

    def forward(self, x):
        return self.head(self.backbone(x))

def get_model(options):
    model = (getattr(sys.modules[__name__], options.model)(options))
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device=options.cuda_device)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[options.local_rank],
        output_device=options.local_rank
    )
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
