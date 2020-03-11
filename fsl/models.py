import sys
import os

import torch
nn = torch.nn

import backbones
import heads

__all__ = ['get_model']

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
    model = model.to(device=f"cuda:{options.local_rank}")
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[options.local_rank],
        output_device=options.local_rank
    )
    return model
