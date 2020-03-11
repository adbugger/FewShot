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
    # if options.multi_gpu:
        # gpu_ids = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(',')]
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # torch.distributed.init_process_group(backend='nccl', world_size=len(gpu_ids), rank=0)
        # model = nn.parallel.DistributedDataParallel(model)
        # model = nn.DataParallel(model)
    return model
