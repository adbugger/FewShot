from __future__ import division

import random
import os
import abc

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def get_gpu_ids():
    return [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]

def getattr_or_default(obj, prop, def_val):
    if not hasattr(obj, prop):
        return def_val

    value = getattr(obj, prop)
    if value is None:
        return def_val

    return value

def get_loader(dataset, options):
    sampler = DistributedSampler(
        dataset,
        num_replicas=len(get_gpu_ids()),
        rank=options.local_rank
    )
    return DataLoader(dataset,
                      batch_size=options.batch_size,
                      # shuffle=options.shuffle,
                      num_workers=options.num_workers,
                      pin_memory=options.pin_memory,
                      sampler=sampler,
                    )

def seed_everything(seed, high_speed=False):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = high_speed
        torch.backends.cudnn.deterministic = not high_speed


class AbstractMeter(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def accumulate(self, value):
        return

class AverageMeter(AbstractMeter):
    def __init__(self, val=0, count=0):
        super().__init__()
        self.last_val = val
        self.sum_val = val
        self.count = count

    def accumulate(self, value):
        self.count += 1
        self.sum_val += value
        self.last_val = value

    def value(self):
        return self.sum_val / self.count if self.count !=0 else 0

    def total(self):
        return self.sum_val

    def latest(self):
        return self.last_val
