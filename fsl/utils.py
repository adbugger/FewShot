from __future__ import division, print_function

import functools
import random
import numpy as np
import os
import abc

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def get_gpu_ids():
    return [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]

def do_nothing(*args, **kwargs):
    pass

def get_func_on_master(func, options):
    if options.local_rank==0:
        return func
    return do_nothing

def get_printer(options):
    if hasattr(options, 'log_file') and options.log_file is not None:
        print_func = functools.partial(print, file=options.log_file, flush=True)
    else:
        print_func = functools.partial(print, flush=True)
    return get_func_on_master(print_func, options)

def getattr_or_default(obj, prop, def_val):
    if not hasattr(obj, prop):
        return def_val

    value = getattr(obj, prop)
    if value is None:
        return def_val

    return value

def get_loader(dataset, options):
    if options.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=len(get_gpu_ids()),
            rank=options.local_rank
        )
        return DataLoader(dataset,
                          batch_size=options.batch_size,
                          # shuffle=options.shuffle,
                          # num_workers=options.num_workers,
                          pin_memory=options.pin_memory,
                          sampler=sampler,
                        )

    return DataLoader(dataset,
                      batch_size=options.batch_size,
                      shuffle=options.shuffle,
                      num_workers=options.num_workers,
                      pin_memory=options.pin_memory,
                    )

def seed_everything(seed, high_speed=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = not high_speed
        torch.backends.cudnn.benchmark = high_speed


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

class ValuePrinter():
    GREEN = u"\u001b[32m"
    RESET = u"\u001b[0m"

    def __init__(self):
        self.values = list()

    def track(self, value):
        self.values.append(value)
        
    def get_formatted_line(self):
        s = ""
        for v in self.values:
            s += f" {v.name}="
            if v.current_is_best:
                s += self.__class__.GREEN
            s += f"{v.current:<8.6f}"
            s += self.__class__.RESET
        # print(s)
        return s

class Value():
    def __init__(self, init_val, better_func, name="value name", tol=1e-7):
        self.current = init_val
        self.best = init_val
        self.current_is_best = True
        self.better_func = better_func
        self.tol = tol
        self.name = name
    
    def is_better_than(self, new_val):
        return abs(self.best - self.better_func(self.best, new_val)) > self.tol

    def update(self, other_val):
        self.current_is_best = self.is_better_than(other_val)
        self.current = other_val
        if self.current_is_best:
            self.best = self.current

if __name__ == "__main__":
    vp = ValuePrinter()
    val1 = Value(-0.5, max, "want more")
    val2 = Value(-5000.236, min, "want less")

    vp.track(val1)
    vp.track(val2)

    vp.get_formatted_line()
    
    val1.update(1.0)
    val2.update(1.0)

    vp.get_formatted_line()
