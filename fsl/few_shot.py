from __future__ import print_function, division

import time
import torch

import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, Normalizer

from arguments import parse_args
from models import ( get_model, get_old_state )
from optimizers import get_optimizer

import datasets
import losses
import episode_strat
import testing_strat

from utils import ( get_printer,
                    get_gpu_ids,
                    AverageMeter)


def few_shot_loop(options):
    Print = get_printer(options)

    # In --no_distributed / single GPU mode, the GPU id may not be the local rank
    options.cuda_device = f"cuda:{get_gpu_ids()[0]}"
    # distributed stuff
    if options.distributed:
        gpus = get_gpu_ids()
        options.cuda_device = f"cuda:{options.local_rank}"
        torch.cuda.set_device(options.local_rank)
        if options.distributed:
            torch.distributed.init_process_group(
                backend='nccl',
                init_method="env://",
                world_size=len(gpus),
                rank=options.local_rank
            )

    model, old_opts = get_old_state(options)
    model.eval()

    options.scaler = old_opts.train_scaler if hasattr(old_opts, "train_scaler") else None
    options.normalizer = Normalizer(copy=False)

    episode_loader = getattr(episode_strat, options.episode_strat)(old_opts).episode_loader(options)
    classifier = getattr(testing_strat, options.testing_strat)

    score_track = AverageMeter()
    time_track = AverageMeter()
    for full_data, full_labels in episode_loader:
        start_time = time.time()

        # full_data = model.module.backbone(full_data.to(options.cuda_device))
        full_data = model(full_data.to(options.cuda_device))
        # full_labels = full_labels.cpu().numpy()

        # call to classifier here
        score = classifier(options, full_data, full_labels)
        score_track.accumulate(score)
        
        time_track.accumulate(time.time() - start_time)

    Print(f"({time_track.latest():.3f}s avg / {time_track.total():.3f}s) Using file {options.load_from.name}: {score_track.value()}")
    return

if __name__ == '__main__':
    options = parse_args()
    few_shot_loop(options)
