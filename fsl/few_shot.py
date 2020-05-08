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

from utils import ( get_printer,
                    get_gpu_ids,
                    AverageMeter)

def few_shot_loop(options):
    Print = get_printer(options)

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

    classifier = KNeighborsClassifier(n_neighbors=1)
    scaler = old_opts.train_scaler if hasattr(old_opts, "train_scaler") else None
    normalizer = Normalizer(copy=False)

    episode_loader = getattr(episode_strat, options.episode_strat)(old_opts).episode_loader(options)
    # episode_num = 1

    score_track = AverageMeter()
    time_track = AverageMeter()
    for full_data, full_labels in episode_loader:
        start_time = time.time()

        # full_data = model.module.backbone(full_data.to(options.cuda_device)).detach().cpu().numpy()
        full_data = model(full_data.to(options.cuda_device)).detach().cpu().numpy()
        full_labels = full_labels.cpu().numpy()

        # SimpleShot center and unit norm
        if scaler is not None:
            full_data = scaler.transform(full_data)
        full_data = normalizer.transform(full_data)

        num_train = options.n_way * options.k_shot
        train_data, test_data = full_data[:num_train], full_data[num_train:]
        train_labels, test_labels = full_labels[:num_train], full_labels[num_train:]

        classifier.fit(train_data, train_labels)
        score = classifier.score(test_data, test_labels)
        score_track.accumulate(score)
        
        time_track.accumulate(time.time() - start_time)

    Print(f"({time_track.latest():.3f}s avg / {time_track.total():.3f}s) Using file {options.load_from.name}: {score_track.value()}")
    return

if __name__ == '__main__':
    options = parse_args()
    few_shot_loop(options)
