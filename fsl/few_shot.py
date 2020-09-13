from __future__ import print_function, division

import time
import torch

import numpy
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import IncrementalPCA

from arguments import parse_args
from models import ( get_model, get_old_state )
from optimizers import get_optimizer

import datasets
import losses
import episode_strat
import testing_strat

from utils import ( get_printer,
                    get_gpu_ids,
                    AverageMeter,
                    get_loader)


def get_pre_classifier_pipeline(options, model):
    full_train_set = getattr(datasets, options.dataset)(options).plain_train_set
    options.batch_size = 2048
    loader = get_loader(full_train_set, options)
    
    # standardize the scaler for all classifiers
    scaler = StandardScaler(copy=False)
    for batch, _ in loader:
        out = model(batch).detach().cpu().numpy()
        scaler.partial_fit(out)

    if options.model == "MoCoModel":
        steps = [('scaler', scaler)]
    elif options.model == "SelfLabelModel":
        if options.ipca:
            ipca = IncrementalPCA(copy=False, n_components=options.ipca_dim,
                                batch_size=options.batch_size)
            for batch, _ in loader:
                out = model(batch).detach().cpu().numpy()
                ipca.partial_fit(scaler.transform(out))

            steps = [('scaler', scaler), ('ipca', ipca)]
        else:
            steps = [('scaler', scaler)]
    elif options.model == "SimCLRModel":
        steps = [('scaler', scaler)]
    else:
        raise NotImplementedError(f"Classifier pipeline for {model} not implemented")

    pipeline = Pipeline(steps)
    options.pre_classifier_pipeline = pipeline
    return pipeline

@torch.no_grad()
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

    # define sklearn.pipeline.Pipeline to be applied to network outputs
    if options.model == 'MoCoModel':
        model = get_model(options)
        episode_loader = getattr(episode_strat, options.episode_strat)(options).episode_loader(options)
    elif options.model == "SelfLabelModel":
        model = get_model(options)
        episode_loader = getattr(episode_strat, options.episode_strat)(options).episode_loader(options)
    elif options.model == "SimCLRModel":
        model, old_opts = get_old_state(options)
        episode_loader = getattr(episode_strat, options.episode_strat)(old_opts).episode_loader(options)
    else:
        raise NotImplementedError(f"Few Shot on {options.model} not implemented")

    score_track = AverageMeter()
    time_track = AverageMeter()
    model.eval()
    get_pre_classifier_pipeline(options, model)
    classifier = getattr(testing_strat, options.testing_strat)

    for full_data, full_labels in episode_loader:
        start_time = time.time()
        full_data = model(full_data.to(options.cuda_device))
        score = classifier(options, full_data, full_labels)

        score_track.accumulate(score)
        time_track.accumulate(time.time() - start_time)

    m, h = score_track.conf()
    Print(f"({time_track.latest():.3f}s avg / {time_track.total():.3f}s) "
          f"{m*100:.4f} \u00b1 {h*100:.4f}")
    return

if __name__ == '__main__':
    options = parse_args()
    few_shot_loop(options)
