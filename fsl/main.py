from __future__ import division

import time

import torch

from utils import ( getattr_or_default,
                    get_loader,
                    seed_everything,
                    AverageMeter,
                    get_gpu_ids,
                    get_func_on_master
                )

import datasets
import losses

from arguments import parse_args
from models import get_model
from optimizers import get_optimizer, get_scheduler
from evaluators import evaluate_on_test


def train_loop(options):
    Print = get_func_on_master(print, options)
    Print(options)

    Save = get_func_on_master(torch.save, options)

    # distributed stuff
    gpus = get_gpu_ids()
    options.cuda_device = f"cuda:{options.local_rank}"
    torch.cuda.set_device(options.local_rank)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method="env://",
        world_size=len(gpus),
        rank=options.local_rank
    )

    model = get_model(options)
    Print(model)

    dataset = getattr(datasets, options.dataset)(options)
    train_loader = get_loader(dataset.train_set, options)
    # Switch off for validation and testing
    options.shuffle = False
    test_loader = get_loader(dataset.test_set, options)
    # valid_loader = get_loader(dataset.valid_set, options)
    criterion = getattr(losses, options.loss_function)(options)
    final_optimizer = get_optimizer(model, options)
    scheduler = get_scheduler(final_optimizer, options)

    time_track = AverageMeter()
    best_model_state = model.state_dict()
    min_loss = 1e5

    Print(("Starting Training\n"
           "-----------------"))
    for epoch in range(options.num_epochs):
        model.train()
        epoch_loss_track = AverageMeter()
        # epoch start
        epoch_start = time.time()

        for aug1, aug2, targets in train_loader:
            final_optimizer.zero_grad()
            feat1 = model(aug1.to(device=options.cuda_device))
            feat2 = model(aug2.to(device=options.cuda_device))
            loss = criterion(feat1, feat2)
            loss.backward()
            final_optimizer.step()

            epoch_loss_track.accumulate(loss.item())

        scheduler.step()
        # epoch end
        time_track.accumulate(time.time() - epoch_start)

        avg_loss = epoch_loss_track.value()

        Print((f"({time_track.latest():>8.3f}s) Epoch {epoch+1:0>3}/{options.num_epochs:>3}: "
               f"Loss={avg_loss:<f}"), end='')

        if avg_loss < min_loss:
            Print(" (best so far)", end='')
            min_loss = avg_loss
            best_model_state = model.state_dict()

        if epoch % options.eval_freq == options.eval_freq-1:
            accuracy = evaluate_on_test(model, test_loader, options)
            Print(f" test set cluster acc: {accuracy * 100:<.3f}", end='')

        Print()

    Print((f"Training for {options.num_epochs} epochs took {time_track.total():.3f}s total "
           f"and {time_track.value():.3f}s average"))

    Print(f"Saving best model and options to {options.save_path}")
    Save({
        'option': options,
        'model_state_dict': best_model_state,
    }, options.save_path)

if __name__ == '__main__':
    seed_everything(1337)
    options = parse_args()
    train_loop(options)
