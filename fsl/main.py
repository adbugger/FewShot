from __future__ import division

import time

import torch

from utils import ( getattr_or_default,
                    get_loader,
                    seed_everything,
                    AverageMeter,
                    get_gpu_ids,
                    get_printer
                )

import datasets
import losses

from arguments import parse_args
from models import get_model
from optimizers import get_optimizer, get_scheduler


if __name__ == '__main__':
    seed_everything(1337)

    options = parse_args()

    Print = get_printer(options)
    Print(options)

    # distributed stuff
    gpus = get_gpu_ids()
    torch.cuda.set_device(options.local_rank)
    cuda_device = f"cuda:{options.local_rank}"
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
    # options.shuffle = False
    # test_loader = get_loader(dataset.test_set, options)
    # valid_loader = get_loader(dataset.valid_set, options)

    criterion = getattr(losses, options.loss_function)(options)

    final_optimizer = get_optimizer(model, options)
    scheduler = get_scheduler(final_optimizer, options)

    Print(("Starting Training\n"
             "-----------------"))
    time_track = AverageMeter()
    best_model_state = model.state_dict()
    min_loss = 1e5
    for epoch in range(options.num_epochs):
        loss_track = AverageMeter()
        epoch_start = time.time()

        for aug1, aug2, targets in train_loader:
            aug1 = aug1.to(device=cuda_device)
            aug2 = aug2.to(device=cuda_device)

            final_optimizer.zero_grad()
            feat1 = model(aug1)
            feat2 = model(aug2)
            loss = criterion(feat1, feat2)
            loss.backward()
            final_optimizer.step()
            loss_track.accumulate(loss.item())

        scheduler.step()
        time_track.accumulate(time.time() - epoch_start)

        loss_value = loss_track.value()

        Print((f"({time_track.latest():>8.3f}s) Epoch {epoch+1:0>3}/{options.num_epochs:>3}: "
                 f"Loss={loss_value:<f}"), end='')
        if loss_value < min_loss:
            Print(" (best so far)", end='')
            min_loss = loss_value
            best_model_state = model.state_dict()
        Print()

    Print((f"Training for {options.num_epochs} epochs took {time_track.total():.3f}s total "
             f"and {time_track.value():.3f}s average"))

    if options.local_rank==0:
        Print(f"Saving best model and options to {options.save_path}")
        torch.save({
            'options': options,
            'model_state_dict': best_model_state,
        }, options.save_path)
