from __future__ import division

import time

import torch

from utils import ( getattr_or_default,
                    get_loader,
                    seed_everything,
                    AverageMeter,
                    get_gpu_ids,
                    get_func_on_master,
                    get_printer,
                )

import datasets
import losses

from arguments import parse_args
from models import get_model
from optimizers import get_optimizer, get_scheduler
from evaluators import kmeans_on_data


def train_loop(options):
    Print = get_printer(options)
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
    num_train_classes = len(dataset.train_set.dataset.classes)

    # Switch off for validation and testing
    options.shuffle = False

    test_loader = get_loader(dataset.test_set, options)
    num_test_classes = len(test_loader.dataset.classes)

    valid_loader = get_loader(dataset.valid_set, options)
    num_valid_classes = len(valid_loader.dataset.classes)

    criterion = getattr(losses, options.loss_function)(options)
    final_optimizer = get_optimizer(model, options)
    scheduler = get_scheduler(final_optimizer, options)

    time_track = AverageMeter()
    best_model_state = model.state_dict()
    
    min_loss = 1e6
    max_test_eval = 0.0
    max_val_eval = 0.0

    Print((f"Starting Training on:\n"
           f"Train: {num_train_classes:>3d} classes\n"
           f"Valid: {num_valid_classes:>3d} classes\n"
           f"Test:  {num_test_classes:>3d} classes"))
    Print("-"*18)

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

        Print(f"({time_track.latest():>8.3f}s) Epoch {epoch+1:0>3}/{options.num_epochs:>3}:", end='')

        if avg_loss < min_loss:
            Print(f" loss=\u001b[32m{avg_loss:<f}\u001b[0m", end='')
            min_loss = avg_loss
            best_model_state = model.state_dict()
        else:
            Print(f" loss={avg_loss:<f}", end='')

        if options.local_rank==0 and epoch % options.eval_freq == options.eval_freq-1:
            val_start = time.time()
            val_acc = kmeans_on_data(model, valid_loader, options)
            val_time = time.time() - val_start

            test_start = time.time()
            test_acc = kmeans_on_data(model, test_loader, options)
            test_time = time.time() - test_start
            
            if val_acc > max_val_eval:
                Print(f" ({val_time:>8.3f}s) val_acc=\u001b[32m{val_acc * 100:<9.6f}\u001b[0m", end='')
                max_val_eval = val_acc
            else:
                Print(f" ({val_time:>8.3f}s) val_acc={val_acc * 100:<9.6f}", end='')

            if test_acc > max_test_eval:
                Print(f" ({test_time:>8.3f}s) test_acc=\u001b[32m{test_acc * 100:<9.6f}\u001b[0m", end='')
                max_test_eval = test_acc
            else:
                Print(f" ({test_time:>8.3f}s) test_acc={test_acc * 100:<9.6f}", end='')
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
