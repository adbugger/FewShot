from __future__ import division

import time
import numpy as np

import torch

from utils import ( getattr_or_default,
                    get_loader,
                    seed_everything,
                    AverageMeter,
                    get_gpu_ids,
                    get_func_on_master,
                    get_printer,
                    Value,
                    ValuePrinter
                )

import datasets
import losses

from arguments import parse_args
from models import get_model
from optimizers import get_optimizer, get_scheduler
from evaluators import kmeans_on_data

from sklearn.preprocessing import StandardScaler, Normalizer

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
    # Print(model)

    dataset = getattr(datasets, options.dataset)(options)
    train_loader = get_loader(dataset.train_set, options)
    num_train_classes = len(dataset.train_set.dataset.classes)

    # Switch off for validation and testing
    options.shuffle = False
    
    plain_train_loader = get_loader(dataset.plain_train_set, options)

    test_loader = get_loader(dataset.test_set, options)
    num_test_classes = len(test_loader.dataset.classes)

    valid_loader = get_loader(dataset.valid_set, options)
    num_valid_classes = len(valid_loader.dataset.classes)

    criterion = getattr(losses, options.loss_function)(options)
    final_optimizer = get_optimizer(model, options)
    scheduler = get_scheduler(final_optimizer, options)

    time_track = AverageMeter()
    best_model_state = model.state_dict()

    loss_val = Value(1e6, min, name="loss")
    loss_printer = ValuePrinter()
    loss_printer.track(loss_val)

    test_eval = Value(-1e6, max, name="test_acc")
    val_eval = Value(-1e6, max, name="val_acc")
    train_eval = Value(-1e6, max, name="train_acc")
    eval_printer = ValuePrinter()
    eval_printer.track(train_eval)
    eval_printer.track(val_eval)
    eval_printer.track(test_eval)

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

        loss_val.update(epoch_loss_track.value())

        Print(f"({time_track.latest():>7.3f}s) Epoch {epoch+1:0>3}/{options.num_epochs:>3}:", end='')
        Print(loss_printer.get_formatted_line(), end='')
        if loss_val.current_is_best:
            best_model_state = model.state_dict()

        if options.local_rank==0 and epoch % options.eval_freq == options.eval_freq-1:
            eval_start = time.time()
            model.eval()

            train_eval.update(kmeans_on_data(model, plain_train_loader, options))
            val_eval.update(kmeans_on_data(model, valid_loader, options))
            test_eval.update(kmeans_on_data(model, test_loader, options))

            model.train()
            eval_time = time.time() - eval_start

            Print(f"  ({eval_time:>7.3f}s) ", end='')
            Print(eval_printer.get_formatted_line(), end='')
        Print()

    Print((f"Training for {options.num_epochs} epochs took {time_track.total():.3f}s total "
           f"and {time_track.value():.3f}s average"))

    Print("Calculating mean of transformed dataset using the best model state ...", end='')
    # since this is what will be saved later
    model.load_state_dict(best_model_state)
    model.eval()
    scaler = StandardScaler(copy=False, with_std=False)

    mean_time = time.time()
    for data, _ in plain_train_loader:
        # feat = model.module.backbone(data.to(device=options.cuda_device)).detach().cpu().numpy()
        feat = model(data.to(device=options.cuda_device)).detach().cpu().numpy()
        scaler.partial_fit(feat)
    mean_time = time.time() - mean_time
    Print(f" {mean_time:.3f}s") 
    options.train_scaler = scaler

    Print(f"Saving best model and options to {options.save_path}")
    save_dict = {'option': options}
    if options.save_model:
        save_dict['model_state_dict'] = best_model_state
    Save(save_dict, options.save_path)


if __name__ == '__main__':
    # The runs are non-deterministic anyway, no point sacrificing speed.
    # seed_everything(1337)
    options = parse_args()
    train_loop(options)
