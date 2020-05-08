"""Increase supervision on the completely unsupervised pipeline.

Uses:
    options.data_percent : x % of dataset
    options.fine_tune_epochs : fine tune for these epochs
    options.load_from : load this model, use dataset from here as well
"""
from __future__ import print_function, division

import time

import datasets
import torch
import random

from arguments import parse_args
from models import ( get_model, get_old_state )
from optimizers import ( get_optimizer, get_scheduler )
from utils import (
    get_gpu_ids, get_printer, AverageMeter,
    get_func_on_master,
    ValuePrinter, Value)

def choose_indices(options, dataset):
    class_indices = dict()
    for idx, (_, label) in enumerate(dataset):
        if label in class_indices:
            class_indices[label].append(idx)
        else:
            class_indices[label] = [idx]
    
    chosen_indices = []
    for idx_list in class_indices.values():
        num_choose = int(options.data_percent / 100 * len(idx_list))
        chosen_indices.extend(random.sample(idx_list, num_choose))
    
    return chosen_indices

def fine_tune(options):
    # get print and save functions
    Print = get_printer(options)
    Save = get_func_on_master(torch.save, options)

    # get old_options
    model, old_opts = get_old_state(options)

    # subsample old dataset
    dataset = getattr(datasets, old_opts.dataset)(old_opts).plain_train_set
    indices = choose_indices(options, dataset)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=options.batch_size)
    full_loader = torch.utils.data.DataLoader(dataset, batch_size=options.batch_size)

    # complete model
    num_classes = len(dataset.classes)
    intermediate_dim = int((num_classes + old_opts.projection_dim) / 2)
    full_model = torch.nn.Sequential(
        model,
        torch.nn.Linear(old_opts.projection_dim, intermediate_dim),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(intermediate_dim, num_classes),
        torch.nn.LogSoftmax(dim=1)
    ).to(device=options.cuda_device)

    # get loss
    criterion = torch.nn.NLLLoss()
    optimizer = get_optimizer(full_model, options)
    scheduler = get_scheduler(optimizer, options)

    # train for num_epochs
    full_model.train()
    
    # pretty printer for loss
    loss_val = Value(1e6, min, name="loss")
    loss_printer = ValuePrinter()
    loss_printer.track(loss_val)

    timer = AverageMeter()
    for epoch in range(options.fine_tune_epochs):
        t = time.time()
        epoch_loss = AverageMeter()
        for data, labels in loader:
            Print('.', end='')
            optimizer.zero_grad()

            out = full_model(data.to(device=options.cuda_device))

            loss = criterion(out, labels.to(device=options.cuda_device))
            loss.backward()
            optimizer.step()

            epoch_loss.accumulate(loss.item())
        scheduler.step()
        loss_val.update(epoch_loss.value())
        timer.accumulate(time.time() - t)
        Print(f" ({timer.latest():>6.2f}s) epoch {epoch+1:>3}/{options.fine_tune_epochs:>3}:"
              f"{loss_printer.get_formatted_line()}")

    Print(f"Fine tuning: {timer.total():.3f}s {options.fine_tune_epochs} epochs / {timer.value():.3f}s avg")
           
    # evaluate on train set once for sanity
    full_model.eval()
    acc = AverageMeter()

    for data, labels in full_loader:
        predicts = full_model(data.to(device=options.cuda_device))
        predicts = predicts.argmax(dim=1)
        labels = labels.to(device=options.cuda_device)

        acc.accumulate((predicts == labels).sum().item() / predicts.size(0))

    Print(f"Saving old options, model state, and base path to {options.save_path}")
    Save({
        'options': old_opts,
        'model_state_dict': model.state_dict(),
        'loaded_from': options.load_from.name
    },  options.save_path)
    
    Print(acc.value())
    return acc.value()

if __name__ == "__main__":
    options = parse_args()
    options.cuda_device = f"cuda:{get_gpu_ids()[0]}"
    fine_tune(options)