from __future__ import division

import time
import sys

import torch
nn = torch.nn
DataLoader = torch.utils.data.DataLoader
optim = torch.optim

from utils import getattr_or_default, get_loader, seed_everything, AverageMeter
from arguments import parse_args
import datasets
import models
import losses

from torchlars import LARS

if __name__ == '__main__':
    seed_everything(1337)

    options = parse_args()

    dataset = getattr(datasets, options.dataset)(options)
    print(options)

    train_loader = get_loader(dataset.train_set, options)

    # Switch off for validation and testing
    # options.shuffle = False
    # test_loader = get_loader(dataset.test_set, options)
    # print('got test_loader!!')
    # valid_loader = get_loader(dataset.valid_set, options)
    # print('got valid_loader!!')

    model = nn.DataParallel(getattr(models, options.model)(options).cuda())
    print(model)

    criterion = getattr(losses, options.loss_function)(options)

    base_optimizer = getattr(optim, options.base_optimizer)(model.parameters(),
                                lr=0.3*options.batch_size/256,
                                momentum=options.momentum,
                                weight_decay=options.weight_decay,
                                dampening=options.dampening,
                                nesterov=options.nesterov)
    final_optimizer = base_optimizer
    scheduler = None
    if not options.simple_opt:
        final_optimizer = getattr(sys.modules[__name__],
                                  options.secondary_optimizer)(optimizer=base_optimizer)
        scheduler = getattr(optim.lr_scheduler, options.scheduler)(final_optimizer,T_max=options.T_max)

    print(("Starting Training\n"
           "-----------------"))
    time_track = AverageMeter()
    best_model_state = model.state_dict()
    min_loss = 1e5
    for epoch in range(options.num_epochs):
        loss_track = AverageMeter()
        epoch_start = time.time()

        for aug1, aug2, targets in train_loader:
            final_optimizer.zero_grad()
            feat1 = model(aug1.cuda())
            feat2 = model(aug2.cuda())
            loss = criterion(feat1, feat2)
            loss.backward()
            final_optimizer.step()
            loss_track.accumulate(loss.item())

        if scheduler is not None: scheduler.step()
        time_track.accumulate(time.time() - epoch_start)

        loss_value = loss_track.value()
        print((f"({time_track.latest():>8.3f}s) Epoch {epoch+1:0>3}/{options.num_epochs:>3}: "
               f"Loss={loss_value:<f}"), end='')
        if loss_value < min_loss:
            print(" (best so far)", end='')
            min_loss = loss_value
            best_model_state = model.state_dict()
        print()

    print((f"Training for {options.num_epochs} epochs took {time_track.total():.3f}s total "
           f"and {time_track.value():.3f}s average"))

    print(f"Saving best model and options to {options.save_path}")
    torch.save({
        'options': options,
        'model_state_dict': best_model_state,
    }, options.save_path)

    # for data, labels in test_loader:
    #     print(data.shape, labels.shape)
    #     print('test loader works!!')
    #     out = backbone(data)
    #     print(out.shape)
    #     print('single test batch went through model!!')
    #     break
    #
    # for data, labels, in valid_loader:
    #     print(data.shape, labels.shape)
    #     print('valid loader works!!')
    #     out = backbone(data)
    #     print(out.shape)
    #     print('single valid batch went through model!!')
    #     break
