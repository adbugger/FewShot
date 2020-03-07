from __future__ import division

import time

import torch
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

    model = getattr(models, options.model)(options).cuda()
    print(model)

    criterion = getattr(losses, options.loss_function)(options)

    base_optimizer = optim.SGD(model.parameters(),
                                lr=0.3*options.batch_size/256,
                                weight_decay=1e-6)
    optimizer = LARS(optimizer=base_optimizer)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=options.num_epochs)

    print(("Starting Training\n"
           "-----------------"))
    time_track = AverageMeter()
    for epoch in range(options.num_epochs):
        loss_track = AverageMeter()
        epoch_start = time.time()

        for aug1, aug2, targets in train_loader:
            optimizer.zero_grad()
            feat1 = model(aug1.cuda())
            feat2 = model(aug2.cuda())
            loss = criterion(feat1, feat2)
            loss.backward()
            optimizer.step()
            loss_track.accumulate(loss.item())

        scheduler.step()
        time_track.accumulate(time.time() - epoch_start)

        print((f"({time_track.latest():>10.3f}s) Epoch {epoch+1:0>3}/{options.num_epochs:>3}: "
               f"Loss={loss_track.value():<f}"))
    print((f"Training for {options.num_epochs} epochs took {time_track.total():.3f}s total "
           f"and {time_track.value():.3f}s average"))

    print(f"Saving model and options to {options.save_path}")
    torch.save({
        'options': options,
        'model_state_dict': model.state_dict(),
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

    print('done!!!')
