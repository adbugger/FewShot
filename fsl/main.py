from __future__ import division

import time

from torch.utils.data import DataLoader
import torch.optim as optim

from utils import getattr_or_default, get_loader, seed_everything, AverageMeter
from arguments import parse_args
import datasets
import models

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

    base_optimizer = getattr(optim, options.base_optimizer)(
                                model.parameters(),
                                lr=0.3*options.batch_size/256)
    optimizer = LARS(optimizer=base_optimizer)

    print(("Starting Training\n"
           "-----------------"))

    time_track = AverageMeter()
    for epoch in range(options.num_epochs):
        loss_track = AverageMeter()
        epoch_start = time.time()

        for aug1, aug2, targets in train_loader:
            optimizer.zero_grad()
            aug1 = aug1.cuda()
            aug2 = aug2.cuda()
            loss = model(aug1, aug2)
            loss.backward()
            optimizer.step()

            loss_track.accumulate(loss.item())

        time_track.accumulate(time.time() - epoch_start)

        print((f"({time_track.latest():>10.3f}s) Epoch {epoch+1:0>3}/{options.num_epochs:>3}: "
               f"Loss={loss_track.value():<f}"))
    print((f"Training for {options.num_epochs} epochs took {time_track.total():.3f}s total "
           f"and {time_track.value():.3f}s average"))
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
