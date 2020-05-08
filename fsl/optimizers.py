import sys

from torch import optim

__all__ = ['get_optimizer', 'get_scheduler']


def get_optimizer(model, options):
    base_optimizer = getattr(optim, options.base_optimizer)(model.parameters(),
        lr=options.base_learning_rate * options.batch_size / 256,
        momentum=options.momentum,
        weight_decay=options.weight_decay,
        dampening=options.dampening,
        nesterov=options.nesterov)

    return base_optimizer

def get_scheduler(optimizer, options):
    return getattr(optim.lr_scheduler, options.scheduler)(
        optimizer, T_max=options.T_max
    )
