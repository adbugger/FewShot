from __future__ import division

import time
import math

import numpy as np

import torch
nn = torch.nn

from pytorch_metric_learning.losses import NTXentLoss

__all__ = ["NTXent", "PyMetricNTXent"]

class PyMetricNTXent(nn.Module):
    # https://github.com/KevinMusgrave/pytorch-metric-learning/issues/6
    def __init__(self, options):
        super().__init__()
        self.temp = options.ntxent_temp
        self.loss_func = NTXentLoss(temperature=self.temp)
    
    def forward(self, x, y):
        assert x.shape == y.shape, (f"Input to loss functions must be of same shape. "
            "Got {x.shape} and {y.shape}.")
        num_data = x.shape[0]

        embeddings = torch.cat((x, y))
        indices = torch.arange(0, num_data, device=x.device)
        labels = torch.cat((indices, indices))

        return self.loss_func(embeddings, labels)


class NTXent(nn.Module):
    """
    Eq. 1 from https://arxiv.org/pdf/2002.05709.pdf .
    """
    def __init__(self, options):
        super(NTXent, self).__init__()
        self.temp = options.ntxent_temp
        self.eps = 1e-8

    def forward(self, x, y):
        assert x.shape == y.shape, (f"Input to loss functions must be of same shape. "
            "Got {x.shape} and {y.shape}.")
        num_data = x.shape[0]
        feat_dim = x.shape[1]

        # each row is unit norm
        samples = nn.functional.normalize(torch.cat((x,y),dim=0), dim=1, p=2, eps=self.eps)
        assert samples.shape == (2*num_data, feat_dim), (f"What did you do?? "
                    "Expected size {(2*num_data, feat_dim)}, got {samples.shape}.")

        sim_matrix = (torch.mm(samples, samples.transpose(1,0)) / self.temp).exp_()
        assert sim_matrix.shape == (2*num_data, 2*num_data)

        term11 = (sim_matrix.diag(num_data)  + self.eps).log_().sum()
        term12 = (sim_matrix.diag(-num_data) + self.eps).log_().sum()
        term1 = term11 + term12
        term2 = (sim_matrix.sum(dim=1) - math.exp(1 / self.temp) + self.eps).log_().sum()

        loss = (-term1 + term2) / (2*num_data)

        return loss


