from __future__ import division

import torch
from pytorch_metric_learning.losses import NTXentLoss

__all__ = ["NTXent"]

# def NTXent(options):
#     loss_func = NTXentLoss(temperature=options.ntxent_temp)
#
#     def compute_loss(x, y):
#         assert x.shape == y.shape, (f"Input to loss functions must be of same shape. "
#             "Got {x.shape} and {y.shape}.")
#         assert x.device == y.device, (f"Both x and y must be on the same device. "
#             "Got {x.device} and {y.device}.")
#
#         num_data = x.shape[0]
#         feat_dim = x.shape[1]
#
#         all_feat = torch.cat((x, y), dim=0)
#         match_idx = torch.arange(0, num_data, device=x.device)
#         labels = torch.cat((match_idx, match_idx))
#
#         return loss_func(all_feat, labels)
#     return compute_loss

import math
nn = torch.nn

class NTXent(nn.Module):
    """
    Eq. 1 from https://arxiv.org/pdf/2002.05709.pdf .
    """
    def __init__(self, options):
        super(NTXent, self).__init__()
        self.temp = options.ntxent_temp

    def forward(self, x, y):
        assert x.shape == y.shape, (f"Input to loss functions must be of same shape. "
            "Got {x.shape} and {y.shape}.")
        num_data = x.shape[0]
        feat_dim = x.shape[1]

        # each row is unit norm
        samples = nn.functional.normalize(torch.cat((x,y),dim=0), dim=1, p=2)
        assert samples.shape == (2*num_data, feat_dim), (f"What did you do?? "
                    "Expected size {(2*num_data, feat_dim)}, got {samples.shape}.")

        sim_matrix = (torch.mm(samples, samples.transpose(1,0)) / self.temp).exp_()
        assert sim_matrix.shape == (2*num_data, 2*num_data)

        term1 = (sim_matrix.diag(num_data) + sim_matrix.diag(-num_data)).log_().sum()
        term2 = (sim_matrix.sum(dim=1) - math.exp(1 / self.temp)).log_().sum()

        loss = (-term1 + term2) / (2*num_data)

        return loss
