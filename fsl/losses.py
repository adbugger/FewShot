from __future__ import division
import torch
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
        assert x.shape == y.shape, ("Input to loss functions must be of same shape. "
            "Got {} and {}.".format(x.shape, y.shape))
        num_data = x.shape[0]
        feat_dim = x.shape[1]

        # each row is unit norm
        samples = nn.functional.normalize(torch.cat((x,y),dim=0), dim=1, p=2)
        assert samples.shape == (2*num_data, feat_dim), ("What did you do?? "
                    "Expected size {}, got {}.".format(
                        (2*num_data, feat_dim), samples.shape))

        sim_matrix = (torch.mm(samples, samples.transpose(1,0)) / self.temp).exp_()
        assert sim_matrix.shape == (2*num_data, 2*num_data)

        term1 = (sim_matrix.diag(num_data) + sim_matrix.diag(-num_data)).log_().sum()
        term2 = (sim_matrix.sum(dim=1) - math.exp(1 / self.temp)).log_().sum()

        loss = (-term1 + term2) / (2*num_data)

        return loss
