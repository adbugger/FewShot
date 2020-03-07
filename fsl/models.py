import torch
nn = torch.nn

import backbones
import heads
import losses

class SimCLRModel(nn.Module):
    def __init__(self, options):
        super(SimCLRModel, self).__init__()

        self.backbone = getattr(backbones, options.backbone)(options)
        self.head = getattr(heads, options.head)(options)
        self.criterion = getattr(losses, options.loss_function)(options)

    def forward(self, x, y):
        feat1 = self.head(self.backbone(x))
        feat2 = self.head(self.backbone(y))
        loss = self.criterion(feat1, feat2)
        return loss
