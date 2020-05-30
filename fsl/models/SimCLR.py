import torch.nn as nn
import backbones
import heads

class SimCLRModel(nn.Module):
    def __init__(self, options):
        super(SimCLRModel, self).__init__()

        self.backbone = getattr(backbones, options.backbone)(options)
        self.head = getattr(heads, options.head)(options)

    def forward(self, x):
        return self.head(self.backbone(x))