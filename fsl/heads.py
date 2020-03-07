import torch
nn = torch.nn
F = nn.functional

class SimpleMLP(nn.Module):
    def __init__(self, options):
        super(SimpleMLP, self).__init__()
        l1_dim = min(512, options.backbone_output_size)

        self.l1 = nn.Linear(options.backbone_output_size, l1_dim)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(l1_dim, options.projection_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x
