"""Load resnet50 weights from MoCo model"""
# Code adapted from https://github.com/facebookresearch/moco/blob/master/moco/builder.py

import torch
import torch.nn as nn
import torchvision.models


def MoCoModel(options):
    # Load the file options.load_from
    load_dict = torch.load(options.load_from.name)

    # Init empty network
    network = getattr(torchvision.models, load_dict['arch'])(options)

    # The pretrained weights have mlp=True. Make the appropriate changes.
    # Adapted from https://github.com/facebookresearch/moco/blob/master/moco/builder.py#L29
    dim_mlp = network.fc.weight.shape[1]
    network.fc = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 128))

    # Rename the state_dict keys. We only need the encoder network.
    # Adapted from https://github.com/facebookresearch/moco/blob/master/main_lincls.py#L161
    state_dict = load_dict['state_dict']
    for k in list(state_dict.keys()):
        # we only need encoder_q
        if k.startswith("module.encoder_q"):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # remove unused or renamed key
        del state_dict[k]

    # Load the damn thing, and hope to god this hack works
    network.load_state_dict(state_dict)

    options.projection_dim = 128
    options.backbone_output_size = 2048
    return network
