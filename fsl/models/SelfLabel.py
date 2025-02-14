"""Load resnet50 model weights from self-label VGG model"""
# Code adapted from https://github.com/yukimasano/self-label

import torch
import torch.nn as nn
import torch.nn.functional as F

# Code taken from https://github.com/yukimasano/self-label/blob/master/models/resnetv2.py
class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    def __init__(self, in_planes, planes, stride=1,expansion=4):
        super(PreActBottleneck, self).__init__()
        self.expansion = expansion
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,expansion=4):
        super(PreActResNet, self).__init__()
        self.in_planes = 16*expansion

        self.features = nn.Sequential(*[
            nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(block, 16*expansion, num_blocks[0], stride=1, expansion=4),
            self._make_layer(block, 2*16*expansion, num_blocks[1], stride=2, expansion=4),
            self._make_layer(block, 4*16*expansion, num_blocks[2], stride=2, expansion=4),
            self._make_layer(block, 8*16*expansion, num_blocks[3], stride=2, expansion=4),
            nn.AdaptiveAvgPool2d((1, 1))
        ])
        self.headcount = len(num_classes)
        if len(num_classes) == 1:
            self.top_layer = nn.Sequential(*[nn.Linear(512*expansion, num_classes[0])]) # for later compatib.
        else:
            for a,i in enumerate(num_classes):
                setattr(self, "top_layer%d" % a, nn.Linear(512*expansion, i))
            self.top_layer = None  # this way headcount can act as switch.

    def _make_layer(self, block, planes, num_blocks, stride,expansion):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,expansion))
            self.in_planes = planes * expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        if self.headcount == 1:
            if self.top_layer:
                out = self.top_layer(out)
            return out
        else:
            outp = []
            for i in range(self.headcount):
                outp.append(getattr(self, "top_layer%d" % i)(out))
            return outp



def PreActResNet50(num_classes):
    return PreActResNet(PreActBottleneck, [3,4,6,3],num_classes)

def resnetv2(nlayers=50, num_classes=[1000], expansion=1):
    if nlayers == 50:
        return PreActResNet(PreActBottleneck, [3,4,6,3], num_classes, expansion=4*expansion)
    else:
        raise NotImplementedError


class SelfLabelModel(nn.Module):
    def __init__(self, options):
        super(SelfLabelModel, self).__init__()
        # Load the file options.load_from
        load_dict = torch.load(options.load_from.name)

        # Init empty network
        model = resnetv2(50, [3000]*10, 1)

        # Load weights
        model.load_state_dict(load_dict)

        options.backbone_output_size = 2048
        options.projection_dim = 2048

        # We delete the "top_layers" since we're only interested in
        # the representations. They are Linear layers anyway.
        delattr(model, "top_layer")
        for i in range(10):
            delattr(model, f"top_layer{i}")
        
        self.model = model.features

    def forward(self, x):
        out = self.model(x)
        # out.squeeze().shape == [batch_size, 2048]
        return out.squeeze()

if __name__=="__main__":
    import torch
    from argparse import Namespace

    opts = Namespace(load_from=Namespace(name="/home/aditya.bharti/FewShot/pretrained_weights/self-label-resnet-10x3k.pth"))
    with torch.no_grad():
        model = SelfLabelModel(opts).cuda()
        x = torch.randn(32,3,84,84).cuda()
        out = model(x)
        print(out.shape)