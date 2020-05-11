import torchvision.models as tv_models


def resnet50(options):
    options.backbone_output_size = 1000
    return tv_models.resnet50(pretrained=options.pretrained,
                              progress=options.progress)

def resnet18(options):
    options.backbone_output_size = 1000
    return tv_models.resnet18(pretrained=options.pretrained,
                              progress=options.progress)


if __name__ == "__main__":
    from argparse import Namespace
    import torch

    opts = Namespace(pretrained=False, progress=True)
    backbone = resnet18(opts).cuda()

    # B, C, H, W
    inp = torch.randn(5,3,32,32).cuda()
    print(backbone(inp).size())
