import torchvision.models as tv_models


def resnet50(options):
    options.backbone_output_size = 1000
    return tv_models.resnet50(pretrained=options.pretrained,
                                 progress=options.progress)
