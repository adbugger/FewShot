import torchvision.models as tv_models

__all__ = ['resnet50', 'resnet18']

def resnet50(options):
    options.backbone_output_size = 1000
    return tv_models.resnet50(pretrained=options.pretrained,
                              progress=options.progress)

def resnet18(options):
    options.backbone_output_size = 1000
    return tv_models.resnet18(pretrained=options.pretrained,
                              progress=options.progress)