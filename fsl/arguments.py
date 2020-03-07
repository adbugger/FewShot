import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Two Phase Few Shot Learning")

    # Train Arguments
    train_args = parser.add_argument_group("Train Arguments")
    train_args.add_argument("--num_epochs", type=int, default=10)
    train_args.add_argument("--base_optimizer", type=str, default="SGD")
    # train_args.add_argument("--secondary_optimizer", type=str, default="LARS")
    # train_args.add_argument("--scheduler", type=str)

    # Model Arugments
    model_args = parser.add_argument_group("Model Arguments")
    model_args.add_argument("--model", type=str, default='SimCLRModel', choices=['SimCLRModel'])

    # Backbone arguments
    backbone_args = parser.add_argument_group("Backbone Arguments")
    backbone_args.add_argument("--backbone", type=str, default='resnet50', choices=['resnet50'])

    backbone_args.add_argument("--pretrained", dest='pretrained', action='store_true')
    backbone_args.add_argument("--no_pretrained", dest='pretrained', action='store_false')
    backbone_args.set_defaults(pretrained=False)

    backbone_args.add_argument("--progress", dest='progress', action='store_true')
    backbone_args.add_argument("--no_progress", dest='progress', action='store_false')
    backbone_args.set_defaults(progress=False)

    # Head arguments
    head_args = parser.add_argument_group("Projection Head Arguments")
    head_args.add_argument("--head", type=str, default="SimpleMLP", choices=["SimpleMLP"])
    head_args.add_argument("--projection_dim", type=int, default=128)

    # Data arguments
    data_args = parser.add_argument_group("Dataset Arguments")
    data_args.add_argument("--dataset", type=str, default='cifar100fs', choices=['cifar100fs'])
    data_args.add_argument("--dataset_root", type=str, required=False)

    # Data loader arguments
    dataloader_args = parser.add_argument_group("Data Loader Arguments")
    dataloader_args.add_argument("--batch_size", type=int, default=2048)

    dataloader_args.add_argument("--shuffle", dest='shuffle', action='store_true')
    dataloader_args.add_argument("--no_shuffle", dest='shuffle', action='store_false')
    dataloader_args.set_defaults(shuffle=False)

    dataloader_args.add_argument("--num_workers", type=int, default=4)

    dataloader_args.add_argument("--pin_memory", dest='pin_memory', action='store_true')
    dataloader_args.add_argument("--no_pin_memory", dest='pin_memory', action='store_false')
    dataloader_args.set_defaults(pin_memory=True)

    # Augmentation Transform arguments
    transform_args = parser.add_argument_group("Image Transform Arguments")
    transform_args.add_argument("--first_augment", type=str,
                                default='CropResize',
                                choices=['CropResize', 'ColorDistort', 'GaussBlur'])
    transform_args.add_argument("--second_augment", type=str,
                                default='GaussBlur',
                                choices=['CropResize', 'ColorDistort', 'GaussBlur'])
    transform_args.add_argument("--jitter_strength", type=float, default=1.0)

    # Loss function parameters
    loss_args = parser.add_argument_group("Loss Function Arguments")
    loss_args.add_argument("--loss_function", type=str, default="NTXent", choices=["NTXent"])
    loss_args.add_argument("--ntxent_temp", type=float, default=1.0)

    options = parser.parse_args()
    return options
