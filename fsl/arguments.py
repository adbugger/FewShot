import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Two Phase Few Shot Learning")

    # Distributed stuff
    dist_args = parser.add_argument_group("Distributed Arguments")
    dist_args.add_argument("--local_rank", type=int, default=0)
    
    dist_args.add_argument("--distributed", dest='distributed', action='store_true')
    dist_args.add_argument("--no_distributed", dest='distributed', action='store_false')
    dist_args.set_defaults(distributed=True)

    # Train Arguments
    train_args = parser.add_argument_group("Train Arguments")
    train_args.add_argument("--num_epochs", type=int, default=10)
    train_args.add_argument("--save_path", type=str, default='exp0.pth')

    train_args.add_argument("--save_model", dest='save_model', action='store_true')
    train_args.add_argument("--no_save_model", dest='save_model', action='store_false')
    train_args.set_defaults(save_model=True)

    train_args.add_argument("--use_trainval", dest='use_trainval', action='store_true')
    train_args.add_argument("--no_use_trainval", dest='use_trainval', action='store_false')
    train_args.set_defaults(use_trainval=False)

    # Few Shot Arguments
    fewshot_args = parser.add_argument_group("Fewshot Arguments")
    fewshot_args.add_argument("--load_from", type=argparse.FileType('r'))
    fewshot_args.add_argument("--episode_strat", type=str,
        choices=['SimpleShotEpisodes'], default='SimpleShotEpisodes')
    fewshot_args.add_argument("--testing_strat", type=str,
        choices=['Classify1NN', 'SoftCosAttn'], default='Classify1NN')
    fewshot_args.add_argument("--n_way", type=int, default=5)
    fewshot_args.add_argument("--k_shot", type=int, default=5)
    fewshot_args.add_argument("--num_test_tasks", type=int, default=int(1e4))
    fewshot_args.add_argument("--num_query", type=int, default=15)
    
    fewshot_args.add_argument("--centroid", dest='centroid', action='store_true')
    fewshot_args.add_argument("--no_centroid", dest='centroid', action='store_false')
    fewshot_args.set_defaults(centroid=False)

    # Fine Tune Arguments
    ft_args = parser.add_argument_group("Fine Tune Arguments")
    ft_args.add_argument("--data_percent", type=int, default=20)
    ft_args.add_argument("--fine_tune_epochs", type=int, default=20)
    ft_args.add_argument("--fine_tune_save_path", type=str, default='ft0.pth')

    # Optimizer Arguments
    opt_args = parser.add_argument_group("Optimizer Arguments")
    opt_args.add_argument("--base_learning_rate", type=float, default=1e-2)

    opt_args.add_argument("--simple_opt", dest='simple_opt', action='store_true')
    opt_args.add_argument("--complex_opt", dest='simple_opt', action='store_false')
    opt_args.set_defaults(simple_opt=True)

    opt_args.add_argument("--base_optimizer", type=str, default="SGD")
    # opt_args.add_argument("--secondary_optimizer", type=str, default="LARS")

    opt_args.add_argument("--momentum", type=float, default=0.0)
    opt_args.add_argument("--weight_decay", type=float, default=0.0)
    opt_args.add_argument("--dampening", type=float, default=0.0)

    opt_args.add_argument("--nesterov", dest='nesterov', action='store_true')
    opt_args.add_argument("--no_nesterov", dest='nesterov', action='store_false')
    opt_args.set_defaults(nesterov=False)

    # Scheduler Arguments
    sched_args = parser.add_argument_group("Scheduler Arguements")
    sched_args.add_argument("--scheduler", type=str, default="CosineAnnealingLR")
    sched_args.add_argument("--T_max", type=int, default=20)

    # Model Arugments
    model_args = parser.add_argument_group("Model Arguments")
    model_args.add_argument("--model", type=str, default='SimCLRModel',
                            choices=['SimCLRModel', 'MoCoModel', 'SelfLabelModel'])

    # Backbone arguments
    backbone_args = parser.add_argument_group("Backbone Arguments")
    backbone_args.add_argument("--backbone", type=str, default='resnet50',
                                choices=['resnet50', 'resnet18'])

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
    data_args.add_argument("--dataset", type=str, default='cifar100fs',
            choices=['cifar100fs', 'fc100', 'miniImagenet', 'Imagenet1k'])
    data_args.add_argument("--dataset_root", type=str, required=False)

    # Data loader arguments
    dataloader_args = parser.add_argument_group("Data Loader Arguments")
    dataloader_args.add_argument("--batch_size", type=int, default=256)

    dataloader_args.add_argument("--shuffle", dest='shuffle', action='store_true')
    dataloader_args.add_argument("--no_shuffle", dest='shuffle', action='store_false')
    dataloader_args.set_defaults(shuffle=True)

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

    # Do PCA on features before sending to classifier
    pre_class_args = parser.add_argument_group("Pre Classifier Pipeline Arguments")
    pre_class_args.add_argument("--scaler", dest='scaler', action='store_true')
    pre_class_args.add_argument("--no_scaler", dest='scaler', action='store_false')
    pre_class_args.set_defaults(scaler=True)

    pre_class_args.add_argument("--ipca", dest="ipca", action='store_true')
    pre_class_args.add_argument("--no_ipca", dest="ipca", action='store_false')
    pre_class_args.set_defaults(ipca=False)

    pre_class_args.add_argument("--ipca_dim", type=int, default=128)

    # Test set kmeans evaluation
    eval_args = parser.add_argument_group("Test Evaluation Arguments")
    eval_args.add_argument("--eval_freq", type=int, default=1, help="Evaluate KMeans on test set after these many epochs")

    # Logging arguments
    log_args = parser.add_argument_group("Logging Arguments")
    log_args.add_argument("--log_file", type=argparse.FileType('a'), required=False)

    options = parser.parse_args()
    return options
