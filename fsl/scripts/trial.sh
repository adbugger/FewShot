#!/bin/bash
#SBATCH -A research
#SBATCH --cpus-per-gpu=3
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=aditya.bharti@research.iiit.ac.in
#SBATCH --mail-type=END
#SBATCH --job-name=resnet18_tests

# cuda and cudnn already loaded in .bashrc
function tester {
    test_name="$1";
    save_dir="$2";
    load_from="$3";
    num_shot="$4";
    test_strat="$5";
    
    mkdir -p "$save_dir";
    out_file="$save_dir/$test_name.out";
    echo -n "Doing $3, saving to ${out_file}, ";
    echo -n "5way-${num_shot}shot ${test_strat} " | tee -a "$out_file";
    echo;

    # python -u fsl/few_shot.py --no_distributed \
    python -u -m torch.distributed.launch --nproc_per_node=1 fsl/few_shot.py --distributed \
        --model="MoCoModel" --dataset="miniImageNet"
        --load_from="$load_from" --log_file="$out_file" \
        --n_way=5 --k_shot="$num_shot" --testing_strat="$test_strat";
}

source "/home/aditya.bharti/python_env/bin/activate";
pushd "/home/aditya.bharti/FewShot";

# resnet18 with both test strats
for pth_file in $( find *_resnet18 -type f -name "*.pth" -exec readlink -f {} \; ); do
    # tester filename directory model_file num_shot test_strat
    for num_shot in 1 5; do
        tester "resnet18_5way_${num_shot}shot" "tests" "$pth_file" "$num_shot" "Classify1NN";
        tester "resnet18_5way_${num_shot}shot" "tests_matchnet" "$pth_file" "$num_shot" "SoftCosAttn";        
    done;
done;

# do the old resnet50 with SoftCosAttn strat
for pth_file in $( find *_500epoch *_1000epoch -type f -name "*.pth" -exec readlink -f {} \; ); do
    # tester filename directory model_file num_shot test_strat
    for num_shot in 1 5; do
        tester "resnet50_5way_${num_shot}shot" "tests_matchnet" "$pth_file" "$num_shot" "SoftCosAttn";
    done;
done;

popd;
