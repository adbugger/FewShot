#!/bin/bash
#SBATCH -A research
#SBATCH --cpus-per-gpu=3
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=aditya.bharti@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --job-name=trainval_tests
#SBATCH --nodes=1

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
        --load_from="$load_from" --log_file="$out_file" \
        --n_way=5 --k_shot="$num_shot" --testing_strat="$test_strat";
}

source "/home/aditya.bharti/python_env/bin/activate";
pushd "/home/aditya.bharti/FewShot";

# trainval with both test strats
for pth_file in $( find *_trainval/ -type f -name "*.pth"; ); do
    for class in "Classify1NN" "SoftCosAttn"; do
        for ns in 1 5; do
            # tester filename directory model_file num_shot test_strat
            tester "trainval_5way_${ns}shot" "tests_trainval" "$pth_file" "$ns" "$class";
        done;
    done;
done;

# tester "trainval_5way_5shot" "tests_trainval" "cifar100fs_trainval/CropGauss.pth" "5" "SoftCosAttn";

popd;
