#!/bin/bash
#SBATCH -A aditya.bharti
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mail-user=aditya.bharti@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --job-name=testers

# cuda and cudnn already loaded in .bashrc
function tester {
    test_name="$1";
    save_dir="$2";
    load_from="$3";

    mkdir -p "$save_dir";

    out_file="$save_dir/$test_name.out"

    python -m torch.distributed.launch --nproc_per_node=1 few_shot.py \
        --load_from="$load_from" --log_file="$out_file" \
        --n_way=5 --k_shot=1;
}

# cuda and cudnn already loaded in .bashrc
source "/home/aditya.bharti/python_env/bin/activate";
pushd "/home/aditya.bharti/FewShot/fsl";

{
    test_dir="tests";
    for pth_file in $( find "fc100_experiments/" -type f -name "*.pth" -exec readlink -f {} \; ); do
        echo "doing $pth_file";
        tester "fc100_runs" "$test_dir" "$pth_file";
    done;
}

{
    test_dir="tests";
    for pth_file in $( find "miniImgnet_experiments/" -type f -name "*.pth" -exec readlink -f {} \; ); do
        echo "doing $pth_file";
        tester "miniImgnet_runs" "$test_dir" "$pth_file";
    done;
}

popd;
