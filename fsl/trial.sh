#!/bin/bash
#SBATCH -A research
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mail-user=aditya.bharti@research.iiit.ac.in
#SBATCH --mail-type=END
#SBATCH --job-name=tester

# cuda and cudnn already loaded in .bashrc
function tester {
    test_name="$1";
    save_dir="$2";
    load_from="$3";
    
    echo "testing $1 $2 $3";
    mkdir -p "$save_dir";
    out_file="$save_dir/$test_name.out"

    # python -m torch.distributed.launch --nproc_per_node=1
    python few_shot.py --no_distributed \
        --load_from="$load_from" --log_file="$out_file" \
        --n_way=5 --k_shot=1;
}

source "/home/aditya.bharti/python_env/bin/activate";
pushd "/home/aditya.bharti/FewShot/fsl";

for pth_file in $( find "miniImgnet_round3" -type f -name "*.pth" -exec readlink -f {} \; ); do
    tester "fc100_round3" "tests" "$pth_file";
done;

popd;
