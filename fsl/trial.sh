#!/bin/bash
#SBATCH -A research
#SBATCH --cpus-per-gpu=3
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=12:00:00
#SBATCH --mail-user=aditya.bharti@research.iiit.ac.in
#SBATCH --mail-type=END
#SBATCH --job-name=500ep_5way_1shot

# cuda and cudnn already loaded in .bashrc
function tester {
    test_name="$1";
    save_dir="$2";
    load_from="$3";
    num_shot="$4";
    
    echo "testing $1 $2 $3";
    mkdir -p "$save_dir";
    out_file="$save_dir/$test_name.out"

    # python -u fsl/few_shot.py --no_distributed \
    python -u -m torch.distributed.launch --nproc_per_node=1 fsl/few_shot.py  --distributed \
        --load_from="$load_from" --log_file="$out_file" \
        --n_way=5 --k_shot="$num_shot";
}

source "/home/aditya.bharti/python_env/bin/activate";
pushd "/home/aditya.bharti/FewShot";

for pth_file in $( find *_1000epoch -type f -name "*.pth" -exec readlink -f {} \; ); do
    # tester filename directory model_file num_shot
    tester "1000ep_5way_5shot" "tests" "$pth_file" 5;
    tester "1000ep_5way_1shot" "tests" "$pth_file" 1;
done;

popd;
