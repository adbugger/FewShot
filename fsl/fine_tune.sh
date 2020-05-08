#!/bin/bash
#SBATCH -A aditya.bharti
#SBATCH --cpus-per-gpu=3
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=24:00:00
#SBATCH --job-name=finetune
#SBATCH --mail-user=aditya.bharti@research.iiit.ac.in
#SBATCH --mail-type=END

function finetune {
    full_save_path="$1";
    load_file="$2";

    mkdir -p $( dirname "$full_save_path" );
    echo "fine tuning $full_save_path $load_file";

    python fsl/fine_tune.py \
        --load_from="$load_file" --save_path="$full_save_path" \
        --no_distributed --data_percent=10 --fine_tune_epochs=20 \
        --batch_size=256 --base_learning_rate=0.1 \
        --nesterov --momentum=0.9 --weight_decay=1e-4 --T_max=20;
}

source "/home/aditya.bharti/python_env/bin/activate";
pushd "/home/aditya.bharti/FewShot";

exp_dir="ft_10p20e";
for pth_file in $( find cifar100fs_round3 fc100_round3 miniImgnet_round3 -type f -name "*.pth" ); do
    finetune "$exp_dir/$pth_file" "$pth_file";
done;

popd;