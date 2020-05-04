#!/bin/bash
#SBATCH -A aditya.bharti
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-cpu=4096
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --job-name=fc100
#SBATCH --mail-user=aditya.bharti@research.iiit.ac.in
#SBATCH --mail-type=END

NUM_EPOCHS=100;
function runner {
    exp_name="$1";
    save_dir="$2";
    aug1="$3";
    aug2="$4";

    echo "doing $1 $2 $3 $4";
    mkdir -p "$save_dir";

    out_file="$save_dir/$exp_name.out"
    model_save="$save_dir/$exp_name.pth"

    # python -m torch.distributed.launch --nproc_per_node=4
    python -u main.py --num_workers=2 \
        --no_distributed \
        --complex_opt --dataset="fc100" \
        --eval_freq=5 --loss_function="NTXent" \
        --first_augment="$aug1" --second_augment="$aug2" \
        --num_epochs=$NUM_EPOCHS --batch_size=256 \
        --base_learning_rate=0.1 --nesterov --momentum=0.9 --weight_decay=1e-4 \
        --T_max=$NUM_EPOCHS \
        --save_path="$model_save" --log_file="$out_file";
}

# cuda and cudnn already loaded in .bashrc
source "/home/aditya.bharti/python_env/bin/activate";
pushd "/home/aditya.bharti/FewShot/fsl";
# runner exp_name save_dir aug1 aug2

exp2_dir="fc100_round3/CropResize_GaussBlur_runs";
CUDA_VISIBLE_DEVICES=0 runner "run1" "$exp2_dir" "CropResize" "GaussBlur" &
CUDA_VISIBLE_DEVICES=1 runner "run2" "$exp2_dir" "CropResize" "GaussBlur" &
CUDA_VISIBLE_DEVICES=2 runner "run3" "$exp2_dir" "CropResize" "GaussBlur" &
CUDA_VISIBLE_DEVICES=3 runner "run4" "$exp2_dir" "CropResize" "GaussBlur" &
 wait;

exp3_dir="fc100_round3/ColorDistort_GaussBlur_runs";
CUDA_VISIBLE_DEVICES=0 runner "run1" "$exp3_dir" "ColorDistort" "GaussBlur" &
CUDA_VISIBLE_DEVICES=1 runner "run2" "$exp3_dir" "ColorDistort" "GaussBlur" &
CUDA_VISIBLE_DEVICES=2 runner "run3" "$exp3_dir" "ColorDistort" "GaussBlur" &
CUDA_VISIBLE_DEVICES=3 runner "run4" "$exp3_dir" "ColorDistort" "GaussBlur" &
wait;

exp4_dir="fc100_round3/CropResize_ColorDistort_runs";
CUDA_VISIBLE_DEVICES=0 runner "run1" "$exp4_dir" "CropResize" "ColorDistort" &
CUDA_VISIBLE_DEVICES=1 runner "run2" "$exp4_dir" "CropResize" "ColorDistort" &
CUDA_VISIBLE_DEVICES=2 runner "run3" "$exp4_dir" "CropResize" "ColorDistort" &
CUDA_VISIBLE_DEVICES=3 runner "run4" "$exp4_dir" "CropResize" "ColorDistort" &
wait;

popd;
