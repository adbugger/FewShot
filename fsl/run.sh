#!/bin/bash
#SBATCH -A aditya.bharti
#SBATCH --cpus-per-task=12
#SBATCH --gpus=4
#SBATCH --time=12:00:00
#SBATCH --mail-user=aditya.bharti@research.iiit.ac.in
#SBATCH --mail-type=ALL

NUM_EPOCHS=50;
function runner {
    exp_name="$1";
    save_dir="$2";
    aug1="$3";
    aug2="$4";

    mkdir -p "$save_dir";

    out_file="$save_dir/$exp_name.out"
    model_save="$save_dir/$exp_name.pth"

    python -m torch.distributed.launch --nproc_per_node=4 main.py \
        --complex_opt \
        --eval_freq=5 \
        --first_augment="$aug1" --second_augment="$aug2" \
        --num_epochs=$NUM_EPOCHS --batch_size=1024 \
        --base_learning_rate=0.01 --nesterov --momentum=0.9 --weight_decay=1e-4 \
        --T_max=$NUM_EPOCHS \
        --save_path="$model_save" --log_file="$out_file";
}


# cuda and cudnn already loaded in .bashrc
source "/home/aditya.bharti/python_env/bin/activate";
pushd "/home/aditya.bharti/FewShot/fsl";
# runner exp_name save_dir aug1 aug2

{
    exp2_dir="CropResize_GaussBlur_runs";
    runner "run1" "$exp2_dir" "CropResize" "GaussBlur";
    runner "run2" "$exp2_dir" "CropResize" "GaussBlur";
    runner "run3" "$exp2_dir" "CropResize" "GaussBlur";
    runner "run4" "$exp2_dir" "CropResize" "GaussBlur";
}

{
    exp3_dir="ColorDistort_GaussBlur_runs";
    runner "run1" "$exp3_dir" "ColorDistort" "GaussBlur";
    runner "run2" "$exp3_dir" "ColorDistort" "GaussBlur";
    runner "run3" "$exp3_dir" "ColorDistort" "GaussBlur";
    runner "run4" "$exp3_dir" "ColorDistort" "GaussBlur";
}

popd;
