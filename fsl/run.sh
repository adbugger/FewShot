#!/bin/bash
#SBATCH -A aditya.bharti
#SBATCH --cpus-per-task=12
#SBATCH --gpus=4
#SBATCH --time=12:00:00
#SBATCH --mail-user=aditya.bharti@research.iiit.ac.in
#SBATCH --mail-type=ALL

EXP_NAME="exp15";
NUM_EPOCHS=10;
OUT_FILE="saves/${EXP_NAME}.out";

# cuda and cudnn already loaded in .bashrc

source "/home/aditya.bharti/python_env/bin/activate";
pushd "/home/aditya.bharti/FewShot/fsl";
# commands
python -m torch.distributed.launch --nproc_per_node=4 main.py \
    --complex_opt \
    --eval_freq=5 \
    --first_augment="CropResize" --second_augment="GaussBlur" \
    --num_epochs=$NUM_EPOCHS --batch_size=1024 \
    --nesterov --momentum=5e-2 --weight_decay=1e-6 \
    --T_max=$NUM_EPOCHS \
    --save_path="saves/${EXP_NAME}.pth" --log_file="$OUT_FILE";
popd;
