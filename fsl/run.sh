#!/bin/bash
#SBATCH -A aditya.bharti
#SBATCH --cpus-per-task=12
#SBATCH --gpus=4
#SBATCH --time=12:00:00
#SBATCH --mail-user=aditya.bharti@research.iiit.ac.in
#SBATCH --mail-type=ALL

EXP_NAME="exp15";
NUM_EPOCHS=10;

# cuda and cudnn already loaded in .bashrc
# pykeops requires cmake 1.10 minimum
module load cmake/3.15.2;

source "/home/aditya.bharti/python_env/bin/activate";
pushd "/home/aditya.bharti/FewShot/fsl";
# commands
python -m torch.distributed.launch --nproc_per_node=4 main.py \
    --complex_opt \
    --eval_freq=1 --cluster_iters=3 \
    --first_augment="CropResize" --second_augment="GaussBlur" \
    --num_epochs=$NUM_EPOCHS --batch_size=2048 \
    --nesterov --momentum=5e-2 --weight_decay=1e-6 \
    --T_max=$NUM_EPOCHS \
    --save_path="saves/${EXP_NAME}.pth" |& tee "saves/${EXP_NAME}.out";
popd;
