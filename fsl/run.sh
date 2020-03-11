#!/bin/bash
#SBATCH -A aditya.bharti
#SBATCH --cpus-per-task=12
#SBATCH --gpus=4
#SBATCH --time=2-00:00:00

EXP_NAME="exp12";

source "/home/aditya.bharti/python_env/bin/activate";
pushd "/home/aditya.bharti/FewShot/fsl";
# commands
python -m torch.distributed.launch --nproc_per_node=4 main.py \
    --complex_opt \
    --num_epochs=200 --batch_size=2048 \
    --nesterov --momentum=5e-2 --weight_decay=1e-6 \
    --T_max=200 \
    --save_path="saves/${EXP_NAME}.pth" |& tee "saves/${EXP_NAME}.out";
popd;
