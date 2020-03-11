#!/bin/bash
#SBATCH -A aditya.bharti
#SBATCH --cpus-per-task=10
#SBATCH --gpus=4
#SBATCH --time=2-00:00:00

EXP_NAME="exp11";

source "/home/aditya.bharti/python_env/bin/activate";
pushd "/home/aditya.bharti/FewShot/fsl";
# commands
python main.py --complex_opt \
    --num_epochs=200 --batch_size=4096 \
    --nesterov --momentum=5e-2 --weight_decay=1e-6 \
    --T_max=200 \
    --save_path="saves/${EXP_NAME}.pth" |& tee "saves/${EXP_NAME}.out";
popd;
