#!/bin/bash
#SBATCH -A aditya.bharti
#SBATCH --cpus-per-task=10
#SBATCH --gpus=4
#SBATCH --time=2-00:00:00

EXP_NAME="exp10";

source "/home/aditya.bharti/python_env/bin/activate";
pushd "/home/aditya.bharti/FewShot/fsl";
# commands
python main.py --complex_opt \
    --num_epochs=200 --batch_size=4096 \
    --no_nesterov --momentum=5e-2 --weight_decay=1e-6 \
    --T_max=200 \
    --save_path="${EXP_NAME}.pth" |& tee "${EXP_NAME}.out";
popd;
