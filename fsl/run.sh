#!/bin/bash
#SBATCH -A aditya.bharti
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00

EXP_NAME="exp1";

source "/home/aditya.bharti/python_env/bin/activate";
pushd "/home/aditya.bharti/FewShot/fsl";
# commands
python main.py --shuffle --num_epochs=200 --T_max=20 --save_path="${EXP_NAME}.pth" |& tee "${EXP_NAME}.out";
popd;
