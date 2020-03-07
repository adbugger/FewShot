#!/bin/bash
#SBATCH -A aditya.bharti
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00

source "/home/aditya.bharti/python_env/bin/activate";

pushd "/home/aditya.bharti/fsl";
# commands
python main.py --shuffle --num_epochs=10;
popd;
