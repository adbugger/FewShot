#!/bin/bash
#SBATCH -A aditya.bharti
#SBATCH --cpus-per-gpu=3
#SBATCH --gpus=4
#SBATCH --mem-per-cpu=4096
#SBATCH --time=3-00:00:00
#SBATCH --job-name=fc100
#SBATCH --mail-user=aditya.bharti@research.iiit.ac.in
#SBATCH --mail-type=END
#SBATCH --nodes=1

NUM_EPOCHS=500;
HALF_EPOCHS=250;
function runner {
  exp_name="$1";
  save_dir="$2";
  aug1="$3";
  aug2="$4";

  echo "doing $1 $2 $3 $4";
  mkdir -p "$save_dir";

  out_file="$save_dir/$exp_name.out"
  model_save="$save_dir/$exp_name.pth"

  # python -u fsl/main.py --no_distributed \
  python -u -m torch.distributed.launch --nproc_per_node=4 fsl/main.py --distributed \
    --simple_opt --dataset="fc100" --backbone="resnet50" --use_trainval \
    --eval_freq=2000 --loss_function="NTXent" \
    --first_augment="$aug1" --second_augment="$aug2" \
    --num_epochs=$NUM_EPOCHS --batch_size=256 \
    --base_learning_rate=0.1 --nesterov --momentum=0.9 --weight_decay=1e-4 \
    --T_max=$HALF_EPOCHS --ntxent_temp=0.1 \
    --save_path="$model_save" --log_file="$out_file";
}

# cuda and cudnn already loaded in .bashrc
source "/home/aditya.bharti/python_env/bin/activate";
pushd "/home/aditya.bharti/FewShot";
# runner exp_name save_dir aug1 aug2

exp_dir=$( readlink -f "fc100_hyperparam" );
runner "CropGauss" "$exp_dir" "CropResize" "GaussBlur";
runner "ColorGauss" "$exp_dir" "ColorDistort" "GaussBlur";
runner "CropColor" "$exp_dir" "CropResize" "ColorDistort";

popd;
