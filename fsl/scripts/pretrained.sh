#!/bin/bash
#SBATCH -A aditya.bharti
#SBATCH --cpus-per-gpu=3
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=aditya.bharti@research.iiit.ac.in
#SBATCH --mail-type=END
#SBATCH --job-name=pre50

# cuda and cudnn already loaded in .bashrc
function test_moco {
  dataset="$1";
  test_strat="$2";
  num_shot="$3";

  moco_weights="pretrained_weights/moco_v2_800ep_pretrain.pth.tar";
  save_file="tests_moco/tests_moco.out";
  mkdir -p $( dirname "$save_file" );

  echo -n "${dataset} 5-way-${num_shot}-shot ${test_strat} " | tee -a "$save_file";
  echo;

  python -u fsl/few_shot.py --no_distributed \
  --model="MoCoModel" --dataset="$dataset" --load_from="$moco_weights" \
  --n_way=5 --k_shot="$num_shot" --testing_strat="$test_strat" \
  --log_file="$save_file";
}

source "/home/aditya.bharti/python_env/bin/activate";
pushd "/home/aditya.bharti/FewShot";

# iterate over datasets and num shot and testing strategy
for dataset in "miniImagenet" "cifar100fs" "fc100"; do
  for testing_strat in "Classify1NN" "SoftCosAttn"; do
    for num_shot in 1 5; do
      test_moco "$dataset" "$testing_strat" "$num_shot";
    done;
  done;
done;

popd;
