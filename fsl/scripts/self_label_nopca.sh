#!/bin/bash
#SBATCH -A aditya.bharti
#SBATCH --cpus-per-gpu=3
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=aditya.bharti@research.iiit.ac.in
#SBATCH --mail-type=END
#SBATCH --job-name=cent_sf

# cuda and cudnn already loaded in .bashrc
function test_sf {
  dataset="$1";
  test_strat="$2";
  num_shot="$3";

  sf_weights="pretrained_weights/self-label-resnet-10x3k.pth";
  save_file="tests_centroid/sf_centroid.out";
  mkdir -p $( dirname "$save_file" );

  echo -n "${dataset} 5-way-${num_shot}-shot ${test_strat} " | tee -a "$save_file";
  echo;

  python -u fsl/few_shot.py --no_distributed \
  --model="SelfLabelModel" --dataset="$dataset" --load_from="$sf_weights" \
  --n_way=5 --k_shot="$num_shot" --testing_strat="$test_strat" \
  --no_ipca --centroid \
  --log_file="$save_file";
}

source "/home/aditya.bharti/python_env/bin/activate";
pushd "/home/aditya.bharti/FewShot";

# iterate over datasets and num shot and testing strategy
for dataset in "miniImagenet" "cifar100fs" "fc100"; do
  for testing_strat in "Classify1NN" "SoftCosAttn"; do
    test_sf "$dataset" "$testing_strat" 5;
  done;
done;

popd;
