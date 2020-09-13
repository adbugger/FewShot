#!/bin/bash
#SBATCH -A research
#SBATCH --cpus-per-gpu=10
#SBATCH --gpus=4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=aditya.bharti@research.iiit.ac.in
#SBATCH --mail-type=END
#SBATCH --job-name=moco_specific

# cuda and cudnn already loaded in .bashrc
function test_moco {
  dataset="$1";
  test_strat="$2";
  num_shot="$3";

  moco_weights="pretrained_weights/moco_v2_800ep_pretrain.pth.tar";
  save_file="tests_centroid/moco_centroid.out";
  mkdir -p $( dirname "$save_file" );

  echo -n "${moco_weights} ${dataset} 5-way-${num_shot}-shot ${test_strat} " | tee -a "$save_file";
  echo;

  python -u fsl/few_shot.py --no_distributed \
  --model="MoCoModel" --dataset="$dataset" --load_from="$moco_weights" \
  --n_way=5 --k_shot="$num_shot" --testing_strat="$test_strat" \
  --log_file="$save_file";
}

function test_moco_on_dataset {
  moco_weights="$1";
  dataset="$2";
  test_strat="$3";
  num_shot="$4";
  save_file="$5";
  
  mkdir -p $( dirname "$save_file" );

  echo -n "$(basename -z "$moco_weights" .pth.tar) ${dataset} 5-way-${num_shot}-shot ${test_strat} " | tee -a "$save_file";
  echo;

  # python -u fsl/few_shot.py --no_distributed \
  python -u -m torch.distributed.launch --nproc_per_node=4 fsl/few_shot.py --distributed \
  --model="MoCoModel" --dataset="$dataset" --load_from="$moco_weights" \
  --n_way=5 --k_shot="$num_shot" --testing_strat="$test_strat" \
  --log_file="$save_file";
}

source "/home/aditya.bharti/python_env/bin/activate";
pushd "/home/aditya.bharti/FewShot";

save_dir="moco_high_lr";
# dataset testing_strat num_shot num_episodes
for num_episodes in 200 400 600 800; do
  save_file="${save_dir}/${num_episodes}.out";
  for testing_strat in "Classify1NN" "SoftCosAttn"; do
    for num_shot in 1 5; do
      test_moco_on_dataset "/home/aditya.bharti/moco-mod/moco-mini-imagenet-${num_episodes}ep.pth.tar" "miniImagenet" "$testing_strat" "$num_shot" "$save_file";
      test_moco_on_dataset "/home/aditya.bharti/moco-mod/moco-cifar100-${num_episodes}ep.pth.tar" "cifar100fs" "$testing_strat" "$num_shot" "$save_file";
      test_moco_on_dataset "/home/aditya.bharti/moco-mod/moco-fc100-${num_episodes}ep.pth.tar" "fc100" "$testing_strat" "$num_shot" "$save_file";
    done;
  done;
done;
popd;
