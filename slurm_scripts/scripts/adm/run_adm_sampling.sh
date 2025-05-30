#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --account=<SLURM_ACCOUNT_NAME>
#SBATCH --partition=<SLURM_PARTITION_NAME>
#SBATCH --output=slurm_out/adm64/4k_t10.out
#SBATCH --job-name=adm_64_4k_T10

module load ML-bundle/24.06a
cd <PROJECT_PATH>
export PYTHONPATH=$PWD
source ./venv/bin/activate

echo USER: $USER
which python

accelerate launch --num_processes 4 taba/scripts/sampling/run_adm_sampling.py --model_name imagenet_pixel_64 --num_inference_steps 10 --with_inversion --seed 420 --batch_size 128 --n_samples 4096 --input_image_path experiments/corr_T/adm64/T100/samples.pt --save_dir experiments/corr_T/adm64/T10