#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --account=<SLURM_ACCOUNT_NAME>
#SBATCH --partition=<SLURM_PARTITION_NAME>
#SBATCH --output=slurm_out/ldm/internal_10k_T50.out
#SBATCH --job-name=ldm_internal_10k_T50

module load ML-bundle/24.06a
cd <PROJECT_PATH>
export PYTHONPATH=$PWD
source ./venv/bin/activate

echo USER: $USER
which python

accelerate launch --num_processes 4 taba/scripts/sampling/run_ldm_sampling.py --num_inference_steps 50 --with_inversion --seed 420 --batch_size 32 --n_samples 10240 --internal --with_reconstruction

# --input_path <INP_PATH> --with_reconstruction --internal