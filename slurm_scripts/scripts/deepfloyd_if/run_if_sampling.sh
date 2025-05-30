#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --account=<SLURM_ACCOUNT_NAME>
#SBATCH --partition=<SLURM_PARTITION_NAME>
#SBATCH --output=slurm_out/deepfloyd_if/guidance/guidance_1.0_cond_seed_42.out
#SBATCH --job-name=deepfloyd_guidance_1.0_cond_seed_42

module load ML-bundle/24.06a
cd <PROJECT_PATH>
export PYTHONPATH=$PWD
source ./venv/bin/activate

echo USER: $USER
which python

accelerate launch --num_processes 4 taba/scripts/sampling/run_if_sampling.py --seed 420 --noises_per_prompt 8 --n_prompts 1024 --batch_size 32 --num_inference_steps 100 --guidance_scale 1.0 --prompts dataset --cond_seed 42

# --input_path <INP_PATH>