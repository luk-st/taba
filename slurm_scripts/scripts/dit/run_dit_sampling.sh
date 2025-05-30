#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --account=<SLURM_ACCOUNT_NAME>
#SBATCH --partition=<SLURM_PARTITION_NAME>
#SBATCH --output=slurm_out/dit/guidance/guidance_1.0_cond_seed_10.out
#SBATCH --job-name=dit_guidance_1.0_cond_seed_10

module load ML-bundle/24.06a
cd <PROJECT_PATH>
export PYTHONPATH=$PWD
source ./venv/bin/activate

echo USER: $USER
which python

accelerate launch --num_processes 4 taba/scripts/sampling/run_dit_sampling.py --seed 420 --noises_per_prompt 8 --n_prompts 1024 --batch_size 32 --num_inference_steps 100 --guidance_scale 1.0 --cond_seed 10 --with_inversion --with_reconstruction

# --input_path <INP_PATH>