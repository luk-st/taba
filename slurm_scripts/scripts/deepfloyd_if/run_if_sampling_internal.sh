#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --account=<SLURM_ACCOUNT_NAME>
#SBATCH --partition=<SLURM_PARTITION_NAME>
#SBATCH --output=slurm_out/deepfloyd_if/internal/change_cond_cond_seed_10_T50_5k_latents_swapeps10.out
#SBATCH --job-name=deepfloyd_internal_change_cond_cond_seed_10_T50_5k_latents_swapeps10

module load ML-bundle/24.06a
cd <PROJECT_PATH>
export PYTHONPATH=$PWD
source ./venv/bin/activate

echo USER: $USER
which python

accelerate launch --num_processes 4 taba/scripts/sampling/run_if_sampling.py --seed 420 --noises_per_prompt 4 --n_prompts 1280 --batch_size 32 --num_inference_steps 50 --guidance_scale 1.0 --prompts dataset --cond_seed 42 --internal --input_path experiments/deepfloyd_if/invert/swap_eps_before10/20250227_001401_seed_420_batch_size_32_num_inference_steps_50_guidance_scale_1.0_internal_Truewith_reconstruction_True/latents.pts
