#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --account=<SLURM_ACCOUNT_NAME>
#SBATCH --partition=<SLURM_PARTITION_NAME>
#SBATCH --output=slurm_out/dit/invert_swap/invert_swap_eps.out
#SBATCH --job-name=dit_invert_swap_eps

module load ML-bundle/24.06a
cd <PROJECT_PATH>
export PYTHONPATH=$PWD
source ./venv/bin/activate

echo USER: $USER
which python

accelerate launch --num_processes 4 taba/scripts/invert_swap/run_dit_invert_swap.py --seed 420 --batch_size 32 --num_inference_steps 50 --internal --with_reconstruction --input_image_path experiments/dit/internal/sampling_invert_reconstruction/noise_seed_221/20250228_100735_seed_221_noises_per_prompt_8_n_prompts_1280_batch_size_32_num_inference_steps_50_guidance_scale_1.0_cond_seed_10_with_inversion_True_with_reconstruction_True/samples.pt --input_cond_path experiments/dit/internal/sampling_invert_reconstruction/noise_seed_221/20250228_100735_seed_221_noises_per_prompt_8_n_prompts_1280_batch_size_32_num_inference_steps_50_guidance_scale_1.0_cond_seed_10_with_inversion_True_with_reconstruction_True/conds.pt --swap_path experiments/dit/internal/sampling_invert_reconstruction/noise_seed_221/20250228_100735_seed_221_noises_per_prompt_8_n_prompts_1280_batch_size_32_num_inference_steps_50_guidance_scale_1.0_cond_seed_10_with_inversion_True_with_reconstruction_True/all_t_eps_samples.pt --swap_before_t 1 --swap_type eps

# --input_path <INP_PATH>