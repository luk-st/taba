#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --account=<SLURM_ACCOUNT_NAME>
#SBATCH --partition=<SLURM_PARTITION_NAME>
#SBATCH --output=slurm_out/dit/from_forward/edit/T50_10k_forward50.out
#SBATCH --job-name=dit-forward-edit-forward50

module load ML-bundle/24.06a
cd <PROJECT_PATH>
export PYTHONPATH=$PWD
source ./venv/bin/activate

echo USER: $USER
which python

accelerate launch --num_processes 4 taba/scripts/sampling/run_dit_sampling.py --seed 420 --noises_per_prompt 8 --n_prompts 1280 --batch_size 32 --num_inference_steps 50 --guidance_scale 1.0 --cond_seed 42 --input_cond_path experiments/dit/start_interpolations_forward/conds_edit.pt --input_path experiments/dit/invert/forward_seed420/forward_before50/20250506_193852_seed_420_forward_seed_42_T_50_batch_size_32_guidance_scale_1.0_with_reconstruction_True_internal_False/latents.pt
# --input_path <INP_PATH>

# noise: --input_path experiments/dit/internal/sampling_invert_reconstruction/noise_seed_420/20250225_141652_seed_420_noises_per_prompt_8_n_prompts_1280_batch_size_32_num_inference_steps_50_guidance_scale_1.0_cond_seed_10_with_inversion_True_with_reconstruction_True/noise.pt
# forward0: --input_path experiments/dit/invert/forward_seed420/forward_before0/20250506_191035_seed_420_forward_seed_42_T_50_batch_size_32_guidance_scale_1.0_with_reconstruction_True_internal_False/latents.pt
# forward1: --input_path experiments/dit/invert/forward_seed420/forward_before1/20250506_191217_seed_420_forward_seed_42_T_50_batch_size_32_guidance_scale_1.0_with_reconstruction_True_internal_False/latents.pt
# forward2: --input_path experiments/dit/invert/forward_seed420/forward_before2/20250506_191217_seed_420_forward_seed_42_T_50_batch_size_32_guidance_scale_1.0_with_reconstruction_True_internal_False/latents.pt
# forward3: --input_path experiments/dit/invert/forward_seed420/forward_before3/20250506_191220_seed_420_forward_seed_42_T_50_batch_size_32_guidance_scale_1.0_with_reconstruction_True_internal_False/latents.pt
# forward5: --input_path experiments/dit/invert/forward_seed420/forward_before5/20250506_191242_seed_420_forward_seed_42_T_50_batch_size_32_guidance_scale_1.0_with_reconstruction_True_internal_False/latents.pt
# forward10: --input_path experiments/dit/invert/forward_seed420/forward_before10/20250506_191252_seed_420_forward_seed_42_T_50_batch_size_32_guidance_scale_1.0_with_reconstruction_True_internal_False/latents.pt
# forward50: --input_path experiments/dit/invert/forward_seed420/forward_before50/20250506_193852_seed_420_forward_seed_42_T_50_batch_size_32_guidance_scale_1.0_with_reconstruction_True_internal_False/latents.pt