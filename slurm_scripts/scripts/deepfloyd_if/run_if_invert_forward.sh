#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --account=<SLURM_ACCOUNT_NAME>
#SBATCH --partition=<SLURM_PARTITION_NAME>
#SBATCH --output=slurm_out/if/invert_forward/s420_conds10_forws42_t50
#SBATCH --job-name=if_invert_forward_s420_conds10_t50

module load ML-bundle/24.06a
cd <PROJECT_PATH>
export PYTHONPATH=$PWD
source ./venv/bin/activate

echo USER: $USER
which python

accelerate launch --num_processes 4 taba/scripts/invert_forward/run_if_invert_forward.py --seed 420 --batch_size 32 --num_inference_steps 50 --guidance_scale 1.0 --with_reconstruction --input_samples_path experiments/deepfloyd_if/internal/sampling_invert_reconstruction/dataset_20250226_152228_seed_420_noises_per_prompt_4_n_prompts_1280_batch_size_32_num_inference_steps_50_guidance_scale_1.0cond_seed_10_with_inversion_True_with_reconstruction_True/samples.pt --input_prompts_path experiments/deepfloyd_if/internal/sampling_invert_reconstruction/dataset_20250226_152228_seed_420_noises_per_prompt_4_n_prompts_1280_batch_size_32_num_inference_steps_50_guidance_scale_1.0cond_seed_10_with_inversion_True_with_reconstruction_True/prompts.pkl --forward_before_t 50 --forward_seed 42
