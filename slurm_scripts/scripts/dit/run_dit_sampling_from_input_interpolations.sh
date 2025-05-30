#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --account=<SLURM_ACCOUNT_NAME>
#SBATCH --partition=<SLURM_PARTITION_NAME>
#SBATCH --output=slurm_out/dit/from_forward/interpolations/T50_10k_forward50.out
#SBATCH --job-name=dit-forward-interpolations-forward50

module load ML-bundle/24.06a
cd <PROJECT_PATH>
export PYTHONPATH=$PWD
source ./venv/bin/activate

echo USER: $USER
which python

accelerate launch --num_processes 4 taba/scripts/sampling/run_dit_sampling.py --seed 221 --noises_per_prompt 8 --n_prompts 1280 --batch_size 32 --num_inference_steps 50 --guidance_scale 1.0 --cond_seed 10 --input_cond_path experiments/dit/start_interpolations_forward/conds.pt --input_path experiments/dit/start_interpolations_forward/forward50_05.pt
# --input_path <INP_PATH>