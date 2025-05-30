#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --account=<SLURM_ACCOUNT_NAME>
#SBATCH --partition=<SLURM_PARTITION_NAME>
#SBATCH --output=slurm_out/dit/4k_t1000.out
#SBATCH --job-name=dit_4k_T1000

module load ML-bundle/24.06a
cd <PROJECT_PATH>
export PYTHONPATH=$PWD
source ./venv/bin/activate

echo USER: $USER
which python

accelerate launch --num_processes 4 taba/scripts/invert_forward/run_dit_invert_forward.py --seed 420 --batch_size 32 --num_inference_steps 1000 --input_image_path experiments/corr_T/dit/T100/samples.pt --input_cond_path experiments/corr_T/dit/T100/conds.pt --save_dir experiments/corr_T/dit/T1000


# accelerate launch --num_processes 4 taba/scripts/sampling/run_dit_sampling.py --seed 420 --noises_per_prompt 4 --n_prompts 1024 --batch_size 32 --num_inference_steps 100 --guidance_scale 1.0 --cond_seed 42 --with_inversion --save_dir experiments/corr_T/dit/T100

# --input_path <INP_PATH>