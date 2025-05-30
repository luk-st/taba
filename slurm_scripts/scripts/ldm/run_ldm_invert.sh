#!/bin/bash -l
#SBATCH -A <SLURM_ACCOUNT_NAME>
#SBATCH --partition=<SLURM_PARTITION_NAME>
#SBATCH --time=12:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=14
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --output=slurm_out/ldm/invert/seed0_swap50_eps.out
#SBATCH --job-name=ldm_invert_seed0_swap50_eps

cd <PROJECT_PATH>
module use /appl/local/csc/modulefiles/
module load pytorch >/dev/null 2>&1
source ./venv/bin/activate
export PYTHONPATH=$PWD

echo USER: $USER
echo PWD: $(pwd)
echo Python: $(which python)
export PYTHONUNBUFFERED=1

accelerate launch --num_processes 4 taba/scripts/invert_swap/run_ldm_invert_swap.py --seed 0 --batch_size 32 --num_inference_steps 50 --internal --with_reconstruction --input_image_path experiments/celeba_ldm_256/internal/sampling_invert_reconstruction/20250226_182210_seed_0_T_50_batch_size_32_n_samples_5120__with_inversion_True__with_reconstruction_True/samples.pt --swap_path experiments/celeba_ldm_256/internal/sampling_invert_reconstruction/20250226_182210_seed_0_T_50_batch_size_32_n_samples_5120__with_inversion_True__with_reconstruction_True/all_t_eps_samples.pt --swap_before_t 50 --swap_type eps

# --input_path <INP_PATH>