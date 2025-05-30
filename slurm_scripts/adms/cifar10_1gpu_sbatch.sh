#!/bin/bash
#SBATCH -A <GRANT NAME>
#SBATCH -p <PARTITION>
#SBATCH -t 48:00:00
#SBATCH --ntasks 1
#SBATCH --gres gpu:1
#SBATCH --mem 120G
#SBATCH --cpus-per-task=16
#SBATCH --nodes 1
#SBATCH --mail-type=BEGIN
#SBATCH -o slurm_out/diff_cifar.log
#SBATCH --job-name=cifar_1gpu

module load GCC/11.2.0
module load OpenMPI/4.1.2-CUDA-11.6.0

conda activate <ENV PATH>

cd <PROJECT PATH>
sh ./slurm_scripts/sh/train_cifar10_checkpointed_1gpu.sh
