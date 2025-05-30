#!/bin/bash
#SBATCH -A <GRANT NAME>
#SBATCH -p <PARTITION>
#SBATCH -t 48:00:00
#SBATCH --ntasks 4
#SBATCH --gres gpu:4
#SBATCH --mem 480G
#SBATCH --cpus-per-task=16
#SBATCH --nodes 1
#SBATCH --mail-type=BEGIN
#SBATCH -o slurm_out/diff_imagenet.log
#SBATCH --job-name=inet_4gpu

module load GCC/11.2.0
module load OpenMPI/4.1.2-CUDA-11.6.0

conda activate <ENV PATH>

cd <PROJECT PATH>
sh ./slurm_scripts/sh/train_imagenet_checkpointed_4gpu.sh
