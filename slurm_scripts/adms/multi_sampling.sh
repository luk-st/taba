#!/bin/bash

seeds=(0 10 42)
# model_name="imagenet64"
# starts=(0 250000 800000 1100000)
# stops=(250000 800000 1100000 1500000)

model_name="cifar32"
starts=(0 27300 175890 331500 495000 595000)
stops=(27300 175890 331500 495000 595000 700000)

for seed in "${seeds[@]}"; do
    for i in "${!starts[@]}"; do
        start=${starts[$i]}
        stop=${stops[$i]}
        log_file="slurm_out/20240917_${model_name}_s${seed}_st${start}.log"
        job_name="20240917${model_name}_s${seed}_st${start}"

        sbatch --output=$log_file --job-name=$job_name <<EOT
#!/bin/bash
#SBATCH -A <GRANT NAME>
#SBATCH -p <PARTITION>
#SBATCH -t 24:00:00
#SBATCH --ntasks 1
#SBATCH --gres gpu:1
#SBATCH --mem 120G
#SBATCH --cpus-per-task=16
#SBATCH --nodes 1

module load GCC/11.2.0
module load Miniconda3/23.3.1-0

eval "\$(conda shell.bash hook)"

conda activate d_a

cd <PROJECT PATH>

export PYTHONPATH=\$PWD

python3 taba/exp_pipelines/steps_statistics_last_sample.py -m $model_name -s $seed -start $start -stop $stop
EOT

    done
done