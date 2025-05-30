#!/bin/bash

seeds=(0 10 42)

date="20241223"
is_last_denoiser=true
model_name="imagenet64"
starts=(0 250000 600000 900000 1250000)
stops=(250000 600000 900000 1250000 1500000)

# model_name="cifar32"
# starts=(0 27300 175890 331500 495000 595000)
# stops=(27300 175890 331500 495000 595000 700000)

if [ "$is_last_denoiser" = true ]; then
    denoiser_argument="--is_last_denoiser"
else
    denoiser_argument=""
fi

for seed in "${seeds[@]}"; do
    for i in "${!starts[@]}"; do
        start=${starts[$i]}
        stop=${stops[$i]}
        log_file="slurm_out/last_denoiser_${is_last_denoiser}/${model_name}/${date}/${model_name}_s${seed}_st${start}.log"
        job_name="${model_name}_s${seed}_st${start}"

        sbatch --output=$log_file --job-name=$job_name <<EOT
#!/bin/bash -l
#SBATCH -A <SLURM_ACCOUNT_NAME>
#SBATCH -p <SLURM_PARTITION_NAME>
#SBATCH -t 24:00:00
#SBATCH --ntasks 1
#SBATCH --gres gpu:1
#SBATCH --mem 400G
#SBATCH --cpus-per-task=70
#SBATCH --nodes 1

cd <PROJECT_PATH>

module load ML-bundle/24.06a
source ./venv/bin/activate
export PYTHONPATH=\$PWD

echo USER: $USER
which python

python3 taba/exp_pipelines/noising_similarity/lastsample_noising_denoising.py -m $model_name -s $seed -start $start -stop $stop $denoiser_argument
EOT

    done
done