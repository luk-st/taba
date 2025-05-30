#!/bin/bash

seeds=(0 10 42)
model_name="imagenet64"
starts=(0 144298 412280 666798 934780 1192455 1460437)
stops=(144298 412280 666798 934780 1192455 1460437 1481051)

# model_name="cifar32"
# starts=(0 29250 175890 370890 495000 595000)
# stops=(29250 175890 370890 495000 595000 700000)

for seed in "${seeds[@]}"; do
    for i in "${!starts[@]}"; do
        start=${starts[$i]}
        stop=${stops[$i]}
        log_file="slurm_out/latent_similarity/20241227/${model_name}_s${seed}_st${start}.log"
        job_name="${model_name}_s${seed}_st${start}"

        sbatch --output=$log_file --job-name=$job_name <<EOT
#!/bin/bash -l
#SBATCH -A <GRANT NAME>
#SBATCH -p <PARTITION>
#SBATCH -t 4:00:00
#SBATCH --ntasks 1
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --nodes 1

cd <PROJECT PATH>

module load ML-bundle/24.06a
source ./venv/bin/activate
export PYTHONPATH=\$PWD

echo USER: $USER
which python

python3 taba/exp_pipelines/noising_similarity/latents_sim_metrics.py -m $model_name -s $seed -start $start -stop $stop
EOT

    done
done