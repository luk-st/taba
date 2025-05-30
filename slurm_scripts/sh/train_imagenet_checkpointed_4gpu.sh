#!/bin/bash

export PYTHONPATH="<PROJECT PATH>"
export OPENAI_LOGDIR="<PROJECT PATH>/res/ckpt_models/imagenet_64/seed_10"

DATA_DIR="<PROJECT PATH>/datasets/imagenet64/imagenet64/train"

MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --rescale_learned_sigmas True --rescale_timesteps True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 32 --save_interval 2500 --log_wandb True --log_interval 1000 --seed 10 --additional_save_steps True"

mpiexec -n 4 python3 scripts/image_train.py --data_dir $DATA_DIR $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS 

# --resume_checkpoint res/ckpt_models/imagenet_64/seed_10/model1130613.pt in case of resume
