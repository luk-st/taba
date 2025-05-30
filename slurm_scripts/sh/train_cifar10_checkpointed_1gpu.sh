#!/bin/bash

# seeds: 42, 0, 10

export PYTHONPATH="<PROJECT PATH>"
export OPENAI_LOGDIR="<PROJECT PATH>/res/ckpt_models/cifar10_32/seed_10"

DATA_DIR="<PROJECT PATH>/datasets/cifar_train"

MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --rescale_learned_sigmas True --rescale_timesteps True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --save_interval 2500 --log_wandb True --log_interval 1000 --seed 10 --additional_save_steps True"

mpiexec -n 1 python3 scripts/image_train.py --data_dir $DATA_DIR $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS 
# --resume_checkpoint res/ckpt_models/cifar10_32/seed_10/model427500.pt in case of resume
