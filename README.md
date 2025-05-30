# There and Back Again: On the relation between Noise and Image Inversions in Diffusion Models



This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion) and [diffusers 🧨 DDIM Scheduler implementation](https://huggingface.co/docs/diffusers/api/schedulers/ddim#ddimscheduler).

## Installation
```sh
pip install -r requirements.txt
```

## Download pre-trained models

To use `ADM` models, download the following checkpoints and place them in the `res/openai-models/` directory:
* ADM-32: [cifar10_uncond_50M_500K.pt](https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_50M_500K.pt)
* ADM-64: [imagenet64_uncond_100M_1500K.pt](https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_uncond_100M_1500K.pt)
* ADM-256: [256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)

`LDM`, `DiT`, and `Deepfloyd-IF` models are downloaded **automatically**

## Experiment runs

### Sampling, inversion, and reconstruction

ADM models:
```sh
$ accelerate launch --num_processes 1 taba/scripts/sampling/run_adm_sampling.py --model_name cifar_pixel_32 --num_inference_steps 100 --with_inversion --with_reconstruction --seed 420 --batch_size 256 --n_samples 10240 --save_dir experiments/sample_invert_reconstruct/adm32

$ accelerate launch --num_processes 1 taba/scripts/sampling/run_adm_sampling.py --model_name imagenet_pixel_64 --num_inference_steps 100 --with_inversion --with_reconstruction --seed 420 --batch_size 128 --n_samples 10240 --save_dir experiments/sample_invert_reconstruct/adm64

$ accelerate launch --num_processes 1 taba/scripts/sampling/run_adm_sampling.py --model_name imagenet_pixel_256 --num_inference_steps 100 --with_inversion --with_reconstruction --seed 420 --batch_size 64 --n_samples 10240 --save_dir experiments/sample_invert_reconstruct/adm256

# use --internal to collect intermediate steps
# use --n_parts and --part_idx to split the dataset to multiple parts
# use --num_processes N to sample data split with N GPUs
# use --input_noise_path PATH to sample from tensor of sampled Noises instead of torch.randn()
# use --input_image_path PATH to start with inversion from provided images
```

LDM model:
```sh
$ accelerate launch --num_processes 1 taba/scripts/sampling/run_ldm_sampling.py --num_inference_steps 100 --with_inversion --seed 420 --batch_size 128 --n_samples 10240 --with_reconstruction --save_dir experiments/sample_invert_reconstruct/ldm

# use --internal to collect intermediate steps
# use --n_parts and --part_idx to split the dataset to multiple parts
# use --num_processes N to sample data split with N GPUs
# use --input_noise_path PATH to sample from tensor of sampled Noises instead of torch.randn()
```

DiT model:
```sh
$ accelerate launch --num_processes 1 taba/scripts/sampling/run_dit_sampling.py --seed 420 --noises_per_prompt 8 --n_prompts 1280 --batch_size 128 --num_inference_steps 100 --guidance_scale 1.0 --cond_seed 10 --with_inversion --with_reconstruction --save_dir experiments/sample_invert_reconstruct/dit

# use --internal to collect intermediate steps
# use --num_processes N to sample data split with N GPUs
# use --input_path PATH and --input_cond_path PATH2 to sample from a ready tensor of sampled Noises with given conditioning
```

Deepfloyd-IF model:
```sh
$ accelerate launch --num_processes 1 taba/scripts/sampling/run_if_sampling.py --seed 420 --noises_per_prompt 8 --n_prompts 1024 --batch_size 64 --num_inference_steps 100 --guidance_scale 1.0 --prompts dataset --cond_seed 10 --with_inversion --with_reconstruction --save_dir experiments/sample_invert_reconstruct/if

# use --internal to collect intermediate steps
# use --prompts null to sample with null prompt embedding
# use --num_processes N to sample data split with N GPUs
# use --input_path PATH to sample from a ready tensor of sampled Noises
```

### Replacing first inversion predictions with ground truth one
Example: `5` = number of first DM predictions during inversion to replace

LDM model:
```sh
$ accelerate launch --num_processes 1 taba/scripts/invert_swap/run_ldm_invert_swap.py --seed 420 --batch_size 128 --num_inference_steps 100 --with_reconstruction --input_image_path experiments/sample_invert_reconstruct/ldm/samples.pt --swap_path experiments/sample_invert_reconstruct/ldm/all_t_eps_samples.pt --swap_before_t 5 --swap_type eps --save_dir experiments/invert_swap/ldm5

# use --internal to collect intermediate steps
# use --n_parts and --part_idx to split the dataset to multiple parts
# use --num_processes N to sample data split with N GPUs
# use --input_image_path PATH is a tensor of generated images
# use --swap_type eps to replace model predictions, use --swap_type xt to replace whole step (avoid machine precision issues)
```

DiT model:
```sh
$ accelerate launch --num_processes 1 taba/scripts/invert_swap/run_dit_invert_swap.py --seed 420 --batch_size 128 --num_inference_steps 100 --with_reconstruction --input_image_path experiments/sample_invert_reconstruct/dit/samples.pt --input_cond_path experiments/sample_invert_reconstruct/dit/conds.pt --swap_path experiments/sample_invert_reconstruct/dit/all_t_eps_samples.pt --swap_before_t 5 --swap_type eps --save_dir experiments/invert_swap/dit5

# use --internal to collect intermediate steps
# use --num_processes N to sample data split with N GPUs
# use --input_image_path PATH and --input_cond_path PATH2 to make sure that inversion is done with the same conditions
# use --swap_type eps to replace model predictions, use --swap_type xt to replace whole step (avoid machine precision issues)
```

Deepfloyd-IF model:
```sh
$ accelerate launch --num_processes 1 taba/scripts/invert_swap/run_if_invert_swap.py --seed 420 --batch_size 64 --num_inference_steps 100 --guidance_scale 1.0 --internal --with_reconstruction --input_samples_path experiments/sample_invert_reconstruct/if/samples.pt --input_prompts_path experiments/sample_invert_reconstruct/if/prompts.pkl --swap_path experiments/sample_invert_reconstruct/if/all_t_eps_samples.pt --swap_type eps --swap_before_t 5 --save_dir experiments/invert_swap/if5

# use --internal to collect intermediate steps
# use --num_processes N to sample data split with N GPUs
# use --input_samples_path PATH and --input_prompts_path PATH2 to make sure that inversion is done with the same conditions
# use --swap_type eps to replace model predictions, use --swap_type xt to replace whole step (avoid machine precision issues)
```

### Invert with forward diffusion
Example: `3` = number of first inversion steps to replace with forward diffusion

DiT model:
```sh
$ accelerate launch --num_processes 1 taba/scripts/invert_forward/run_dit_invert_forward.py --seed 420 --batch_size 128 --num_inference_steps 100 --guidance_scale 1.0 --with_reconstruction --input_image_path experiments/sample_invert_reconstruct/dit/samples.pt --input_cond_path experiments/sample_invert_reconstruct/dit/conds.pt --forward_before_t 3 --forward_seed 999 --save_dir experiments/invert_forward/dit3

# use --num_processes N to sample data split with N GPUs
# use --input_image_path PATH and --input_cond_path PATH2 to make sure that inversion is done with the same conditions
# make sure --forward_seed SEED is different from --seed
```

Deepfloyd-IF model:
```sh
$ accelerate launch --num_processes 1 taba/scripts/invert_forward/run_if_invert_forward.py --seed 420 --batch_size 64 --num_inference_steps 100 --guidance_scale 1.0 --with_reconstruction --input_samples_path experiments/sample_invert_reconstruct/if/samples.pt --input_prompts_path experiments/sample_invert_reconstruct/if/prompts.pkl --forward_before_t 3 --forward_seed 999 --save_dir experiments/invert_forward/if3

# use --num_processes N to sample data split with N GPUs
# use --input_image_path PATH and --input_cond_path PATH2 to make sure that inversion is done with the same conditions
# make sure --forward_seed SEED is different from --seed
```
