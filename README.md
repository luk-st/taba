<h1 align="center">
     There and Back Again: On the relation between Noise and Image Inversions in Diffusion Models
</h1>
<h4 align="center">
Łukasz Staniszewski, Łukasz Kuciński, Kamil Deja
</h4>
<p align="center">
  <a href="https://arxiv.org/abs/2410.23530"><img src="https://img.shields.io/badge/arXiv-2410.23530-b31b1b.svg"></a>
</p>

<div style="max-width: 10px; margin: 0 auto; text-align: center; font-size: 14px; border: 1px solid red;">
    <img src="https://github.com/user-attachments/assets/c95445ee-66a5-4030-bbfc-50563f3897bd" style="width: 100%; height: auto;" alt="Image"><br>
    This is the implementation for the paper <em>"There and Back Again: On the relation between Noise and Image Inversions in Diffusion Models"</em>. We show that DDIM inverted latents exhibit input image patterns and propose to replace first inversion steps with forward diffusion process, boosting image editability and interpolation with Diffusion Models. Please refer to <a href="https://www.arxiv.org/abs/2410.23530">our paper</a> for more details.
</div>



## ⚙️ Setup

### Installation
```sh
pip install -r requirements.txt
```

### Download pre-trained models

To use `ADM` models, download the following checkpoints and place them in the `res/openai-models/` directory:
* ADM-32: [cifar10_uncond_50M_500K.pt](https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_50M_500K.pt)
* ADM-64: [imagenet64_uncond_100M_1500K.pt](https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_uncond_100M_1500K.pt)
* ADM-256: [256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)

`LDM`, `DiT`, `Deepfloyd-IF`, and `SDXL` models are downloaded **automatically**

## 🧪 Run experiments

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

SDXL model:
```sh
$ accelerate launch --num_processes 1 taba/scripts/sdxl/sdxl_ddim_sample_inv_recon.py --seed 420 --n_noises_per_prompt 4 --n_prompts 512 --batch_size 4 --num_inference_steps 100 --guidance_scale 1.0 --cond_seed 10 --save_dir experiments/sample_invert_reconstruct/sdxl

# use --with_forward to use our inversion method:
#   use --forward_before_t to specify the number of first inversion steps to replace
#   use --forward_seed to specify the seed for the forward diffusion
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


ADM model:
```sh
$ accelerate launch --num_processes 1 taba/scripts/invert_forward/run_adm_invert_forward.py --model_name cifar_pixel_32 --seed 420 --batch_size 256 --num_inference_steps 100 --guidance_scale 1.0 --with_reconstruction --input_image_path experiments/sample_invert_reconstruct/adm32/samples.pt --forward_before_t 3 --forward_seed 999 --save_dir experiments/invert_forward/adm32_3
$ accelerate launch --num_processes 1 taba/scripts/invert_forward/run_adm_invert_forward.py --model_name imagenet_pixel_64 --seed 420 --batch_size 128 --num_inference_steps 100 --guidance_scale 1.0 --with_reconstruction --input_image_path experiments/sample_invert_reconstruct/adm64/samples.pt --forward_before_t 3 --forward_seed 999 --save_dir experiments/invert_forward/adm64_3
$ accelerate launch --num_processes 1 taba/scripts/invert_forward/run_adm_invert_forward.py --model_name imagenet_pixel_256 --seed 420 --batch_size 64 --num_inference_steps 100 --guidance_scale 1.0 --with_reconstruction --input_image_path experiments/sample_invert_reconstruct/adm256/samples.pt --forward_before_t 3 --forward_seed 999 --save_dir experiments/invert_forward/adm256_3

# use --num_processes N to sample data split with N GPUs
# use --input_image_path PATH and --input_cond_path PATH2 to make sure that inversion is done with the same conditions
# make sure --forward_seed SEED is different from --seed
# if needed, you can divide the dataset to multiple parts with --n_parts N and --part_idx {0, ..., N-1}
```

DiT model:
```sh
$ accelerate launch --num_processes 1 taba/scripts/invert_forward/run_dit_invert_forward.py --seed 420 --batch_size 128 --num_inference_steps 100 --guidance_scale 1.0 --with_reconstruction --input_image_path experiments/sample_invert_reconstruct/dit/samples.pt --input_cond_path experiments/sample_invert_reconstruct/dit/conds.pt --forward_before_t 3 --forward_seed 999 --save_dir experiments/invert_forward/dit3

# use --num_processes N to sample data split with N GPUs
# use --input_image_path PATH and --input_cond_path PATH2 to make sure that inversion is done with the same conditions
# make sure --forward_seed SEED is different from --seed
```

LDM model:
```sh
$ accelerate launch --num_processes 1 taba/scripts/invert_forward/run_adm_invert_forward.py --seed 420 --batch_size 128 --num_inference_steps 100 --guidance_scale 1.0 --with_reconstruction --input_image_path experiments/sample_invert_reconstruct/ldm/samples.pt --forward_before_t 3 --forward_seed 999 --save_dir experiments/invert_forward/ldm_3

# use --internal to collect intermediate steps
# use --num_processes N to sample data split with N GPUs
# use --input_image_path PATH and --input_cond_path PATH2 to make sure that inversion is done with the same conditions
# make sure --forward_seed SEED is different from --seed
# if needed, you can divide the dataset to multiple parts with --n_parts N and --part_idx {0, ..., N-1}
```

Deepfloyd-IF model:
```sh
$ accelerate launch --num_processes 1 taba/scripts/invert_forward/run_if_invert_forward.py --seed 420 --batch_size 64 --num_inference_steps 100 --guidance_scale 1.0 --with_reconstruction --input_samples_path experiments/sample_invert_reconstruct/if/samples.pt --input_prompts_path experiments/sample_invert_reconstruct/if/prompts.pkl --forward_before_t 3 --forward_seed 999 --save_dir experiments/invert_forward/if3

# use --num_processes N to sample data split with N GPUs
# use --input_image_path PATH and --input_cond_path PATH2 to make sure that inversion is done with the same conditions
# make sure --forward_seed SEED is different from --seed
```

## 💗 Acknowledgements
This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion) and [diffusers 🧨 DDIM Scheduler implementation](https://huggingface.co/docs/diffusers/api/schedulers/ddim#ddimscheduler).

## :black_nib: Citation

If you found our work helpful, please consider citing:

```bibtex
@misc{staniszewski2025taba,
      title={There and Back Again: On the relation between Noise and Image Inversions in Diffusion Models}, 
      author={Łukasz Staniszewski and Łukasz Kuciński and Kamil Deja},
      year={2025},
      eprint={2410.23530},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.23530}, 
}
```
