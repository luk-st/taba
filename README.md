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

This project uses [`uv`](https://docs.astral.sh/uv/) for environment and dependency management. Install `uv` (see the [docs](https://docs.astral.sh/uv/getting-started/installation/)), then from the repository root run:

```sh
uv sync
```

This creates a `.venv/` with the exact, locked dependencies from `uv.lock` (Python 3.10, CUDA 12.1 PyTorch wheels) and installs the `taba` package in editable mode. Prefix commands with `uv run` (e.g. `uv run accelerate launch ...`) or activate the environment with `source .venv/bin/activate`.

To also install the optional notebook dependencies:

```sh
uv sync --extra notebooks
```

### Download pre-trained models

To use `ADM` models, download the following checkpoints and place them in the `res/openai-models/` directory:
* ADM-32: [cifar10_uncond_50M_500K.pt](https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_50M_500K.pt)
* ADM-64: [imagenet64_uncond_100M_1500K.pt](https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_uncond_100M_1500K.pt)
* ADM-256: [256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)

`LDM`, `DiT`, `Deepfloyd-IF`, and `SDXL` models are downloaded **automatically**.

## 🛠️ Configuration (Hydra)

Every experiment script is configured with [Hydra](https://hydra.cc/). Default configs live in [`configs/`](configs/), mirroring the script layout (e.g. `configs/sampling/dit.yaml` configures `taba/scripts/sampling/run_dit_sampling.py`). Any field can be overridden on the command line with `key=value` syntax, for example:

```sh
uv run accelerate launch taba/scripts/sampling/run_dit_sampling.py n_prompts=1280 with_inversion=true with_reconstruction=true
```

Boolean flags are set with `key=true` / `key=false`, paths default to `null`, and lists use `'key=[a,b]'`. To see the fully resolved config for a script without running it, append `--cfg job`.

## 🧪 Run experiments

### Sampling, inversion, and reconstruction

ADM models:
```sh
$ uv run accelerate launch --num_processes 1 taba/scripts/sampling/run_adm_sampling.py model_name=cifar_pixel_32 num_inference_steps=100 with_inversion=true with_reconstruction=true seed=420 batch_size=256 n_samples=10240 save_dir=experiments/sample_invert_reconstruct/adm32

$ uv run accelerate launch --num_processes 1 taba/scripts/sampling/run_adm_sampling.py model_name=imagenet_pixel_64 num_inference_steps=100 with_inversion=true with_reconstruction=true seed=420 batch_size=128 n_samples=10240 save_dir=experiments/sample_invert_reconstruct/adm64

$ uv run accelerate launch --num_processes 1 taba/scripts/sampling/run_adm_sampling.py model_name=imagenet_pixel_256 num_inference_steps=100 with_inversion=true with_reconstruction=true seed=420 batch_size=64 n_samples=10240 save_dir=experiments/sample_invert_reconstruct/adm256

# use internal=true to collect intermediate steps
# use n_parts and part_idx to split the dataset to multiple parts
# use --num_processes N (an accelerate flag) to sample data split with N GPUs
# use input_noise_path=PATH to sample from a tensor of sampled noises instead of torch.randn()
# use input_image_path=PATH to start with inversion from provided images
```

LDM model:
```sh
$ uv run accelerate launch --num_processes 1 taba/scripts/sampling/run_ldm_sampling.py num_inference_steps=100 with_inversion=true with_reconstruction=true seed=420 batch_size=128 n_samples=10240 save_dir=experiments/sample_invert_reconstruct/ldm

# use internal=true to collect intermediate steps
# use n_parts and part_idx to split the dataset to multiple parts
# use input_noise_path=PATH to sample from a tensor of sampled noises instead of torch.randn()
```

DiT model:
```sh
$ uv run accelerate launch --num_processes 1 taba/scripts/sampling/run_dit_sampling.py seed=420 noises_per_prompt=8 n_prompts=1280 batch_size=128 num_inference_steps=100 guidance_scale=1.0 cond_seed=10 with_inversion=true with_reconstruction=true save_dir=experiments/sample_invert_reconstruct/dit

# use internal=true to collect intermediate steps
# use --num_processes N (an accelerate flag) to sample data split with N GPUs
# use input_path=PATH and input_cond_path=PATH2 to sample from a ready tensor of sampled noises with given conditioning
```

Deepfloyd-IF model:
```sh
$ uv run accelerate launch --num_processes 1 taba/scripts/sampling/run_if_sampling.py seed=420 noises_per_prompt=8 n_prompts=1024 batch_size=64 num_inference_steps=100 guidance_scale=1.0 prompts_dataset=dataset cond_seed=10 with_inversion=true with_reconstruction=true save_dir=experiments/sample_invert_reconstruct/if

# use internal=true to collect intermediate steps
# use prompts_dataset=null to sample with null prompt embedding
# use --num_processes N (an accelerate flag) to sample data split with N GPUs
# use input_path=PATH to sample from a ready tensor of sampled noises
```

SDXL model:
```sh
$ uv run accelerate launch --num_processes 1 taba/scripts/sdxl/sdxl_ddim_sample_inv_recon.py seed=88 n_noises_per_prompt=4 n_prompts=512 batch_size=4 num_inference_steps=50 guidance_scale=1.0 cond_seed=11

# results are written under experiments/sdxl/ddim
# use with_forward=true to use our inversion method:
#   use forward_before_t=K to set the number of first inversion steps to replace with forward diffusion
#   use forward_seed=SEED to set the seed for the forward diffusion (keep it different from seed)
```

### Replacing first inversion predictions with the ground-truth one
Example: `swap_before_t=5` = number of first DM predictions during inversion to replace.

LDM model:
```sh
$ uv run accelerate launch --num_processes 1 taba/scripts/invert_swap/run_ldm_invert_swap.py seed=420 batch_size=128 num_inference_steps=100 with_reconstruction=true input_image_path=experiments/sample_invert_reconstruct/ldm/samples.pt swap_path=experiments/sample_invert_reconstruct/ldm/all_t_eps_samples.pt swap_before_t=5 swap_type=eps save_dir=experiments/invert_swap/ldm5

# use internal=true to collect intermediate steps
# use n_parts and part_idx to split the dataset to multiple parts
# use --num_processes N (an accelerate flag) to sample data split with N GPUs
# input_image_path is a tensor of generated images
# use swap_type=eps to replace model predictions, use swap_type=xt to replace the whole step (avoid machine precision issues)
```

DiT model:
```sh
$ uv run accelerate launch --num_processes 1 taba/scripts/invert_swap/run_dit_invert_swap.py seed=420 batch_size=128 num_inference_steps=100 with_reconstruction=true input_image_path=experiments/sample_invert_reconstruct/dit/samples.pt input_cond_path=experiments/sample_invert_reconstruct/dit/conds.pt swap_path=experiments/sample_invert_reconstruct/dit/all_t_eps_samples.pt swap_before_t=5 swap_type=eps save_dir=experiments/invert_swap/dit5

# use internal=true to collect intermediate steps
# use --num_processes N (an accelerate flag) to sample data split with N GPUs
# use input_image_path=PATH and input_cond_path=PATH2 to make sure that inversion is done with the same conditions
# use swap_type=eps to replace model predictions, use swap_type=xt to replace the whole step (avoid machine precision issues)
```

Deepfloyd-IF model:
```sh
$ uv run accelerate launch --num_processes 1 taba/scripts/invert_swap/run_if_invert_swap.py seed=420 batch_size=64 num_inference_steps=100 guidance_scale=1.0 internal=true with_reconstruction=true input_samples_path=experiments/sample_invert_reconstruct/if/samples.pt input_prompts_path=experiments/sample_invert_reconstruct/if/prompts.pkl swap_path=experiments/sample_invert_reconstruct/if/all_t_eps_samples.pt swap_type=eps swap_before_t=5 save_dir=experiments/invert_swap/if5

# use internal=true to collect intermediate steps
# use --num_processes N (an accelerate flag) to sample data split with N GPUs
# use input_samples_path=PATH and input_prompts_path=PATH2 to make sure that inversion is done with the same conditions
# use swap_type=eps to replace model predictions, use swap_type=xt to replace the whole step (avoid machine precision issues)
```

### Invert with forward diffusion (our method)
Example: `forward_before_t=3` = number of first inversion steps to replace with forward diffusion.

ADM model:
```sh
$ uv run accelerate launch --num_processes 1 taba/scripts/invert_forward/run_adm_invert_forward.py model_name=cifar_pixel_32 seed=420 batch_size=256 num_inference_steps=100 with_reconstruction=true input_image_path=experiments/sample_invert_reconstruct/adm32/samples.pt forward_before_t=3 forward_seed=999 save_dir=experiments/invert_forward/adm32_3
$ uv run accelerate launch --num_processes 1 taba/scripts/invert_forward/run_adm_invert_forward.py model_name=imagenet_pixel_64 seed=420 batch_size=128 num_inference_steps=100 with_reconstruction=true input_image_path=experiments/sample_invert_reconstruct/adm64/samples.pt forward_before_t=3 forward_seed=999 save_dir=experiments/invert_forward/adm64_3
$ uv run accelerate launch --num_processes 1 taba/scripts/invert_forward/run_adm_invert_forward.py model_name=imagenet_pixel_256 seed=420 batch_size=64 num_inference_steps=100 with_reconstruction=true input_image_path=experiments/sample_invert_reconstruct/adm256/samples.pt forward_before_t=3 forward_seed=999 save_dir=experiments/invert_forward/adm256_3

# use --num_processes N (an accelerate flag) to sample data split with N GPUs
# use input_image_path=PATH to make sure that inversion is done with the same images
# make sure forward_seed is different from seed
# if needed, divide the dataset to multiple parts with n_parts=N and part_idx={0, ..., N-1}
```

DiT model:
```sh
$ uv run accelerate launch --num_processes 1 taba/scripts/invert_forward/run_dit_invert_forward.py seed=420 batch_size=128 num_inference_steps=100 guidance_scale=1.0 with_reconstruction=true input_image_path=experiments/sample_invert_reconstruct/dit/samples.pt input_cond_path=experiments/sample_invert_reconstruct/dit/conds.pt forward_before_t=3 forward_seed=999 save_dir=experiments/invert_forward/dit3

# use --num_processes N (an accelerate flag) to sample data split with N GPUs
# use input_image_path=PATH and input_cond_path=PATH2 to make sure that inversion is done with the same conditions
# make sure forward_seed is different from seed
```

LDM model:
```sh
$ uv run accelerate launch --num_processes 1 taba/scripts/invert_forward/run_ldm_invert_forward.py seed=420 batch_size=128 num_inference_steps=100 with_reconstruction=true input_image_path=experiments/sample_invert_reconstruct/ldm/samples.pt forward_before_t=3 forward_seed=999 save_dir=experiments/invert_forward/ldm_3

# use internal=true to collect intermediate steps
# use --num_processes N (an accelerate flag) to sample data split with N GPUs
# use input_image_path=PATH to make sure that inversion is done with the same images
# make sure forward_seed is different from seed
```

Deepfloyd-IF model:
```sh
$ uv run accelerate launch --num_processes 1 taba/scripts/invert_forward/run_if_invert_forward.py seed=420 batch_size=64 num_inference_steps=100 guidance_scale=1.0 with_reconstruction=true input_samples_path=experiments/sample_invert_reconstruct/if/samples.pt input_prompts_path=experiments/sample_invert_reconstruct/if/prompts.pkl forward_before_t=3 forward_seed=999 save_dir=experiments/invert_forward/if3

# use --num_processes N (an accelerate flag) to sample data split with N GPUs
# use input_samples_path=PATH and input_prompts_path=PATH2 to make sure that inversion is done with the same conditions
# make sure forward_seed is different from seed
```

## 🎨 Editing real images with our inversion

We provide two real-image editing pipelines that plug our forward-diffusion inversion into existing attention-based editing methods. The vendored, lightly-adapted method code lives under [`taba/ext/`](taba/ext/).

### MasaCtrl + forward diffusion (SD1.5)

Edit a single user image by combining [MasaCtrl](https://github.com/TencentARC/MasaCtrl) mutual self-attention control with our inversion:

```sh
$ uv run python taba/scripts/masactrl/run_masactrl_edit_real.py \
    image_path=/path/to/image.png \
    source_prompt="" \
    target_prompt="a photo of a tiger" \
    num_inference_steps=50 guidance_scale=7.5 \
    forward_t=2 forward_seed=2115 \
    masactrl_step=4 masactrl_layer=10 \
    output_dir=experiments/masactrl/edit_real
```

`forward_t` is the number of first inversion steps replaced with forward diffusion (set `forward_t=0` to fall back to standard DDIM inversion). Outputs (`input.png`, `reconstruction.png`, `source.png`, `edited.png`, and a `grid.png`) are written to `output_dir`.

### StyleAligned + forward diffusion (SDXL)

Transfer the style of a user image to one or more target prompts by combining [StyleAligned](https://github.com/google/style-aligned) shared attention with our inversion:

```sh
$ uv run python taba/scripts/style_aligned/transfer_style.py \
    image_path=/path/to/photo.jpg \
    p_source="a photo of a house" \
    'ps_target=["a house in the style of van gogh","a house as a watercolor painting"]' \
    num_inference_steps=50 guidance_scale_inv=2.0 gs_sampling=12.0 \
    forward_t=3 forward_seed=2115 offset_inv=3 use_forward_diffusion=true \
    output_dir=results/style_aligned/demo
```

Set `use_forward_diffusion=false` to fall back to standard DDIM inversion. The reconstruction, styled variants, and the input image are written to `output_dir`.

> Note: `taba/scripts/eval/eval_fids.py` is a standalone analysis script with hardcoded paths (edit it in place) and is not Hydra-configured.

## 💗 Acknowledgements
This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion) and [diffusers 🧨 DDIM Scheduler implementation](https://huggingface.co/docs/diffusers/api/schedulers/ddim#ddimscheduler). The real-image editing pipelines build on [MasaCtrl](https://github.com/TencentARC/MasaCtrl) and [StyleAligned](https://github.com/google/style-aligned).

## :black_nib: Citation

If you found our work helpful, please consider citing:

```bibtex
@inproceedings{
     staniszewski2026there,
     title={There and Back Again: On the relation between Noise and Image Inversions in Diffusion Models},
     author={{\L}ukasz Staniszewski and {\L}ukasz Kuci{\'n}ski and Kamil Deja},
     booktitle={The Fourteenth International Conference on Learning Representations},
     year={2026},
     url={https://openreview.net/forum?id=8PaDdLuVKN}
}
```
