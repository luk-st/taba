# Based on: https://github.com/huggingface/diffusers/blob/074e12358bc17e7dbe111ea4f62f05dbae8a49d5/src/diffusers/pipelines/dit/pipeline_dit.py

from typing import Dict, List, Optional, Tuple, Union

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.transformers.dit_transformer_2d import DiTTransformer2DModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from tqdm import tqdm

from taba.ddim.schedulers import (
    AdvancedDDIMInverseScheduler,
    AdvancedDDIMScheduler,
    AdvancedDDIMSchedulerOutput,
)
from taba.models.dit.constants import DIT_IMAGENET_CLASSES_SMALL


def _rearrange_t_to_b(tensor: torch.Tensor) -> torch.Tensor:
    return rearrange(tensor, "t b c h w -> b t c h w")


def _rearrange_b_to_t(tensor: torch.Tensor) -> torch.Tensor:
    return rearrange(tensor, "b t c h w -> t b c h w")


class CustomDiTPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation based on a Transformer backbone instead of a UNet.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        transformer ([`DiTTransformer2DModel`]):
            A class conditioned `DiTTransformer2DModel` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        scheduler ([`DDIMScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "transformer->vae"

    def __init__(
        self,
        transformer: DiTTransformer2DModel,
        vae: AutoencoderKL,
        scheduler: KarrasDiffusionSchedulers,
        id2label: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, vae=vae, scheduler=scheduler)

        # create a imagenet -> id dictionary for easier use
        self.labels = {}
        if id2label is not None:
            for key, value in id2label.items():
                for label in value.split(","):
                    self.labels[label.lstrip().rstrip()] = int(key)
            self.labels = dict(sorted(self.labels.items()))

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        r"""

        Map label strings from ImageNet to corresponding class ids.

        Parameters:
            label (`str` or `dict` of `str`):
                Label strings to be mapped to class ids.

        Returns:
            `list` of `int`:
                Class ids to be processed by pipeline.
        """

        if not isinstance(label, list):
            label = list(label)

        for l in label:
            if l not in self.labels:
                raise ValueError(
                    f"{l} does not exist. Please make sure to select one of the following labels: \n {self.labels}."
                )

        return [self.labels[l] for l in label]

    @torch.no_grad()
    def vae_decode(self, latents: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        n = latents.shape[0]
        outs = []
        for idx_start in tqdm(range(0, n, batch_size), desc="VAE Decoding"):
            b_latents = latents[idx_start : idx_start + batch_size].to(self._execution_device)
            b_latents = 1 / self.vae.config.scaling_factor * b_latents
            with torch.no_grad():
                samples = self.vae.decode(b_latents).sample
            outs.append(samples.cpu())
        return torch.cat(outs)

    @torch.no_grad()
    def vae_encode(self, images: torch.Tensor) -> torch.Tensor:
        latents_0 = self.vae.encode(images).latent_dist.sample()
        latents_0 = latents_0 * self.vae.config.scaling_factor
        return latents_0

    @torch.no_grad()
    def fix_timesteps(self, t: Union[int, float], latent_model_input: torch.Tensor) -> torch.Tensor:
        timesteps = t
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = latent_model_input.device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
        elif len(timesteps.shape) == 0:  # type: ignore
            timesteps = timesteps[None].to(latent_model_input.device)  # type: ignore
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(latent_model_input.shape[0])  # type: ignore
        return timesteps

    @torch.no_grad()
    def prepare_denoiser_input(
        self, latents: torch.Tensor, guidance_scale: float, batch_size: int, class_labels: List[int] | torch.Tensor
    ):
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents
        if isinstance(class_labels, list):
            class_labels = torch.tensor(class_labels)
        class_labels = class_labels.to(self._execution_device).reshape(-1)  # type: ignore
        class_null = torch.tensor([1000] * batch_size, device=self._execution_device)
        class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels  # type: ignore
        return latent_model_input, class_labels_input

    @torch.no_grad()
    def noise_denoise(
        self,
        latents: torch.Tensor,
        guidance_scale: float,
        batch_size: int,
        num_inference_steps: int,
        mode: str,
        class_labels: List[int],
        from_each_t: bool = False,
        swap_eps: dict[int, torch.Tensor] = {},
        swap_xt: dict[int, torch.Tensor] = {},
        fixed_noise_generator: torch.Generator | None = None,
        forward_before_t: int | None = None,
        is_first_batch: bool = False,
    ) -> torch.Tensor:

        assert mode in ["noise", "denoise"], "Mode must be either 'noise' or 'denoise'"
        assert (
            mode == "noise" or forward_before_t is None
        ), "forward_diffusion_end_step must be None when mode is 'denoise'"
        if from_each_t:
            assert forward_before_t is None, "forward_diffusion_end_step must be None when from_each_t is True"

        latent_channels = self.transformer.config.in_channels

        latent_model_input, class_labels_input = self.prepare_denoiser_input(
            latents, guidance_scale, batch_size, class_labels
        )

        all_t_samples = []
        all_t_eps = []
        all_t_pred_xstart = []

        timesteps = self.scheduler.timesteps  # type: ignore
        timesteps_ids = list(range(len(timesteps)))
        timesteps_ids = timesteps_ids if mode == "noise" else timesteps_ids[::-1]

        if forward_before_t is not None and forward_before_t > 0:
            assert isinstance(
                self.scheduler, AdvancedDDIMInverseScheduler
            ), "Scheduler must be AdvancedDDIMInverseScheduler when forward_diffusion_end_step is provided"
            assert (
                1 <= forward_before_t <= len(timesteps_ids)
            ), "forward_diffusion_end_step must be less than or equal to the number of timesteps"

            # use correct timestep for forward diffusion
            # if forward_before_t is number of inference steps, use scheduler last output
            timestep_forward = (
                timesteps[forward_before_t]
                if forward_before_t < len(timesteps)
                else torch.tensor(self.scheduler.config.get("num_train_timesteps") - 1).to(
                    dtype=self.scheduler.timesteps[0].dtype
                )
            )
            # we mo
            timesteps = timesteps[forward_before_t:]
            timesteps_ids = timesteps_ids[forward_before_t:]

            fixed_noise = torch.randn(
                latent_model_input.shape,
                generator=fixed_noise_generator,
                dtype=latent_model_input.dtype,
                device=latent_model_input.device,
            )
            latent_model_input = self.scheduler.forward_diffusion(
                x0=latent_model_input,
                timestep=timestep_forward,
                fixed_noise=fixed_noise,
            )
            if is_first_batch:
                print(
                    f"###\nForward diffusion:\n - {len(self.scheduler.timesteps)=}\n - {len(timesteps)=}\n - {(forward_before_t)=}\n - {timestep_forward=}\n - {self.scheduler.timesteps=}\n###"
                )

        for t_idx, t in zip(timesteps_ids, timesteps):
            # if guidance_scale > 1:
            #     half = latent_model_input[: len(latent_model_input) // 2]
            #     latent_model_input = torch.cat([half, half], dim=0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)  # type: ignore

            t_pred = self.fix_timesteps(t, latent_model_input)

            # predict noise model_output
            noise_pred = self.transformer(latent_model_input, timestep=t_pred, class_labels=class_labels_input).sample

            # perform guidance
            # if guidance_scale > 1:
            #     eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
            #     cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

            #     half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            #     eps = torch.cat([half_eps, half_eps], dim=0)

            #     noise_pred = torch.cat([eps, rest], dim=1)

            # learned sigma
            if self.transformer.config.out_channels // 2 == latent_channels:
                model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
            else:
                model_output = noise_pred

            # compute previous image: x_t -> x_t-1
            output: AdvancedDDIMSchedulerOutput = self.scheduler.step(model_output=model_output, timestep=t, sample=latent_model_input, timestep_idx=t_idx, swap_eps=swap_eps)  # type: ignore
            if t_idx in swap_xt.keys():
                output.prev_sample = swap_xt[t_idx]
            latent_model_input = output.prev_sample
            if from_each_t:
                all_t_samples.append(latent_model_input.clone())
                all_t_eps.append(output.eps.clone())
                all_t_pred_xstart.append(output.pred_original_sample.clone())

        latents = latent_model_input
        if from_each_t is True:
            return {
                "samples": latents,
                "all_t_samples": torch.stack(all_t_samples).cpu(),
                "all_t_eps": torch.stack(all_t_eps).cpu(),
                "all_t_pred_xstart": torch.stack(all_t_pred_xstart).cpu(),
            }
        else:
            return latents

    def set_inverse_scheduler(self, num_inference_steps: int):
        self.scheduler = AdvancedDDIMInverseScheduler.from_config(self.scheduler.config)  # type: ignore
        self.scheduler.set_timesteps(num_inference_steps)  # type: ignore

    def set_sampling_scheduler(self, num_inference_steps: int):
        self.scheduler = AdvancedDDIMScheduler.from_config(self.scheduler.config)  # type: ignore
        self.scheduler.set_timesteps(num_inference_steps)  # type: ignore

    @torch.no_grad()
    def ddim_inverse(
        self,
        latents_x_0: torch.Tensor,
        guidance_scale: float,
        batch_size: int,
        num_inference_steps: int,
        class_labels: torch.Tensor,
        from_each_t: bool = False,
        swap_eps: dict[int, torch.Tensor] = {},
        swap_xt: dict[int, torch.Tensor] = {},
        forward_before_t: int | None = None,
        fixed_noise_generator: torch.Generator | None = None,
    ) -> torch.Tensor:

        assert swap_xt == {} or swap_eps == {}, "swap_xt and swap_eps cannot both be provided"
        assert (
            forward_before_t is None or forward_before_t <= num_inference_steps
        ), "forward_diffusion_end_step must be None or not greater than num_inference_steps"

        self.set_inverse_scheduler(num_inference_steps)

        n_latents = latents_x_0.shape[0]
        all_latents = []
        all_t_latents = []
        all_t_eps = []
        all_t_pred_xstart = []
        for idx_start in tqdm(range(0, n_latents, batch_size), desc="DDIM Inversion"):
            b_latents_x_0 = latents_x_0[idx_start : idx_start + batch_size].to(self._execution_device)
            b_class_labels = class_labels[idx_start : idx_start + batch_size].to(self._execution_device)
            curr_swap_eps = {
                k: swap_eps[k][idx_start : idx_start + batch_size].to(self._execution_device) for k in swap_eps.keys()
            }
            curr_swap_xt = {
                k: swap_xt[k][idx_start : idx_start + batch_size].to(self._execution_device) for k in swap_xt.keys()
            }
            output = self.noise_denoise(
                latents=b_latents_x_0,
                guidance_scale=guidance_scale,
                batch_size=batch_size,
                mode="noise",
                num_inference_steps=num_inference_steps,
                class_labels=b_class_labels,
                from_each_t=from_each_t,
                swap_eps=curr_swap_eps,
                swap_xt=curr_swap_xt,
                fixed_noise_generator=fixed_noise_generator,
                forward_before_t=forward_before_t,
                is_first_batch=(idx_start == 0),
            )
            if from_each_t:
                all_t_latents.append(_rearrange_t_to_b(output["all_t_samples"]))
                all_t_eps.append(_rearrange_t_to_b(output["all_t_eps"]))
                all_t_pred_xstart.append(_rearrange_t_to_b(output["all_t_pred_xstart"]))
                latents = output["samples"]
            else:
                latents = output
            all_latents.append(latents.clone())

        all_latents = torch.cat(all_latents, dim=0)
        if from_each_t:
            all_t_latents = _rearrange_b_to_t(torch.cat(all_t_latents, dim=0))
            all_t_eps = _rearrange_b_to_t(torch.cat(all_t_eps, dim=0))
            all_t_pred_xstart = _rearrange_b_to_t(torch.cat(all_t_pred_xstart, dim=0))
            return {
                "latents": all_latents,
                "all_t_latents": all_t_latents,
                "all_t_eps_latents": all_t_eps,
                "all_t_pred_xstart_latents": all_t_pred_xstart,
            }
        else:
            return {"latents": all_latents}

    @torch.no_grad()
    def ddim(
        self,
        latents_x_T: torch.Tensor,
        guidance_scale: float,
        batch_size: int,
        num_inference_steps: int,
        class_labels: torch.Tensor,
        from_each_t: bool = False,
    ) -> torch.Tensor:

        self.set_sampling_scheduler(num_inference_steps)

        n_latents = latents_x_T.shape[0]
        all_samples = []
        all_t_samples = []
        all_t_eps = []
        all_t_pred_xstart = []
        for idx_start in tqdm(range(0, n_latents, batch_size), desc="DDIM Sampling"):
            b_latents_x_T = latents_x_T[idx_start : idx_start + batch_size].to(self._execution_device)
            b_class_labels = class_labels[idx_start : idx_start + batch_size].to(self._execution_device)
            output = self.noise_denoise(
                latents=b_latents_x_T,
                guidance_scale=guidance_scale,
                batch_size=batch_size,
                mode="denoise",
                num_inference_steps=num_inference_steps,
                class_labels=b_class_labels,
                from_each_t=from_each_t,
            )
            if from_each_t:
                all_t_samples.append(_rearrange_t_to_b(output["all_t_samples"]))
                all_t_eps.append(_rearrange_t_to_b(output["all_t_eps"]))
                all_t_pred_xstart.append(_rearrange_t_to_b(output["all_t_pred_xstart"]))
                samples = output["samples"]
            else:
                samples = output
            all_samples.append(samples.clone())

        all_samples = torch.cat(all_samples, dim=0)
        if from_each_t:
            all_t_samples = _rearrange_b_to_t(torch.cat(all_t_samples, dim=0))
            all_t_eps = _rearrange_b_to_t(torch.cat(all_t_eps, dim=0))
            all_t_pred_xstart = _rearrange_b_to_t(torch.cat(all_t_pred_xstart, dim=0))
            return {
                "samples": all_samples,
                "all_t_samples": all_t_samples,
                "all_t_eps_samples": all_t_eps,
                "all_t_pred_xstart_samples": all_t_pred_xstart,
            }
        else:
            return {"samples": all_samples}

    @torch.no_grad()
    def prepare_latents(self, n_samples: int, generator: Optional[torch.Generator] = None):
        latent_size = self.transformer.config.sample_size
        latent_channels = self.transformer.config.in_channels
        latents = randn_tensor(
            shape=(n_samples, latent_channels, latent_size, latent_size),
            generator=generator,
            device=self._execution_device,
            dtype=self.transformer.dtype,
        )
        return latents

    @torch.no_grad()
    def __call__(
        self,
        class_labels: List[int],
        noise: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            class_labels (List[int]):
                List of ImageNet class labels for the images to be generated.
            guidance_scale (`float`, *optional*, defaults to 1.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 250):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.
        """

        if noise is not None:
            assert noise.shape[0] == len(class_labels)

        batch_size = len(class_labels)

        if noise is None:
            latents = self.prepare_latents(n_samples=batch_size, generator=generator)
        else:
            latents = noise

        latents = self.noise_denoise(
            latents=latents,
            guidance_scale=guidance_scale,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            class_labels=class_labels,
        )

        samples = self.vae_decode(latents=latents)
        samples = (samples / 2 + 0.5).clamp(0, 1)
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (samples,)
        return ImagePipelineOutput(images=samples)


if __name__ == "__main__":
    # example run
    example_labels = DIT_IMAGENET_CLASSES_SMALL
    # load pipeline
    pipe = CustomDiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # prepare data
    class_ids = pipe.get_label_ids(example_labels)
    generator = torch.manual_seed(33)
    noise = pipe.prepare_latents(n_samples=16, generator=generator)

    # generate images from noise
    samples_x_0 = pipe.ddim(
        latents_x_T=noise, guidance_scale=1.0, batch_size=16, num_inference_steps=100, class_labels=class_ids
    )
    vae_images1 = pipe.vae_decode(latents=samples_x_0)
    # from taba.utils import plot_diffusion
    # plot_diffusion(vae_images1.cpu().float(), nrow=8)

    # invert images to noise
    samples_x_T = pipe.ddim_inverse(
        latents_x_0=samples_x_0, guidance_scale=1.0, batch_size=16, num_inference_steps=100, class_labels=class_ids
    )

    # reconstruct images from inverted noise
    recons_x_0 = pipe.ddim(
        latents_x_T=samples_x_T, guidance_scale=1.0, batch_size=16, num_inference_steps=100, class_labels=class_ids
    )
    vae_images2 = pipe.vae_decode(latents=recons_x_0)
    # plot_diffusion(vae_images2.cpu().float(), nrow=8)
    # plot_diffusion((1-(vae_images2 - vae_images1).abs()).cpu().float(), nrow=8)
