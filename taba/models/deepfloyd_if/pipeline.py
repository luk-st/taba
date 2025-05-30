from logging import ERROR
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.deepfloyd_if.pipeline_if import IFPipeline
from diffusers.pipelines.deepfloyd_if.pipeline_output import IFPipelineOutput
from diffusers.pipelines.deepfloyd_if.safety_checker import IFSafetyChecker
from diffusers.pipelines.deepfloyd_if.watermark import IFWatermarker
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.utils.logging import disable_progress_bar, enable_progress_bar, get_verbosity, set_verbosity
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPImageProcessor, T5EncoderModel, T5Tokenizer

from taba.ddim.schedulers import (
    AdvancedDDIMInverseScheduler,
    AdvancedDDIMScheduler,
    AdvancedDDIMSchedulerOutput,
)

IF_MODEL_NAME = "DeepFloyd/IF-I-XL-v1.0"
IF_MODEL_DTYPE = torch.float16


class CustomIFPipeline(IFPipeline):

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        safety_checker: Optional[IFSafetyChecker],
        feature_extractor: Optional[CLIPImageProcessor],
        watermarker: Optional[IFWatermarker],
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            watermarker=None,
        )
        self.sampling_scheduler = AdvancedDDIMScheduler.from_config(self.scheduler.config)
        self.inversion_scheduler = AdvancedDDIMInverseScheduler.from_config(self.scheduler.config)

    def prepare_latents(
        self,
        n_images: int,
        height: Optional[int] = None,
        width: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ):
        height = height or self.unet.config.sample_size  # type: ignore
        width = width or self.unet.config.sample_size  # type: ignore
        num_channels = self.unet.config.in_channels  # type: ignore

        if dtype is None:
            dtype = self.unet.dtype

        shape = (n_images, num_channels, height, width)
        if isinstance(generator, list) and len(generator) != n_images:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested {n_images} images."
                f" Make sure number of images matches the length of the generators."
            )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        if self.scheduler.init_noise_sigma != 1.0:
            print(f"Weird initial noise sigma: {self.scheduler.init_noise_sigma}")
        return latents

    @torch.no_grad()
    def sample(
        self,
        prompt: Union[str, List[str]],
        noise: torch.Tensor,
        num_inference_steps: int = 100,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        from_each_t: bool = False,
        swap_eps: dict[int, torch.Tensor] = {},
        swap_xt: dict[int, torch.Tensor] = {},
        edit_prompt: Optional[Union[str, List[str]]] = None,
        edit_t_start: Optional[int] = None,
    ):
        assert swap_xt == {} and swap_eps == {}, "swap_xt and swap_eps cannot both be provided"
        self.scheduler = self.sampling_scheduler  # type: ignore
        return self.__call__(
            prompt=prompt,
            latents=noise,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            height=height,
            width=width,
            eta=eta,
            generator=generator,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type="pt",
            return_dict=True,
            callback=callback,
            callback_steps=callback_steps,
            clean_caption=clean_caption,
            cross_attention_kwargs=cross_attention_kwargs,
            from_each_t=from_each_t,
            is_inversion=False,
            swap_eps=swap_eps,
            swap_xt=swap_xt,
            edit_prompt=edit_prompt,
            edit_t_start=edit_t_start,
        )

    @torch.no_grad()
    def invert(
        self,
        prompt: Union[str, List[str]],
        image: torch.Tensor,
        num_inference_steps: int = 100,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        from_each_t: bool = False,
        swap_eps: dict[int, torch.Tensor] = {},
        swap_xt: dict[int, torch.Tensor] = {},
        forward_before_t: int | None = None,
        fixed_noise_generator: torch.Generator | None = None,
        is_first_batch: bool = False,
    ):
        self.scheduler = self.inversion_scheduler  # type: ignore
        return self.__call__(
            prompt=prompt,
            latents=image,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            height=height,
            width=width,
            eta=eta,
            generator=generator,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type="pt",
            return_dict=True,
            callback=callback,
            callback_steps=callback_steps,
            clean_caption=clean_caption,
            cross_attention_kwargs=cross_attention_kwargs,
            from_each_t=from_each_t,
            is_inversion=True,
            swap_eps=swap_eps,
            swap_xt=swap_xt,
            forward_before_t=forward_before_t,
            fixed_noise_generator=fixed_noise_generator,
            is_first_batch=is_first_batch,
            edit_prompt=None,
            edit_t_start=None,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        edit_prompt: Optional[Union[str, List[str]]] = None,
        edit_t_start: Optional[int] = None,
        latents: Optional[torch.Tensor] = None,
        num_inference_steps: int = 100,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        is_inversion: bool = False,
        from_each_t: bool = False,
        swap_eps: dict[int, torch.Tensor] = {},
        swap_xt: dict[int, torch.Tensor] = {},
        fixed_noise_generator: torch.Generator | None = None,
        forward_before_t: int | None = None,
        is_first_batch: bool = False,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            latents (`torch.Tensor`, *optional*):
                Pre-generated latents. Can be used to easily tweak latents, *e.g.* prompt weighting. If not defined, latents will be generated.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.IFPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.IFPipelineOutput`] if `return_dict` is True, otherwise a `tuple. When
            returning a tuple, the first element is a list with the generated images, and the second element is a list
            of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work" (nsfw)
            or watermarked content, according to the `safety_checker`.
        """
        assert swap_xt == {} or swap_eps == {}, "swap_xt and swap_eps cannot both be provided"
        if edit_prompt is not None:
            assert is_inversion is False, "edit_prompt can only be used when is_inversion is False"
            if edit_t_start is None:
                edit_t_start = 0

        if forward_before_t is not None:
            assert is_inversion is True, "forward only when is_inversion is True"
            assert (
                0 <= forward_before_t <= num_inference_steps
            ), "forward_before_t must be None or not greater than num_inference_steps"
            if forward_before_t != 0:
                assert not from_each_t, "from_each_t must be False when positive forward_before_t is provided"
            else:
                assert isinstance(
                    self.scheduler, AdvancedDDIMInverseScheduler
                ), "Scheduler must be AdvancedDDIMInverseScheduler when forward_before_t is 0"

        device = self._execution_device
        self.check_inputs(prompt, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # prompts
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError("Either prompt or prompt_embeds must be provided")
        do_classifier_free_guidance = guidance_scale > 1.0

        old_verbosity = get_verbosity()
        set_verbosity(ERROR)
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clean_caption=clean_caption,
        )
        if edit_prompt is not None:
            prompt_embeds_edit, negative_prompt_embeds_edit = self.encode_prompt(
                edit_prompt,
                do_classifier_free_guidance,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                negative_prompt=None,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                clean_caption=clean_caption,
            )
            if do_classifier_free_guidance:
                prompt_embeds_edit = torch.cat([negative_prompt_embeds_edit, prompt_embeds_edit])
        else:
            prompt_embeds_edit = None
        set_verbosity(old_verbosity)
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        if hasattr(self, "text_encoder_offload_hook") and self.text_encoder_offload_hook is not None:
            self.text_encoder_offload_hook.offload()

        # timesteps
        if timesteps is not None:
            self.scheduler.set_timesteps(timesteps=timesteps, device=device)
            timesteps = self.scheduler.timesteps  # type: ignore
            num_inference_steps = len(timesteps)
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps  # type: ignore
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(0)
        progress_bar_steps = num_inference_steps

        # latents
        if latents is None:
            latents = self.prepare_latents(
                n_images=batch_size * num_images_per_prompt,
                height=height,
                width=width,
                dtype=self.unet.dtype,
                device=device,
                generator=generator,
            )
        else:
            assert latents.shape[0] == batch_size * num_images_per_prompt
            latents = latents.to(device)

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        all_latents = []
        all_t_latents = []
        all_t_eps = []
        all_t_pred_xstart = []

        if forward_before_t is not None and forward_before_t > 0:
            assert forward_before_t <= len(
                timesteps
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

            # we move forward in time, inversion will start from middle of the timesteps
            timesteps = timesteps[forward_before_t:]
            progress_bar_steps = len(timesteps)

            fixed_noise = torch.randn(
                latents.shape,
                generator=fixed_noise_generator,
                dtype=latents.dtype,
                device=latents.device,
            )
            latents = self.scheduler.forward_diffusion(
                x0=latents,
                timestep=timestep_forward,
                fixed_noise=fixed_noise,
            )
            if is_first_batch:
                print(
                    f"###\nForward diffusion:\n - {len(self.scheduler.timesteps)=}\n - {len(timesteps)=}\n - {(forward_before_t)=}\n - {timestep_forward=}\n - {self.scheduler.timesteps=}\n###"
                )

        with self.progress_bar(total=progress_bar_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                model_input = self.scheduler.scale_model_input(model_input, t)

                # predict the noise residual

                if prompt_embeds_edit is not None and is_inversion is False and i >= edit_t_start:
                    enc_hidden_states = prompt_embeds_edit
                else:
                    enc_hidden_states = prompt_embeds
                noise_pred = self.unet(
                    model_input,
                    t,
                    encoder_hidden_states=enc_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
                    noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

                if self.scheduler.config.variance_type not in ["learned", "learned_range"] or "ddim" in self.scheduler.config._class_name.lower():  # type: ignore
                    noise_pred, _ = noise_pred.split(model_input.shape[1], dim=1)
                outputs: AdvancedDDIMSchedulerOutput = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=True, timestep_idx=i, swap_eps=swap_eps
                )
                if i in swap_xt.keys():
                    outputs.prev_sample = swap_xt[i]
                latents = outputs.prev_sample
                if from_each_t:
                    all_t_latents.append(latents.clone())
                    all_t_eps.append(noise_pred.clone())
                    all_t_pred_xstart.append(outputs.pred_original_sample.clone())

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                else:
                    print("Skipping step, warmup!")
        image = latents

        # if output_type == "pil":
        #     image = (image / 2 + 0.5).clamp(0, 1)  # type: ignore
        #     image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        #     image = self.numpy_to_pil(image)
        if output_type == "pt":
            if hasattr(self, "unet_offload_hook") and self.unet_offload_hook is not None:
                self.unet_offload_hook.offload()
        # else:
        #     image = (image / 2 + 0.5).clamp(0, 1)  # type: ignore
        #     image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        self.maybe_free_model_hooks()

        if from_each_t is True:
            if is_inversion:
                return {
                    "latents": latents,
                    "all_t_latents": torch.stack(all_t_latents).cpu(),
                    "all_t_eps_latents": torch.stack(all_t_eps).cpu(),
                    "all_t_pred_xstart_latents": torch.stack(all_t_pred_xstart).cpu(),
                }
            else:
                return {
                    "samples": latents,
                    "all_t_samples": torch.stack(all_t_latents).cpu(),
                    "all_t_eps_samples": torch.stack(all_t_eps).cpu(),
                    "all_t_pred_xstart_samples": torch.stack(all_t_pred_xstart).cpu(),
                }
        else:
            if is_inversion:
                return {
                    "latents": latents,
                }
            else:
                return {
                    "samples": latents,
                }


def initialize_if_ddim_pipeline() -> CustomIFPipeline:
    disable_progress_bar()
    old_verbosity = get_verbosity()
    set_verbosity(ERROR)
    pipeline = CustomIFPipeline.from_pretrained(IF_MODEL_NAME, torch_dtype=IF_MODEL_DTYPE)
    pipeline.enable_model_cpu_offload()
    set_verbosity(old_verbosity)
    enable_progress_bar()
    return pipeline
