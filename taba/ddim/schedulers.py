# based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim_inverse.py

from dataclasses import dataclass
from typing import Optional

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddim_inverse import DDIMInverseScheduler
from diffusers.utils.outputs import BaseOutput


@dataclass
class AdvancedDDIMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
        eps (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted noise of the current timestep (no scaling).
    """

    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor
    eps: torch.Tensor


class AdvancedDDIMScheduler(DDIMScheduler):
    """
    `AdvancedDDIMScheduler` is edited version of `DDIMScheduler` that returns `eps` as well as `prev_sample` and `pred_original_sample`.
    """

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        timestep_idx: int,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        return_dict: bool = True,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        swap_eps: dict[int, torch.Tensor] = {},
    ) -> AdvancedDDIMSchedulerOutput:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`, defaults to `False`):
                If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`CycleDiffusion`].
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`]:
                [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        assert eta == 0.0, "eta cant be other than 0.0"

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        num_train_timesteps = self.config.num_train_timesteps  # type: ignore
        prev_timestep = timestep - num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prediction_type = self.config.prediction_type  # type: ignore
        if prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        else:
            raise ValueError(
                f"prediction_type given as {prediction_type} must be one of `epsilon`, `sample`, or" " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        thresholding = self.config.thresholding  # type: ignore
        clip_sample = self.config.clip_sample  # type: ignore
        clip_sample_range = self.config.clip_sample_range  # type: ignore
        if thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif clip_sample:
            pred_original_sample = pred_original_sample.clamp(-clip_sample_range, clip_sample_range)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        return AdvancedDDIMSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample, eps=pred_epsilon
        )


class AdvancedDDIMInverseScheduler(DDIMInverseScheduler):
    """
    `AdvancedDDIMInverseScheduler` is edited version of `DDIMInverseScheduler` that returns `eps` as well as `prev_sample` and `pred_original_sample`.
    """

    @torch.no_grad()
    def forward_diffusion(
        self,
        x0: torch.Tensor,
        timestep: torch.LongTensor,
        fixed_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Vectorized closed-form forward diffusion: computes x_t for each t in timesteps.
        Returns a dict mapping t -> x_t.
        """
        if fixed_noise is None:
            fixed_noise = torch.randn_like(x0)

        alphas_bar = self.alphas_cumprod[timestep].to(x0.device)
        sqrt_alpha_bar = alphas_bar**0.5
        sqrt_one_minus_alpha_bar = (1 - alphas_bar) ** 0.5
        xts = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * fixed_noise

        return xts

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        timestep_idx: int,
        swap_eps: dict[int, torch.Tensor] = {},
        return_dict: bool = True,
    ) -> AdvancedDDIMSchedulerOutput:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`, defaults to `False`):
                If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`CycleDiffusion`].
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddim_inverse.DDIMInverseSchedulerOutput`] or
                `tuple`.

        Returns:
            [`~schedulers.scheduling_ddim_inverse.DDIMInverseSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddim_inverse.DDIMInverseSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.

        """
        # 1. get previous step value (=t+1)
        prev_timestep = timestep
        num_train_timesteps = self.config.num_train_timesteps  # type: ignore
        timestep = min(timestep - num_train_timesteps // self.num_inference_steps, num_train_timesteps - 1)

        # 2. compute alphas, betas
        # change original implementation to exactly match noise levels for analogous forward process
        alpha_prod_t = self.alphas_cumprod[timestep] if timestep >= 0 else self.initial_alpha_cumprod
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prediction_type = self.config.prediction_type  # type: ignore
        assert prediction_type == "epsilon", f"prediction_type must be 'epsilon', got {prediction_type}"
        if timestep_idx in swap_eps.keys():
            # print(f"Swapped eps for timestep {timestep_idx}")
            pred_epsilon = swap_eps[timestep_idx].to(model_output.device)
        else:
            pred_epsilon = model_output
        pred_original_sample = (sample - beta_prod_t ** (0.5) * pred_epsilon) / alpha_prod_t ** (0.5)

        clip_sample = self.config.clip_sample  # type: ignore
        clip_sample_range = self.config.clip_sample_range  # type: ignore
        if clip_sample:
            pred_original_sample = pred_original_sample.clamp(-clip_sample_range, clip_sample_range)

        # 5. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon

        # 6. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        return AdvancedDDIMSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample, eps=pred_epsilon
        )
