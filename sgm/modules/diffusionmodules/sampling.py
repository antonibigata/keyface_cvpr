"""
Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""

from collections import defaultdict
from typing import Dict, Union

import torch
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm
from einops import rearrange

from ...modules.diffusionmodules.sampling_utils import (
    get_ancestral_step,
    linear_multistep_coeff,
    to_d,
    to_neg_log_sigma,
    to_sigma,
    chunk_inputs,
)
from ...util import append_dims, default, instantiate_from_config

DEFAULT_GUIDER = {"target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"}


class BaseDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = True,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )
        self.verbose = verbose
        self.device = device

    def set_num_steps(self, num_steps: int):
        self.num_steps = num_steps

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None, strength=1.0):
        print("Num steps: ", self.num_steps if num_steps is None else num_steps)
        sigmas = self.discretization(self.num_steps if num_steps is None else num_steps, device=self.device)
        if strength != 1.0:
            init_timestep = min(int(len(sigmas) * strength), len(sigmas))
            t_start = max(len(sigmas) - init_timestep, 0)
            # sigmas[:t_start] = torch.ones_like(sigmas[:t_start]) * sigmas[t_start]
            sigmas = sigmas[t_start:]
        uc = default(uc, cond)

        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, uc):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc))
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator


class FIFODiffusionSampler(BaseDiffusionSampler):
    def __init__(self, lookahead=False, num_frames=14, num_partitions=4, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_frames = num_frames
        self.lookahead = lookahead
        self.num_partitions = num_partitions
        self.num_steps = self.num_frames * self.num_partitions
        self.fifo = []

    def get_sigma_gen(self, num_sigmas, total_n_frames):
        total = total_n_frames + num_sigmas - self.num_frames
        sigma_generator = range(total_n_frames + num_sigmas - self.num_frames - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=total,
                desc=f"Sampling with {self.__class__.__name__} for {total} steps",
            )
        return sigma_generator

    def prepare_sampling_loop(self, x, cond, uc=None):
        sigmas = self.discretization(self.num_steps, device=self.device)

        uc = default(uc, cond)

        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, sigmas, num_sigmas, cond, uc


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d


class EDMSampler(SingleStepDiffusionSampler):
    def __init__(self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc)
        if x.ndim == 5:
            denoised = rearrange(denoised, "(b t) c h w -> b c t h w", b=x.shape[0])

        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(euler_step, x, d, dt, next_sigma, denoiser, cond, uc)
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, strength=1.0):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps, strength=strength)

        for i in self.get_sigma_gen(num_sigmas):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1) if self.s_tmin <= sigmas[i] <= self.s_tmax else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
            )

        return x


class EDMSampleCFGplusplus(SingleStepDiffusionSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0):
        sigma_hat = sigma

        denoised, x_u = self.denoise(x, denoiser, sigma_hat, cond, uc)
        if x.ndim == 5:
            denoised = rearrange(denoised, "(b t) c h w -> b c t h w", b=x.shape[0])
            x_u = rearrange(x_u, "(b t) c h w -> b c t h w", b=x.shape[0])

        d = to_d(x, sigma_hat, x_u)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)
        next_sigma = append_dims(next_sigma, x.ndim)

        euler_step = self.euler_step(denoised, d, next_sigma)
        x = self.possible_correction_step(euler_step, x, d, dt, next_sigma, denoiser, cond, uc)
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, strength=1.0):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps, strength=strength)

        for i in self.get_sigma_gen(num_sigmas):
            s_in = x.new_ones([x.shape[0]])
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                None,
            )

        return x


def shift_latents(latents):
    # shift latents
    latents[:, :, :-1] = latents[:, :, 1:].clone()

    # add new noise to the last frame
    latents[:, :, -1] = torch.randn_like(latents[:, :, -1])

    return latents


class FIFOEDMSampler(FIFODiffusionSampler):
    """
    The problem is that the original implementation doesn't take into consideration the condition.
    So we need to check if this can work with the condition. Don't have time to check this now.
    """

    def __init__(self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def euler_step(self, x, d, dt):
        return x + dt * d

    def possible_correction_step(self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc):
        return euler_step

    def concatenate_list_dict(self, dict1):
        for k, v in dict1.items():
            if isinstance(v, list):
                dict1[k] = torch.cat(v, dim=0)
            else:
                dict1[k] = v
        return dict1

    def prepare_latents(self, x, c, uc, sigmas, num_sigmas):
        latents_list = []
        sigma_hat_list = []
        sigma_next_list = []
        c_list = defaultdict(list)
        uc_list = defaultdict(list)

        video = torch.load("/data/home/antoni/code/generative-models-dub/samples_z.pt")
        video = rearrange(video, "t c h w -> () c t h w")

        for k, v in c.items():
            if not isinstance(v, torch.Tensor):
                c_list[k] = v
                uc_list[k] = uc[k]

        if self.lookahead:
            for i in range(self.num_frames // 2):
                gamma = (
                    min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                    if self.s_tmin <= sigmas[i] <= self.s_tmax
                    else 0.0
                )
                sigma = sigmas[i]
                sigma_hat = sigma * (gamma + 1.0)
                if gamma > 0:
                    eps = torch.randn_like(video[:, :, [0]]) * self.s_noise
                    latents = video[:, :, [0]] + eps * append_dims(sigma_hat**2 - sigma**2, video.ndim) ** 0.5
                else:
                    latents = video[:, :, [0]]

                for k, v in c.items():
                    if isinstance(v, torch.Tensor):
                        c_list[k].append(v[[0]])
                for k, v in uc.items():
                    if isinstance(v, torch.Tensor):
                        uc_list[k].append(v[[0]])

                latents_list.append(latents)
                sigma_hat_list.append(sigma_hat)
                sigma_next_list.append(sigmas[i + 1])

        for i in range(num_sigmas - 1):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1) if self.s_tmin <= sigmas[i] <= self.s_tmax else 0.0
            )
            sigma = sigmas[i]
            sigma_hat = sigma * (gamma + 1.0)
            frame_idx = max(0, i - (num_sigmas - self.num_frames))
            print(frame_idx)
            if gamma > 0:
                eps = torch.randn_like(video[:, :, [frame_idx]]) * self.s_noise
                latents = video[:, :, [frame_idx]] + eps * append_dims(sigma_hat**2 - sigma**2, video.ndim) ** 0.5
            else:
                latents = video[:, :, [frame_idx]]

            for k, v in c.items():
                if isinstance(v, torch.Tensor):
                    c_list[k].append(
                        v[[frame_idx]] if v.shape[0] == video.shape[2] else v[[frame_idx // self.num_frames]]
                    )
            for k, v in uc.items():
                if isinstance(v, torch.Tensor):
                    uc_list[k].append(
                        v[[frame_idx]] if v.shape[0] == video.shape[2] else v[[frame_idx // self.num_frames]]
                    )

            latents_list.append(latents)
            sigma_hat_list.append(sigma_hat)
            sigma_next_list.append(sigmas[i + 1])

        latents = torch.cat(latents_list, dim=2)
        sigma_hat = torch.stack(sigma_hat_list, dim=0)
        sigma_next = torch.stack(sigma_next_list, dim=0)

        c_list = self.concatenate_list_dict(c_list)
        uc_list = self.concatenate_list_dict(uc_list)

        return latents, sigma_hat, sigma_next, c_list, uc_list

    def sampler_step(self, sigma_hat, next_sigma, denoiser, x, cond, uc=None):
        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc)
        if x.ndim == 5:
            x = rearrange(x, "b c t h w -> (b t) c h w")

        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(euler_step, x, d, dt, next_sigma, denoiser, cond, uc)
        return x

    def merge_cond_dict(self, cond, total_n_frames):
        for k, v in cond.items():
            if not isinstance(v, torch.Tensor):
                cond[k] = v
            else:
                if v.dim() == 5:
                    cond[k] = rearrange(v, "b c t h w -> (b t) c h w")
                elif v.dim() == 3 and v.shape[0] != total_n_frames:
                    cond[k] = rearrange(v, "b t c -> (b t) () c")
                else:
                    cond[k] = v
        return cond

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, strength=1.0):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc)

        x = rearrange(x, "b c h w -> () c b h w")
        cond = self.merge_cond_dict(cond, x.shape[2])
        uc = self.merge_cond_dict(uc, x.shape[2])
        total_n_frames = x.shape[2]
        latents, sigma_hat, sigma_next, cond, uc = self.prepare_latents(x, cond, uc, sigmas, num_sigmas)

        fifo_video_frames = []

        for i in self.get_sigma_gen(num_sigmas, total_n_frames):
            for rank in reversed(range(2 * self.num_partitions if self.lookahead else self.num_partitions)):
                start_idx = rank * (self.num_frames // 2) if self.lookahead else rank * self.num_frames
                midpoint_idx = start_idx + self.num_frames // 2
                end_idx = start_idx + self.num_frames

                chunk_x, sigma_hat_chunk, sigma_next_chunk, cond_chunk, uc_chunk = chunk_inputs(
                    latents, cond, uc, sigma_hat, sigma_next, start_idx, end_idx, self.num_frames
                )

                s_in = chunk_x.new_ones([chunk_x.shape[0]])

                out = self.sampler_step(
                    s_in * sigma_hat_chunk,
                    s_in * sigma_next_chunk,
                    denoiser,
                    chunk_x,
                    cond_chunk,
                    uc=uc_chunk,
                )
                if self.lookahead:
                    latents[:, :, midpoint_idx:end_idx] = rearrange(
                        out[-(self.num_frames // 2) :], "b c h w -> () c b h w"
                    )
                else:
                    latents[:, :, start_idx:end_idx] = rearrange(out, "b c h w -> () c b h w")
                del out

            first_frame_idx = self.num_frames // 2 if self.lookahead else 0
            latents = shift_latents(latents)
            fifo_video_frames.append(latents[:, :, [first_frame_idx]])

        return rearrange(torch.cat(fifo_video_frames, dim=2), "() c b h w -> b c h w")[-total_n_frames:]


class AncestralSampler(SingleStepDiffusionSampler):
    def __init__(self, eta=1.0, s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eta = eta
        self.s_noise = s_noise
        self.noise_sampler = lambda x: torch.randn_like(x)

    def ancestral_euler_step(self, x, denoised, sigma, sigma_down):
        d = to_d(x, sigma, denoised)
        dt = append_dims(sigma_down - sigma, x.ndim)

        return self.euler_step(x, d, dt)

    def ancestral_step(self, x, sigma, next_sigma, sigma_up):
        x = torch.where(
            append_dims(next_sigma, x.ndim) > 0.0,
            x + self.noise_sampler(x) * self.s_noise * append_dims(sigma_up, x.ndim),
            x,
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
            )

        return x


class LinearMultistepSampler(BaseDiffusionSampler):
    def __init__(
        self,
        order=4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.order = order

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        ds = []
        sigmas_cpu = sigmas.detach().cpu().numpy()
        for i in self.get_sigma_gen(num_sigmas):
            sigma = s_in * sigmas[i]
            denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), **kwargs)
            denoised = self.guider(denoised, sigma)
            d = to_d(x, sigma, denoised)
            ds.append(d)
            if len(ds) > self.order:
                ds.pop(0)
            cur_order = min(i + 1, self.order)
            coeffs = [linear_multistep_coeff(cur_order, sigmas_cpu, i, j) for j in range(cur_order)]
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))

        return x


class EulerEDMSampler(EDMSampler):
    def possible_correction_step(self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc):
        return euler_step


class EulerEDMSamplerPlusPlus(EDMSampleCFGplusplus):
    def possible_correction_step(self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc):
        return euler_step


class HeunEDMSampler(EDMSampler):
    def possible_correction_step(self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc):
        if torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            return euler_step
        else:
            denoised = self.denoise(euler_step, denoiser, next_sigma, cond, uc)
            d_new = to_d(euler_step, next_sigma, denoised)
            d_prime = (d + d_new) / 2.0

            # apply correction if noise level is not 0
            x = torch.where(append_dims(next_sigma, x.ndim) > 0.0, x + d_prime * dt, euler_step)
            return x


class EulerAncestralSampler(AncestralSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x = self.ancestral_euler_step(x, denoised, sigma, sigma_down)
        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)

        return x


class DPMPP2SAncestralSampler(AncestralSampler):
    def get_variables(self, sigma, sigma_down):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, sigma_down)]
        h = t_next - t
        s = t + 0.5 * h
        return h, s, t, t_next

    def get_mult(self, h, s, t, t_next):
        mult1 = to_sigma(s) / to_sigma(t)
        mult2 = (-0.5 * h).expm1()
        mult3 = to_sigma(t_next) / to_sigma(t)
        mult4 = (-h).expm1()

        return mult1, mult2, mult3, mult4

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, **kwargs):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x_euler = self.ancestral_euler_step(x, denoised, sigma, sigma_down)

        if torch.sum(sigma_down) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            x = x_euler
        else:
            h, s, t, t_next = self.get_variables(sigma, sigma_down)
            mult = [append_dims(mult, x.ndim) for mult in self.get_mult(h, s, t, t_next)]

            x2 = mult[0] * x - mult[1] * denoised
            denoised2 = self.denoise(x2, denoiser, to_sigma(s), cond, uc)
            x_dpmpp2s = mult[2] * x - mult[3] * denoised2

            # apply correction if noise level is not 0
            x = torch.where(append_dims(sigma_down, x.ndim) > 0.0, x_dpmpp2s, x_euler)

        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)
        return x


class DPMPP2MSampler(BaseDiffusionSampler):
    def get_variables(self, sigma, next_sigma, previous_sigma=None):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, next_sigma)]
        h = t_next - t

        if previous_sigma is not None:
            h_last = t - to_neg_log_sigma(previous_sigma)
            r = h_last / h
            return h, r, t, t_next
        else:
            return h, None, t, t_next

    def get_mult(self, h, r, t, t_next, previous_sigma):
        mult1 = to_sigma(t_next) / to_sigma(t)
        mult2 = (-h).expm1()

        if previous_sigma is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_sigma,
        sigma,
        next_sigma,
        denoiser,
        x,
        cond,
        uc=None,
    ):
        denoised = self.denoise(x, denoiser, sigma, cond, uc)

        h, r, t, t_next = self.get_variables(sigma, next_sigma, previous_sigma)
        mult = [append_dims(mult, x.ndim) for mult in self.get_mult(h, r, t, t_next, previous_sigma)]

        x_standard = mult[0] * x - mult[1] * denoised
        if old_denoised is None or torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d

            # apply correction if noise level is not 0 and not first step
            x = torch.where(append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard)

        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * sigmas[i - 1],
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
            )

        return x
