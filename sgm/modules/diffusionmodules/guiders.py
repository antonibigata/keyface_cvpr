import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from ...util import append_dims, default

logpy = logging.getLogger(__name__)


class Guider(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        pass

    def prepare_inputs(
        self, x: torch.Tensor, s: float, c: Dict, uc: Dict
    ) -> Tuple[torch.Tensor, float, Dict]:
        pass


class VanillaCFG(Guider):
    def __init__(
        self, scale: float, low_sigma: float = 0.0, high_sigma: float = float("inf")
    ):
        self.scale = scale
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma

    def set_scale(self, scale: float):
        self.scale = scale

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x_u, x_c = x.chunk(2)

        x_pred = x_u + self.scale * (x_c - x_u)
        return x_pred

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            if k in [
                "vector",
                "crossattn",
                "concat",
                "audio_emb",
                "image_embeds",
                "landmarks",
                "masks",
                "gt",
                "valence",
                "arousal",
            ]:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            elif k == "reference":
                c_out["reference"] = []
                for i in range(len(c[k])):
                    c_out["reference"].append(torch.cat((uc[k][i], c[k][i]), 0))
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out


class VanillaSTG(Guider):
    def __init__(
        self,
        scale_spatial: float,
        scale_temporal: float,
        low_sigma: float = 0.0,
        high_sigma: float = float("inf"),
        layer_skip: int = 8,
    ):
        self.scale_spatial = scale_spatial
        self.scale_temporal = scale_temporal
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma
        self.layer_skip = layer_skip

    def set_scale(self, scale_spatial: float, scale_temporal: float):
        self.scale_spatial = scale_spatial
        self.scale_temporal = scale_temporal

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x_c, x_spatial, x_temporal = x.chunk(3)
        x_pred = (
            x_c
            + self.scale_spatial * (x_c - x_spatial)
            + self.scale_temporal * (x_c - x_temporal)
        )
        return x_pred

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            if k in [
                "vector",
                "crossattn",
                "concat",
                "audio_emb",
                "image_embeds",
                "landmarks",
                "masks",
                "gt",
                "valence",
                "arousal",
            ]:
                c_out[k] = torch.cat((c[k], c[k], c[k]), 0)
            elif k == "reference":
                c_out["reference"] = []
                for i in range(len(c[k])):
                    c_out["reference"].append(torch.cat((c[k][i], c[k][i], c[k][i]), 0))
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]

        c_out["skip_spatial_attention_at"] = [None, self.layer_skip, None]
        c_out["skip_temporal_attention_at"] = [None, None, self.layer_skip]

        return torch.cat([x] * 3), torch.cat([s] * 3), c_out


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def project(v0: torch.Tensor, v1: torch.Tensor):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = F.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def normalized_guidance(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    momentum_buffer: MomentumBuffer = None,
    eta: float = 1.0,
    norm_threshold: float = 0.0,
):
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
    return pred_guided


class APGGuider(VanillaCFG):
    def __init__(
        self,
        scale: float,
        momentum: float = -0.75,
        eta: float = 0.0,
        norm_threshold: float = 2.5,
    ):
        super().__init__(scale)
        self.momentum_buffer = MomentumBuffer(momentum)
        self.eta = eta
        self.norm_threshold = norm_threshold

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x_u, x_c = x.chunk(2)
        return normalized_guidance(
            x_c, x_u, self.scale, self.momentum_buffer, self.eta, self.norm_threshold
        )


class VanillaCFGplusplus(VanillaCFG):
    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x_u, x_c = x.chunk(2)

        x_pred = x_u + self.scale * (x_c - x_u)
        return x_pred, x_u


class KarrasGuider(VanillaCFG):
    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            if k in [
                "vector",
                "crossattn",
                "concat",
                "audio_emb",
                "image_embeds",
                "landmarks",
                "valence",
                "arousal",
            ]:
                c_out[k] = torch.cat((c[k], c[k]), 0)
            elif k == "reference":
                c_out["reference"] = []
                for i in range(len(c[k])):
                    c_out["reference"].append(torch.cat((c[k][i], c[k][i]), 0))
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out


class MultipleCondVanilla(Guider):
    def __init__(self, scales, condition_names) -> None:
        assert len(scales) == len(condition_names)
        self.scales = scales
        # self.condition_names = condition_names
        self.n_conditions = len(scales)
        self.map_cond_name = {
            "audio_emb": "audio_emb",
            "cond_frames_without_noise": "crossattn",
            "cond_frames": "concat",
        }
        self.condition_names = [
            self.map_cond_name.get(cond_name, cond_name)
            for cond_name in condition_names
        ]
        print("Condition names: ", self.condition_names)

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        outs = x.chunk(self.n_conditions + 1)
        x_full_cond = outs[0]
        x_pred = (1 + sum(self.scales)) * x_full_cond
        for i, scale in enumerate(self.scales):
            x_pred -= scale * outs[i + 1]
        return x_pred

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        # The first element is the full condition
        for k in c:
            if k in [
                "vector",
                "crossattn",
                "concat",
                "audio_emb",
                "image_embeds",
                "landmarks",
                "masks",
                "gt",
            ]:
                c_out[k] = c[k]
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]

        # The rest are the conditions removed from the full condition
        for cond_name in self.condition_names:
            if not isinstance(cond_name, list):
                cond_name = [cond_name]
            for k in c:
                if k in [
                    "vector",
                    "crossattn",
                    "concat",
                    "audio_emb",
                    "image_embeds",
                    "landmarks",
                    "masks",
                    "gt",
                ]:
                    c_out[k] = torch.cat(
                        (c_out[k], uc[k] if k in cond_name else c[k]), 0
                    )

        return (
            torch.cat([x] * (self.n_conditions + 1)),
            torch.cat([s] * (self.n_conditions + 1)),
            c_out,
        )


class AudioRefMultiCondGuider(MultipleCondVanilla):
    def __init__(
        self,
        audio_ratio: float = 5.0,
        ref_ratio: float = 3.0,
        use_normalized: bool = False,
        momentum: float = -0.75,
        eta: float = 0.0,
        norm_threshold: float = 2.5,
    ):
        super().__init__(
            scales=[audio_ratio, ref_ratio], condition_names=["audio_emb", "concat"]
        )
        self.audio_ratio = audio_ratio
        self.ref_ratio = ref_ratio
        self.use_normalized = use_normalized
        print(f"Use normalized: {self.use_normalized}")
        self.momentum_buffer = MomentumBuffer(momentum)
        self.eta = eta
        self.norm_threshold = norm_threshold
        self.momentum_buffer_audio = MomentumBuffer(momentum)
        self.momentum_buffer_ref = MomentumBuffer(momentum)

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        e_uc, e_ref, c_audio_ref = x.chunk(3)

        if self.use_normalized:
            # Normalized guidance version
            # Compute diff for audio guidance
            diff_audio = c_audio_ref - e_uc
            if self.momentum_buffer_audio is not None:
                self.momentum_buffer_audio.update(diff_audio)
                diff_audio = self.momentum_buffer_audio.running_average
            if self.norm_threshold > 0:
                ones = torch.ones_like(diff_audio)
                diff_norm = diff_audio.norm(p=2, dim=[-1, -2, -3], keepdim=True)
                scale_factor = torch.minimum(ones, self.norm_threshold / diff_norm)
                diff_audio = diff_audio * scale_factor
            diff_audio_parallel, diff_audio_orthogonal = project(
                diff_audio, c_audio_ref
            )
            normalized_update_audio = (
                diff_audio_orthogonal + self.eta * diff_audio_parallel
            )
            guidance_audio = (self.audio_ratio - 1) * normalized_update_audio

            # Compute diff for ref guidance
            diff_ref = e_ref - e_uc
            if self.momentum_buffer_ref is not None:
                self.momentum_buffer_ref.update(diff_ref)
                diff_ref = self.momentum_buffer_ref.running_average
            if self.norm_threshold > 0:
                ones = torch.ones_like(diff_ref)
                diff_norm = diff_ref.norm(p=2, dim=[-1, -2, -3], keepdim=True)
                scale_factor = torch.minimum(ones, self.norm_threshold / diff_norm)
                diff_ref = diff_ref * scale_factor
            diff_ref_parallel, diff_ref_orthogonal = project(diff_ref, e_ref)
            normalized_update_ref = diff_ref_orthogonal + self.eta * diff_ref_parallel
            guidance_ref = (self.ref_ratio - 1) * normalized_update_ref

            e_final = e_uc + guidance_audio + guidance_ref
        else:
            # Original version
            e_final = (
                self.audio_ratio * (c_audio_ref - e_ref)
                + self.ref_ratio * (e_ref - e_uc)
                + e_uc
            )

        return e_final

    def set_scale(self, scale: torch.Tensor):
        self.audio_ratio = float(scale[0])
        self.ref_ratio = float(scale[1])
        print(f"Audio ratio: {self.audio_ratio}, Ref ratio: {self.ref_ratio}")

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        # Prepare inputs for e_base (no audio, no ref concat)
        c_base = {k: v for k, v in c.items()}
        c_base["crossattn"] = uc["crossattn"]
        c_base["concat"] = uc["concat"]  # Remove ref concat

        # Prepare inputs for e_ref (no audio, with ref concat)
        c_audio_ref = {k: v for k, v in c.items()}
        # c_ref["concat"] = uc["concat"]  # Remove ref concat

        # Prepare inputs for e_audio (all conditions)
        c_ref = {k: v for k, v in c.items()}
        c_ref["crossattn"] = uc["crossattn"]

        # Combine all conditions
        for k in c:
            if k in [
                "vector",
                "crossattn",
                "concat",
                "audio_emb",
                "image_embeds",
                "landmarks",
                "masks",
                "gt",
            ]:
                c_out[k] = torch.cat((c_base[k], c_ref[k], c_audio_ref[k]), 0)
            else:
                c_out[k] = c[k]

        return torch.cat([x] * 3), torch.cat([s] * 3), c_out


class IdentityGuider(Guider):
    def __init__(self, *args, **kwargs):
        # self.num_frames = num_frames
        pass

    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        return x

    def prepare_inputs(
        self, x: torch.Tensor, s: float, c: Dict, uc: Dict
    ) -> Tuple[torch.Tensor, float, Dict]:
        c_out = dict()

        for k in c:
            c_out[k] = c[k]

        return x, s, c_out


class LinearPredictionGuider(Guider):
    def __init__(
        self,
        max_scale: float,
        num_frames: int,
        min_scale: float = 1.0,
        additional_cond_keys: Optional[Union[List[str], str]] = None,
        only_first=False,
    ):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.num_frames = num_frames
        self.scale = torch.linspace(min_scale, max_scale, num_frames).unsqueeze(0)

        self.only_first = only_first
        if only_first:
            self.scale = torch.ones_like(self.scale) * max_scale
            self.scale[:, 0] = min_scale

        additional_cond_keys = default(additional_cond_keys, [])
        if isinstance(additional_cond_keys, str):
            additional_cond_keys = [additional_cond_keys]
        self.additional_cond_keys = additional_cond_keys

    def set_scale(self, scale: torch.Tensor):
        self.min_scale = scale
        self.scale = torch.linspace(
            self.min_scale, self.max_scale, self.num_frames
        ).unsqueeze(0)

        if self.only_first:
            self.scale = torch.ones_like(self.scale) * self.max_scale
            self.scale[:, 0] = self.min_scale

        print(self.scale)

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x_u, x_c = x.chunk(2)

        x_u = rearrange(x_u, "(b t) ... -> b t ...", t=self.num_frames)
        x_c = rearrange(x_c, "(b t) ... -> b t ...", t=self.num_frames)
        scale = repeat(self.scale, "1 t -> b t", b=x_u.shape[0])
        scale = append_dims(scale, x_u.ndim).to(x_u.device)
        return rearrange(x_u + scale * (x_c - x_u), "b t ... -> (b t) ...")

    def prepare_inputs(
        self, x: torch.Tensor, s: torch.Tensor, c: dict, uc: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        c_out = dict()

        for k in c:
            if (
                k
                in ["vector", "crossattn", "concat", "audio_emb", "masks", "gt"]
                + self.additional_cond_keys
            ):
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]

        return torch.cat([x] * 2), torch.cat([s] * 2), c_out


class LinearPredictionGuiderPlus(LinearPredictionGuider):
    def __init__(
        self,
        max_scale: float,
        num_frames: int,
        min_scale: float = 1.0,
        additional_cond_keys: Optional[Union[List[str], str]] = None,
    ):
        super().__init__(max_scale, num_frames, min_scale, additional_cond_keys)

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x_u, x_c = x.chunk(2)

        x_u = rearrange(x_u, "(b t) ... -> b t ...", t=self.num_frames)
        x_c = rearrange(x_c, "(b t) ... -> b t ...", t=self.num_frames)
        scale = repeat(self.scale, "1 t -> b t", b=x_u.shape[0])
        scale = append_dims(scale, x_u.ndim).to(x_u.device)
        return rearrange(x_u + scale * (x_c - x_u), "b t ... -> (b t) ..."), x_u


class TrianglePredictionGuider(LinearPredictionGuider):
    def __init__(
        self,
        max_scale: float,
        num_frames: int,
        min_scale: float = 1.0,
        period: float | List[float] = 1.0,
        period_fusing: Literal["mean", "multiply", "max"] = "max",
        additional_cond_keys: Optional[Union[List[str], str]] = None,
    ):
        super().__init__(max_scale, num_frames, min_scale, additional_cond_keys)
        values = torch.linspace(0, 1, num_frames)
        # Constructs a triangle wave
        if isinstance(period, float):
            period = [period]

        scales = []
        for p in period:
            scales.append(self.triangle_wave(values, p))

        if period_fusing == "mean":
            scale = sum(scales) / len(period)
        elif period_fusing == "multiply":
            scale = torch.prod(torch.stack(scales), dim=0)
        elif period_fusing == "max":
            scale = torch.max(torch.stack(scales), dim=0).values
        self.scale = (scale * (max_scale - min_scale) + min_scale).unsqueeze(0)

    def triangle_wave(self, values: torch.Tensor, period) -> torch.Tensor:
        return 2 * (values / period - torch.floor(values / period + 0.5)).abs()
