import torch
import torch.nn as nn
from packaging import version
from einops import repeat, rearrange
from diffusers.utils import _get_model_file
from diffusers.models.modeling_utils import load_state_dict
from ...modules.diffusionmodules.augment_pipeline import AugmentPipe
from ...modules.encoders.modules import ConcatTimestepEmbedderND
from ...util import append_dims


OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"


class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    def __init__(
        self,
        diffusion_model,
        compile_model: bool = False,
        ada_aug_percent=0.0,
        fix_image_leak=False,
        add_embeddings=False,
        im_size=[64, 64],
        n_channels=4,
    ):
        super().__init__(diffusion_model, compile_model)
        self.fix_image_leak = fix_image_leak
        if fix_image_leak:
            self.beta_m = 15
            self.a = 5
            self.noise_encoder = ConcatTimestepEmbedderND(256)

        self.augment_pipe = None
        if ada_aug_percent > 0.0:
            augment_kwargs = dict(
                xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1
            )
            self.augment_pipe = AugmentPipe(ada_aug_percent, **augment_kwargs)

        self.add_embeddings = add_embeddings
        if add_embeddings:
            self.learned_mask = nn.Parameter(
                torch.zeros(n_channels, im_size[0], im_size[1])
            )

    def get_noised_input(
        self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        noised_input = input + noise * sigmas_bc
        return noised_input

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        cond_cat = c.get("concat", torch.Tensor([]).type_as(x))

        if len(cond_cat.shape) and cond_cat.shape[0]:
            T = x.shape[0] // cond_cat.shape[0]
        if self.fix_image_leak:
            noise_aug_strength = get_sigma_s(
                rearrange(t, "(b t) ... -> b t ...", b=T)[: cond_cat.shape[0], 0] / 700,
                self.a,
                self.beta_m,
            )
            noise_aug = append_dims(noise_aug_strength, 4).to(x.device)
            noise = torch.randn_like(noise_aug)
            cond_cat = self.get_noised_input(noise_aug, noise, cond_cat)
            noise_emb = self.noise_encoder(noise_aug_strength).to(x.device)
            c["vector"] = (
                noise_emb
                if "vector" not in c
                else torch.cat([c["vector"], noise_emb], dim=1)
            )
            # c["vector"] = noise_emb

        if (
            len(cond_cat.shape)
            and cond_cat.shape[0]
            and x.shape[0] != cond_cat.shape[0]
        ):
            cond_cat = repeat(cond_cat, "b c h w -> b c t h w", t=T)
            cond_cat = rearrange(cond_cat, "b c t h w -> (b t) c h w")
        x = torch.cat((x, cond_cat), dim=1)

        if self.add_embeddings:
            learned_mask = repeat(
                self.learned_mask.to(x.device), "c h w -> b c h w", b=cond_cat.shape[0]
            )
            x = torch.cat((x, learned_mask), dim=1)

        if self.augment_pipe is not None:
            x, labels = self.augment_pipe(x)
        else:
            labels = torch.zeros(x.shape[0], 9, device=x.device)

        # if c.get("reference", None) is not None:
        #     c["crossattn"] = c["reference"]

        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            reference_context=c.get("reference", None),
            y=c.get("vector", None),
            audio_emb=c.get("audio_emb", None),
            landmarks=c.get("landmarks", None),
            aug_labels=labels,
            **kwargs,
        )


class StabilityWrapper(IdentityWrapper):
    def __init__(
        self,
        diffusion_model,
        compile_model: bool = False,
        use_ipadapter: bool = False,
        ipadapter_model: str = "ip-adapter_sd15.bin",
        adapter_scale: float = 1.0,
        n_adapters: int = 1,
        skip_text_emb: bool = False,
        # pass_image_emb_to_hidden_states: bool = False,
    ):
        super().__init__(diffusion_model, compile_model)
        self.use_ipadapter = use_ipadapter
        # self.pass_image_emb_to_hidden_states = pass_image_emb_to_hidden_states

        if use_ipadapter:
            model_file = _get_model_file(
                "h94/IP-Adapter",
                weights_name=ipadapter_model,  # ip-adapter_sd15.bin
                # cache_dir="/vol/paramonos2/projects/antoni/.cache",
                subfolder="models",
            )
            state_dict = load_state_dict(model_file)
            state_dict = [load_state_dict(model_file)] * n_adapters
            print(f"Loading IP-Adapter weights from {model_file}")

            diffusion_model.set_ip_adapter_scale(adapter_scale)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        added_cond_kwargs = None
        if self.use_ipadapter:
            added_cond_kwargs = {"image_embeds": c.get("image_embeds", None)}
            landmarks = c.get("landmarks", None)
            if landmarks is not None:
                added_cond_kwargs["image_embeds"] = [
                    added_cond_kwargs["image_embeds"],
                    landmarks,
                ]

        cond_cat = c.get("concat", torch.Tensor([]).type_as(x))
        if len(cond_cat.shape) and cond_cat.shape[0]:
            cond_cat = repeat(
                cond_cat, "b c h w -> b c t h w", t=x.shape[0] // cond_cat.shape[0]
            )
            cond_cat = rearrange(cond_cat, "b c t h w -> (b t) c h w")
            x = torch.cat((x, cond_cat), dim=1)

        return self.diffusion_model(
            x,
            t,
            encoder_hidden_states=c.get("crossattn", None),
            added_cond_kwargs=added_cond_kwargs,
            audio_emb=c.get("audio_emb", None),
            **kwargs,
        )[0]


def logit_normal_sampler(m, s=1, beta_m=15, sample_num=1000000):
    y_samples = torch.randn(sample_num) * s + m
    x_samples = beta_m * (torch.exp(y_samples) / (1 + torch.exp(y_samples)))
    return x_samples


def mu_t(t, a=5, mu_max=1):
    t = t.to("cpu")
    return 2 * mu_max * t**a - mu_max


def get_sigma_s(t, a, beta_m):
    mu = mu_t(t, a=a)
    sigma_s = logit_normal_sampler(m=mu, sample_num=t.shape[0], beta_m=beta_m)
    return sigma_s


class InterpolationWrapper(IdentityWrapper):
    def __init__(
        self,
        diffusion_model,
        compile_model: bool = False,
        im_size=[512, 512],
        n_channels=4,
        starting_mask_method="zeros",
        add_mask=True,
        fix_image_leak=False,
    ):
        super().__init__(diffusion_model, compile_model)
        im_size = [
            x // 8 for x in im_size
        ]  # 8 is the default downscaling factor in the vae model
        if starting_mask_method == "zeros":
            self.learned_mask = nn.Parameter(
                torch.zeros(n_channels, im_size[0], im_size[1])
            )
        elif starting_mask_method == "ones":
            self.learned_mask = nn.Parameter(
                torch.ones(n_channels, im_size[0], im_size[1])
            )
        elif starting_mask_method == "random":
            self.learned_mask = nn.Parameter(
                torch.randn(n_channels, im_size[0], im_size[1])
            )
        elif starting_mask_method == "none":
            self.learned_mask = None
        elif starting_mask_method == "fixed_ones":
            self.learned_mask = torch.ones(n_channels, im_size[0], im_size[1])
        elif starting_mask_method == "fixed_zeros":
            self.learned_mask = torch.zeros(n_channels, im_size[0], im_size[1])
        else:
            raise NotImplementedError(
                f"Unknown stating_mask_method: {starting_mask_method}"
            )

        self.add_mask = add_mask
        self.fix_image_leak = fix_image_leak
        if fix_image_leak:
            self.beta_m = 15
            self.a = 5
            self.noise_encoder = ConcatTimestepEmbedderND(256)

    def get_noised_input(
        self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        noised_input = input + noise * sigmas_bc
        return noised_input

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        cond_cat = c.get("concat", torch.Tensor([]).type_as(x))
        T = x.shape[0] // cond_cat.shape[0]

        if self.fix_image_leak:
            noise_aug_strength = get_sigma_s(
                rearrange(t, "(b t) ... -> b t ...", b=T)[: cond_cat.shape[0], 0] / 700,
                self.a,
                self.beta_m,
            )
            noise_aug = append_dims(noise_aug_strength, 4).to(x.device)
            noise = torch.randn_like(noise_aug)
            cond_cat = self.get_noised_input(noise_aug, noise, cond_cat)
            noise_emb = self.noise_encoder(noise_aug_strength).to(x.device)
            # cond["vector"] = noise_emb if "vector" not in cond else torch.cat([cond["vector"], noise_emb], dim=1)
            c["vector"] = noise_emb

        cond_cat = rearrange(cond_cat, "b (t c) h w -> b c t h w", t=2)

        start, end = cond_cat.chunk(2, dim=2)
        if self.learned_mask is None:
            learned_mask = torch.stack(
                [start.squeeze(2)] * (T // 2 - 1) + [end.squeeze(2)] * (T // 2 - 1),
                dim=2,
            )
        else:
            learned_mask = repeat(
                self.learned_mask.to(x.device), "c h w -> b c h w", b=cond_cat.shape[0]
            )
        ones_mask = torch.ones_like(learned_mask)[:, 0].unsqueeze(1)
        zeros_mask = torch.zeros_like(learned_mask)[:, 0].unsqueeze(1)
        if self.learned_mask is None:
            cond_seq = torch.cat([start] + [learned_mask] + [end], dim=2)
        else:
            cond_seq = torch.stack(
                [start.squeeze(2)] + [learned_mask] * (T - 2) + [end.squeeze(2)], dim=2
            )
        cond_seq = rearrange(cond_seq, "b c t h w -> (b t) c h w")

        x = torch.cat((x, cond_seq), dim=1)
        if self.add_mask:
            mask_seq = torch.stack(
                [ones_mask] + [zeros_mask] * (T - 2) + [ones_mask], dim=2
            )
            mask_seq = rearrange(mask_seq, "b c t h w -> (b t) c h w")
            x = torch.cat((x, mask_seq), dim=1)

        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            audio_emb=c.get("audio_emb", None),
            **kwargs,
        )
