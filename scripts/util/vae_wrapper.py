import os
import torch
import torch.nn as nn
from einops import rearrange
from diffusers import AutoencoderKL, AutoencoderKLTemporalDecoder, StableDiffusionPipeline


def load_stable_model(model_path):
    vae_model = StableDiffusionPipeline.from_pretrained(model_path)
    vae_model.set_use_memory_efficient_attention_xformers(True)
    return vae_model.vae


class VaeWrapper(nn.Module):
    def __init__(self, latent_type, max_chunk_decode=16, variant="fp16"):
        super().__init__()
        self.vae_model = self.get_vae(latent_type, variant)
        # self.latent_scale = latent_scale
        self.latent_type = latent_type
        self.max_chunk_decode = max_chunk_decode

    def get_vae(self, latent_type, variant="fp16"):
        if latent_type == "stable":
            vae_model = load_stable_model("stabilityai/stable-diffusion-x4-upscaler")
            vae_model.enable_slicing()
            vae_model.set_use_memory_efficient_attention_xformers(True)
            self.down_factor = 4
        elif latent_type == "video":
            vae_model = AutoencoderKLTemporalDecoder.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid",
                subfolder="vae",
                torch_dtype=torch.float16 if variant == "fp16" else torch.float32,
                variant="fp16" if variant == "fp16" else None,
            )
            vae_model.set_use_memory_efficient_attention_xformers(True)
            self.down_factor = 8
        elif latent_type == "refiner":
            vae_model = AutoencoderKL.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0", subfolder="vae", revision=None
            )
            vae_model.enable_slicing()
            vae_model.set_use_memory_efficient_attention_xformers(True)
            self.down_factor = 8
        elif latent_type == "ldm":
            assert False, "Not implemented"
            vae_model = get_ldm_model(
                os.getcwd() + "/src/models/components/autoencoders/latent_diffusion/models/ldm/ffhq256/model.ckpt",
                os.getcwd() + "/src/models/components/autoencoders/latent_diffusion/models/ldm/ffhq256/config.yaml",
            )
            self.down_factor = 4
        vae_model.eval()
        vae_model.requires_grad_(False)
        vae_model.cuda()

        vae_model = torch.compile(vae_model)
        return vae_model

    # def accelerate_model(self, example_shape):
    #     self.vae_model = torch.jit.trace(self.vae_model, torch.randn(example_shape).cuda())
    #     self.vae_model = torch.compile(self.vae_model)
    #     self.is_accelerated = True
    def disable_slicing(self):
        self.vae_model.disable_slicing()

    @torch.no_grad()
    def encode_video(self, video):
        """
        video: (B, C, T, H, W)
        """
        is_video = False
        if len(video.shape) == 5:
            is_video = True
            T = video.shape[2]
            video = rearrange(video, "b c t h w -> (b t) c h w")
        or_dtype = video.dtype
        # if not self.is_accelerated:
        #     self.accelerate_model(video.shape)
        if self.latent_type in ["stable", "refiner", "video"]:
            encoded_video = (
                self.vae_model.encode(video.to(dtype=self.vae_model.dtype)).latent_dist.sample().to(dtype=or_dtype)
                * self.vae_model.config.scaling_factor
            )
        elif self.latent_type == "ldm":
            encoded_video = self.vae_model.encode_first_stage(video) * 0.18215
        if not is_video:
            return encoded_video
        return rearrange(encoded_video, "(b t) c h w -> b c t h w", t=T)

    @torch.no_grad()
    def decode_video(self, encoded_video):
        """
        encoded_video: (B, C, T, H, W)
        """
        is_video = False
        B, T = encoded_video.shape[0], 1
        if len(encoded_video.shape) == 5:
            is_video = True
            T = encoded_video.shape[2]
            encoded_video = rearrange(encoded_video, "b c t h w -> (b t) c h w")
        decoded_full = []
        or_dtype = encoded_video.dtype

        for i in range(0, T * B, self.max_chunk_decode):  # Slow but no memory issues
            if self.latent_type in ["stable", "refiner"]:
                decoded_full.append(
                    self.vae_model.decode(
                        (1 / self.vae_model.config.scaling_factor) * encoded_video[i : i + self.max_chunk_decode]
                    ).sample
                )
            elif self.latent_type == "video":
                chunk = encoded_video[i : i + self.max_chunk_decode].to(dtype=self.vae_model.dtype)
                num_frames_in = chunk.shape[0]
                decode_kwargs = {}
                decode_kwargs["num_frames"] = num_frames_in
                decoded_full.append(
                    self.vae_model.decode(1 / self.vae_model.config.scaling_factor * chunk, **decode_kwargs).sample.to(
                        or_dtype
                    )
                )
            elif self.latent_type == "ldm":
                decoded_full.append(
                    self.vae_model.decode_first_stage(1 / 0.18215 * encoded_video[i : i + self.max_chunk_decode])
                )
        decoded_video = torch.cat(decoded_full, dim=0)
        if not is_video:
            return decoded_video.clamp(-1.0, 1.0)
        return rearrange(decoded_video, "(b t) c h w -> b c t h w", t=T).clamp(-1.0, 1.0)
