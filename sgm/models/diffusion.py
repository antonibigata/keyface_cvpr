import os
import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union
import re
import pytorch_lightning as pl
import torch
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0

from ..modules import UNCONDITIONAL_CONFIG
from ..modules.autoencoding.temporal_ae import VideoDecoder
from ..modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from ..modules.ema import LitEma
from ..util import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    log_txt_as_img,
)


class DiffusionEngine(pl.LightningModule):
    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str, Dict, ListConfig, OmegaConf] = None,
        ckpt_path: Union[None, str] = None,
        remove_keys_from_weights: Union[None, List, Tuple] = None,
        pattern_to_remove: Union[None, str] = None,
        remove_keys_from_unet_weights: Union[None, List, Tuple] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        input_key: str = "jpg",
        log_keys: Union[List, None] = None,
        no_log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        en_and_decode_n_samples_a_time: Optional[int] = None,
        only_train_ipadapter: Optional[bool] = False,
        to_unfreeze: Optional[List[str]] = [],
        to_freeze: Optional[List[str]] = [],
        separate_unet_ckpt: Optional[str] = None,
        use_thunder: Optional[bool] = False,
        is_dubbing: Optional[bool] = False,
        bad_model_path: Optional[str] = None,
        bad_model_config: Optional[Dict] = None,
    ):
        super().__init__()

        # self.automatic_optimization = False
        self.log_keys = log_keys
        self.no_log_keys = no_log_keys
        self.input_key = input_key
        self.is_dubbing = is_dubbing
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        self.model = self.initialize_network(
            network_config, network_wrapper, compile_model=compile_model
        )

        self.denoiser = instantiate_from_config(denoiser_config)

        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.is_guided = True
        if (
            self.sampler
            and "IdentityGuider" in sampler_config["params"]["guider_config"]["target"]
        ):
            self.is_guided = False
        if self.sampler is not None:
            config_guider = sampler_config["params"]["guider_config"]
            sampler_config["params"]["guider_config"] = None
            self.sampler_no_guidance = instantiate_from_config(sampler_config)
            sampler_config["params"]["guider_config"] = config_guider
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        self.scheduler_config = scheduler_config
        self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log

        if ckpt_path is not None:
            self.init_from_ckpt(
                ckpt_path,
                remove_keys_from_weights=remove_keys_from_weights,
                pattern_to_remove=pattern_to_remove,
            )
            if separate_unet_ckpt is not None:
                sd = torch.load(separate_unet_ckpt)["state_dict"]
                if remove_keys_from_unet_weights is not None:
                    for k in list(sd.keys()):
                        for remove_key in remove_keys_from_unet_weights:
                            if remove_key in k:
                                del sd[k]
                self.model.diffusion_model.load_state_dict(sd, strict=False)

        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        print(
            "Using",
            self.en_and_decode_n_samples_a_time,
            "samples at a time for encoding and decoding",
        )

        if to_freeze:
            for name, p in self.model.diffusion_model.named_parameters():
                for layer in to_freeze:
                    if layer[0] == "!":
                        if layer[1:] not in name:
                            # print("Freezing", name)
                            p.requires_grad = False
                    else:
                        if layer in name:
                            # print("Freezing", name)
                            p.requires_grad = False
                # if "time_" in name:
                #     print("Freezing", name)
                #     p.requires_grad = False

        if only_train_ipadapter:
            # Freeze the model
            for p in self.model.parameters():
                p.requires_grad = False
            # Unfreeze the adapter projection layer
            for p in self.model.diffusion_model.encoder_hid_proj.parameters():
                p.requires_grad = True
            # Unfreeze the cross-attention layer
            for att_layer in self.model.diffusion_model.attn_processors.values():
                if isinstance(att_layer, IPAdapterAttnProcessor2_0):
                    for p in att_layer.parameters():
                        p.requires_grad = True

            # for name, p in self.named_parameters():
            #     if p.requires_grad:
            #         print(name)

        if to_unfreeze:
            for name in to_unfreeze:
                for p in getattr(self.model.diffusion_model, name).parameters():
                    p.requires_grad = True

        if use_thunder:
            import thunder

            self.model.diffusion_model = thunder.jit(self.model.diffusion_model)

        if "Karras" in denoiser_config.target:
            assert bad_model_path is not None, (
                "bad_model_path must be provided for KarrasGuidanceDenoiser"
            )
            karras_config = default(bad_model_config, network_config)
            bad_model = self.initialize_network(
                karras_config, network_wrapper, compile_model=compile_model
            )
            state_dict = self.load_bad_model_weights(bad_model_path)
            bad_model.load_state_dict(state_dict)
            self.denoiser.set_bad_network(bad_model)

    def load_bad_model_weights(self, path: str) -> None:
        print(f"Restoring bad model from {path}")
        state_dict = torch.load(path, map_location="cpu")
        new_dict = {}
        for k, v in state_dict["module"].items():
            if "learned_mask" in k:
                new_dict[k.replace("_forward_module.", "").replace("model.", "")] = v
            if "diffusion_model" in k:
                new_dict["diffusion_model" + k.split("diffusion_model")[1]] = v
        return new_dict

    def initialize_network(self, network_config, network_wrapper, compile_model=False):
        model = instantiate_from_config(network_config)
        if isinstance(network_wrapper, str) or network_wrapper is None:
            model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
                model, compile_model=compile_model
            )
        else:
            target = network_wrapper["target"]
            params = network_wrapper.get("params", dict())
            model = get_obj_from_str(target)(
                model, compile_model=compile_model, **params
            )
        return model

    def init_from_ckpt(
        self,
        path: str,
        remove_keys_from_weights: Optional[Union[List, Tuple]] = None,
        pattern_to_remove: str = None,
    ) -> None:
        print(f"Restoring from {path}")
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("pt"):
            sd = torch.load(path, map_location="cpu")["module"]
            # Remove leading _forward_module from keys
            sd = {k.replace("_forward_module.", ""): v for k, v in sd.items()}
        elif path.endswith("bin"):
            sd = torch.load(path, map_location="cpu")
            # Remove leading _forward_module from keys
            sd = {k.replace("_forward_module.", ""): v for k, v in sd.items()}
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            raise NotImplementedError

        print(f"Loaded state dict from {path} with {len(sd)} keys")

        # if remove_keys_from_weights is not None:
        #     for k in list(sd.keys()):
        #         for remove_key in remove_keys_from_weights:
        #             if remove_key in k:
        #                 del sd[k]
        if pattern_to_remove is not None or remove_keys_from_weights is not None:
            sd = self.remove_mismatched_keys(
                sd, pattern_to_remove, remove_keys_from_weights
            )

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def remove_mismatched_keys(self, state_dict, pattern=None, additional_keys=None):
        """Remove keys from the state dictionary based on a pattern and a list of additional specific keys."""
        # Find keys that match the pattern
        if pattern is not None:
            mismatched_keys = [key for key in state_dict if re.search(pattern, key)]
        else:
            mismatched_keys = []

        print(f"Removing {len(mismatched_keys)} keys based on pattern {pattern}")
        print(mismatched_keys)

        # Add specific keys to be removed
        if additional_keys:
            mismatched_keys.extend(
                [key for key in additional_keys if key in state_dict]
            )

        # Remove all identified keys
        for key in mismatched_keys:
            if key in state_dict:
                del state_dict[key]
        return state_dict

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model
        if self.input_key == "latents":
            # Remove encoder to save memory
            self.first_stage_model.encoder = None
        torch.cuda.empty_cache()

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key]

    @torch.no_grad()
    def decode_first_stage(self, z):
        is_video = False
        if len(z.shape) == 5:
            is_video = True
            T = z.shape[2]
            z = rearrange(z, "b c t h w -> (b t) c h w")

        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])

        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                if isinstance(self.first_stage_model.decoder, VideoDecoder):
                    kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
                else:
                    kwargs = {}
                out = self.first_stage_model.decode(
                    z[n * n_samples : (n + 1) * n_samples], **kwargs
                )
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        if is_video:
            out = rearrange(out, "(b t) c h w -> b c t h w", t=T)
        torch.cuda.empty_cache()
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        is_video = False
        if len(x.shape) == 5:
            is_video = True
            T = x.shape[2]
            x = rearrange(x, "b c t h w -> (b t) c h w")
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(
                    x[n * n_samples : (n + 1) * n_samples]
                )
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        if is_video:
            z = rearrange(z, "(b t) c h w -> b c t h w", t=T)
        return z

    def forward(self, x, batch):
        loss_dict = self.loss_fn(
            self.model,
            self.denoiser,
            self.conditioner,
            x,
            batch,
            self.first_stage_model,
        )
        # loss_mean = loss.mean()
        for k in loss_dict:
            loss_dict[k] = loss_dict[k].mean()
        # loss_dict = {"loss": loss_mean}
        return loss_dict["loss"], loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        if self.input_key != "latents":
            x = self.encode_first_stage(x)
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        # debugging_message = "Training step"
        # print(f"RANK - {self.trainer.global_rank}: {debugging_message}")

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        # debugging_message = "Training step - log"
        # print(f"RANK - {self.trainer.global_rank}: {debugging_message}")

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

        # # to prevent other processes from moving forward until all processes are in sync
        # self.trainer.strategy.barrier()

        return loss

    # def validation_step(self, batch, batch_idx):
    #     # loss, loss_dict = self.shared_step(batch)
    #     # self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)
    #     self.log(
    #         "global_step",
    #         self.global_step,
    #         prog_bar=True,
    #         logger=True,
    #         on_step=True,
    #         on_epoch=False,
    #     )
    #     return 0

    # def on_train_epoch_start(self, *args, **kwargs):
    #     print(f"RANK - {self.trainer.global_rank}: on_train_epoch_start")

    def on_train_start(self, *args, **kwargs):
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(self.trainer.global_rank)
        # torch.cuda.set_device(self.trainer.global_rank)
        # torch.cuda.empty_cache()
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    # def on_before_batch_transfer(self, batch, dataloader_idx):
    #     print(f"RANK - {self.trainer.global_rank}: on_before_batch_transfer - {dataloader_idx}")
    #     return batch

    # def on_after_batch_transfer(self, batch, dataloader_idx):
    #     print(f"RANK - {self.trainer.global_rank}: on_after_batch_transfer - {dataloader_idx}")
    #     return batch

    def on_train_batch_end(self, *args, **kwargs):
        # print(f"RANK - {self.trainer.global_rank}: on_train_batch_end")
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                params = params + list(embedder.parameters())
        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)

        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **kwargs
        )
        samples = self.sampler(denoiser, randn, cond, uc=uc)

        return samples

    @torch.no_grad()
    def sample_no_guider(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)

        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **kwargs
        )
        samples = self.sampler_no_guidance(denoiser, randn, cond, uc=uc)

        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[-2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                if embedder.input_key in self.no_log_keys:
                    continue
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    elif x.dim() == 4:  # already an image
                        xc = x
                    elif x.dim() == 5:
                        xc = torch.cat([x[:, :, i] for i in range(x.shape[2])], dim=-1)
                    else:
                        print(x.shape, embedder.input_key)
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )

        sampling_kwargs = {}

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        if self.input_key != "latents":
            log["inputs"] = x
            z = self.encode_first_stage(x)
        else:
            z = x
        log["reconstructions"] = self.decode_first_stage(z)
        log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
                )
            samples = self.decode_first_stage(samples)

            log["samples"] = samples

            with self.ema_scope("Plotting"):
                samples = self.sample_no_guider(
                    c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
                )
            samples = self.decode_first_stage(samples)

            log["samples_no_guidance"] = samples
        return log

    @torch.no_grad()
    def log_videos(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        # conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        # if ucg_keys:
        #     assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
        #         "Each defined ucg key for sampling must be in the provided conditioner input keys,"
        #         f"but we have {ucg_keys} vs. {conditioner_input_keys}"
        #     )
        # else:
        #     ucg_keys = conditioner_input_keys
        log = dict()
        batch_uc = {}

        x = self.get_input(batch)
        num_frames = x.shape[2]  # assuming bcthw format

        for key in batch.keys():
            if key not in batch_uc and isinstance(batch[key], torch.Tensor):
                batch_uc[key] = torch.clone(batch[key])

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            batch_uc=batch_uc,
            force_uc_zero_embeddings=ucg_keys
            if ucg_keys is not None
            else [
                "cond_frames",
                "cond_frames_without_noise",
            ],
        )

        # for k in ["crossattn", "concat"]:
        #     uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
        #     uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
        #     c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
        #     c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

        sampling_kwargs = {}

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]

        if self.input_key != "latents":
            log["inputs"] = x
            z = self.encode_first_stage(x)
        else:
            z = x
        log["reconstructions"] = self.decode_first_stage(z)
        log.update(self.log_conditionings(batch, N))

        if c.get("masks", None) is not None:
            # Create a mask reconstruction
            masks = 1 - c["masks"]
            t = masks.shape[2]
            masks = rearrange(masks, "b c t h w -> (b t) c h w")
            target_size = (
                log["reconstructions"].shape[-2],
                log["reconstructions"].shape[-1],
            )
            masks = torch.nn.functional.interpolate(
                masks, size=target_size, mode="nearest"
            )
            masks = rearrange(masks, "(b t) c h w -> b c t h w", t=t)
            log["mask_reconstructions"] = log["reconstructions"] * masks

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))
            elif isinstance(c[k], list):
                for i in range(len(c[k])):
                    c[k][i], uc[k][i] = map(
                        lambda y: y[k][i][:N].to(self.device), (c, uc)
                    )

        if sample:
            n = 2 if self.is_guided else 1
            # if num_frames == 1:
            #     sampling_kwargs["image_only_indicator"] = torch.ones(n, num_frames).to(self.device)
            # else:
            sampling_kwargs["image_only_indicator"] = torch.zeros(n, num_frames).to(
                self.device
            )
            sampling_kwargs["num_video_frames"] = batch["num_video_frames"]

            with self.ema_scope("Plotting"):
                samples = self.sample(
                    c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
                )
            samples = self.decode_first_stage(samples)
            if self.is_dubbing:
                samples[:, :, :, : samples.shape[-2] // 2] = log["reconstructions"][
                    :, :, :, : samples.shape[-2] // 2
                ]
            log["samples"] = samples

            # Without guidance
            # if num_frames == 1:
            #     sampling_kwargs["image_only_indicator"] = torch.ones(1, num_frames).to(self.device)
            # else:
            sampling_kwargs["image_only_indicator"] = torch.zeros(1, num_frames).to(
                self.device
            )
            sampling_kwargs["num_video_frames"] = batch["num_video_frames"]

            with self.ema_scope("Plotting"):
                samples = self.sample_no_guider(
                    c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs
                )
            samples = self.decode_first_stage(samples)
            if self.is_dubbing:
                samples[:, :, :, : samples.shape[-2] // 2] = log["reconstructions"][
                    :, :, :, : samples.shape[-2] // 2
                ]
            log["samples_no_guidance"] = samples

        torch.cuda.empty_cache()
        return log
