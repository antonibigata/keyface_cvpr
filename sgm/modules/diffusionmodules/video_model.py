from functools import partial
from typing import List, Optional, Union

from einops import rearrange, repeat
import copy

from ...modules.diffusionmodules.openaimodel import *
from ...modules.video_attention import SpatialVideoTransformer
from ...modules.diffusionmodules.model import FaceLocator
from ...util import default
from .util import AlphaBlender


class VideoResBlock(ResBlock):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        video_kernel_size: Union[int, List[int]] = 3,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
        skip_time: bool = False,
    ):
        super().__init__(
            channels,
            emb_channels,
            dropout,
            out_channels=out_channels,
            use_conv=use_conv,
            use_scale_shift_norm=use_scale_shift_norm,
            dims=dims,
            use_checkpoint=use_checkpoint,
            up=up,
            down=down,
        )

        self.time_stack = ResBlock(
            default(out_channels, channels),
            emb_channels,
            dropout=dropout,
            dims=3,
            out_channels=default(out_channels, channels),
            use_scale_shift_norm=False,
            use_conv=False,
            up=False,
            down=False,
            kernel_size=video_kernel_size,
            use_checkpoint=use_checkpoint,
            exchange_temb_dims=True,
        )
        self.time_mixer = AlphaBlender(
            alpha=merge_factor,
            merge_strategy=merge_strategy,
            rearrange_pattern="b t -> b 1 t 1 1",
        )
        self.skip_time = skip_time

    def forward(
        self,
        x: th.Tensor,
        emb: th.Tensor,
        num_video_frames: int,
        image_only_indicator: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        x = super().forward(x, emb)

        if self.skip_time:
            return x

        x_mix = rearrange(x, "(b t) c h w -> b c t h w", t=num_video_frames)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=num_video_frames)

        x = self.time_stack(
            x, rearrange(emb, "(b t) ... -> b t ...", t=num_video_frames)
        )
        x = self.time_mixer(
            x_spatial=x_mix, x_temporal=x, image_only_indicator=image_only_indicator
        )
        x = rearrange(x, "b c t h w -> (b t) c h w")
        return x


class VideoUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: int,
        dropout: float = 0.0,
        channel_mult: List[int] = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: Optional[int] = None,
        use_checkpoint: bool = False,
        num_heads: int = -1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        transformer_depth: Union[List[int], int] = 1,
        transformer_depth_middle: Optional[int] = None,
        context_dim: Optional[int] = None,
        time_downup: bool = False,
        time_context_dim: Optional[int] = None,
        extra_ff_mix_layer: bool = False,
        use_spatial_context: bool = False,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        spatial_transformer_attn_type: str = "softmax",
        video_kernel_size: Union[int, List[int]] = 3,
        use_linear_in_transformer: bool = False,
        adm_in_channels: Optional[int] = None,
        disable_temporal_crossattention: bool = False,
        max_ddpm_temb_period: int = 10000,
        fine_tuning_method: str = None,
        unfreeze_blocks: Optional[List[str]] = None,
        adapter_kwargs: Optional[dict] = {},
        audio_cond_method: str = None,
        audio_dim: Optional[int] = 0,
        additional_audio_frames: Optional[int] = 0,
        skip_time: bool = False,
        use_ada_aug: bool = False,
        encode_landmarks: bool = False,
        reference_to: str = None,
    ):
        super().__init__()
        assert context_dim is not None

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1

        if num_head_channels == -1:
            assert num_heads != -1

        self.additional_audio_frames = additional_audio_frames
        audio_multiplier = additional_audio_frames * 2 + 1
        audio_dim = audio_dim * audio_multiplier

        self.audio_is_context = "both" in audio_cond_method

        if "both" == audio_cond_method:
            audio_cond_method = "to_time_emb_image"
        elif "both_keyframes" == audio_cond_method:
            audio_cond_method = "to_time_emb"

        if "to_time_emb" in audio_cond_method:
            adm_in_channels += audio_dim

        print(adm_in_channels, audio_dim, audio_cond_method)

        self.adapter = None
        self.audio_cond_method = audio_cond_method

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        transformer_depth_middle = default(
            transformer_depth_middle, transformer_depth[-1]
        )

        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.use_ada_aug = use_ada_aug
        if use_ada_aug:
            self.map_aug = linear(9, time_embed_dim)

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ),
                )

            elif self.num_classes == "sequential":
                if adm_in_channels > 0:
                    assert adm_in_channels is not None
                    self.label_emb = nn.Sequential(
                        nn.Sequential(
                            linear(adm_in_channels, time_embed_dim),
                            nn.SiLU(),
                            linear(time_embed_dim, time_embed_dim),
                        )
                    )
                else:
                    # Disabling the label embedding
                    self.num_classes = None
            else:
                raise ValueError()

        self.encode_landmarks = encode_landmarks
        if encode_landmarks:
            self.face_locator = FaceLocator(
                320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)
            )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        def get_attention_layer(
            ch,
            num_heads,
            dim_head,
            depth=1,
            context_dim=None,
            use_checkpoint=False,
            disabled_sa=False,
            audio_context_dim=None,
        ):
            return SpatialVideoTransformer(
                ch,
                num_heads,
                dim_head,
                depth=depth,
                context_dim=context_dim,
                audio_context_dim=audio_context_dim,
                time_context_dim=time_context_dim,
                dropout=dropout,
                ff_in=extra_ff_mix_layer,
                use_spatial_context=use_spatial_context,
                merge_strategy=merge_strategy,
                merge_factor=merge_factor,
                checkpoint=use_checkpoint,
                use_linear=use_linear_in_transformer,
                attn_mode=spatial_transformer_attn_type,
                disable_self_attn=disabled_sa,
                disable_temporal_crossattention=disable_temporal_crossattention,
                max_time_embed_period=max_ddpm_temb_period,
                skip_time=skip_time,
                reference_to=reference_to,
            )

        def get_resblock(
            merge_factor,
            merge_strategy,
            video_kernel_size,
            ch,
            time_embed_dim,
            dropout,
            out_ch,
            dims,
            use_checkpoint,
            use_scale_shift_norm,
            down=False,
            up=False,
        ):
            return VideoResBlock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                out_channels=out_ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                down=down,
                up=up,
                skip_time=skip_time,
            )

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_ch=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    layers.append(
                        get_attention_layer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            audio_context_dim=audio_dim
                            if "cross_attention" in audio_cond_method
                            else None,
                            use_checkpoint=use_checkpoint,
                            disabled_sa=False,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_ch=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            third_down=time_downup,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)

                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        self.middle_block = TimestepEmbedSequential(
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                time_embed_dim=time_embed_dim,
                out_ch=None,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            get_attention_layer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth_middle,
                context_dim=context_dim,
                audio_context_dim=audio_dim
                if "new_cross_attention" in audio_cond_method
                else None,
                use_checkpoint=use_checkpoint,
            ),
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                out_ch=None,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()

                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch + ich,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_ch=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    layers.append(
                        get_attention_layer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            audio_context_dim=audio_dim
                            if "new_cross_attention" == audio_cond_method
                            else None,
                            use_checkpoint=use_checkpoint,
                            disabled_sa=False,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    ds //= 2
                    layers.append(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_ch=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            third_up=time_downup,
                        )
                    )

                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        if fine_tuning_method is not None:
            # Freeze everything except the adapter
            for param in self.parameters():
                param.requires_grad = False
            if self.adapter is not None:
                for param in self.adapter.parameters():
                    param.requires_grad = True
            if len(unfreeze_blocks):
                if "input" in unfreeze_blocks:
                    for param in self.input_blocks[0].parameters():
                        param.requires_grad = True
                    # break  # only unfreeze the first input block
                if "label_emb" in unfreeze_blocks:
                    for param in self.label_emb.parameters():
                        param.requires_grad = True

    def get_skip_attention_at(
        self,
        skip_attention_at: List[int],
        curr_layer: int,
        batch_size: int,
        num_video_frames: int,
    ):
        if skip_attention_at is None:
            return None

        skip_attention = th.zeros(len(skip_attention_at), 1, dtype=th.bool)

        for i, layer in enumerate(skip_attention_at):
            skip_attention[i] = layer == curr_layer
        skip_attention = repeat(
            skip_attention, "b ... -> (b t) ...", t=num_video_frames
        )
        assert skip_attention.shape[0] == batch_size, (
            f"{skip_attention.shape[0]} != {batch_size}"
        )
        return skip_attention

    def forward(
        self,
        x: th.Tensor,
        timesteps: th.Tensor,
        context: Optional[th.Tensor] = None,
        reference_context: Optional[th.Tensor] = None,
        y: Optional[th.Tensor] = None,
        audio_emb: Optional[th.Tensor] = None,
        landmarks: Optional[th.Tensor] = None,
        aug_labels: Optional[th.Tensor] = None,
        time_context: Optional[th.Tensor] = None,
        num_video_frames: Optional[int] = 1,
        image_only_indicator: Optional[th.Tensor] = None,
        skip_spatial_attention_at: Optional[List[int]] = None,
        skip_temporal_attention_at: Optional[List[int]] = None,
    ):
        if self.audio_is_context:
            assert audio_emb is None
            audio_emb = context.clone()

        curr_context_idx = 0
        num_video_frames = (
            num_video_frames
            if isinstance(num_video_frames, int)
            else num_video_frames[0]
        )
        if reference_context is not None:
            copy_context = copy.deepcopy(reference_context)
            mid = copy_context.pop(-1)
            copy_context.insert((len(copy_context) // 2) - 1, mid)
            reference_context = copy_context
            curr_context_idx = 0
            if num_video_frames > 1:
                reference_context = [
                    repeat(ref_context, "b h w -> (b t) h w", t=num_video_frames)
                    for ref_context in reference_context
                ]

        or_batch_size = x.shape[0] // num_video_frames
        if (
            image_only_indicator is not None
            and image_only_indicator.shape[0] != or_batch_size
        ):
            # TODO: fix this
            image_only_indicator = repeat(
                image_only_indicator, "b ... -> (b t) ...", t=2
            )

        if context is not None and x.shape[0] != context.shape[0]:
            context = repeat(context, "b ... -> b t ...", t=num_video_frames)
            context = rearrange(context, "b t ... -> (b t) ...", t=num_video_frames)

        if "cross_attention" in self.audio_cond_method:
            assert audio_emb is not None
            if audio_emb.ndim == 4:
                audio_emb = rearrange(audio_emb, "b t d c -> b (t d) c")

        #     context = th.cat([context, audio_emb], dim=1)

        if self.audio_cond_method == "cross_time":
            assert audio_emb is not None
            time_context = audio_emb

        if y is not None and y.shape[0] != x.shape[0]:
            y = repeat(y, "b ... -> b t ...", t=num_video_frames)
            y = rearrange(y, "b t ... -> (b t) ...", t=num_video_frames)

        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)

        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y is not None or "to_time_emb" in self.audio_cond_method

            if self.audio_cond_method == "to_time_emb":
                assert audio_emb is not None
                audio_emb = rearrange(audio_emb, "b t c -> (b t) c")
                if y is not None:
                    y = th.cat([y, audio_emb], dim=1)
                else:
                    y = audio_emb
            elif self.audio_cond_method == "to_time_emb_image":
                assert audio_emb is not None

                audio_emb = rearrange(audio_emb, "b t c -> b (t c)")
                if y is not None:
                    y = th.cat([y, audio_emb], dim=1)
                else:
                    y = audio_emb
            assert y.shape[0] == x.shape[0], (
                f"{y.shape} != {x.shape} and audio_emb.shape: {audio_emb.shape}"
            )

            emb = emb + self.label_emb(y)

        if self.use_ada_aug:
            assert aug_labels is not None, (
                "must provide aug_labels if use_ada_aug is True"
            )
            emb = emb + self.map_aug(aug_labels)

        h = x

        if self.encode_landmarks:
            landmarks_emb = self.face_locator(landmarks)
            landmarks_emb = rearrange(landmarks_emb, "b c t h w -> (b t) c h w")
            # print("landmarks_emb:", landmarks_emb.shape)
        for i, module in enumerate(self.input_blocks):
            # print(image_only_indicator.shape, num_video_frames, h.shape)
            if i == 1 and self.encode_landmarks:
                h = h + landmarks_emb
            # print("h.shape:", h.shape, i)
            skip_spatial_attention = self.get_skip_attention_at(
                skip_spatial_attention_at,
                curr_context_idx,
                x.shape[0],
                num_video_frames,
            )
            skip_temporal_attention = self.get_skip_attention_at(
                skip_temporal_attention_at,
                curr_context_idx,
                x.shape[0],
                num_video_frames,
            )
            h, is_attention = module(
                h,
                emb,
                context=context,
                reference_context=reference_context[curr_context_idx]
                if reference_context is not None
                else None,
                audio_context=audio_emb
                if "cross_attention" in self.audio_cond_method
                else None,
                image_only_indicator=image_only_indicator,
                time_context=time_context,
                num_video_frames=num_video_frames,
                skip_spatial_attention=skip_spatial_attention,
                skip_temporal_attention=skip_temporal_attention,
            )
            if is_attention:
                curr_context_idx = (
                    None if curr_context_idx is None else curr_context_idx + 1
                )
            hs.append(h)
        skip_spatial_attention = self.get_skip_attention_at(
            skip_spatial_attention_at, curr_context_idx, x.shape[0], num_video_frames
        )
        skip_temporal_attention = self.get_skip_attention_at(
            skip_temporal_attention_at, curr_context_idx, x.shape[0], num_video_frames
        )
        h, is_attention = self.middle_block(
            h,
            emb,
            context=context,
            reference_context=reference_context[curr_context_idx]
            if reference_context is not None
            else None,
            audio_context=audio_emb
            if "cross_attention" in self.audio_cond_method
            else None,
            image_only_indicator=image_only_indicator,
            time_context=time_context,
            num_video_frames=num_video_frames,
            skip_spatial_attention=skip_spatial_attention,
            skip_temporal_attention=skip_temporal_attention,
        )
        curr_context_idx = None if curr_context_idx is None else curr_context_idx + 1
        for i, module in enumerate(self.output_blocks):
            skip_x = hs.pop()
            if self.adapter is not None:
                skip_x = self.adapter[i](
                    skip_x, n_frames=num_video_frames, condition=audio_emb
                )
            h = th.cat([h, skip_x], dim=1)
            skip_spatial_attention = self.get_skip_attention_at(
                skip_spatial_attention_at,
                curr_context_idx,
                x.shape[0],
                num_video_frames,
            )
            skip_temporal_attention = self.get_skip_attention_at(
                skip_temporal_attention_at,
                curr_context_idx,
                x.shape[0],
                num_video_frames,
            )
            h, is_attention = module(
                h,
                emb,
                context=context,
                reference_context=reference_context[curr_context_idx]
                if reference_context is not None
                else None,
                audio_context=audio_emb
                if "cross_attention" in self.audio_cond_method
                else None,
                image_only_indicator=image_only_indicator,
                time_context=time_context,
                num_video_frames=num_video_frames,
                skip_spatial_attention=skip_spatial_attention,
                skip_temporal_attention=skip_temporal_attention,
            )
            if is_attention:
                curr_context_idx = (
                    None if curr_context_idx is None else curr_context_idx + 1
                )
        # h = h.type(x.dtype)
        return self.out(h)
