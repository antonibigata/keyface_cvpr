import torch
from einops import repeat
from ..modules.attention import *
from ..modules.diffusionmodules.util import AlphaBlender, linear, timestep_embedding


class TimeMixSequential(nn.Sequential):
    def forward(self, x, context=None, timesteps=None):
        for layer in self:
            x = layer(x, context, timesteps)

        return x


class VideoTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,
        "softmax-xformers": MemoryEfficientCrossAttention,
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        timesteps=None,
        ff_in=False,
        inner_dim=None,
        attn_mode="softmax",
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        switch_temporal_ca_to_sa=False,
    ):
        super().__init__()

        attn_cls = self.ATTENTION_MODES[attn_mode]

        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim

        assert int(n_heads * d_head) == inner_dim

        self.is_res = inner_dim == dim

        if self.ff_in:
            self.norm_in = nn.LayerNorm(dim)
            self.ff_in = FeedForward(dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff)

        self.timesteps = timesteps
        self.disable_self_attn = disable_self_attn
        if self.disable_self_attn:
            self.attn1 = attn_cls(
                query_dim=inner_dim,
                heads=n_heads,
                dim_head=d_head,
                context_dim=context_dim,
                dropout=dropout,
            )  # is a cross-attention
        else:
            self.attn1 = attn_cls(
                query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout
            )  # is a self-attention

        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff)

        if disable_temporal_crossattention:
            if switch_temporal_ca_to_sa:
                raise ValueError
            else:
                self.attn2 = None
        else:
            self.norm2 = nn.LayerNorm(inner_dim)
            if switch_temporal_ca_to_sa:
                self.attn2 = attn_cls(
                    query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout
                )  # is a self-attention
            else:
                self.attn2 = attn_cls(
                    query_dim=inner_dim,
                    context_dim=context_dim,
                    heads=n_heads,
                    dim_head=d_head,
                    dropout=dropout,
                )  # is self-attn if context is none

        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

        self.checkpoint = checkpoint
        if self.checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")

    def forward(
        self, x: torch.Tensor, context: torch.Tensor = None, timesteps: int = None, skip_attention=False
    ) -> torch.Tensor:
        if self.checkpoint:
            return checkpoint(self._forward, x, context, timesteps, skip_attention)
        else:
            return self._forward(x, context, timesteps=timesteps, skip_attention=skip_attention)

    def _forward(self, x, context=None, timesteps=None, skip_attention=None):
        assert self.timesteps or timesteps
        assert not (self.timesteps and timesteps) or self.timesteps == timesteps
        timesteps = self.timesteps or timesteps
        B, S, C = x.shape
        x = rearrange(x, "(b t) s c -> (b s) t c", t=timesteps)

        if skip_attention is not None:
            skip_attention = repeat(skip_attention[: B // timesteps], "b ... -> (b s) ...", s=S)
        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip

        if self.disable_self_attn:
            x = self.attn1(self.norm1(x), context=context) + x
        else:
            x = self.attn1(self.norm1(x), skip_attention=skip_attention) + x

        if self.attn2 is not None:
            if self.switch_temporal_ca_to_sa:
                x = self.attn2(self.norm2(x), skip_attention=skip_attention) + x
            else:
                x = self.attn2(self.norm2(x), context=context) + x
        x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip

        x = rearrange(x, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps)
        return x

    def get_last_layer(self):
        return self.ff.net[-1].weight


class SpatialVideoTransformer(SpatialTransformer):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        use_linear=False,
        context_dim=None,
        use_spatial_context=False,
        timesteps=None,
        merge_strategy: str = "fixed",
        merge_factor: float = 0.5,
        merge_audio_factor: float = 5.0,  # Almost 0 audio at first
        time_context_dim=None,
        audio_context_dim=None,
        ff_in=False,
        checkpoint=False,
        time_depth=1,
        attn_mode="softmax",
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        max_time_embed_period: int = 10000,
        skip_time=False,
        reference_to=None,
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth,
            dropout=dropout,
            attn_type=attn_mode,
            use_checkpoint=checkpoint,
            context_dim=context_dim,
            use_linear=use_linear,
            disable_self_attn=disable_self_attn,
            reference_to=reference_to,
        )
        self.time_depth = time_depth
        self.depth = depth
        self.max_time_embed_period = max_time_embed_period
        self.skip_time = skip_time

        time_mix_d_head = d_head
        n_time_mix_heads = n_heads

        time_mix_inner_dim = int(time_mix_d_head * n_time_mix_heads)

        inner_dim = n_heads * d_head
        if use_spatial_context:
            time_context_dim = context_dim

        if not self.skip_time:
            self.time_stack = nn.ModuleList(
                [
                    VideoTransformerBlock(
                        inner_dim,
                        n_time_mix_heads,
                        time_mix_d_head,
                        dropout=dropout,
                        context_dim=time_context_dim,
                        timesteps=timesteps,
                        checkpoint=checkpoint,
                        ff_in=ff_in,
                        inner_dim=time_mix_inner_dim,
                        attn_mode=attn_mode,
                        disable_self_attn=disable_self_attn,
                        disable_temporal_crossattention=disable_temporal_crossattention,
                    )
                    for _ in range(self.depth)
                ]
            )
        else:
            self.time_stack = None

        self.audio_stack = None
        if audio_context_dim is not None:
            self.audio_stack = nn.ModuleList(
                [
                    VideoTransformerBlock(
                        inner_dim,
                        n_time_mix_heads,
                        time_mix_d_head,
                        dropout=dropout,
                        context_dim=audio_context_dim,
                        timesteps=timesteps,
                        checkpoint=checkpoint,
                        ff_in=ff_in,
                        inner_dim=time_mix_inner_dim,
                        attn_mode=attn_mode,
                        disable_self_attn=disable_self_attn,
                        disable_temporal_crossattention=disable_temporal_crossattention or self.skip_time,
                    )
                    for _ in range(self.depth)
                ]
            )
            self.audio_mixer = AlphaBlender(alpha=merge_audio_factor, merge_strategy=merge_strategy)

        if self.time_stack is None:
            self.time_stack = [None] * len(self.transformer_blocks)
        assert len(self.time_stack) == len(self.transformer_blocks)

        self.use_spatial_context = use_spatial_context
        self.in_channels = in_channels

        time_embed_dim = self.in_channels * 4
        self.time_pos_embed = nn.Sequential(
            linear(self.in_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, self.in_channels),
        )

        self.time_mixer = AlphaBlender(alpha=merge_factor, merge_strategy=merge_strategy)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        reference_context: Optional[torch.Tensor] = None,
        time_context: Optional[torch.Tensor] = None,
        audio_context: Optional[torch.Tensor] = None,
        timesteps: Optional[int] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        skip_spatial_attention: bool = False,
        skip_temporal_attention: bool = False,
    ) -> torch.Tensor:
        _, _, h, w = x.shape
        x_in = x
        spatial_context = None
        if exists(context):
            spatial_context = context

        if not isinstance(spatial_context, list):
            spatial_context = [spatial_context]
        if reference_context is not None and not isinstance(reference_context, list):
            reference_context = [reference_context]
        # else:
        #     # spatial_context.reverse()
        #     print([c.shape for c in spatial_context])

        if self.use_spatial_context and not self.skip_time:
            assert (
                isinstance(context, list) or context.ndim == 3
            ), f"n dims of spatial context should be 3 but are {context.ndim}"
            time_context = context
            if not isinstance(context, list):
                time_context_first_timestep = time_context[::timesteps]
                time_context = repeat(time_context_first_timestep, "b ... -> (b n) ...", n=h * w)
        elif time_context is not None and not self.use_spatial_context:
            time_context = repeat(time_context, "b ... -> (b n) ...", n=h * w)
            if time_context.ndim == 2:
                time_context = rearrange(time_context, "b c -> b 1 c")

        if audio_context is not None:
            audio_context = repeat(audio_context, "b ... -> (b n) ...", n=h * w)
            if audio_context.ndim == 2:
                audio_context = rearrange(audio_context, "b c -> b 1 c")

        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.use_linear:
            x = self.proj_in(x)

        if not self.skip_time:
            num_frames = torch.arange(timesteps, device=x.device, dtype=x.dtype)
            num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
            num_frames = rearrange(num_frames, "b t -> (b t)")
            t_emb = timestep_embedding(
                num_frames,
                self.in_channels,
                repeat_only=False,
                max_period=self.max_time_embed_period,
            )
            emb = self.time_pos_embed(t_emb)
            emb = emb[:, None, :]

        for it_, (block, mix_block) in enumerate(zip(self.transformer_blocks, self.time_stack)):
            if it_ > 0 and len(spatial_context) == 1:
                it_ = 0  # use same context for each block

            x = block(
                x,
                context=spatial_context[it_],
                reference_context=reference_context[it_] if reference_context is not None else None,
                skip_attention=skip_spatial_attention,
            )

            if not self.skip_time:
                x_mix = x
                x_mix = x_mix + emb

                x_mix = mix_block(
                    x_mix,
                    context=time_context,
                    timesteps=timesteps,
                    skip_attention=skip_temporal_attention,
                )
                x = self.time_mixer(
                    x_spatial=x,
                    x_temporal=x_mix,
                    image_only_indicator=image_only_indicator,
                )

            if self.audio_stack is not None:
                audio_mix_block = self.audio_stack[it_]
                x_audio = x
                # x_audio = x_audio + emb
                x_audio = audio_mix_block(x_audio, context=audio_context, timesteps=timesteps)
                x = self.audio_mixer(x, x_audio, image_only_indicator=image_only_indicator)

        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        return out
