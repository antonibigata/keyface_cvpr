import torch
import whisper
import math
from whisper.audio import log_mel_spectrogram, pad_or_trim
from einops import rearrange


class Whisper(torch.nn.Module):
    def __init__(self, model_size="base", fps=25, merge_type="cat") -> None:
        super().__init__()
        assert model_size in whisper.available_models()
        self.model = whisper.load_model(model_size)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.code_size = 512 * 2 if merge_type == "cat" else 512
        if model_size in ["large-v2", "large-v3"]:
            self.code_size = 1280 * 2 if merge_type == "cat" else 1280

        self.fps = fps
        self.whisper_n_seconds = 30
        self.whisper_n_frames = 1500
        self.feed_as_frames = False
        self.merge_type = merge_type

    def forward(self, x):
        if len(x.shape) == 4:
            return x  # already embedded
        T = x.shape[1]
        x = rearrange(x, "b ... -> b (...)")
        x = pad_or_trim(x)
        x = log_mel_spectrogram(x, n_mels=self.model.dims.n_mels)
        x = self.model.embed_audio(x)[
            :, : math.ceil(((T / self.fps) * self.whisper_n_frames) / self.whisper_n_seconds)
        ]

        x = rearrange(x[:, : 2 * T], "b (f d) c -> b f d c", d=2)

        if self.merge_type == "cat":
            if x.dim() == 3:
                return rearrange(x, "b d c -> b (d c)")
            return rearrange(x, "b f d c -> b f (d c)")
        elif self.merge_type == "sum":
            return x.sum(dim=-2)
        elif self.merge_type == "mean":
            return x.mean(dim=-2)
        elif self.merge_type == "None":
            return x
        else:
            raise NotImplementedError
