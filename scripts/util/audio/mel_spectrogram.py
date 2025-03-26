import torch
import torch.nn as nn
import torchaudio
from einops import rearrange


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate=16_000,
        n_fft=1024,
        win_length=1024,
        hop_length=160,
        f_min=0,
        f_max=8_000,
        n_mels=80,
        normalized=True,
        center=True,
    ):
        super().__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            normalized=normalized,
            center=center,
        )
        self.code_size = 4 * n_mels
        self.n_mels = n_mels

    @torch.no_grad()
    def forward(self, x):
        x = rearrange(x, "b ... -> b (...)")
        spec = self.transform(x)[:, :, :-1]
        spec = rearrange(spec, "b c (f d) -> b f d c", d=4)
        return spec


if __name__ == "__main__":
    func = MelSpectrogram(n_mels=128).cuda()
    linear = nn.Linear(4, 128).cuda()
    n_frames = 16
    wav = torch.ones(10, n_frames, 640).cuda()
    spec = func(wav)
    # spec = linear(spec.transpose(-1, -2)).transpose(-1, -2)
    print(spec.min(), spec.max(), spec.mean(), spec.std())
    assert list(spec.size()) == [10, n_frames, 4, 80], spec.size()
