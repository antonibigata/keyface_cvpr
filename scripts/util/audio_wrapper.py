import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from scripts.util.audio.Whisper import Whisper
except ModuleNotFoundError:
    print("Whisper not found")

from scripts.util.audio.WavLM import WavLM_wrapper
from scripts.util.audio.BEATs import BEATWrapper

import torch
import torch.nn as nn
from einops import rearrange
from transformers import Wav2Vec2ForCTC, HubertModel


def default(value, default):
    return default if value is None else value


import functools


def handle_oom(func):
    """
    Decorator to handle CUDA Out of Memory errors by moving computation to CPU.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Try running the function normally (usually on GPU)
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(
                    f"CUDA OOM encountered in {func.__name__}: Moving computation to CPU."
                )
                torch.cuda.empty_cache()  # Clear unused GPU memory

                # Move all tensor arguments and the model itself to CPU
                new_args = [
                    arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in args
                ]
                new_kwargs = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in kwargs.items()
                }

                # Ensure the model (assumed to be part of the 'self' in method args) is moved to CPU
                if hasattr(args[0], "model"):
                    args[0].model.cpu()
                return func(*new_args, **new_kwargs)
            else:
                raise  # Re-raise if the error is not due to CUDA OOM

    return wrapper


class AudioWrapper(nn.Module):
    def __init__(self, model_type="whisper", model_size="large-v3", fps=25) -> None:
        super().__init__()

        if model_type == "whisper":
            self.model = Whisper(model_size, fps, "None")
            self.encode_audio = self.whisper_encoding
        elif model_type == "wavlm":
            self.model = WavLM_wrapper(
                model_size="Base+",
                feed_as_frames=False,
                merge_type="None",
                model_path="",
            )
            self.encode_audio = self.wavlm_encoding
        elif model_type == "wav2vec2":
            self.model = Wav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-base-960h",
                # attn_implementation="flash_attention_2",
                # torch_dtype=torch.bfloat16,
            )
            # self.model = self.model.to(torch.float16)
            self.encode_audio = self.wav2vec2_encoding
        elif model_type == "hubert":
            self.model = HubertModel.from_pretrained(
                "facebook/hubert-base-ls960",
                # attn_implementation="flash_attention_2",
                # torch_dtype=torch.bfloat16,
            )
            self.encode_audio = self.hubert_encoding
        elif model_type == "beats":
            self.model = BEATWrapper(
                fine_tuned=True,
                feed_as_frames=False,
                model_path="pretrained_models/checkpoints/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
            )
            self.encode_audio = self.beats_encoding
        else:
            raise ValueError(f"Model type {model_type} not supported")

    @torch.no_grad()
    def whisper_encoding(self, audio_frames, chunks=None):
        chunks = default(chunks, 750)
        # Get audio embeddings
        audio_embeddings = []
        for chunk in torch.split(
            audio_frames, chunks, dim=0
        ):  # 750 is the max size of the audio chunks that can be processed by the model (= 30 seconds)
            audio_embeddings.append(self.model(chunk.unsqueeze(0).cuda()))
        audio_embeddings = torch.cat(audio_embeddings, dim=1)
        # audio_embeddings = model(audio_frames.unsqueeze(0).cuda())

        # # Save audio embeddings
        # assert audio_embeddings.shape[1] == audio_frames.shape[0], (
        #     f"{audio_embeddings.shape[1]} != {audio_frames.shape[0]}"
        # )

        return audio_embeddings.squeeze(0)

    @torch.no_grad()
    def beats_encoding(self, audio_frames):
        assert audio_frames.dim() == 2, (
            f"Audio frames must be 2D, got {audio_frames.dim()}D"
        )
        audio_embeddings = self.model(audio_frames.unsqueeze(0).cuda())
        return audio_embeddings.squeeze(0)

    @torch.no_grad()
    def wavlm_encoding(self, audio_frames):
        assert audio_frames.dim() == 2, (
            f"Audio frames must be 2D, got {audio_frames.dim()}D"
        )
        audio_embeddings = self.model(audio_frames.unsqueeze(0).cuda())

        assert audio_embeddings.shape[1] == audio_frames.shape[0], (
            f"{audio_embeddings.shape[1]} != {audio_frames.shape[0]}"
        )
        return audio_embeddings.squeeze(0)

    def calculate_splits(self, tensor, min_last_size):
        # Check the total number of elements in the tensor
        total_size = tensor.size(1)  # size along the second dimension

        # If total size is less than the minimum size for the last split, return the tensor as a single split
        if total_size <= min_last_size:
            return [tensor]

        # Calculate number of splits and size of each split
        num_splits = (total_size - min_last_size) // min_last_size + 1
        base_size = (total_size - min_last_size) // num_splits

        # Create split sizes list
        split_sizes = [base_size] * (num_splits - 1)
        split_sizes.append(
            total_size - sum(split_sizes)
        )  # Ensure the last split has at least min_last_size

        # Adjust sizes to ensure they sum exactly to total_size
        sum_sizes = sum(split_sizes)
        while sum_sizes != total_size:
            for i in range(num_splits):
                if sum_sizes < total_size:
                    split_sizes[i] += 1
                    sum_sizes += 1
                if sum_sizes >= total_size:
                    break

        # Split the tensor
        splits = torch.split(tensor, split_sizes, dim=1)

        return splits

    @torch.no_grad()
    def wav2vec2_encoding(self, audio_frames, chunks=None):
        chunks = default(chunks, 16000 * 1000)
        # chunk_size = max(audio_frames.shape[1] // 4, chunks)
        # Get audio embeddings
        audio_embeddings = []
        for chunk in self.calculate_splits(audio_frames, chunks):
            hidden_states = self.model.wav2vec2(chunk.cuda())[0]
            audio_embeddings.append(hidden_states)
        audio_embeddings = torch.cat(audio_embeddings, dim=1)

        # audio_embeddings = self.model.wav2vec2(rearrange(audio_frames, "f s -> () (f s)"))[0]
        if audio_embeddings.shape[1] % 2 != 0:
            audio_embeddings = torch.cat(
                [audio_embeddings, torch.zeros_like(audio_embeddings[:, :1])], dim=1
            )
        audio_embeddings = rearrange(audio_embeddings, "() (f d) c -> f d c", d=2)

        # torch.cuda.empty_cache()

        # assert audio_embeddings.shape[1] == audio_frames.shape[0], f"{audio_embeddings.shape} != {audio_frames.shape}"
        return audio_embeddings

    @torch.no_grad()
    def hubert_encoding(self, audio_frames, chunks=None):
        chunks = default(chunks, 16000 * 1000)
        # chunk_size = max(audio_frames.shape[1] // 4, chunks)
        # Get audio embeddings
        audio_embeddings = []
        for chunk in self.calculate_splits(audio_frames, chunks):
            hidden_states = self.model(chunk.cuda())[0]
            audio_embeddings.append(hidden_states)
        audio_embeddings = torch.cat(audio_embeddings, dim=1)

        # audio_embeddings = self.model.wav2vec2(rearrange(audio_frames, "f s -> () (f s)"))[0]
        if audio_embeddings.shape[1] % 2 != 0:
            audio_embeddings = torch.cat(
                [audio_embeddings, torch.zeros_like(audio_embeddings[:, :1])], dim=1
            )
        audio_embeddings = rearrange(audio_embeddings, "() (f d) c -> f d c", d=2)

        # torch.cuda.empty_cache()
        assert len(audio_embeddings.shape) == 3, (
            f"{audio_embeddings.shape} != (t, f, d)"
        )
        assert audio_embeddings.shape[1] == 2, f"{audio_embeddings.shape} != (t, 2, d)"

        # assert audio_embeddings.shape[1] == audio_frames.shape[0], f"{audio_embeddings.shape} != {audio_frames.shape}"
        return audio_embeddings
