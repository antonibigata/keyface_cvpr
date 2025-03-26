import math
import os
import sys
from typing import Optional, List, Tuple, Union

import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from torchvision.io import read_video
import torchaudio
from safetensors.torch import load_file as load_safetensors

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from sgm.util import (  # noqa
    default,
    instantiate_from_config,
    trim_pad_audio,
    get_raw_audio,
    save_audio_video,
)


def merge_overlapping_segments(segments: torch.Tensor, overlap: int) -> torch.Tensor:
    """
    Merges overlapping segments by averaging overlapping frames.

    Args:
        segments: Tensor of shape (b, t, ...), where 'b' is the number of segments,
                 't' is frames per segment, and '...' are other dimensions.
        overlap: Number of frames that overlap between consecutive segments.

    Returns:
        Tensor of the merged video.
    """
    # Get the shape details
    b, t, *other_dims = segments.shape
    num_frames = (b - 1) * (
        t - overlap
    ) + t  # Total number of frames in the merged video

    # Initialize the output tensor and a count tensor for averaging
    output_shape = [num_frames] + other_dims
    output = torch.zeros(output_shape, dtype=segments.dtype, device=segments.device)
    count = torch.zeros(output_shape, dtype=torch.float32, device=segments.device)

    current_index = 0
    for i in range(b):
        end_index = current_index + t
        # Add the segment to the output tensor
        output[current_index:end_index] += rearrange(segments[i], "... -> ...")
        # Increment the count tensor for each frame that's added
        count[current_index:end_index] += 1
        # Update the starting index for the next segment
        current_index += t - overlap

    # Avoid division by zero
    count[count == 0] = 1
    # Average the frames where there's overlap
    output /= count

    return output


def create_emotion_list(emotion_states, total_frames, accentuate=False):
    emotion_values = {
        "happy": (0.85, 0.75),  # Joy/Happiness
        "angry": (-0.443, 0.908),  # Anger
        "surprised": (0.0, 0.85),  # Surprise
        "sad": (-0.85, -0.35),  # Sadness
        "neutral": (0.0, 0.0),  # Neutral
        "fear": (0.181, 0.949),  # Fear
        "disgusted": (-0.8, 0.5),  # Disgust
        "contempt": (0.307, 0.535),  # Contempt
        "calm": (0.65, -0.5),  # Calmness
        "excited": (0.9, 0.9),  # Excitement
        "bored": (-0.6, -0.9),  # Boredom
        "confused": (-0.3, 0.4),  # Confusion
        "anxious": (-0.85, 0.9),  # Anxiety
        "confident": (0.7, 0.4),  # Confidence
        "frustrated": (-0.8, 0.6),  # Frustration
        "amused": (0.7, 0.5),  # Amusement
        "proud": (0.8, 0.4),  # Pride
        "ashamed": (-0.8, -0.3),  # Shame
        "grateful": (0.7, 0.2),  # Gratitude
        "jealous": (-0.7, 0.5),  # Jealousy
        "hopeful": (0.7, 0.3),  # Hope
        "disappointed": (-0.7, -0.3),  # Disappointment
        "curious": (0.5, 0.5),  # Curiosity
        "overwhelmed": (-0.6, 0.8),  # Overwhelm
        # Add more emotions as needed
    }

    if accentuate:
        accentuated_values = {
            k: (max(min(v[0] * 1.5, 1), -1), max(min(v[1] * 1.5, 1), -1))
            for k, v in emotion_values.items()
        }
        accentuated_values["neutral"] = (0.0, 0.0)  # Keep neutral as is
        emotion_values = accentuated_values

    if len(emotion_states) == 1:
        v, a = emotion_values[emotion_states[0]]
        valence = [v] * (total_frames + 2)
        arousal = [a] * (total_frames + 2)
    else:
        frames_per_transition = total_frames // (len(emotion_states) - 1)

        valence = []
        arousal = []

        for i in range(len(emotion_states) - 1):
            start_v, start_a = emotion_values[emotion_states[i]]
            end_v, end_a = emotion_values[emotion_states[i + 1]]

            v_values = np.linspace(start_v, end_v, frames_per_transition)
            a_values = np.linspace(start_a, end_a, frames_per_transition)

            valence.extend(v_values)
            arousal.extend(a_values)

        valence = valence[:total_frames]
        arousal = arousal[:total_frames]
        # Save valence and arousal as numpy arrays
        valence = np.array(valence)
        arousal = np.array(arousal)

    return (torch.tensor(valence), torch.tensor(arousal))


def get_audio_indexes(main_index: int, n_audio_frames: int, max_len: int) -> List[int]:
    """
    Get indexes for audio frames centered around the main index.

    Args:
        main_index: Center frame index
        n_audio_frames: Number of audio frames to include on each side
        max_len: Maximum length of the audio sequence

    Returns:
        List of audio frame indexes
    """
    audio_ids = []
    # Add padding at the beginning if needed
    audio_ids += [0] * max(n_audio_frames - main_index, 0)
    # Add actual frame indexes
    for i in range(
        max(main_index - n_audio_frames, 0),
        min(main_index + n_audio_frames + 1, max_len),
    ):
        audio_ids += [i]
    # Add padding at the end if needed
    audio_ids += [max_len - 1] * max(main_index + n_audio_frames - max_len + 1, 0)
    return audio_ids


def create_pipeline_inputs(
    video: torch.Tensor,
    audio: torch.Tensor,
    audio_interpolation: torch.Tensor,
    num_frames: int,
    video_emb: Optional[torch.Tensor] = None,
    emotions: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    overlap: int = 1,
    add_zero_flag: bool = False,
    is_image_model: bool = False,
    accentuate=False,
) -> Tuple:
    """
    Create inputs for the keyframe generation and interpolation pipeline.

    Args:
        video: Input video tensor
        audio: Audio embeddings for keyframe generation
        audio_interpolation: Audio embeddings for interpolation
        num_frames: Number of frames per segment
        video_emb: Optional video embeddings
        emotions: Optional emotion tensors (valence, arousal)
        overlap: Number of frames to overlap between segments
        add_zero_flag: Whether to add zero flag every num_frames
        is_image_model: Whether using an image-based model

    Returns:
        Tuple containing all necessary inputs for the pipeline
    """
    audio_interpolation_chunks = []
    audio_image_preds = []
    gt_chunks = []
    emotions_chunks = []
    step = num_frames - overlap

    if emotions is not None:
        emotions = create_emotion_list(emotions, audio.shape[0], accentuate=accentuate)
    else:
        emotions = create_emotion_list(
            ["neutral"], audio.shape[0], accentuate=accentuate
        )

    if accentuate:
        emotions = (
            torch.clamp(emotions[0] * 1.5, -1, 1),
            torch.clamp(emotions[1] * 1.5, -1, 1),
        )
    # Ensure there's at least one step forward on each iteration
    if step < 1:
        step = 1

    audio_image_preds_idx = []
    audio_interp_preds_idx = []
    for i in range(0, audio.shape[0] - num_frames + 1, step):
        try:
            video[i + num_frames - 1]
        except IndexError:
            break  # Last chunk is smaller than num_frames
        segment_end = i + num_frames
        gt_chunks.append(video[i:segment_end])

        # Process first frame of segment
        if i not in audio_image_preds_idx:
            if is_image_model:
                audio_indexes = get_audio_indexes(i, 2, len(audio))
                audio_image_preds.append(audio[audio_indexes])
            else:
                audio_image_preds.append(audio[i])
            audio_image_preds_idx.append(i)
            if emotions is not None:
                emotions_chunks.append((emotions[0][i], emotions[1][i]))

        # Process last frame of segment
        if segment_end - 1 not in audio_image_preds_idx:
            audio_image_preds_idx.append(segment_end - 1)
            if is_image_model:
                audio_indexes = get_audio_indexes(segment_end - 1, 2, len(audio))
                audio_image_preds.append(audio[audio_indexes])
            else:
                audio_image_preds.append(audio[segment_end - 1])
            if emotions is not None:
                emotions_chunks.append(
                    (emotions[0][segment_end - 1], emotions[1][segment_end - 1])
                )

        audio_interpolation_chunks.append(audio_interpolation[i:segment_end])
        audio_interp_preds_idx.append([i, segment_end - 1])

    # Handle remaining audio interpolation data
    remaining_frames = len(audio_interpolation) - segment_end
    if remaining_frames > 0:
        # Create a new chunk with the remaining frames
        last_chunk = audio_interpolation[segment_end:]

        # If the remaining chunk is smaller than num_frames, pad it by repeating the last element
        if len(last_chunk) < num_frames:
            padding_needed = num_frames - len(last_chunk)
            last_element = last_chunk[-1]
            padded_chunk = torch.cat(
                [last_chunk, last_element.repeat(padding_needed, 1, 1)], dim=0
            )
            audio_interpolation_chunks.append(padded_chunk)
        else:
            # If it's already the right size or larger, just take the first num_frames
            audio_interpolation_chunks.append(last_chunk[:num_frames])

        # Update the indices for this chunk
        audio_interp_preds_idx.append([segment_end, segment_end + num_frames - 1])

    # Add element 0 every num_frames elements if flag is on
    if add_zero_flag:
        first_element = audio_image_preds[0]
        if emotions is not None:
            first_element_emotions = (emotions[0][0], emotions[1][0])
        len_audio_image_preds = (
            len(audio_image_preds) + (len(audio_image_preds) + 1) % num_frames
        )
        for i in range(0, len_audio_image_preds, num_frames):
            audio_image_preds.insert(i, first_element)
            audio_image_preds_idx.insert(i, None)
            if emotions is not None:
                emotions_chunks.insert(i, first_element_emotions)

    to_remove = [idx is None for idx in audio_image_preds_idx]
    audio_image_preds_idx_clone = [idx for idx in audio_image_preds_idx]

    if add_zero_flag:
        # Remove the added elements from the list
        audio_image_preds_idx = [
            sample for i, sample in zip(to_remove, audio_image_preds_idx) if not i
        ]

    # Create interpolation condition list
    interpolation_cond_list = []
    for i in range(0, len(audio_image_preds_idx) - 1, overlap if overlap > 0 else 2):
        interpolation_cond_list.append(
            [audio_image_preds_idx[i], audio_image_preds_idx[i + 1]]
        )

    # Ensure the last chunk is of size num_frames
    frames_needed = (num_frames - (len(audio_image_preds) % num_frames)) % num_frames

    # Extend audio_image_preds
    audio_image_preds = audio_image_preds + [audio_image_preds[-1]] * frames_needed
    if emotions is not None:
        emotions_chunks = emotions_chunks + [emotions_chunks[-1]] * frames_needed
    to_remove = to_remove + [True] * frames_needed
    audio_image_preds_idx_clone = (
        audio_image_preds_idx_clone + [audio_image_preds_idx_clone[-1]] * frames_needed
    )

    print(
        f"Added {frames_needed} frames from the start to make audio_image_preds a multiple of {num_frames}"
    )

    random_cond_idx = 0

    assert len(to_remove) == len(audio_image_preds), (
        "to_remove and audio_image_preds must have the same length"
    )

    return (
        gt_chunks,
        audio_interpolation_chunks,
        audio_image_preds,
        video_emb[random_cond_idx] if video_emb is not None else None,
        video[random_cond_idx],
        emotions_chunks,
        random_cond_idx,
        frames_needed,
        to_remove,
        audio_interp_preds_idx,
        audio_image_preds_idx_clone,
    )


def get_audio_embeddings(
    audio_path: str,
    audio_rate: int = 16000,
    fps: int = 25,
    audio_emb_type: str = "wav2vec2",
    audio_folder: Optional[str] = None,
    audio_emb_folder: Optional[str] = None,
    extra_audio: Union[bool, str] = False,
    max_frames: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Load audio embeddings from file or generate them from raw audio.

    Args:
        audio_path: Path to audio file or embeddings
        audio_rate: Audio sample rate
        fps: Frames per second
        audio_emb_type: Type of audio embeddings
        audio_folder: Folder containing raw audio files
        audio_emb_folder: Folder containing audio embedding files
        extra_audio: Whether to include extra audio embeddings
        max_frames: Maximum number of frames to process

    Returns:
        Tuple of (audio embeddings, interpolation audio embeddings, raw audio)
    """
    audio = None
    raw_audio = None
    audio_interpolation = None

    if audio_path is not None and (
        audio_path.endswith(".wav") or audio_path.endswith(".mp3")
    ):
        # Process raw audio file
        audio, sr = torchaudio.load(audio_path, channels_first=True)
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(
            audio, orig_freq=sr, new_freq=audio_rate
        )[0]
        samples_per_frame = math.ceil(audio_rate / fps)
        n_frames = audio.shape[-1] / samples_per_frame
        if not n_frames.is_integer():
            print("Audio shape before trim_pad_audio: ", audio.shape)
            audio = trim_pad_audio(
                audio, audio_rate, max_len_raw=math.ceil(n_frames) * samples_per_frame
            )
            print("Audio shape after trim_pad_audio: ", audio.shape)
        raw_audio = rearrange(audio, "(f s) -> f s", s=samples_per_frame)

        if "whisper" in audio_path.lower():
            raise NotImplementedError("Whisper audio embeddings are not yet supported.")

    elif audio_path is not None and audio_path.endswith(".safetensors"):
        # Load pre-computed audio embeddings
        audio = load_safetensors(audio_path)["audio"]
        if audio_emb_type != "wav2vec2":
            audio_interpolation = load_safetensors(
                audio_path.replace(f"_{audio_emb_type}_emb", "_wav2vec2_emb")
            )["audio"]
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        print(audio.shape)

        if max_frames is not None:
            audio = audio[:max_frames]
            if audio_interpolation is not None:
                audio_interpolation = audio_interpolation[:max_frames]

        # Handle extra audio embeddings
        if extra_audio in ["key", "both"]:
            extra_audio_emb = load_safetensors(
                audio_path.replace(f"_{audio_emb_type}_emb", "_beats_emb")
            )["audio"]
            if max_frames is not None:
                extra_audio_emb = extra_audio_emb[:max_frames]
            print(
                f"Loaded extra audio embeddings from {audio_path.replace(f'_{audio_emb_type}_emb', '_beats_emb')} {extra_audio_emb.shape}."
            )
            min_size = min(audio.shape[0], extra_audio_emb.shape[0])
            audio = torch.cat([audio[:min_size], extra_audio_emb[:min_size]], dim=-1)
            print(f"Loaded audio embeddings from {audio_path} {audio.shape}.")

        print(audio.shape)

        if audio_interpolation is None:
            audio_interpolation = audio
        elif extra_audio in ["interp", "both"]:
            extra_audio_emb = load_safetensors(
                audio_path.replace(f"_{audio_emb_type}_emb", "_beats_emb")
            )["audio"]
            if max_frames is not None:
                extra_audio_emb = extra_audio_emb[:max_frames]
            print(
                f"Loaded extra audio embeddings from {audio_path.replace(f'_{audio_emb_type}_emb', '_beats_emb')} {extra_audio_emb.shape}."
            )
            min_size = min(audio_interpolation.shape[0], extra_audio_emb.shape[0])
            audio = audio[:min_size]
            audio_interpolation = torch.cat(
                [audio_interpolation[:min_size], extra_audio_emb[:min_size]], dim=-1
            )
            print(f"Loaded audio embeddings from {audio_path} {audio.shape}.")

        print(f"Loaded audio embeddings from {audio_path} {audio.shape}.")

        # Try to load raw audio if available
        raw_audio_path = audio_path.replace(".safetensors", ".wav").replace(
            f"_{audio_emb_type}_emb", ""
        )
        if audio_folder is not None:
            raw_audio_path = raw_audio_path.replace(audio_emb_folder, audio_folder)

        if os.path.exists(raw_audio_path):
            raw_audio = get_raw_audio(raw_audio_path, audio_rate)
        else:
            print(f"WARNING: Could not find raw audio file at {raw_audio_path}.")

    return audio, audio_interpolation, raw_audio


def sample_keyframes(
    model_keyframes,
    audio_list: torch.Tensor,
    condition: torch.Tensor,
    num_frames: int,
    fps_id: int,
    cond_aug: float,
    device: str,
    embbedings: Optional[torch.Tensor],
    valence_list: Optional[torch.Tensor],
    arousal_list: Optional[torch.Tensor],
    force_uc_zero_embeddings: List[str],
    n_batch_keyframes: int,
    added_frames: int,
    strength: float,
    scale: Optional[float],
    num_steps: Optional[int],
    is_image_model: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate keyframes using the keyframe model.

    Args:
        model_keyframes: The keyframe generation model
        audio_list: Audio embeddings for conditioning
        condition: Visual conditioning
        num_frames: Number of frames to generate
        fps_id: FPS ID for conditioning
        cond_aug: Conditioning augmentation strength
        device: Device to run inference on
        embbedings: Optional embeddings for conditioning
        valence_list: Optional valence values for emotion conditioning
        arousal_list: Optional arousal values for emotion conditioning
        force_uc_zero_embeddings: Keys to zero out in unconditional embeddings
        n_batch_keyframes: Batch size for keyframe generation
        added_frames: Number of frames added for padding
        strength: Strength parameter for sampling
        scale: Optional classifier-free guidance scale
        num_steps: Optional number of sampling steps
        is_image_model: Whether using an image-based model

    Returns:
        Tuple of (latent samples, decoded samples)
    """
    if scale is not None:
        model_keyframes.sampler.guider.set_scale(scale)
    if num_steps is not None:
        model_keyframes.sampler.set_num_steps(num_steps)

    samples_list = []
    samples_z_list = []
    samples_x_list = []

    for i in range(audio_list.shape[0]):
        H, W = condition.shape[-2:]
        assert condition.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)

        if fps_id < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")
        if fps_id > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        audio_cond = audio_list[i].unsqueeze(0)

        value_dict = {}
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames_without_noise"] = condition

        if embbedings is not None:
            value_dict["cond_frames"] = embbedings + cond_aug * torch.randn_like(
                embbedings
            )
        else:
            value_dict["cond_frames"] = condition + cond_aug * torch.randn_like(
                condition
            )

        value_dict["cond_aug"] = cond_aug
        value_dict["audio_emb"] = audio_cond

        if valence_list is not None:
            value_dict["valence"] = valence_list[i].unsqueeze(0)
            value_dict["arousal"] = arousal_list[i].unsqueeze(0)

        if is_image_model:
            value_dict["audio_emb"] = value_dict["audio_emb"].squeeze(0)

        with torch.no_grad():
            with torch.autocast(device):
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(
                        model_keyframes.conditioner
                    ),
                    value_dict,
                    [1, 1],
                    T=num_frames,
                    device=device,
                )

                c, uc = model_keyframes.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                )

                for k in ["crossattn"]:
                    if c[k].shape[1] != num_frames:
                        uc[k] = repeat(
                            uc[k],
                            "b ... -> b t ...",
                            t=num_frames if not is_image_model else 1,
                        )
                        uc[k] = rearrange(
                            uc[k],
                            "b t ... -> (b t) ...",
                            t=num_frames if not is_image_model else 1,
                        )
                        c[k] = repeat(
                            c[k],
                            "b ... -> b t ...",
                            t=num_frames if not is_image_model else 1,
                        )
                        c[k] = rearrange(
                            c[k],
                            "b t ... -> (b t) ...",
                            t=num_frames if not is_image_model else 1,
                        )

                video = torch.randn(shape, device=device)

                for k in c:
                    if isinstance(c[k], torch.Tensor):
                        print(k, c[k].shape)

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(
                    n_batch_keyframes, num_frames
                ).to(device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                def denoiser(input, sigma, c):
                    return model_keyframes.denoiser(
                        model_keyframes.model,
                        input,
                        sigma,
                        c,
                        **additional_model_inputs,
                    )

                samples_z = model_keyframes.sampler(
                    denoiser, video, cond=c, uc=uc, strength=strength
                )
                samples_z_list.append(samples_z)

                samples_x = model_keyframes.decode_first_stage(samples_z)
                samples_x_list.append(samples_x)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                samples_list.append(samples)

                video = None

    samples = (
        torch.concat(samples_list)[:-added_frames]
        if added_frames > 0
        else torch.concat(samples_list)
    )
    samples_z = (
        torch.concat(samples_z_list)[:-added_frames]
        if added_frames > 0
        else torch.concat(samples_z_list)
    )
    samples_x = (
        torch.concat(samples_x_list)[:-added_frames]
        if added_frames > 0
        else torch.concat(samples_x_list)
    )

    return samples_z, samples_x


def sample_interpolation(
    model,
    samples_z: torch.Tensor,
    samples_x: torch.Tensor,
    audio_interpolation_list: List[torch.Tensor],
    condition: torch.Tensor,
    num_frames: int,
    device: str,
    overlap: int,
    fps_id: int,
    cond_aug: float,
    force_uc_zero_embeddings: List[str],
    n_batch: int,
    chunk_size: Optional[int],
    strength: float,
    scale: Optional[float],
    num_steps: Optional[int],
    cut_audio: bool = False,
    to_remove: List[bool] = [],
) -> np.ndarray:
    """
    Generate interpolated frames between keyframes.

    Args:
        model: The interpolation model
        samples_z: Latent keyframes
        samples_x: Decoded keyframes
        audio_interpolation_list: Audio embeddings for interpolation
        condition: Visual conditioning
        num_frames: Number of frames per segment
        device: Device to run inference on
        overlap: Number of frames to overlap
        fps_id: FPS ID for conditioning
        cond_aug: Conditioning augmentation strength
        force_uc_zero_embeddings: Keys to zero out in unconditional embeddings
        n_batch: Batch size for interpolation
        chunk_size: Optional chunk size for processing
        strength: Strength parameter for sampling
        scale: Optional classifier-free guidance scale
        num_steps: Optional number of sampling steps
        cut_audio: Whether to cut audio embeddings
        to_remove: List indicating which frames to remove

    Returns:
        Numpy array of generated video frames
    """
    if scale is not None:
        model.sampler.guider.set_scale(scale)
    if num_steps is not None:
        model.sampler.set_num_steps(num_steps)

    # Filter out frames marked for removal
    samples_x = [sample for i, sample in zip(to_remove, samples_x) if not i]
    samples_z = [sample for i, sample in zip(to_remove, samples_z) if not i]

    # Create interpolation condition lists
    interpolation_cond_list = []
    interpolation_cond_list_emb = []

    for i in range(0, len(samples_z) - 1, overlap if overlap > 0 else 2):
        interpolation_cond_list.append(
            torch.stack([samples_x[i], samples_x[i + 1]], dim=1)
        )
        interpolation_cond_list_emb.append(
            torch.stack([samples_z[i], samples_z[i + 1]], dim=1)
        )

    condition = torch.stack(interpolation_cond_list).to(device)
    audio_cond = torch.stack(audio_interpolation_list).to(device)
    embbedings = torch.stack(interpolation_cond_list_emb).to(device)

    H, W = condition.shape[-2:]
    F = 8
    C = 4
    shape = (num_frames * audio_cond.shape[0], C, H // F, W // F)

    if (H, W) != (576, 1024):
        print(
            "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
        )

    if fps_id < 5:
        print("WARNING: Small fps value! This may lead to suboptimal performance.")
    if fps_id > 30:
        print("WARNING: Large fps value! This may lead to suboptimal performance.")
    # Prepare value dictionary for conditioning
    value_dict = {
        "fps_id": fps_id,
        "cond_aug": cond_aug,
        "cond_frames_without_noise": condition,
        "cond_frames": embbedings,
    }

    # Handle audio embeddings based on cut_audio flag
    if cut_audio:
        value_dict["audio_emb"] = audio_cond[:, :, :, :768]
    else:
        value_dict["audio_emb"] = audio_cond

    with torch.no_grad():
        with torch.autocast(device):
            # Prepare batch and unconditional batch
            batch, batch_uc = get_batch_overlap(
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict,
                [1, num_frames],
                T=num_frames,
                device=device,
            )

            # Get conditional and unconditional embeddings
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )

            # Handle cross-attention conditioning
            for k in ["crossattn"]:
                if c[k].shape[1] != num_frames:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

            # Initialize noise for diffusion process
            video = torch.randn(shape, device=device)

            # Prepare additional model inputs
            additional_model_inputs = {
                "image_only_indicator": torch.zeros(n_batch, num_frames).to(device),
                "num_video_frames": batch["num_video_frames"],
            }

            # Debug information
            print(
                f"Shapes - Condition: {condition.shape}, Embeddings: {embbedings.shape}, "
                f"Audio: {audio_cond.shape}, Video: {shape}, Additional inputs: {additional_model_inputs}"
            )

            # Configure chunk size for memory efficiency if specified
            if chunk_size is not None:
                chunk_size = chunk_size * num_frames

            # Define denoiser function for the sampler
            def denoiser(input, sigma, c):
                return model.denoiser(
                    model.model,
                    input,
                    sigma,
                    c,
                    num_overlap_frames=overlap,
                    num_frames=num_frames,
                    n_skips=n_batch,
                    chunk_size=chunk_size,
                    **additional_model_inputs,
                )

            # Run the sampling process
            samples_z = model.sampler(denoiser, video, cond=c, uc=uc, strength=strength)

            # Decode the latent samples to pixel space
            samples_x = model.decode_first_stage(samples_z)

            # Normalize the output samples to [0, 1] range
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

            # Free up memory
            video = None

    # Reshape samples and merge overlapping segments
    samples = rearrange(samples, "(b t) c h w -> b t c h w", t=num_frames)
    samples = merge_overlapping_segments(samples, overlap)

    # Convert to numpy array for output
    vid = (
        (rearrange(samples, "t c h w -> t c h w") * 255).cpu().numpy().astype(np.uint8)
    )

    return vid


def sample(
    model,
    model_keyframes,
    video_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    resize_size: Optional[int] = None,
    video_folder: Optional[str] = None,
    latent_folder: Optional[str] = None,
    audio_folder: Optional[str] = None,
    audio_emb_folder: Optional[str] = None,
    version: str = "svd",
    fps_id: int = 24,
    cond_aug: float = 0.00,
    seed: int = 23,
    decoding_t: int = 14,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = None,
    strength: float = 1.0,
    model_config: Optional[str] = None,
    model_keyframes_config: Optional[str] = None,
    min_seconds: Optional[int] = None,
    force_uc_zero_embeddings=[
        "cond_frames",
        "cond_frames_without_noise",
    ],
    chunk_size: int = None,  # Useful if the model gets OOM
    overlap: int = 1,  # Overlap between frames (i.e Multi-diffusion)
    keyframes_ckpt: Optional[str] = None,
    interpolation_ckpt: Optional[str] = None,
    add_zero_flag: bool = False,
    n_batch: int = 1,
    n_batch_keyframes: int = 1,
    compute_until: float = "end",
    extra_audio: bool = False,
    audio_emb_type: str = "wav2vec2",
    extra_naming: str = "",
    is_image_model: bool = False,
    scale: list = None,
    emotion_states: Optional[list[str]] = None,
    accentuate: bool = False,
    recompute: bool = False,
):
    if version == "svd":
        num_frames = default(num_frames, 14)
        output_folder = default(output_folder, "outputs/full_pipeline/svd/")
    elif version == "svd_xt":
        num_frames = default(num_frames, 25)
        output_folder = default(output_folder, "outputs/full_pipeline/svd_xt/")
    elif version == "svd_image_decoder":
        num_frames = default(num_frames, 14)
        output_folder = default(
            output_folder, "outputs/full_pipeline/svd_image_decoder/"
        )
    elif version == "svd_xt_image_decoder":
        num_frames = default(num_frames, 25)
        output_folder = default(
            output_folder, "outputs/full_pipeline/svd_xt_image_decoder/"
        )
    else:
        raise ValueError(f"Version {version} does not exist.")

    os.makedirs(output_folder, exist_ok=True)

    if extra_naming != "":
        video_out_name = (
            os.path.basename(video_path).replace(".mp4", "")
            + "_"
            + extra_naming
            + ".mp4"
        )
    else:
        video_out_name = os.path.basename(video_path)

    out_video_path = os.path.join(output_folder, video_out_name)

    if os.path.exists(out_video_path) and not recompute:
        print(f"Video already exists at {out_video_path}. Skipping.")
        return

    torch.manual_seed(seed)

    video = read_video(video_path, output_format="TCHW")[0]
    video = (video / 255.0) * 2.0 - 1.0
    h, w = video.shape[2:]
    video = torch.nn.functional.interpolate(video, (512, 512), mode="bilinear")

    video_embedding_path = video_path.replace(".mp4", "_video_512_latent.safetensors")
    if video_folder is not None and latent_folder is not None:
        video_embedding_path = video_embedding_path.replace(video_folder, latent_folder)
    video_emb = load_safetensors(video_embedding_path)["latents"].cpu()

    if compute_until == "end":
        compute_until = int((video.shape[0] * 10) / 25)

    if compute_until is not None:
        max_frames = compute_until * (fps_id + 1)

    audio, audio_interpolation, raw_audio = get_audio_embeddings(
        audio_path,
        16000,
        fps_id + 1,
        audio_folder=audio_folder,
        audio_emb_folder=audio_emb_folder,
        extra_audio=extra_audio,
        max_frames=max_frames,
        audio_emb_type=audio_emb_type,
    )
    if compute_until is not None:
        if video.shape[0] > max_frames:
            video = video[:max_frames]
            audio = audio[:max_frames]
            video_emb = video_emb[:max_frames] if video_emb is not None else None
            raw_audio = raw_audio[:max_frames] if raw_audio is not None else None
    if min_seconds is not None:
        min_frames = min_seconds * (fps_id + 1)
        video = video[min_frames:]
        audio = audio[min_frames:]
        video_emb = video_emb[min_frames:] if video_emb is not None else None
        raw_audio = raw_audio[min_frames:] if raw_audio is not None else None
    audio = audio

    print(
        "Video has ",
        video.shape[0],
        "frames",
        "and",
        video.shape[0] / 25,
        "seconds",
    )

    h, w = video.shape[2:]

    model_input = video
    if h % 64 != 0 or w % 64 != 0:
        width, height = map(lambda x: x - x % 64, (w, h))
        if resize_size is not None:
            width, height = (
                (resize_size, resize_size)
                if isinstance(resize_size, int)
                else resize_size
            )
        else:
            width = min(width, 1024)
            height = min(height, 576)
        model_input = torch.nn.functional.interpolate(
            model_input, (height, width), mode="bilinear"
        ).squeeze(0)
        print(
            f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
        )

    (
        gt_chunks,
        audio_interpolation_list,
        audio_list,
        emb,
        cond,
        emotions_chunks,
        _,
        added_frames,
        to_remove,
        test_interpolation_list,
        test_keyframes_list,
    ) = create_pipeline_inputs(
        model_input,
        audio,
        audio_interpolation,
        num_frames,
        video_emb,
        emotions=emotion_states,
        overlap=overlap,
        add_zero_flag=add_zero_flag,
        is_image_model=is_image_model,
        accentuate=accentuate,
    )

    model_keyframes.en_and_decode_n_samples_a_time = decoding_t
    model.en_and_decode_n_samples_a_time = decoding_t

    audio_list = torch.stack(audio_list).to(device)
    if is_image_model:
        audio_list = rearrange(audio_list, "(b t) x c d  -> b t x c d", t=num_frames)
    else:
        audio_list = rearrange(audio_list, "(b t) c d  -> b t c d", t=num_frames)

    # Convert to_remove into chunks of num_frames
    to_remove_chunks = [
        to_remove[i : i + num_frames] for i in range(0, len(to_remove), num_frames)
    ]
    test_keyframes_list = [
        test_keyframes_list[i : i + num_frames]
        for i in range(0, len(test_keyframes_list), num_frames)
    ]

    valence_list = None
    arousal_list = None
    if emotions_chunks is not None:
        valence_list = torch.stack([x[0] for x in emotions_chunks]).to(device)
        arousal_list = torch.stack([x[1] for x in emotions_chunks]).to(device)
        valence_list = rearrange(valence_list, "(b t) -> b t", t=num_frames)
        arousal_list = rearrange(arousal_list, "(b t) -> b t", t=num_frames)

    condition = cond

    audio_cond = audio_list
    condition = condition.unsqueeze(0).to(device)
    embbedings = emb.unsqueeze(0).to(device) if emb is not None else None

    # One batch of keframes is approximately 7 seconds
    chunk_size = 2
    complete_video = []
    complete_audio = []
    start_idx = 0
    last_frame_z = None
    last_frame_x = None
    last_keyframe_idx = None
    last_to_remove = None

    for chunk_start in range(0, len(audio_cond), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(audio_cond))
        is_last_chunk = chunk_end == len(audio_cond)  # Flag for last chunk

        chunk_audio_cond = audio_cond[chunk_start:chunk_end].cuda()
        chunk_valence_list = (
            valence_list[chunk_start:chunk_end].cuda()
            if valence_list is not None
            else None
        )
        chunk_arousal_list = (
            arousal_list[chunk_start:chunk_end].cuda()
            if arousal_list is not None
            else None
        )

        test_keyframes_list_unwrapped = [
            elem
            for sublist in test_keyframes_list[chunk_start:chunk_end]
            for elem in sublist
        ]
        to_remove_chunks_unwrapped = [
            elem
            for sublist in to_remove_chunks[chunk_start:chunk_end]
            for elem in sublist
        ]

        if last_keyframe_idx is not None:
            test_keyframes_list_unwrapped = [
                last_keyframe_idx
            ] + test_keyframes_list_unwrapped
            to_remove_chunks_unwrapped = [last_to_remove] + to_remove_chunks_unwrapped

        last_keyframe_idx = test_keyframes_list_unwrapped[-1]
        last_to_remove = to_remove_chunks_unwrapped[-1]
        # Find the first non-None keyframe in the chunk
        first_keyframe = next(
            (kf for kf in test_keyframes_list_unwrapped if kf is not None), None
        )

        # Find the last non-None keyframe in the chunk
        last_keyframe = next(
            (kf for kf in reversed(test_keyframes_list_unwrapped) if kf is not None),
            None,
        )

        start_idx = next(
            (
                idx
                for idx, comb in enumerate(test_interpolation_list)
                if comb[0] == first_keyframe
            ),
            None,
        )
        end_idx = next(
            (
                idx
                for idx, comb in enumerate(reversed(test_interpolation_list))
                if comb[1] == last_keyframe
            ),
            None,
        )

        if start_idx is not None and end_idx is not None:
            end_idx = (
                len(test_interpolation_list) - 1 - end_idx
            )  # Adjust for reversed enumeration
        end_idx += 1
        if start_idx is None:
            break
        if end_idx < start_idx:
            end_idx = len(audio_interpolation_list)

        if is_last_chunk:
            end_idx += 1
        audio_interpolation_list_chunk = audio_interpolation_list[start_idx:end_idx]

        samples_z, samples_x = sample_keyframes(
            model_keyframes,
            chunk_audio_cond,
            condition.cuda(),
            num_frames,
            fps_id,
            cond_aug,
            device,
            embbedings.cuda(),
            chunk_valence_list,
            chunk_arousal_list,
            force_uc_zero_embeddings,
            n_batch_keyframes,
            0,
            strength,
            None,
            None,
            is_image_model=is_image_model,
        )

        if last_frame_x is not None:
            samples_x = torch.cat([last_frame_x.unsqueeze(0), samples_x], axis=0)
            samples_z = torch.cat([last_frame_z.unsqueeze(0), samples_z], axis=0)

        last_frame_x = samples_x[-1]
        last_frame_z = samples_z[-1]

        if is_last_chunk:
            # For the last chunk, make the first True after the last False in to_remove_chunks_unwrapped True
            # Skip the first element if it's True
            start_idx = 1 if to_remove_chunks_unwrapped[0] else 0

            # Go through the list and make the first True element False, then break
            for i in range(start_idx, len(to_remove_chunks_unwrapped)):
                if to_remove_chunks_unwrapped[i]:
                    to_remove_chunks_unwrapped[i] = False
                    break

        vid = sample_interpolation(
            model,
            samples_z,
            samples_x,
            audio_interpolation_list_chunk,
            condition.cuda(),
            num_frames,
            device,
            overlap,
            fps_id,
            cond_aug,
            force_uc_zero_embeddings,
            n_batch,
            chunk_size,
            strength,
            None,
            None,
            cut_audio=extra_audio not in ["both", "interp"],
            to_remove=to_remove_chunks_unwrapped,
        )

        if chunk_start == 0:
            complete_video = vid
        else:
            complete_video = np.concatenate([complete_video[:-1], vid], axis=0)

    complete_video = complete_video[: raw_audio.shape[0]]

    assert complete_video.shape[0] == raw_audio.shape[0]

    if raw_audio is not None:
        complete_audio = rearrange(raw_audio, "f s -> () (f s)")

    save_audio_video(
        complete_video,
        audio=complete_audio,
        frame_rate=fps_id + 1,
        sample_rate=16000,
        save_path=out_video_path,
        keep_intermediate=False,
    )

    print(f"Saved video to {out_video_path}")


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def get_batch_overlap(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "b ... -> (b t) ...", t=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "b ... -> (b t) ...", t=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    input_key: str,
    ckpt: Optional[str] = None,
    low_sigma: float = 0.0,
    high_sigma: float = float("inf"),
):
    config = OmegaConf.load(config)
    config["model"]["params"]["input_key"] = input_key

    if ckpt is not None:
        config.model.params.ckpt_path = ckpt

    if num_steps is not None:
        config.model.params.sampler_config.params.num_steps = num_steps
    if "num_frames" in config.model.params.sampler_config.params.guider_config.params:
        config.model.params.sampler_config.params.guider_config.params.num_frames = (
            num_frames
        )

    if (
        "IdentityGuider"
        in config.model.params.sampler_config.params.guider_config.target
    ):
        n_batch = 1
    elif (
        "MultipleCondVanilla"
        in config.model.params.sampler_config.params.guider_config.target
    ):
        n_batch = 3
    elif (
        "AudioRefMultiCondGuider"
        in config.model.params.sampler_config.params.guider_config.target
    ):
        n_batch = 3
    else:
        n_batch = 2  # Conditional and unconditional
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    return model, filter, n_batch


def main(
    filelist: str = "",
    filelist_audio: str = "",
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    resize_size: Optional[int] = None,
    video_folder: Optional[str] = None,
    latent_folder: Optional[str] = None,
    audio_folder: Optional[str] = None,
    audio_emb_folder: Optional[str] = None,
    version: str = "svd",
    fps_id: int = 24,
    cond_aug: float = 0.00,
    seed: int = 23,
    decoding_t: int = 14,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = None,
    strength: float = 1.0,
    model_config: Optional[str] = None,
    model_keyframes_config: Optional[str] = None,
    min_seconds: Optional[int] = None,
    force_uc_zero_embeddings=[
        "cond_frames",
        "cond_frames_without_noise",
    ],
    chunk_size: int = None,  # Useful if the model gets OOM
    overlap: int = 1,  # Overlap between frames (i.e Multi-diffusion)
    keyframes_ckpt: Optional[str] = None,
    interpolation_ckpt: Optional[str] = None,
    add_zero_flag: bool = False,
    extra_audio: str = None,
    compute_until: str = "end",
    starting_index: int = 0,
    audio_emb_type: str = "wav2vec2",
    is_image_model: bool = False,
    scale: list = None,
    accentuate: bool = False,
    emotion_states: Optional[list[str]] = None,
    recompute: bool = False,
):
    print("Scale: ", scale)
    num_frames = default(num_frames, 14)
    model, filter, n_batch = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        "latents",
        interpolation_ckpt,
    )

    model_keyframes, filter, n_batch_keyframes = load_model(
        model_keyframes_config,
        device,
        num_frames,
        num_steps,
        "latents",
        keyframes_ckpt,
    )
    if scale is not None:
        if len(scale) == 1:
            scale = scale[0]
        model_keyframes.sampler.guider.set_scale(scale)

    # Open the filelist and read the video paths
    with open(filelist, "r") as f:
        video_paths = f.readlines()

    # Remove the newline character from each path
    video_paths = [path.strip() for path in video_paths]

    if filelist_audio:
        with open(filelist_audio, "r") as f:
            audio_paths = f.readlines()
        audio_paths = [
            path.strip()
            .replace(video_folder, audio_folder)
            .replace(".mp4", f"_{audio_emb_type}_emb.safetensors")
            for path in audio_paths
        ]
    else:
        audio_paths = [
            video_path.replace(video_folder, audio_folder).replace(
                ".mp4", f"_{audio_emb_type}_emb.safetensors"
            )
            for video_path in video_paths
        ]

    if starting_index:
        video_paths = video_paths[starting_index:]
        audio_paths = audio_paths[starting_index:]

    for video_path, audio_path in zip(video_paths, audio_paths):
        sample(
            model,
            model_keyframes,
            video_path=video_path,
            audio_path=audio_path,
            num_frames=num_frames,
            num_steps=num_steps,
            resize_size=resize_size,
            video_folder=video_folder,
            latent_folder=latent_folder,
            audio_folder=audio_folder,
            audio_emb_folder=audio_emb_folder,
            version=version,
            fps_id=fps_id,
            cond_aug=cond_aug,
            seed=seed,
            decoding_t=decoding_t,
            device=device,
            output_folder=output_folder,
            strength=strength,
            model_config=model_config,
            model_keyframes_config=model_keyframes_config,
            min_seconds=min_seconds,
            force_uc_zero_embeddings=force_uc_zero_embeddings,
            chunk_size=chunk_size,
            overlap=overlap,
            keyframes_ckpt=keyframes_ckpt,
            interpolation_ckpt=interpolation_ckpt,
            add_zero_flag=add_zero_flag,
            n_batch=n_batch,
            n_batch_keyframes=n_batch_keyframes,
            extra_audio=extra_audio,
            compute_until=compute_until,
            audio_emb_type=audio_emb_type,
            extra_naming=os.path.basename(audio_path).split(".")[0]
            if filelist_audio
            else "",
            is_image_model=is_image_model,
            accentuate=accentuate,
            emotion_states=emotion_states,
            recompute=recompute,
        )


if __name__ == "__main__":
    Fire(main)
