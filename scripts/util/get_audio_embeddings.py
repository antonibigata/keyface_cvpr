"""
Script to get audio embeddings for the audio files in the dataset.
"""

import argparse
import glob
import math
import os
import random
import sys
import gc

import torch
import torchaudio
from einops import rearrange
from tqdm import tqdm
import cv2
from safetensors.torch import save_file

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.util.audio_wrapper import AudioWrapper
from sgm.util import trim_pad_audio


def make_into_multiple_of(x, multiple, dim=0):
    """Make the torch tensor into a multiple of the given number."""
    if x.shape[dim] % multiple != 0:
        x = torch.cat(
            [
                x,
                torch.zeros(
                    *x.shape[:dim],
                    multiple - (x.shape[dim] % multiple),
                    *x.shape[dim + 1 :],
                ).to(x.device),
            ],
            dim=dim,
        )
    return x


argparser = argparse.ArgumentParser()
argparser.add_argument("--audio_path", type=str, default="data/audio_files.txt")
argparser.add_argument(
    "--model_type", type=str, default="whisper", help="Model type: whisper or wavlm"
)
argparser.add_argument(
    "--output_path",
    type=str,
    default=None,
    help="Path to save the embeddings, if None save in same directory as audio file",
)
argparser.add_argument(
    "--model_size",
    type=str,
    default="base",
    help="Model size: base, small, medium, large or large-v2",
)
argparser.add_argument("--fps", type=int, default=25)
argparser.add_argument("--video_fps", type=int, default=25)
argparser.add_argument("--audio_rate", type=int, default=16000)
argparser.add_argument(
    "--audio_folder",
    type=str,
    default="",
    help="Name of audio folder following structure in README file",
)
argparser.add_argument(
    "--video_folder",
    type=str,
    default="",
    help="Name of video folder following structure in README file",
)
argparser.add_argument("--in_dir", type=str, default="NA")
argparser.add_argument("--out_dir", type=str, default="NA")
argparser.add_argument(
    "--recompute",
    action="store_true",
    help="Recompute audio embeddings even if they already exist",
)
argparser.add_argument(
    "--skip_video", action="store_true", help="Skip video processing"
)
argparser.add_argument(
    "--max_size",
    type=int,
    default=None,
    help="Maximum size of audio frames to process at once",
)
args = argparser.parse_args()


def calculate_new_frame_count(original_fps, target_fps, original_frame_count):
    """Calculate new frame count when changing FPS."""
    duration = original_frame_count / original_fps
    new_frame_count = duration * target_fps
    return round(new_frame_count)


def get_video_duration(video_path):
    """Get duration of video in frames."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


@torch.no_grad()
def get_audio_embeddings(audio_path, output_path, model_size, fps, video_fps):
    """Get audio embeddings for the audio files in the dataset."""
    # Load audio files
    audio_files = []
    if audio_path.endswith(".txt"):
        with open(audio_path, "r") as f:
            audio_files = [line.strip() for line in f]
    else:
        audio_files = glob.glob(audio_path)

    audio_rate = args.audio_rate
    a2v_ratio = fps / float(audio_rate)
    samples_per_frame = math.ceil(1 / a2v_ratio)

    # Initialize model
    model = AudioWrapper(model_type=args.model_type, model_size=model_size, fps=fps)
    model.eval()
    model.cuda()

    random.shuffle(audio_files)

    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            if audio_file.endswith(".mp4"):
                audio_file = audio_file.replace(".mp4", ".wav").replace(
                    "video_crop", "audio"
                )

            audio_file_name = os.path.basename(audio_file)
            audio_file_name = (
                f"{os.path.splitext(audio_file_name)[0]}_{args.model_type}_emb.pt"
            )
            audio_file_name = os.path.join(os.path.dirname(audio_file), audio_file_name)

            output_path = audio_file_name.replace(args.in_dir, args.out_dir)
            if os.path.exists(output_path) and not args.recompute:
                continue

            video_fps = args.fps
            video_path = audio_file.replace(
                args.audio_folder, args.video_folder
            ).replace(".wav", ".mp4")

            if "AA_processed" in video_path or "1000actors_nsv" in video_path:
                video_fps = 60
                if "AA_processed" in video_path:
                    video_path = video_path.replace(".mp4", "_output_output.mp4")

            if not args.skip_video and not os.path.exists(video_path):
                print(f"Video file {video_path} does not exist. Skipping...")
                continue

            # Get video length if needed
            max_len_sec = (
                None if args.skip_video else get_video_duration(video_path) / video_fps
            )
            len_video = (
                None
                if args.skip_video
                else calculate_new_frame_count(
                    video_fps, fps, get_video_duration(video_path)
                )
            )

            # Load and process audio
            audio, sr = torchaudio.load(audio_file)
            if sr != audio_rate:
                audio = torchaudio.functional.resample(
                    audio, orig_freq=sr, new_freq=audio_rate
                )[0]
            if audio.dim() == 2:
                audio = audio.mean(0, keepdim=True)

            # Process based on model type
            if args.model_type in ["wav2vec2", "hubert"]:
                audio_embeddings = process_wav2vec_audio(
                    audio, audio_rate, samples_per_frame, max_len_sec, len_video, model
                )
            else:
                audio_embeddings = process_other_audio(
                    audio, audio_rate, samples_per_frame, max_len_sec, len_video, model
                )

            # Save embeddings
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_file(
                {"audio": audio_embeddings.contiguous()},
                output_path.replace(".pt", ".safetensors"),
            )

        except Exception as e:
            print(f"Failed processing {audio_file}: {str(e)}")
            continue

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


def process_wav2vec_audio(
    audio, audio_rate, samples_per_frame, max_len_sec, len_video, model
):
    """Process audio for wav2vec2/hubert models."""
    audio = (audio - audio.mean()) / torch.sqrt(audio.var() + 1e-7)
    if max_len_sec is not None:
        audio = trim_pad_audio(audio, audio_rate, max_len_sec=max_len_sec)[0]
    if audio.dim() == 2:
        audio = audio[0]

    audio = make_into_multiple_of(audio, samples_per_frame, dim=0)
    audio_frames = rearrange(audio, "(f s) -> f s", s=samples_per_frame)

    if not args.skip_video:
        assert audio_frames.shape[0] == len_video, (
            f"Frame count mismatch: {audio_frames.shape[0]} != {len_video}"
        )

    audio = rearrange(audio_frames, "f s -> () (f s)")
    return process_audio_chunks(audio, model)


def process_other_audio(
    audio, audio_rate, samples_per_frame, max_len_sec, len_video, model
):
    """Process audio for other model types."""
    if max_len_sec is not None:
        audio = trim_pad_audio(audio, audio_rate, max_len_sec=max_len_sec)[0]
    if audio.dim() == 2:
        audio = audio[0]

    audio = make_into_multiple_of(audio, samples_per_frame, dim=0)
    audio_frames = rearrange(audio, "(f s) -> f s", s=samples_per_frame)

    if not args.skip_video and audio_frames.shape[0] - len_video == 1:
        audio_frames = audio_frames[:len_video]

    if audio_frames.shape[0] % 2 != 0:
        audio_frames = torch.cat(
            [audio_frames, torch.zeros(1, samples_per_frame)], dim=0
        )

    return process_audio_chunks(audio_frames, model)


def process_audio_chunks(audio_input, model):
    """Process audio in chunks if needed."""
    if (
        args.max_size is not None
        and (audio_input.shape[0] if audio_input.dim() == 2 else audio_input.shape[1])
        > args.max_size
    ):
        mid = (
            audio_input.shape[1]
            if audio_input.dim() == 1
            else audio_input.shape[0] // 2
        )
        chunk1 = (
            audio_input[:mid].cuda()
            if audio_input.dim() == 2
            else audio_input[:, :mid].cuda()
        )
        chunk2 = (
            audio_input[mid:].cuda()
            if audio_input.dim() == 2
            else audio_input[:, mid:].cuda()
        )

        embeddings1 = model.encode_audio(chunk1)
        embeddings2 = model.encode_audio(chunk2)

        return torch.cat(
            [embeddings1.cpu(), embeddings2.cpu()],
            dim=0 if audio_input.dim() == 2 else 1,
        )
    else:
        return model.encode_audio(audio_input.cuda()).cpu()


if __name__ == "__main__":
    get_audio_embeddings(
        args.audio_path,
        args.output_path,
        args.model_size,
        args.fps,
        args.video_fps,
    )
