"""
Script to generate latent vectors from a video file.

This script processes video files or images and generates corresponding latent vectors
using a VAE model. It supports batch processing, chunked video encoding, and various
output formats.
"""

import argparse
import os
import sys
from pathlib import Path
import random
import glob
from typing import List, Optional

import torch
import decord
from PIL import Image
import torchvision.transforms as T
from einops import rearrange
from safetensors.torch import save_file
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.util.vae_wrapper import VaeWrapper

# Configure decord to use PyTorch tensors
decord.bridge.set_bridge("torch")


def default(value: Optional[any], default_value: any) -> any:
    """Return default_value if value is None, otherwise return value."""
    return default_value if value is None else value


def process_image(
    image: torch.Tensor, resolution: Optional[int] = None
) -> torch.Tensor:
    """
    Process image tensor by resizing and normalizing.

    Args:
        image: Input image tensor
        resolution: Target resolution for resizing

    Returns:
        Processed image tensor
    """
    if resolution is not None:
        image = torch.nn.functional.interpolate(
            image.float(), size=resolution, mode="bilinear", align_corners=False
        )
    return image / 127.5 - 1.0


def encode_video_chunk(
    model: VaeWrapper,
    video: torch.Tensor,
    target_resolution: Optional[int] = None,
) -> torch.Tensor:
    """
    Encode a chunk of video frames into latent space.

    Args:
        model: VAE model for encoding
        video: Video tensor to encode
        target_resolution: Target resolution for processing

    Returns:
        Encoded latent tensor
    """
    video = rearrange(video, "t h w c -> c t h w")
    vid_rez = min(video.shape[-1], video.shape[-2])
    to_rez = default(target_resolution, vid_rez)
    video = process_image(video, to_rez)

    encoded = model.encode_video(video.cuda().unsqueeze(0)).squeeze(0)
    return rearrange(encoded, "c t h w -> t c h w")


def main():
    parser = argparse.ArgumentParser(
        description="Generate latent vectors from video/image files"
    )
    parser.add_argument(
        "--filelist",
        type=str,
        required=True,
        nargs="+",
        help="Path(s) to file list or glob pattern",
    )
    parser.add_argument(
        "--in_dir", type=str, default="NA", help="Input directory to replace in paths"
    )
    parser.add_argument(
        "--out_dir", type=str, default="NA", help="Output directory to replace in paths"
    )
    parser.add_argument("--resolution", type=int, default=512, help="Target resolution")
    parser.add_argument(
        "--chunk_size", type=int, default=10, help="Number of frames to process at once"
    )
    parser.add_argument(
        "--diffusion_type", type=str, default="video", help="Type of diffusion model"
    )
    parser.add_argument(
        "--save_as_tensor",
        action="store_true",
        help="Save as .pt instead of .safetensors",
    )
    parser.add_argument(
        "--only_missing", action="store_true", help="Only count missing files"
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force recomputation of existing files",
    )
    args = parser.parse_args()

    # Collect all input files
    video_files: List[str] = []
    for filelist in args.filelist:
        if filelist.endswith(".txt"):
            with open(filelist, "r") as f:
                video_files.extend(line.strip() for line in f)
        else:
            video_files.extend(glob.glob(filelist))

    model = VaeWrapper(args.diffusion_type)
    random.shuffle(video_files)
    missing_count = 0

    for video_file in tqdm(video_files, desc="Processing files"):
        try:
            video_file = str(video_file)
            file_suffix = f"_{args.diffusion_type}_{args.resolution}_latent"
            file_ext = "pt" if args.save_as_tensor else "safetensors"
            out_path = (
                Path(video_file)
                .with_stem(Path(video_file).stem + file_suffix)
                .with_suffix(f".{file_ext}")
            )
            out_path = Path(str(out_path).replace(args.in_dir, args.out_dir))
            os.makedirs(out_path.parent, exist_ok=True)

            if out_path.exists() and not args.force_recompute:
                continue

            if args.only_missing:
                missing_count += 1
                continue

            with torch.no_grad():
                if video_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    # Process single image
                    img = Image.open(video_file).convert("RGB")
                    video = T.ToTensor()(img).unsqueeze(0) * 255.0
                    encoded = encode_video_chunk(
                        model, rearrange(video, "t c h w -> t h w c"), args.resolution
                    )
                    video_reader = [None]
                else:
                    # Process video
                    video_reader = decord.VideoReader(video_file)
                    total_frames = len(video_reader)

                    if args.chunk_size:
                        encoded = []
                        for start_idx in tqdm(
                            range(0, total_frames, args.chunk_size),
                            leave=False,
                            desc="Chunks",
                        ):
                            end_idx = min(start_idx + args.chunk_size, total_frames)
                            video_chunk = video_reader.get_batch(
                                range(start_idx, end_idx)
                            )
                            encoded.append(
                                encode_video_chunk(model, video_chunk, args.resolution)
                            )
                        encoded = torch.cat(encoded, dim=0)
                    else:
                        video = video_reader.get_batch(range(total_frames))
                        encoded = encode_video_chunk(model, video, args.resolution)

                # Ensure encoded frames match input
                if encoded.shape[0] > len(video_reader):
                    encoded = encoded[: len(video_reader)]
                assert encoded.shape[0] == len(video_reader), (
                    f"Frame count mismatch: {encoded.shape[0]} != {len(video_reader)}"
                )

                # Save output
                if args.save_as_tensor:
                    torch.save(encoded.cpu(), out_path)
                else:
                    save_file(
                        {"latents": encoded.cpu()},
                        out_path,
                    )

        except Exception as e:
            print(f"Failed processing {video_file}: {str(e)}")
            raise

    if args.only_missing:
        print(f"Missing files: {missing_count}")


if __name__ == "__main__":
    main()
