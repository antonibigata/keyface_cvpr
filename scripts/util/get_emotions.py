import os
import argparse
from typing import List
import torch
from scripts.util.emotion_predictor import EmotionDetector
from tqdm import tqdm
import numpy as np


def process_videos(video_list: List[str], model_path: str, output_base: str, video_folder: str, landmarks_folder: str):
    # Initialize EmotionDetector
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emotion_detector = EmotionDetector(model_path, device=device, provide_landmarks=True)

    for video_path in tqdm(video_list, desc="Processing videos", total=len(video_list)):
        # Create output path
        output_path = video_path.replace(video_folder, output_base)
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if os.path.exists(str(output_path).replace(".mp4", ".pt")):
            print(f"Skipping {video_path} -> {output_path} because it already exists")
            continue

        # Load landmarks
        landmarks_path = (
            video_path.replace(video_folder, landmarks_folder)
            .replace(".mp4", ".npy")
            .replace("_output_output", "_output_keypoints")
        )
        if not os.path.exists(landmarks_path):
            print(f"Skipping {video_path} -> {output_path} because landmarks do not exist")
            continue

        if "AA_processed" in landmarks_path:
            landmarks_np = np.load(landmarks_path, allow_pickle=True)
            # Convert object array to float array
            landmarks_np = np.array(landmarks_np.tolist(), dtype=np.float32)
            landmarks = torch.from_numpy(landmarks_np)
        else:
            landmarks_np = np.load(landmarks_path, allow_pickle=False)
            landmarks = torch.from_numpy(landmarks_np)

        # Process the video with landmarks
        try:
            results = emotion_detector.process_video(video_path, landmarks=landmarks)
            # Convert labels list to a tensor
            results["labels"] = torch.tensor(
                [
                    int(key)
                    for label in results["labels"]
                    for key, value in emotion_detector.info["classes"].items()
                    if value == label
                ]
            )

            # Save results as tensor (.pt)
            torch.save(results, str(output_path).replace(".mp4", ".pt"))
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue

        # print(f"Processed {video_path} -> {output_path}")


def get_video_list(video_folder: str) -> List[str]:
    video_list = []
    for root, _, files in os.walk(video_folder):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov")):  # Add more video extensions if needed
                video_list.append(os.path.join(root, file))
    return video_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos for emotion detection")
    parser.add_argument(
        "--model_path",
        default="/data/home/antoni/code/generative-models/scripts/util/detector_8class_va.pt",
        help="Path to the emotion detection model",
    )
    parser.add_argument("--filelist", required=True, help="Path to the file containing list of video paths")
    parser.add_argument("--video_folder", default="video_crop", help="Path to the folder containing videos")
    parser.add_argument("--landmarks_folder", default="landmarks_crop", help="Path to the folder containing landmarks")
    parser.add_argument("--output_folder", default="emotions", help="Path to the folder to save emotion results")
    args = parser.parse_args()

    # Read list of video files from the filelist
    with open(args.filelist, "r") as f:
        video_list = [line.strip() for line in f.readlines()]

    # Randomize the video list
    import random

    random.shuffle(video_list)
    # Process videos
    process_videos(video_list, args.model_path, args.output_folder, args.video_folder, args.landmarks_folder)
