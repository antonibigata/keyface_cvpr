import os
import tarfile
import argparse
from tqdm import tqdm
import webdataset as wds
from itertools import islice


def create_webdataset(filelist_path, audio_model=""):
    with open(filelist_path, "r") as filelist:
        filelist = filelist.readlines()

    for line in filelist:
        video_file_path = line.strip()
        speaker_id, video_file = os.path.split(video_file_path)
        audio_path = speaker_id.replace("video", "audio")
        speaker_id = os.path.basename(speaker_id)
        base_filename, _ = os.path.splitext(video_file)
        audio_file = base_filename + ".wav"
        if audio_model:
            audio_embedding_file = base_filename + f"_{audio_model}_emb.pt"
        else:
            audio_embedding_file = base_filename + "_emb.pt"

        audio_file_path = os.path.join(audio_path, audio_file)
        audio_emb_file_path = os.path.join(audio_path, audio_embedding_file)

        if not os.path.exists(audio_file_path):
            print(f"No audio file for video: {video_file_path}")
            continue

        with open(video_file_path, "rb") as f:
            video_bytes = f.read()

        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()

        with open(audio_emb_file_path, "rb") as f:
            audio_emb_bytes = f.read()

        sample = {
            "__key__": speaker_id + "/" + base_filename,
            "mp4": video_bytes,
            "wav": audio_bytes,
            "pt": audio_emb_bytes,
        }

        yield sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a WebDataset from video and audio files.")
    # parser.add_argument("root_path", help="Root path of the video and audio directories.")
    parser.add_argument("tar_output_path", help="Output path for the tar files.")
    parser.add_argument("filelist_path", help="Path to the text file containing the list of video files.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of samples per tar file.")
    parser.add_argument("--audio_model", type=str, default="", help="Audio model to use for feature extraction.")

    args = parser.parse_args()

    os.makedirs(args.tar_output_path, exist_ok=True)

    with wds.ShardWriter(f"{args.tar_output_path}/out-%06d.tar", maxcount=args.batch_size) as sink:
        for sample in islice(create_webdataset(args.filelist_path, args.audio_model), 0, 10000):
            sink.write(sample)
