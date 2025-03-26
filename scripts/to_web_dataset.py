import os
import tarfile
import argparse
from tqdm import tqdm


def create_webdataset(root_path, tar_output_path, batch_size, filelist_path):
    audio_path = os.path.join(root_path, "audio")

    os.makedirs(tar_output_path, exist_ok=True)

    batch_count = 0
    tar_file = None
    tar_file_path = ""

    print(f"Creating tar files from {filelist_path}")

    with open(filelist_path, "r") as filelist:
        filelist = filelist.readlines()

    max_files = len(filelist) // batch_size
    n_number_length = len(str(max_files))

    for line in tqdm(filelist, desc="Creating tar files", total=len(filelist)):
        video_file_path = line.strip()
        speaker_id, video_file = os.path.split(video_file_path)
        speaker_id = os.path.basename(speaker_id)
        base_filename, _ = os.path.splitext(video_file)
        audio_file = base_filename + ".wav"
        audio_embedding_file = base_filename + "_emb.pt"

        audio_file_path = os.path.join(audio_path, speaker_id, audio_file)
        audio_emb_file_path = os.path.join(audio_path, speaker_id, audio_embedding_file)

        if not os.path.exists(audio_file_path):
            print(f"No audio file for video: {video_file_path}")
            continue

        if batch_size and batch_count % batch_size == 0:
            if tar_file is not None:
                tar_file.close()
                print(f"Created tar file: {tar_file_path}")
            file_number = str(batch_count // batch_size).zfill(n_number_length)
            tar_file_path = os.path.join(tar_output_path, f"batch_{file_number}.tar")
            tar_file = tarfile.open(tar_file_path, "w")

        tar_file.add(video_file_path, arcname=os.path.join("video", speaker_id, video_file))
        tar_file.add(audio_file_path, arcname=os.path.join("audio", speaker_id, audio_file))
        tar_file.add(audio_emb_file_path, arcname=os.path.join("audio", speaker_id, audio_embedding_file))
        batch_count += 1

    if tar_file is not None:
        tar_file.close()
        print(f"Created tar file: {tar_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a WebDataset from video and audio files.")
    parser.add_argument("root_path", help="Root path of the video and audio directories.")
    parser.add_argument("tar_output_path", help="Output path for the tar files.")
    parser.add_argument("filelist_path", help="Path to the text file containing the list of video files.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of samples per tar file.")

    args = parser.parse_args()

    create_webdataset(args.root_path, args.tar_output_path, args.batch_size, args.filelist_path)
