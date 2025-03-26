import random
import numpy as np
from functools import partial
from torch.utils.data import Dataset, WeightedRandomSampler
import torch.nn.functional as F
import torch
import math
import decord
from einops import rearrange
from more_itertools import sliding_window
from omegaconf import ListConfig
import torchaudio
import soundfile as sf
from torchvision.transforms import RandomHorizontalFlip
from audiomentations import Compose, AddGaussianNoise, PitchShift
from safetensors.torch import load_file
from tqdm import tqdm
import cv2

torchaudio.set_audio_backend("sox_io")
decord.bridge.set_bridge("torch")


def exists(x):
    return x is not None


def trim_pad_audio(audio, sr, max_len_sec=None, max_len_raw=None):
    len_file = audio.shape[-1]

    if max_len_sec or max_len_raw:
        max_len = max_len_raw if max_len_raw is not None else int(max_len_sec * sr)
        if len_file < int(max_len):
            # dummy = np.zeros((1, int(max_len_sec * sr) - len_file))
            # extened_wav = np.concatenate((audio_data, dummy[0]))
            extened_wav = torch.nn.functional.pad(
                audio, (0, int(max_len) - len_file), "constant"
            )
        else:
            extened_wav = audio[:, : int(max_len)]
    else:
        extened_wav = audio

    return extened_wav


# Similar to regular video dataset but trades flexibility for speed
class VideoDataset(Dataset):
    def __init__(
        self,
        filelist,
        resize_size=None,
        audio_folder="Audio",
        video_folder="CroppedVideos",
        emotions_folder="emotions",
        audio_emb_folder=None,
        video_extension=".avi",
        audio_extension=".wav",
        audio_rate=16000,
        latent_folder=None,
        audio_in_video=False,
        fps=25,
        num_frames=5,
        need_cond=True,
        step=1,
        mode="prediction",
        max_missing_audio_files=10,
        scale_audio=False,
        augment=False,
        augment_audio=False,
        use_latent=False,
        # precomputed_latent=False,
        latent_type="stable",
        latent_scale=1,  # For backwards compatibility
        from_audio_embedding=False,
        load_all_possible_indexes=False,
        audio_emb_type="wavlm",
        cond_noise=[-3.0, 0.5],
        motion_id=255.0,
        data_mean=None,
        data_std=None,
        use_latent_condition=False,
        skip_frames=0,
        get_separate_id=False,
        virtual_increase=1,
        filter_by_length=False,
        select_randomly=False,
        balance_datasets=True,
        use_emotions=False,
        get_original_frames=False,
        add_extra_audio_emb=False,
        expand_box=0.0,
        nose_index=28,
    ):
        self.audio_folder = audio_folder
        self.from_audio_embedding = from_audio_embedding
        self.audio_emb_type = audio_emb_type
        self.cond_noise = cond_noise
        self.latent_condition = use_latent_condition
        precomputed_latent = latent_type
        self.audio_emb_folder = (
            audio_emb_folder if audio_emb_folder is not None else audio_folder
        )
        self.skip_frames = skip_frames
        self.get_separate_id = get_separate_id
        self.fps = fps
        self.virtual_increase = virtual_increase
        self.select_randomly = select_randomly
        self.use_emotions = use_emotions
        self.emotions_folder = emotions_folder
        self.get_original_frames = get_original_frames
        self.add_extra_audio_emb = add_extra_audio_emb
        self.expand_box = expand_box
        self.nose_index = nose_index
        # self.fps = fps

        assert not (exists(data_mean) ^ exists(data_std)), (
            "Both data_mean and data_std should be provided"
        )

        if data_mean is not None:
            data_mean = rearrange(torch.as_tensor(data_mean), "c -> c () () ()")
            data_std = rearrange(torch.as_tensor(data_std), "c -> c () () ()")
        self.data_mean = data_mean
        self.data_std = data_std
        self.motion_id = motion_id
        self.latent_folder = (
            latent_folder if latent_folder is not None else video_folder
        )
        self.audio_in_video = audio_in_video

        self.filelist = []
        self.audio_filelist = []
        # missing_audio = 0
        with open(filelist, "r") as files:
            for f in files.readlines():
                f = f.rstrip()

                audio_path = f.replace(video_folder, audio_folder).replace(
                    video_extension, audio_extension
                )
                # if not self.audio_in_video and not os.path.exists(audio_path):
                #     missing_audio += 1
                #     print("Missing audio file: ", audio_path)
                #     if missing_audio > max_missing_audio_files:
                #         raise FileNotFoundError(f"Missing more than {max_missing_audio_files} audio files")
                #     continue
                self.filelist += [f]
                self.audio_filelist += [audio_path]

        self.resize_size = resize_size
        if use_latent and not precomputed_latent:
            self.resize_size *= 4 if latent_type in ["stable", "ldm"] else 8
        self.scale_audio = scale_audio
        self.step = step
        self.use_latent = use_latent
        self.precomputed_latent = precomputed_latent
        self.latent_type = latent_type
        self.latent_scale = latent_scale
        self.video_ext = video_extension
        self.video_folder = video_folder

        self.augment = augment
        self.maybe_augment = RandomHorizontalFlip(p=0.5) if augment else lambda x: x
        self.maybe_augment_audio = (
            Compose(
                [
                    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.002, p=0.25),
                    # TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3),
                    PitchShift(min_semitones=-1, max_semitones=1, p=0.25),
                    # Shift(min_fraction=-0.5, max_fraction=0.5, p=0.333),
                ]
            )
            if augment_audio
            else lambda x, sample_rate: x
        )
        self.maybe_augment_audio = partial(
            self.maybe_augment_audio, sample_rate=audio_rate
        )

        self.mode = mode
        if mode == "interpolation":
            need_cond = False  # Interpolation does not need condition as first and last frame becomes the condition
        self.need_cond = need_cond  # If need cond will extract one more frame than the number of frames
        if get_separate_id:
            self.need_cond = True
        # It is used for the conditional model when the condition is not on the temporal dimension
        num_frames = num_frames if not self.need_cond else num_frames + 1
        # print(f"Num frames: {num_frames}")

        # Get metadata about video and audio
        # _, self.audio_rate = torchaudio.load(self.audio_filelist[0], channels_first=False)
        vr = decord.VideoReader(self.filelist[0])
        self.video_rate = math.ceil(vr.get_avg_fps())
        print(f"Video rate: {self.video_rate}")
        self.audio_rate = audio_rate
        a2v_ratio = fps / float(self.audio_rate)
        self.samples_per_frame = math.ceil(1 / a2v_ratio)

        if get_separate_id:
            assert mode == "prediction", (
                "Separate identity frame is only supported for prediction mode"
            )
            # No need for extra frame if we are getting a separate identity frame
            self.need_cond = True
            num_frames -= 1
        self.num_frames = num_frames
        self.load_all_possible_indexes = load_all_possible_indexes
        if load_all_possible_indexes:
            self._indexes = self._get_indexes(self.filelist, self.audio_filelist)
        else:
            if filter_by_length:
                self._indexes = self.filter_by_length(
                    self.filelist, self.audio_filelist
                )
            else:
                self._indexes = list(zip(self.filelist, self.audio_filelist))

        self.balance_datasets = balance_datasets
        if self.balance_datasets:
            self.weights = self._calculate_weights()
            self.sampler = WeightedRandomSampler(
                self.weights, num_samples=len(self._indexes), replacement=True
            )

    def __len__(self):
        return len(self._indexes) * self.virtual_increase

    def get_emotions(self, video_file, video_indexes):
        emotions_path = video_file.replace(
            self.video_folder, self.emotions_folder
        ).replace(self.video_ext, ".pt")
        emotions = torch.load(emotions_path)
        return (
            emotions["valence"][video_indexes],
            emotions["arousal"][video_indexes],
            emotions["labels"][video_indexes],
        )

    def get_frame_indices(self, total_video_frames, select_randomly=False, start_idx=0):
        if select_randomly:
            # Randomly select self.num_frames indices from the available range
            available_indices = list(range(start_idx, total_video_frames))
            if len(available_indices) < self.num_frames:
                raise ValueError(
                    "Not enough frames in the video to sample with given parameters."
                )
            indexes = random.sample(available_indices, self.num_frames)
            return sorted(indexes)  # Sort to maintain temporal order
        else:
            # Calculate the maximum possible start index
            max_start_idx = total_video_frames - (
                (self.num_frames - 1) * (self.skip_frames + 1) + 1
            )

            # Generate a random start index
            if max_start_idx > 0:
                start_idx = np.random.randint(start_idx, max_start_idx)
            else:
                raise ValueError(
                    "Not enough frames in the video to sample with given parameters."
                )

            # Generate the indices
            indexes = [
                start_idx + i * (self.skip_frames + 1) for i in range(self.num_frames)
            ]

        return indexes

    def _load_audio(self, filename, max_len_sec, start=None, indexes=None):
        audio, sr = sf.read(
            filename,
            start=math.ceil(start * self.audio_rate),
            frames=math.ceil(self.audio_rate * max_len_sec),
            always_2d=True,
        )  # e.g (16000, 1)
        audio = audio.T  # (1, 16000)
        assert sr == self.audio_rate, (
            f"Audio rate is {sr} but should be {self.audio_rate}"
        )
        audio = audio.mean(0, keepdims=True)
        audio = self.maybe_augment_audio(audio)
        audio = torch.from_numpy(audio).float()
        # audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.audio_rate)
        audio = trim_pad_audio(audio, self.audio_rate, max_len_sec=max_len_sec)
        return audio[0]

    def ensure_shape(self, tensors):
        target_length = self.samples_per_frame
        processed_tensors = []
        for tensor in tensors:
            current_length = tensor.shape[1]
            diff = current_length - target_length
            assert abs(diff) <= 5, (
                f"Expected shape {target_length}, but got {current_length}"
            )
            if diff < 0:
                # Calculate how much padding is needed
                padding_needed = target_length - current_length
                # Pad the tensor
                padded_tensor = F.pad(tensor, (0, padding_needed))
                processed_tensors.append(padded_tensor)
            elif diff > 0:
                # Trim the tensor
                trimmed_tensor = tensor[:, :target_length]
                processed_tensors.append(trimmed_tensor)
            else:
                # If it's already the correct size
                processed_tensors.append(tensor)
        return torch.cat(processed_tensors)

    def normalize_latents(self, latents):
        if self.data_mean is not None:
            # Normalize latents to 0 mean and 0.5 std
            latents = ((latents - self.data_mean) / self.data_std) * 0.5
        return latents

    def convert_indexes(self, indexes_25fps, fps_from=25, fps_to=60):
        ratio = fps_to / fps_from
        indexes_60fps = [int(index * ratio) for index in indexes_25fps]
        return indexes_60fps

    def _get_frames_and_audio(self, idx):
        if self.load_all_possible_indexes:
            indexes, video_file, audio_file = self._indexes[idx]
            if self.audio_in_video:
                vr = decord.AVReader(video_file, sample_rate=self.audio_rate)
            else:
                vr = decord.VideoReader(video_file)
            len_video = len(vr)
            if "AA_processed" in video_file or "1000actors_nsv" in video_file:
                len_video *= 25 / 60
                len_video = int(len_video)
        else:
            video_file, audio_file = self._indexes[idx]
            if self.audio_in_video:
                vr = decord.AVReader(video_file, sample_rate=self.audio_rate)
            else:
                vr = decord.VideoReader(video_file)
            len_video = len(vr)
            if "AA_processed" in video_file or "1000actors_nsv" in video_file:
                len_video *= 25 / 60
                len_video = int(len_video)

            indexes = self.get_frame_indices(
                len_video,
                select_randomly=self.select_randomly,
                start_idx=120 if "1000actors_nsv" in video_file else 0,
            )

        if self.get_separate_id:
            id_idx = np.random.randint(0, len_video)
            indexes.insert(0, id_idx)

        if "AA_processed" in video_file or "1000actors_nsv" in video_file:
            video_indexes = self.convert_indexes(indexes, fps_from=25, fps_to=60)
            audio_file = audio_file.replace("_output_output", "")
            if self.audio_emb_type == "wav2vec2" and "AA_processed" in video_file:
                audio_path_extra = ".safetensors"
            else:
                audio_path_extra = f"_{self.audio_emb_type}_emb.safetensors"

            video_path_extra = f"_{self.latent_type}_512_latent.safetensors"
            audio_path_extra_extra = (
                ".pt" if "AA_processed" in video_file else "_beats_emb.pt"
            )

        else:
            video_indexes = indexes
            audio_path_extra = f"_{self.audio_emb_type}_emb.safetensors"
            video_path_extra = f"_{self.latent_type}_512_latent.safetensors"
            audio_path_extra_extra = "_beats_emb.pt"

        emotions = None
        if self.use_emotions:
            emotions = self.get_emotions(video_file, video_indexes)
            if self.get_separate_id:
                emotions = (emotions[0][1:], emotions[1][1:], emotions[2][1:])

        raw_audio = None
        if self.audio_in_video:
            raw_audio, frames_video = vr.get_batch(video_indexes)
            raw_audio = rearrange(self.ensure_shape(raw_audio), "f s -> (f s)")

        if self.use_latent and self.precomputed_latent:
            latent_file = video_file.replace(self.video_ext, video_path_extra).replace(
                self.video_folder, self.latent_folder
            )
            frames = load_file(latent_file)["latents"][video_indexes, :, :, :]

            if frames.shape[-1] != 64:
                print(f"Frames shape: {frames.shape}, video file: {video_file}")

            frames = rearrange(frames, "t c h w -> c t h w") * self.latent_scale
            frames = self.normalize_latents(frames)
        else:
            if self.audio_in_video:
                frames = frames_video.permute(3, 0, 1, 2).float()
            else:
                frames = vr.get_batch(video_indexes).permute(3, 0, 1, 2).float()

        if raw_audio is None:
            # Audio is not in video
            raw_audio = self._load_audio(
                audio_file,
                max_len_sec=frames.shape[1] / self.fps,
                start=indexes[0] / self.fps,
                # indexes=indexes,
            )
        if not self.from_audio_embedding:
            audio = raw_audio
            audio_frames = rearrange(audio, "(f s) -> f s", s=self.samples_per_frame)
        else:
            audio = load_file(
                audio_file.replace(self.audio_folder, self.audio_emb_folder).split(".")[
                    0
                ]
                + audio_path_extra
            )["audio"]
            audio_frames = audio[indexes, :]
            if self.add_extra_audio_emb:
                audio_extra = torch.load(
                    audio_file.replace(self.audio_folder, self.audio_emb_folder).split(
                        "."
                    )[0]
                    + audio_path_extra_extra
                )
                audio_extra = audio_extra[indexes, :]
                audio_frames = torch.cat([audio_frames, audio_extra], dim=-1)

        audio_frames = (
            audio_frames[1:] if self.need_cond else audio_frames
        )  # Remove audio of first frame

        if self.get_original_frames:
            original_frames = vr.get_batch(video_indexes).permute(3, 0, 1, 2).float()
            original_frames = self.scale_and_crop((original_frames / 255.0) * 2 - 1)
            original_frames = (
                original_frames[:, 1:] if self.need_cond else original_frames
            )
        else:
            original_frames = None

        if not self.use_latent or (self.use_latent and not self.precomputed_latent):
            frames = self.scale_and_crop((frames / 255.0) * 2 - 1)

        target = frames[:, 1:] if self.need_cond else frames
        if self.mode == "prediction":
            if self.use_latent:
                if self.audio_in_video:
                    clean_cond = (
                        frames_video[0].unsqueeze(0).permute(3, 0, 1, 2).float()
                    )
                else:
                    clean_cond = (
                        vr[video_indexes[0]].unsqueeze(0).permute(3, 0, 1, 2).float()
                    )
                original_size = clean_cond.shape[-2:]
                clean_cond = self.scale_and_crop((clean_cond / 255.0) * 2 - 1).squeeze(
                    0
                )
                if self.latent_condition:
                    noisy_cond = frames[:, 0]
                else:
                    noisy_cond = clean_cond
            else:
                clean_cond = frames[:, 0]
                noisy_cond = clean_cond
        elif self.mode == "interpolation":
            if self.use_latent:
                # vr = decord.VideoReader(video_file) if vr is None else vr
                if self.audio_in_video:
                    clean_cond = frames_video[[0, -1]].permute(3, 0, 1, 2).float()
                else:
                    clean_cond = (
                        vr.get_batch([video_indexes[0], video_indexes[-1]])
                        .permute(3, 0, 1, 2)
                        .float()
                    )
                original_size = clean_cond.shape[-2:]
                clean_cond = self.scale_and_crop((clean_cond / 255.0) * 2 - 1)
                if self.latent_condition:
                    noisy_cond = torch.stack([target[:, 0], target[:, -1]], dim=1)
                else:
                    noisy_cond = clean_cond
            else:
                clean_cond = torch.stack([target[:, 0], target[:, -1]], dim=1)
                noisy_cond = clean_cond

        # Add noise to conditional frame
        # noisy_cond = None
        if self.cond_noise and isinstance(self.cond_noise, ListConfig):
            cond_noise = (
                self.cond_noise[0] + self.cond_noise[1] * torch.randn((1,))
            ).exp()
            noisy_cond = noisy_cond + cond_noise * torch.randn_like(noisy_cond)
        else:
            noisy_cond = noisy_cond + self.cond_noise * torch.randn_like(noisy_cond)
            cond_noise = self.cond_noise
        # else:
        #     cond_noise = None

        return (
            original_frames,
            clean_cond,
            noisy_cond,
            target,
            audio_frames,
            raw_audio,
            cond_noise,
            emotions,
        )

    def filter_by_length(self, video_filelist, audio_filelist):
        def with_opencv(filename):
            video = cv2.VideoCapture(filename)
            frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

            return int(frame_count)

        filtered_video = []
        filtered_audio = []
        min_length = (self.num_frames - 1) * (self.skip_frames + 1) + 1
        for vid_file, audio_file in tqdm(
            zip(video_filelist, audio_filelist),
            total=len(video_filelist),
            desc="Filtering",
        ):
            # vr = decord.VideoReader(vid_file)

            len_video = with_opencv(vid_file)
            # Short videos
            if len_video < min_length:
                continue
            filtered_video.append(vid_file)
            filtered_audio.append(audio_file)
        print(f"New number of files: {len(filtered_video)}")
        return filtered_video, filtered_audio

    def _get_indexes(self, video_filelist, audio_filelist):
        indexes = []
        self.og_shape = None
        for vid_file, audio_file in zip(video_filelist, audio_filelist):
            vr = decord.VideoReader(vid_file)
            if self.og_shape is None:
                self.og_shape = vr[0].shape[-2]
            len_video = len(vr)
            # Short videos
            if len_video < self.num_frames:
                continue
            else:
                possible_indexes = list(
                    sliding_window(range(len_video), self.num_frames)
                )[:: self.step]
                possible_indexes = list(
                    map(lambda x: (x, vid_file, audio_file), possible_indexes)
                )
                indexes.extend(possible_indexes)
        print("Indexes", len(indexes), "\n")
        return indexes

    def scale_and_crop(self, video):
        h, w = video.shape[-2], video.shape[-1]
        # scale shorter side to resolution

        if self.resize_size is not None:
            scale = self.resize_size / min(h, w)
            if h < w:
                target_size = (self.resize_size, math.ceil(w * scale))
            else:
                target_size = (math.ceil(h * scale), self.resize_size)
            video = F.interpolate(
                video,
                size=target_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            # center crop
            h, w = video.shape[-2], video.shape[-1]
            w_start = (w - self.resize_size) // 2
            h_start = (h - self.resize_size) // 2
            video = video[
                :,
                :,
                h_start : h_start + self.resize_size,
                w_start : w_start + self.resize_size,
            ]
        return self.maybe_augment(video)

    def _calculate_weights(self):
        aa_processed_count = sum(
            1
            for item in self._indexes
            if "AA_processed" in (item[1] if len(item) == 3 else item[0])
        )
        nsv_processed_count = sum(
            1
            for item in self._indexes
            if "1000actors_nsv" in (item[1] if len(item) == 3 else item[0])
        )
        other_count = len(self._indexes) - aa_processed_count - nsv_processed_count

        aa_processed_weight = 1 / aa_processed_count if aa_processed_count > 0 else 0
        nsv_processed_weight = 1 / nsv_processed_count if nsv_processed_count > 0 else 0
        other_weight = 1 / other_count if other_count > 0 else 0

        print(
            f"AA processed count: {aa_processed_count}, NSV processed count: {nsv_processed_count}, other count: {other_count}"
        )
        print(f"AA processed weight: {aa_processed_weight}")
        print(f"NSV processed weight: {nsv_processed_weight}")
        print(f"Other weight: {other_weight}")

        weights = [
            aa_processed_weight
            if "AA_processed" in (item[1] if len(item) == 3 else item[0])
            else nsv_processed_weight
            if "1000actors_nsv" in (item[1] if len(item) == 3 else item[0])
            else other_weight
            for item in self._indexes
        ]
        return weights

    def __getitem__(self, idx):
        if self.balance_datasets:
            idx = self.sampler.__iter__().__next__()

        try:
            (
                original_frames,
                clean_cond,
                noisy_cond,
                target,
                audio,
                raw_audio,
                cond_noise,
                emotions,
            ) = self._get_frames_and_audio(idx % len(self._indexes))
        except Exception as e:
            print(f"Error with index {idx}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))
        out_data = {}

        if original_frames is not None:
            out_data["original_frames"] = original_frames

        if audio is not None:
            out_data["audio_emb"] = audio
            out_data["raw_audio"] = raw_audio

        if self.use_emotions:
            out_data["valence"] = emotions[0]
            out_data["arousal"] = emotions[1]
            out_data["emo_labels"] = emotions[2]
        if self.use_latent:
            input_key = "latents"
        else:
            input_key = "frames"
        out_data[input_key] = target
        if noisy_cond is not None:
            out_data["cond_frames"] = noisy_cond
        out_data["cond_frames_without_noise"] = clean_cond
        if cond_noise is not None:
            out_data["cond_aug"] = cond_noise

        out_data["motion_bucket_id"] = torch.tensor([self.motion_id])
        out_data["fps_id"] = torch.tensor([self.fps - 1])
        out_data["num_video_frames"] = self.num_frames
        out_data["image_only_indicator"] = torch.zeros(self.num_frames)
        return out_data


if __name__ == "__main__":
    import torchvision.transforms as transforms
    import cv2

    transform = transforms.Compose(transforms=[transforms.Resize((256, 256))])
    dataset = VideoDataset(
        "/vol/paramonos2/projects/antoni/datasets/mahnob/filelist_videos_val.txt",
        transform=transform,
        num_frames=25,
    )
    print(len(dataset))
    idx = np.random.randint(0, len(dataset))

    for i in range(10):
        print(dataset[i][0].shape, dataset[i][1].shape)

    image_identity = (dataset[idx][0].permute(1, 2, 0).numpy() + 1) / 2 * 255
    image_other = (dataset[idx][1][:, -1].permute(1, 2, 0).numpy() + 1) / 2 * 255
    cv2.imwrite("image_identity.png", image_identity[:, :, ::-1])
    for i in range(25):
        image = (dataset[idx][1][:, i].permute(1, 2, 0).numpy() + 1) / 2 * 255
        cv2.imwrite(f"tmp_vid_dataset/image_{i}.png", image[:, :, ::-1])
