from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from pytorch_lightning.utilities import rank_zero_only
from typing import Union
import pytorch_lightning as pl
import os
from sgm.util import exists, suppress_output, default
import torchvision
from PIL import Image
import torch
import wandb
import moviepy.editor as mpy
from einops import rearrange
import torchaudio
# import tempfile
# import cv2
# import scipy.io.wavfile as wav
# import ffmpeg


@suppress_output
def save_audio_video(
    video, audio=None, frame_rate=25, sample_rate=16000, save_path="temp.mp4", keep_intermediate=False
):
    """Save audio and video to a single file.
    video: (t, c, h, w)
    audio: (channels t)
    """

    # temp_filename = next(tempfile._get_candidate_names())
    # if save_path:
    #     save_path = save_path
    # else:
    #     save_path = "/tmp/" + next(tempfile._get_candidate_names()) + ".mp4"
    save_path = str(save_path)
    try:
        torchvision.io.write_video(
            "temp_video.mp4", rearrange(video.detach().cpu(), "t c h w -> t h w c").to(torch.uint8), frame_rate
        )
        video_clip = mpy.VideoFileClip("temp_video.mp4")
        if audio is not None:
            torchaudio.save("temp_audio.wav", audio.detach().cpu(), sample_rate)
            audio_clip = mpy.AudioFileClip("temp_audio.wav")
            video_clip = video_clip.set_audio(audio_clip)
        video_clip.write_videofile(save_path, fps=frame_rate, codec="libx264", audio_codec="aac", verbose=False)
        if not keep_intermediate:
            os.remove("temp_video.mp4")
            if audio is not None:
                os.remove("temp_audio.wav")
        return 1
    except Exception as e:
        print(e)
        print("Saving video to file failed")
        return 0


# def write_video_opencv(video, video_rate, video_path):
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out = cv2.VideoWriter(video_path, fourcc, video_rate, (video.shape[2], video.shape[3]), 0)
#     for frame in list(video):
#         frame = np.squeeze(frame)
#         out.write(np.squeeze(frame))
#     out.release()


# # Code mostly inherited from bulletin
# def save_av_sample(video, video_rate, audio=None, audio_rate=16_000, path=None):
#     # Save video sample in train dir for debugging
#     # video_save = 0.5 * video.detach().cpu().numpy() + 0.5
#     video_save = rearrange(video, "t c h w -> t h w c").detach().cpu().numpy()
#     temp_filename = next(tempfile._get_candidate_names())
#     if path:
#         video_path = path
#     else:
#         video_path = "/tmp/" + next(tempfile._get_candidate_names()) + ".mp4"
#     write_video_opencv((video_save).astype(np.uint8), video_rate, "/tmp/" + temp_filename + ".mp4")
#     audio_save = audio.detach().squeeze().cpu().numpy()
#     wav.write("/tmp/" + temp_filename + ".wav", audio_rate, audio_save)
#     try:
#         in1 = ffmpeg.input("/tmp/" + temp_filename + ".mp4")
#         in2 = ffmpeg.input("/tmp/" + temp_filename + ".wav")
#         out = ffmpeg.output(in1["v"], in2["a"], video_path, loglevel="panic").overwrite_output()
#         out.run(capture_stdout=True, capture_stderr=True)
#     except ffmpeg.Error as e:
#         print("stdout:", e.stdout.decode("utf8"))
#         print("stderr:", e.stderr.decode("utf8"))
#         raise e
#     return video_path


class VideoLogger(Callback):
    def __init__(
        self,
        batch_frequency,
        max_videos,
        clamp=True,
        increase_log_steps=True,
        rescale=True,
        disabled=False,
        log_on_batch_idx=False,
        log_first_step=False,
        log_videos_kwargs=None,
        log_before_first_step=False,
        enable_autocast=True,
        batch_frequency_val=None,
    ):
        super().__init__()
        self.enable_autocast = enable_autocast
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_videos = max_videos
        self.log_steps = [2**n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.batch_freq_val = default(batch_frequency_val, self.batch_freq)
        self.log_steps_val = [2**n for n in range(int(np.log2(self.batch_freq_val)) + 1)]
        if not increase_log_steps:
            self.log_steps_val = [self.batch_freq_val]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_videos_kwargs = log_videos_kwargs if log_videos_kwargs else {}
        self.log_first_step = log_first_step
        self.log_before_first_step = log_before_first_step

    @rank_zero_only
    def log_local(
        self,
        save_dir,
        split,
        log_elements,
        raw_audio,
        global_step,
        current_epoch,
        batch_idx,
        pl_module: Union[None, pl.LightningModule] = None,
    ):
        root = os.path.join(save_dir, "videos", split)
        for k in log_elements:
            element = log_elements[k]
            if len(element.shape) == 4:
                grid = torchvision.utils.make_grid(element, nrow=4)
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                img = Image.fromarray(grid)
                img.save(path)
                if exists(pl_module):
                    assert isinstance(
                        pl_module.logger, WandbLogger
                    ), "logger_log_image only supports WandbLogger currently"
                    pl_module.logger.log_image(
                        key=f"{split}/{k}",
                        images=[
                            img,
                        ],
                        step=pl_module.global_step,
                    )
            elif len(element.shape) == 5:
                video = element
                if self.rescale:
                    video = (video + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                video = video * 255.0
                video = video.permute(0, 2, 1, 3, 4).cpu().detach().to(torch.uint8)  # b,t,c,h,w
                for i in range(video.shape[0]):
                    filename = "{}_gs-{:06}_e-{:06}_b-{:06}_{}.mp4".format(k, global_step, current_epoch, batch_idx, i)
                    path = os.path.join(root, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    log_audio = raw_audio[i] if raw_audio is not None else None
                    success = save_audio_video(
                        video[i],
                        audio=log_audio.unsqueeze(0) if log_audio is not None else None,
                        frame_rate=25,
                        sample_rate=16000,
                        save_path=path,
                        keep_intermediate=False,
                    )

                    # video_path = save_av_sample(video[i], 25, audio=raw_audio, audio_rate=16000, path=None)
                    if exists(pl_module):
                        assert isinstance(
                            pl_module.logger, WandbLogger
                        ), "logger_log_image only supports WandbLogger currently"
                        pl_module.logger.experiment.log(
                            {
                                f"{split}/{k}": wandb.Video(
                                    path if success else video,
                                    # caption=f"diffused videos w {n_frames} frames (condition left, generated right)",
                                    fps=25,
                                    format="mp4",
                                )
                            },
                        )

    @rank_zero_only
    def log_video(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        # print(f"check_idx: {check_idx}", f"split: {split}")
        if (
            self.check_frequency(check_idx, split=split)
            and hasattr(pl_module, "log_videos")  # batch_idx % self.batch_freq == 0
            and callable(pl_module.log_videos)
            and
            # batch_idx > 5 and
            self.max_videos > 0
        ):
            logger = type(pl_module.logger)
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            gpu_autocast_kwargs = {
                "enabled": self.enable_autocast,  # torch.is_autocast_enabled(),
                "dtype": torch.get_autocast_gpu_dtype(),
                "cache_enabled": torch.is_autocast_cache_enabled(),
            }
            with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
                videos = pl_module.log_videos(batch, split=split, **self.log_videos_kwargs)

            for k in videos:
                N = min(videos[k].shape[0], self.max_videos)
                videos[k] = videos[k][:N]
                if isinstance(videos[k], torch.Tensor):
                    videos[k] = videos[k].detach().float().cpu()
                    if self.clamp:
                        videos[k] = torch.clamp(videos[k], -1.0, 1.0)

            raw_audio = batch.get("raw_audio", None)

            self.log_local(
                pl_module.logger.save_dir,
                split,
                videos,
                raw_audio,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
                pl_module=pl_module if isinstance(pl_module.logger, WandbLogger) else None,
            )

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx, split="train"):
        if split == "val":
            if check_idx:
                check_idx -= 1
            if ((check_idx % self.batch_freq_val) == 0 or (check_idx in self.log_steps_val)) and (
                check_idx > 0 or self.log_first_step
            ):
                try:
                    self.log_steps_val.pop(0)
                except IndexError as e:
                    print(e)
                    pass
                return True
            return False
        if check_idx:
            check_idx -= 1
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
            check_idx > 0 or self.log_first_step
        ):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_video(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.log_before_first_step and pl_module.global_step == 0:
            print(f"{self.__class__.__name__}: logging before training")
            self.log_video(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_video(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, "calibrate_grad_norm"):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)
