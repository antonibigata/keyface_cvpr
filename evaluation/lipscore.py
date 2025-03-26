import os
import torch
from torchmetrics import Metric
from typing import List
import numpy as np

import hydra

from auto_avsr.lightning import ModelModule
from auto_avsr.datamodule.transforms import VideoTransform
from auto_avsr.preparation.detectors.retinaface.detector import LandmarksDetector
from auto_avsr.preparation.detectors.retinaface.video_process import VideoProcess
import torch.nn.functional as F


class LipScore(Metric):
    """
    LipScore Metric using the auto_avsr lip reading model.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        modality: str = "video",
        pretrained_model_path: str = "PATH_TO_CHECKPOINTS/vsr_trlrs2lrs3vox2avsp_base_updated.pth",
        config_dir: str = "PATH_TO_EVALUATION_FOLDER/auto_avsr/configs",
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__()

        if not os.path.exists(pretrained_model_path):
            raise FileNotFoundError(f"Model file not found at {pretrained_model_path}")

        if not os.path.exists(config_dir):
            raise FileNotFoundError(f"Config directory not found at {config_dir}")

        # Initialize Hydra configuration using absolute path
        with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = hydra.compose(
                config_name="config",
                overrides=[
                    f"data.modality={modality}",
                    f"pretrained_model_path={pretrained_model_path}",
                ],
            )
        self.cfg = cfg
        self.modality = cfg.data.modality

        self.my_device = device  # Renamed to avoid conflict

        # Initialize the model components
        self.video_transform = VideoTransform(subset="test")
        self.landmarks_detector = LandmarksDetector(device=str(device))
        self.video_process = VideoProcess(convert_gray=False)

        # Initialize the ModelModule
        self.modelmodule = ModelModule(self.cfg)
        self.modelmodule.model.load_state_dict(
            torch.load(
                self.cfg.pretrained_model_path,
                map_location=lambda storage, loc: storage,
            )
        )
        self.modelmodule.to(device)
        self.modelmodule.eval()

        # States for the metric
        self.add_state("errors", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_words", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "cosine_similarities", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

        self.counter = 0

    def update(self, videos_pred: List[torch.Tensor], videos_gt: List[torch.Tensor]):
        """
        Update the cosine similarity metric with batches of videos.

        Args:
            videos_pred (List[str]): List of predicted video tensors.
            videos_gt (List[str]): List of ground truth video tensors.
        """
        assert len(videos_pred) == len(videos_gt), (
            "The number of predicted and ground truth videos must be the same."
        )

        landmarks_pred_list = []
        landmarks_gt_list = []

        for pred, gt in zip(videos_pred, videos_gt):
            pred = pred.permute(0, 2, 3, 1).numpy() * 255
            gt = gt.permute(0, 2, 3, 1).numpy() * 255

            # Get features
            feature_pred, landmarks_pred = self.get_feature(pred)
            feature_gt, landmarks_gt = self.get_feature(gt)

            landmarks_pred_list.append(np.array(landmarks_pred))
            landmarks_gt_list.append(np.array(landmarks_gt))

            # Compute cosine similarity
            similarity = F.cosine_similarity(feature_pred, feature_gt, dim=1)

            # Update states
            self.cosine_similarities += similarity.mean().cpu().item()
            self.counter += 1

        return landmarks_pred_list, landmarks_gt_list

    def compute(self):
        result = {}

        result["cosine_sim"] = self.cosine_similarities.item() / self.counter

        return result

    def get_feature(self, video: torch.Tensor):
        # Apply landmarks detector
        landmarks = self.landmarks_detector(video)

        # Process video
        video_processed = self.video_process(video, landmarks)
        video_tensor = torch.tensor(video_processed).permute(0, 3, 1, 2)  # (T, C, H, W)
        video_tensor = self.video_transform(video_tensor)

        # Move to device
        video_tensor = video_tensor.to(self.my_device)

        # Get transcript
        with torch.no_grad():
            feature = self.modelmodule.forward_no_beam(
                video_tensor
            )  # Add batch dimension

        return feature, landmarks


if __name__ == "__main__":
    import torchvision

    # Create sample videos for testing
    batch_size = 2
    frames = 10
    height = 512
    width = 512
    channels = 3

    video_path = "PATH_TO_VIDEO"
    video = torchvision.io.read_video(video_path, pts_unit="sec")[
        0
    ]  # Returns (T, H, W, C)
    video = video.permute(0, 3, 1, 2) / 255  # Convert to (T, C, H, W)
    videos_pred = [video]
    videos_gt = [video]  # Using same video for demo

    lipscore = LipScore()
    lipscore.update(videos_pred, videos_gt)
    print(lipscore.compute())
