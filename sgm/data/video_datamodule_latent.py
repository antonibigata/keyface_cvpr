from typing import Any, Dict, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from omegaconf import DictConfig

import sys
import pyrootutils

root = pyrootutils.setup_root(__file__, pythonpath=True)
sys.path.append(root)
from sgm.data.video_dataset_latent import VideoDataset


class VideoDataModule(LightningDataModule):
    """
    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        train: DictConfig,
        validation: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
        skip_val_loader: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.train_config = train
        assert "datapipeline" in self.train_config and "loader" in self.train_config, (
            "train config requires the fields `datapipeline` and `loader`"
        )

        self.val_config = validation
        if not skip_val_loader:
            if self.val_config is not None:
                assert (
                    "datapipeline" in self.val_config and "loader" in self.val_config
                ), "validation config requires the fields `datapipeline` and `loader`"
            else:
                print(
                    "Warning: No Validation datapipeline defined, using that one from training"
                )
                self.val_config = train

        self.test_config = test
        if self.test_config is not None:
            assert (
                "datapipeline" in self.test_config and "loader" in self.test_config
            ), "test config requires the fields `datapipeline` and `loader`"

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        print("Preparing datasets")

        self.train_datapipeline = VideoDataset(**self.train_config.datapipeline)
        if self.val_config:
            self.val_datapipeline = VideoDataset(**self.val_config.datapipeline)
        if self.test_config:
            self.test_datapipeline = VideoDataset(**self.test_config.datapipeline)

    def train_dataloader(self):
        return DataLoader(self.train_datapipeline, **self.train_config.loader)

    def val_dataloader(self):
        if self.val_datapipeline:
            return DataLoader(self.val_datapipeline, **self.val_config.loader)
        else:
            return None

    def test_dataloader(self):
        if self.test_datapipeline:
            return DataLoader(self.test_datapipeline, **self.test_config.loader)
        else:
            return None

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils
    import cv2

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(
        root / "configs" / "datamodule" / "image_datamodule.yaml"
    )
    # cfg.data_dir = str(root / "data")
    data = hydra.utils.instantiate(cfg)
    data.prepare_data()
    data.setup()
    print(data.data_train.__getitem__(0)[0].shape)
    batch = next(iter(data.train_dataloader()))
    identity, target = batch
    image_identity = (identity[0].permute(1, 2, 0).numpy() + 1) / 2 * 255
    image_other = (target[0].permute(1, 2, 0).numpy() + 1) / 2 * 255
    cv2.imwrite("image_identity.png", image_identity[:, :, ::-1])
    cv2.imwrite("image_other.png", image_other[:, :, ::-1])
