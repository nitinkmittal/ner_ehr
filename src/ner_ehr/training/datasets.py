# import os
# from torch.utils.data import Dataset, DataLoader
# import torch
# from torch import Tensor
# from pathlib import Path
# from typing import Union, Tuple, Optional
# from PIL import Image
# import numpy as np
# from tqdm import tqdm
# import pytorch_lightning as pl

# from shutil import copyfile

# from .transforms import OneHotMask


# def train_val_test_split(
#     img_dir: Union[Path, str],
#     mask_dir: Union[Path, str],
#     split_ratio: Tuple[float, float],
#     seed: int = None,
# ):
#     """Create train, validation and test subsets.
#     Args:
#         img_dir: directory with images
#         mask_dir: directory with masks
#         split_ratio: train and validation split ratio
#         seed: seed for random number generator
#     """

#     assert np.sum(split_ratio) <= 1.0
#     rng = np.random.default_rng(seed)

#     fps = os.listdir(img_dir)
#     total = len(fps)
#     rng.shuffle(fps)

#     base_dir = os.path.dirname(img_dir)

#     def subset(fps: Union[Path, str], mode: str):
#         """Create subset."""
#         mode_dir = os.path.join(base_dir, mode)
#         os.makedirs(mode_dir, exist_ok=True)
#         mode_img_dir = os.path.join(mode_dir, "images")
#         os.makedirs(mode_img_dir, exist_ok=True)
#         mode_mask_dir = os.path.join(mode_dir, "masks")
#         os.makedirs(mode_mask_dir, exist_ok=True)
#         t = tqdm(enumerate(fps), position=0, leave=False)
#         for i, fp in t:
#             copyfile(os.path.join(img_dir, fp), os.path.join(mode_img_dir, fp))
#             copyfile(
#                 os.path.join(mask_dir, fp), os.path.join(mode_mask_dir, fp)
#             )
#             t.set_description(
#                 f"Creating {mode} subset, copied {i+1}/{len(fps)} files"
#             )

#     # train
#     train_fps = fps[: int(total * split_ratio[0])]
#     subset(train_fps, mode="train")

#     # val
#     train_fps = fps[
#         int(total * split_ratio[0]) : int(total * np.sum(split_ratio))
#     ]
#     subset(train_fps, mode="val")

#     # test
#     train_fps = fps[int(total * np.sum(split_ratio)) :]
#     subset(train_fps, mode="test")


# class NERDataset(Dataset):
#     """Synthetic aperture radar (SAR) images and masks dataset reader."""

#     def __init__(
#         self,
#         img_dir: Union[Path, str],
#         mask_dir: Union[Path, str],
#         num_classes: int,
#     ):
#         """
#         Initialize SAR dataset.
#         Args:
#             img_dir: directory with images
#             mask_dir: directory with masks
#             num_classes: number of distinct classes to classify
#         Returns:
#             None
#         """
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.num_classes = num_classes

#         self.img_fps = os.listdir(self.img_dir)
#         self.mask_fps = os.listdir(self.mask_dir)

#         if len(self.img_fps) != len(self.mask_fps):
#             raise ValueError(
#                 f"Number of files in {img_dir} and {mask_dir} are not equal"
#             )

#     def __len__(self) -> int:
#         return len(self.img_fps)

#     def __getitem__(self, idx: int) -> Tuple[str, Tensor, Tensor]:
#         """
#         Return data for given idx.
#         Args:
#             idx: file index.
#         Returns:
#             fp: file pointer
#             img: A tensor of shape (1, H, W)
#             mask: A tensor of shape (num_classes, H, W)
#         """
#         fp = self.img_fps[idx]
#         img = Image.open(os.path.join(self.img_dir, fp))
#         mask = Image.open(os.path.join(self.mask_dir, fp))

#         # normalizing 16 bit image
#         img = np.array(img) / 2 ** 16

#         mask = torch.unsqueeze(
#             torch.tensor(np.array(mask), dtype=torch.uint8), dim=0
#         )
#         mask = OneHotMask(num_classes=self.num_classes)(mask)
#         img = ToTensor()(img)

#         return fp, img, mask.type(torch.get_default_dtype())


# class SARDataModule(pl.LightningDataModule):
#     def __init__(
#         self,
#         train_dirs: Tuple[Union[Path, str], Union[Path, str]],
#         val_dirs: Tuple[Union[Path, str], Union[Path, str]],
#         num_classes: int,
#         batch_sizes: Union[int, Tuple[int, int]],
#         num_workers: Union[int, Tuple[int, int, int]],
#     ):
#         super().__init__()

#         self.train_ds = SARDataset(
#             img_dir=train_dirs[0],
#             mask_dir=train_dirs[1],
#             num_classes=num_classes,
#         )
#         self.val_ds = SARDataset(
#             img_dir=val_dirs[0], mask_dir=val_dirs[1], num_classes=num_classes
#         )

#         if isinstance(num_workers, int):
#             self.train_workers = self.val_workers = num_workers

#         else:
#             (
#                 self.train_workers,
#                 self.val_workers,
#             ) = num_workers

#         if isinstance(batch_sizes, int):
#             self.train_batch_size = self.val_batch_size = batch_sizes

#         else:
#             (
#                 self.train_batch_size,
#                 self.val_batch_size,
#             ) = batch_sizes

#     def setup(self, stage: Optional[str] = None):
#         pass

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_ds,
#             batch_size=self.train_batch_size,
#             shuffle=True,
#             num_workers=self.train_workers,
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_ds,
#             batch_size=self.val_batch_size,
#             shuffle=False,
#             num_workers=self.val_workers,
#         )

#     def test_dataloader(self):
#         pass

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Union, Optional, Tuple
import os
from ner_ehr.data.variables import AnnotationTuple, TokenTuple
from ner_ehr.data.ehr import EHR
from ner_ehr.data.utils import df_to_namedtuples
from glob import glob
from pytorch_lightning import LightningDataModule


class EHRDataset(Dataset):
    def __init__(self, dir: Union[Path, str], annotated: bool = False):
        self.dir = dir
        self.annotated = annotated
        self.tokens: Union[List[TokenTuple], List[AnnotationTuple]] = []
        self._setup()

    def _setup(
        self,
    ) -> Union[List[TokenTuple], List[AnnotationTuple]]:

        for fp in glob(os.path.join(self.dir, r"*.csv")):

            if self.annotated:
                self.tokens += df_to_namedtuples(
                    name=AnnotationTuple.__name__,
                    df=EHR.read_csv_tokens_with_annotations(fp=fp),
                )

            else:
                self.tokens += df_to_namedtuples(
                    name=TokenTuple.__name__,
                    df=EHR.read_csv_tokens_without_annotations(fp=fp),
                )

    def __getitem__(
        self, i: int
    ) -> Union[List[TokenTuple], List[AnnotationTuple]]:
        return self.tokens[i]

    def __len__(
        self,
    ) -> int:
        return len(self.tokens)


class EHRDataModule(LightningDataModule):
    def __init__(
        self,
        train_dir: Optional[Union[Path, str]] = None,
        val_dir: Optional[Union[Path, str]] = None,
        test_dir: Optional[Union[Path, str]] = None,
        annotated: bool = False,
        batch_sizes: Union[int, Tuple[int, int], Tuple[int, int, int]] = 1,
        num_workers: Union[int, Tuple[int, int], Tuple[int, int, int]] = 1,
        shuffle: bool = False,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.train_dataset: EHRDataset = None
        self.val_dataset: EHRDataset = None
        self.test_dataset: EHRDataset = None
        self.annotated = annotated
        if isinstance(num_workers, int):
            self.train_workers = self.val_workers = num_workers
            if self.test_dir is not None:
                self.test_workers = num_workers
        else:
            if self.test_dir is not None:
                (
                    self.train_workers,
                    self.val_workers,
                    self.test_workers,
                ) = num_workers
            else:
                (
                    self.train_workers,
                    self.val_workers,
                ) = num_workers

        if isinstance(batch_sizes, int):
            self.train_batch_size = self.val_batch_size = batch_sizes

    def setup(self, stage: Optional[str] = None):
        if self.train_dir is not None:
            self.train_dataset = EHRDataset(
                dir=self.train_dir, annotated=self.annotated
            )
        if self.val_dir is not None:
            self.val_dataset = EHRDataset(
                dir=self.val_dir, annotated=self.annotated
            )
        if self.test_dir is not None:
            self.test_dataset = EHRDataset(
                dir=self.test_dir, annotated=self.annotated
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.train_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.val_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.val_workers,
        )
